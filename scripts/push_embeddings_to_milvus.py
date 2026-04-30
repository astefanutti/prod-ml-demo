"""Generate review embeddings and push directly to Milvus (no Feast).

Reads raw review parquets from S3, generates 384-dim embeddings with
all-MiniLM-L6-v2, and bulk-inserts into Milvus via pymilvus.

Designed to run as a K8s Job on the feast-spark-executor-embeddings image
(which has sentence-transformers + model weights baked in).

Usage:
    python scripts/push_embeddings_to_milvus.py \
        --input s3://smartshop-raw/raw/reviews/ \
        --milvus-uri http://milvus.smartshop.svc.cluster.local:19530 \
        --collection smartshop_review_embeddings \
        --max-rows 500000 \
        --batch-size 10000
"""

import argparse
import os
import time

import numpy as np
import pyarrow.parquet as pq
import s3fs
from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    utility,
)
from sentence_transformers import SentenceTransformer

EMBEDDING_DIM = 384
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="S3 path to raw review parquet (comma-sep for multiple)")
    parser.add_argument("--milvus-uri", default="http://milvus.smartshop.svc.cluster.local:19530")
    parser.add_argument("--collection", default="smartshop_review_embeddings")
    parser.add_argument("--max-rows", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--embed-batch-size", type=int, default=256,
                        help="Batch size for sentence-transformers encoding")
    return parser.parse_args()


def get_s3_fs():
    return s3fs.S3FileSystem(
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3",
                                    "http://minio.smartshop.svc.cluster.local:9000"),
        client_kwargs={"verify": False},
    )


def read_reviews(input_paths: str, max_rows: int):
    """Read raw review parquets from S3, return relevant columns."""
    fs = get_s3_fs()
    paths = [p.strip().replace("s3://", "").replace("s3a://", "")
             for p in input_paths.split(",") if p.strip()]

    all_files = []
    for path in paths:
        if fs.isdir(path):
            files = [f for f in fs.ls(path, detail=False)
                     if f.endswith(".parquet") or "/part-" in f]
            all_files.extend(files)
        else:
            all_files.append(path)

    print(f"Found {len(all_files)} parquet files")

    columns = ["parent_asin", "asin", "user_id", "rating", "title", "text"]
    rows = []
    for f in all_files:
        table = pq.read_table(f, filesystem=fs,
                              columns=[c for c in columns if c in
                                       pq.read_schema(f, filesystem=fs).names])
        rows.append(table)
        total = sum(len(t) for t in rows)
        if total >= max_rows:
            break

    import pyarrow as pa
    combined = pa.concat_tables(rows).to_pandas()

    if "asin" in combined.columns and "parent_asin" not in combined.columns:
        combined = combined.rename(columns={"asin": "parent_asin"})

    combined["embed_text"] = (
        combined.get("title", "").fillna("") + " " +
        combined.get("text", "").fillna("")
    ).str.strip()

    combined = combined[combined["embed_text"].str.len() >= 20]

    if len(combined) > max_rows:
        combined = combined.head(max_rows)

    combined["item_id"] = combined["parent_asin"].astype(str)
    combined["review_id"] = [str(i) for i in range(len(combined))]

    print(f"Loaded {len(combined):,} reviews")
    return combined


def ensure_collection(client: MilvusClient, collection_name: str):
    """Create Milvus collection if it doesn't exist."""
    if client.has_collection(collection_name):
        stats = client.get_collection_stats(collection_name)
        print(f"Collection '{collection_name}' exists: {stats}")
        return

    schema = CollectionSchema(fields=[
        FieldSchema("review_id", DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema("item_id", DataType.VARCHAR, max_length=64),
        FieldSchema("rating", DataType.FLOAT),
        FieldSchema("embed_text", DataType.VARCHAR, max_length=512),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ])

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )

    client.create_index(
        collection_name=collection_name,
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}},
    )
    print(f"Created collection '{collection_name}' with IVF_FLAT COSINE index")


def main():
    args = parse_args()

    print("=" * 60)
    print("SmartShop — Embeddings → Milvus (direct, no Feast)")
    print("=" * 60)
    print(f"  Input        : {args.input}")
    print(f"  Milvus       : {args.milvus_uri}")
    print(f"  Collection   : {args.collection}")
    print(f"  Max rows     : {args.max_rows:,}")
    print(f"  Batch size   : {args.batch_size:,}")
    print()

    # 1. Read reviews from S3
    t0 = time.time()
    df = read_reviews(args.input, args.max_rows)
    print(f"  Read time: {time.time() - t0:.1f}s")
    print()

    # 2. Generate embeddings
    print(f"Generating embeddings with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = df["embed_text"].tolist()

    all_embeddings = []
    for i in range(0, len(texts), args.embed_batch_size):
        batch = texts[i:i + args.embed_batch_size]
        embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.append(embs)
        if (i // args.embed_batch_size) % 50 == 0:
            print(f"  Encoded {min(i + args.embed_batch_size, len(texts)):,}/{len(texts):,}")

    embeddings = np.vstack(all_embeddings)
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Encode time: {time.time() - t0:.1f}s total")
    print()

    # 3. Connect to Milvus and ensure collection
    print(f"Connecting to Milvus at {args.milvus_uri}...")
    client = MilvusClient(uri=args.milvus_uri)
    ensure_collection(client, args.collection)
    print()

    # 4. Bulk insert in batches
    print("Inserting into Milvus...")
    t_insert = time.time()
    total_inserted = 0

    for i in range(0, len(df), args.batch_size):
        batch_df = df.iloc[i:i + args.batch_size]
        batch_emb = embeddings[i:i + args.batch_size]

        data = [
            {"review_id": row["review_id"],
             "item_id": row["item_id"],
             "rating": float(row.get("rating", 0) or 0),
             "embed_text": str(row["embed_text"])[:511],
             "embedding": batch_emb[j].tolist()}
            for j, (_, row) in enumerate(batch_df.iterrows())
        ]

        client.insert(collection_name=args.collection, data=data)
        total_inserted += len(data)
        print(f"  Inserted {total_inserted:,}/{len(df):,}")

    insert_time = time.time() - t_insert
    total_time = time.time() - t0

    print()
    print("=" * 60)
    print(f"Done — {total_inserted:,} embeddings in Milvus")
    print(f"  Insert time  : {insert_time:.1f}s")
    print(f"  Total time   : {total_time:.1f}s")
    print(f"  Throughput   : {total_inserted / total_time:,.0f} rows/s")
    print("=" * 60)

    # 5. Quick verification
    print("\nVerification — searching with random vector...")
    dummy = np.random.randn(EMBEDDING_DIM).tolist()
    results = client.search(
        collection_name=args.collection,
        data=[dummy],
        limit=3,
        output_fields=["item_id", "rating", "embed_text"],
    )
    for hit in results[0]:
        print(f"  score={hit['distance']:.3f} item={hit['entity']['item_id']} "
              f"text={hit['entity']['embed_text'][:60]}...")


if __name__ == "__main__":
    main()
