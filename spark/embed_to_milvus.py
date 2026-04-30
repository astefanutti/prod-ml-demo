"""Spark → Milvus: Generate embeddings and bulk-insert directly (no Feast).

Executors generate 384-dim embeddings via sentence-transformers pandas UDF.
Driver collects partitions via foreachPartition and inserts into Milvus via pymilvus.

Usage (SparkApplication):
    spark-submit spark/embed_to_milvus.py \
        --input s3a://smartshop-raw/raw/reviews/Electronics/ \
        --parquet-output s3a://smartshop-embeddings/review_embeddings/ \
        --milvus-uri http://milvus.smartshop.svc.cluster.local:19530 \
        --collection smartshop_review_embeddings \
        --max-reviews 1000000
"""

import argparse
import importlib
import os
import subprocess
import sys
import time

_PIP_TARGET = "/tmp/pip-pkgs"


def _pip_install(*packages):
    """Install packages to /tmp/pip-pkgs (works as non-root)."""
    os.makedirs(_PIP_TARGET, exist_ok=True)
    if _PIP_TARGET not in sys.path:
        sys.path.insert(0, _PIP_TARGET)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--no-cache-dir", "--target", _PIP_TARGET] + list(packages))


if importlib.util.find_spec("pymilvus") is None:
    print("Installing pymilvus + sentence-transformers on driver...")
    _pip_install("pymilvus", "sentence-transformers")

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient, utility

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

EMBEDDING_DIM = 384
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def make_embedding_udf():
    model_name = EMBEDDING_MODEL

    @F.pandas_udf(T.ArrayType(T.FloatType()))
    def embed_texts(texts):
        import importlib, os, subprocess, sys
        import pandas as pd
        target = "/tmp/pip-pkgs"
        if target not in sys.path:
            sys.path.insert(0, target)
        if importlib.util.find_spec("sentence_transformers") is None:
            os.makedirs(target, exist_ok=True)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet",
                 "--no-cache-dir", "--target", target, "sentence-transformers"])
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        text_list = texts.fillna("").tolist()
        embeddings = model.encode(text_list, batch_size=64, show_progress_bar=False)
        return pd.Series([emb.tolist() for emb in embeddings])
    return embed_texts


def ensure_collection(client: MilvusClient, name: str):
    if client.has_collection(name):
        stats = client.get_collection_stats(name)
        print(f"Collection '{name}' exists: {stats}")
        client.drop_collection(name)
        print(f"Dropped existing collection for fresh load")

    schema = CollectionSchema(fields=[
        FieldSchema("review_id", DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema("item_id", DataType.VARCHAR, max_length=64),
        FieldSchema("rating", DataType.FLOAT),
        FieldSchema("embed_text", DataType.VARCHAR, max_length=1024),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ])
    client.create_collection(collection_name=name, schema=schema)

    idx_params = client.prepare_index_params()
    idx_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128},
    )
    client.create_index(collection_name=name, index_params=idx_params)
    print(f"Created collection '{name}' with IVF_FLAT COSINE index")


def _truncate_to_bytes(s: str, max_bytes: int = 1020) -> str:
    """Truncate string so its UTF-8 byte length stays under max_bytes."""
    encoded = s.encode("utf-8")
    if len(encoded) <= max_bytes:
        return s
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def insert_to_milvus(client: MilvusClient, collection: str, df, batch_size: int):
    """Collect embeddings to driver and batch-insert into Milvus.

    Uses toLocalIterator to stream rows without loading entire DF into memory.
    """
    total = 0
    buf = []
    for row in df.toLocalIterator():
        emb = list(row["embedding"]) if row["embedding"] else [0.0] * EMBEDDING_DIM
        text = str(row["embed_text"]) if row["embed_text"] else ""
        buf.append({
            "review_id": str(row["review_id"]),
            "item_id": str(row["item_id"]) if row["item_id"] else "",
            "rating": float(row["rating"]) if row["rating"] else 0.0,
            "embed_text": _truncate_to_bytes(text),
            "embedding": emb,
        })
        if len(buf) >= batch_size:
            client.insert(collection_name=collection, data=buf)
            total += len(buf)
            if total % 50000 == 0:
                print(f"  [driver] Inserted {total:,} rows...")
            buf = []
    if buf:
        client.insert(collection_name=collection, data=buf)
        total += len(buf)
    print(f"  [driver] Total inserted: {total:,} rows")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--parquet-output", default=None,
                        help="Optional S3 path to also write parquet backup")
    parser.add_argument("--milvus-uri", required=True)
    parser.add_argument("--collection", default="smartshop_review_embeddings")
    parser.add_argument("--max-reviews", type=int, default=1_000_000)
    parser.add_argument("--milvus-batch-size", type=int, default=5000)
    args = parser.parse_args()

    job_start = time.time()

    spark = (SparkSession.builder
             .appName("SmartShop-Embed-to-Milvus")
             .config("spark.sql.adaptive.enabled", "true")
             .config("spark.sql.shuffle.partitions", "200")
             .getOrCreate())

    print("=" * 60)
    print("SmartShop — Spark Embeddings → Milvus (direct)")
    print("=" * 60)
    print(f"  Input       : {args.input}")
    print(f"  Milvus      : {args.milvus_uri}")
    print(f"  Collection  : {args.collection}")
    print(f"  Max reviews : {args.max_reviews:,}")
    print()

    # --- Read raw reviews ---
    input_paths = [p.strip() for p in args.input.split(",") if p.strip()]
    df = spark.read.parquet(*input_paths)

    if "asin" in df.columns and "parent_asin" not in df.columns:
        df = df.withColumnRenamed("asin", "parent_asin")

    df = df.withColumn(
        "embed_text",
        F.concat_ws(" ",
                     F.coalesce(F.col("title"), F.lit("")),
                     F.coalesce(F.col("text"), F.lit(""))),
    ).filter(F.length(F.col("embed_text")) >= 20)

    if args.max_reviews:
        df = df.limit(args.max_reviews).repartition(200)

    df = df.select(
        F.monotonically_increasing_id().cast("string").alias("review_id"),
        F.col("parent_asin").alias("item_id"),
        F.col("rating").cast("float"),
        F.substring(F.col("embed_text"), 1, 500).alias("embed_text"),
    )

    # --- Generate embeddings (distributed across executors) ---
    print(f"Generating embeddings with {EMBEDDING_MODEL}...")
    embed_udf = make_embedding_udf()
    df = df.withColumn("embedding", embed_udf(F.col("embed_text")))

    # Cache so we don't re-compute for parquet + Milvus writes
    df.cache()
    total_rows = df.count()
    print(f"  Total rows with embeddings: {total_rows:,}")
    embed_time = time.time() - job_start
    print(f"  Embedding time: {embed_time:.1f}s ({total_rows / embed_time:,.0f} rows/s)")

    # --- Write parquet backup ---
    if args.parquet_output:
        print(f"\nWriting parquet to {args.parquet_output}...")
        t_pq = time.time()
        df.write.mode("overwrite").parquet(args.parquet_output)
        print(f"  Parquet write: {time.time() - t_pq:.1f}s")

    # --- Create Milvus collection (from driver) ---
    print(f"\nConnecting to Milvus at {args.milvus_uri}...")
    client = MilvusClient(uri=args.milvus_uri)
    ensure_collection(client, args.collection)

    # --- Write to Milvus (driver-side collection + batch insert) ---
    print(f"\nInserting into Milvus from driver (batch_size={args.milvus_batch_size})...")
    t_mv = time.time()
    insert_to_milvus(client, args.collection, df, args.milvus_batch_size)
    milvus_time = time.time() - t_mv
    print(f"  Milvus insert: {milvus_time:.1f}s")

    # --- Verify ---
    stats = client.get_collection_stats(args.collection)
    total_time = time.time() - job_start

    print()
    print("=" * 60)
    print(f"Done — embeddings in Milvus")
    print(f"  Collection   : {args.collection}")
    print(f"  Milvus stats : {stats}")
    print(f"  Total time   : {total_time:.1f}s")
    print(f"  Throughput   : {total_rows / total_time:,.0f} rows/s")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
