"""Load review embeddings from S3 into Milvus via Feast.

NOTE: This script is the legacy bulk-load path.
      The preferred approach is now feast materialize, which runs the Spark
      embedding job and Milvus write in a single command:

          feast -c feast/feature_repo/feature_store_milvus.yaml \\
              materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S) \\
              --feature-views review_embeddings

      The review_embeddings @batch_feature_view in features.py handles:
        - Reading raw reviews from s3a://smartshop-raw/raw/reviews/*/
        - Generating embeddings via all-MiniLM-L6-v2 pandas UDF
        - Writing to Milvus (no separate Spark job or this script needed)

Use this script ONLY if you have a pre-computed review_embeddings parquet
(e.g. from an earlier spark-application-embedding.yaml run) and want to skip
re-embedding by loading directly from the parquet.

Usage:
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_ENDPOINT_URL_S3=http://minio.smartshop.svc.cluster.local:9000

    python scripts/load_embeddings_to_milvus.py \\
        --parquet-path s3://smartshop-embeddings/review_embeddings/ \\
        --feast-repo feast/feature_repo \\
        --batch-size 50000

Requirements:
    pip install feast[milvus] pandas pyarrow s3fs
"""

import argparse
import os
import sys
import time

import pandas as pd
import pyarrow.parquet as pq

EMBEDDING_DIM = 384


def parse_args():
    parser = argparse.ArgumentParser(description="Load review embeddings → Milvus via Feast")
    parser.add_argument(
        "--parquet-path",
        default=os.environ.get(
            "EMBEDDINGS_PARQUET_PATH",
            "s3://smartshop-embeddings/review_embeddings/",
        ),
        help="S3/MinIO path to review_embeddings parquet (folder or glob)",
    )
    parser.add_argument(
        "--feast-repo",
        default=os.environ.get("FEAST_REPO_PATH", "feast/feature_repo"),
        help="Path to feast feature_repo directory",
    )
    parser.add_argument(
        "--feast-config",
        default="feature_store_milvus.yaml",
        help="Feast config filename inside --feast-repo (default: feature_store_milvus.yaml)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Rows per write_to_online_store batch (default: 50000)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap total rows loaded (useful for smoke testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and validate parquet only — do not write to Milvus",
    )
    return parser.parse_args()


def get_s3_filesystem(endpoint_url: str):
    """Return an s3fs filesystem pointed at MinIO."""
    import s3fs

    return s3fs.S3FileSystem(
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        endpoint_url=endpoint_url,
        client_kwargs={"verify": False},
    )


def iter_parquet_batches(parquet_path: str, batch_size: int, max_rows: int | None):
    """Yield pandas DataFrames from a parquet dataset (local or S3)."""
    endpoint = os.environ.get(
        "AWS_ENDPOINT_URL_S3",
        "http://minio.smartshop.svc.cluster.local:9000",
    )

    if parquet_path.startswith("s3://") or parquet_path.startswith("s3a://"):
        # Normalise s3a → s3 for s3fs
        path = parquet_path.replace("s3a://", "s3://")
        fs = get_s3_filesystem(endpoint)
        dataset = pq.ParquetDataset(path.replace("s3://", ""), filesystem=fs)
    else:
        dataset = pq.ParquetDataset(parquet_path)

    total_yielded = 0
    for fragment in dataset.fragments:
        table = fragment.to_table()
        for batch in table.to_batches(max_chunksize=batch_size):
            df = batch.to_pandas()
            if max_rows is not None:
                remaining = max_rows - total_yielded
                if remaining <= 0:
                    return
                df = df.head(remaining)
            yield df
            total_yielded += len(df)


def validate_schema(df: pd.DataFrame):
    required = {"review_id", "item_id", "user_id", "rating", "embed_text", "embedding", "event_timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing expected columns: {missing}")

    sample_emb = df["embedding"].iloc[0]
    if not hasattr(sample_emb, "__len__") or len(sample_emb) != EMBEDDING_DIM:
        raise ValueError(
            f"embedding column must be list of length {EMBEDDING_DIM}, "
            f"got length {len(sample_emb) if hasattr(sample_emb, '__len__') else type(sample_emb)}"
        )
    print(f"  Schema OK — {len(df.columns)} columns, embedding dim={len(sample_emb)}")


def load_feast_store(feast_repo: str, config_file: str):
    """Load FeatureStore using the Milvus-specific config yaml.

    Uses fs_yaml_file parameter (Feast 0.60+) to load a named config file
    directly rather than requiring it to be named feature_store.yaml.
    """
    from pathlib import Path
    import feast

    config_path = os.path.join(feast_repo, config_file)
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Feast config not found: {config_path}\n"
            f"Expected at feast/feature_repo/feature_store_milvus.yaml"
        )
    store = feast.FeatureStore(fs_yaml_file=Path(config_path))
    return store


def main():
    args = parse_args()

    print("=" * 60)
    print("SmartShop — Load review embeddings → Milvus")
    print("=" * 60)
    print(f"  Parquet path : {args.parquet_path}")
    print(f"  Feast repo   : {args.feast_repo}")
    print(f"  Config       : {args.feast_config}")
    print(f"  Batch size   : {args.batch_size:,}")
    print(f"  Max rows     : {args.max_rows or 'all'}")
    print(f"  Dry run      : {args.dry_run}")
    print()

    # ------------------------------------------------------------------ #
    # 1. Initialise Feast with Milvus config
    # ------------------------------------------------------------------ #
    if not args.dry_run:
        print("Connecting to Feast (Milvus online store)...")
        store = load_feast_store(args.feast_repo, args.feast_config)
        print(f"  Project: {store.project}")
        print(f"  Online store: {store.config.online_store.type}")
        print()

        # Verify review_embeddings is in the registry — it must have been registered
        # via `feast apply` before this script is run for the first time.
        try:
            store.get_feature_view("review_embeddings")
            print("  review_embeddings feature view found in registry ✓")
        except Exception:
            config_path = os.path.join(args.feast_repo, args.feast_config)
            print(
                f"\nERROR: 'review_embeddings' not found in the Feast registry.\n"
                f"Run this first:\n\n"
                f"  feast -c {config_path} apply\n\n"
                f"Then re-run this script."
            )
            sys.exit(1)
        print()

    # ------------------------------------------------------------------ #
    # 2. Stream parquet → write in batches
    # ------------------------------------------------------------------ #
    total_written = 0
    batch_num = 0
    t_start = time.time()

    for df in iter_parquet_batches(args.parquet_path, args.batch_size, args.max_rows):
        batch_num += 1

        if batch_num == 1:
            print(f"First batch ({len(df):,} rows) — validating schema...")
            validate_schema(df)
            print()

        # Ensure review_id is a string — Feast entity keys must be string-serialisable
        df["review_id"] = df["review_id"].astype(str)

        # Milvus online store hardcodes VARCHAR max_length=512 for all non-vector fields.
        # Amazon reviews easily exceed this — truncate to avoid upsert errors.
        for col in ("embed_text", "review_title"):
            if col in df.columns:
                df[col] = df[col].str.slice(0, 511)

        # embedding must be list[float], not numpy array
        if hasattr(df["embedding"].iloc[0], "tolist"):
            df["embedding"] = df["embedding"].apply(lambda x: x.tolist())

        t_batch = time.time()

        if not args.dry_run:
            store.write_to_online_store(
                feature_view_name="review_embeddings",
                df=df,
                transform_on_write=False,  # embeddings are pre-computed by Spark job
            )

        elapsed_batch = time.time() - t_batch
        total_written += len(df)
        rate = len(df) / elapsed_batch if elapsed_batch > 0 else 0

        print(
            f"  Batch {batch_num:3d} | {len(df):>7,} rows | "
            f"{elapsed_batch:5.1f}s | {rate:,.0f} rows/s | "
            f"total={total_written:,}"
        )

    total_elapsed = time.time() - t_start

    print()
    print("=" * 60)
    if args.dry_run:
        print(f"Dry run complete — {total_written:,} rows validated, nothing written")
    else:
        print(f"Done — {total_written:,} embeddings loaded into Milvus")
    print(f"  Total elapsed : {total_elapsed:.1f}s")
    print(f"  Avg throughput: {total_written / total_elapsed:,.0f} rows/s")
    print()

    if not args.dry_run and total_written > 0:
        print("Smoke test — retrieve top-1 for a dummy query vector...")
        import random

        dummy_query = [random.uniform(-1, 1) for _ in range(EMBEDDING_DIM)]
        result = store.retrieve_online_documents_v2(
            features=[
                "review_embeddings:embedding",
                "review_embeddings:embed_text",
                "review_embeddings:item_id",
                "review_embeddings:rating",
            ],
            query=dummy_query,
            top_k=1,
            distance_metric="COSINE",
        )
        df_result = result.to_df()
        if df_result.empty:
            print("  WARNING: retrieve returned 0 results — check Milvus index build")
            sys.exit(1)
        else:
            print(f"  OK — top result: item_id={df_result['review_embeddings__item_id'].iloc[0]}")
            print(f"       embed_text={str(df_result['review_embeddings__embed_text'].iloc[0])[:80]}...")

    print()
    print("Next steps:")
    print("  Verify: oc exec -it <feast-pod> -- feast feature-views list")
    print("  RAG server can now call retrieve_online_documents_v2()")


if __name__ == "__main__":
    main()
