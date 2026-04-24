"""Offline feature fetch utility — generates a training-ready parquet via Feast.

Decouples feature fetching from training so train.py just reads a flat parquet.
Run this once before kicking off TrainJob; all DDP ranks then load the same file.

What it does:
  1. Reads interactions (user_id, item_id, event_timestamp, rating) from MinIO
  2. Derives a binary label (rating >= 4 → positive)
  3. Calls store.get_historical_features() for user_features + item_features
     (point-in-time correct via SparkOfflineStore)
  4. Optionally joins pre-aggregated item_text_embedding (mean-pooled 384-dim)
     from the embedding parquet — disabled by default, ready for post-Summit
  5. Writes the merged DataFrame to s3://smartshop-features/training_dataset/

The resulting parquet schema is what train.py --no-feast reads directly:
  user_id, item_id, event_timestamp, label,
  user_avg_rating, user_review_count, user_unique_items,
  user_avg_review_length, user_category_count, user_tenure_days,
  item_avg_rating, item_rating_stddev, item_review_count,
  item_total_helpful_votes, item_avg_review_length, item_price
  [optional] item_embedding_avg: list[float] (384-dim, --include-embeddings)

Usage:
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_ENDPOINT_URL_S3=http://minio.smartshop.svc.cluster.local:9000
    export FEAST_REPO_PATH=feast/feature_repo

    # Standard: user + item features only
    python scripts/fetch_training_features.py \\
        --interactions s3://smartshop-features/interactions/ \\
        --output s3://smartshop-features/training_dataset/

    # With item text embeddings (post-Summit, requires embedding job done)
    python scripts/fetch_training_features.py \\
        --interactions s3://smartshop-features/interactions/ \\
        --embeddings  s3://smartshop-embeddings/review_embeddings/ \\
        --output      s3://smartshop-features/training_dataset/ \\
        --include-embeddings

    # Dry run (fetch features, print schema, don't write)
    python scripts/fetch_training_features.py --dry-run

Requirements:
    pip install feast[spark,redis] pandas pyarrow s3fs pyspark
"""

import argparse
import os
import sys
import time

import pandas as pd

# ---------------------------------------------------------------------------
# Feature columns — must match features.py exactly
# ---------------------------------------------------------------------------
_USER_FEAT_COLS = [
    "user_avg_rating",
    "user_review_count",
    "user_unique_items",
    "user_avg_review_length",
    "user_category_count",
    "user_tenure_days",
]
_ITEM_FEAT_COLS = [
    "item_avg_rating",
    "item_rating_stddev",
    "item_review_count",
    "item_total_helpful_votes",
    "item_avg_review_length",
    "item_price",
]
_FEAST_FEATURES = (
    [f"user_features:{c}" for c in _USER_FEAT_COLS]
    + [f"item_features:{c}" for c in _ITEM_FEAT_COLS]
)

EMBEDDING_DIM = 384
POSITIVE_RATING_THRESHOLD = 4  # rating >= 4 → label=1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch training features from Feast offline store"
    )
    parser.add_argument(
        "--interactions",
        default=os.environ.get(
            "INTERACTIONS_PATH",
            "s3://smartshop-features/interactions/",
        ),
        help="S3 path to interactions parquet (output of Spark Job A)",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get(
            "TRAINING_DATASET_PATH",
            "s3://smartshop-features/training_dataset/",
        ),
        help="S3 path to write training-ready parquet",
    )
    parser.add_argument(
        "--feast-repo",
        default=os.environ.get("FEAST_REPO_PATH", "feast/feature_repo"),
        help="Path to feast feature_repo (default: feast/feature_repo)",
    )
    parser.add_argument(
        "--embeddings",
        default=os.environ.get(
            "EMBEDDINGS_PARQUET_PATH",
            "s3://smartshop-embeddings/review_embeddings/",
        ),
        help="S3 path to review_embeddings parquet (Spark Job C output)",
    )
    parser.add_argument(
        "--include-embeddings",
        action="store_true",
        default=False,
        help=(
            "Join mean-pooled item embeddings as item_embedding_avg column. "
            "Requires --embeddings path to exist. "
            "Disabled by default — rec model architecture change needed first."
        ),
    )
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=None,
        help="Cap interactions for smoke testing (e.g. --max-interactions 10000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch features, print schema + stats, do not write output",
    )
    return parser.parse_args()


def s3_storage_options() -> dict:
    endpoint = os.environ.get(
        "AWS_ENDPOINT_URL_S3",
        "http://minio.smartshop.svc.cluster.local:9000",
    )
    return {
        "key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
        "client_kwargs": {"endpoint_url": endpoint},
    }


def load_interactions(path: str, max_rows: int | None) -> pd.DataFrame:
    """Read interactions parquet (Job A output) and derive binary label."""
    print(f"Reading interactions from {path} ...")
    opts = s3_storage_options() if path.startswith("s3") else {}
    df = pd.read_parquet(path, storage_options=opts or None)

    # Normalise column names — Job A may output 'parent_asin' or 'item_id'
    if "parent_asin" in df.columns and "item_id" not in df.columns:
        df = df.rename(columns={"parent_asin": "item_id"})

    # Ensure event_timestamp is UTC-aware (required by Feast point-in-time join)
    if "event_timestamp" not in df.columns:
        if "timestamp" in df.columns:
            df["event_timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        else:
            raise ValueError("interactions parquet has no 'event_timestamp' or 'timestamp' column")
    else:
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)

    # Binary label: rating >= threshold → positive interaction
    if "label" not in df.columns:
        if "rating" not in df.columns:
            raise ValueError("interactions parquet has no 'rating' or 'label' column")
        df["label"] = (df["rating"] >= POSITIVE_RATING_THRESHOLD).astype(float)

    required = {"user_id", "item_id", "event_timestamp", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Interactions missing required columns: {missing}")

    if max_rows:
        df = df.head(max_rows)

    print(f"  {len(df):,} interactions | positive rate: {df['label'].mean():.1%}")
    return df[["user_id", "item_id", "event_timestamp", "label", "rating"]]


def fetch_feast_features(interactions_df: pd.DataFrame, feast_repo: str) -> pd.DataFrame:
    """Call store.get_historical_features() — point-in-time correct join via SparkOfflineStore."""
    try:
        from feast import FeatureStore
    except ImportError:
        print("ERROR: feast not installed. Run: pip install 'feast[spark,redis]'")
        sys.exit(1)

    print(f"\nConnecting to Feast (repo: {feast_repo}) ...")
    store = FeatureStore(repo_path=feast_repo)
    print(f"  Project  : {store.project}")
    print(f"  Registry : {store.config.registry}")

    entity_df = interactions_df[["user_id", "item_id", "event_timestamp", "label"]].copy()

    print(f"\nCalling get_historical_features() for {len(entity_df):,} rows ...")
    print(f"  Features : {_FEAST_FEATURES}")
    t0 = time.time()

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=_FEAST_FEATURES,
    ).to_df()

    elapsed = time.time() - t0
    print(f"  Retrieved {len(training_df):,} rows in {elapsed:.1f}s")

    # Fill NaN features — entities with no matching feature row get 0
    for col in _USER_FEAT_COLS + _ITEM_FEAT_COLS:
        if col not in training_df.columns:
            training_df[col] = 0.0
        else:
            training_df[col] = training_df[col].fillna(0.0)

    null_frac = training_df[_USER_FEAT_COLS + _ITEM_FEAT_COLS].isnull().mean().mean()
    if null_frac > 0:
        print(f"  WARNING: {null_frac:.1%} null values after fill — check Feast materialization")

    return training_df


def compute_item_embeddings(embeddings_path: str) -> pd.DataFrame:
    """Mean-pool review embeddings per item_id → one 384-dim vector per item.

    This is the bridge between Job C output and the item tower.
    NOT used in current rec model training — item_feat_dim is hardcoded to 6.
    To use this: update TwoTowerModel to accept item_embedding_avg as an
    additional concat input after the hand-crafted item features.

    Returns DataFrame with columns: [item_id, item_embedding_avg: list[float]]
    """
    import numpy as np
    import pyarrow.parquet as pq

    print(f"\nLoading embeddings from {embeddings_path} for item mean-pooling ...")
    opts = s3_storage_options() if embeddings_path.startswith("s3") else {}

    if embeddings_path.startswith("s3://") or embeddings_path.startswith("s3a://"):
        import s3fs
        path = embeddings_path.replace("s3a://", "s3://")
        endpoint = os.environ.get(
            "AWS_ENDPOINT_URL_S3",
            "http://minio.smartshop.svc.cluster.local:9000",
        )
        fs = s3fs.S3FileSystem(
            key=os.environ["AWS_ACCESS_KEY_ID"],
            secret=os.environ["AWS_SECRET_ACCESS_KEY"],
            endpoint_url=endpoint,
            client_kwargs={"verify": False},
        )
        dataset = pq.ParquetDataset(path.replace("s3://", ""), filesystem=fs)
    else:
        dataset = pq.ParquetDataset(embeddings_path)

    # Stream in chunks to avoid loading all 5M rows into memory at once
    item_sums: dict[str, list] = {}
    item_counts: dict[str, int] = {}

    for fragment in dataset.fragments:
        table = fragment.to_table(columns=["item_id", "embedding"])
        df_chunk = table.to_pandas()

        for _, row in df_chunk.iterrows():
            iid = str(row["item_id"])
            emb = np.array(row["embedding"], dtype=np.float32)
            if iid not in item_sums:
                item_sums[iid] = emb
                item_counts[iid] = 1
            else:
                item_sums[iid] += emb
                item_counts[iid] += 1

    print(f"  Mean-pooling embeddings for {len(item_sums):,} unique items ...")
    records = []
    for iid, emb_sum in item_sums.items():
        mean_emb = (emb_sum / item_counts[iid]).tolist()
        records.append({"item_id": iid, "item_embedding_avg": mean_emb})

    result_df = pd.DataFrame(records)
    print(f"  Done — {len(result_df):,} item embeddings (dim={EMBEDDING_DIM})")
    return result_df


def write_output(df: pd.DataFrame, output_path: str):
    """Write training dataset parquet to S3 or local path."""
    opts = s3_storage_options() if output_path.startswith("s3") else {}
    print(f"\nWriting {len(df):,} rows → {output_path}")
    df.to_parquet(output_path, index=False, storage_options=opts or None)
    print("  Done.")


def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("Training dataset summary")
    print("=" * 60)
    print(f"  Rows        : {len(df):,}")
    print(f"  Columns     : {len(df.columns)}")
    print(f"  Users       : {df['user_id'].nunique():,}")
    print(f"  Items       : {df['item_id'].nunique():,}")
    print(f"  Positive rate: {df['label'].mean():.1%}")
    print()
    print("  Feature null rates:")
    for col in _USER_FEAT_COLS + _ITEM_FEAT_COLS:
        if col in df.columns:
            null_rate = df[col].isnull().mean()
            flag = " ⚠" if null_rate > 0.05 else ""
            print(f"    {col:<35} {null_rate:.1%}{flag}")
    if "item_embedding_avg" in df.columns:
        print(f"    item_embedding_avg (dim={EMBEDDING_DIM})      present ✓")
    print()
    print("  Schema:")
    print(df.dtypes.to_string())


def main():
    args = parse_args()

    print("=" * 60)
    print("SmartShop — Fetch offline training features from Feast")
    print("=" * 60)
    print(f"  Interactions : {args.interactions}")
    print(f"  Feast repo   : {args.feast_repo}")
    print(f"  Output       : {args.output}")
    print(f"  Embeddings   : {'enabled — ' + args.embeddings if args.include_embeddings else 'disabled'}")
    print(f"  Max rows     : {args.max_interactions or 'all'}")
    print(f"  Dry run      : {args.dry_run}")
    print()

    # 1. Load interactions + derive labels
    interactions_df = load_interactions(args.interactions, args.max_interactions)

    # 2. Feast get_historical_features → user + item features (point-in-time correct)
    training_df = fetch_feast_features(interactions_df, args.feast_repo)

    # 3. Optionally join mean-pooled item embeddings
    if args.include_embeddings:
        print("\n[--include-embeddings] Computing item_embedding_avg ...")
        print(
            "  NOTE: TwoTowerModel.item_feat_dim must be updated to accept this column. "
            "Currently hardcoded to 6 (hand-crafted features only)."
        )
        item_emb_df = compute_item_embeddings(args.embeddings)
        before = len(training_df)
        training_df = training_df.merge(item_emb_df, on="item_id", how="left")
        matched = training_df["item_embedding_avg"].notna().sum()
        print(f"  Joined {matched:,}/{before:,} rows with item embeddings")

    # 4. Summary
    print_summary(training_df)

    # 5. Write
    if not args.dry_run:
        write_output(training_df, args.output)
        print()
        print("Next steps:")
        print(f"  torchrun --nproc_per_node=4 training/recommendation/train.py \\")
        print(f"    --no-feast \\")
        print(f"    --data-dir {args.output}")
        print()
        print("  Or pass --data-dir to point train.py directly at this output.")
    else:
        print("\nDry run — nothing written.")


if __name__ == "__main__":
    main()
