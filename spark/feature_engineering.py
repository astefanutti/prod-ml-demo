"""Spark Job A: Structured Feature Engineering.

Reads raw Amazon Reviews parquet files and computes:
  - User features: avg_rating, review_count, category_preferences
  - Item features: avg_rating, price_bucket, review_volume, avg_helpful_votes
  - Interaction pairs: user-item co-occurrence for training

Outputs features as Parquet files compatible with Feast offline store.

Usage:
    spark-submit spark/feature_engineering.py \
        --input s3a://smartshop-raw/ \
        --output s3a://smartshop-features/
"""

import argparse
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

try:
    from utils.mlflow_metrics import SparkRunLogger
except ImportError:
    SparkRunLogger = None  # graceful fallback when running outside the image


def _log_metric(label: str, value, unit: str = "") -> None:
    """Emit a structured metric line for easy grep in pod logs and Spark history."""
    unit_str = f" {unit}" if unit else ""
    print(f"[METRIC] {label}={value}{unit_str}")


def create_spark_session(app_name: str = "SmartShop-FeatureEngineering") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def compute_user_features(reviews_df):
    """Compute per-user aggregate features."""
    return (
        reviews_df.groupBy("user_id")
        .agg(
            F.avg("rating").alias("user_avg_rating"),
            F.count("*").alias("user_review_count"),
            F.countDistinct("parent_asin").alias("user_unique_items"),
            F.avg(F.length("text")).alias("user_avg_review_length"),
            F.collect_set("category").alias("user_categories"),
            F.max("timestamp").alias("user_last_active"),
            F.min("timestamp").alias("user_first_active"),
        )
        .withColumn("user_category_count", F.size("user_categories"))
        .withColumn(
            "user_tenure_days",
            # timestamp is Unix ms epoch (BIGINT) → convert diff to days
            ((F.col("user_last_active") - F.col("user_first_active")) / 86_400_000).cast("int"),
        )
        .drop("user_categories")
        .withColumn("event_timestamp", F.current_timestamp())
    )


def compute_item_features(reviews_df, metadata_df):
    """Compute per-item aggregate features from reviews and metadata."""
    review_aggs = reviews_df.groupBy("parent_asin").agg(
        F.avg("rating").alias("item_avg_rating"),
        F.stddev("rating").alias("item_rating_stddev"),
        F.count("*").alias("item_review_count"),
        F.sum(F.col("helpful_vote").cast("int")).alias("item_total_helpful_votes"),
        F.avg(F.length("text")).alias("item_avg_review_length"),
        F.first("category").alias("category"),
    )

    # Join with metadata for price info
    if metadata_df is not None:
        item_features = review_aggs.join(
            metadata_df.select(
                F.col("parent_asin"),
                F.col("price").cast("float").alias("item_price"),
                F.col("average_rating").alias("item_meta_rating"),
                F.col("rating_number").alias("item_meta_rating_count"),
            ),
            on="parent_asin",
            how="left",
        )
    else:
        item_features = review_aggs.withColumn("item_price", F.lit(None).cast("float"))

    # Price bucketing
    item_features = (
        item_features
        .withColumn("event_timestamp", F.current_timestamp())
        # Rename to match Feast entity key — Feast entity is `item_id`, not `parent_asin`
        .withColumnRenamed("parent_asin", "item_id")
        # Drop string columns not consumed by the TwoTower model
        .drop("item_price_bucket", "category", "item_meta_rating", "item_meta_rating_count")
    )

    return item_features


def compute_interactions(reviews_df):
    """Build user-item interaction pairs for recommendation training."""
    return (
        reviews_df.select(
            F.col("user_id"),
            F.col("parent_asin").alias("item_id"),
            F.col("rating"),
            F.col("timestamp").alias("event_timestamp"),
            # Binary label: rating >= 4 is positive
            F.when(F.col("rating") >= 4, 1.0).otherwise(0.0).alias("label"),
        )
    )


def _rapids_active(spark: SparkSession) -> bool:
    """Return True if the RAPIDS SQL plugin is loaded and enabled."""
    try:
        plugins = spark.conf.get("spark.plugins", "")
        enabled = spark.conf.get("spark.rapids.sql.enabled", "false")
        return "com.nvidia.spark.SQLPlugin" in plugins and enabled.lower() == "true"
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input path (S3 or local)")
    parser.add_argument("--output", required=True, help="Output path for features")
    parser.add_argument("--metadata-input", default=None, help="Metadata parquet path")
    args = parser.parse_args()

    job_start = time.time()
    spark = create_spark_session()

    gpu_mode = _rapids_active(spark)
    run_name = f"{'rapids' if gpu_mode else 'cpu'}-{spark.conf.get('spark.executor.instances', '?')}ex"
    mlflow_logger = SparkRunLogger(spark, experiment="smartshop-feature-engineering") if SparkRunLogger else None

    _log_metric("gpu_accelerated", gpu_mode)
    _log_metric("spark_app_name", spark.sparkContext.appName)
    print(f"GPU/RAPIDS acceleration: {'ON' if gpu_mode else 'OFF (CPU mode)'}")

    _run_ctx = mlflow_logger.start_run(run_name=run_name) if mlflow_logger else None
    if _run_ctx:
        _run_ctx.__enter__()

    def _metric(key: str, value, unit: str = "") -> None:
        _log_metric(key, value, unit)
        if mlflow_logger:
            mlflow_logger.log_metric(key, value)

    print(f"Reading reviews from {args.input}")
    read_start = time.time()
    # Support comma-separated paths for multi-directory reads (non-Hive-partitioned shards)
    input_paths = [p.strip() for p in args.input.split(",") if p.strip()]
    reviews_df = spark.read.parquet(*input_paths)

    # Normalize column names
    if "asin" in reviews_df.columns and "parent_asin" not in reviews_df.columns:
        reviews_df = reviews_df.withColumnRenamed("asin", "parent_asin")

    # Ensure timestamp column exists
    if "timestamp" not in reviews_df.columns and "unixReviewTime" in reviews_df.columns:
        reviews_df = reviews_df.withColumn(
            "timestamp", F.from_unixtime("unixReviewTime").cast("timestamp")
        )

    # Add category from file path if not present
    if "category" not in reviews_df.columns:
        reviews_df = reviews_df.withColumn(
            "category", F.input_file_name()
        )

    # Ensure helpful_vote exists
    if "helpful_vote" not in reviews_df.columns:
        reviews_df = reviews_df.withColumn("helpful_vote", F.lit(0))

    reviews_df.cache()
    total_reviews = reviews_df.count()
    read_elapsed = time.time() - read_start
    print(f"Total reviews: {total_reviews:,}")
    _metric("total_reviews", total_reviews)
    _metric("read_elapsed_s", round(read_elapsed, 2), "s")

    # Load metadata if provided
    metadata_df = None
    if args.metadata_input:
        print(f"Reading metadata from {args.metadata_input}")
        metadata_df = spark.read.parquet(args.metadata_input)

    # Compute features
    print("Computing user features...")
    t = time.time()
    user_features = compute_user_features(reviews_df)
    user_features.write.parquet(f"{args.output}/user_features", mode="overwrite")
    n_users = user_features.count()
    _metric("unique_users", n_users)
    _metric("user_features_elapsed_s", round(time.time() - t, 2), "s")
    print(f"  Users: {n_users:,}")

    print("Computing item features...")
    t = time.time()
    item_features = compute_item_features(reviews_df, metadata_df)
    item_features.write.parquet(f"{args.output}/item_features", mode="overwrite")
    n_items = item_features.count()
    _metric("unique_items", n_items)
    _metric("item_features_elapsed_s", round(time.time() - t, 2), "s")
    print(f"  Items: {n_items:,}")

    print("Computing interactions...")
    t = time.time()
    interactions = compute_interactions(reviews_df)
    interactions.write.parquet(f"{args.output}/interactions", mode="overwrite")
    n_interactions = interactions.count()
    _metric("total_interactions", n_interactions)
    _metric("interactions_elapsed_s", round(time.time() - t, 2), "s")
    print(f"  Interactions: {n_interactions:,}")

    job_elapsed = time.time() - job_start
    throughput = int(total_reviews / job_elapsed) if job_elapsed > 0 else 0
    _metric("total_elapsed_s", round(job_elapsed, 2), "s")
    _metric("throughput_rows_per_s", throughput, "rows/s")

    if mlflow_logger:
        mlflow_logger.log_rapids_coverage()
        mlflow_logger.finalize(output_path="/tmp/feature_engineering_metrics.json")
    if _run_ctx:
        _run_ctx.__exit__(None, None, None)

    print(
        f"\n{'='*60}\n"
        f"Feature engineering complete.\n"
        f"  Mode     : {'GPU (RAPIDS)' if gpu_mode else 'CPU'}\n"
        f"  Reviews  : {total_reviews:,}\n"
        f"  Users    : {n_users:,}  |  Items: {n_items:,}\n"
        f"  Elapsed  : {job_elapsed:.1f}s\n"
        f"  Throughput: {throughput:,} rows/s\n"
        f"{'='*60}"
    )

    spark.stop()


if __name__ == "__main__":
    main()
