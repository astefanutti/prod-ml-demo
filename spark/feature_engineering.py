"""Spark Job A: Structured Feature Engineering.

Reads raw Amazon Reviews parquet files and computes:
  - User features: avg_rating, review_count, category_preferences
  - Item features: avg_rating, price_bucket, review_volume, avg_helpful_votes
  - Interaction pairs: user-item co-occurrence for training

Outputs features as Parquet files compatible with Feast offline store.

Usage:
    spark-submit spark/feature_engineering.py \
        --input s3a://smartshop/raw/ \
        --output s3a://smartshop/features/
"""

import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


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
            F.datediff("user_last_active", "user_first_active"),
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
    item_features = item_features.withColumn(
        "item_price_bucket",
        F.when(F.col("item_price") < 10, "budget")
        .when(F.col("item_price") < 50, "mid")
        .when(F.col("item_price") < 200, "premium")
        .when(F.col("item_price").isNotNull(), "luxury")
        .otherwise("unknown"),
    ).withColumn("event_timestamp", F.current_timestamp())

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input path (S3 or local)")
    parser.add_argument("--output", required=True, help="Output path for features")
    parser.add_argument("--metadata-input", default=None, help="Metadata parquet path")
    args = parser.parse_args()

    spark = create_spark_session()

    print(f"Reading reviews from {args.input}")
    reviews_df = spark.read.parquet(args.input)

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
    print(f"Total reviews: {total_reviews:,}")

    # Load metadata if provided
    metadata_df = None
    if args.metadata_input:
        print(f"Reading metadata from {args.metadata_input}")
        metadata_df = spark.read.parquet(args.metadata_input)

    # Compute features
    print("Computing user features...")
    user_features = compute_user_features(reviews_df)
    user_features.write.parquet(f"{args.output}/user_features", mode="overwrite")
    print(f"  Users: {user_features.count():,}")

    print("Computing item features...")
    item_features = compute_item_features(reviews_df, metadata_df)
    item_features.write.parquet(f"{args.output}/item_features", mode="overwrite")
    print(f"  Items: {item_features.count():,}")

    print("Computing interactions...")
    interactions = compute_interactions(reviews_df)
    interactions.write.parquet(f"{args.output}/interactions", mode="overwrite")
    print(f"  Interactions: {interactions.count():,}")

    spark.stop()
    print("Feature engineering complete.")


if __name__ == "__main__":
    main()
