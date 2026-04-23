"""Feast feature definitions for SmartShop AI — @batch_feature_view edition.

Pipeline:
  Raw Parquet (s3a://smartshop-raw/raw/reviews/)
      ↓  SparkComputeEngine (k8s executor pods)
  @batch_feature_view transformation UDFs
      ↓
  Redis online store  (online=True)
  (no write-back to raw source — offline=False, see FEAST-BFV-DESIGN.md §4)

Design reference: docs/FEAST-BFV-DESIGN.md
"""

from datetime import timedelta

from feast import Entity
from feast.batch_feature_view import batch_feature_view
from feast.field import Field
from feast.infra.offline_stores.contrib.spark_offline_store.spark_source import (
    SparkSource,
)
from feast.transformation.mode import TransformationMode
from feast.types import Float32, Float64, Int64, String  # noqa: F401 — String/Float32 kept for review_embeddings re-enable
from feast.value_type import ValueType

# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

user = Entity(
    name="user_id",
    value_type=ValueType.STRING,
    description="Unique user identifier from Amazon Reviews",
)

item = Entity(
    name="item_id",
    value_type=ValueType.STRING,
    description="Product ASIN identifier (parent_asin)",
)

review = Entity(
    name="review_id",
    value_type=ValueType.STRING,
    description="Unique review identifier for embedding lookup",
)

# ---------------------------------------------------------------------------
# Raw source — reads all parquet files under raw/reviews/ directly.
# `query=` does an inline CAST(timestamp / 1000 AS TIMESTAMP) so Feast's
# SQL time-range filter works without any separate preprocessing step.
# Covers all 29.6 GB / 282 files in the raw dataset.
# ---------------------------------------------------------------------------

raw_reviews_source = SparkSource(
    name="raw_reviews_source",
    # Read raw data directly — no preprocessing script needed.
    # The raw `timestamp` column is Unix-ms BIGINT; Feast's SQL filter expects TIMESTAMP.
    # Inline cast via `query=` avoids DATATYPE_MISMATCH.BINARY_OP_DIFF_TYPES.
    # This covers all 29.6 GB / 282 files in raw/reviews/ without a separate ETL step.
    # parquet.`path` in Spark SQL doesn't recurse into subdirectories.
    # `raw/reviews/` has per-category subdirs (Books/, Electronics/, …).
    # The glob `*/` makes Spark read all immediate subdirectory parquet files.
    query=(
        "SELECT *, CAST(timestamp / 1000 AS TIMESTAMP) AS event_timestamp "
        "FROM parquet.`s3a://smartshop-raw/raw/reviews/*/`"
    ),
    timestamp_field="event_timestamp",
)

# Embeddings source — disabled until embedding job populates review_embeddings/
# review_embeddings_source = SparkSource(
#     name="review_embeddings_source",
#     path="s3a://smartshop-embeddings/review_embeddings/",
#     file_format="parquet",
#     timestamp_field="event_timestamp",
# )

# ---------------------------------------------------------------------------
# @batch_feature_view — user_features
#
# Reads raw reviews → computes per-user aggregates → writes to Redis.
# All imports MUST be inside the function body (dill serialization requirement).
# ---------------------------------------------------------------------------


@batch_feature_view(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="user_avg_rating", dtype=Float64),
        Field(name="user_review_count", dtype=Int64),
        Field(name="user_unique_items", dtype=Int64),
        Field(name="user_avg_review_length", dtype=Float64),
        Field(name="user_category_count", dtype=Int64),
        Field(name="user_tenure_days", dtype=Int64),
    ],
    source=raw_reviews_source,
    mode=TransformationMode.PYTHON,
    online=True,
    offline=False,
)
def user_features(df):
    from pyspark.sql import functions as F

    # Schema normalisation — raw parquet may use 'asin' instead of 'parent_asin'
    if "asin" in df.columns and "parent_asin" not in df.columns:
        df = df.withColumnRenamed("asin", "parent_asin")
    if "helpful_vote" not in df.columns:
        df = df.withColumn("helpful_vote", F.lit(0))
    if "category" not in df.columns:
        df = df.withColumn("category", F.input_file_name())

    return (
        df.groupBy("user_id")
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
            (
                (F.col("user_last_active") - F.col("user_first_active"))
                / 86_400_000
            ).cast("int"),
        )
        .drop("user_categories", "user_last_active", "user_first_active")
        .withColumn("event_timestamp", F.current_timestamp())
    )


# ---------------------------------------------------------------------------
# @batch_feature_view — item_features
#
# Reads raw reviews → computes per-item aggregates → writes to Redis.
# Metadata join (price) is best-effort: items not in metadata get item_price=null.
# ---------------------------------------------------------------------------


@batch_feature_view(
    name="item_features",
    entities=[item],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="item_avg_rating", dtype=Float64),
        Field(name="item_rating_stddev", dtype=Float64),
        Field(name="item_review_count", dtype=Int64),
        Field(name="item_total_helpful_votes", dtype=Int64),
        Field(name="item_avg_review_length", dtype=Float64),
        Field(name="item_price", dtype=Float32),
    ],
    source=raw_reviews_source,
    mode=TransformationMode.PYTHON,
    online=True,
    offline=False,
)
def item_features(df):
    from pyspark.sql import functions as F

    # Schema normalisation
    if "asin" in df.columns and "parent_asin" not in df.columns:
        df = df.withColumnRenamed("asin", "parent_asin")
    if "helpful_vote" not in df.columns:
        df = df.withColumn("helpful_vote", F.lit(0))

    return (
        df.groupBy("parent_asin")
        .agg(
            F.avg("rating").alias("item_avg_rating"),
            F.stddev("rating").alias("item_rating_stddev"),
            F.count("*").alias("item_review_count"),
            F.sum(F.col("helpful_vote").cast("int")).alias("item_total_helpful_votes"),
            F.avg(F.length("text")).alias("item_avg_review_length"),
            # price not in reviews — materialise as null; join with metadata offline
            F.lit(None).cast("float").alias("item_price"),
        )
        .withColumnRenamed("parent_asin", "item_id")
        .withColumn("event_timestamp", F.current_timestamp())
    )


# ---------------------------------------------------------------------------
# FeatureView — review_embeddings (commented out until embedding job runs)
#
# The embedding job (rec-trainer / separate pipeline step) must run first to
# populate s3a://smartshop-embeddings/review_embeddings/ before this view
# can be registered.  Uncomment after the embedding Parquet files exist.
# ---------------------------------------------------------------------------

# from feast import FeatureView
#
# review_embeddings_view = FeatureView(
#     name="review_embeddings",
#     entities=[review],
#     ttl=timedelta(days=90),
#     schema=[
#         Field(name="item_id", dtype=String),
#         Field(name="user_id", dtype=String),
#         Field(name="rating", dtype=Float64),
#         Field(name="review_title", dtype=String),
#         Field(name="embed_text", dtype=String),
#         Field(name="embedding", dtype=Array(Float32)),
#     ],
#     source=review_embeddings_source,
#     online=True,
# )
