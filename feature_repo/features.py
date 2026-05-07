"""Feast feature definitions — @batch_feature_view edition.

Pipeline:
  Raw Parquet (s3a://smartshop-raw/raw/reviews/)
      |  SparkComputeEngine (RAPIDS GPU)
  @batch_feature_view transformation UDFs
      |
  Redis online store  (online=True)
  S3 offline store    (offline=True, path=s3a://smartshop-features/offline/...)
      |
  get_historical_features reads pre-computed offline parquet for training data

Feature views:
  user_features  — per-user review aggregates + category preference (from reviews)
  item_features  — per-item review aggregates (from reviews)
  item_metadata  — product catalog fields    (from metadata)
"""

from datetime import timedelta

from feast import Entity
from feast.batch_feature_view import batch_feature_view
from feast.field import Field
from feast.infra.offline_stores.contrib.spark_offline_store.spark_source import (
    SparkSource,
)
from feast.transformation.mode import TransformationMode
from feast.types import Float32, Float64, Int64, String
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

# ---------------------------------------------------------------------------
# Sources
#
# Each BFV needs its own source because offline write-back (offline=True)
# writes UDF output to batch_source.path. Different BFVs produce different
# schemas, so each needs a unique offline destination.
#
# query = raw data input for UDF execution during materialization
# path  = offline write destination + get_historical_features read source
# ---------------------------------------------------------------------------

_REVIEWS_QUERY = (
    "SELECT *, CAST(timestamp / 1000 AS TIMESTAMP) AS event_timestamp "
    "FROM parquet.`s3a://smartshop-raw/raw/reviews/*/`"
)

user_reviews_source = SparkSource(
    name="user_reviews_source",
    query=_REVIEWS_QUERY,
    path="s3a://smartshop-features/offline/user_features/",
    file_format="parquet",
    timestamp_field="event_timestamp",
)

item_reviews_source = SparkSource(
    name="item_reviews_source",
    query=_REVIEWS_QUERY,
    path="s3a://smartshop-features/offline/item_features/",
    file_format="parquet",
    timestamp_field="event_timestamp",
)

interactions_source = SparkSource(
    name="interactions_source",
    query=_REVIEWS_QUERY,
    path="s3a://smartshop-features/offline/interactions/",
    file_format="parquet",
    timestamp_field="event_timestamp",
)

raw_metadata_source = SparkSource(
    name="raw_metadata_source",
    query=(
        "SELECT parent_asin, title, "
        "CASE WHEN main_category IN ('None', '') THEN 'Electronics' "
        "     ELSE main_category END AS main_category, "
        "CAST(price AS FLOAT) AS price, "
        "CASE WHEN store IS NOT NULL AND CAST(store AS STRING) != 'None' "
        "     THEN store END AS brand, "
        "TIMESTAMP('2020-01-01') AS event_timestamp "
        "FROM parquet.`s3a://smartshop-raw/raw/metadata/Electronics_meta/` "
        "UNION ALL "
        "SELECT parent_asin, title, "
        "CASE WHEN main_category IN ('None', '') THEN 'Books' "
        "     ELSE main_category END AS main_category, "
        "CAST(price AS FLOAT) AS price, "
        "CASE WHEN store IS NOT NULL AND CAST(store AS STRING) != 'None' "
        "     THEN store END AS brand, "
        "TIMESTAMP('2020-01-01') AS event_timestamp "
        "FROM parquet.`s3a://smartshop-raw/raw/metadata/Books_meta/` "
        "UNION ALL "
        "SELECT parent_asin, title, "
        "CASE WHEN main_category IN ('None', '') THEN 'Home & Kitchen' "
        "     ELSE main_category END AS main_category, "
        "CAST(price AS FLOAT) AS price, "
        "CASE WHEN store IS NOT NULL AND CAST(store AS STRING) != 'None' "
        "     THEN store END AS brand, "
        "TIMESTAMP('2020-01-01') AS event_timestamp "
        "FROM parquet.`s3a://smartshop-raw/raw/metadata/Home_and_Kitchen_meta/`"
    ),
    path="s3a://smartshop-features/offline/item_metadata/",
    file_format="parquet",
    timestamp_field="event_timestamp",
)


# ---------------------------------------------------------------------------
# @batch_feature_view — user_features
#
# Reads raw reviews -> computes per-user aggregates -> writes to Redis.
# All imports MUST be inside the function body (dill serialization).
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
        Field(name="user_primary_category", dtype=String),
    ],
    source=user_reviews_source,
    mode=TransformationMode.PYTHON,
    online=True,
    offline=True,
)
def user_features(df):
    from pyspark.sql import functions as F
    from pyspark.sql import Window

    if "asin" in df.columns and "parent_asin" not in df.columns:
        df = df.withColumnRenamed("asin", "parent_asin")
    if "helpful_vote" not in df.columns:
        df = df.withColumn("helpful_vote", F.lit(0))

    df = df.filter(F.col("rating") > 0)

    df = df.withColumn(
        "category",
        F.regexp_extract(F.input_file_name(), r"/reviews/([^/]+)/", 1),
    )

    user_cat_counts = (
        df.groupBy("user_id", "category")
        .agg(F.count("*").alias("cat_count"))
    )
    w = Window.partitionBy("user_id").orderBy(F.desc("cat_count"))
    user_primary = (
        user_cat_counts
        .withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .select("user_id", F.col("category").alias("user_primary_category"))
    )

    user_aggs = (
        df.groupBy("user_id")
        .agg(
            F.avg("rating").alias("user_avg_rating"),
            F.count("*").alias("user_review_count"),
            F.countDistinct("parent_asin").alias("user_unique_items"),
            F.avg(F.length("text")).alias("user_avg_review_length"),
            F.countDistinct("category").alias("user_category_count"),
            F.max("timestamp").alias("user_last_active"),
            F.min("timestamp").alias("user_first_active"),
        )
        .withColumn(
            "user_tenure_days",
            (
                (F.col("user_last_active") - F.col("user_first_active"))
                / 86_400_000
            ).cast("int"),
        )
        .drop("user_last_active", "user_first_active")
    )

    return (
        user_aggs.join(user_primary, on="user_id", how="left")
        .withColumn("event_timestamp", F.current_timestamp())
    )


# ---------------------------------------------------------------------------
# @batch_feature_view — item_features
#
# Reads raw reviews -> computes per-item review aggregates -> writes to Redis.
# Product catalog fields (title, brand, category, price) are in item_metadata.
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
    ],
    source=item_reviews_source,
    mode=TransformationMode.PYTHON,
    online=True,
    offline=True,
)
def item_features(df):
    from pyspark.sql import functions as F

    if "asin" in df.columns and "parent_asin" not in df.columns:
        df = df.withColumnRenamed("asin", "parent_asin")
    if "helpful_vote" not in df.columns:
        df = df.withColumn("helpful_vote", F.lit(0))

    df = df.filter(F.col("rating") > 0)

    return (
        df.groupBy("parent_asin")
        .agg(
            F.avg("rating").alias("item_avg_rating"),
            F.stddev("rating").alias("item_rating_stddev"),
            F.count("*").alias("item_review_count"),
            F.sum(F.col("helpful_vote").cast("int")).alias("item_total_helpful_votes"),
            F.avg(F.length("text")).alias("item_avg_review_length"),
        )
        .withColumnRenamed("parent_asin", "item_id")
        .withColumn("event_timestamp", F.current_timestamp())
    )


# ---------------------------------------------------------------------------
# @batch_feature_view — item_metadata
#
# Reads product catalog (metadata parquet) -> writes to Redis.
# Separate source from reviews — no nested spark.read inside UDFs.
# ---------------------------------------------------------------------------


@batch_feature_view(
    name="item_metadata",
    entities=[item],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="item_title", dtype=String),
        Field(name="item_brand", dtype=String),
        Field(name="item_category", dtype=String),
        Field(name="item_price", dtype=Float32),
    ],
    source=raw_metadata_source,
    mode=TransformationMode.PYTHON,
    online=True,
    offline=True,
)
def item_metadata(df):
    from pyspark.sql import functions as F

    return (
        df.select(
            F.col("parent_asin").alias("item_id"),
            F.col("title").alias("item_title"),
            F.col("main_category").alias("item_category"),
            F.col("price").cast("float").alias("item_price"),
            F.col("brand").alias("item_brand"),
        )
        .dropDuplicates(["item_id"])
        .withColumn("event_timestamp", F.current_timestamp())
    )


# ---------------------------------------------------------------------------
# @batch_feature_view — interactions
#
# Reads raw reviews -> produces user-item interaction pairs for training.
# label = 1.0 when rating >= 4, else 0.0 (implicit positive signal).
# online=False: interactions are too large for Redis.
# Training reads from S3 offline parquet, so online serving is not needed.
# ---------------------------------------------------------------------------


@batch_feature_view(
    name="interactions",
    entities=[user, item],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="label", dtype=Float64),
    ],
    source=interactions_source,
    mode=TransformationMode.PYTHON,
    online=False,
    offline=True,
)
def interactions(df):
    from pyspark.sql import functions as F

    if "asin" in df.columns and "parent_asin" not in df.columns:
        df = df.withColumnRenamed("asin", "parent_asin")

    return (
        df.select(
            F.col("user_id"),
            F.col("parent_asin").alias("item_id"),
            F.when(F.col("rating") >= 4, 1.0).otherwise(0.0).alias("label"),
            F.col("event_timestamp"),
        )
    )
