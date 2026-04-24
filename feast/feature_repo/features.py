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

from feast import Entity, FeatureView  # FeatureView used by commented item_text_embedding_view below
from feast.batch_feature_view import batch_feature_view
from feast.field import Field
from feast.infra.offline_stores.contrib.spark_offline_store.spark_source import (
    SparkSource,
)
from feast.transformation.mode import TransformationMode
from feast.types import Array, Float32, Float64, Int64, String
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
# @batch_feature_view — review_embeddings
#
# Reads raw reviews directly → generates 384-dim sentence embeddings inline →
# materializes to Milvus online store via feast materialize.
# The pandas UDF loads all-MiniLM-L6-v2 once per executor partition.
#
# review_id is a stable sha256(user_id_item_id_timestamp) — deterministic
# across reruns, unlike monotonically_increasing_id().
#
# Materialize command (run from Feast pod or local with cluster access):
#   feast -c feature_store_milvus.yaml materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
#
# Online retrieve (RAG server):
#   store.retrieve_online_documents_v2(
#       features=["review_embeddings:embedding", "review_embeddings:embed_text",
#                 "review_embeddings:item_id", "review_embeddings:rating"],
#       query=<384-dim float list>, top_k=5,
#   )
#
# NOT used by rec model training — that uses user_features + item_features only.
# Milvus VARCHAR schema: max_length=512 (Feast hardcoded) — review_title and
# embed_text are sliced to 511 chars before write. Embeddings use full text.
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@batch_feature_view(
    name="review_embeddings",
    entities=[review],
    ttl=timedelta(days=90),
    schema=[
        Field(name="item_id", dtype=String),
        Field(name="user_id", dtype=String),
        Field(name="rating", dtype=Float64),
        Field(name="review_title", dtype=String),
        Field(name="embed_text", dtype=String),
        Field(
            name="embedding",
            dtype=Array(Float32),
            vector_index=True,
            vector_search_metric="COSINE",
        ),
    ],
    source=raw_reviews_source,
    mode=TransformationMode.PYTHON,
    online=True,
    offline=False,
)
def review_embeddings(df):
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    model_id = EMBEDDING_MODEL

    if "asin" in df.columns and "parent_asin" not in df.columns:
        df = df.withColumnRenamed("asin", "parent_asin")

    # Combine title + review body into embed_text; drop very short entries
    df = df.withColumn(
        "embed_text",
        F.concat_ws(
            " ",
            F.coalesce(F.col("title"), F.lit("")),
            F.coalesce(F.col("text"), F.lit("")),
        ),
    ).filter(F.length("embed_text") >= 20)

    # Stable, deterministic review_id across reruns (sha256 hex = 64 chars)
    df = df.withColumn(
        "review_id",
        F.sha2(
            F.concat_ws("_", F.col("user_id"), F.col("parent_asin"), F.col("timestamp").cast("string")),
            256,
        ),
    )

    # Pandas UDF: loads model once per executor partition
    @F.pandas_udf(T.ArrayType(T.FloatType()))
    def _embed(texts):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_id)
        return [
            emb.tolist()
            for emb in model.encode(
                texts.fillna("").tolist(), batch_size=64, show_progress_bar=False
            )
        ]

    df = df.withColumn("embedding", _embed(F.col("embed_text")))

    # Milvus online store VARCHAR fields are hardcoded max_length=512 — slice to 511
    return df.select(
        F.col("review_id"),
        F.col("parent_asin").alias("item_id"),
        F.col("user_id"),
        F.col("rating").cast("double"),
        F.col("title").alias("review_title"),
        F.col("embed_text").substr(1, 511).alias("embed_text"),
        F.col("embedding"),
        F.current_timestamp().alias("event_timestamp"),
    )

# ---------------------------------------------------------------------------
# FeatureView — item_text_embedding  [POST-SUMMIT — not registered yet]
#
# Mean-pooled review embeddings per item_id → one 384-dim vector per item.
# Pre-computed by scripts/fetch_training_features.py --include-embeddings.
# Source parquet: s3a://smartshop-embeddings/item_embeddings_avg/ (Job C output
# aggregated by fetch_training_features.py or a dedicated Spark aggregation job).
#
# To use in training:
#   1. Run: python scripts/fetch_training_features.py --include-embeddings
#   2. Add a Linear(384, hidden_dim) projection to TwoTowerModel.item_tower
#   3. Update train.py _ITEM_FEAT_COLS or pass embedding as separate tensor
#   4. Uncomment this view and feast apply
#
# item_feat_dim will change from 6 → 6 + hidden_dim (projected).
# ---------------------------------------------------------------------------

# item_embeddings_avg_source = SparkSource(
#     name="item_embeddings_avg_source",
#     path="s3a://smartshop-embeddings/item_embeddings_avg/",
#     file_format="parquet",
#     timestamp_field="event_timestamp",
# )
#
# item_text_embedding_view = FeatureView(
#     name="item_text_embedding",
#     entities=[item],
#     ttl=timedelta(days=90),
#     schema=[
#         Field(
#             name="item_embedding_avg",
#             dtype=Array(Float32),  # 384-dim mean-pooled review embedding
#         ),
#     ],
#     source=item_embeddings_avg_source,
#     online=True,
#     offline=True,   # needed for get_historical_features() in training
# )
