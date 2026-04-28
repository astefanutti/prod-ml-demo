"""Feast feature definitions — Milvus-only slice.

Contains ONLY the review_embeddings @batch_feature_view and its dependencies.
Used with feature_store_milvus.yaml to avoid feast apply trying to create
Milvus collections for user_features / item_features (which have no vector field).

Flow:
  raw_reviews_source (SparkSource, S3)
      ↓  SparkComputeEngine  (k8s:// executor pods — feast-spark-executor-embeddings)
  review_embeddings BFV  →  spark_embed() (sentence-transformers, GPU/RAPIDS)
      ↓
  Milvus online store  (384-dim COSINE index)
"""

from datetime import timedelta

from feast import Entity
from feast.batch_feature_view import batch_feature_view
from feast.field import Field
from feast.infra.offline_stores.contrib.spark_offline_store.spark_source import SparkSource
from feast.transformation.mode import TransformationMode
from feast.types import Array, Float32, Float64, String
from feast.value_type import ValueType

# ── Entity ───────────────────────────────────────────────────────────────────

review = Entity(
    name="review_id",
    value_type=ValueType.STRING,
    description="Unique review identifier for embedding lookup",
)

# ── Raw source ────────────────────────────────────────────────────────────────

raw_reviews_source = SparkSource(
    name="raw_reviews_source",
    query="SELECT *, CAST(timestamp / 1000 AS TIMESTAMP) AS event_timestamp FROM parquet.`s3a://smartshop-raw/raw/reviews/*/`",
    timestamp_field="event_timestamp",
)

# ── Embedding model constants ─────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


# ── @batch_feature_view: review_embeddings ───────────────────────────────────
#
# Run:
#   feast -c /tmp/feast-milvus-repo/feature_store.yaml \
#     materialize 2023-06-01T00:00:00 2023-06-08T00:00:00 \
#     --feature-views review_embeddings

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

    from feast.infra.compute_engines.spark.utils import spark_embed

    if "asin" in df.columns and "parent_asin" not in df.columns:
        df = df.withColumnRenamed("asin", "parent_asin")

    df = df.withColumn(
        "embed_text",
        F.concat_ws(
            " ",
            F.coalesce(F.col("title"), F.lit("")),
            F.coalesce(F.col("text"), F.lit("")),
        ),
    ).filter(F.length("embed_text") >= 20)

    df = df.withColumn(
        "review_id",
        F.sha2(
            F.concat_ws(
                "_",
                F.col("user_id"),
                F.col("parent_asin"),
                F.col("timestamp").cast("string"),
            ),
            256,
        ),
    )

    staging = df.select(
        F.col("review_id"),
        F.col("parent_asin").alias("item_id"),
        F.col("user_id"),
        F.col("rating").cast("double"),
        F.col("title").alias("review_title"),
        F.col("embed_text").substr(1, 511).alias("embed_text"),
        F.current_timestamp().alias("event_timestamp"),
    )

    return spark_embed(
        staging,
        text_col="embed_text",
        model=EMBEDDING_MODEL,
        output_col="embedding",
        batch_size=64,
    )
