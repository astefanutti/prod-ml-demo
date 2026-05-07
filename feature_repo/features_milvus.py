"""Feast feature definitions — Milvus embedding slice.

Defines review_embeddings @batch_feature_view and its SparkSource.
Separate from features.py (user/item → Redis) to keep Milvus-only
materialization isolated.

Flow:
  raw_reviews_source (SparkSource, S3 parquet)
    → SparkComputeEngine (k8s:// GPU executor pods)
      → _embed_udf: sentence-transformers all-MiniLM-L6-v2, 384-dim
        → Milvus online store (IVF_FLAT COSINE index)
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
    query=(
        "SELECT *, "
        "CAST(timestamp / 1000 AS TIMESTAMP) AS event_timestamp, "
        "SHA2(CONCAT_WS('_', user_id, COALESCE(parent_asin, asin), CAST(timestamp AS STRING)), 256) AS review_id "
        "FROM parquet.`s3a://smartshop-raw/raw/reviews/*/`"
    ),
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
    import numpy as np
    import pandas as pd
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, FloatType

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

    @F.pandas_udf(ArrayType(FloatType()))
    def _embed_udf(texts: pd.Series) -> pd.Series:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = model.encode(
            texts.tolist(), normalize_embeddings=True, batch_size=64, show_progress_bar=False
        )
        return pd.Series([e.tolist() for e in embeddings])

    return staging.withColumn("embedding", _embed_udf(F.col("embed_text")))
