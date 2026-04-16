"""Feast feature definitions for SmartShop AI.

Defines three feature views:
  1. user_features - Per-user aggregates from review history
  2. item_features - Per-item aggregates from reviews + metadata
  3. review_embeddings - Vector embeddings for RAG similarity search

Data sources live in MinIO (S3-compatible). Set FEAST_S3_ENDPOINT_URL or
AWS_ENDPOINT_URL_S3=http://minio.smartshop.svc.cluster.local:9000 in the
serving environment (the Feast operator injects this via envFrom).
"""

from datetime import timedelta

import os

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Array, Float32, Float64, Int64, String
from feast.value_type import ValueType

# PyArrow S3FileSystem ignores AWS_ENDPOINT_URL_S3 when endpoint_override is not
# passed explicitly. Always read the endpoint from the environment so that both
# local dev (external URL) and in-cluster (internal svc URL) work transparently.
_S3_ENDPOINT = os.environ.get(
    "AWS_ENDPOINT_URL_S3", "http://minio.smartshop.svc.cluster.local:9000"
)

# -- Entities --

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

# -- Data Sources (MinIO / S3) --
# Paths are populated by the Spark ETL job (infrastructure/openshift/spark-application-rapids.yaml).
# `feast apply` registers the schema even before data exists; `feast materialize` reads the data.

user_features_source = FileSource(
    path="s3://smartshop-features/user_features/",
    timestamp_field="event_timestamp",
    s3_endpoint_override=_S3_ENDPOINT,
)

item_features_source = FileSource(
    path="s3://smartshop-features/item_features/",
    timestamp_field="event_timestamp",
    s3_endpoint_override=_S3_ENDPOINT,
)

review_embeddings_source = FileSource(
    path="s3://smartshop-embeddings/review_embeddings/",
    timestamp_field="event_timestamp",
    s3_endpoint_override=_S3_ENDPOINT,
)

# -- Feature Views --

user_features_view = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=30),
    schema=[
        Field(name="user_avg_rating", dtype=Float64),
        Field(name="user_review_count", dtype=Int64),
        Field(name="user_unique_items", dtype=Int64),
        Field(name="user_avg_review_length", dtype=Float64),
        Field(name="user_category_count", dtype=Int64),
        Field(name="user_tenure_days", dtype=Int64),
    ],
    source=user_features_source,
    online=True,
)

item_features_view = FeatureView(
    name="item_features",
    entities=[item],
    ttl=timedelta(days=30),
    schema=[
        Field(name="item_avg_rating", dtype=Float64),
        Field(name="item_rating_stddev", dtype=Float64),
        Field(name="item_review_count", dtype=Int64),
        Field(name="item_total_helpful_votes", dtype=Int64),
        Field(name="item_avg_review_length", dtype=Float64),
        Field(name="item_price", dtype=Float32),
        Field(name="item_price_bucket", dtype=String),
        Field(name="category", dtype=String),
    ],
    source=item_features_source,
    online=True,
)

review_embeddings_view = FeatureView(
    name="review_embeddings",
    entities=[review],
    ttl=timedelta(days=90),
    schema=[
        Field(name="item_id", dtype=String),
        Field(name="user_id", dtype=String),
        Field(name="rating", dtype=Float64),
        Field(name="review_title", dtype=String),
        Field(name="embed_text", dtype=String),
        Field(name="embedding", dtype=Array(Float32)),
    ],
    source=review_embeddings_source,
    online=True,
)
