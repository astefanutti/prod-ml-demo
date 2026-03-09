"""Feast feature definitions for SmartShop AI.

Defines three feature views:
  1. user_features - Per-user aggregates from review history
  2. item_features - Per-item aggregates from reviews + metadata
  3. review_embeddings - Vector embeddings for RAG similarity search
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Array, Float32, Float64, Int64, String

# -- Entities --

user = Entity(
    name="user_id",
    description="Unique user identifier from Amazon Reviews",
)

item = Entity(
    name="item_id",
    description="Product ASIN identifier",
)

review = Entity(
    name="review_id",
    description="Unique review identifier for embedding lookup",
)

# -- Data Sources --

user_features_source = FileSource(
    path="data/user_features.parquet",
    timestamp_field="event_timestamp",
)

item_features_source = FileSource(
    path="data/item_features.parquet",
    timestamp_field="event_timestamp",
)

review_embeddings_source = FileSource(
    path="data/review_embeddings.parquet",
    timestamp_field="event_timestamp",
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
