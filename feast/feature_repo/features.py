"""Feast feature definitions for SmartShop AI.

Pipeline:
  spark/feature_engineering.py pre-computes user + item features to S3 parquet.
  Plain FeatureViews point at those pre-computed outputs:
    s3a://smartshop-features/user_features/   → user_features  FeatureView
    s3a://smartshop-features/item_features/   → item_features  FeatureView

  get_historical_features() reads pre-computed parquets (PIT join works).
  feast materialize pushes the same data to Redis online store.

  Review embeddings remain as @batch_feature_view (online-only, Milvus).
"""

from datetime import timedelta

from feast import Entity, FeatureView
from feast.field import Field
from feast.infra.offline_stores.contrib.spark_offline_store.spark_source import (
    SparkSource,
)
from feast.types import Float32, Float64, Int64
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
# Pre-computed sources — output of spark/feature_engineering.py
# ---------------------------------------------------------------------------

user_features_source = SparkSource(
    name="user_features_source",
    path="s3a://smartshop-features/user_features/",
    file_format="parquet",
    timestamp_field="event_timestamp",
)

item_features_source = SparkSource(
    name="item_features_source",
    path="s3a://smartshop-features/item_features/",
    file_format="parquet",
    timestamp_field="event_timestamp",
)

# ---------------------------------------------------------------------------
# FeatureView — user_features
#
# Reads pre-computed parquet from S3 (written by feature_engineering.py).
# offline=True → get_historical_features() PIT join works.
# online=True  → feast materialize pushes to Redis.
# ---------------------------------------------------------------------------

user_features_view = FeatureView(
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
    source=user_features_source,
    online=True,
    offline=True,
)

# ---------------------------------------------------------------------------
# FeatureView — item_features
#
# Reads pre-computed parquet from S3 (written by feature_engineering.py).
# ---------------------------------------------------------------------------

item_features_view = FeatureView(
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
    source=item_features_source,
    online=True,
    offline=True,
)
