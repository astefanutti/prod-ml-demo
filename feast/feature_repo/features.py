"""Feast feature definitions for SmartShop AI.

Defines three feature views:
  1. user_features - Per-user aggregates from review history
  2. item_features - Per-item aggregates from reviews + metadata
  3. review_embeddings - Vector embeddings for RAG similarity search

Data sources use SparkSource (s3a://) — requires SparkOfflineStore in feature_store.yaml.
Spark reads from MinIO via the hadoop-aws S3A connector; S3A endpoint + credentials are
injected by the Feast operator from feast-spark-config secret.

In-cluster config: feast-operator.yaml sets offline_store.type=spark with
  spark.hadoop.fs.s3a.endpoint = http://minio.smartshop.svc.cluster.local:9000
Local dev: override SPARK_MASTER + S3A env vars (see feast/feature_repo/feature_store.yaml).

Ref: https://github.com/ntkathole/feast/blob/prod_deploy/docs/how-to-guides/production-deployment-topologies.md
     On-Prem/OpenShift section: Spark + MinIO is the recommended offline store
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.infra.offline_stores.contrib.spark_offline_store.spark_source import SparkSource
from feast.types import Array, Float32, Float64, Int64, String
from feast.value_type import ValueType

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

# -- Data Sources (MinIO / S3A via Spark) --
# Paths are written by the Spark ETL job (spark-application-rapids.yaml).
# s3a:// scheme is required by hadoop-aws; s3:// (boto3) won't work with SparkOfflineStore.
# `feast apply` registers schema even before data exists; `feast materialize` triggers Spark jobs.

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

review_embeddings_source = SparkSource(
    name="review_embeddings_source",
    path="s3a://smartshop-embeddings/review_embeddings/",
    file_format="parquet",
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
        # item_price_bucket and category excluded — string fields, not usable
        # as float32 tensor inputs in TwoTowerModel (item_feat_dim=6 numeric only)
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
