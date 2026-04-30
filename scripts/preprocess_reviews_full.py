"""preprocess_reviews_full.py — Add event_timestamp column to all 3 categories.

Reads raw/reviews/{Category}/ (full partitioned parquet) for all 3 categories,
adds event_timestamp as TIMESTAMP (from unix milliseconds in `timestamp` column),
and writes to processed/reviews/ as a single merged Parquet dataset.

Feast requires a TIMESTAMP column for time-based filtering in materialize.
The raw `timestamp` field is BIGINT (unix ms) — this converts it.

Run inside the feast pod after full download completes:

  oc exec -n $NAMESPACE <feast-pod> -c offline -- bash -c '
    export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
    export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
    python3 /feast-data/smartshop/feast/feature_repo/preprocess_reviews_full.py
  '
"""

import os
import time
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

MINIO = os.environ.get("AWS_ENDPOINT_URL_S3", os.environ.get("MINIO_ENDPOINT", "http://minio.smartshop.svc.cluster.local:9000"))
_BUCKET = os.environ.get("S3_RAW_BUCKET", "smartshop-raw")
RAW   = f"s3a://{_BUCKET}/raw/reviews"
OUT   = f"s3a://{_BUCKET}/processed/reviews"
CATEGORIES = ["Electronics", "Books", "Home_and_Kitchen"]

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("SmartShop-PreprocessReviews-Full")
    .config("spark.driver.memory",                          "12g")
    .config("spark.sql.shuffle.partitions",                "400")
    .config("spark.sql.session.timeZone",                  "UTC")
    .config("spark.hadoop.fs.s3a.endpoint",                MINIO)
    .config("spark.hadoop.fs.s3a.path.style.access",       "true")
    .config("spark.hadoop.fs.s3a.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.EnvironmentVariableCredentialsProvider")
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled",  "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

t0 = time.time()
total_rows = 0

for cat in CATEGORIES:
    t_cat = time.time()
    path = f"{RAW}/{cat}/"
    print(f"[{cat}] reading {path} ...", flush=True)
    df = spark.read.parquet(path)
    n_raw = df.count()
    print(f"[{cat}] raw rows: {n_raw:,}", flush=True)

    df = (df
          .withColumn(
              "event_timestamp",
              F.to_timestamp(F.from_unixtime(F.col("timestamp") / 1000))
          )
          .withColumnRenamed("asin", "item_id")
          .withColumnRenamed("user_id", "user_id")
          .filter(F.col("event_timestamp").isNotNull())
          .filter(F.col("item_id").isNotNull())
          .filter(F.col("user_id").isNotNull())
         )

    df.write.mode("append").parquet(OUT)
    elapsed = time.time() - t_cat
    total_rows += n_raw
    print(f"[{cat}] done: {n_raw:,} rows in {elapsed:.0f}s → {OUT}", flush=True)

total_elapsed = time.time() - t0
print(f"\n=== Preprocessing complete ===", flush=True)
print(f"Total rows: {total_rows:,}", flush=True)
print(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)", flush=True)
print(f"Output:     {OUT}", flush=True)
