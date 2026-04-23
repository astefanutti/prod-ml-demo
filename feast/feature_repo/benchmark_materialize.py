"""benchmark_materialize.py - Timed feast materialize with Spark + Redis metrics.

Runs feast materialize (full historical range, not incremental) and emits [METRIC]
lines in the same format as SparkRunLogger, enabling direct comparison with
SmartShop-FeatureEngineering SparkApp runs in the Spark History Server.

Pipeline being benchmarked:
    s3a://smartshop-raw/processed/reviews/
        -> SparkComputeEngine (local[*])
        -> @batch_feature_view UDF (groupBy/agg)
        -> Redis online store

Reference SparkApp numbers (from Spark History Server 2026-04-21):
    RAPIDS  (SmartShop-FeatureEngineering): 1h 18m  (4680s)
    CPU     (SmartShop-FeatureEngineering): 1h 57m  (7020s)
    Note: SparkApp output = Parquet files. Feast BFV goes raw -> Redis directly.

Usage (inside feast pod):
    export AWS_ACCESS_KEY_ID=minio
    export AWS_SECRET_ACCESS_KEY=minio123
    export MLFLOW_TRACKING_URI=http://mlflow...  (optional)
    cd /feast-data/smartshop/feast/feature_repo
    python3 benchmark_materialize.py [--flush-redis]

    --flush-redis   Clear Redis before run for a clean key-count baseline.
"""

import argparse
import json
import os
import time
import urllib.request
from datetime import datetime, timezone

# Guard: Feast CLI imports all .py files in the feature_repo to discover feature
# definitions. If parse_args() runs at module level it intercepts sys.argv and
# breaks `feast apply` / `feast materialize`. Always guard with __name__ check.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flush-redis",  action="store_true")
    parser.add_argument("--no-mlflow",    action="store_true")
    parser.add_argument("--event-log",    action="store_true",
                        help="Enable spark.eventLog so this run appears in Spark History Server")
    parser.add_argument("--rapids-s",     type=float, default=0,
                        help="RAPIDS SparkApp elapsed seconds for comparison (0 = skip)")
    parser.add_argument("--cpu-s",        type=float, default=0,
                        help="CPU SparkApp elapsed seconds for comparison (0 = skip)")
    args = parser.parse_args()
else:
    # Imported by feast CLI during repo scan — provide safe defaults.
    import types
    args = types.SimpleNamespace(
        flush_redis=False, no_mlflow=True, event_log=False, rapids_s=0, cpu_s=0
    )

FEATURE_REPO  = "/feast-data/smartshop/feast/feature_repo"
REDIS_HOST    = "redis.smartshop.svc.cluster.local"
REDIS_PORT    = 6379
REDIS_PASS    = os.environ.get("REDIS_PASSWORD", "smartshop-redis-2026")
SPARK_UI_PORT = 4040
START_DATE    = datetime(2010, 1, 1, tzinfo=timezone.utc)
END_DATE      = datetime.now(tz=timezone.utc)

# Confirmed numbers from DEMO-ARTIFACTS.md — SparkApp runs on 2026-04-21
# (140,772,341 rows, full dataset: Electronics + Books + Home & Kitchen)
RAPIDS_ELAPSED_S = args.rapids_s or float(os.environ.get("RAPIDS_ELAPSED_S", "4755"))  # 1h 19m
CPU_ELAPSED_S    = args.cpu_s    or float(os.environ.get("CPU_ELAPSED_S",    "7057"))  # 1h 57m


def log(msg):
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}", flush=True)


def metric(key, value, unit=""):
    line = f"[METRIC] {key}: {value}"
    if unit:
        line += f" {unit}"
    print(line, flush=True)


def spark_api(path):
    try:
        url = f"http://localhost:{SPARK_UI_PORT}{path}"
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return None


# ── 1. SparkSession with UI enabled ────────────────────────────────────────────
# Create the session before feast does so getOrCreate() reuses it with UI on.

log("Starting SparkSession with Spark UI enabled on port 4040...")
from pyspark.sql import SparkSession

builder = (
    SparkSession.builder
    .master("local[*]")
    .appName("SmartShop-FeastBVF-Benchmark")
    .config("spark.driver.memory",                          "8g")
    .config("spark.ui.enabled",                            "true")
    .config("spark.ui.port",                               str(SPARK_UI_PORT))
    .config("spark.sql.session.timeZone",                  "UTC")
    .config("spark.sql.shuffle.partitions",                "200")
    .config("spark.sql.execution.arrow.pyspark.enabled",   "true")
    .config("spark.hadoop.fs.s3a.endpoint",
            "http://minio.smartshop.svc.cluster.local:9000")
    .config("spark.hadoop.fs.s3a.path.style.access",       "true")
    .config("spark.hadoop.fs.s3a.impl",
            "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.EnvironmentVariableCredentialsProvider")
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled",  "false")
)

if args.event_log:
    builder = (builder
        .config("spark.eventLog.enabled",  "true")
        .config("spark.eventLog.dir",      "s3a://smartshop-spark-logs/events")
    )
    log("spark.eventLog enabled -> s3a://smartshop-spark-logs/events")
spark = builder.getOrCreate()
spark.sparkContext.setLogLevel("WARN")
app_id = spark.sparkContext.applicationId
log(f"Spark {spark.version} | app: {app_id}")

# ── 2. Redis baseline ───────────────────────────────────────────────────────────

import redis as redislib

rc = redislib.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASS)

if args.flush_redis:
    log("Flushing Redis...")
    rc.flushall()

keys_before    = rc.dbsize()
mem_before_mb  = rc.info("memory")["used_memory"] / 1_048_576
log(f"Redis before: {keys_before:,} keys  {mem_before_mb:.1f} MB")

# ── 3. Count source rows ────────────────────────────────────────────────────────

log("Counting source rows (processed/reviews/)...")
t0 = time.time()
source_rows = spark.read.parquet("s3a://smartshop-raw/processed/reviews/").count()
count_s = round(time.time() - t0, 1)
log(f"Source rows: {source_rows:,}  ({count_s}s)")

# ── 4. Full feast materialize ───────────────────────────────────────────────────

from feast import FeatureStore

fs = FeatureStore(repo_path=FEATURE_REPO)
log(f"feast materialize: {START_DATE.date()} -> {END_DATE.strftime('%Y-%m-%dT%H:%M:%SZ')}")

t0_mat = time.time()
fs.materialize(start_date=START_DATE, end_date=END_DATE)
elapsed_s = round(time.time() - t0_mat, 1)
log(f"materialize done in {elapsed_s}s ({elapsed_s/60:.1f} min)")

# ── 5. Redis post-run ───────────────────────────────────────────────────────────

time.sleep(2)
keys_after     = rc.dbsize()
mem_after_mb   = rc.info("memory")["used_memory"] / 1_048_576
keys_written   = keys_after - keys_before
mem_delta_mb   = round(mem_after_mb - mem_before_mb, 1)
keys_per_sec   = round(keys_written / elapsed_s, 0) if elapsed_s > 0 else 0
log(f"Redis after:  {keys_after:,} keys  {mem_after_mb:.1f} MB")
log(f"Keys written: {keys_written:,}  throughput: {keys_per_sec:.0f} keys/s")

# ── 6. Spark REST API ───────────────────────────────────────────────────────────

time.sleep(1)
log("Collecting Spark REST metrics...")
spark_summary = {}
apps = spark_api("/api/v1/applications")
if apps:
    aid      = apps[0]["id"]
    stages   = spark_api(f"/api/v1/applications/{aid}/stages")   or []
    execs    = spark_api(f"/api/v1/applications/{aid}/executors") or []
    sql_list = spark_api(f"/api/v1/applications/{aid}/sql")       or []

    spark_summary = {
        "stage_count":            len(stages),
        "total_task_ms":          sum(s.get("executorRunTime", 0)   for s in stages),
        "shuffle_read_mb":  round(sum(s.get("shuffleReadBytes", 0)  for s in stages) / 1_048_576, 1),
        "shuffle_write_mb": round(sum(s.get("shuffleWriteBytes", 0) for s in stages) / 1_048_576, 1),
        "input_mb":         round(sum(s.get("inputBytes", 0)        for s in stages) / 1_048_576, 1),
        "max_exec_mem_mb":  round(max((e.get("memoryUsed", 0) for e in execs), default=0) / 1_048_576, 1),
        "sql_queries":            len(sql_list),
    }
    log(f"Spark: {spark_summary['stage_count']} stages | "
        f"task={spark_summary['total_task_ms']/1000:.0f}s | "
        f"shuffle_read={spark_summary['shuffle_read_mb']}MB")
else:
    log("WARN: Spark REST API not reachable")

# ── 7. MLflow ───────────────────────────────────────────────────────────────────

mlflow_run_id = None
if not args.no_mlflow:
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment("smartshop-feature-engineering")
            with mlflow.start_run(run_name="feast-bfv-full-benchmark") as run:
                mlflow_run_id = run.info.run_id
                mlflow.set_tags({
                    "run_type": "feast_bfv",
                    "spark_mode": "local[*]",
                    "feast_version": "0.62.0",
                })
                mlflow.log_params({
                    "pyspark_version":    spark.version,
                    "source_path":        "s3a://smartshop-raw/processed/reviews/",
                    "online_store":       "redis",
                    "partitions":         10,
                    "materialize_mode":   "full",
                })
                mlflow.log_metrics({
                    "total_elapsed_s":        elapsed_s,
                    "source_rows":            source_rows,
                    "redis_keys_written":     keys_written,
                    "redis_keys_total":       keys_after,
                    "redis_write_throughput": keys_per_sec,
                    "redis_mem_delta_mb":     mem_delta_mb,
                    **{f"spark_{k}": v for k, v in spark_summary.items()},
                })
            log(f"MLflow: {mlflow_run_id}")
        except Exception as e:
            log(f"MLflow failed (non-fatal): {e}")
    else:
        log("MLFLOW_TRACKING_URI not set — skipping")

# ── 8. [METRIC] output ──────────────────────────────────────────────────────────

sep = "=" * 62
print(f"\n{sep}", flush=True)
metric("run_type",              "feast_bfv_full")
metric("app_name",              "SmartShop-FeastBVF-Benchmark")
metric("spark_version",         spark.version)
metric("total_elapsed_s",       elapsed_s,                     "s")
metric("total_elapsed_min",     round(elapsed_s / 60, 2),      "min")
metric("source_rows",           f"{source_rows:,}")
metric("redis_keys_before",     f"{keys_before:,}")
metric("redis_keys_after",      f"{keys_after:,}")
metric("redis_keys_written",    f"{keys_written:,}")
metric("redis_write_throughput",f"{keys_per_sec:.0f}",          "keys/sec")
metric("redis_mem_delta_mb",    f"{mem_delta_mb:.1f}",          "MB")
for k, v in spark_summary.items():
    metric(f"spark_{k}", v)
if mlflow_run_id:
    metric("mlflow_run_id",     mlflow_run_id)
print(sep, flush=True)

# ── 9. Comparison table ─────────────────────────────────────────────────────────

print(f"\n{sep}", flush=True)
print("  Feature Engineering Benchmark — SmartShop AI", flush=True)
print(f"  {'Approach':<32} {'Time':>9}  {'Output'}", flush=True)
print(f"  {'-'*32} {'-'*9}  {'-'*34}", flush=True)
print(f"  {'SparkApp + RAPIDS (GPU)':<32} {RAPIDS_ELAPSED_S/60:>7.1f}m  "
      f"Parquet -> feast materialize -> Redis", flush=True)
print(f"  {'SparkApp + CPU baseline':<32} {CPU_ELAPSED_S/60:>7.1f}m  "
      f"Parquet -> feast materialize -> Redis", flush=True)
print(f"  {'Feast BFV (this run)':<32} {elapsed_s/60:>7.1f}m  "
      f"Raw reviews -> Redis (1 step)", flush=True)
print(f"", flush=True)
print(f"  Speedup vs RAPIDS : {RAPIDS_ELAPSED_S / elapsed_s:.2f}x", flush=True)
print(f"  Speedup vs CPU    : {CPU_ELAPSED_S / elapsed_s:.2f}x", flush=True)
print(f"", flush=True)
print(f"  Keys in Redis     : {keys_after:,}", flush=True)
print(f"  Write throughput  : {keys_per_sec:.0f} keys/sec", flush=True)
print(f"  Source rows read  : {source_rows:,}", flush=True)
print(f"", flush=True)
print(f"  SparkApp baseline: 140,772,341 rows, both jobs run 2026-04-21", flush=True)
print(f"  SparkApp pipeline writes intermediate Parquet + separate feast materialize.", flush=True)
print(f"  Feast BFV skips intermediate Parquet — raw data goes directly to Redis.", flush=True)
print(sep, flush=True)
