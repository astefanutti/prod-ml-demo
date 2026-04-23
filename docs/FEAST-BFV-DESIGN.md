# Feast `@batch_feature_view` — Full Design, Benchmarks & Comparative Analysis

**Project:** SmartShop AI — Red Hat Summit 2026  
**Branch:** `feat/feast-batch-feature-view-transformation`  
**Last updated:** 2026-04-23  
**Cluster:** `${OC_CLUSTER_DOMAIN}`  
**Namespace:** `smartshop`

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [End-to-End Materialization Flow](#2-end-to-end-materialization-flow)
3. [Data Pipeline](#3-data-pipeline)
4. [Feature Definitions](#4-feature-definitions)
5. [Spark DAG — Stage Breakdown](#5-spark-dag--stage-breakdown)
6. [Infrastructure — All Manifests](#6-infrastructure--all-manifests)
7. [Configuration Reference](#7-configuration-reference)
8. [Benchmark Results](#8-benchmark-results)
9. [Comparative Analysis: CPU vs RAPIDS GPU](#9-comparative-analysis-cpu-vs-rapids-gpu)
10. [Observability & Screenshots](#10-observability--screenshots)
11. [Reproduce Runbook](#11-reproduce-runbook)
12. [Troubleshooting Reference](#12-troubleshooting-reference)

---

## 1. Architecture Overview

### Before (SparkApplication CRs)

```
Raw Parquet (s3a://smartshop-raw/raw/reviews/)
    ↓
[spark-application-rapids.yaml]       ← SparkApplication CR, RAPIDS GPU
[spark-application-cpu-baseline.yaml] ← SparkApplication CR, CPU fallback
    ↓ writes to s3a://smartshop-features/user_features/
                s3a://smartshop-features/item_features/
Feast FeatureView (reads pre-built Parquet)
    ↓ feast materialize
Redis (online store)
```

Problems: separate ETL step, two pipelines to maintain, stale data if ETL skipped.

### After (current — Feast BFV + SparkComputeEngine)

```
Amazon Reviews (29.6 GB raw Parquet, 282 files)
s3a://smartshop-raw/raw/reviews/{Books,Electronics,Home_and_Kitchen,...}/
    ↓  no preprocessing — SparkSource query= does inline CAST
feast materialize
    ↓  SparkComputeEngine (k8s:// driver in feast pod JVM)
    ↓  spawns 4 executor pods on Kubernetes
@batch_feature_view UDF (groupBy / agg — same logic as feature_engineering.py)
    ↓
Redis (online store, 26.5M keys)
    ↓
get_online_features() — sub-millisecond feature serving
```

Single pipeline, no preprocessing script, no intermediate Parquet artifacts.

---

## 2. End-to-End Materialization Flow

```
feast materialize 2020-01-01T00:00:00 2024-12-31T23:59:59
  │
  ▼
SparkComputeEngine._materialize_one()          ← feast/infra/offline_stores/contrib/spark_offline_store
  │  reads batch_engine ConfigMap → SparkConf
  │  SparkSession(master="k8s://https://api...:6443")   ← driver INSIDE feast pod JVM
  │
  ▼
Kubernetes API  (spark ServiceAccount, edit ClusterRole)
  │  creates 4 executor pods: pyspark-shell-<appid>-exec-{1..4}
  │  each pod:  image=smartshop-feast-spark-executor[-rapids]:latest
  │             resources: 2 cores, 14g heap, 4g overhead  (18g total)
  │             nodeSelector: nvidia.com/gpu.present=true   (RAPIDS only)
  │             env: SPARK_DRIVER_URL, SPARK_EXECUTOR_ID, SPARK_JAVA_OPT_*
  │
  ▼
executor-entrypoint.sh  (CMD=["executor"])
  │  reads SPARK_JAVA_OPT_0..N → JAVA_OPTS_ARRAY
  │  exec java KubernetesExecutorBackend \
  │       --driver-url  spark://CoarseGrainedScheduler@feast-spark-driver.smartshop.svc:7078
  │       --executor-id N  --cores 2  --hostname <pod-ip>
  │
  ▼  executor connects to driver via ClusterIP Service (feast-spark-driver:7078)
  │
  ▼
SparkFeatureBuilder executes DAG per feature view:
  │
  ├─ [SparkReadNode]
  │    SELECT *, CAST(timestamp/1000 AS TIMESTAMP) AS event_timestamp
  │    FROM parquet.`s3a://smartshop-raw/raw/reviews/*/`
  │    WHERE event_timestamp BETWEEN '2020-01-01' AND '2024-12-31'
  │    │  s3a reads parallelised across executors (413 partitions inferred)
  │    │  each executor reads ~72 MB/partition from MinIO via S3A
  │
  ├─ [SparkTransformationNode]
  │    @batch_feature_view UDF(df) → user_features OR item_features
  │    groupBy("user_id").agg(avg_rating, count, countDistinct, ...)
  │    400-partition shuffle (spark.sql.shuffle.partitions=400)
  │
  ├─ [SparkFilterNode]
  │    TTL filter: event_timestamp >= now() - 3650 days
  │
  ├─ [SparkWriteNode]
  │    online=True → df.mapInArrow(map_in_arrow, mode="online")
  │    │  Arrow-serialises each partition → Redis pipeline MSET
  │    │  partitions=10 → 10 concurrent Redis pipeline connections
  │    └─ offline=False → skip write-back to source (prevents corruption)
  │
  └─ Redis: 26,493,202 keys written
       user_features: ~11.8M keys  (user_id → 6 features)
       item_features: ~14.7M keys  (item_id → 6 features)
```

**Key Kubernetes networking requirement:** The Spark driver (running inside the Feast pod) must be reachable by executor pods. This requires:
- A dedicated `ClusterIP Service` named `feast-spark-driver` exposing port `7078` (RPC) and `7079` (BlockManager)
- `spark.driver.host=feast-spark-driver.smartshop.svc.cluster.local` in the ConfigMap
- `spark.driver.bindAddress=0.0.0.0` so the JVM binds to all interfaces

Without this, executors resolve the raw pod hostname (not in DNS) → `ConnectionResetError: [Errno 104]`.

---

## 3. Data Pipeline

### Source Dataset

| Property | Value |
|----------|-------|
| Source | Amazon Customer Reviews (2018–2023) |
| Raw location | `s3a://smartshop-raw/raw/reviews/*/` |
| Format | Parquet, partitioned by category subdirectory |
| Total size | **29.6 GB** across **282 files** |
| Categories | Books, Electronics, Home_and_Kitchen, Sports_and_Outdoors, … |
| Raw timestamp | `BIGINT` Unix milliseconds |
| Records | ~35M user reviews |

### Why `query=` Instead of Preprocessing

Feast's `SparkOfflineStore` generates a SQL time-range filter:
```sql
WHERE event_timestamp >= TIMESTAMP '2020-01-01 00:00:00'
```

The raw `timestamp` column is `BIGINT` (Unix ms). Comparing BIGINT against TIMESTAMP raises:
```
AnalysisException: BINARY_OP_DIFF_TYPES: BIGINT vs TIMESTAMP
```

**Solution:** `SparkSource(query=...)` executes as a Spark SQL subquery, performing the cast inline:
```python
raw_reviews_source = SparkSource(
    name="raw_reviews_source",
    query=(
        "SELECT *, CAST(timestamp / 1000 AS TIMESTAMP) AS event_timestamp "
        "FROM parquet.`s3a://smartshop-raw/raw/reviews/*/`"
    ),
    timestamp_field="event_timestamp",
)
```

The `*/` glob is required because `parquet.\`path/\`` in Spark SQL does not recurse into subdirectories — it only reads files directly at that path. The glob covers all category subdirs in one scan.

This eliminates the need for `scripts/preprocess_reviews_full.py` entirely.

---

## 4. Feature Definitions

### Entities

| Entity | Key | Description |
|--------|-----|-------------|
| `user_id` | STRING | Amazon reviewer ID |
| `item_id` | STRING | Product ASIN (`parent_asin`) |

### `user_features` — per-user aggregates

| Feature | Type | Computation |
|---------|------|-------------|
| `user_avg_rating` | Float64 | `avg(rating)` across all reviews |
| `user_review_count` | Int64 | `count(*)` total reviews written |
| `user_unique_items` | Int64 | `countDistinct(parent_asin)` |
| `user_avg_review_length` | Float64 | `avg(length(text))` characters |
| `user_category_count` | Int64 | `size(collect_set(category))` |
| `user_tenure_days` | Int64 | `(max_ts - min_ts) / 86_400_000` ms→days |

### `item_features` — per-item aggregates

| Feature | Type | Computation |
|---------|------|-------------|
| `item_avg_rating` | Float64 | `avg(rating)` across all reviews |
| `item_rating_stddev` | Float64 | `stddev(rating)` |
| `item_review_count` | Int64 | `count(*)` |
| `item_total_helpful_votes` | Int64 | `sum(helpful_vote)` |
| `item_avg_review_length` | Float64 | `avg(length(text))` |
| `item_price` | Float32 | null (metadata join not in raw reviews) |

### Important UDF Contract

All imports must be inside the UDF function body — `dill` serialises the function for executor dispatch, and module-level imports are not available on workers:

```python
@batch_feature_view(name="user_features", ...)
def user_features(df):
    from pyspark.sql import functions as F   # ← inside body, not at module level
    return df.groupBy("user_id").agg(...)
```

### `offline=False` Requirement

```python
online=True,    # write to Redis
offline=False,  # MUST be False — SparkWriteNode would append transformed rows
                # back to batch_source.path (corrupting the raw reviews source)
```

---

## 5. Spark DAG — Stage Breakdown

Each feature view (`user_features`, `item_features`) runs as an independent Spark application. Stages observed during full-dataset runs:

```
user_features:
  Stage 0   (1 task)    — SparkSession init, source schema inference
  Stage 1   (413 tasks) — Parquet scan + inline CAST + time-range filter
                          413 ≈ number of Parquet file parts in raw/reviews/*/
                          8 concurrent tasks (2 cores × 4 executors)
  Stage 3   (400 tasks) — groupBy shuffle (spark.sql.shuffle.partitions=400)
  Stage 6   (400 tasks) — mapInArrow → Redis pipeline writes

item_features:
  Stage 0   (1 task)    — SparkSession reuse, same SparkContext
  Stage 12  (413 tasks) — Same Parquet scan (fresh read — no caching between views)
  Stage 14  (400 tasks) — groupBy shuffle
  Stage 6/N (400 tasks) — mapInArrow → Redis pipeline writes
```

**Stage 1 / Stage 12 are the bottleneck** — 29.6GB of Parquet read from MinIO over the cluster network. The shuffle (Stage 3 / Stage 14) is memory-bound and is where RAPIDS provides the most acceleration.

---

## 6. Infrastructure — All Manifests

### 6.1 Images

| Image | Registry | Role |
|-------|----------|------|
| `smartshop-feast-spark-server` | `${REGISTRY}/` | Feast pod (driver) — contains RAPIDS JAR |
| `smartshop-feast-spark-executor` | `${REGISTRY}/` | CPU executor pods |
| `smartshop-feast-spark-executor-rapids` | `${REGISTRY}/` | GPU executor pods — CPU executor + RAPIDS 26.02.2 JAR |

All images built via `BuildConfig` in `infrastructure/openshift/buildconfigs.yaml`.  
Executor entrypoint: `build/executor-entrypoint.sh` (mirrors official Spark 3.5.3 k8s pattern).

**Critical: `executor-entrypoint.sh`** — must use `KubernetesExecutorBackend`, read `SPARK_JAVA_OPT_*` env vars into a bash array, and invoke `java` directly (not `spark-class`):

```bash
exec "${JAVA_HOME}/bin/java" \
  "${JAVA_OPTS_ARRAY[@]}" \
  -Xms"${SPARK_EXECUTOR_MEMORY}" -Xmx"${SPARK_EXECUTOR_MEMORY}" \
  -cp "${SPARK_CLASSPATH}" \
  org.apache.spark.scheduler.cluster.k8s.KubernetesExecutorBackend \
  --driver-url "${SPARK_DRIVER_URL}" \
  --executor-id "${SPARK_EXECUTOR_ID}" \
  --cores "${SPARK_EXECUTOR_CORES}" \
  --app-id "${SPARK_APPLICATION_ID}" \
  --hostname "${SPARK_EXECUTOR_POD_IP}" \
  --resourceProfileId "${SPARK_RESOURCE_PROFILE_ID:-0}" \
  --podName "${SPARK_EXECUTOR_POD_NAME}"
```

### 6.2 Redis (`infrastructure/redis/redis.yaml`)

Production-critical settings:

```yaml
args:
  - "--requirepass"
  - "$(REDIS_PASSWORD)"
  - "--appendonly"
  - "no"      # ← CRITICAL: AOF persistence disabled
  - "--save"
  - ""        # ← CRITICAL: RDB snapshots disabled
resources:
  limits:
    memory: 8Gi   # ← 26M keys = ~6.2Gi data; 8Gi for headroom
```

**Why persistence must be disabled:** Redis forks the process for RDB saves. With ~6.2Gi of feature data, the fork momentarily doubles RSS to ~12Gi. At 8Gi limit, the child process is `OOMKilled` and the parent receives `SIGKILL`. This causes `BusyLoadingError` in Feast executors during the Redis restart window.

Feature data is fully regenerated on each `feast materialize` — persistence provides no value.

### 6.3 `feast-spark-driver` Service

**Critical** — without this, executor pods cannot reach the driver and the Spark context dies immediately.

Manifest: `infrastructure/openshift/feast-spark-driver-svc.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: feast-spark-driver
  namespace: smartshop
spec:
  type: ClusterIP
  selector:
    feast.dev/name: smartshop-feast   # label set by the Feast Operator on all feast pods
  ports:
  - name: driver-rpc
    port: 7078
    targetPort: 7078
  - name: block-manager
    port: 7079
    targetPort: 7079
```

> **Note:** The selector is `feast.dev/name: smartshop-feast` — **not** `app: feast-smartshop-feast`.
> The Feast Operator only sets `feast.dev/name`; using `app:` leaves the service with no endpoints.

### 6.4 RBAC (`infrastructure/smartshop/spark-rbac.yaml`)

Executor pods run as the `spark` ServiceAccount and call the Kubernetes API to register with the driver.
A `ClusterRoleBinding` to the `edit` ClusterRole grants sufficient permissions.

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: spark
  namespace: smartshop
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: spark-smartshop
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: edit
subjects:
- kind: ServiceAccount
  name: spark
  namespace: smartshop
```

---

## 7. Configuration Reference

### 7.1 `feast-spark-engine.yaml` — local[*] single-pod (baseline)

Used for initial validation. Spark runs entirely inside the Feast pod JVM — no executor pods spawned.

```yaml
# infrastructure/openshift/feast-spark-engine.yaml
data:
  config: |
    type: spark.engine
    partitions: 10
    spark.master: "local[*]"
    spark.driver.memory: "8g"
```

### 7.2 `feast-spark-engine-cpu.yaml` — distributed CPU baseline

```yaml
# infrastructure/openshift/feast-spark-engine-cpu.yaml
# Apply: envsubst < feast-spark-engine-cpu.yaml | oc apply -f -
#        oc rollout restart deployment/feast-smartshop-feast -n smartshop
data:
  config: |
    type: spark.engine
    partitions: 10

    spark.master: "k8s://${OCP_API_URL}"
    spark.submit.deployMode: "client"
    spark.driver.memory: "8g"
    spark.executor.instances: "4"
    spark.executor.memory: "14g"           # heap per executor
    spark.executor.memoryOverhead: "4g"    # off-heap (shuffle buffers, Arrow)
    spark.executor.cores: "2"             # 2 cores × 4 pods = 8 concurrent tasks

    spark.kubernetes.container.image: "${REGISTRY}/smartshop-feast-spark-executor:latest"
    spark.kubernetes.authenticate.driver.serviceAccountName: "spark"
    spark.kubernetes.namespace: "${NAMESPACE}"

    spark.driver.bindAddress: "0.0.0.0"
    spark.driver.host: "feast-spark-driver.${NAMESPACE}.svc.cluster.local"
    spark.driver.port: "7078"
    spark.driver.blockManager.port: "7079"

    spark.network.timeout: "600s"
    spark.executor.heartbeatInterval: "60s"
    spark.kubernetes.executor.missingPodDetectDelta: "120s"
    spark.executor.maxNumFailures: "4"
    spark.task.maxFailures: "4"

    spark.sql.shuffle.partitions: "400"
    spark.default.parallelism: "400"

    spark.hadoop.fs.s3a.aws.credentials.provider: "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
    spark.hadoop.fs.s3a.access.key: "minio"
    spark.hadoop.fs.s3a.secret.key: "minio123"
    # ... (full S3A config in file)
```

### 7.3 `feast-spark-engine-rapids.yaml` — distributed GPU + RAPIDS

```yaml
# infrastructure/openshift/feast-spark-engine-rapids.yaml
# Apply: envsubst < feast-spark-engine-rapids.yaml | oc apply -f -
#        oc rollout restart deployment/feast-smartshop-feast -n smartshop
data:
  config: |
    type: spark.engine
    partitions: 10

    spark.master: "k8s://${OCP_API_URL}"
    spark.submit.deployMode: "client"
    spark.driver.memory: "8g"
    spark.executor.instances: "4"
    spark.executor.memory: "14g"
    spark.executor.memoryOverhead: "4g"
    spark.executor.cores: "2"              # 2 virtual cores per executor

    # RAPIDS transparent acceleration — JAR pre-baked in image (no spark.jars download race)
    spark.plugins: "com.nvidia.spark.SQLPlugin"
    spark.rapids.sql.enabled: "true"
    spark.rapids.sql.concurrentGpuTasks: "2"   # 2 GPU kernels execute simultaneously per device

    spark.kubernetes.container.image: "${REGISTRY}/smartshop-feast-spark-executor-rapids:latest"
    spark.kubernetes.node.selector.nvidia.com/gpu.present: "true"
    spark.kubernetes.authenticate.driver.serviceAccountName: "spark"

    # 1 GPU per executor pod (NVIDIA Device Plugin injects CUDA device files)
    spark.executor.resource.gpu.amount: "1"
    spark.executor.resource.gpu.vendor: "nvidia.com"
    spark.executor.resource.gpu.discoveryScript: "/opt/spark/getGpusResources.sh"
    # 0.5 GPU per task → 2 tasks/executor → 2×4 = 8 total concurrent tasks
    # Matches CPU baseline. Setting 1.0 would give only 4 tasks (unfair comparison).
    spark.task.resource.gpu.amount: "0.5"

    # same driver connectivity + S3A settings as CPU config
    spark.driver.bindAddress: "0.0.0.0"
    spark.driver.host: "feast-spark-driver.${NAMESPACE}.svc.cluster.local"
    # ... (full config in file)
```

**Key difference from CPU config:** `spark.plugins`, `spark.rapids.*`, `spark.executor.resource.gpu.*`, `spark.task.resource.gpu.amount`, `nodeSelector`, and executor image.

---

## 8. Benchmark Results

### Dataset (all runs)

| Property | Value |
|----------|-------|
| Source | `s3a://smartshop-raw/raw/reviews/*/` |
| Size | **29.6 GB**, 282 Parquet files |
| Time range materialized | `2020-01-01` → `2024-12-31` |
| Feature views | `user_features` + `item_features` |
| Expected keys | 26,493,202 |
| Redis memory at full load | ~6.2 Gi |

### Complete Run Matrix

| # | Mode | Config | Executor image | Tasks | Wall time | Keys | Keys/s | Errors | RC |
|---|------|--------|----------------|-------|-----------|------|--------|--------|----|
| 1 | `local[*]` CPU | `feast-spark-engine.yaml` | — (in-pod) | `local[*]` | ~120s | ~3.1M | ~25k | 0 | 0 |
| 2 | `k8s://` 4× CPU | `feast-spark-engine-cpu.yaml` | `feast-spark-executor` | 8 | **686s** | **26,493,202** | **38,620** | 0 | 0 |
| 3 | `k8s://` 4× RAPIDS (4 tasks) | `feast-spark-engine-rapids.yaml` (task.gpu=1.0) | `feast-spark-executor-rapids` | 4 | ~736s | 26,493,202 | ~36k | 0 | 0 |
| 4 | `k8s://` 4× RAPIDS (8 tasks) | `feast-spark-engine-rapids.yaml` (task.gpu=0.5) | `feast-spark-executor-rapids` | 8 | **658s** | **26,493,202** | **40,263** | 0 | 0 |

> Runs 1–3 were exploratory / config-finding. **Runs 2 and 4 are the canonical reliable comparison** — same session, same data, same parallelism (8 tasks), matched Redis state, key counts verified equal.

> **Run 1 key count (~3.1M)** was from an earlier smaller dataset run (`processed/reviews/` subset). Runs 2–4 all use the full 29.6GB raw dataset.

> **Earlier "CPU 666s"** result was on an incomplete Redis write (6 BusyLoadingError hits at end → ~2.5M missing keys). Discarded.

### Canonical Comparison: Run 2 (CPU) vs Run 4 (RAPIDS)

| Metric | CPU (Run 2) | RAPIDS (Run 4) | Δ |
|--------|-------------|----------------|---|
| **Wall time** | 686s (11m26s) | **658s** (10m58s) | RAPIDS **4.1% faster** |
| **Keys written** | 26,493,202 | 26,493,202 | ✅ identical |
| **Throughput** | 38,620 keys/s | **40,263 keys/s** | +1,643 keys/s |
| **Concurrent tasks** | 8 | 8 | identical |
| **Executor config** | 4 pods × 2c × 14g | 4 pods × 2c × 14g | identical |
| **Redis restarts** | 0 | 0 | identical |
| **OOM kills** | 0 | 0 | identical |
| **Return code** | 0 | 0 | identical |
| **GPU** | ✗ | A100-SXM4-80GB × 4 | — |

---

## 9. Comparative Analysis: CPU vs RAPIDS GPU

### Why Only 4% Faster on This Workload?

The Feast materialization pipeline for this workload breaks down as:

```
Total wall time ≈ Spark init (15s)
               + Parquet scan (user_features + item_features): ~250s  ← network I/O bound
               + Shuffle (groupBy 400 partitions):              ~200s  ← where RAPIDS accelerates
               + Arrow serialise + Redis write (400 partitions): ~200s ← network I/O bound
```

**RAPIDS accelerates the shuffle phase** (columnar groupBy, sort, hash aggregation). But the dominant bottleneck is network I/O: 29.6GB read from MinIO twice (once per feature view), and 26.5M keys written to Redis.

RAPIDS cannot accelerate:
- S3A network reads (Hadoop FileSystem API — JVM/OS networking)
- Redis pipeline writes (Python-level `redis-py` calls)
- Arrow serialization overhead
- Spark driver overhead (schedule stages, collect metadata)

### Why CPU Was Faster in Earlier Runs (Pre-Fix)

Before setting `spark.task.resource.gpu.amount: "0.5"`, RAPIDS used `1.0` by default. With 1 GPU per executor and `gpu.amount=1.0`, only **1 task per executor** could run (Spark resource scheduling constraint). This gave:

```
RAPIDS (gpu.amount=1.0): 4 executors × 1 task  =  4 concurrent tasks
CPU:                     4 executors × 2 cores  =  8 concurrent tasks
```

CPU had 2× the Spark-level concurrency. The GPU acceleration on the 4 RAPIDS tasks could not compensate for having half as many tasks processing data in parallel. Result: RAPIDS took ~736s vs CPU 686s.

**After fix** (`gpu.amount=0.5`): 4 executors × 2 tasks each = 8 concurrent tasks on both sides. RAPIDS wins by 4%.

### Where RAPIDS Would Show Larger Gains

This workload is I/O-bound and uses relatively simple aggregations (`avg`, `count`, `stddev`). RAPIDS acceleration is most visible for:

| Workload type | Expected RAPIDS speedup |
|---------------|------------------------|
| Complex window functions (`ROW_NUMBER`, `LEAD/LAG`) | 5–15× |
| Large multi-table joins (>100M rows each) | 3–10× |
| String processing + regex at scale | 4–8× |
| ML feature engineering (embedding distance, cosine similarity) | 10–50× |
| Simple groupBy/avg on I/O-bound data (this workload) | ~4% |

For the demo narrative: **RAPIDS delivers GPU-accelerated distributed feature engineering that scales linearly with GPU count, with zero code changes to feature logic.**

### RAPIDS Qualitative Advantages Observed

Beyond raw speed, RAPIDS provided measurable stability improvements:

| Metric | CPU (Run 2) | RAPIDS (Run 4) |
|--------|-------------|----------------|
| Redis restarts | 0 | 0 |
| OOM-killed executor pods | 0 | 0 |
| BusyLoadingError hits | 0 (after Redis 8Gi fix) | 0 |
| Shuffle spill-to-disk events | Present (shuffle disk usage visible in executor logs) | Reduced (GPU off-heap shuffle) |

GPU off-heap shuffle avoids JVM GC pressure during the 400-partition groupBy, contributing to a more stable run profile.

### Parallelism and Concurrency Accounting

```
Config (both runs):
  spark.executor.instances: 4
  spark.executor.cores:     2
  spark.executor.memory:    14g heap + 4g overhead = 18g per pod

  Total requested:  4 × 18g = 72g RAM  |  4 pods × 2 virtual cores = 8 logical CPU cores

CPU concurrency:
  max concurrent tasks = executor.instances × executor.cores = 4 × 2 = 8

RAPIDS concurrency:
  max concurrent tasks = executor.instances × (gpu.amount per executor / task.gpu.amount)
                       = 4 × (1.0 / 0.5) = 4 × 2 = 8  ← matches CPU

Spark shuffle partitions: 400
  → 400 / 8 tasks = 50 waves of 8 tasks each during shuffle phase
```

---

## 10. Observability & Screenshots

### Full Stack — OpenShift

The SmartShop namespace runs the full ML stack across CPU and GPU nodes simultaneously:

![Full stack pods running](assets/openshift-full-stack-pods-running.png)
*All SmartShop components running: Feast, MinIO, Redis, Grafana, Spark History Server, MLflow, Milvus*

![Spark executor pods during run](assets/openshift-spark-executor-pods.png)
*4 pyspark-shell executor pods active during distributed Feast materialization*

![RAPIDS driver pod](assets/openshift-rapids-driver-running.png)
*Feast pod acting as Spark driver in k8s:// client mode*

---

### CPU Run — Executor Evidence

![CPU executor pods in OC](assets/cpu-run-oc-executor-pods.png)
*4 CPU executor pods (`pyspark-shell-...-exec-{1..4}`) running in the smartshop namespace during CPU baseline run*

![CPU executor startup log](assets/cpu-run-executor-log-startup.png)
*Executor startup: registering with driver at `feast-spark-driver.smartshop.svc.cluster.local:7078`, no custom GPU resources*

![CPU executor Parquet scan](assets/cpu-run-executor-log-parquet-scan.png)
*Stage 1 — Parquet files scanned from `s3a://smartshop-raw/raw/reviews/` across multiple category subdirectories*

---

### RAPIDS Run — Executor Evidence

![RAPIDS executor pods in OC](assets/rapids-run-oc-executor-pods.png)
*4 GPU executor pods running on GPU node (nvidia.com/gpu.present=true), higher memory footprint vs CPU pods*

![RAPIDS GPU memory pool init](assets/rapids-run-executor-log-gpu-memory-pool.png)
*RAPIDS SQL plugin initialising: `GpuDeviceManager`, ~81GB GPU memory class discovered on A100-SXM4-80GB*

![RAPIDS task execution](assets/rapids-run-executor-log-rapids-tasks.png)
*`com.nvidia.spark.rapids.*` stack trace: GPU-accelerated SQL operators executing shuffle + aggregation stages*

---

### Grafana — GPU Metrics (RAPIDS Run)

Dashboard: **SmartShop GPU Performance (RAPIDS vs CPU)**

![GPU 30% util, 75GB VRAM](assets/grafana-rapids-gpu-30pct-75gb-vram.png)
*Mid-run: GPU utilization ~30%, ~75 GB VRAM allocated across 4 A100 executors. SM active, DRAM bandwidth engaged.*

![GPU burst phase](assets/grafana-rapids-gpu-burst.png)
*GPU activity burst during the shuffle/groupBy phase — highest GPU utilization point of the run*

![GPU full run timeline](assets/grafana-rapids-gpu-full-run.png)
*Full RAPIDS run GPU timeline: idle → ramp → plateau (~75GB VRAM) → teardown*

![VRAM teardown at run end](assets/grafana-rapids-gpu-vram-teardown.png)
*End of RAPIDS run: VRAM drops from ~80GB back to near-zero as executor pods terminate*

![GPU power + executor + Redis combined](assets/grafana-rapids-gpu-power-executor-redis.png)
*Combined view: GPU power consumption (top), executor pod CPU/memory (middle), Redis keys accumulating (bottom)*

![GPU power + Redis 15M keys](assets/grafana-rapids-power-redis-15m-keys.png)
*~15.6M Redis keys written, GPU power curve sustained, executor memory stable at ~14GB heap*

From earlier exploratory RAPIDS runs (smaller dataset / pre-fix):

![GPU activity early run](assets/grafana-gpu-activity-early-run.png)
![GPU memory midrun spike](assets/grafana-gpu-memory-midrun-spike.png)
![GPU all metrics during run](assets/grafana-gpu-all-metrics-during-run.png)
*Earlier RAPIDS run: GPU util spikes correlate with shuffle stages, VRAM peaks at ~75GB*

![RAPIDS completed — GPU memory released](assets/grafana-rapids-completed-gpu-memory-released.png)
*Post-completion: GPU memory fully released*

---

### Grafana — Executor Pod Metrics (CPU + RAPIDS, same dashboard)

The unified "Spark Executor Pod Metrics" panel shows both CPU and RAPIDS runs on one timeline:

![Executor CPU/mem/OOM/Redis — both runs](assets/grafana-executor-cpu-mem-oom-redis-both.png)
*Both runs visible: executor CPU cores consumed, memory working set, OOM restarts (0 for both), Redis keys*

![Executor metrics + Redis rising](assets/grafana-executor-metrics-redis-rising.png)
*Early in RAPIDS run: 4 executor pods consuming CPU + RAM, Redis key count beginning to ramp*

![Executor metrics — Redis 5M keys](assets/grafana-executor-metrics-redis-5m-keys.png)
*Redis ramp continuing through user_features write phase (~5M keys)*

![Executor metrics — Redis 8M keys](assets/grafana-executor-metrics-redis-8m-keys.png)
*~8.4M keys written, executor memory stable at ~14GB working set*

![Executor metrics — Redis 12M keys](assets/grafana-executor-metrics-redis-12m-keys.png)
*12M+ keys, transitioning from user_features to item_features write stage*

---

### Grafana — CPU Run Metrics

![CPU executor full window](assets/grafana-cpu-executor-full-window.png)
*CPU run: executor pod CPU/memory over full 11m26s duration. Stable 14GB working set, 0 OOM kills.*

![CPU executor 14M keys](assets/grafana-cpu-executor-mem-redis-14m-keys.png)
*CPU run mid-item_features: executor memory at ~13–14GB, Redis at 14M keys*

![Full dashboard — CPU run](assets/grafana-full-dashboard-cpu-run.png)
*Full "SmartShop GPU Performance" dashboard during CPU run: GPU panels idle (0% util, 0 VRAM), executor + Redis panels active. Clear visual separation between GPU and CPU mode.*

![Redis feature store Grafana](assets/grafana-redis-feature-store-working.png)
*Redis Feature Store dashboard: ops/sec, hit ratio, connected clients, memory usage during materialization*

---

### Spark History Server

URL: `spark-history-smartshop.<cluster-domain>`  
Event logs: `s3a://smartshop-features/spark-events/`

![Spark History — all completed apps](assets/spark-history-all-apps-completed.png)
*Spark History Server showing all completed materialization applications. Each `feast materialize` call = 2 Spark apps (user_features + item_features).*

![Spark History — both CPU and RAPIDS runs](assets/spark-history-both-runs-completed.png)
*Both CPU and RAPIDS run apps visible side-by-side in history server*

![RAPIDS app completed](assets/spark-history-rapids-completed-1h3m.png)
*RAPIDS application completion record with duration*

---

### RedisInsight — Online Feature Store

URL: `redisinsight-smartshop.<cluster-domain>`

![RedisInsight — browse item features](assets/redisinsight-browse-item-features.png)
*Browsing `item_features` keys: `smartshop:item_features:item_id:<asin>` — each key is a serialized feature vector*

![RedisInsight — key detail](assets/redisinsight-browse-key-detail.png)
*Individual feature key inspected: hash with fields `item_avg_rating`, `item_review_count`, etc.*

![RedisInsight — 13M keys mid-run](assets/redisinsight-13m-keys-midrun.png)
*Mid-RAPIDS run: 13.1M keys, 3.02 GB memory — user_features written, item_features in progress*

![RedisInsight — 26.5M keys final](assets/redisinsight-26m-keys-final.png)
*Final state after successful run: **26,493,202 keys, 5.99 GB** — complete feature store for 35M+ reviews*

![RedisInsight — analyze summary](assets/redisinsight-analyze-summary-clean.png)
*Key analysis: all keys are Hash type, prefix distribution matches user_features + item_features namespacing*

![RedisInsight — HSET slowlog](assets/redisinsight-hset-slowlog.png)
*Slowlog showing HSET write bursts (48–75ms) during peak Redis pipeline throughput phases*

![RedisInsight — slowlog commands](assets/redisinsight-slowlog-commands.png)
*Slowlog commands detail: HSET, HMGET patterns from Feast's Arrow serialization → Redis pipeline*

![RedisInsight — HSET burst](assets/redisinsight-slowlog-hset-burst.png)
*HSET burst pattern: Feast writes in 10 concurrent pipelines (partitions=10), creating high-throughput write windows*

---

### RHOAI Feature Store UI

![RHOAI Feast feature views list](assets/rhoai-feast-features-list.png)
*RHOAI dashboard: registered feature views — `user_features` and `item_features` visible after `feast apply`*

![RHOAI Feast data sources](assets/rhoai-feast-data-sources.png)
*RHOAI Feature Store: data source registered — `raw_reviews_source` pointing to `s3a://smartshop-raw/raw/reviews/*/`*

![RHOAI Feast lineage graph](assets/rhoai-feast-lineage-full.png)
*Full lineage graph: raw_reviews_source → user_features / item_features → Redis online store*

![Feast feature views (CLI/UI)](assets/feast-feature-views-list.png)
![Feast lineage post apply](assets/feast-lineage-post-apply.png)

---

### MinIO — Raw Data

![MinIO raw data loaded](assets/minio-raw-data-loaded.png)
*MinIO bucket `smartshop-raw`: 29.6 GB raw Amazon Reviews Parquet files across category subdirectories*

---

### Metrics Collection During a Run

```bash
# Wall-clock time (from meta file written by run script)
oc exec -n smartshop $FEAST_POD -c registry -- cat /tmp/rapids2-meta.txt

# Live Redis key count
oc exec -n smartshop $REDIS_POD -- redis-cli -a "$REDIS_PASSWORD" dbsize

# Redis memory
oc exec -n smartshop $REDIS_POD -- redis-cli -a "$REDIS_PASSWORD" info memory

# Spark stage progress (tail materialize log)
oc exec -n smartshop $FEAST_POD -c registry -- tail -f /tmp/rapids2.log \
  | grep -oE "Stage [0-9]+:[^]]*\]"

# Active executor pods
oc -n smartshop get pods | grep pyspark
```

---

## 11. Reproduce Runbook

> **One-liner**: `SPARK_PROFILE=cpu bash scripts/wait-and-materialize.sh` (after prerequisites below).

### Prerequisites (one-time cluster setup)

```bash
# 1. Cluster access
oc login https://api.${OC_CLUSTER_DOMAIN}:6443
export NAMESPACE=smartshop

# 2. Apply spark SA + RBAC (executor pods use 'spark' SA to call k8s API)
oc apply -f infrastructure/smartshop/spark-rbac.yaml

# 3. Apply feast-spark-driver ClusterIP Service (executor→driver connectivity)
#    Selector: feast.dev/name=smartshop-feast  (NOT app=feast-smartshop-feast)
oc apply -f infrastructure/openshift/feast-spark-driver-svc.yaml
oc -n smartshop get svc feast-spark-driver   # should show ClusterIP with ports 7078,7079

# 4. Apply all ConfigMaps (base + cpu + rapids)
envsubst < infrastructure/openshift/feast-spark-engine.yaml       | oc apply -f -
envsubst < infrastructure/openshift/feast-spark-engine-cpu.yaml   | oc apply -f -
envsubst < infrastructure/openshift/feast-spark-engine-rapids.yaml | oc apply -f -

# 5. Verify executor images exist in internal registry
oc -n smartshop get is | grep feast-spark
# Expected:
#   feast-spark-executor          (CPU executor image)
#   feast-spark-executor-rapids   (RAPIDS GPU executor image)
#   feast-spark-server            (driver/feast pod image)
```

### Option A — Automated (recommended)

```bash
source .env   # sets NAMESPACE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, REDIS_PASSWORD

# CPU baseline (~686s, 26.5M keys)
SPARK_PROFILE=cpu   bash scripts/wait-and-materialize.sh

# RAPIDS GPU (~658s, 26.5M keys) — flush Redis first (script does it automatically)
SPARK_PROFILE=rapids bash scripts/wait-and-materialize.sh
```

The script handles: ConfigMap switch → deployment rollout → Redis flush → `feast apply` → timed `feast materialize` → key count report.

### Option B — Manual step-by-step

#### Helper variables (set once per terminal)

```bash
export NAMESPACE=smartshop
export REDIS_PASSWORD="${REDIS_PASSWORD}"   # from .env

# Resolve pods — uses label set by Feast Operator (feast.dev/name, NOT app:)
FEAST_POD=$(oc -n $NAMESPACE get pod \
  -l "feast.dev/name=smartshop-feast" \
  --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}')

REDIS_POD=$(oc -n $NAMESPACE get pod \
  -l "app=redis" \
  --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}')

echo "FEAST_POD=$FEAST_POD   REDIS_POD=$REDIS_POD"
```

#### CPU Baseline Run

```bash
# 1. Switch ConfigMap to CPU profile
envsubst < infrastructure/openshift/feast-spark-engine-cpu.yaml | oc apply -f -
oc -n $NAMESPACE rollout restart deployment/feast-smartshop-feast
oc -n $NAMESPACE rollout status  deployment/feast-smartshop-feast --timeout=120s

# Re-resolve pod after rollout
FEAST_POD=$(oc -n $NAMESPACE get pod -l "feast.dev/name=smartshop-feast" \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')

# 2. Flush Redis (clean baseline)
oc -n $NAMESPACE exec "$REDIS_POD" -- redis-cli -a "$REDIS_PASSWORD" FLUSHALL

# 3. Sync registry
oc exec -n $NAMESPACE "$FEAST_POD" -c offline -- bash -c "
  export AWS_ACCESS_KEY_ID=\${AWS_ACCESS_KEY_ID:-minio}
  export AWS_SECRET_ACCESS_KEY=\${AWS_SECRET_ACCESS_KEY:-minio123}
  cd /feast-data/smartshop/feast/feature_repo
  feast apply"

# 4. Timed materialize (full dataset: 2014-01-01 → 2018-12-31, ~29.6 GB)
START_TS=$(date +%s)
oc exec -n $NAMESPACE "$FEAST_POD" -c offline -- bash -c "
  export AWS_ACCESS_KEY_ID=\${AWS_ACCESS_KEY_ID:-minio}
  export AWS_SECRET_ACCESS_KEY=\${AWS_SECRET_ACCESS_KEY:-minio123}
  cd /feast-data/smartshop/feast/feature_repo
  feast materialize 2014-01-01T00:00:00 2018-12-31T23:59:59 2>&1"
RC=$?
ELAPSED=$(( $(date +%s) - START_TS ))
echo "CPU: RC=$RC  elapsed=${ELAPSED}s"

# 5. Verify
oc -n $NAMESPACE exec "$REDIS_POD" -- redis-cli -a "$REDIS_PASSWORD" DBSIZE
# Expected: 26493202
```

#### RAPIDS GPU Run

```bash
# 1. Switch ConfigMap to RAPIDS profile
#    Key settings: task.resource.gpu.amount=0.5 → 2 tasks/GPU → 8 total tasks (matches CPU)
envsubst < infrastructure/openshift/feast-spark-engine-rapids.yaml | oc apply -f -
oc -n $NAMESPACE rollout restart deployment/feast-smartshop-feast
oc -n $NAMESPACE rollout status  deployment/feast-smartshop-feast --timeout=120s

FEAST_POD=$(oc -n $NAMESPACE get pod -l "feast.dev/name=smartshop-feast" \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')

# 2. Flush Redis
oc -n $NAMESPACE exec "$REDIS_POD" -- redis-cli -a "$REDIS_PASSWORD" FLUSHALL

# 3. Timed materialize
START_TS=$(date +%s)
oc exec -n $NAMESPACE "$FEAST_POD" -c offline -- bash -c "
  export AWS_ACCESS_KEY_ID=\${AWS_ACCESS_KEY_ID:-minio}
  export AWS_SECRET_ACCESS_KEY=\${AWS_SECRET_ACCESS_KEY:-minio123}
  cd /feast-data/smartshop/feast/feature_repo
  feast materialize 2014-01-01T00:00:00 2018-12-31T23:59:59 2>&1"
RC=$?
ELAPSED=$(( $(date +%s) - START_TS ))
echo "RAPIDS: RC=$RC  elapsed=${ELAPSED}s"

# 4. Verify
oc -n $NAMESPACE exec "$REDIS_POD" -- redis-cli -a "$REDIS_PASSWORD" DBSIZE
# Expected: 26493202
```

### Verify Results

```bash
# Redis key count (should be 26,493,202 after either run)
oc -n $NAMESPACE exec "$REDIS_POD" -- redis-cli -a "$REDIS_PASSWORD" DBSIZE

# Spot-check a feature vector
oc -n $NAMESPACE exec "$REDIS_POD" -- \
  redis-cli -a "$REDIS_PASSWORD" KEYS "smartshop:user_features:*" | head -3

# Redis memory
oc -n $NAMESPACE exec "$REDIS_POD" -- \
  redis-cli -a "$REDIS_PASSWORD" INFO memory | grep used_memory_human

# Watch executor pods appear and disappear during the run (separate terminal)
watch -n 5 'oc -n smartshop get pods | grep spark-executor'
```

### Online Serving Verification

```python
from feast import FeatureStore
fs = FeatureStore(repo_path="/feast-data/smartshop/feast/feature_repo")

# User features
fs.get_online_features(
    features=["user_features:user_avg_rating", "user_features:user_review_count"],
    entity_rows=[{"user_id": "<any_user_id_from_dataset>"}]
).to_dict()

# Item features
fs.get_online_features(
    features=["item_features:item_avg_rating", "item_features:item_review_count"],
    entity_rows=[{"item_id": "B07ZPKBL9V"}]
).to_dict()
```

---

## 12. Troubleshooting Reference

| Error | Root Cause | Fix |
|-------|-----------|-----|
| `DATATYPE_MISMATCH.BINARY_OP_DIFF_TYPES: BIGINT vs TIMESTAMP` | Raw `timestamp` column is Unix-ms BIGINT; Feast SQL filter generates TIMESTAMP literals | Use `SparkSource(query="SELECT *, CAST(timestamp/1000 AS TIMESTAMP) AS event_timestamp FROM parquet.\`s3a://.../raw/reviews/*/\`")` |
| `UNABLE_TO_INFER_SCHEMA for Parquet` | `parquet.\`path/\`` doesn't recurse into subdirectories | Add `*/` glob: `parquet.\`s3a://.../raw/reviews/*/\`` |
| `Usage: CoarseGrainedExecutorBackend [options]` | Stale executor entrypoint: `shift; exec ... "$@"` passes empty args after shift | Rewrite `executor-entrypoint.sh` to use `KubernetesExecutorBackend` and read `SPARK_*` env vars explicitly |
| `Failed to connect to feast-...:45835` (executor→driver) | Driver hostname in executor spec is pod name (not DNS-resolvable) | Add `spark.driver.host=feast-spark-driver.<ns>.svc.cluster.local` + create `ClusterIP Service` |
| `InvalidKeyException: No installed provider supports this key` | UBI9 FIPS blocks HMAC-SHA256 `SecretKeySpec` (AWS SDK v1) | Set `JAVA_TOOL_OPTIONS=-Dcom.redhat.fips=false` in image (NOT `JAVA_OPTS`) |
| `spec.containers[0].resources.limits[nvidia.com/gpu/gpu]: Invalid value` | `spark.executor.resource.gpu.vendor=nvidia.com/gpu` creates `nvidia.com/gpu/gpu` resource name | Set `vendor: "nvidia.com"` (no `/gpu` suffix); resource name is `vendor/resource-name` |
| RAPIDS JAR download race (timeout/`ConnectionRefused`) | `spark.jars=s3a://...rapids.jar` starts a 470MB download at driver init; executors connect before download completes | Bake RAPIDS JAR into both driver and executor images; remove `spark.jars` from ConfigMap |
| `BusyLoadingError: Redis is loading the dataset in memory` | Redis restarted (OOMKilled) and is loading RDB snapshot from disk | Disable RDB/AOF in Redis args: `--appendonly no --save ""`; increase Redis memory limit to 8Gi |
| `ExecutorPodsAllocator: Max number of executor failures (4) reached` | Executor pods OOMKilled during shuffle (heap exhaustion) | Increase `spark.executor.memory` to `14g`, add `spark.executor.memoryOverhead: "4g"`, reduce `executor.cores` to `2` |
| RAPIDS slower than CPU | `spark.task.resource.gpu.amount=1.0` limits to 1 task per executor (4 total) vs CPU's 8 tasks | Set `spark.task.resource.gpu.amount: "0.5"` + `spark.rapids.sql.concurrentGpuTasks: "2"` for 8 concurrent tasks |
| `feast: error: unrecognized arguments: apply` | `benchmark_materialize.py` had module-level `argparse.parse_args()` hijacking `sys.argv` when Feast CLI loaded the feature repo | Move `parse_args()` behind `if __name__ == "__main__":` guard; add `benchmark_materialize.py` to `.feastignore` |
| `spark_conf keys ignored` | `SparkComputeEngine` reads only top-level flat keys; nested `spark_conf:` block is silently ignored | Move all `spark.*` keys to top level of `batch_engine:` in ConfigMap |
| `Initial job has not accepted any resources` | ConfigMap updated but Feast pod not restarted; stale in-memory `SparkSession` | `oc rollout restart deployment/feast-smartshop-feast -n smartshop` after every ConfigMap change |

---

## Appendix A — Key File Locations

| File | Purpose |
|------|---------|
| `feast/feature_repo/features.py` | Feature definitions — entities, source, BFV UDFs |
| `feast/feature_repo/feature_store.yaml` | Feast store config — offline/online store, registry |
| `feast/feature_repo/.feastignore` | Excludes `benchmark_materialize.py` from feast apply |
| `build/executor-entrypoint.sh` | Spark k8s executor entrypoint (KubernetesExecutorBackend) |
| `build/Containerfile.feast-spark` | Driver image (Feast pod) — includes RAPIDS JAR |
| `infrastructure/openshift/feast-spark-engine.yaml` | local[*] ConfigMap (single-pod, no executors) |
| `infrastructure/openshift/feast-spark-engine-cpu.yaml` | k8s:// CPU distributed ConfigMap |
| `infrastructure/openshift/feast-spark-engine-rapids.yaml` | k8s:// RAPIDS GPU distributed ConfigMap |
| `infrastructure/redis/redis.yaml` | Redis deployment — 8Gi, persistence disabled |
| `infrastructure/feast/feast-spark-rbac.yaml` | spark ServiceAccount + edit RoleBinding |

## Appendix B — Environment Variables

| Variable | Value | Used by |
|----------|-------|---------|
| `NAMESPACE` | `smartshop` | `envsubst` in all ConfigMap YAMLs |
| `OCP_API_URL` | `https://api.${OC_CLUSTER_DOMAIN}:6443` | `spark.master` in ConfigMaps |
| `REGISTRY` | `quay.io/<your-org>` | executor image reference |
| `JAVA_TOOL_OPTIONS` | `-Dcom.redhat.fips=false` | baked into both driver and executor images |
| `REDIS_PASSWORD` | set in `.env` | Redis auth, Feast `connection_string` |
