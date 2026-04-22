# Feast `@batch_feature_view` — Implementation Reference

**Branch:** `feat/feast-batch-feature-view-transformation`  
**Goal:** Replace `SparkApplication` CRs for `user_features` / `item_features` with Feast-native
transformation + materialization using `@batch_feature_view` + `SparkComputeEngine`.

---

## 1. Architecture

### Before
```
Raw Parquet (s3a://smartshop-raw/raw/reviews/)
    ↓
[spark-application-rapids.yaml]       ← SparkApplication CR, RAPIDS GPU
[spark-application-cpu-baseline.yaml] ← SparkApplication CR, CPU fallback
    ↓ writes to s3a://smartshop-features/user_features/
                s3a://smartshop-features/item_features/
                s3a://smartshop-features/interactions/
Feast FeatureView (reads pre-built Parquet)
    ↓ feast materialize-incremental
Redis (online store)
```

### After (current)
```
Amazon Reviews download (setup phase)
    ↓ raw parquet: s3a://smartshop-raw/raw/reviews/ — BIGINT timestamp
    ↓ preprocess_reviews_full.py [ONE-TIME — adds event_timestamp TIMESTAMP column]
s3a://smartshop-raw/processed/reviews/
    ↓
feast apply && feast materialize-incremental
    ↓  SparkComputeEngine (local[*] in feast pod JVM)
    ↓  SELECT * FROM processed/reviews/ WHERE event_timestamp BETWEEN ...
@batch_feature_view UDF (groupBy/agg — same logic as feature_engineering.py)
    ↓
Redis (online store)   ← online=True
                       ← offline=False (see §4)

Training via get_historical_features()
    ↓  SparkComputeEngine reads raw → applies UDF on-demand
    ↓  no pre-computed Parquet needed
```

### What is NOT replaced by Feast BFV

| SparkApplication | Output | Why |
|---|---|---|
| `spark-application-text-preprocessing.yaml` | `s3a://smartshop-features/llm_data/` | Pipeline artifact for embedding model — not a feature store concern |
| `spark-application-embedding.yaml` | `s3a://smartshop-embeddings/` | Vector embeddings — separate pipeline, not a feature store concern |
| `interactions` output in `feature_engineering.py` | `s3a://smartshop-features/interactions/` | Could become a BFV with `offline=True` + dedicated sink; not needed for demo |

---

## 2. Materialization Command

```bash
FEAST_POD=$(oc get pod -n smartshop -l "feast.dev/name=smartshop-feast" \
  -o jsonpath='{.items[0].metadata.name}')

oc exec -n smartshop $FEAST_POD -c offline -- bash -c '
  export AWS_ACCESS_KEY_ID=minio
  export AWS_SECRET_ACCESS_KEY=minio123
  cd /feast-data/smartshop/feast/feature_repo
  feast apply
  feast materialize-incremental "$(date -u +%Y-%m-%dT%H:%M:%S)"
'
```

**Result:** ~2.9M Redis keys — `user_features` (per `user_id`) + `item_features` (per `item_id`).  
`JAVA_TOOL_OPTIONS=-Dcom.redhat.fips=false` is baked into the image — no manual export needed.

Verified online serving:
```python
fs.get_online_features(
    features=['user_features:user_avg_rating', 'user_features:user_review_count'],
    entity_rows=[{'user_id': 'AHO2HFBROLSL5BHJ2IOMLHHFFHWQ'}]
).to_dict()
# → user_avg_rating: [1.89], user_review_count: [9]
```

---

## 3. The `offline=False` Requirement (CRITICAL)

From `nodes.py — SparkWriteNode.execute()`:
```python
if self.feature_view.online:
    spark_df.mapInArrow(lambda x: map_in_arrow(x, serialized_artifacts, mode="online"), ...)

if self.feature_view.offline:
    # ⚠️ APPENDS transformed df BACK TO batch_source.path
    dest_path = self.feature_view.batch_source.path
    spark_df.write.format(file_format).mode("append").save(dest_path)
```

`batch_source.path = "s3a://smartshop-raw/processed/reviews/"`. With `offline=True`, the engine
appends output feature rows back into the raw review data — **corrupting the source**.

```python
online=True,    # ✅ write to Redis
offline=False,  # ✅ skip write-back to raw source
```

**Training is unaffected.** `get_historical_features()` reads from `batch_source` directly
(raw → UDF → return features) — the `offline` flag is not consulted during reads.

---

## 4. Source Data: Why `processed/reviews/` Instead of `raw/reviews/`

### `processed/reviews/` is NOT from `feature_engineering.py`

Data lineage:
```
s3a://smartshop-raw/raw/reviews/          ← BIGINT timestamp (from Amazon Reviews download)
    ↓ preprocess_reviews_full.py (one-time, scripts/ → copy to feast pod)
s3a://smartshop-raw/processed/reviews/   ← adds event_timestamp TIMESTAMP column only
    ↓ @batch_feature_view UDF (the actual feature engineering)
Redis
```

`feature_engineering.py` writes to `s3a://smartshop-features/user_features/` and `item_features/`.
The BFV approach reads directly from `processed/reviews/` and recomputes the same aggregations in
the UDF. The preprocessing step is separate and trivial.

### Why preprocessing is needed

Feast's `SparkOfflineStore` generates SQL comparing `timestamp_field` against `TIMESTAMP` literals:
```sql
WHERE timestamp >= TIMESTAMP '2023-01-01 00:00:00'
-- AnalysisException: BINARY_OP_DIFF_TYPES: BIGINT vs TIMESTAMP
```
Raw data has `timestamp` as `BIGINT` (Unix milliseconds). The one-time preprocessing converts it:

```python
# scripts/preprocess_reviews_full.py — run once, copy to feast pod and execute
# Reads raw/reviews/{Electronics,Books,Home_and_Kitchen}/ (partitioned Parquet)
# and writes merged output with event_timestamp to processed/reviews/
df.withColumn("event_timestamp",
              F.to_timestamp(F.from_unixtime(F.col("timestamp") / 1000))) \
  .write.mode("append").parquet("s3a://smartshop-raw/processed/reviews/")
```

All three categories are written into a single flat `processed/reviews/` directory so Spark
infers schema from the top-level path. `features.py` `SparkSource` points to `processed/reviews/`
with `timestamp_field="event_timestamp"`.

The script lives at `scripts/preprocess_reviews_full.py` in the repo. Copy it to the feast pod
before running (there is no automated sync — the operator only mounts `feast/feature_repo/`):

```bash
FEAST_POD=$(oc get pod -n smartshop -l app=feast-smartshop-feast -o name | head -1)
SCRIPT=$(python3 -c "import json; print(json.dumps(open('scripts/preprocess_reviews_full.py').read()))")
oc exec -n smartshop $FEAST_POD -c offline -- python3 -c "
import json
with open('/feast-data/smartshop/feast/feature_repo/preprocess_reviews_full.py','w') as f:
    f.write(json.loads('$SCRIPT'))
"
oc exec -n smartshop $FEAST_POD -c offline -- bash -c '
  export AWS_ACCESS_KEY_ID=minio
  export AWS_SECRET_ACCESS_KEY=minio123
  python3 /feast-data/smartshop/feast/feature_repo/preprocess_reviews_full.py 2>&1
'
```

Once `processed/reviews/` exists, `feast materialize-incremental` runs with no SparkApplication
dependency.

---

## 5. Full DAG Execution

```
feast materialize-incremental
    ↓
SparkComputeEngine._materialize_one()
    ↓
SparkFeatureBuilder.build()
    ↓

[SparkReadNode]
  ↓ get_column_info() → feature_cols=[]
  ↓ pull_all_from_table_or_query()
  ↓   → SELECT * FROM processed/reviews/
  ↓   → PySpark DataFrame (all raw columns)
  ↓ Applies start_time / end_time filter on event_timestamp

[SparkTransformationNode]
  ↓ udf(raw_df) → transformed_df
  ↓ This is the @batch_feature_view function body

[SparkFilterNode]
  ↓ TTL filter: event_timestamp >= now - ttl

[SparkDedupNode]   ← only for HistoricalRetrievalTask or task.only_latest=True
  ↓ ROW_NUMBER() OVER (PARTITION BY entity ORDER BY event_timestamp DESC) = 1

[SparkValidationNode]
  ↓ checks declared schema[] columns exist + type compatibility

[SparkWriteNode]
  ↓ online=True  → mapInArrow → map_in_arrow(mode="online") → Redis
  ↓ offline=False → skip
```

Dedup is skipped for `MaterializationTask` with default config — `groupBy` already produces
one row per entity.

---

## 6. UDF Contract

The UDF receives a PySpark DataFrame with all raw source columns and must return:

### `user_features`
| Column | Type | Note |
|--------|------|------|
| `user_id` | String | entity join key |
| `event_timestamp` | Timestamp | `F.current_timestamp()` |
| `user_avg_rating` | Double | |
| `user_review_count` | Long | |
| `user_unique_items` | Long | |
| `user_avg_review_length` | Double | |
| `user_category_count` | Long | |
| `user_tenure_days` | Long | |

### `item_features`
| Column | Type | Note |
|--------|------|------|
| `item_id` | String | renamed from `parent_asin` |
| `event_timestamp` | Timestamp | |
| `item_avg_rating` | Double | |
| `item_rating_stddev` | Double | |
| `item_review_count` | Long | |
| `item_total_helpful_votes` | Long | |
| `item_avg_review_length` | Double | |
| `item_price` | Float | null — metadata join not in BFV |

**All imports must be inside the function body.** `dill` serializes the UDF — module-level
imports are not available on executors.

---

## 7. Infrastructure Configuration

### Image: `${REGISTRY}/smartshop-feast-spark-server:latest`

Built by `build/Containerfile.feast-spark` from `quay.io/feastdev/feature-server:0.62.0` +
JDK 11 + PySpark 3.5.3 + hadoop-aws JARs.

Key decisions:
- **`JAVA_TOOL_OPTIONS=-Dcom.redhat.fips=false`** — the standard JVM env var read at startup by
  all JVM processes including PySpark's Py4J gateway. Required on UBI9/RHEL9 to allow
  AWS SDK v1 HMAC-SHA256 signing (`javax.crypto.spec.SecretKeySpec`). `JAVA_OPTS` is NOT
  sufficient — it is only read by shell scripts that explicitly pass `$JAVA_OPTS` to `java`.
- **PySpark 3.5.3** (not 4.0.0) — PySpark 4.0 bundles Hadoop 3.4 which requires AWS SDK v2;
  the JARs in the image target SDK v1. Pinned: `hadoop-aws-3.3.4.jar` + `aws-java-sdk-bundle-1.12.367.jar`.
- **JDK 11** — minimum required by PySpark 3.5.3 to launch the JVM (SparkContext).

### Redis memory

Default 512Mi is too small for ~2.9M feature vectors. Patched to 2Gi:
```bash
oc patch deployment redis -n smartshop --type=json \
  -p='[{"op":"replace","path":"/spec/template/spec/containers/0/resources/limits/memory","value":"2Gi"}]'
```

### `feast-spark-engine` ConfigMap

Controls `batch_engine` config — injected by the Feast Operator into `feature_store.yaml`.
Source: `infrastructure/openshift/feast-spark-engine.yaml`.

Key settings:
```yaml
partitions: 10          # 10 concurrent Redis pipeline connections (200 caused OOMKill)
spark.master: local[*]  # Spark in feast pod JVM; k8s:// requires dedicated executor image
spark.driver.memory: 8g
```

### `SparkComputeEngineConfig` layout

All `spark.*` keys must be **flat at the top level** of `batch_engine` — NOT nested under `spark_conf:`:
```yaml
# ✅ Correct
batch_engine:
  type: spark.engine
  partitions: 10
  spark.master: "local[*]"
  spark.driver.memory: "8g"

# ❌ Wrong — SparkComputeEngine ignores nested spark_conf
batch_engine:
  type: spark.engine
  spark_conf:
    spark.master: "local[*]"
```

`feature_store.yaml` does NOT support `${VAR:-default}` syntax — only plain `${VAR}` via
`os.path.expandvars`. Use static values.

---

## 8. RAPIDS GPU Path (Future)

When running with a proper Spark executor image, add to `feast-spark-engine` ConfigMap:
```yaml
spark.master: "k8s://https://api.cluster:6443"
spark.plugins: "com.nvidia.spark.SQLPlugin"
spark.rapids.sql.enabled: "true"
spark.rapids.sql.concurrentGpuTasks: "2"
spark.kubernetes.executor.resource.gpu.amount: "1"
spark.kubernetes.executor.resource.gpu.vendor: "nvidia.com/gpu"
spark.executor.instances: "4"
spark.executor.memory: "8g"
spark.kubernetes.container.image: "${REGISTRY}/smartshop-feast-spark-server:latest"
spark.kubernetes.authenticate.driver.serviceAccountName: "spark"
```

Requirement: the executor image needs `/opt/spark/bin/executor`. The current `feast-server`
base image does not include it. Use a Spark-native base image for executors, or build a
combined image from `apache/spark:3.5.3`.

---

## 9. Implementation Checklist

- [x] `feast/feature_repo/features.py` — `@batch_feature_view` for `user_features` + `item_features`
- [x] `processed/reviews/` — one-time preprocessing with `event_timestamp TIMESTAMP`
- [x] `Containerfile.feast-spark` — JDK 11 + `JAVA_TOOL_OPTIONS` + PySpark 3.5.3 + S3A JARs
- [x] `feast-spark-engine` ConfigMap — `local[*]` + `partitions: 10`
- [x] `feast apply` — views registered in registry
- [x] `feast materialize-incremental` — ~2.9M keys in Redis, online serving verified
- [x] `scripts/wait-and-materialize.sh` — updated to use `feast materialize-incremental` (bfv3)
- [ ] RAPIDS GPU `batch_engine` config — pending dedicated executor image (bfv2)

---

## 10. Troubleshooting Reference

| Error | Cause | Fix |
|-------|-------|-----|
| `DATATYPE_MISMATCH.BINARY_OP_DIFF_TYPES` | Raw `timestamp` BIGINT vs Feast SQL TIMESTAMP | One-time preprocess to `processed/reviews/` |
| `UNABLE_TO_INFER_SCHEMA` | Subdirectories under processed reviews | Write all categories to flat `processed/reviews/` dir |
| `InvalidKeyException: No installed provider supports this key` | UBI9 FIPS blocks HMAC-SHA256 | `JAVA_TOOL_OPTIONS=-Dcom.redhat.fips=false` in image (NOT `JAVA_OPTS`) |
| `NoClassDefFoundError: software/amazon/awssdk` | PySpark 4.0 + Hadoop 3.4 requires SDK v2 | Pin PySpark to 3.5.3 + hadoop-aws-3.3.4 + aws-java-sdk-bundle-1.12.367 |
| `JAVA_HOME is not set` | Base feature-server image has no JDK | `microdnf install java-11-openjdk-headless` in Containerfile |
| `spark_conf keys ignored in batch_engine` | SparkComputeEngine reads flat keys only | Move all `spark.*` to top level of `batch_engine`, not under `spark_conf` |
| Redis OOMKilled (exit 137) | 200 concurrent Spark partitions → 200 Redis pipeline connections | `partitions: 10` in `batch_engine` + Redis limit 2Gi |
| `CreateContainerError: executor not found` | `k8s://` executor pods use feast-server image lacking Spark executor binary | Use `local[*]` for demo; separate executor image needed for k8s:// |
| `Initial job has not accepted any resources` | ConfigMap-injected `batch_engine` had `k8s://` master, pod not restarted after ConfigMap patch | Restart feast deployment after patching ConfigMap |
