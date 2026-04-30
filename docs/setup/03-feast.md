## 8. Deploy Feast Feature Store

Feast is installed and managed by the RHOAI Feast Operator (enabled by default in the DSC as `FeastOperatorReady`).

### Architecture

SmartShop uses **`@batch_feature_view` (BFV)** — Feast-native transformations that compute features on-demand from raw data, bypassing the intermediate Parquet step:

```
                Amazon Reviews download (HuggingFace → MinIO)
                    s3a://smartshop-raw/raw/reviews/{category}/
                              │
                              │ preprocess_reviews_full.py  ← ONE-TIME
                              │ adds event_timestamp TIMESTAMP column
                              ▼
                    s3a://smartshop-raw/processed/reviews/
                              │
              ┌───────────────┼───────────────────────────┐
              ▼               ▼                           ▼
    ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │  feast-spark-   │  │  rec-trainer pod │  │  KServe          │
    │  server pod     │  │  (TrainJob)      │  │  InferenceService│
    │                 │  │                  │  │                  │
    │  @batch_feature │  │  get_historical  │  │  get_online      │
    │  _view UDF      │  │  _features()     │  │  _features()     │
    │  groupBy/agg    │  │  BFV UDF on-     │  │  Redis → model   │
    │  → Redis online │  │  demand (no      │  └──────────────────┘
    └─────────────────┘  │  Parquet needed) │
                         └──────────────────┘
```

| Component | Role |
|-----------|------|
| **`@batch_feature_view` UDF** | Python function registered with Feast. Receives raw review DataFrame, returns aggregated user/item features. Same `groupBy/agg` logic as `feature_engineering.py` — runs inside Feast instead of a separate SparkApp. |
| **SparkComputeEngine** | Executes the BFV UDF during `feast materialize-incremental`. Runs pyspark `local[*]` **inside** the feast server pod. Reads `processed/reviews/`, applies UDF, writes to Redis. |
| **SparkOfflineStore** | Powers `get_historical_features()` in the training pod. Reads raw Parquet → applies BFV UDF on-demand. No pre-computed feature Parquet needed. |
| **Redis online store** | Sub-ms feature lookup at inference time. Holds ~3.1M keys after materialization (user + item entities). |
| **Remote registry** | Training pod reads feature definitions via gRPC (`feast-smartshop-feast-registry:6570`) — no local registry file needed. |

> **Why BFV over separate SparkApps?** BFV eliminates the intermediate Parquet write entirely.
> Raw reviews → 3.1M feature vectors in Redis in one step — no `smartshop-features/` bucket needed for materialization.

The feast server uses the **`feast-spark-server` image**: `feature-server:0.62.0 + pyspark + Java 11 + hadoop-aws JARs + FIPS disabled` (built from `build/Containerfile.feast-spark`).
The rec-trainer uses a separate image with `pyspark + feast[grpc]==0.62.0` (built from `build/Containerfile.rec-trainer`).
**Note:** feast-spark-server must use pyspark 3.5.3 — pyspark 4.0 drops support for AWS SDK v1 which is required for S3A on this cluster.

### 8a — Build feast-spark-server image

The default RHOAI feature server image (`quay.io/feastdev/feature-server:0.62.0`) ships only `feast[minimal]` (no pyspark). Build the custom image:

```bash
source .env

# Apply ImageStream
envsubst < infrastructure/openshift/imagestreams.yaml | oc apply -f -

# Apply BuildConfig
envsubst < infrastructure/openshift/buildconfigs.yaml | oc apply -f -

# Start build (~5-10 min: pip install pyspark + 2 JAR downloads ~150MB total)
oc start-build feast-spark-server -n smartshop --follow
```

Verify:
```bash
oc get build -n smartshop | grep feast-spark
# feast-spark-server-1   Docker   Git@<sha>   Complete
```

### 8b — Apply Spark engine ConfigMap + Secret

```bash
source .env
envsubst < infrastructure/openshift/feast-spark-engine.yaml | oc apply -f -
```

- `feast-spark-engine` ConfigMap — batch engine config (`type: spark.engine`, spark_conf with s3a MinIO endpoint)
- `feast-spark-config` Secret — offline store spark_conf injected into `feature_store.yaml`

### 8c — Apply FeatureStore CR

```bash
source .env
envsubst < infrastructure/openshift/feast-operator.yaml | oc apply -f -
```

The CR:
- Sets `spec.batchEngine.configMapRef: feast-spark-engine` — `SparkComputeEngine`
- Sets `offlineStore.persistence.store.type: spark` — `SparkOfflineStore`
- Uses custom `feast-spark-server` image for all containers — includes pyspark + Hadoop-AWS JARs required by SparkOfflineStore
- Overrides all service images to `feast-spark-server:latest`
- Online store: Redis via `feast-redis-secret`
- Registry: file-backed PVC (1Gi `nfs-csi`)

> **NFS double-mount gotcha:** The Feast pod runs registry, offline, and online containers
> in the same pod. NFS CSI cannot mount the same PV twice within a single pod.
> This is why the registry gets its own dedicated 1Gi PVC instead of reusing
> `smartshop-shared-storage`. The offline/online containers use no PVC.

**Watch rollout (~2 min):**

```bash
oc get pods -n smartshop -l feast.dev/name=smartshop-feast -w
# feast-smartshop-feast-xxxxxxxxx-xxxxx   0/4   Pending → 4/4   Running
# (no Init: stages — init containers disabled)
```

**Verify:**

```bash
oc get featurestore -n smartshop
# NAME              STATUS   AGE
# smartshop-feast   Ready    Xm

# Confirm pyspark is available in the pod (must be 3.5.3 — not 4.0.0)
oc exec -n smartshop deploy/feast-smartshop-feast -c offline -- python3 -c "import pyspark; print(pyspark.__version__)"
# 3.5.3
```

### 8f — Preprocess reviews + `feast materialize-incremental` (SparkComputeEngine → Redis)

#### Step 1 — Preprocess reviews (ONE-TIME prerequisite)

Feast's `SparkOfflineStore` generates SQL that compares `timestamp_field` against `TIMESTAMP` literals. The raw Amazon Reviews data stores timestamps as `BIGINT` (Unix milliseconds), which causes `DATATYPE_MISMATCH: BIGINT vs TIMESTAMP` during materialization.

`scripts/preprocess_reviews_full.py` converts `timestamp BIGINT → event_timestamp TIMESTAMP` and writes to `processed/reviews/`:

```
s3a://smartshop-raw/raw/reviews/{Electronics,Books,Home_and_Kitchen}/  ← BIGINT timestamp
    ↓ preprocess_reviews_full.py
s3a://smartshop-raw/processed/reviews/                                  ← TIMESTAMP column added
```

The Feast Operator only mounts `feast/feature_repo/` into the pod — `scripts/` is not synced automatically. Copy and run the script manually:

```bash
source .env

FEAST_POD=$(oc get pod -n smartshop -l feast.dev/name=smartshop-feast \
  -o jsonpath='{.items[0].metadata.name}')

# Copy script into the feast pod
SCRIPT=$(python3 -c "import json; print(json.dumps(open('scripts/preprocess_reviews_full.py').read()))")
oc exec -n smartshop "$FEAST_POD" -c offline -- python3 -c "
import json
with open('/feast-data/smartshop/feast/feature_repo/preprocess_reviews_full.py','w') as f:
    f.write(json.loads('$SCRIPT'))
"

# Run preprocessing (~15–30 min for full 140M-row dataset)
oc exec -n smartshop "$FEAST_POD" -c offline -- bash -c "
  export AWS_ACCESS_KEY_ID=${MINIO_ACCESS_KEY:-minio}
  export AWS_SECRET_ACCESS_KEY=${MINIO_SECRET_KEY:-minio123}
  python3 /feast-data/smartshop/feast/feature_repo/preprocess_reviews_full.py 2>&1
"
```

Verify the output exists:
```bash
oc exec -n smartshop "$FEAST_POD" -c offline -- python3 -c "
import os, s3fs
fs = s3fs.S3FileSystem(
    key=os.environ['AWS_ACCESS_KEY_ID'],
    secret=os.environ['AWS_SECRET_ACCESS_KEY'],
    endpoint_url=os.environ['AWS_ENDPOINT_URL_S3'],
    client_kwargs={'verify': False},
)
files = fs.ls('smartshop-raw/processed/reviews/')
print(f'{len(files)} files in processed/reviews/')
"
# 30+ files in processed/reviews/
```

#### Step 2 — `feast apply` + `feast materialize-incremental`

`feast materialize-incremental` uses `SparkComputeEngine` to read `processed/reviews/`, run the `@batch_feature_view` UDF (`groupBy/agg`), and write features to Redis. No `smartshop-features/` Parquet is needed.

```bash
source .env

FEAST_POD=$(oc get pod -n smartshop -l "feast.dev/name=smartshop-feast" \
  -o jsonpath='{.items[0].metadata.name}')

oc exec -n smartshop "$FEAST_POD" -c offline -- bash -c "
  export AWS_ACCESS_KEY_ID=${MINIO_ACCESS_KEY:-minio}
  export AWS_SECRET_ACCESS_KEY=${MINIO_SECRET_KEY:-minio123}
  cd /feast-data/smartshop/feast/feature_repo
  feast apply
  feast materialize-incremental \"$(date -u +%Y-%m-%dT%H:%M:%S)\" 2>&1
"
```

What happens internally:
1. `SparkComputeEngine` reads `batch_engine` config from `feature_store.yaml` (injected by `feast-spark-engine` ConfigMap)
2. SparkSession starts (`local[*]` — runs in feast pod JVM, `JAVA_TOOL_OPTIONS=-Dcom.redhat.fips=false` baked into image)
3. `SELECT * FROM s3a://smartshop-raw/processed/reviews/ WHERE event_timestamp BETWEEN ...`
4. BFV UDF: `groupBy(user_id).agg(avg_rating, review_count, ...)` + `groupBy(item_id).agg(...)` → transformed DataFrame
5. `mapInArrow` serializes + writes each entity row to Redis online store (`partitions: 10` to avoid OOMKill)
6. Redis DBSIZE increases to ~3.1M keys

Verify Redis after materialization:
```bash
oc exec -n smartshop deploy/redis -c redis -- \
  redis-cli -a "${REDIS_PASSWORD}" --no-auth-warning DBSIZE
# (integer) 3143XXX  ← ~3.1M user + item feature keys

# Spot-check a user feature
oc exec -n smartshop "$FEAST_POD" -c online -- python3 -c "
from feast import FeatureStore
fs = FeatureStore(repo_path='/feast-data/smartshop/feast/feature_repo')
result = fs.get_online_features(
    features=['user_features:user_avg_rating', 'user_features:user_review_count'],
    entity_rows=[{'user_id': 'AHO2HFBROLSL5BHJ2IOMLHHFFHWQ'}]
).to_dict()
print(result)
# → {'user_avg_rating': [1.89], 'user_review_count': [9]}
"
```

#### Performance benchmarks (full 140M-row dataset)

| Stage | Time | Notes |
|-------|------|-------|
| `preprocess_reviews_full.py` | ~22 min | One-time, 3 categories → single flat partition |
| `feast materialize-incremental` | ~2 min | SparkComputeEngine `local[*]`, `partitions: 10` |
| Redis write throughput | ~26K keys/sec | `mapInArrow`, 10 pipeline partitions |
| Total features materialized | ~3.1M keys | `user_features` + `item_features` entities |

Full benchmark details and MLflow logging: `feast/feature_repo/benchmark_materialize.py`.

### 8g — Training Integration (SparkOfflineStore in rec-trainer)

The rec-trainer image includes `feast[grpc]==0.62.0 + pyspark`. When `FEAST_REPO_PATH` is set in the TrainJob env, `train.py` uses `feast.get_historical_features()` instead of direct `pd.read_parquet`. A `--max-rows` argument is available for partial data iteration:

```python
# train.py — rank-0 only, result saved to /tmp, all ranks load from /tmp
store = FeatureStore(repo_path=feast_repo_path)
entity_df = interactions[["user_id", "item_id", "event_timestamp"]]
entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"], unit="ms", utc=True)
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["user_features:user_avg_rating", ..., "item_features:item_price"],
).to_df()
```

The training pod connects to the feast **registry server** (not the file PVC) via remote gRPC:
- `FEAST_REGISTRY_TYPE=remote`
- `FEAST_REGISTRY_PATH=feast-smartshop-feast-registry.smartshop.svc.cluster.local:6570`

The training pod's SparkOfflineStore reads Parquet from MinIO directly (same s3a:// paths, same hadoop-aws JARs). It does **not** go through the feast server — only the registry lookup is remote.

### 8h — Enable Feature Store in RHOAI Dashboard

The Feature Store tab in the RHOAI dashboard is gated behind a feature flag that defaults to **disabled**. Patch it once per cluster:

```bash
oc patch odhdashboardconfig odh-dashboard-config \
  -n redhat-ods-applications \
  --type=merge \
  -p '{"spec":{"dashboardConfig":{"disableFeatureStore":false}}}'

# Restart dashboard to pick up the config change
oc rollout restart deployment/rhods-dashboard -n redhat-ods-applications
oc rollout status deployment/rhods-dashboard -n redhat-ods-applications
```

The FeatureStore CR also needs the label `feature-store-ui: enabled` — this is already present in `infrastructure/openshift/feast-operator.yaml`. Verify:

```bash
oc get featurestore smartshop-feast -n smartshop \
  -o jsonpath='{.metadata.labels.feature-store-ui}'
# enabled
```

After the dashboard restarts, the **Feature Store** section appears in the RHOAI sidebar under the `smartshop` project.

### 8i — Run `feast apply` (register schema)

`feast apply` registers the feature schema and metadata into the Feast registry. It does **not** move any data — it only tells Feast what features exist, where they live, and which entities own them.

> **Must run in the `offline` container** — the `registry` container has no access to S3 credentials or pyspark. The feature repo path includes `feast/` (not just the top-level PVC mount).

```bash
source .env

FEAST_POD=$(oc get pod -n smartshop -l feast.dev/name=smartshop-feast \
  -o jsonpath='{.items[0].metadata.name}')

oc exec -n smartshop "$FEAST_POD" -c offline -- bash -c "
  export AWS_ACCESS_KEY_ID=${MINIO_ACCESS_KEY:-minio}
  export AWS_SECRET_ACCESS_KEY=${MINIO_SECRET_KEY:-minio123}
  cd /feast-data/smartshop/feast/feature_repo
  feast apply 2>&1
"

# Expected output:
# Applying changes for project smartshop
# Deploying infrastructure for user_features
# Deploying infrastructure for item_features
# Deploying infrastructure for review_embeddings
```

> **Prerequisite — placeholder Parquet files:** `feast apply` reads the schema
> from the S3 data sources even when `schema=` is explicitly declared (Feast 0.60
> always calls `get_table_column_names_and_types_from_data_source` to populate
> entity columns). Before the Spark ETL has run, the buckets are empty and PyArrow
> returns `ACCESS_DENIED` (MinIO returns HTTP 403 for `HeadObject` on missing keys,
> not 404). Run this once to write empty schema-carrying Parquet files:
>
> ```bash
> oc exec -n smartshop $FEAST_POD -c offline -- python3 << 'EOF'
> import os, pyarrow as pa, pyarrow.parquet as pq, pyarrow.fs as pafs
> from datetime import datetime, timezone
>
> endpoint = os.environ["AWS_ENDPOINT_URL_S3"].replace("http://", "")
> s3 = pafs.S3FileSystem(
>     access_key=os.environ["AWS_ACCESS_KEY_ID"],
>     secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
>     endpoint_override=endpoint, scheme="http", force_virtual_addressing=False,
> )
> ts = pa.timestamp("us", tz="UTC")
> placeholders = {
>     "smartshop-features/user_features/_placeholder.parquet": pa.schema([
>         pa.field("user_id", pa.string()), pa.field("event_timestamp", ts),
>         pa.field("user_avg_rating", pa.float64()), pa.field("user_review_count", pa.int64()),
>         pa.field("user_unique_items", pa.int64()), pa.field("user_avg_review_length", pa.float64()),
>         pa.field("user_category_count", pa.int64()), pa.field("user_tenure_days", pa.int64()),
>     ]),
>     "smartshop-features/item_features/_placeholder.parquet": pa.schema([
>         pa.field("item_id", pa.string()), pa.field("event_timestamp", ts),
>         pa.field("item_avg_rating", pa.float64()), pa.field("item_rating_stddev", pa.float64()),
>         pa.field("item_review_count", pa.int64()), pa.field("item_total_helpful_votes", pa.int64()),
>         pa.field("item_avg_review_length", pa.float64()), pa.field("item_price", pa.float32()),
>     ]),
>     "smartshop-embeddings/review_embeddings/_placeholder.parquet": pa.schema([
>         pa.field("review_id", pa.string()), pa.field("event_timestamp", ts),
>         pa.field("item_id", pa.string()), pa.field("user_id", pa.string()),
>         pa.field("rating", pa.float64()), pa.field("review_title", pa.string()),
>         pa.field("embed_text", pa.string()), pa.field("embedding", pa.list_(pa.float32())),
>     ]),
> }
> for path, schema in placeholders.items():
>     with s3.open_output_stream(path) as f:
>         pq.write_table(schema.empty_table(), f)
>     print("OK", path)
> EOF
> ```

---

### 8d — Feature View Definitions (`@batch_feature_view`)

Both feature views are declared in `feast/feature_repo/features.py` using the `@batch_feature_view` decorator with `TransformationMode.PYTHON`. The UDF receives the full raw reviews DataFrame and returns the aggregated features:

**`user_features`** — aggregated per `user_id`:

| Feature | Type | Description |
|---------|------|-------------|
| `user_avg_rating` | Float64 | Mean star rating across all reviews |
| `user_review_count` | Int64 | Total reviews written |
| `user_unique_items` | Int64 | Distinct items reviewed |
| `user_avg_review_length` | Float64 | Mean character length of review text |
| `user_category_count` | Int64 | Distinct product categories reviewed |
| `user_tenure_days` | Int64 | Days from first to last review |

**`item_features`** — aggregated per `item_id` (from `parent_asin`):

| Feature | Type | Description |
|---------|------|-------------|
| `item_avg_rating` | Float64 | Mean star rating |
| `item_rating_stddev` | Float64 | Rating standard deviation (controversy signal) |
| `item_review_count` | Int64 | Total reviews received |
| `item_total_helpful_votes` | Int64 | Sum of helpful votes across all reviews |
| `item_avg_review_length` | Float64 | Mean review length |
| `item_price` | Float32 | `null` — metadata join not in BFV scope |

Both views: `TTL=3650d`, `online=True`, `offline=False`.

> **Why `offline=False`?** Feast's `SparkWriteNode` uses `offline=True` to append the transformed DataFrame **back into `batch_source.path`** — which is `processed/reviews/`. Leaving it `True` would corrupt the raw source on every materialization run. `offline=False` skips the write-back while still serving `get_historical_features()` (reads apply the UDF on-demand from source).

### 8e — Full Data Pipeline (BFV approach)

`feast apply` ✅ only registers schema. Real data flows through these stages:

```
[✅ Step 0]  deploy_infra complete — all pods Running

[Step 1]  Download Amazon Reviews 2023 (~49GB)
             make data-full   (or data-sample for a 5% subset)
             Raw Parquet → s3://smartshop-raw/raw/{Electronics,Books,Home_and_Kitchen}/

[Step 2]  Preprocess reviews — ONE-TIME, adds event_timestamp column
             scripts/preprocess_reviews_full.py (copy to feast pod, see §8f below)
             Reads  → smartshop-raw/raw/reviews/{category}/  (BIGINT timestamp)
             Writes → smartshop-raw/processed/reviews/       (TIMESTAMP column added)

[Step 3]  feast apply + feast materialize-incremental (see §8f below)
             SparkComputeEngine reads processed/reviews/
             @batch_feature_view UDF: groupBy/agg → user_features + item_features
             Writes → Redis (~3.1M keys, ~2 min on full dataset)
             NO smartshop-features/ Parquet needed for materialization

[Step 4]  Spark jobs — text preprocessing + embeddings (separate pipeline)
             spark-application-text-preprocessing.yaml → smartshop-features/llm_data/
             spark-application-embedding.yaml          → smartshop-embeddings/

[Step 5]  Training reads features via SparkOfflineStore
             get_historical_features() → BFV UDF on-demand from processed/reviews/
             Saves model → smartshop-models/

[Step 6]  KServe serving calls get_online_features() → Redis at inference time
```

> **What BFV replaces:** `spark-application-rapids.yaml` and `spark-application-cpu-baseline.yaml` are still run as SparkApps for benchmarking/comparison (they write to `smartshop-features/`), but materialization no longer depends on their output — the BFV reads raw reviews directly.

### 8j — Verify Feature Store in RHOAI Dashboard

The RHOAI Dashboard shows the registered feature views, entities, and lineage graph:

**Feature views list** — 3 views registered, all Online-enabled:

![Feast feature views list in RHOAI dashboard](./assets/feast-feature-views-list.png)

**Lineage graph** — data sources → entities → feature views:

![Feast lineage graph showing data source to feature view relationships](./assets/feast-lineage-post-apply.png)

**Lineage graph (RHOAI Feature Store full view)** — entities flowing into `@batch_feature_view` nodes:

![RHOAI Feature Store — lineage: entities to @batch_feature_view nodes (dark theme)](./assets/rhoai-feast-lineage-full.png)

**Data sources** — registered Spark data sources visible in RHOAI dashboard:

![RHOAI Feature Store — data sources: raw_reviews_source and review_embeddings_source](./assets/rhoai-feast-data-sources.png)

**Features** — 12 features across `item_features` and `user_features` views:

![RHOAI Feature Store — features list: item_avg_rating, item_price, user_avg_rating, user_review_count, …](./assets/rhoai-feast-features-list.png)

> **Note on `__dummy` entity in lineage:** The lineage graph shows an internal
> Feast `__no_join_key` placeholder rendered as `Entity: __dummy`. This is a
> RHOAI Dashboard UI rendering artifact — `feast entities list` returns only
> the 3 correct entities. No functional impact.

---

