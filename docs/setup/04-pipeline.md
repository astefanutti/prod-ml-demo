## 9. Deploy MLflow (PostgreSQL backend)

MLflow is operator-managed in `redhat-ods-applications`. It requires a PostgreSQL database for its backend store.

### 9a — Deploy PostgreSQL in `smartshop`

```bash
# Secrets use ${VAR} markers — use envsubst
envsubst < infrastructure/mlflow/postgres.yaml | oc apply -f -
oc rollout status deployment/postgres -n smartshop
```

The manifest creates:
- `postgres` Deployment — OpenShift-native `postgresql:15-el9` image, 5Gi NFS PVC (`postgres-data`)
- `postgres` Service on port 5432
- `postgres-credentials` Secret populated from `PG_USER`, `PG_PASSWORD`, `PG_DATABASE` in `.env`

### 9b — Update MLflow Backend Secret

> **OVN-K DNS gotcha:** MLflow's `NetworkPolicy` in `redhat-ods-applications` has
> `policyTypes: [Ingress, Egress]` (operator-managed). DNS lookups for cross-namespace
> CNAME chains fail under OVN-K even though port 53 is listed in the egress rules.
> Use the postgres Service **ClusterIP** directly — TCP connectivity works fine.

After postgres is running, get its ClusterIP and add it to `.env`:

```bash
oc get svc postgres -n smartshop -o jsonpath='{.spec.clusterIP}'
# e.g. 172.30.26.132 — set PG_CLUSTERIP=<this value> in .env
```

Then re-run `make setup-secrets` to create/update `mlflow-postgres-secret` and `mlflow-s3-credentials` in `redhat-ods-applications` with the correct values from `.env`.

### 9c — Patch MLflow CR for S3 artifacts

Patch the MLflow CR to use S3 as artifact destination (see `infrastructure/mlflow/mlflow-cr.yaml`):

```bash
oc patch mlflow mlflow -n redhat-ods-applications --type=merge -p '{
  "spec": {
    "artifactsDestination": "s3://smartshop-models",
    "serveArtifacts": true,
    "envFrom": [{"secretRef": {"name": "mlflow-s3-credentials"}}]
  }
}'
```

### 9d — Verify MLflow is Running

```bash
oc rollout restart deployment/mlflow -n redhat-ods-applications
oc rollout status deployment/mlflow -n redhat-ods-applications

oc get pods -n redhat-ods-applications | grep mlflow
# mlflow-xxxxx   2/2   Running
```

> **Fixed:** MLflow was crash-looping because the backend secret pointed to
> `postgres.feast-trainer-demo.svc.cluster.local` — an old namespace that no longer exists.
> Steps 9a–9d above resolve this.

**MLflow UI:**

```bash
oc get route mlflow -n redhat-ods-applications -o jsonpath='{.spec.host}'
```

![MLflow dashboard running on RHOAI](./assets/mlflow-dashboard.png)

---

## 10. Container Images ✅

All images are built via OpenShift `BuildConfig` and pushed directly to `${REGISTRY}` using the `quay-push-secret` in the `smartshop` namespace.

| Image | `.env` variable | Containerfile | Used by |
|---|---|---|---|
| `smartshop-feast-spark-server` | `FEAST_SERVER_IMAGE` | `build/Containerfile.feast-spark` — `feature-server:0.62.0` + pyspark + Java 11 + hadoop-aws JARs | Feast pod (offline, online, registry, ui containers) |
| `smartshop-feast-spark-executor` | `FEAST_EXECUTOR_IMAGE` | `build/Containerfile.feast-spark-executor` — `apache/spark:3.5.3` + hadoop-aws JARs + `executor-entrypoint.sh` | Executor pods — CPU k8s:// runs (`feast-spark-engine-cpu.yaml`) |
| `smartshop-feast-spark-executor-rapids` | `FEAST_EXECUTOR_RAPIDS_IMAGE` | `build/Containerfile.feast-spark-executor-rapids` — executor + RAPIDS 26.02.2 JAR (cuda12) | Executor pods — RAPIDS k8s:// runs (`feast-spark-engine-rapids.yaml`) |
| `smartshop-spark-jobs` | `SPARK_JOBS_IMAGE` | `build/Containerfile.spark` — UBI9/Python 3.12, PySpark + Feast | SparkApplication ETL jobs |
| `smartshop-spark-jobs-rapids` | `SPARK_RAPIDS_IMAGE` | `build/Containerfile.spark-rapids` — `apache/spark:3.5.3` + RAPIDS 26.02.2 JAR (cuda12) | RAPIDS GPU SparkApplication jobs |
| `smartshop-rec-trainer` | `REC_TRAINER_IMAGE` | `build/Containerfile.rec-trainer` — UBI9/Python 3.12, PyTorch DDP | TrainJob — two-tower recommendation |
| `smartshop-llm-trainer` | `LLM_TRAINER_IMAGE` | `build/Containerfile.llm-trainer` — UBI9/Python 3.12, QLoRA + FSDP | TrainJob — Mistral-7B QLoRA fine-tuning on K8s |
| `smartshop-rec-server` | `REC_SERVER_IMAGE` | `build/Containerfile.serving` — UBI9/Python 3.12, FastAPI | KServe — recommendation endpoint |
| `vllm/vllm-openai:v0.6.4` | `VLLM_IMAGE` | External — **not built here**; pin in `.env` before upgrade | KServe — LLM + RAG InferenceService containers |

**Rebuild any image:**
```bash
# Create push secret once (credentials from ~/.config/containers/auth.json)
make setup-push-secret QUAY_USER=abdhumal QUAY_TOKEN=<token>

# Apply/update BuildConfigs
make setup-builds

# Rebuild all
make build-images

# Rebuild one
make build-spark        # or build-rec / build-llm / build-serving / build-spark-rapids
```

**`.env` image variables** — all images are set here; `envsubst` injects them into manifests:

```ini
# Feast images (built via BuildConfig)
FEAST_SERVER_IMAGE=quay.io/abdhumal/smartshop-feast-spark-server:latest
FEAST_EXECUTOR_IMAGE=quay.io/abdhumal/smartshop-feast-spark-executor:latest
FEAST_EXECUTOR_RAPIDS_IMAGE=quay.io/abdhumal/smartshop-feast-spark-executor-rapids:latest

# Spark ETL + training
SPARK_JOBS_IMAGE=quay.io/abdhumal/smartshop-spark-jobs:latest
SPARK_RAPIDS_IMAGE=quay.io/abdhumal/smartshop-spark-jobs-rapids:latest
REC_TRAINER_IMAGE=quay.io/abdhumal/smartshop-rec-trainer:latest
LLM_TRAINER_IMAGE=quay.io/abdhumal/smartshop-llm-trainer:latest

# Serving
REC_SERVER_IMAGE=quay.io/abdhumal/smartshop-rec-server:latest
VLLM_IMAGE=vllm/vllm-openai:v0.6.4   # external — pin before upgrading
```

---

## 11. Run Spark ETL (Feature Engineering)

> **Prerequisite:** Data must land in `s3://smartshop-raw/raw/reviews/` and `raw/metadata/` first.
> Images are already built ✅. Run the download Job below — it streams HF → MinIO directly,
> no local disk required.

### 11a — Download data on-cluster (recommended)

Instead of downloading locally and uploading, run a Kubernetes Job that streams from
HuggingFace directly into MinIO. Avoids local disk/bandwidth entirely.

```bash
# Submit the download Job (streams ~1M reviews/category in sample mode, ~500MB total)
/bin/bash -c 'set -a; source .env; set +a; envsubst < infrastructure/openshift/data-download-job.yaml | oc apply -f -'

# Tail logs
oc logs -n smartshop -f job/smartshop-data-download

# Verify data landed
source .env
S3=https://$(oc get route minio-s3 -n smartshop -o jsonpath='{.spec.host}')
AWS_ACCESS_KEY_ID="$MINIO_ACCESS_KEY" AWS_SECRET_ACCESS_KEY="$MINIO_SECRET_KEY" \
  aws s3 ls s3://smartshop-raw/raw/reviews/ --endpoint-url $S3 --no-verify-ssl
```

**Verify data landed in MinIO:**

![MinIO smartshop-raw bucket with reviews parquet files loaded](./assets/minio-raw-data-loaded.png)

To run full dataset (49GB) for the Summit recording, patch the Job's env before apply:

```bash
# Full mode — delete sample Job first, patch env, resubmit
oc delete job smartshop-data-download -n smartshop
# Edit DATA_DOWNLOAD_MODE: full in infrastructure/openshift/data-download-job.yaml
/bin/bash -c 'set -a; source .env; set +a; envsubst < infrastructure/openshift/data-download-job.yaml | oc apply -f -'
```

### 11b — RAPIDS GPU vs CPU baseline (A/B comparison for demo)

The driver logs emit structured `[METRIC]` lines for easy comparison. The canonical Summit story:
*"Same Python code, same manifest — swap one image and the RAPIDS plugin handles the rest."*

**Step 1 — CPU baseline:**

```bash
/bin/bash -c 'set -a; source .env; set +a; \
  envsubst < infrastructure/openshift/spark-application-cpu-baseline.yaml | oc apply -f -'

# Wait for completion, then capture metrics
CPU_DRIVER=$(oc get pod -n smartshop -l spark-app-name=smartshop-feature-engineering-cpu-baseline,spark-role=driver -o name)
oc logs -n smartshop $CPU_DRIVER | grep METRIC
# [METRIC] gpu_accelerated=False
# [METRIC] total_elapsed_s=<N> s
# [METRIC] throughput_rows_per_s=<N> rows/s
```

**Step 2 — RAPIDS GPU:**

```bash
/bin/bash -c 'set -a; source .env; set +a; \
  envsubst < infrastructure/openshift/spark-application-rapids.yaml | oc apply -f -'

GPU_DRIVER=$(oc get pod -n smartshop -l spark-app-name=smartshop-feature-engineering-rapids,spark-role=driver -o name)
oc logs -n smartshop $GPU_DRIVER | grep METRIC
# [METRIC] gpu_accelerated=True
# [METRIC] total_elapsed_s=<M> s   ← should be significantly lower
# [METRIC] throughput_rows_per_s=<M> rows/s
```

**GPU utilization (DCGM — already scraped by OCP monitoring):**

```bash
# Query from the OCP Observe → Metrics console, or:
oc exec -n openshift-monitoring prometheus-k8s-0 -- \
  curl -sg 'http://localhost:9090/api/v1/query?query=DCGM_FI_DEV_GPU_UTIL' | \
  python3 -c "import sys,json; [print(m['metric']['gpu'], m['value'][1]) for m in json.load(sys.stdin)['data']['result']]"
```

**Spark UI (shows GPU operators in the DAG):**

```bash
# Port-forward the Spark UI from the running driver pod
oc port-forward -n smartshop $GPU_DRIVER 4040:4040
# Open http://localhost:4040 → SQL/DataFrame tab → look for GpuHashAggregateExec
```

**CPU path (full pipeline — text preprocessing + embeddings):**

```bash
/bin/bash -c 'set -a; source .env; set +a; envsubst < infrastructure/openshift/spark-application.yaml | oc apply -f -'
```

**Monitor any job:**

```bash
oc get sparkapplication -n smartshop
oc logs -n smartshop -f $(oc get pod -n smartshop -l spark-role=driver -o name | head -1)
```

When running both CPU baseline and RAPIDS simultaneously, you'll see all three SparkApplications in the console:

![OpenShift console showing cpu-baseline, rapids, and text-preprocessing SparkApplications all RUNNING](./assets/openshift-3-sparkapps-running.png)

Executor pods spin up per job — each gets ~12 GB RAM and the RAPIDS driver gets a GPU:

![Executor pods for rapids and cpu-baseline running with 12 GB memory each](./assets/openshift-spark-executor-pods.png)

The RAPIDS driver pod appears as a separate running pod before executors are created:

![OpenShift — RAPIDS driver pod running, waiting for executors to register](./assets/openshift-rapids-driver-running.png)

Once the full stack (including Spark History Server) is deployed, the complete pod listing looks like this:

![OpenShift — full stack with Spark History Server pod added to smartshop namespace](./assets/openshift-pods-with-history-server.png)

**CPU executor startup logs** show Spark registering with the driver over `feast-spark-driver.smartshop.svc.cluster.local`:

![CPU executor log — startup, registering with driver via ClusterIP Service](./assets/cpu-run-executor-log-startup.png)

**CPU executor logs** mid-run show the distributed Parquet scan from MinIO:

![CPU executor log — Parquet scan from s3a://smartshop-raw/processed/reviews/](./assets/cpu-run-executor-log-parquet-scan.png)

**CPU run executor pods** — 4 executor pods (2 cores each, 14Gi RAM each):

![oc get pods — 4 CPU executor pods running during feast materialize](./assets/cpu-run-oc-executor-pods.png)

**RAPIDS executor startup logs** show RAPIDS memory pool initialization and task scheduling:

![RAPIDS executor log — RAPIDS memory pool (76 GB) and task concurrency](./assets/rapids-run-executor-log-gpu-memory-pool.png)

![RAPIDS executor log — RAPIDS SQL plugin active, GPU tasks scheduled](./assets/rapids-run-executor-log-rapids-tasks.png)

**RAPIDS run executor pods** — same 4 pods but with GPU resource limit (`nvidia.com/gpu: 1`):

![oc get pods — 4 RAPIDS executor pods running with GPU resource during feast materialize](./assets/rapids-run-oc-executor-pods.png)

When the job completes, the following MinIO buckets will be populated with Parquet files:
- `s3://smartshop-features/user_features/`
- `s3://smartshop-features/item_features/`
- `s3://smartshop-embeddings/review_embeddings/`

**Grafana — GPU utilization during the RAPIDS run:**

The SmartShop GPU Performance dashboard shows real GPU activity throughout the job lifecycle:

![Grafana — all 6 DCGM panels during the RAPIDS run (utilization, framebuffer, SM active, DRAM, NVLink, power)](./assets/grafana-gpu-all-metrics-during-run.png)

Key observations to highlight during the demo:
- **Framebuffer** holds ~75 GB continuously — the full dataset stays in GPU VRAM between stages
- **SM Active peaks** visible during user feature aggregation (~20:20) — the most compute-heavy stage
- **GPU power** 80–100 W (vs 400 W TDP) — workload is I/O-bound waiting on MinIO reads

GPU activity in the early run phase (Parquet scan from MinIO):

![Grafana — GPU activity early in the run, low utilization during I/O-bound Parquet scan](./assets/grafana-gpu-activity-early-run.png)

GPU utilization at 30% with 75 GB VRAM occupied — typical mid-run steady state:

![Grafana — RAPIDS GPU 30% utilization, 75 GB framebuffer in use](./assets/grafana-rapids-gpu-30pct-75gb-vram.png)

GPU memory mid-run spike when all raw data is loaded into VRAM:

![Grafana — framebuffer memory spike to 75 GB at mid-run](./assets/grafana-gpu-memory-midrun-spike.png)

![Grafana — framebuffer at 75 GB mid-run (zoomed)](./assets/grafana-rapids-midrun-gpu-memory-75gb.png)

GPU burst phase — SM active peak during aggregation (most compute-heavy stage):

![Grafana — RAPIDS GPU burst: SM active spike during groupBy aggregation](./assets/grafana-rapids-gpu-burst.png)

GPU power + executor memory + Redis keys growing together (mid-run overview):

![Grafana — GPU power, executor memory, Redis keys rising in lockstep mid-run](./assets/grafana-rapids-gpu-power-executor-redis.png)

GPU power + Redis keys at 15M (half of expected 26.5M) — approximately 50% through materialization:

![Grafana — GPU power stable, Redis keys at 15M (~50% through)](./assets/grafana-rapids-power-redis-15m-keys.png)

Full run arc — memory rises, job completes, GPU released:

![Grafana — complete run timeline showing memory rise then drop to 0 at job completion](./assets/grafana-gpu-complete-run-timeline.png)

RAPIDS full run timeline — GPU memory ramps up then drops cleanly to 0 after completion:

![Grafana — RAPIDS full run from start to finish, memory curve and final release](./assets/grafana-rapids-gpu-full-run.png)

GPU VRAM teardown — memory drops from 75 GB to 0 as executors are terminated:

![Grafana — RAPIDS VRAM teardown: framebuffer drops from 75 GB to 0 at job end](./assets/grafana-rapids-gpu-vram-teardown.png)

GPU memory fully released, executor pods terminated:

![Grafana — RAPIDS completed: GPU memory released, all executor pods gone](./assets/grafana-rapids-completed-gpu-memory-released.png)

**CPU run — Grafana executor + Redis metrics:**

Full Grafana window during the CPU run (executor CPU, memory, Redis keys across full duration):

![Grafana — full dashboard view during CPU run showing executor CPU/memory and Redis growth](./assets/grafana-full-dashboard-cpu-run.png)

CPU executor memory stable at ~14Gi, Redis at 14M keys:

![Grafana — CPU executor memory steady at 14Gi, Redis 14M keys](./assets/grafana-cpu-executor-mem-redis-14m-keys.png)

Full CPU executor monitoring window (executor restarts visible, memory stable):

![Grafana — CPU executor full monitoring window: memory, CPU, and restart events](./assets/grafana-cpu-executor-full-window.png)

CPU vs RAPIDS combined executor + Redis dashboard (side-by-side):

![Grafana — CPU and RAPIDS combined: executor CPU/memory + OOM events + Redis keys](./assets/grafana-executor-cpu-mem-oom-redis-both.png)

**Redis key growth progression** during materialization:

Redis at 5M keys (early stage, user_features writing):

![Grafana — Redis keys at 5M, early materialization phase](./assets/grafana-executor-metrics-redis-5m-keys.png)

Redis at 8M keys (mid user_features):

![Grafana — Redis keys at 8M](./assets/grafana-executor-metrics-redis-8m-keys.png)

Redis at 12M keys (item_features starting):

![Grafana — Redis keys at 12M, item_features phase](./assets/grafana-executor-metrics-redis-12m-keys.png)

Redis keys rising steeply — final write phase, approaching 26.5M:

![Grafana — Redis keys curve rising toward 26.5M target](./assets/grafana-executor-metrics-redis-rising.png)

**Spark History Server — completed runs:**

Both completed runs are visible in the History Server at `https://spark-history-smartshop.${OC_CLUSTER_DOMAIN}`:

![Spark History Server showing CPU baseline 2.0h and RAPIDS 1.3h runs side by side](./assets/spark-history-both-runs-completed.png)

RAPIDS run detail — completed in 1h 3m:

![Spark History Server — RAPIDS run completed in 1h 3m](./assets/spark-history-rapids-completed-1h3m.png)

All three SparkApplications (feature engineering RAPIDS, CPU baseline, text preprocessing) visible in History Server after completion:

![Spark History Server — all 3 apps completed](./assets/spark-history-all-apps-completed.png)

The RAPIDS run completes in 1h 3m vs 2h for CPU baseline — **1.48× wall-clock speedup**.

### Materialize features into Redis (online store)

`feast materialize-incremental` reads `processed/reviews/` from MinIO, runs the
`@batch_feature_view` UDFs (same groupBy/agg logic as the RAPIDS Spark job), and writes
features to Redis. The `wait-and-materialize.sh` script handles this automatically after
the SparkApps complete. To run manually:

```bash
FEAST_POD=$(oc get pod -n smartshop -l "feast.dev/name=smartshop-feast" \
  -o jsonpath='{.items[0].metadata.name}')

oc exec -n smartshop $FEAST_POD -c offline -- bash -c '
  export AWS_ACCESS_KEY_ID=minio
  export AWS_SECRET_ACCESS_KEY=minio123
  cd /feast-data/smartshop/feast/feature_repo
  feast materialize-incremental "'"$(date -u +%Y-%m-%dT%H:%M:%S)"'"
'
```

After materialization, verify features landed in Redis via RedisInsight and the Grafana Redis Feature Store dashboard:

**RedisInsight — key growth mid-run** (13M keys, materialization in progress):

![RedisInsight — 13M keys at mid-run, feast materialize in progress](./assets/redisinsight-13m-keys-midrun.png)

**RedisInsight — final key count** (26,493,202 keys — both user_features and item_features complete):

![RedisInsight — 26,493,202 keys final count after complete materialization](./assets/redisinsight-26m-keys-final.png)

**RedisInsight — Browse: item_features HASH keys** with field data visible:

![RedisInsight — Browse: 3.1M item_id HASH keys after Feast BFV materialization](./assets/redisinsight-browse-item-features.png)

**RedisInsight — key detail** showing HSET fields (feature names and serialized values):

![RedisInsight — single key detail: HSET fields for a user_features entity](./assets/redisinsight-browse-key-detail.png)

**RedisInsight — database analysis summary** (key type distribution, memory, top key patterns):

![RedisInsight — analysis summary: 26M HASH keys, ~6.2 GB, smartshop:* namespace patterns](./assets/redisinsight-analyze-summary-clean.png)

**RedisInsight — Slowlog** showing HSET burst pattern during materialization (expected):

![RedisInsight — slowlog showing HSET burst commands during feast materialize](./assets/redisinsight-slowlog-hset-burst.png)

![RedisInsight — slowlog commands list: HSET entries in sequence](./assets/redisinsight-slowlog-commands.png)

![RedisInsight — HSET slowlog entry detail with duration and key pattern](./assets/redisinsight-hset-slowlog.png)

**Grafana — Redis Feature Store dashboard** (post-materialization, serving traffic):

![Grafana — SmartShop Redis Feature Store: Commands/sec, 100% Cache Hit Ratio, 711 MB memory](./assets/grafana-redis-feature-store-working.png)

---

