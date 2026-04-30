## Full Demo Automation

> End-to-end deployment from a clean namespace. Use `scripts/apply-all.sh` or run phases manually.

### One-command deploy

```bash
# 1. Fill in .env
cp .env.example .env
# Edit .env — at minimum set: MINIO_ACCESS_KEY, MINIO_SECRET_KEY, REDIS_PASSWORD,
# HF_TOKEN, QUAY_USER, QUAY_TOKEN, PG_USER, PG_PASSWORD, PG_CLUSTERIP, OC_CLUSTER_DOMAIN

# 2. Login to cluster
oc login <cluster-api-url>

# 3. Deploy everything
set -a && source .env && set +a
bash scripts/apply-all.sh all
```

### Phase-by-phase deploy (recommended for first run)

```bash
set -a && source .env && set +a

# Phase 1: Namespace, secrets, RBAC, monitoring
bash scripts/apply-all.sh infra

# Phase 2: ImageStreams + BuildConfigs → wait for all 6 builds to Complete
bash scripts/apply-all.sh images
oc get builds -n smartshop -w

# Phase 3: Upload Spark JARs + download full dataset to MinIO
bash scripts/apply-all.sh data

# Phase 4: Grafana, redis-exporter, Spark metrics
bash scripts/apply-all.sh observability

# Phase 5: Spark ETL (RAPIDS, CPU baseline, text preprocessing)
bash scripts/apply-all.sh spark
oc get sparkapplication -n smartshop -w

# Phase 6: Feast operator → feast apply → materialize
bash scripts/apply-all.sh feast
bash scripts/wait-and-materialize.sh

# Phase 7: Training (Notebook CR runs rec + LLM training notebooks)
bash scripts/apply-all.sh training
# OR direct TrainJob submission:
envsubst < infrastructure/openshift/trainjobs.yaml | oc apply -f -
oc get trainjob -n smartshop -w

# Phase 8: Serving (ServingRuntimes + InferenceServices + ServiceMonitor)
bash scripts/apply-all.sh serving
oc get inferenceservice -n smartshop -w

# Phase 9: Notebook runner (optional — metrics analysis)
bash scripts/apply-all.sh notebook
```

### Manifest inventory

All manifests live in `infrastructure/openshift/`. Key files by phase:

| Phase | Manifests |
|-------|-----------|
| Infra | `user-workload-monitoring.yaml` |
| Images | `imagestreams.yaml`, `buildconfigs.yaml` |
| Data | `upload-spark-jars-job.yaml`, `data-download-job.yaml` |
| Observability | `grafana.yaml`, `redis-exporter.yaml`, `spark-metrics-configmap.yaml` |
| Spark | `spark-history-server.yaml`, `spark-application-rapids.yaml`, `spark-application-cpu-baseline.yaml`, `spark-application-text-preprocessing.yaml`, `spark-application-embedding.yaml` |
| Feast | `feast-spark-engine.yaml`, `feast-operator.yaml`, `feast-training-configmap.yaml`, `feast-spark-driver-svc.yaml` |
| Training | `trainjobs.yaml`, `notebook-rec-train.yaml`, `notebook-llm-train.yaml`, `e2e-notebook.yaml` |
| Serving | `serving-runtimes.yaml` *(consolidated — Secret, SA, 2 ServingRuntimes, 3 InferenceServices, ServiceMonitor)* |
| Monitoring | `serving-metrics-monitor.yaml` *(also included in serving-runtimes.yaml)* |

### Secrets (all created by `apply-all.sh infra` or `make setup-secrets`)

| Secret | Source | Used by |
|--------|--------|---------|
| `smartshop-credentials` | `.env` → MinIO + Redis + MLflow | Spark, Training, Feast |
| `hf-credentials` | `.env` → HF_TOKEN | LLM training, vLLM runtime |
| `quay-push-secret` | `.env` → QUAY_USER/TOKEN | BuildConfigs |
| `feast-redis-secret` | `.env` → Redis connection | Feast operator |
| `feast-s3-credentials` | `.env` → MinIO creds | Feast operator |
| `smartshop-mlflow-token` | `.env` → SA token | Spark, Training (MLflow tracking) |
| `smartshop-s3-serving` | `serving-runtimes.yaml` | KServe adapter download |

### Verified working state (Apr 30 2026)

| Component | Status | Notes |
|-----------|--------|-------|
| `smartshop-rec` ISVC | Ready | Two-Tower model, s3://smartshop-models/recommendation/ |
| `smartshop-llm` ISVC | Ready | vLLM v0.13.0 + Mistral-7B + LoRA from checkpoint-200 |
| `smartshop-rag` ISVC | Ready (scaled to 0) | Needs RAG server image build |
| Grafana dashboards | 4 dashboards | GPU, Redis, Spark, Inference Metrics |
| ServiceMonitors | 3 active | redis-exporter, smartshop-rec-metrics, smartshop-serving-metrics |
| Prometheus scraping | Working | Via `ocp-thanos` datasource (port 9091) |

### Known issues and fixes

| Issue | Root cause | Fix |
|-------|-----------|-----|
| vLLM `PermissionError: /.triton`, `/.config` | OpenShift non-root UID | emptyDir volumes + `TRITON_CACHE_DIR`/`XDG_CONFIG_HOME` env vars (fixed in `serving-runtimes.yaml`) |
| KServe storage initializer OOM for 15GB model | Default init container memory too low | RawDeployment mode — vLLM pulls base model from HF directly |
| RHOAI vLLM image pull failure on some nodes | `registry.redhat.io/rhaii/` needs auth | Switched to public `vllm/vllm-openai:v0.13.0` |
| Grafana "No data" on inference panels | Dashboard used `ocp-uwm` datasource (port 9092) | Changed all panels to `ocp-thanos` (port 9091) |
| LLM adapter path mismatch | Training saves to `llm-checkpoints/checkpoint-200/`, manifest had `llm-adapter/` | Fixed init container S3 prefix |
| Rec server `KeyError: num_users` | Old image had stale `server.py` | Rebuilt via BuildConfig + `imagePullPolicy: Always` |

### Teardown

```bash
# Delete all workloads (preserves PVCs and data)
oc delete inferenceservice --all -n smartshop
oc delete trainjob --all -n smartshop
oc delete sparkapplication --all -n smartshop
oc delete notebook --all -n smartshop
oc delete job --all -n smartshop

# Full namespace teardown (destroys everything including data)
oc delete namespace smartshop
```
