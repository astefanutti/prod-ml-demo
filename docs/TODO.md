# SmartShop AI — Demo Execution TODO

**Cluster:** `${OC_CLUSTER_DOMAIN}`
**Namespace:** `smartshop`
**Registry:** `${REGISTRY}`
**Branch:** `refine-cluster-infra-setup`

> Legend: ✅ Done · 🔄 In progress · ⏳ Blocked (dependency) · 🔲 Not started · ❌ Failed/needs fix

---

## Already completed

| Status | What |
|---|---|
| ✅ | Namespace, RBAC, secrets, MinIO buckets, Redis, PostgreSQL, Milvus created |
| ✅ | BuildConfigs updated to push to `${REGISTRY}` |
| ✅ | All images built and pushed: `spark-jobs`, `spark-jobs-rapids`, `rec-trainer`, `llm-trainer` |
| ✅ | Feast operator deployed — `smartshop-feast` pod Ready |
| ✅ | `spark-application.yaml`, `spark-application-rapids.yaml`, `spark-application-cpu-baseline.yaml` |
| ✅ | `data-download-job.yaml` — on-cluster HF → MinIO streaming Job |
| ✅ | Full observability stack: MLflow helper, PrometheusServlet, DCGM, redis_exporter, Grafana, collect script |
| ✅ | `notebooks/metrics_analysis.ipynb` — publication charts |
| ✅ | Gradio UI redesign — Platform Metrics tab + Architecture tab |
| ✅ | `docs/OBSERVABILITY.md`, `SETUP.md` up to date |
| ✅ | Git history cleaned (single-line commits, `Signed-off-by`) |

---

## Phase 1 — Observability

> **Goal:** metrics capture everything from the first Spark job onwards.
> **Blocker for:** Phase 3 metrics collection (DCGM data won't exist if deployed late).

| Status | # | Step | Command | Done when |
|---|---|---|---|---|
| ✅ | 1.1 | Enable user-workload monitoring (cluster-admin, once) | `oc apply -f infrastructure/openshift/user-workload-monitoring.yaml` | `cluster-monitoring-config` ConfigMap created |
| ✅ | 1.2 | Apply Spark metrics ConfigMap | `envsubst < infrastructure/openshift/spark-metrics-configmap.yaml \| oc apply -f -` | `spark-metrics-config` CM created in smartshop |
| ✅ | 1.3 | Deploy `redis_exporter` + ServiceMonitor | `envsubst < infrastructure/openshift/redis-exporter.yaml \| oc apply -f -` | Pod `redis-exporter-6c58cd8fbf-zlfkp` Running, 25 metrics exposed |
| ✅ | 1.4 | Deploy Grafana | `envsubst < infrastructure/openshift/grafana.yaml \| oc apply -f -` | Pod `grafana-7c755b8f6f-4fkj9` Running — https://grafana-smartshop.${OC_CLUSTER_DOMAIN} |
| ✅ | 1.5 | Get Prometheus token → `.env` | `oc create token grafana-sa -n smartshop --duration=8760h` | Token saved to `PROMETHEUS_TOKEN` in `.env` |
| 🔲 | 1.6 | Verify Grafana loads GPU metrics | Open `https://grafana-smartshop.${OC_CLUSTER_DOMAIN}` → GPU dashboard | DCGM panels show data |

**Verification:**
```bash
# redis_exporter scraping Redis
oc port-forward -n smartshop svc/redis-exporter 9121:9121
curl http://localhost:9121/metrics | grep redis_commands

# Grafana route
oc get route grafana -n smartshop -o jsonpath='{.spec.host}'
```

---

## Phase 2 — Data Ingestion

> **Goal:** `s3://smartshop-raw/raw/reviews/{Category}.parquet` and `raw/metadata/{Category}_meta.parquet` exist in MinIO.
> **Blocker for:** All Spark ETL jobs, Feast, training.

| Status | # | Step | Command | Done when |
|---|---|---|---|---|
| 🔲 | 2.1 | Upload download script to ConfigMap | `oc create configmap smartshop-download-script --from-file=download_to_minio.py=<(python3 -c "import json; cm=open('infrastructure/openshift/data-download-job.yaml').read(); ...") -n smartshop` | See note below |
| ✅ | 2.2 | Submit data download Job | Fixed script (no `datasets` lib needed), resubmitted | Job `smartshop-data-download` completed successfully |
| ✅ | 2.3 | Tail download logs | `oc logs -n smartshop job/smartshop-data-download` | `=== Download complete ===` — 333K reviews/category streamed at ~40K rows/s |
| ✅ | 2.4 | Verify reviews in MinIO | `aws s3 ls s3://smartshop-raw/raw/reviews/` | `Books.parquet` (125MB), `Electronics.parquet` (81MB), `Home_and_Kitchen.parquet` (76MB) |
| ⚠️ | 2.5 | Verify metadata in MinIO | `aws s3 ls s3://smartshop-raw/raw/metadata/` | Only `Electronics_meta.parquet` (66MB) — Books and Home_and_Kitchen have no metadata shards in HF repo |

> **Note on 2.1:** The download script is embedded in the ConfigMap inside `data-download-job.yaml`.
> The easiest apply path is:
> ```bash
> # Extract the script from the YAML and create the ConfigMap separately
> source .env
> python3 - <<'EOF'
> import yaml, subprocess
> docs = list(yaml.safe_load_all(open('infrastructure/openshift/data-download-job.yaml').read()))
> cm = next(d for d in docs if d['metadata']['name'] == 'smartshop-download-script')
> script = cm['data']['download_to_minio.py']
> with open('/tmp/download_to_minio.py', 'w') as f:
>     f.write(script)
> print("Extracted to /tmp/download_to_minio.py")
> EOF
> oc create configmap smartshop-download-script \
>   --from-file=download_to_minio.py=/tmp/download_to_minio.py \
>   -n smartshop --dry-run=client -o yaml | oc apply -f -
> ```

**Expected download times (sample mode, cluster bandwidth):**
- Electronics: ~5 min (22 GB JSONL streamed, 333K rows kept)
- Books: ~8 min (largest category)
- Home_and_Kitchen: ~3 min
- Total: ~15–20 min

---

## Phase 3 — Spark ETL + RAPIDS A/B Proof

> **Goal:** Feature Parquet files in `s3://smartshop-features/`, plus a measured GPU speedup number.
> **Requires:** Phase 2 complete (data in MinIO).
> **Blocker for:** Feast materialization, model training.

### 3a — CPU baseline (run first to establish the benchmark)

| Status | # | Step | Command |
|---|---|---|---|
| 🔲 | 3.1 | Apply CPU baseline SparkApp | `source .env && envsubst < infrastructure/openshift/spark-application-cpu-baseline.yaml \| oc apply -f -` |
| 🔲 | 3.2 | Watch until COMPLETED | `oc get sparkapplication smartshop-feature-engineering-cpu-baseline -n smartshop -w` |
| 🔲 | 3.3 | Capture `[METRIC]` lines | `oc logs -n smartshop $(oc get pod -n smartshop -l spark-app-name=smartshop-feature-engineering-cpu-baseline,spark-role=driver -o name) \| grep METRIC` |
| 🔲 | 3.4 | Collect full metrics bundle | `RUN_TYPE=cpu APP_NAME=smartshop-feature-engineering-cpu-baseline bash scripts/collect-run-metrics.sh` |

### 3b — RAPIDS GPU (the headline demo run)

| Status | # | Step | Command |
|---|---|---|---|
| 🔲 | 3.5 | Apply RAPIDS SparkApp | `source .env && envsubst < infrastructure/openshift/spark-application-rapids.yaml \| oc apply -f -` |
| 🔲 | 3.6 | Watch Spark UI during run (GPU operators visible) | `oc port-forward -n smartshop <driver-pod> 4040:4040` → http://localhost:4040/SQL |
| 🔲 | 3.7 | Capture `[METRIC]` lines | grep for `[METRIC] gpu_accelerated=True` and `total_elapsed_s` |
| 🔲 | 3.8 | Collect full metrics bundle (speedup auto-computed vs CPU run) | `RUN_TYPE=rapids APP_NAME=smartshop-feature-engineering-rapids bash scripts/collect-run-metrics.sh` |

### 3c — Text preprocessing + embeddings (CPU path, needed for RAG)

| Status | # | Step | Command |
|---|---|---|---|
| 🔲 | 3.9 | Apply full pipeline SparkApp (text + embeddings) | `source .env && envsubst < infrastructure/openshift/spark-application.yaml \| oc apply -f -` |
| 🔲 | 3.10 | Wait for all 3 SparkApps COMPLETED | `oc get sparkapplication -n smartshop` |
| 🔲 | 3.11 | Verify feature files in MinIO | `aws s3 ls s3://smartshop-features/ --recursive --endpoint-url ...` |

**Expected Spark runtimes (sample dataset ~1M reviews):**
- CPU feature engineering: ~10–20 min
- RAPIDS feature engineering: estimated ~2–5 min (depends on GPU speedup)
- Text preprocessing: ~5–10 min
- Embedding generation: ~15–30 min (sentence-transformers on CPU/GPU)

---

## Phase 4 — Feast Materialization (SparkComputeEngine → Redis)

> **Goal:** Redis has materialized user and item features; `redis_db_keys > 0`.  
> **Architecture:** `feast-spark-server` pod runs SparkComputeEngine `local[*]` → reads `SparkSource` Parquet from MinIO → writes to Redis.  
> **New in this phase:** `FileSource` replaced with `SparkSource` (`s3a://`); custom image with `pyspark==4.0.0`.  
> **See:** `docs/FEAST-SPARK.md` for full architecture. `docs/SETUP.md` §8 for step-by-step.  
> **Blocker for:** Phase 5 online serving lookup. Phase 5 training (Feast path).

### 4a — Build feast-spark-server image ✅ (files created)

> `build/Containerfile.feast-spark` extends `feature-server:0.62.0` with `pyspark==4.0.0` + hadoop-aws JARs.  
> BuildConfig + ImageStream already added to `infrastructure/openshift/`.

| Status | # | Step | Command | Done when |
|---|---|---|---|---|
| ✅ | 4.1 | `build/Containerfile.feast-spark` created | — | File in repo |
| ✅ | 4.2 | ImageStream `feast-spark-server` added to `imagestreams.yaml` | — | File updated |
| ✅ | 4.3 | BuildConfig `feast-spark-server` added to `buildconfigs.yaml` | — | File updated |
| 🔲 | 4.4 | Apply ImageStream + BuildConfig | `source .env && envsubst < infrastructure/openshift/imagestreams.yaml \| oc apply -f - && envsubst < infrastructure/openshift/buildconfigs.yaml \| oc apply -f -` | Resources created |
| 🔲 | 4.5 | Start build (~5-10 min) | `oc start-build feast-spark-server -n smartshop --follow` | Build `Complete` |

### 4b — Feature repo updated for SparkSource ✅

| Status | # | Step | Done when |
|---|---|---|---|
| ✅ | 4.6 | `feast/feature_repo/features.py`: `FileSource` → `SparkSource`, `s3://` → `s3a://` | `from feast.infra.offline_stores.contrib.spark_offline_store.spark_source import SparkSource` |
| ✅ | 4.7 | `feast/feature_repo/feature_store.yaml`: `offline_store.type: spark` + full spark_conf + env-var registry | `type: spark`, `FEAST_REGISTRY_TYPE` env var |

### 4c — Apply Feast Kubernetes config

| Status | # | Step | Command | Done when |
|---|---|---|---|---|
| ✅ | 4.8 | `feast-spark-engine.yaml` (ConfigMap + Secret) created | — | File in repo |
| ✅ | 4.9 | `feast-operator.yaml` (FeatureStore CR with SparkComputeEngine) created | — | File in repo |
| 🔲 | 4.10 | Apply ConfigMap + Secret | `source .env && envsubst < infrastructure/openshift/feast-spark-engine.yaml \| oc apply -f -` | CM + Secret exist |
| 🔲 | 4.11 | Apply FeatureStore CR (triggers Deployment restart) | `source .env && envsubst < infrastructure/openshift/feast-operator.yaml \| oc apply -f -` | Feast pod restarts |
| 🔲 | 4.12 | Wait for pod ready + verify pyspark | `oc get pod -l feast.dev/name=smartshop-feast -n smartshop -w` then `oc exec ... -c offline -- python3 -c "import pyspark; print(pyspark.__version__)"` | `4.0.0` |

### 4d — Run feast apply + materialize-incremental

| Status | # | Step | Command | Done when |
|---|---|---|---|---|
| 🔲 | 4.13 | Clone repo + `feast apply` in pod | `oc exec $FEAST_POD -c offline -- bash -c "git clone ... /tmp/repo && cd /tmp/repo/feast/feature_repo && FEAST_REGISTRY_TYPE=file FEAST_REGISTRY_PATH=/feast-registry/registry.db feast apply"` | Feature views registered (SparkSource) |
| 🔲 | 4.14 | `feast materialize-incremental` (SparkComputeEngine) | `oc exec $FEAST_POD -c offline -- bash -c "cd /tmp/repo/feast/feature_repo && FEAST_REGISTRY_TYPE=file FEAST_REGISTRY_PATH=/feast-registry/registry.db feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)"` | Spark log + rows → Redis |
| 🔲 | 4.15 | Verify Redis keys | `oc exec -n smartshop deploy/redis -- redis-cli -a $REDIS_PASSWORD DBSIZE` | Count > 0 |
| 🔲 | 4.16 | Redis Grafana dashboard | Open Grafana → Redis dashboard | Keys visible |

---

## Phase 5 — Model Training (Feast SparkOfflineStore + DDP)

> **Goal:** Trained model artifacts in `s3://smartshop-models/`.  
> **Requires:** Phase 3 Parquet in MinIO (for both Feast + direct paths); Phase 4 Redis keys (for serving).  
> **New in this phase:** `train.py` uses `feast.get_historical_features()` via `SparkOfflineStore local[*]` for point-in-time correct training data. Direct `pd.read_parquet` is the `--no-feast` fallback.  
> **rec-trainer image:** now includes `pyspark==4.0.0 + feast==0.62.0` (from `build/requirements/training.txt`).

### 5a — Recommendation Model (PyTorch DDP 4× A100 + Feast SparkOfflineStore)

> Training flow: rank-0 calls `feast.get_historical_features(entity_df)` → SparkSession `local[*]` reads  
> `SparkSource` Parquet → point-in-time join → saves to `/tmp/training_data.parquet` → `dist.barrier()` →  
> all ranks load from `/tmp` → `RecommendationDataset` → DDP training.

| Status | # | Step | Command | Done when |
|---|---|---|---|---|
| ✅ | 5.1 | `train.py`: added `_load_features_via_feast()` + `--use-feast/--no-feast` flag | — | Code in repo |
| ✅ | 5.2 | `build/requirements/training.txt`: added `pyspark==4.0.0 feast==0.62.0` | — | File updated |
| 🔲 | 5.3 | Rebuild rec-trainer image | `oc start-build rec-trainer -n smartshop --follow` | Build `Complete` |
| 🔲 | 5.4 | Set `FEAST_REPO_PATH` + registry env in `.env` | `FEAST_REPO_PATH=/tmp/smartshop-repo/feast/feature_repo` + `FEAST_REGISTRY_TYPE=remote` + `FEAST_REGISTRY_PATH=feast-smartshop-feast-registry.smartshop.svc.cluster.local:6570` | `.env` updated |
| 🔲 | 5.5 | Apply TrainingRuntime + TrainJob (rec) | `source .env && envsubst < infrastructure/openshift/trainjobs.yaml \| oc apply -f - --field-manager=rec` | TrainJob created |
| 🔲 | 5.6 | Watch rank-0 Feast log | `oc logs -n smartshop <rec-train-worker-0-pod> -f \| grep "\[Feast\]"` | `[Feast] retrieved N rows in Xs` |
| 🔲 | 5.7 | Watch DDP training | `oc get trainjob smartshop-rec-train -n smartshop -w` | Status = Complete |
| 🔲 | 5.8 | MLflow loss curves | Open MLflow → `smartshop-rec-training` experiment | Loss converging, `feature_source=feast_spark` logged |
| 🔲 | 5.9 | Verify model in MinIO | `aws s3 ls s3://smartshop-models/recommendation/` | `best_model.pt` exists |

### 5b — LLM Fine-Tuning (Mistral-7B, FSDP + QLoRA, Slurm, 2 nodes × 4 A100)

| Status | # | Step | Command | Done when |
|---|---|---|---|---|
| 🔲 | 5.10 | Verify Slurm partition available | `oc exec -n smartshop <llm-trainer-pod> -- sinfo -p slinky` | Nodes idle or allocated |
| 🔲 | 5.11 | Apply LLM TrainJob | `source .env && envsubst < infrastructure/openshift/trainjobs.yaml \| oc apply -f -` | `oc get trainjob smartshop-llm-finetune -n smartshop` |
| 🔲 | 5.12 | Verify NCCL bandwidth (cross-node) | `oc logs -n smartshop <worker-pod> \| grep "busBw\|Avg bus"` | `> 100 GB/s` |
| 🔲 | 5.13 | Collect Slurm metrics bundle | `RUN_TYPE=slurm TRAINJOB_NAME=smartshop-llm-finetune bash scripts/collect-run-metrics.sh` | Bundle in MinIO |
| 🔲 | 5.14 | Verify adapter in MinIO | `aws s3 ls s3://smartshop-models/llm-adapter/` | Adapter weights exist |

---

## Phase 6 — Serving (KServe InferenceServices)

> **Goal:** All 3 InferenceServices Ready; endpoints returning valid responses.
> **Requires:** Phase 5 model artifacts in MinIO.

| Status | # | Step | Command | Done when |
|---|---|---|---|---|
| 🔲 | 6.1 | Apply all 3 InferenceServices | `source .env && envsubst < infrastructure/openshift/inferenceservices.yaml \| oc apply -f -` | Resources created |
| 🔲 | 6.2 | Wait for rec model Ready | `oc get isvc smartshop-rec -n smartshop -w` | `READY=True` |
| 🔲 | 6.3 | Wait for LLM model Ready (may take 5–10 min to load) | `oc get isvc smartshop-llm -n smartshop -w` | `READY=True` |
| 🔲 | 6.4 | Wait for RAG Ready | `oc get isvc smartshop-rag -n smartshop -w` | `READY=True` |
| 🔲 | 6.5 | Smoke test rec endpoint | `curl -X POST $RECOMMEND_URL -H 'Content-Type: application/json' -d '{"user_id":"test123","top_k":5}'` | JSON with `recommendations` array |
| 🔲 | 6.6 | Smoke test LLM endpoint | `curl -X POST $SUMMARIZE_URL -d '{"product_name":"Headphones","review_text":"Great sound quality"}'` | JSON with `summary` |
| 🔲 | 6.7 | Smoke test RAG endpoint | `curl -X POST $RAG_URL -d '{"question":"Is this good for gaming?","product_id":""}'` | JSON with `answer` |

---

## Phase 7 — Demo + Summit Proof Artifacts

> **Goal:** Live Gradio UI working end-to-end; all charts generated for Summit slides/blog.

### 7a — Gradio UI

| Status | # | Step | Command | Done when |
|---|---|---|---|---|
| 🔲 | 7.1 | Set `PROMETHEUS_TOKEN` in `.env` | `oc sa get-token grafana-sa -n smartshop` | Gradio Platform Metrics tab shows live data |
| 🔲 | 7.2 | Run Gradio locally or deploy as pod | `source .env && python demo/app.py` | http://localhost:7860 opens |
| 🔲 | 7.3 | Test recommendation flow end-to-end | Enter user ID → Get Recommendations → watch GPU spike | Response in <100ms |
| 🔲 | 7.4 | Screenshot Platform Metrics tab | GPU utilization spike during inference | Summit slide ready |

### 7b — Summit Charts (via analysis notebook)

| Status | # | Step | Output |
|---|---|---|---|
| 🔲 | 7.5 | Run `notebooks/metrics_analysis.ipynb` | `mlflow_gpu_vs_cpu.png` — wall-clock speedup bar chart |
| 🔲 | 7.6 | — | `dcgm_combined.png` — 6-panel GPU metrics |
| 🔲 | 7.7 | — | `redis_metrics.png` — ops/sec, hit ratio, memory |
| 🔲 | 7.8 | — | `rapids_coverage.png` — GPU operator coverage pie |
| 🔲 | 7.9 | — | `mlflow_stage_breakdown.png` — per-stage timing comparison |
| 🔲 | 7.10 | Bundle all charts as zip | `summit_charts.zip` uploaded to MinIO |

---

## Blocked / Deferred

| Item | Reason | When to revisit |
|---|---|---|
| Feast `SQLRegistry` (psycopg2) | Default feast image has no psycopg2; upgrade to feast-spark image also fixes this | Phase 4 (use feast-spark image) |
| Feast `SparkOfflineStore` k8s executor mode | `spark.master: k8s://...` requires executor pod RBAC + image with pyspark; `local[*]` used for Summit | Post-Summit scale-up |
| OTEL distributed tracing (Gradio→KServe→Feast→Redis) | Needs OTEL Collector + Tempo backend | Post-Summit infra upgrade |
| RAPIDS full dataset (571M reviews) | 49 GB download, ~4h ETL | Summit recording run (not demo) |
| Embedding generation on GPU | `sentence-transformers==2.7.0` (Python 3.8 compatible) on `spark-jobs-rapids` image | build `spark-jobs-rapids-6` in progress |

---

## Quick reference — one-liner to apply each phase

```bash
# Phase 1 — Observability
source .env
oc apply -f infrastructure/openshift/user-workload-monitoring.yaml
envsubst < infrastructure/openshift/spark-metrics-configmap.yaml | oc apply -f -
envsubst < infrastructure/openshift/redis-exporter.yaml | oc apply -f -
envsubst < infrastructure/openshift/grafana.yaml | oc apply -f -
PROMETHEUS_TOKEN=$(oc sa get-token grafana-sa -n smartshop)

# Phase 2 — Data
envsubst < infrastructure/openshift/data-download-job.yaml | oc apply -f -
oc logs -n smartshop -f job/smartshop-data-download

# Phase 3 — Spark ETL
envsubst < infrastructure/openshift/spark-application-cpu-baseline.yaml | oc apply -f -
# (wait for COMPLETED)
envsubst < infrastructure/openshift/spark-application-rapids.yaml | oc apply -f -
envsubst < infrastructure/openshift/spark-application.yaml | oc apply -f -

# Phase 4 — Feast
oc apply -f infrastructure/feast/feast-spark-rbac.yaml
FEAST_POD=$(oc get pod -n smartshop -l app=feast-smartshop-feast -o jsonpath='{.items[0].metadata.name}')
oc exec -n smartshop $FEAST_POD -c offline -- bash -c "cd /feast/feature_repo && feast apply"
oc exec -n smartshop $FEAST_POD -c offline -- bash -c "cd /feast/feature_repo && feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)"

# Phase 5 — Training
envsubst < infrastructure/openshift/trainjobs.yaml | oc apply -f -

# Phase 6 — Serving
envsubst < infrastructure/openshift/inferenceservices.yaml | oc apply -f -
oc get isvc -n smartshop -w

# Phase 7 — Demo
python demo/app.py
# Open notebooks/metrics_analysis.ipynb in JupyterLab
```

---

## Status update log

| Date | Update |
|---|---|
| 2026-04-08 | Phases 1–7 documented; cluster state: Feast Ready, Redis 0 keys, no data, no Spark jobs, no training, no serving |
| 2026-04-20 | Phase 1 complete ✅ — user-workload monitoring, spark-metrics-config, redis_exporter, Grafana all deployed. PROMETHEUS_TOKEN saved. |
| 2026-04-20 | Phase 2 complete ✅ (partial) — 1M reviews across 3 categories in MinIO. Only Electronics has metadata. Books/Home_and_Kitchen metadata missing from HF repo. Spark jobs will use metadata only for Electronics. |
| 2026-04-20 | Issues found: (1) `datasets` lib missing from spark-jobs image — fixed by using `requests` streaming instead. (2) `smartshop-credentials` secret had empty MINIO/AWS creds — patched directly with `oc patch`. Tracked in IMPROVEMENTS.md. |
| 2026-04-21 | Phase 3 partial ✅ — RAPIDS + CPU baseline SparkApps COMPLETED. `text-preprocessing` RUNNING. Build `spark-jobs-rapids-6` running (fix: sentence-transformers 2.7.0 for Python 3.8). |
| 2026-04-21 | Phase 4 redesigned — switching from dask `FileSource` → `SparkComputeEngine` + `SparkSource`. ODH feast repo (`opendatahub-io/feast` v0.62.0) investigated: `spark` is a valid `offlineStore.type` in the CRD; `spec.batchEngine.configMapRef` injects `type: spark.engine` into feature_store.yaml; requires custom image with `pyspark==4.0.0 feast[spark]`. See `docs/FEAST-SPARK.md` for full findings. |
