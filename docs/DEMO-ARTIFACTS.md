# SmartShop AI — Summit 2026 Demo Artifacts

Tracks every metric, chart, and screenshot needed for the Red Hat Summit 2026 demo.
Run `bash scripts/apply-all.sh notebook` after all Spark/training jobs complete to
auto-generate and upload most of these to `s3://smartshop-models/notebooks/`.

---

## Cluster Hardware

| Resource | Detail |
|----------|--------|
| GPU nodes | 2 × `gpu-node-{1,2}` |
| GPU model | NVIDIA A100-SXM4-80GB (8 per node = 16 total) |
| GPU Operator | `nvcr.io/nvidia/gpu-operator@sha256:634471cf…` |
| DCGM Exporter | `nvcr.io/nvidia/k8s/dcgm-exporter@sha256:7c0ac443…` |
| Spark Operator | `registry.redhat.io/rhoai/odh-spark-operator-rhel9@sha256:82fb7c45…` |
| MLflow | RHOAI-managed, `kubernetes-auth` mode — workspace `smartshop` |

---

## Dataset

### Amazon Reviews 2023 — Full Download

| Category | Size | Parquet files | Download completed |
|----------|------|---------------|--------------------|
| Books | 8.6 GiB | 59 | 2026-04-22 |
| Electronics | 8.4 GiB | 88 | 2026-04-22 |
| Home_and_Kitchen | 11.0 GiB | 135 | 2026-04-22 |
| **Total** | **28.0 GiB** | **282** | |

- Bucket: `s3://smartshop-raw/raw/reviews/{category}/`
- Row count (all categories): **140,772,341** rows
- Download job: `smartshop-data-download-full` — `Complete` (ran ~54 min)

---

## Feast BFV Materialization Benchmarks

### Spark Config — Three-point comparison

| Mode | ConfigMap | Executor image | `spark.master` |
|------|-----------|---------------|----------------|
| `local[*]` (single pod) | `feast-spark-engine.yaml` | _(none — runs in feast pod JVM)_ | `local[*]` |
| k8s:// CPU workers | `feast-spark-engine-cpu.yaml` | `smartshop-feast-spark-executor` | `k8s://https://${OCP_API_URL}` |
| k8s:// RAPIDS GPU | `feast-spark-engine-rapids.yaml` | `smartshop-feast-spark-executor-rapids` | `k8s://https://${OCP_API_URL}` |

### RAPIDS GPU executor config (feast-spark-engine-rapids.yaml)

```yaml
spark.master: "k8s://https://${OCP_API_URL}"
spark.plugins: "com.nvidia.spark.SQLPlugin"
spark.rapids.sql.enabled: "true"
spark.rapids.sql.concurrentGpuTasks: "2"
spark.executor.resource.gpu.amount: "1"
spark.executor.resource.gpu.vendor: "nvidia.com/gpu"
spark.executor.resource.gpu.discoveryScript: "/opt/spark/getGpusResources.sh"
spark.executor.instances: "4"
spark.executor.memory: "8g"
spark.kubernetes.container.image: "${REGISTRY}/smartshop-feast-spark-executor-rapids:latest"
spark.kubernetes.authenticate.driver.serviceAccountName: "spark"
spark.kubernetes.namespace: "smartshop"
```

### Results (to be filled after runs)

| Mode | Rows | Preprocessing time | Materialization time | Redis keys | Notes |
|------|------|--------------------|----------------------|------------|-------|
| `local[*]` CPU | 140.8M | ~22 min | ~2 min | ~3.1M | Baseline — already done |
| k8s:// 4×CPU workers | 140.8M | 🔲 pending | 🔲 pending | 🔲 | Build: `feast-spark-executor` |
| k8s:// 4×GPU RAPIDS | 140.8M | 🔲 pending | 🔲 pending | 🔲 | Build: `feast-spark-executor-rapids` |

---

## Performance Metrics

### Phase 3 — Spark ETL / Feature Engineering

Both jobs processed the full Amazon review dataset (Electronics + Books + Home & Kitchen).

> **Run date:** 2026-04-21 — full dataset, both jobs submitted simultaneously from a clean cluster state.

| Metric | CPU Baseline | RAPIDS GPU | Speedup |
|--------|-------------|------------|---------|
| Total rows processed | 140,772,341 ✅ | 140,772,341 ✅ | same |
| Read time (s) | 1,546.6 | **1,091.4** | **1.42×** |
| User feature aggregation (s) | 3,020.8 | **1,912.7** | **1.58×** |
| Item feature aggregation (s) | 1,366.7 | **786.4** | **1.74×** |
| Interaction join/write (s) | 1,088.2 | **844.3** | **1.29×** |
| **Total elapsed (s)** | **7,029.1** | **4,667.7** | **1.51× overall** |
| **Wall-clock (s)** | **7,057** | **4,755** | **1.48×** |
| Throughput (rows/s) | 20,026 | **30,159** | **+51%** |
| Unique users | 35,049,327 ✅ | 35,049,327 ✅ | same |
| Unique items | 9,790,339 ✅ | 9,790,339 ✅ | same |
| `gpu_accelerated` flag | `False` ✅ | `True` ✅ | |

> **Bottleneck analysis:** Both jobs are I/O-bound (MinIO S3 over NFS).
> RAPIDS advantage is columnar in-memory processing via CUDF — no JVM serialization overhead.
> SM Active Ratio < 2.5%, GPU power 80–100W (vs 400W TDP) throughout — GPU waiting on S3 reads.
> On a compute-bound workload or with local NVMe storage the speedup would be significantly higher.
> the cluster team flagged storage as the bottleneck — SSD nodes would improve both jobs.

**RAPIDS job completed:** `2026-04-21T15:25:39Z`  
**CPU job completed:** `2026-04-21T16:03:05Z`

## Key Numbers for Slides

```
SparkApp GPU vs CPU (same 140.8M row dataset — valid A/B comparison):
  RAPIDS speedup:        1.51× overall  (CPU 7,057s → RAPIDS 4,755s)
  Best stage speedup:    1.74× (item feature aggregation)
  Throughput gain:       +51%  (20,026 → 30,159 rows/s)
  Rows processed:        140.8M rows, 49 GB full dataset
  Unique users → Redis:  35M
  Unique items:          9.8M

Feast BFV pipeline (standalone — do not compare directly to SparkApp numbers):
  Elapsed:               138s (2.3 min)
  Source rows:           10,333,334
  Throughput:            74,878 rows/sec
  Redis keys written:    3,109,935 HASH keys
  Write throughput:      22,700 keys/sec
  Redis memory:          711 MB
  Steps:                 1  (raw parquet → Redis, no intermediate Parquet)

Architectural story (valid regardless of dataset size):
  SparkApp pipeline:     2 steps — transform to Parquet, then feast materialize
  Feast BFV pipeline:    1 step  — transform directly to Redis online store
  Eliminated:            Intermediate Parquet files in MinIO (~tens of GB)
```

```bash
# Reproduce metrics from driver logs:
oc logs smartshop-feature-engineering-rapids-driver -n smartshop | grep '\[METRIC\]'
oc logs smartshop-feature-engineering-cpu-baseline-driver -n smartshop | grep '\[METRIC\]'
```

#### RAPIDS image details

| Item | Value |
|------|-------|
| Image | `${REGISTRY}/smartshop-spark-jobs-rapids:latest` |
| Push digest | `sha256:4213f31c5ccef381422872ee409aebc77c9f16e75a398a8a514bb4e869d9ee19` |
| RAPIDS JAR | `rapids-4-spark_2.12-26.02.2-cuda13.jar` (Maven Central) |
| CUDA variant | `cuda13` — cluster runs CUDA 13.0 / driver 580.x |
| Base image | `apache/spark:3.5.3` |
| Built via | `BuildConfig/spark-jobs-rapids` in `smartshop` ns |

No code changes to `feature_engineering.py` — `spark.plugins=com.nvidia.spark.SQLPlugin`
intercepts all DataFrame ops at runtime.

---

### Phase 4 — Feast BFV Materialization (feature store pipeline)

> **Run date:** 2026-04-22 — full historical materialize, clean Redis slate, `local[*]` Spark inside feast pod.

| Metric | Value | Confidence |
|--------|-------|------------|
| Source rows read | 10,333,334 | ✅ Spark count verified |
| Total elapsed | **138s (2.3 min)** | ✅ wall-clock timed |
| Rows/sec throughput | **74,878 rows/sec** | ✅ computed |
| Redis keys written | **3,109,935** | ✅ verified RedisInsight |
| Redis write throughput | **22,700 keys/sec** | ✅ keys/elapsed |
| Redis memory used | **711 MB** | ✅ verified RedisInsight |
| Key type | 100% HASH | ✅ Analyze tab |
| Key TTL | None — persist indefinitely | ✅ flat expiry chart |
| Spark mode | `local[*]` inside feast pod | ✅ |
| Write burst window | 10:16:03–10:17:14 IST (~71s) | ✅ RedisInsight Slow Log |
| HSET latency | 44–75ms per write | ✅ Slow Log |

> **Slow log evidence:** HSET + HMGET burst confirmed by RedisInsight. Key pattern:
> `item_id_XXXXXXXXXX_smartshop`, fields: `_ts:item_features` + 6 aggregated feature fields.

#### Comparison with SparkApp pipeline

The SparkApp metrics (140.8M rows, 4,755s RAPIDS) are from a **different dataset size** —
the current `raw/reviews/` bucket holds 10.3M rows. A direct speedup ratio is not valid.

**Use the architectural argument instead:**

| Approach | Steps | Output | Dataset |
|---|---|---|---|
| SparkApp + RAPIDS | 1. SparkApp → Parquet files<br>2. `feast materialize` reads Parquet → Redis | 2-step pipeline | 140.8M rows (Apr 21) |
| Feast BFV | 1. `feast materialize` reads raw → Redis directly | **1-step pipeline** | 10.3M rows (Apr 22) |

**Key message:** Feast BFV eliminates the intermediate Parquet write entirely.
On 10.3M rows: **138 seconds, raw reviews → 3.1M feature vectors in Redis, no intermediate storage**.

```bash
# Reproduce:
oc exec -n smartshop $(oc get pod -n smartshop -l app=feast-smartshop-feast -o name | head -1) \
  -c offline -- bash -c '
  export AWS_ACCESS_KEY_ID=minio
  export AWS_SECRET_ACCESS_KEY=minio123
  cd /feast-data/smartshop/feast/feature_repo
  python3 benchmark_materialize.py --flush-redis --event-log 2>&1
'
```

---

### Phase 3 — Text Preprocessing (LLM instruction-tuning data)

> Job was RUNNING at snapshot time — final metrics pending completion.

| Metric | Value | Status |
|--------|-------|--------|
| Total input reviews | 140,772,341 | ✅ (from driver logs) |
| Clean reviews after dedup | 104,608,088 | ✅ |
| Read time (s) | 7.8 | ✅ |
| Train examples | _pending_ | job still running |
| Val examples | _pending_ | job still running |
| Total elapsed (s) | _pending_ | job still running |

```bash
oc logs smartshop-text-preprocessing-driver -n smartshop | grep '\[METRIC\]'
```

---

### Phase 5 — Recommendation Training (DDP, 4× A100)

| Metric | Value | Where |
|--------|-------|-------|
| Best val loss | _pending_ | MLflow `best_val_loss` |
| Val accuracy | _pending_ | MLflow `val_accuracy` |
| Epochs | 10 | `.env: REC_TRAIN_EPOCHS` |
| GPUs used | 4 × A100-80GB | `.env: REC_GPUS_PER_NODE=4` |
| Training time (s) | _pending_ | MLflow `total_training_time_s` |
| Throughput (samples/s) | _pending_ | MLflow `throughput_samples_per_s` |

---

### Phase 5 — LLM Fine-Tuning (FSDP, 2 nodes × 4 A100 = 8 GPUs)

| Metric | Value | Where |
|--------|-------|-------|
| Base model | `mistralai/Mistral-7B-Instruct-v0.3` | `.env` |
| Nodes × GPUs | 2 × 4 A100 = 8 total | `.env` |
| NCCL bus bandwidth | _pending_ | pod logs `busBw` |
| Train loss (final epoch) | _pending_ | MLflow `train_loss` |
| Adapter size (MB) | _pending_ | `s3://smartshop-models/llm-adapter/` |

---

## GitHub Manifest Links

All on branch [`refine-cluster-infra-setup`](https://github.com/abhijeet-dhumal/prod-ml-demo/tree/refine-cluster-infra-setup).

| File | Purpose | Link |
|------|---------|------|
| `spark-application-rapids.yaml` | RAPIDS SparkApplication | [link](https://github.com/abhijeet-dhumal/prod-ml-demo/blob/refine-cluster-infra-setup/infrastructure/openshift/spark-application-rapids.yaml) |
| `spark-application-cpu-baseline.yaml` | CPU baseline for A/B | [link](https://github.com/abhijeet-dhumal/prod-ml-demo/blob/refine-cluster-infra-setup/infrastructure/openshift/spark-application-cpu-baseline.yaml) |
| `spark-application-text-preprocessing.yaml` | LLM data prep | [link](https://github.com/abhijeet-dhumal/prod-ml-demo/blob/refine-cluster-infra-setup/infrastructure/openshift/spark-application-text-preprocessing.yaml) |
| `build/Containerfile.spark-rapids` | RAPIDS container image | [link](https://github.com/abhijeet-dhumal/prod-ml-demo/blob/refine-cluster-infra-setup/build/Containerfile.spark-rapids) |
| `spark/feature_engineering.py` | ETL script (CPU + GPU) | [link](https://github.com/abhijeet-dhumal/prod-ml-demo/blob/refine-cluster-infra-setup/spark/feature_engineering.py) |
| `spark/utils/mlflow_metrics.py` | MLflow logging helper | [link](https://github.com/abhijeet-dhumal/prod-ml-demo/blob/refine-cluster-infra-setup/spark/utils/mlflow_metrics.py) |
| `spark-metrics-configmap.yaml` | PrometheusServlet + PodMonitor | [link](https://github.com/abhijeet-dhumal/prod-ml-demo/blob/refine-cluster-infra-setup/infrastructure/openshift/spark-metrics-configmap.yaml) |
| `grafana.yaml` | Grafana + 3 dashboards | [link](https://github.com/abhijeet-dhumal/prod-ml-demo/blob/refine-cluster-infra-setup/infrastructure/openshift/grafana.yaml) |
| `trainjobs.yaml` | Kubeflow TrainJob (DDP + FSDP) | [link](https://github.com/abhijeet-dhumal/prod-ml-demo/blob/refine-cluster-infra-setup/infrastructure/openshift/trainjobs.yaml) |
| `scripts/apply-all.sh` | One-shot deploy script | [link](https://github.com/abhijeet-dhumal/prod-ml-demo/blob/refine-cluster-infra-setup/scripts/apply-all.sh) |

---

## Screenshots to Capture

### Grafana Dashboards

| Screenshot | File | What it shows | Status |
|-----------|------|--------------|--------|
| RAPIDS run — all 6 GPU metrics | `grafana-gpu-all-metrics-during-run.png` | Full dashboard: util, framebuffer, SM active, DRAM, NVLink, power | ✅ captured |
| RAPIDS run — early GPU activity (19:20–20:15) | `grafana-gpu-activity-early-run.png` | SM active spikes during feature aggregation phase | ✅ captured |
| RAPIDS job mid-run — GPU memory spike | `grafana-gpu-memory-midrun-spike.png` | Framebuffer at 75 GB at ~20:50, power 80–100W | ✅ captured |
| RAPIDS job mid-run — 75 GB loaded (PNG ref) | `grafana-rapids-midrun-gpu-memory-75gb.png` | Framebuffer at 75 GB, SM active, power at 80–100W during I/O-bound ETL | ✅ captured |
| RAPIDS job — complete run full timeline | `grafana-gpu-complete-run-timeline.png` | Full arc: memory spike → job completes → memory released (best overview) | ✅ captured |
| RAPIDS job completed — memory released to 0 | `grafana-rapids-completed-gpu-memory-released.png` | Memory drops to 0 at 20:55 (job completion), DRAM spike on final flush | ✅ captured |
| NVLink Bandwidth during TrainJob | — | Inter-GPU comms during DDP/FSDP training | 🔲 pending |
| Redis ops/s during Feast materialize | — | Write throughput as features land in online store | 🔲 pending |

**Grafana dashboard:** `https://grafana-smartshop.${OC_CLUSTER_DOMAIN}`  
**Dashboard name:** SmartShop GPU Performance (RAPIDS vs CPU)

**Key insight from DCGM data (captured in screenshots):**
- SM Active Ratio peak: ~2.5% — workload is **I/O-bound**, not compute-bound
- Framebuffer sustained at ~75 GB — full dataset held in GPU VRAM throughout
- Power: 80–100W (vs 400W TDP) — GPU waiting on MinIO S3 reads between stages
- NVLink near-zero — expected, single-node ETL (no inter-GPU comms needed)
- Large SM spike at ~20:20 IST = user feature aggregation stage (1,912s — heaviest compute stage)

```bash
# Admin password:
oc get secret grafana-admin-credentials -n smartshop \
  -o jsonpath='{.data.GF_SECURITY_ADMIN_PASSWORD}' | base64 -d
```

### OpenShift / Cluster State

| Screenshot | File | What it shows | Status |
|-----------|------|--------------|--------|
| 3 SparkApps running simultaneously | `openshift-3-sparkapps-running.png` | cpu-baseline + rapids + text-preprocessing all RUNNING in parallel | ✅ captured |
| Full smartshop pod stack | `openshift-full-stack-pods-running.png` | feast, grafana, milvus, postgres, redis, redsinsight all Running | ✅ captured |
| Spark executor pods with memory | `openshift-spark-executor-pods.png` | ~12 GB/executor, 4 executors each for rapids + cpu-baseline | ✅ captured |
| Pods view incl. Spark History Server | `openshift-pods-with-history-server.png` | history server + driver pods + completed build pods | ✅ captured |
| RAPIDS driver pod running | `openshift-rapids-driver-running.png` | rapids driver Running + build pods Completed | ✅ captured |

### Spark History Server

| Screenshot | File | What it shows | Status |
|-----------|------|--------------|--------|
| Both runs completed | `spark-history-both-runs-completed.png` | CPU baseline 2.0h + RAPIDS 1.3h side-by-side — best speedup evidence | ✅ captured |
| RAPIDS job completed (1h 3m) | `spark-history-rapids-completed-1h3m.png` | Single job view with event log download link | ✅ captured |
| All 3 apps completed | `spark-history-all-apps-completed.png` | Feature-engineering RAPIDS, CPU-baseline, text-preprocessing all in history | ✅ captured |
| Text preprocessing — stages (top) | `spark-history-textpreprocessing-jobs-top.png` | SmartShop-TextPreprocessing: 21 Spark stages | ✅ captured |
| Text preprocessing — stages (bottom) | `spark-history-textpreprocessing-jobs-bottom.png` | Lower stages list for text preprocessing job | ✅ captured |

![Spark History Server — CPU baseline 2.0h and RAPIDS 1.3h runs side-by-side](./assets/spark-history-both-runs-completed.png)

![Spark History Server — all 3 apps completed including text preprocessing](./assets/spark-history-all-apps-completed.png)

### Redis / Feature Store

| Screenshot | File | What it shows | Status |
|-----------|------|--------------|--------|
| RedisInsight — Browse tab | `redisinsight-browse-item-features.png` | 3,109,935 HASH keys, `item_id_XXXXXX_smartshop` pattern, 711 MB, 2 clients | ✅ captured |
| RedisInsight — Browse with key detail | `redisinsight-browse-key-detail.png` | Individual HASH key expanded: `_ts:item_features` + 6 aggregated fields | ✅ captured |
| RedisInsight — Analyze (summary) | `redisinsight-analyze-summary-clean.png` | 677 MB, 3,109,935 keys, 100% HASH type | ✅ captured |
| RedisInsight — Analyze (full view) | `redisinsight-analyze-3m-keys.png` | Same + memory-to-be-freed chart (flat — no TTL expiry) | ✅ captured |
| RedisInsight — Slow Log | `redisinsight-slowlog-hset-burst.png` | HSET + HMGET burst 10:15:54–10:17:14 IST, 44–75ms/write, Feast BFV materialize evidence | ✅ captured |
| RedisInsight — Slow Log (commands) | `redisinsight-slowlog-commands.png` | Full command text: key pattern `item_id_XXXXXXXXXX_smartshop` | ✅ captured |
| Grafana — Redis Feature Store | `grafana-redis-feature-store-working.png` | Commands/sec, Cache Hit Ratio 100%, Memory 711 MB, Keys 2M+, 3 clients, Ops/sec live | ✅ captured |

![RedisInsight — Browse: 3.1M HASH keys with item_id pattern](./assets/redisinsight-browse-item-features.png)

![RedisInsight — Browse: individual key expanded showing 6 feature fields](./assets/redisinsight-browse-key-detail.png)

![RedisInsight — Analyze: 677 MB, 3,109,935 keys, 100% HASH, no TTL](./assets/redisinsight-analyze-summary-clean.png)

![RedisInsight — Slow Log: HSET burst confirming Feast BFV materialize window (10:16–10:17)](./assets/redisinsight-slowlog-commands.png)

![Grafana — SmartShop Redis Feature Store: live metrics post-materialization](./assets/grafana-redis-feature-store-working.png)

### RHOAI Feature Store UI

| Screenshot | File | What it shows | Status |
|-----------|------|--------------|--------|
| Feature lineage — full view | `rhoai-feast-lineage-full.png` | Entities (user_id, item_id, review_id) → BFVs: user_features (6), item_features (6) | ✅ captured |
| Data sources | `rhoai-feast-data-sources.png` | `raw_reviews_source` + `review_embeddings_source`, linked to 2 feature views | ✅ captured |
| Features list | `rhoai-feast-features-list.png` | 12 features: item_avg_rating, item_price, user_avg_rating, user_review_count, … | ✅ captured |

![RHOAI Feature Store — lineage: entities → @batch_feature_view nodes](./assets/rhoai-feast-lineage-full.png)

![RHOAI Feature Store — data sources: raw_reviews_source feeds 2 feature views](./assets/rhoai-feast-data-sources.png)

![RHOAI Feature Store — features: 12 features across item_features and user_features](./assets/rhoai-feast-features-list.png)

### MinIO

| Screenshot | File | What it shows | Status |
|-----------|------|--------------|--------|
| Raw data loaded in MinIO | `minio-raw-data-loaded.png` | smartshop-raw bucket: Books/Electronics/Home_and_Kitchen parquet files (32.8 GiB, 300 objects) | ✅ captured |

### MLflow UI

| Screenshot | What to show | Status |
|-----------|-------------|--------|
| Experiment list — `smartshop-feature-engineering` | 2 runs: cpu-baseline + rapids | 🔲 pending |
| Side-by-side run comparison | `total_elapsed_s`, `throughput_rows_per_s`, speedup | 🔲 pending |
| Rec training loss curves | `train_loss` vs `val_loss` per epoch | 🔲 pending |

```bash
# MLflow browser URL (SSO via RHOAI dashboard):
open https://rh-ai.${OC_CLUSTER_DOMAIN}/mlflow/#/?workspace=smartshop

# Internal URI used by in-cluster pods:
# https://mlflow.redhat-ods-applications.svc.cluster.local:8443/mlflow
```

---

## Notebook-Generated PNGs (auto-uploaded to MinIO)

| File | Content | Status |
|------|---------|--------|
| `mlflow_gpu_vs_cpu.png` | 3-panel: elapsed time, throughput, 1.34× speedup bar chart | 🔲 |
| `dcgm_combined.png` | 6-panel DCGM grid (util, mem, power, SM active, DRAM active, NVLink) | 🔲 |
| `redis_ops.png` | Redis ops/s + hit ratio during Feast materialize + inference | 🔲 |
| `training_loss_curves.png` | Rec model loss per epoch (train + val) | 🔲 |

```bash
# Generate all PNGs:
bash scripts/apply-all.sh notebook

# Download from MinIO after job completes:
source .env
aws s3 cp s3://smartshop-models/notebooks/ ./notebooks/output/ \
  --recursive --endpoint-url $MINIO_ENDPOINT_EXTERNAL
```

---

## Pending Numbers for Slides

```
Rec model val accuracy:        ___ (pending TrainJob)
LLM fine-tune NCCL bandwidth:  ___ GB/s (pending LLM TrainJob)
Clean LLM training examples:   104.6M (after dedup from 140.8M)
```

---

## Artifact Collection Commands

```bash
# After Phase 3 (Spark ETL) — already completed:
source .env
oc create configmap smartshop-collect-script \
  --from-file=collect-run-metrics.sh=scripts/collect-run-metrics.sh \
  -n smartshop --dry-run=client -o yaml | oc apply -f -

RUN_TYPE=rapids APP_NAME=smartshop-feature-engineering-rapids \
  bash scripts/collect-run-metrics.sh
RUN_TYPE=cpu APP_NAME=smartshop-feature-engineering-cpu-baseline \
  bash scripts/collect-run-metrics.sh

# After Phase 4 (Feast materialization):
RUN_TYPE=feast bash scripts/collect-run-metrics.sh

# After Phase 5 (training):
RUN_TYPE=slurm TRAINJOB_NAME=smartshop-llm-finetune \
  bash scripts/collect-run-metrics.sh

# Full notebook run (generates all PNGs):
bash scripts/apply-all.sh notebook
oc logs -n smartshop job/smartshop-notebook-runner -f
aws s3 cp s3://smartshop-models/notebooks/ ./notebooks/output/ \
  --recursive --endpoint-url $MINIO_ENDPOINT_EXTERNAL
```

---

## Manifest Inventory

All working manifests in `infrastructure/openshift/*.yaml`.
Dot-prefixed files (`.*.yaml`) are temporary debugging artifacts — not tracked by git.
Deploy everything with `bash scripts/apply-all.sh all`.

| Manifest | Phase | Applied by | Status |
|----------|-------|-----------|--------|
| `user-workload-monitoring.yaml` | 1 | `apply-all.sh infra` | ✅ applied |
| `spark-metrics-configmap.yaml` | 1 | `apply-all.sh observability` | ✅ applied |
| `redis-exporter.yaml` | 1 | `apply-all.sh observability` | ✅ applied |
| `grafana.yaml` | 1 | `apply-all.sh observability` | ✅ applied |
| `metrics-collection-job.yaml` | 1 | `apply-all.sh observability` + per-run | ✅ applied |
| `imagestreams.yaml` | 2 | `apply-all.sh images` | ✅ applied |
| `buildconfigs.yaml` | 2 | `apply-all.sh images` | ✅ applied |
| `data-download-job.yaml` | 2 | `apply-all.sh data` | ✅ applied |
| `upload-spark-jars-job.yaml` | 2 | `apply-all.sh data` | ✅ applied |
| `spark-application-rapids.yaml` | 3 | `apply-all.sh spark` | ✅ COMPLETED |
| `spark-application-cpu-baseline.yaml` | 3 | `apply-all.sh spark` | ✅ COMPLETED |
| `spark-application-text-preprocessing.yaml` | 3 | `apply-all.sh spark` | 🔄 RUNNING |
| `spark-application-embedding.yaml` | 3 | `apply-all.sh spark` | 🔲 pending |
| `trainjobs.yaml` | 5 | `apply-all.sh training` | 🔲 pending |
| `inferenceservices.yaml` | 6 | `apply-all.sh serving` | 🔲 pending |
| `notebook-runner-job.yaml` | 7 | `apply-all.sh notebook` | 🔲 pending |
