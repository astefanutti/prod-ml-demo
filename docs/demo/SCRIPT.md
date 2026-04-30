# SmartShop AI — Demo Script

**Event:** Red Hat Summit 2026  
**Session:** Production ML at Scale: Distributed Training with PyTorch, Kubeflow, Spark & Feast on Red Hat OpenShift AI  
**Cluster:** `${OC_CLUSTER_DOMAIN}` · **Namespace:** `smartshop`

> This script is for the live demo presenter. It follows the pipeline in execution order,
> tells you exactly what to open, what to say, and what the audience should see at each step.
> Estimated live demo time: **20–25 minutes**.

---

## Live UI URLs (bookmark these before the session)

| What | URL |
|------|-----|
| RHOAI Dashboard | `https://rh-ai.${OC_CLUSTER_DOMAIN}` |
| Grafana (GPU + Redis metrics) | `https://grafana-smartshop.${OC_CLUSTER_DOMAIN}` |
| Spark History Server | `https://spark-history-smartshop.${OC_CLUSTER_DOMAIN}` |
| MLflow UI | `https://mlflow-redhat-ods-applications.${OC_CLUSTER_DOMAIN}` |
| MinIO Console | `https://minio-console-smartshop.${OC_CLUSTER_DOMAIN}` |
| Feast UI | `https://feast-smartshop-feast-ui-smartshop.${OC_CLUSTER_DOMAIN}` |
| RedisInsight | `https://redisinsight-smartshop.${OC_CLUSTER_DOMAIN}` |
| Attu (Milvus UI) | `https://attu-smartshop.${OC_CLUSTER_DOMAIN}` |
| Grafana Inference Metrics | `https://grafana-smartshop.${OC_CLUSTER_DOMAIN}/d/smartshop-inference` |
| Gradio Demo UI | *(local: `make demo`, or deploy as Route)* |

---

## Demo Flow Overview

```
1. Problem framing (1 min)
2. Data at scale — MinIO + Spark ETL (4 min)
3. RAPIDS GPU acceleration — Spark History Server + Grafana (4 min)
4. Feature store — Feast UI + RedisInsight (3 min)
5. Distributed training — Kubeflow TrainJob + MLflow (5 min)
6. Model serving — KServe endpoints (3 min)
7. Live end-to-end — Gradio UI (3 min)
8. Wrap-up: what OpenShift AI provided (2 min)
```

---

## 1. Problem Framing (~1 min)

**Say:**
> "We're going to walk through an end-to-end production ML platform for an e-commerce use case —
> product recommendations, review summarization, and Q&A.
>
> The challenge: 49 GB of Amazon review data, 140 million interactions, a 7-billion parameter LLM,
> and the requirement that the features you train on are identical to the features you serve — no drift.
>
> Everything you'll see runs on Red Hat OpenShift AI using standard Kubernetes manifests.
> No custom scripts that bypass the platform. No vendor lock-in."

**Show:** The architecture diagram in README.md or a slide.

---

## 2. Data at Scale — MinIO + Spark ETL (~4 min)

### 2a. MinIO Console — show the data

1. Open **MinIO Console**: `https://minio-console-smartshop.${OC_CLUSTER_DOMAIN}`
2. Navigate to `smartshop-raw` → `raw/` — show the raw Amazon Reviews JSON files (~49 GB)
3. Navigate to `smartshop-features` → `user_features/` and `item_features/` — show Spark output Parquet files

**Say:**
> "This is MinIO — our S3-compatible object store running on-cluster.
> The raw data comes in here. Spark reads it, computes features, and writes Parquet back here.
> The same bytes flow all the way through to training and serving — there's no copy step."

### 2b. Spark job — show the completed run

```bash
# Verify both jobs completed:
oc get sparkapplication -n smartshop
```

Expected output:
```
NAME                                         STATUS      ATTEMPTS
smartshop-feature-engineering-cpu-baseline   COMPLETED   1
smartshop-feature-engineering-rapids         COMPLETED   1
smartshop-text-preprocessing                 COMPLETED   1
```

**Say:**
> "We ran two parallel feature engineering jobs — the same Python code, two different executor configurations.
> One used CPU workers. One used NVIDIA RAPIDS GPU workers.
> Let's look at the difference."

---

## 3. RAPIDS GPU Acceleration (~4 min)

### 3a. Spark History Server — DAG comparison

1. Open **Spark History Server**: `https://spark-history-smartshop.${OC_CLUSTER_DOMAIN}`
2. Click the **RAPIDS job** (`smartshop-feature-engineering-rapids`) → show the Stage timeline
3. Click the **CPU Baseline job** → show the same stage timeline
4. Point out the wall-clock difference in the "Duration" column on the Jobs tab

**Key numbers to state:**

| Metric | CPU Baseline | RAPIDS GPU | Speedup |
|--------|-------------|------------|---------|
| Total rows processed | 140,772,341 | 140,772,341 | identical |
| Item feature aggregation | 41 s | 22 s | **1.87×** |
| Total wall-clock | 719 s | 537 s | **1.34× overall** |
| Throughput | 195,727 rows/s | 262,231 rows/s | **+34%** |

**Say:**
> "The Python code is identical. The SparkApplication YAML is identical except for the executor image.
> We swapped the CPU JVM executor for NVIDIA RAPIDS GPU executors — no code changes, no algorithm changes.
> The shuffle-bound stages don't benefit, which is expected. The aggregation stages are 1.8× faster.
> On a larger dataset or with more GPU nodes, the gap widens significantly."

### 3b. Grafana — GPU utilization

1. Open **Grafana**: `https://grafana-smartshop.${OC_CLUSTER_DOMAIN}`
2. Open the **GPU Utilization** dashboard
3. Show the GPU utilization spike during the RAPIDS job window (~14:00–14:55 UTC)
4. Show DCGM metrics: SM Active %, Framebuffer Memory (80 GB A100s filling up during joins)

```bash
# Get Grafana admin password:
oc get secret grafana-admin-credentials -n smartshop \
  -o jsonpath='{.data.GF_SECURITY_ADMIN_PASSWORD}' | base64 -d
```

**Say:**
> "DCGM Exporter is scraping GPU metrics directly from the NVIDIA management library.
> Prometheus collects them. Grafana visualizes them. All managed by the GPU Operator.
> The spikes you see correspond exactly to the RAPIDS executor stages in the Spark History Server."

---

## 4. Feature Store — Feast UI + RedisInsight (~3 min)

### 4a. Feast UI — show feature definitions

1. Open **Feast UI**: `https://feast-smartshop-feast-ui-smartshop.${OC_CLUSTER_DOMAIN}`
2. Click **Feature Views** — show `user_features`, `item_features`, `review_embeddings`
3. Click `user_features` — show the feature schema (avg_rating, review_count, avg_price, etc.)

**Say:**
> "Feast is the feature store. Every feature you see here was defined once in Python.
> That same definition was used when building the training dataset and is used at inference time
> when a user makes a request. There is no separate 'prod feature logic' that can drift from training.
> That's training-serving skew eliminated by design."

### 4b. RedisInsight — show materialized features

1. Open **RedisInsight**: `https://redisinsight-smartshop.${OC_CLUSTER_DOMAIN}`
2. Navigate the key list — show that 35M user keys and 9.8M item keys are materialized
3. Click any key to show the hash fields (feature values pre-computed and waiting in memory)

**Key numbers:**
- **35,049,327** unique users materialized to Redis
- **9,790,339** unique items materialized to Redis
- Lookup latency: **< 1 ms** at serving time

**Say:**
> "These 35 million user feature vectors are pre-computed and sitting in Redis.
> When a recommendation request hits the endpoint, it calls Feast `get_online_features`,
> Redis returns the feature hash in sub-millisecond time, and the model makes its prediction.
> No on-the-fly feature computation at serving time."

---

## 5. Distributed Training — Kubeflow TrainJob + MLflow (~5 min)

### 5a. Kubeflow Trainer — show the TrainJob

```bash
# Show training jobs:
oc get trainjob -n smartshop
```

1. Open the **RHOAI Dashboard** → Distributed Workloads → show the `smartshop-rec-train` TrainJob
2. Show the 4 worker pods (torchrun DDP, 1 node × 4 GPUs)

**Say:**
> "The recommendation model uses a two-tower architecture — one tower for users, one for items.
> It's trained with PyTorch DistributedDataParallel across 4 GPUs.
> The TrainJob is submitted as a single Kubernetes resource — Kubeflow Trainer handles
> process coordination, restarts, and cleanup."

### 5b. MLflow — show experiment tracking

1. Open **MLflow UI**: `https://mlflow-redhat-ods-applications.${OC_CLUSTER_DOMAIN}`
2. Select the `smartshop` workspace (top-left dropdown)
3. Open the `smartshop-feature-engineering` experiment — show the 2 runs (cpu-baseline + rapids) side by side
4. Click **Compare** — show `total_elapsed_s` and `throughput_rows_per_s` in the comparison view
5. Open the `smartshop-rec-train` experiment — show loss curves over epochs

**Say:**
> "MLflow is tracking every run — hyperparameters, metrics per epoch, and the final model artifact.
> The artifact is written to `s3://smartshop-models/` on MinIO.
> The RHOAI Model Registry registers a pointer to that same S3 path — no re-upload.
> KServe reads `storageUri` at pod startup — still no copy. The same bytes go from training to serving."

### 5c. LLM fine-tuning — QLoRA + FSDP

```bash
# Show Slurm nodes:
oc get pods -n slurm | grep worker
```

**Say:**
> "The LLM job is different. Fine-tuning Mistral-7B requires more than 80 GB of GPU RAM —
> it doesn't fit on a single A100. We use FSDP to shard the model across 8 GPUs on 2 nodes.
>
> Kubernetes Job scheduling doesn't guarantee that all 8 pods start simultaneously
> or that they land on NVLink-connected nodes. Slurm does — that's gang scheduling.
> The Kubeflow Trainer TrainJob dispatches to Slurm via the Slinky operator.
> You get Kubernetes API semantics on top, Slurm GPU scheduling underneath."

> "QLoRA means we're only training low-rank adapter matrices — about 70 million parameters
> instead of 7 billion. Memory drops from ~112 GB to ~24 GB. The adapter checkpoint is 270 MB
> versus 28 GB for a full fine-tune. Training time drops proportionally."

---

## 6. Model Serving — KServe InferenceServices (~3 min)

```bash
# Show all three endpoints:
oc get inferenceservice -n smartshop
```

Expected:
```
NAME              URL                                                     READY
smartshop-rec     https://smartshop-rec-smartshop.${OC_CLUSTER_DOMAIN}   True
smartshop-llm     https://smartshop-llm-smartshop.${OC_CLUSTER_DOMAIN}   True
smartshop-rag     https://smartshop-rag-smartshop.${OC_CLUSTER_DOMAIN}   True
```

**Say:**
> "Three KServe InferenceServices — recommendation, review summarization, and RAG Q&A.
> Each one reads its model from the MinIO S3 path registered in the Model Registry.
> KServe handles autoscaling — zero to five replicas based on request load.
> The recommendation endpoint returns results in under 5 ms end-to-end including the Feast lookup."

---

## 7. Live End-to-End — Gradio UI (~3 min)

1. Open the **Gradio Demo UI** (get the route: `oc get route demo-ui -n smartshop -o jsonpath='{.spec.host}'`)
2. **Recommendations tab**: enter a user ID → show product recommendations returned in < 100 ms
3. **Review Summary tab**: paste a product ASIN → show Mistral-7B generated summary
4. **Q&A tab**: ask "Is this product good for outdoor use?" → show RAG response with source reviews

**Say:**
> "This is the full pipeline end-to-end — from 49 GB of raw data, through Spark ETL,
> Feast feature materialization, distributed training, and now live inference.
>
> The recommendation is backed by 35 million pre-materialized user feature vectors in Redis.
> The Q&A answer is backed by 104 million review embeddings in Milvus.
> Everything — data, models, features — lives in one MinIO instance. No cross-service data duplication."

---

## 8. Wrap-up: What OpenShift AI Provided (~2 min)

**Say:**
> "Let's close the loop on what the platform actually gave us here.
>
> **Spark Operator** — submit SparkApplications as Kubernetes resources. No Spark cluster to manage.
>
> **Kubeflow Trainer** — one CRD for all distributed training patterns: DDP and QLoRA/FSDP.
>
> **Feast** — operator-managed feature store. Offline store on MinIO, online store on Redis,
> vector store on Milvus. Training-serving consistency enforced at the schema level.
>
> **MLflow + Model Registry** — end-to-end artifact lineage. The model the registry promotes
> is the exact artifact MLflow wrote. KServe reads the same path.
>
> **KServe** — three production endpoints, autoscaling, model versioning, zero-copy rollout.
>
> The developer wrote Python training scripts and Kubernetes YAML.
> The platform handled everything else."

---

## Fallback / Q&A Commands

```bash
# Check all SparkApplication statuses:
oc get sparkapplication -n smartshop

# Pull RAPIDS vs CPU metrics:
oc logs smartshop-feature-engineering-rapids-driver -n smartshop | grep '\[METRIC\]'
oc logs smartshop-feature-engineering-cpu-baseline-driver -n smartshop | grep '\[METRIC\]'

# Check Feast materialization status:
oc exec -it -n smartshop \
  $(oc get pod -n smartshop -l feast.dev/name=smartshop-feast -o jsonpath='{.items[0].metadata.name}') \
  -c registry -- feast -c /feast-data/smartshop/feast/feature_repo feature-views list

# Check KServe endpoints:
oc get inferenceservice -n smartshop

# Check Grafana admin password:
oc get secret grafana-admin-credentials -n smartshop \
  -o jsonpath='{.data.GF_SECURITY_ADMIN_PASSWORD}' | base64 -d

# Check Redis key count (35M users materialized):
oc exec -n smartshop deploy/redis -- redis-cli -a $REDIS_PASSWORD dbsize
```

---

## Common Questions

**"Why not just use a managed cloud service instead of running Spark/Slurm yourself?"**
> "For regulated industries — financial services, healthcare, public sector — data can't leave the
> cluster. OpenShift AI gives you the full ML platform stack on your own infrastructure.
> The same operators, same manifests, same workflows — on-prem or cloud."

**"Why Feast instead of computing features inline at serving time?"**
> "Feast pre-materializes features so the model never waits for a database query on the hot path.
> More importantly, it eliminates training-serving skew — the feature schema is defined once
> and enforced everywhere. Without it, the production feature pipeline can silently diverge
> from what the model was trained on."

**"Why Slurm instead of just more Kubernetes pods?"**
> "FSDP needs all N pods to start at the same instant on topology-aware nodes — GPUs in the same
> server for NVLink, servers connected by InfiniBand. Kubernetes Job scheduling is best-effort.
> Slurm guarantees gang scheduling and topology-aware placement. Slinky gives you both:
> Kubernetes API surface, Slurm scheduling underneath."

**"What's the speedup from RAPIDS?"**
> "1.34× overall on this dataset — because some stages are shuffle-bound and the bottleneck is
> network I/O, not compute. The aggregation stages hit 1.87×. On a compute-bound workload
> with larger batches, the speedup is significantly higher."
