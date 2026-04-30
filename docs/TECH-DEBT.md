# Tech Debt & Productisation Gaps

**Context:** Findings from building a production-grade distributed ML pipeline
(Spark RAPIDS + Kubeflow + Feast + QLoRA/FSDP) on Red Hat OpenShift AI.
Intended as a community reference for what works, what is rough, and what needs
platform-level support for teams to adopt this stack at scale.

---

## 1. Data Ingestion

**Current approach:** Kubernetes Job streams HuggingFace Hub → MinIO via `requests` streaming + `huggingface_hub`. Works but is hand-rolled.

| Gap | Impact | Fix |
|---|---|---|
| No native dataset ingestion primitive | Every project reinvents a download Job | RHOAI should provide a Data Connection + Dataset Import wizard (HF Hub, S3, GCS, HTTP URI → K8s Job) |
| No progress visibility for long downloads | 22 GB Electronics JSONL has no progress in `oc logs` | Periodic `[METRIC]` logging + Prometheus counter `rows_downloaded_total` |
| No data versioning | Re-running download job overwrites data silently | MinIO object versioning or DVC integration |
| Books/Home_and_Kitchen have no metadata shards in HF repo | `raw_meta_Books/` does not exist | Feature engineering uses metadata only for Electronics; other categories still produce features via left join — no action needed |

---

## 2. Spark on Kubeflow (SparkApplication)

**Current approach:** `SparkApplication` CR via `spark-operator`. Two image variants: CPU (`spark-jobs`) and GPU/RAPIDS (`spark-jobs-rapids`). Feast materialization uses `SparkComputeEngine` with `k8s://` distributed mode.

| Gap | Impact | Fix |
|---|---|---|
| No official RHOAI-supported Spark runtime image | Must build and maintain `Containerfile.spark` manually; version drift risk | RHOAI should ship a supported UBI9-based Spark 3.5 image with s3a connector, PyArrow, sentence-transformers |
| RAPIDS image must be hand-built | `rapids-4-spark` JAR + CUDA compatibility is fragile (cuda12 JAR on CUDA 13.0 cluster works via NVIDIA backward compat, but fragile) | NVIDIA + Red Hat should co-publish a certified RAPIDS + Spark image in the RHOAI catalog |
| `spark.plugins` requires manual CUDA version matching | Wrong JAR falls back to CPU silently | RHOAI GPU Operator should auto-inject the correct RAPIDS plugin version based on detected CUDA driver |
| No Spark History Server deployed by default | Post-run stage/SQL analysis requires port-forwarding to a live driver | RHOAI should deploy a Spark History Server backed by S3 when SparkOperator is enabled |
| `envsubst` for manifest substitution is fragile | Shell quoting + zsh plugins + unset vars cause silent failures | Kustomize overlays or a `make render-manifests` target using Python `string.Template` |
| S3A credentials provider mismatch | `EnvironmentVariableCredentialsProvider` expects `AWS_*` vars; MinIO secrets use `MINIO_*` naming | RHOAI Data Connection secrets should always inject both `AWS_*` and provider-specific aliases |

---

## 3. RAPIDS GPU Acceleration

**Current state:** `rapids-4-spark` JAR pre-placed in executor image. Verified working on A100 CUDA 13.0 with cuda12 JAR. CPU 686s vs RAPIDS 658s (4% faster) on feature engineering workload — I/O bound, not compute bound.

| Gap | Impact | Fix |
|---|---|---|
| CUDA driver/JAR version mismatch is silent | Wrong JAR falls back to CPU with no warning | RAPIDS should emit a startup warning when JAR CUDA version ≠ driver CUDA version |
| UDFs and `input_file_name()` silently fall back to CPU | Developers don't know which operations are not accelerated | RAPIDS docs should list all non-acceleratable operations prominently |
| No GPU-aware autoscaling for SparkApplication | Fixed executor count; can't scale based on queue depth | KEDA + GPU metrics for dynamic executor scaling |
| `spark.task.resource.gpu.amount` default of `1.0` limits concurrency | With 4 GPUs × 1 task/GPU = 4 concurrent tasks vs CPU's 8; masked a performance gap until explicitly fixed to `0.5` | Must be set to `0.5` for fair comparison — documented in `feast/BFV-DESIGN.md` |

**What works:**
- CUDA 13.0 + cuda12 JAR backward compatibility ✅
- Zero Python code changes — SQL plugin intercepts DataFrame ops automatically ✅
- `spark.ui.prometheus.enabled=true` + DCGM gives full observability stack ✅

---

## 4. Feast Feature Store on RHOAI

**Current state:** `FeatureStore` CR via RHOAI Feast Operator. Custom `feast-spark-server` image with `pyspark==3.5.3` + `feast[spark]`. SparkComputeEngine running in `k8s://` distributed mode (executor pods). Online store: Redis (26.5M keys). Milvus vector store deployed but embeddings not yet loaded.

| Gap | Impact | Fix |
|---|---|---|
| `feast materialize` is manual and not scheduled | Features go stale unless triggered manually | RHOAI should provide a `spec.materializationSchedule` cron field in the FeatureStore CR |
| `feast[spark]` requires `pyspark>=4.0.0` but ETL images use 3.5.3 | The Feast pod SparkSession is independent of the ETL Spark Operator — no conflict — but must be explicitly documented to avoid confusion | Already handled via custom image; document pyspark version split clearly |
| NFS double-mount issue with `smartshop-shared-storage` | Feast registry PVC must be a separate dedicated PVC; sharing NFS PVC in the same pod fails | Document as known NFS CSI limitation; recommend dedicated PVC per component |
| Milvus materialization not yet triggered | `review_embeddings_view` is registered but vectors not loaded; RAG retrieval will return empty results | Run `scripts/load_embeddings_to_milvus.py` after Spark embedding job completes |
| No built-in feature drift detection | Schema changes break downstream models silently | Great Expectations or Evidently AI on `feast apply` |
| Feast SDK version must match server | `feast apply` errors on version mismatch | Lock SDK + server versions; document upgrade path |

---

## 5. Kubeflow Training Operator (DDP + FSDP)

**Current state:** `TrainJob` CR for PyTorch DDP rec model (1 node × 4 GPU). QLoRA fine-tuning with FSDP on plain K8s (1 node × 4 GPU). Slurm integration deferred to post-Summit pending upstream Kubeflow Trainer support (issue #2249).

| Gap | Impact | Fix |
|---|---|---|
| `ClusterTrainingRuntime` for Slurm is brand new (weeks old) | Zero community documentation; debugging dispatch failures is opaque | RHOAI should publish a Slurm integration guide with example configs |
| NCCL over OVN-K SDN needs `NCCL_IB_DISABLE=1` or SR-IOV | Full NVLink bandwidth not achievable over default OVN network | Document SR-IOV / InfiniBand config for multi-node training |
| No native training job queue/priority | Multiple TrainJobs compete for GPUs with no fairness policy | Kueue (already in ecosystem) for GPU quota-based queuing |
| FSDP checkpoint format not standardized | `torch.save` vs `torch.distributed.checkpoint` incompatibility | RHOAI should recommend and document checkpoint format |
| `item_feat_dim` default in `model.py` was 8 but data has 6 features | Latent trap: `TwoTowerModel()` without checkpoint would silently build wrong shape | ✅ Fixed — default now 6; checkpoint saves/restores actual dims (RHOAIENG-57388 Bug 1) |
| N+1 Feast lookups in recommendation server | 100 sequential Redis round-trips per request | ✅ Fixed — single batched `get_online_features()` call (RHOAIENG-57388 Bug 6) |
| Training pod had no `feature_store.yaml` | `FeatureStore(repo_path="/feast/feature_repo")` would crash — path doesn't exist in trainer image | ✅ Fixed — `feature_store_training.yaml` (remote registry) baked into image via Containerfile |

**What works:**
- `TrainJob` CR is clean and declarative ✅
- RHOAI-hosted MLflow accessible from training pods via `MLFLOW_TRACKING_URI` ✅

---

## 6. KServe Model Serving

**Current state:** Not yet deployed. Target: 3 InferenceServices (`/recommend`, `/summarize`, `/ask`). Tracked in RHOAIENG-57390.

| Gap | Impact | Fix |
|---|---|---|
| Model Registry → KServe deployment is manual | Must copy S3 URI from training output → InferenceService YAML | RHOAI Model Registry → KServe deploy button should auto-generate InferenceService |
| Feast feature retrieval latency not tracked per request | Cannot measure end-to-end P99 for recommendation | Add OTEL span for Feast online store call in serving code |
| GPU memory fragmentation across 3 InferenceServices | rec + LLM + RAG sharing GPU nodes may OOM | RHOAI multi-model serving runtime (Triton) that shares GPU memory |
| vLLM autoscaling requires Knative serverless mode | `RawDeployment` only scales on CPU/memory, not token throughput | Confirm InferenceService uses serverless deployment mode |

---

## 7. Observability

**Current state:** DCGM → Prometheus, redis_exporter, Spark PrometheusServlet, MLflow, Grafana with 3 dashboards. Documented in `observability/MONITORING.md`.

| Gap | Impact | Fix |
|---|---|---|
| No OTEL distributed tracing | Cannot trace a single request from Gradio → KServe → Feast → Redis | OpenTelemetry Collector + Tempo; instrument Python services |
| User-workload monitoring disabled by default | ServiceMonitors in user namespaces don't work without cluster-admin enabling it | RHOAI should enable this automatically for Data Science projects |
| Grafana not included in RHOAI by default | Must self-deploy; no persistent storage | Bundle Grafana Operator + pre-wire to OCP Thanos instance |
| No alerting on training job failures | Silent failures; must watch `oc get trainjob` manually | RHOAI should ship `PrometheusRule` CRs for common ML job failure conditions |

**What works:**
- DCGM → Prometheus is the right GPU metrics path for OpenShift ✅
- `redis_exporter` + ServiceMonitor gives Feast online store visibility ✅
- Spark PrometheusServlet exposes per-executor JVM + shuffle metrics ✅

---

## 8. Runtime Images (hand-built — should be platform-provided)

| Image | Base | Status |
|---|---|---|
| `spark-jobs` | UBI9 Python 3.11 | ❌ Hand-built — PySpark 3.5, PyArrow, sentence-transformers, boto3 |
| `spark-jobs-rapids` | apache/spark:3.5.3 | ❌ Hand-built — adds `rapids-4-spark` JAR, `getGpusResources.sh` |
| `feast-spark-server` | odh-feature-server-rhel9 | ✅ Built via BuildConfig — pyspark 3.5.3 + feast[spark] |
| `feast-spark-executor` | apache/spark:3.5.3 | ✅ Built — CPU executor image for k8s:// distributed mode |
| `feast-spark-executor-rapids` | apache/spark:3.5.3 | ✅ Built — RAPIDS GPU executor image |
| `rec-trainer` | UBI9 Python 3.11 | ❌ Hand-built — PyTorch 2.x, mlflow, feast, boto3 |
| `llm-trainer` | UBI9 Python 3.11 | ❌ Hand-built — PyTorch 2.x + FSDP, peft, trl, mlflow |
| `rec-server` / `rag-server` / `llm-server` | UBI9 Python 3.11 | ❌ Hand-built — fastapi, feast, sentence-transformers, vllm |

**Ask for RHOAI product team:** publish these as optional add-on images in the RHOAI
workbench catalog, versioned alongside RHOAI releases.

---

## 9. Developer Experience

| Gap | Pain point | Fix |
|---|---|---|
| `envsubst` + `.env` is the only templating | Fragile with zsh plugins and unset vars | Kustomize overlays or `make render-manifests` using Python `string.Template` |
| No local development path for Spark jobs | Requires Java + Spark install locally | Dev Container (`devcontainer.json`) with Spark, Python, `oc` pre-installed |
| No CI/CD for the ML pipeline | Every step is manual `oc apply` | Tekton Pipeline: build → ETL → materialize → train → serve |
| No local MinIO/Redis for development | Must connect to cluster for all dev work | `docker-compose.yml` with MinIO + Redis + PostgreSQL |
| KFP pipeline Spark components use `local[*]` | Negates distributed Spark narrative inside the pipeline | Replace with `SparkApplication` CRD dispatch + poll for `COMPLETED` — tracked in RHOAIENG-57388 Bug 2 |

---

## 10. Security

| Gap | Risk | Fix |
|---|---|---|
| Spark executor pods use `edit` ClusterRole | Broader than needed | Scope down to minimum: `pods/create`, `pods/delete`, `pods/get` in the target namespace only |
| MinIO root credentials in `smartshop-credentials` | Broad access; no bucket-level IAM | Migrate to MinIO IAM policies per service account |
| No network policy between components | Feast → Redis, Spark → MinIO unrestricted | `NetworkPolicy` CRs: Feast only talks to Redis; Spark only to MinIO |
| Model artifacts unsigned | No provenance for `best_model.pt` | Sigstore/Cosign for model artifact signing via RHOAI Model Registry |

---

## 11. Prerequisites Before Starting (for platform teams)

### Must have (blockers)
- GPU node pool with DCGM exporter DaemonSet deployed
- User-workload monitoring enabled (`cluster-monitoring-config` ConfigMap)
- Persistent storage: NFS CSI or ODF with `ReadWriteMany` StorageClass
- MinIO with pre-created buckets and a Data Connection in RHOAI
- Redis deployed
- Kubeflow Spark Operator with GPU-capable `spark` ServiceAccount + RBAC
- Kubeflow Training Operator (`TrainJob` CRD)
- RHOAI Feast Operator with a working `FeatureStore` CR template
- MLflow deployed (RHOAI provides this ✅)
- External container registry (quay.io) with push credentials

### Should have
- Spark History Server backed by S3
- Grafana with OCP Thanos datasource pre-configured
- Slurm `slinky` plugin + GPU partition available
- `anyuid` or custom Spark SCC pre-approved for the namespace
- HuggingFace token stored as `hf-credentials` secret

### Nice to have
- Kueue for GPU quota-based job queuing
- OpenTelemetry Collector + Tempo for distributed tracing
- Dev Container definition for local development
- Tekton Pipeline for full ML CI/CD

---

## Top 5 Platform Investments for RHOAI

| Priority | Investment | Why |
|---|---|---|
| 1 | **Feast feature server image with pyspark** | Unlocks SparkOfflineStore + SQLRegistry; current default image forces dask fallback |
| 2 | **Supported Spark 3.5 + RAPIDS runtime images** | Every team rebuilds these from scratch; fragile CUDA compatibility |
| 3 | **User-workload monitoring enabled by default** | Blocks all custom ServiceMonitors; one-line fix with high impact |
| 4 | **Feast `materializationSchedule` in CR** | Features go stale without scheduled materialization; critical for production |
| 5 | **Slurm integration guide + ClusterTrainingRuntime template** | Zero community documentation; high barrier to adoption |
