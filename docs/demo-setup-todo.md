# SmartShop AI — Demo Setup TODO
**Target:** Red Hat Summit 2026 recording
**Deadline:** ~20 days from kick-off
**Epic:** [RHOAIENG-57382](https://redhat.atlassian.net/browse/RHOAIENG-57382)

---

## Phase 0 — Cluster Access & Infra Setup ✅ MOSTLY DONE
> Owner: cluster admin + team infra lead
> Blocker for everything else

- [ ] Confirm GPU node availability: at least 2 nodes × 4 GPUs for FSDP LLM job
- [x] Create `smartshop` namespace on OpenShift — `infrastructure/smartshop/namespace.yaml`
- [x] Install / verify Spark Operator is available on cluster — confirmed, screenshots in `docs/assets/`
- [ ] Install / verify Kubeflow Trainer v2 operator (`TrainJob` CRD present)
- [x] Install / verify Slurm + Slinky operator — deployed via OperatorHub, screenshots in `docs/assets/`
- [x] Deploy MinIO — AIStor operator installed, `smartshop-raw/features/embeddings/models` buckets created
- [x] Deploy Redis — running in `smartshop`, online store reachable; RedisInsight UI exposed
- [x] Deploy Milvus — running in `smartshop` via Helm (`infrastructure/milvus/`), NFS + MinIO S3 backend
- [x] Deploy Feast Feature Store — operator-managed, visible in RHOAI dashboard (`infrastructure/feast/`)
- [x] Deploy MLflow — running in `redhat-ods-applications`, Postgres backend + MinIO S3 artifacts (`infrastructure/mlflow/`)
- [x] All secrets wired via `.env` + `make setup-secrets` — no hardcoded credentials in repo
- [ ] Confirm RHOAI Model Registry is enabled in the namespace
- [ ] Confirm KServe is available and can schedule GPU InferenceServices
- [ ] Set resource quotas / limits to not conflict with other teams on cluster

---

## Phase 1 — Container Images
> Owner: dev team
> Build and push before any K8s job can run

- [ ] `make build-images` — builds rec-trainer, llm-trainer, rec-server, llm-server, rag-server
- [ ] `make push-images` — pushes all images to registry
- [ ] Confirm Containerfile for rec-trainer: `Containerfile.rec-trainer`
- [ ] Confirm Containerfile for llm-trainer: `Containerfile.llm-trainer`
- [ ] Add/verify Containerfiles for serving images (rec, llm, rag servers)
- [ ] Verify image tags match what's in `infrastructure/openshift/inferenceservices.yaml` and `trainjobs.yaml`

---

## Phase 2 — Data Pipeline (Spark)
> Jira: [RHOAIENG-57386](https://redhat.atlassian.net/browse/RHOAIENG-57386)
> Owner: Nikhil

- [ ] Download Amazon Reviews 2023 dataset: `make data-full` (49GB, ~571M reviews)
  - Alternative: `make data-sample` for local dev (5% subset)
- [ ] Upload raw data to MinIO `smartshop/raw/` bucket
- [ ] Fix entity key mismatch in `spark/feature_engineering.py`:
  - Item features written with `parent_asin` but Feast entity is `item_id` — rename column
- [ ] Add null filters for `user_id` and `parent_asin` in feature engineering
- [ ] Handle `item_rating_stddev` null for single-review items (fill 0 or drop)
- [ ] Fix non-deterministic `review_id` in `spark/embedding_generation.py`:
  - Replace `monotonically_increasing_id()` with stable hash or `concat(user_id, asin, timestamp)`
- [ ] Submit Spark feature engineering job: `make spark-run`
- [ ] Verify output parquet files land in MinIO `smartshop/features/`
- [ ] Submit Spark embedding generation job
- [ ] Spot-check output: row count, null rate, schema correctness
- [ ] _(Optional)_ Run RAPIDS variant: `make spark-rapids` — validate output matches CPU run

---

## Phase 3 — Feast Feature Store
> Jira: [RHOAIENG-57387](https://redhat.atlassian.net/browse/RHOAIENG-57387)
> Owner: Nikhil

- [ ] Update `feast/feature_repo/feature_store.yaml` with cluster MinIO, Redis, Milvus endpoints
- [ ] Run `feast apply` to register feature views and entities
- [ ] Run `feast materialize` to push user/item features from S3 → Redis (online store):
  `make feast-materialize`
- [ ] Push review embeddings to Milvus vector store
- [ ] Fix `feast/test_features.py`: replace fake entity IDs with real ones from dataset
- [ ] Run `feast/test_features.py` and verify:
  - [ ] Offline retrieval returns non-null user features
  - [ ] Online retrieval (`get_online_features`) returns non-null item features from Redis
  - [ ] Vector retrieval (`retrieve_online_documents`) returns nearest embeddings from Milvus
- [ ] Confirm Feast feature dimensions match training expectations:
  - `user_features_view` → 6 fields (for rec model `user_feat_dim=6`)
  - `item_features_view` → numeric fields only (exclude string `item_price_bucket`, `category`)

---

## Phase 4 — Fix Code Bugs
> Jira: [RHOAIENG-57388](https://redhat.atlassian.net/browse/RHOAIENG-57388)
> Owner: dev team

- [ ] Fix `serving/recommendation/server.py` critical bug:
  - `item_feat_dim` default is 8, trained model uses 6 — pass correct value or load from model metadata
- [ ] Fix N+1 Feast lookup in rec server: replace per-item `get_online_features()` loop with single batched call
- [ ] Replace deprecated `@app.on_event("startup")` in all 3 servers with `lifespan` context manager:
  - `serving/recommendation/server.py`
  - `serving/llm/server.py`
  - `serving/rag/server.py`
- [ ] Fix `pipelines/e2e_pipeline.py`: Spark step uses `--master local[*]` — update to use Spark Operator on K8s
- [ ] Fix `data/download.py` docstring: dataset has ~571M reviews, not 233M
- [ ] Add Model Registry registration step to `e2e_pipeline.py` after training

---

## Phase 5 — MLflow Experiment Tracking _(best effort)_
> Jira: [RHOAIENG-57407](https://redhat.atlassian.net/browse/RHOAIENG-57407)
> Owner: dev team

- [x] Deploy MLflow tracking server — running in `redhat-ods-applications` (RHOAI-managed operator)
  - [x] PostgreSQL backend (`infrastructure/mlflow/postgres.yaml`) — OpenShift-native `postgresql:15-el9` image
  - [x] S3 artifact store → `s3://smartshop-models` via MinIO (`infrastructure/mlflow/mlflow-cr.yaml`)
  - [x] OVN-K DNS workaround documented — using Postgres ClusterIP directly in backend secret
- [ ] Update `training/recommendation/train.py` to log:
  - Hyperparams (lr, batch size, embed_dim, hidden_dim)
  - Per-epoch train/val loss
  - Final model artifact via `mlflow.pytorch.log_model()`
- [ ] Update `training/llm/finetune.py` to log:
  - QLoRA hyperparams (r, lora_alpha, target_modules)
  - Training loss curve
- [ ] Add `MLFLOW_TRACKING_URI` env var to both TrainJob specs in `trainjobs.yaml`
- [ ] Verify shared artifact path: MLflow and RHOAI Model Registry both point to `s3://smartshop/models/`
- [ ] Confirm MLflow UI shows runs after training completes

---

## Phase 6 — Distributed Training + Model Registry
> Jira: [RHOAIENG-57389](https://redhat.atlassian.net/browse/RHOAIENG-57389)
> Owner: dev team (Kubeflow/training side)

- [ ] Apply TrainJob manifests: `kubectl apply -f infrastructure/openshift/trainjobs.yaml`
- [ ] Submit rec model DDP training job (`smartshop-rec-train`):
  - 1 node × 4 GPUs, `torchrun --nproc_per_node=4`
  - Monitor with `kubectl logs -f`
- [ ] Submit LLM FSDP fine-tune via Slurm (`smartshop-llm-finetune`):
  - 2 nodes × 4 GPUs via Slurm sbatch (direct)
  - Monitor: `kubectl get trainjob`, `squeue`, Slurm logs
- [ ] Verify both jobs complete without OOM or NCCL errors
- [ ] Register rec model artifact in RHOAI Model Registry
- [ ] Register LLM fine-tune artifact (LoRA adapter) in RHOAI Model Registry
- [ ] Confirm model versions visible in RHOAI UI Model Registry tab

---

## Phase 7 — KServe Model Serving
> Jira: [RHOAIENG-57390](https://redhat.atlassian.net/browse/RHOAIENG-57390)
> Owner: dev team

- [ ] Apply InferenceService manifests: `kubectl apply -f infrastructure/openshift/inferenceservices.yaml`
- [ ] Set env vars on InferenceServices: `MILVUS_HOST`, `REDIS_HOST`, `FEAST_REPO_PATH`, `MODEL_PATH`
- [ ] Wait for all 3 services to reach `Ready` state:
  - `smartshop-rec` — recommendation server
  - `smartshop-llm` — vLLM with LoRA adapter (Mistral-7B)
  - `smartshop-rag` — RAG Q&A server
- [ ] Smoke-test each endpoint:
  ```bash
  curl -X POST https://<rec-url>/recommend -d '{"user_id": "U123"}'
  curl -X POST https://<llm-url>/summarize -d '{"asin": "B08N5KWB9H", "reviews": [...]}'
  curl -X POST https://<rag-url>/ask -d '{"question": "Is this headphone good for calls?"}'
  ```
- [ ] Fix `item_title` display in rec server response (add `item_title` Feast field + return in payload)

---

## Phase 8 — Demo UI (Gradio)
> Jira: [RHOAIENG-57391](https://redhat.atlassian.net/browse/RHOAIENG-57391)
> Owner: dev team / Amita (demo script)

- [ ] Set env vars in `demo/app.py`: `RECOMMEND_URL`, `SUMMARIZE_URL`, `RAG_URL` → cluster InferenceService URLs
- [ ] Verify Recommendations tab shows readable product names (not raw item IDs)
- [ ] Verify Review Summarizer tab returns coherent summaries
- [ ] Verify Product Q&A tab returns grounded answers from real reviews
- [ ] Do a full dry run: walk all 3 tabs on cluster, measure latency
- [ ] Align with Amita on demo script and talking points (target: ≤5 minutes)
- [ ] Identify any rough edges (slow cold starts, awkward UI copy, missing error handling)

---

## Phase 9 — RAPIDS GPU Spark Variant _(nice to have)_
> Jira: [RHOAIENG-57408](https://redhat.atlassian.net/browse/RHOAIENG-57408)
> Owner: Nikhil (if GPUs are available on cluster)

- [ ] Confirm RAPIDS-compatible GPU nodes available on cluster
- [ ] Validate `infrastructure/openshift/spark-application-rapids.yaml`:
  - Check `spark-rapids-sql` JAR version matches Spark version in image
- [ ] Submit RAPIDS SparkApplication: `make spark-rapids`
- [ ] Verify output matches CPU Spark run (spot-check key feature columns)
- [ ] Note wall-clock time difference for demo narrative (target: ~10× faster claim)

---

## Phase 10 — End-to-End Validation & Recording
> Jira: [RHOAIENG-57385](https://redhat.atlassian.net/browse/RHOAIENG-57385) (infra), [RHOAIENG-57391](https://redhat.atlassian.net/browse/RHOAIENG-57391) (recording)
> Owner: Amita (recording) + full team (sign-off)

- [ ] Run full pipeline via KFP: submit `pipelines/e2e_pipeline.py` and watch DAG in RHOAI UI
- [ ] Confirm pipeline DAG shows all stages: Spark → Feast → Training → Registry → Serving
- [ ] Review Kubeflow Trainer TrainJob view in RHOAI UI (show multi-worker pods)
- [ ] Review RHOAI Model Registry showing both registered models
- [ ] Final walkthrough all 3 Gradio tabs — no errors, responses feel real
- [ ] Record:
  - [ ] KFP pipeline DAG run in RHOAI UI
  - [ ] Kubeflow Trainer live TrainJob view
  - [ ] Gradio UI — all 3 tabs with realistic inputs
  - [ ] (Optional) RHOAI Model Registry promoting model to KServe
- [ ] Review recording with team — flag any issues
- [ ] Amita approves final cut
- [ ] Submit video to Summit organizers before deadline

---

## Phase 11 — Code & Infra Quality Fixes (from review)
> Identified via full codebase review — fix before training/serving runs are attempted

### 🔴 Critical (silent failures / demo breaks)

- [x] **Fix `hf-credentials` secret key mismatch** — Makefile now uses `--from-literal=token=`; secret recreated on cluster
- [x] **Standardize S3 bucket names** — all files now use canonical buckets: `smartshop-raw`, `smartshop-features`, `smartshop-embeddings`, `smartshop-models`
  - Fixed: `spark-application-rapids.yaml`, `e2e_pipeline.py`, `trainjobs.yaml`, `sbatch_llm_finetune.sh`
- [ ] **Fix RAG → LLM URL/contract mismatch** — `serving/rag/server.py` calls `POST /v1/completions` (OpenAI-style) but `serving/llm/server.py` exposes `POST /v1/summarize` with its own JSON schema
  - Fix: point `LLM_URL` at `/v1/summarize`, send `{"product_name": ..., "review_text": ...}`
- [x] **Fix TrainJob DDP topology** — `trainjobs.yaml` now has `numNodes: 1`, `nproc_per_node=4`, resources `4×GPU / 64Gi` per node
- [ ] **Fix rec model dimension mismatch** — serving pads item features with zeros to reach dim=8; training uses 6 item columns → checkpoint mismatch
  - Fix: save `user_feat_dim`/`item_feat_dim` in checkpoint dict at train time; load from checkpoint at serve time; remove padding

### 🟠 High (correctness / performance)

- [ ] **Remove `sys.path.insert` hack in rec server** — `serving/recommendation/server.py` hacks `sys.path` to import `model.py` from training dir
  - Fix: copy `model.py` into `serving/recommendation/` and update Containerfile, or extract to shared `smartshop_ml/` package
- [ ] **Fix SentenceTransformer reload per UDF batch** — `spark/embedding_generation.py` reloads model inside every pandas UDF call
  - Fix: use `mapPartitions` so model loads once per executor partition
- [ ] **Fix N+1 Feast lookups in rec server** — `get_online_features()` called in a loop per candidate item
  - Fix: one batched call `entity_rows=[{"item_id": id} for id in candidates]`
- [ ] **Replace deprecated `@app.on_event("startup")` in all 3 servers** with FastAPI `lifespan` context manager
  - Files: `serving/recommendation/server.py`, `serving/llm/server.py`, `serving/rag/server.py`
- [ ] **Fix non-deterministic `review_id` in embeddings** — `monotonically_increasing_id()` changes on every Spark run, breaking Feast vector lookups
  - Fix: `sha2(concat_ws("_", user_id, parent_asin, timestamp), 256)` as `review_id`

### 🟡 Medium (infra / manifests)

- [ ] **Narrow Spark SA RBAC** — `spark-rbac.yaml` binds ClusterRole `edit` to Spark SA (too broad)
  - Fix: custom Role with only pods, configmaps, services in `smartshop` namespace
- [ ] **Pin Feast git ref to `main`** — `feast-operator.yaml` has `ref: refine-cluster-infra-setup` (feature branch)
  - Fix: update to `main` after merge, or pin to a tag/SHA
- [ ] **Add missing `smartshop-models` PVC manifest** — `inferenceservices.yaml` mounts it but no manifest exists in repo
  - Fix: add PVC manifest under `infrastructure/smartshop/` or switch InferenceServices to S3 `storageUri`
- [ ] **Mark `ETCD_UNSAFE_NO_FSYNC=true` as demo-only** in `infrastructure/milvus/values.yaml` with prominent comment

### 📄 Docs fixes

- [ ] **Fix README `make spark-rapids`** — Makefile target is actually `spark-features-rapids`; add alias or fix README
- [ ] **Update SETUP.md Section 9** — MLflow status callout still says "Broken"; update to "Fixed ✅"
- [ ] **Update SETUP.md Pending table** — Gradio listed as "Not started" but `demo/app.py` exists
- [ ] **Align README architecture diagram** — shows "1 node × 4 GPUs" for DDP; update after TrainJob topology fix

---

## Dependency Order (Critical Path)

```
Phase 0 (Cluster)
  └─► Phase 1 (Images) ──────────────────────────────────────────────┐
  └─► Phase 2 (Spark data)                                           │
        └─► Phase 3 (Feast materialize)                              │
              └─► Phase 4 (Bug fixes) ─► Phase 5 (MLflow, optional)  │
                    └─────────────────► Phase 6 (Training) ◄─────────┘
                                              └─► Phase 7 (KServe)
                                                    └─► Phase 8 (Gradio UI)
                                                          └─► Phase 10 (Record)
Phase 2 ──► Phase 9 (RAPIDS, optional, parallel)
```

---

## Known Blockers / Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Cluster access not granted | Blocks everything | Ping the cluster admin + the cluster admin ASAP |
| Slurm/Slinky not installed | Blocks LLM FSDP job | Confirm with the cluster admin before relying on it; fallback: DDP-only on K8s |
| GPU quota too small for 2-node FSDP | LLM job fails | Negotiate quota or reduce to 1 node × 4 GPUs |
| Mistral-7B download (14GB) slow | Delays LLM training | Pre-pull to MinIO or use a smaller model |
| Feast entity mismatch causes null features | Training silently broken | Fix `parent_asin` → `item_id` in Spark job before materializing |
| `item_feat_dim` bug in rec server | Serving crashes at inference | Fix before any serving test |
| RAPIDS JAR version mismatch | RAPIDS job fails | Verify version pinning in SparkApplication manifest |
