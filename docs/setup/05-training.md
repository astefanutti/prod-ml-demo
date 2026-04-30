## 12. Run PyTorch Training Jobs

### Training Approach

Training uses the **Kubeflow SDK** via Jupyter notebooks executed with **Papermill**.
Both notebooks reference the **built-in `torch-distributed` ClusterTrainingRuntime** —
no custom runtimes are needed.

| Notebook | Trainer | What it trains |
|---|---|---|
| `01_training_rec.ipynb` | `CustomTrainer(func=train_fn)` | Two-Tower DDP recommendation model |
| `02_training_llm.ipynb` | `TransformersTrainer(func=train_llm)` | Mistral-7B QLoRA fine-tune with progression tracking |

The controller automatically injects `PET_*` env vars and wraps execution with `torchrun`.

### Prerequisites

| Check | Why |
|---|---|
| Spark feature engineering job completed | Rec model reads `s3://smartshop-features/interactions/` |
| LLM text preprocessing completed | LLM reads from `s3://smartshop-features/llm_data/` |
| `smartshop-credentials` secret exists | S3 (MinIO) creds for data access |
| `smartshop-mlflow-token` secret exists | MLflow SA token for experiment tracking |
| `hf-credentials` secret exists | HuggingFace token for gated model access (LLM only) |
| `torch-distributed` ClusterTrainingRuntime exists | Built-in runtime from Kubeflow Trainer operator |

```bash
# Verify data is present
oc exec -n smartshop deploy/minio -c minio -- \
  mc ls local/smartshop-features/interactions/ 2>/dev/null | head -5

# Verify runtime exists
oc get clustertrainingruntime torch-distributed

# Verify secrets exist
oc get secret smartshop-credentials smartshop-mlflow-token hf-credentials -n smartshop
```

### 12a — Run training via Notebook CR (recommended)

The e2e Notebook CR runs all notebooks sequentially with Papermill:

```bash
# Create ConfigMap with notebooks
oc create configmap e2e-notebooks \
  --from-file=01_training_rec.ipynb=notebooks/01_training_rec.ipynb \
  --from-file=02_training_llm.ipynb=notebooks/02_training_llm.ipynb \
  --from-file=03_serving.ipynb=notebooks/03_serving.ipynb \
  -n smartshop --dry-run=client -o yaml | oc apply -f -

# Deploy the Notebook CR
set -a && source .env && set +a
envsubst '$NAMESPACE $REC_TRAINER_IMAGE $MINIO_ENDPOINT $AWS_DEFAULT_REGION
  $MLFLOW_TRACKING_URI $S3_FEATURES_BUCKET $S3_MODELS_BUCKET $REC_MAX_ROWS
  $REC_TRAIN_EPOCHS $REC_TRAIN_BATCH_SIZE $REC_TRAIN_NODES $REC_GPUS_PER_NODE
  $LLM_BASE_MODEL $LLM_MAX_FILES $LLM_TRAIN_EPOCHS $LLM_MAX_STEPS
  $LLM_GRAD_ACCUM $LLM_TRAIN_NODES $LLM_GPUS_PER_NODE' \
  < infrastructure/openshift/e2e-notebook.yaml | oc apply -f -

# Monitor
oc logs -f statefulset/smartshop-e2e-test -n smartshop
```

### 12b — Run training via standalone TrainJob YAML (alternative)

For direct YAML-based submission without notebooks (uses custom trainer images):

```bash
set -a && source .env && set +a
envsubst < infrastructure/openshift/trainjobs.yaml | oc apply -f -

# Watch rec training
oc get trainjob smartshop-rec-train -n smartshop -w

# Watch LLM training
oc get trainjob smartshop-llm-finetune -n smartshop -w
```

### 12c — SDK details

The notebooks install `kubeflow[rhai]` from the RHAI PyPI index:

```
pip install "kubeflow[rhai]" --no-cache-dir \
  --index-url https://console.redhat.com/api/pypi/public-rhai/rhoai/3.3/cuda12.9-ubi9/simple/
```

Key SDK features used:
- **`CustomTrainer(func=...)`** — Serializes a Python function into the TrainJob pod
- **`TransformersTrainer(func=...)`** — Same as above + auto-injects progression tracking callback
- Secrets are read from the cluster via `kubernetes.client` and passed as `env` dict
- `/dev/shm` emptyDir is injected via `pod_template_overrides` for NCCL shared memory

### Training parameters (defaults from `.env`)

| Parameter | Rec Model | LLM |
|---|---|---|
| Nodes | 2 | 2 |
| GPUs/node | 4 | 4 |
| Runtime | `torch-distributed` | `torch-distributed` |
| Epochs | 10 | 1 |
| Max steps | — | 1500 |
| Gradient accumulation | — | 2 |
| Batch size | 2048 | 4 |
| Max rows | 5000000 | — |

### Papermill parameter reference

Each notebook cell tagged `parameters` defines defaults that Papermill can override at runtime.
The e2e Notebook CR passes these via `-p KEY VALUE` flags:

**01_training_rec.ipynb** — Recommendation model:
```
-p NAMESPACE smartshop
-p RUNTIME torch-distributed
-p DATA_DIR s3://smartshop-features
-p OUTPUT_DIR s3://smartshop-models/recommendation
-p MAX_ROWS 5000000
-p EPOCHS 10
-p BATCH_SIZE 2048
-p NUM_NODES 2
-p GPUS_PER_NODE 4
-p MINIO_ENDPOINT <minio-url>
-p MLFLOW_TRACKING_URI <mlflow-url>
-p S3_CREDENTIALS_SECRET smartshop-credentials
-p MLFLOW_SECRET smartshop-mlflow-token
-p TIMEOUT_SECONDS 7200
```

**02_training_llm.ipynb** — LLM fine-tuning:
```
-p NAMESPACE smartshop
-p RUNTIME torch-distributed
-p DATA_DIR s3://smartshop-features/llm_data
-p OUTPUT_DIR s3://smartshop-models/llm-adapter
-p BASE_MODEL mistralai/Mistral-7B-Instruct-v0.3
-p MAX_FILES 3
-p EPOCHS 1
-p MAX_STEPS 1500
-p GRADIENT_ACCUMULATION 2
-p NUM_NODES 2
-p GPUS_PER_NODE 4
-p MINIO_ENDPOINT <minio-url>
-p MLFLOW_TRACKING_URI <mlflow-url>
-p S3_CREDENTIALS_SECRET smartshop-credentials
-p MLFLOW_SECRET smartshop-mlflow-token
-p HF_SECRET hf-credentials
-p TIMEOUT_SECONDS 7200
```

---

