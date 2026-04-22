.PHONY: help install \
       data-sample data-full \
       spark-local spark-features-local spark-text-local spark-embeddings-local \
       spark-run spark-features-rapids spark-rapids \
       feast-apply feast-materialize feast-test \
       train-rec train-rec-k8s train-llm train-llm-slurm \
       serve serve-rec serve-llm serve-rag serve-k8s \
       demo \
       setup-builds build-images build-spark build-rec build-llm build-serving \
       build-spark-rapids build-status push-images \
       setup-push-secret setup-secrets deploy clean

# Load .env if present — exports all vars so envsubst and shell commands pick them up.
# Skipped when .env doesn't exist (CI or first-run before setup).
-include .env
export

PYTHON     ?= python
SPARK_SUBMIT ?= spark-submit
KUBECTL    ?= oc
NAMESPACE  ?= smartshop
REGISTRY   ?= $(error REGISTRY not set — define in .env: REGISTRY=quay.io/<your-org>)
INTERNAL_REGISTRY ?= $(error INTERNAL_REGISTRY not set — define in .env)
GIT_REPO   ?= https://github.com/abhijeet-dhumal/prod-ml-demo.git
GIT_BRANCH ?= main

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all local dev dependencies
	pip install -r build/requirements/dev.txt

# ============================================================================
# Data
# ============================================================================

data-sample: ## Download sample dataset (~1M reviews)
	$(PYTHON) data/download.py --mode sample

data-full: ## Download full dataset (~233M reviews, ~49GB)
	$(PYTHON) data/download.py --mode full

# ============================================================================
# Spark Jobs (local mode for development)
# ============================================================================

spark-local: ## Run all Spark jobs locally on sample data
	$(MAKE) spark-features-local
	$(MAKE) spark-text-local
	$(MAKE) spark-embeddings-local

spark-features-local: ## Run feature engineering locally
	$(SPARK_SUBMIT) --master "local[*]" \
		spark/feature_engineering.py \
		--input data/sample/ \
		--output data/processed/

spark-text-local: ## Run text preprocessing locally
	$(SPARK_SUBMIT) --master "local[*]" \
		spark/text_preprocessing.py \
		--input data/sample/ \
		--output data/processed/llm_data/ \
		--max-examples 10000

spark-embeddings-local: ## Run embedding generation locally
	$(SPARK_SUBMIT) --master "local[*]" \
		--packages org.apache.spark:spark-sql_2.12:3.5.0 \
		spark/embedding_generation.py \
		--input data/sample/ \
		--output data/processed/review_embeddings/ \
		--max-reviews 5000

# ============================================================================
# Spark Jobs (OpenShift via Spark Operator)
# ============================================================================

spark-run: ## Submit all Spark jobs to OpenShift (RAPIDS + CPU baseline + text preprocessing)
	bash scripts/apply-all.sh spark

spark-features-rapids: ## Submit RAPIDS GPU-accelerated Spark job only
	envsubst < infrastructure/openshift/spark-application-rapids.yaml | $(KUBECTL) apply -f -

spark-rapids: spark-features-rapids ## Alias for spark-features-rapids

# ============================================================================
# Feast Feature Store
# ============================================================================

feast-apply: ## Register Feast feature views
	cd feast/feature_repo && feast apply

feast-materialize: ## Materialize features to online store
	cd feast/feature_repo && feast materialize-incremental $$(date -u +"%Y-%m-%dT%H:%M:%S")

feast-test: ## Test feature retrieval
	$(PYTHON) feast/test_features.py

# ============================================================================
# Training
# ============================================================================

train-rec: ## Train recommendation model (DDP)
	torchrun --nproc_per_node=1 \
		training/recommendation/train.py \
		--data-dir data/processed \
		--output-dir models/recommendation \
		--epochs 10 \
		--batch-size 1024

train-rec-k8s: ## Submit rec model TrainJob to K8s
	envsubst < infrastructure/openshift/trainjobs.yaml | $(KUBECTL) apply -f -

train-llm: ## Fine-tune Mistral-7B with QLoRA (local, sample)
	$(PYTHON) training/llm/finetune.py \
		--data-dir data/processed/llm_data \
		--output-dir models/llm-adapter \
		--sample

train-llm-slurm: ## Submit LLM fine-tuning to Slurm
	sbatch infrastructure/slurm/sbatch_llm_finetune.sh

# ============================================================================
# Serving (local development)
# ============================================================================

serve: ## Start all serving endpoints locally
	$(MAKE) -j3 serve-rec serve-llm serve-rag

serve-rec: ## Start recommendation server
	uvicorn serving.recommendation.server:app --host 0.0.0.0 --port 8000

serve-llm: ## Start LLM summarization server
	$(PYTHON) serving/llm/server.py --port 8001

serve-rag: ## Start RAG Q&A server
	uvicorn serving.rag.server:app --host 0.0.0.0 --port 8002

serve-k8s: ## Deploy KServe InferenceServices
	envsubst < infrastructure/openshift/inferenceservices.yaml | $(KUBECTL) apply -f -

# ============================================================================
# Demo
# ============================================================================

demo: ## Launch Gradio demo UI
	$(PYTHON) demo/app.py

# ============================================================================
# Container Images — built on-cluster via OpenShift BuildConfig + ImageStream
# ============================================================================

setup-builds: ## Create ImageStreams and BuildConfigs on the cluster
	envsubst < infrastructure/openshift/imagestreams.yaml  | $(KUBECTL) apply -f -
	envsubst < infrastructure/openshift/buildconfigs.yaml  | $(KUBECTL) apply -f -

build-images: ## Trigger all image builds on the cluster (requires setup-builds first)
	$(KUBECTL) start-build spark-jobs         -n $(NAMESPACE) --follow
	$(KUBECTL) start-build spark-jobs-rapids  -n $(NAMESPACE) --follow
	$(KUBECTL) start-build feast-spark-server -n $(NAMESPACE) --follow
	$(KUBECTL) start-build rec-trainer        -n $(NAMESPACE) --follow
	$(KUBECTL) start-build llm-trainer        -n $(NAMESPACE) --follow
	$(KUBECTL) start-build rec-server         -n $(NAMESPACE) --follow

build-spark:     ## Build only spark-jobs image
	$(KUBECTL) start-build spark-jobs  -n $(NAMESPACE) --follow

build-rec:       ## Build only rec-trainer image
	$(KUBECTL) start-build rec-trainer -n $(NAMESPACE) --follow

build-llm:       ## Build only llm-trainer image
	$(KUBECTL) start-build llm-trainer -n $(NAMESPACE) --follow

build-serving:   ## Build only rec-server (serving) image
	$(KUBECTL) start-build rec-server  -n $(NAMESPACE) --follow

build-spark-rapids: ## Build only spark-jobs-rapids image (RAPIDS JAR; ~850 MiB download at build time)
	$(KUBECTL) start-build spark-jobs-rapids -n $(NAMESPACE) --follow

build-status:    ## Show status of all builds
	$(KUBECTL) get builds -n $(NAMESPACE)

push-images: ## Mirror all images from internal registry → ${REGISTRY}
	$(KUBECTL) image mirror \
	  "$(INTERNAL_REGISTRY)/$(NAMESPACE)/spark-jobs:latest=$(REGISTRY)/smartshop-spark-jobs:latest" \
	  "$(INTERNAL_REGISTRY)/$(NAMESPACE)/spark-jobs-rapids:latest=$(REGISTRY)/smartshop-spark-jobs-rapids:latest" \
	  "$(INTERNAL_REGISTRY)/$(NAMESPACE)/rec-trainer:latest=$(REGISTRY)/smartshop-rec-trainer:latest" \
	  "$(INTERNAL_REGISTRY)/$(NAMESPACE)/llm-trainer:latest=$(REGISTRY)/smartshop-llm-trainer:latest" \
	  "$(INTERNAL_REGISTRY)/$(NAMESPACE)/rec-server:latest=$(REGISTRY)/smartshop-rec-server:latest" \
	  --insecure=true -a ~/.config/containers/auth.json

# ============================================================================
# Infrastructure
# ============================================================================

setup-push-secret: ## Create quay.io push secret used by BuildConfigs (run once before setup-builds)
	@if [ -z "$(QUAY_USER)" ] || [ -z "$(QUAY_TOKEN)" ]; then \
	  echo "ERROR: pass QUAY_USER and QUAY_TOKEN: make setup-push-secret QUAY_USER=abdhumal QUAY_TOKEN=<robot-token>"; exit 1; \
	fi
	$(KUBECTL) create secret docker-registry $(QUAY_PUSH_SECRET) \
	  --docker-server=quay.io \
	  --docker-username=$(QUAY_USER) \
	  --docker-password=$(QUAY_TOKEN) \
	  -n $(NAMESPACE) --dry-run=client -o yaml | $(KUBECTL) apply -f -
	@echo "==> $(QUAY_PUSH_SECRET) secret ready in $(NAMESPACE)"

setup-secrets: ## Create/update all Kubernetes secrets from .env values
	@if [ ! -f .env ]; then \
	  echo "ERROR: .env not found. Copy .env.example → .env and fill in values."; exit 1; \
	fi
	@echo "==> smartshop namespace secrets..."
	envsubst < infrastructure/smartshop/credentials.yaml  | $(KUBECTL) apply -f -
	envsubst < infrastructure/redis/redis.yaml             | $(KUBECTL) apply -f - --prune=false
	envsubst < infrastructure/mlflow/postgres.yaml        | $(KUBECTL) apply -f -
	# NOTE: Feast FeatureStore CR (Spark backend) is applied by scripts/apply-all.sh phase_feast
	# Do NOT apply infrastructure/feast/feast-operator.yaml (dask backend, superseded)
	@echo "==> redhat-ods-applications secrets..."
	$(KUBECTL) create secret generic mlflow-s3-credentials \
	  --from-literal=AWS_ACCESS_KEY_ID=$(MINIO_ACCESS_KEY) \
	  --from-literal=AWS_SECRET_ACCESS_KEY=$(MINIO_SECRET_KEY) \
	  --from-literal=MLFLOW_S3_ENDPOINT_URL=$(MINIO_ENDPOINT) \
	  --from-literal=AWS_DEFAULT_REGION=$(AWS_DEFAULT_REGION) \
	  -n redhat-ods-applications --dry-run=client -o yaml | $(KUBECTL) apply -f -
	$(KUBECTL) create secret generic mlflow-postgres-secret \
	  --from-literal=uri="postgresql+psycopg2://$(PG_USER):$(PG_PASSWORD)@$(PG_CLUSTERIP):5432/$(PG_DATABASE)?sslmode=disable" \
	  -n redhat-ods-applications --dry-run=client -o yaml | $(KUBECTL) apply -f -
	$(KUBECTL) create secret generic minio-root-user \
	  --from-literal=MINIO_ROOT_USER=$(MINIO_ACCESS_KEY) \
	  --from-literal=MINIO_ROOT_PASSWORD=$(MINIO_SECRET_KEY) \
	  -n smartshop --dry-run=client -o yaml | $(KUBECTL) apply -f -
	$(KUBECTL) create secret generic hf-credentials \
	  --from-literal=token=$(HF_TOKEN) \
	  -n smartshop --dry-run=client -o yaml | $(KUBECTL) apply -f -
	@echo "==> Secrets synced."

deploy: setup-secrets ## Deploy all infrastructure to OpenShift (see docs/SETUP.md for full guide)
	$(KUBECTL) apply -f infrastructure/smartshop/namespace.yaml
	$(KUBECTL) apply -f infrastructure/smartshop/shared-storage.yaml
	$(KUBECTL) apply -f infrastructure/smartshop/spark-rbac.yaml

clean: ## Clean generated data and models
	rm -rf data/processed/ data/sample/*.parquet models/
	cd feast/feature_repo && rm -f data/registry.db data/online_store.db
