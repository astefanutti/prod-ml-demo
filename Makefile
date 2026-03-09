.PHONY: help data-sample data-full spark-run spark-features-rapids spark-local \
       feast-apply feast-materialize \
       train-rec train-llm serve serve-rec serve-llm serve-rag demo \
       build-images build-image-spark-rapids push-images deploy clean

PYTHON ?= python
SPARK_SUBMIT ?= spark-submit
KUBECTL ?= kubectl
NAMESPACE ?= smartshop
REGISTRY ?= quay.io/smartshop

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

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

spark-run: ## Submit all Spark jobs to OpenShift
	$(KUBECTL) apply -f infrastructure/openshift/spark-application.yaml -n $(NAMESPACE)

spark-features-rapids: ## Submit RAPIDS GPU-accelerated feature engineering job
	$(KUBECTL) apply -f infrastructure/openshift/spark-application-rapids.yaml -n $(NAMESPACE)

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
	$(KUBECTL) apply -f infrastructure/openshift/trainjobs.yaml -n $(NAMESPACE)

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
	$(KUBECTL) apply -f infrastructure/openshift/inferenceservices.yaml -n $(NAMESPACE)

# ============================================================================
# Demo
# ============================================================================

demo: ## Launch Gradio demo UI
	$(PYTHON) demo/app.py

# ============================================================================
# Container Images
# ============================================================================

build-images: ## Build all container images
	podman build -f Containerfile.spark -t $(REGISTRY)/spark-jobs:latest .
	podman build -f Containerfile.rec-trainer -t $(REGISTRY)/rec-trainer:latest .
	podman build -f Containerfile.llm-trainer -t $(REGISTRY)/llm-trainer:latest .
	podman build -f Containerfile.serving -t $(REGISTRY)/rec-server:latest .

build-image-spark-rapids: ## Build RAPIDS GPU-accelerated Spark image
	podman build -f Containerfile.spark-rapids -t $(REGISTRY)/spark-jobs-rapids:latest .

push-images: ## Push images to registry
	podman push $(REGISTRY)/spark-jobs:latest
	podman push $(REGISTRY)/rec-trainer:latest
	podman push $(REGISTRY)/llm-trainer:latest
	podman push $(REGISTRY)/rec-server:latest

# ============================================================================
# Infrastructure
# ============================================================================

deploy: ## Deploy all infrastructure to OpenShift
	$(KUBECTL) apply -f infrastructure/openshift/namespace.yaml
	$(KUBECTL) apply -f infrastructure/slurm/kueue-config.yaml

clean: ## Clean generated data and models
	rm -rf data/processed/ data/sample/*.parquet models/
	cd feast/feature_repo && rm -f data/registry.db data/online_store.db
