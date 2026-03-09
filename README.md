# SmartShop AI: Production ML at Scale

Reference implementation for the Red Hat Summit 2026 presentation on production ML at scale.

Demonstrates **PyTorch Distributed**, **Kubeflow Trainer**, **Apache Spark**, **Feast Feature Store**, and **Slurm** on **Red Hat OpenShift AI** through a realistic e-commerce use case.

## What It Does

SmartShop AI is a production e-commerce platform that:

1. **Recommends products** using a PyTorch two-tower model trained on purchase/rating history (DDP on K8s)
2. **Summarizes reviews** using a fine-tuned Mistral-7B (QLoRA + FSDP on Slurm)
3. **Answers product questions** via RAG over review embeddings stored in Feast's vector store

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full architecture diagram and component rationale.

```
Spark Preprocessing → Feast Feature Store → Distributed Training → KServe Endpoints → Demo UI
     (49GB)            (offline+online+vector)   (DDP + FSDP)       (3 services)      (Gradio)
```

## Quick Start (Local / Sample Data)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download sample dataset (~1M reviews, ~500MB)
make data-sample

# 3. Run Spark preprocessing locally
make spark-local

# 4. Set up Feast feature store
make feast-apply

# 5. Train recommendation model
make train-rec

# 6. (Optional) Fine-tune LLM on sample data
make train-llm

# 7. Start serving endpoints
make serve

# 8. Launch demo UI
make demo
```

## Full-Scale Run (OpenShift AI)

```bash
# 1. Deploy infrastructure (MinIO, Redis, operators)
make deploy

# 2. Build and push container images
make build-images push-images

# 3. Download full dataset (~49GB)
make data-full

# 4. Submit Spark jobs to cluster
make spark-run

# 5. Materialize Feast features
make feast-materialize

# 6. Submit training jobs
make train-rec-k8s    # DDP on K8s
make train-llm-slurm  # FSDP on Slurm

# 7. Deploy serving endpoints
make serve-k8s
```

## Project Structure

```
summit26/
├── docs/                    # Architecture documentation
├── data/                    # Download scripts, sample data
├── spark/                   # Spark jobs (feature eng, text preprocessing, embeddings)
├── feast/feature_repo/      # Feast feature view definitions
├── training/
│   ├── recommendation/      # Two-Tower model + DDP training script
│   └── llm/                 # QLoRA fine-tuning + FSDP config
├── serving/
│   ├── recommendation/      # KServe recommendation server
│   ├── llm/                 # vLLM-based summarization server
│   └── rag/                 # RAG Q&A pipeline (Feast vector search + LLM)
├── pipelines/               # Kubeflow Pipeline (e2e orchestration)
├── demo/                    # Gradio demo UI
├── infrastructure/
│   ├── openshift/           # K8s manifests (SparkApplication, TrainJob, InferenceService)
│   └── slurm/               # Slurm sbatch scripts, Kueue config
├── Containerfile.*          # Container images for each component
├── Makefile                 # All build/run/deploy targets
└── requirements*.txt        # Python dependencies
```

## Key Components

| Component | Role | Why It Earns Its Place |
|---|---|---|
| **Apache Spark** | ETL, feature engineering, embeddings | 233M reviews at 49GB can't be processed in pandas |
| **Feast** | Feature store + vector store | Same features for train/serve (no skew); vector store for RAG |
| **Kubeflow Trainer** | Distributed training orchestration | Two TrainJobs: DDP rec model + FSDP LLM |
| **Slurm** | HPC GPU scheduling for LLM | Multi-node FSDP needs gang scheduling + topology awareness |
| **KServe** | Model serving | Three autoscaling endpoints |
| **Mistral-7B** | Review summarization + RAG Q&A | The data IS text; LLM is the natural fit |

## Dataset

- **Amazon Reviews 2023** from McAuley Lab (HuggingFace)
- Categories: Electronics, Books, Home & Kitchen
- Sample: ~1M reviews for local testing
- Full: ~50GB across 3 categories

## Extends

This project extends the [Red Hat AI Quickstart for Product Recommender](https://developers.redhat.com/articles/2026/01/20/ai-quickstart-product-recommender-openshift-ai) by adding:
- Large-scale Spark preprocessing
- Distributed training with Kubeflow Trainer
- LLM fine-tuning with FSDP on Slurm
- RAG with Feast vector store
