# Production ML at Scale: SmartShop AI Architecture

## Overview

SmartShop AI is a production e-commerce platform that demonstrates ML at scale using
PyTorch Distributed, Kubeflow Trainer, Apache Spark, Feast Feature Store, and Slurm
on Red Hat OpenShift AI.

The platform:
- **Recommends products** using a PyTorch two-tower model trained on purchase/rating history
- **Summarizes reviews** using a fine-tuned Mistral-7B
- **Answers product questions** via RAG over review embeddings stored in Feast's vector store

## Architecture Diagram

```
                        RED HAT OPENSHIFT AI PLATFORM
 ===========================================================================================

 STAGE 1: DATA INGESTION
 ┌────────────────────┐
 │ Amazon Reviews     │──── Raw Parquet ───► Object Storage (S3/MinIO)
 │ Dataset (49GB)     │
 └────────────────────┘

 STAGE 2: SPARK PREPROCESSING (Kubeflow Spark Operator)
 ┌──────────────────────────────────────────────────────────────────────┐
 │  Job A: Structured Feature Engineering                               │
 │  - User features (avg_rating, review_count, category_prefs)          │
 │  - Item features (avg_rating, price_bucket, review_volume)           │
 │  - Interaction features (user-item co-occurrence)                    │
 │  Output ──► Feast Offline Store                                      │
 │                                                                      │
 │  Job B: Text Preprocessing for LLM Fine-Tuning                       │
 │  - Cleaning, dedup, instruction-format dataset creation              │
 │  - Train/val/test split at scale                                     │
 │  Output ──► Object Storage (JSONL)                                   │
 │                                                                      │
 │  Job C: Embedding Generation (sentence-transformer via Spark UDF)    │
 │  Output ──► Feast Vector Store                                       │
 └──────────────────────────────────────────────────────────────────────┘

 STAGE 3: FEAST FEATURE STORE
 ┌──────────────────────────────────────────────────────────────────────┐
 │  Offline Store (S3/Parquet)  │ Training: get_historical_features()   │
 │  Online Store (Redis)        │ Serving:  get_online_features()       │
 │  Vector Store (Milvus)       │ RAG:      similarity_search()         │
 │                                                                      │
 │  feast apply → register feature views                                │
 │  feast materialize → push offline → online                           │
 └──────────────────────────────────────────────────────────────────────┘

 STAGE 4: DISTRIBUTED TRAINING (Kubeflow Trainer, parallel TrainJobs)
 ┌─────────────────────────────────┐  ┌─────────────────────────────────┐
 │ 4a: Recommendation Model        │  │ 4b: LLM Fine-Tuning             │
 │ - Two-Tower Neural CF (PyTorch) │  │ - Mistral-7B + QLoRA            │
 │ - DDP, 4 workers on K8s         │  │ - FSDP, multi-node multi-GPU    │
 │ - Data from Feast offline store │  │ - Dispatched to Slurm cluster   │
 │ - Kubeflow Trainer TrainJob     │  │   via Project Slinky / Kueue    │
 │ - TrainingRuntime: pytorch-ddp  │  │ - TrainJob + ClusterTraining-   │
 │ Output ──► Model Registry       │  │   Runtime: llm-fsdp-slurm       │
 └─────────────────────────────────┘  │ Output ──► Model Registry       │
                                      └─────────────────────────────────┘

 STAGE 5: MODEL REGISTRY (Kubeflow Model Registry)
 ┌──────────────────────────────────────────────────────────────────────┐
 │  smartshop-rec-model v1.0     │  smartshop-llm-adapter v1.0          │
 │  (metrics, lineage, artifact) │  (base model, LoRA weights, metrics) │
 └──────────────────────────────────────────────────────────────────────┘

 STAGE 6: SERVING (KServe on OpenShift)
 ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐
 │ /recommend           │  │ /summarize           │  │ /ask (RAG)       │
 │ Feast online lookup  │  │ LLM inference        │  │ Feast vector     │
 │ → model inference    │  │ via vLLM             │  │ search → LLM     │
 │ → top-N items        │  │ → summary+sentiment  │  │ → answer         │
 └──────────────────────┘  └──────────────────────┘  └──────────────────┘
```

## Why Each Component Earns Its Place

| Component | Role | Why It's Needed |
|---|---|---|
| **Spark** | ETL, feature eng, embeddings | 233M reviews at 49GB can't be processed in pandas |
| **Feast** | Feature store + vector store | Same features for train/serve (no skew); vector store for RAG |
| **Kubeflow Trainer** | Distributed training orchestration | Two TrainJobs: DDP rec model + FSDP LLM fine-tuning |
| **Slurm** | HPC GPU scheduling for LLM | Multi-node FSDP needs gang scheduling + topology-aware placement |
| **KServe** | Model serving | Three endpoints with autoscaling |
| **GenAI/LLM** | Review summarization + RAG Q&A | The data IS text -- LLM fine-tuning is the natural thing to do |

## Dataset Strategy

- **Primary**: Amazon Reviews 2023 (subset: Electronics + Books + Home categories, ~30-50GB)
- **Source**: huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- **Sample**: ~1M reviews bundled for quick local testing
- **Pre-trained artifacts**: Downloadable to skip training and jump to serving

## Slurm Integration

The LLM fine-tuning job (7B model, FSDP across 2 nodes x 4 GPUs) is the natural Slurm workload:
- **Gang scheduling**: All 8 GPUs must be available simultaneously
- **Topology awareness**: Slurm places workers on nodes with optimal interconnects
- **HPC reuse narrative**: "Use your existing GPU cluster from OpenShift AI"

Integration path: Kubeflow Trainer TrainJob -> ClusterTrainingRuntime -> Kueue AppWrapper -> Slurm sbatch -> GPU nodes -> artifacts back to S3 -> Model Registry

## Optional: GPU-Accelerated Spark

The Spark preprocessing jobs (Stage 2) can optionally run on GPUs using the [RAPIDS Accelerator for Apache Spark](https://nvidia.github.io/spark-rapids/). This is a drop-in plugin -- the same PySpark code runs unchanged, with DataFrame operations offloaded to GPU. The feature engineering job (Job A) benefits most, with 10-30x speedups on its heavy groupBy/join/aggregation workload.

See [docs/rapids.md](rapids.md) for per-job analysis, infrastructure requirements, and configuration.

## Key References

- [Red Hat AI Quickstart Product Recommender](https://developers.redhat.com/articles/2026/01/20/ai-quickstart-product-recommender-openshift-ai)
- [Kubeflow Fraud Detection E2E](https://blog.kubeflow.org/fraud-detection-e2e/)
- [Fine-tune RAG with Feast + Kubeflow Trainer](https://developers.redhat.com/articles/2025/12/17/fine-tune-rag-model-feast-kubeflow-trainer)
- [OpenShift AI 3.3 Fine-Tuning Pipelines](https://developers.redhat.com/articles/2026/02/26/fine-tune-ai-pipelines-red-hat-openshift-ai)
- [Feast GenAI Integration](https://docs.feast.dev/getting-started/genai)
- [Sovereign AI with Kubeflow Trainer + Feast](https://redhat.com/en/blog/sovereign-ai-architecture-scaling-distributed-training-kubeflow-trainer-and-feast-red-hat-openshift-ai)
- [Amazon Reviews 2023 Dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
