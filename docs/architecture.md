# SmartShop AI — Architecture

## Overview

SmartShop AI demonstrates production ML at scale on Red Hat OpenShift AI: distributed
feature engineering with Spark, feature serving with Feast, distributed training with
Kubeflow Trainer (DDP + QLoRA/FSDP on K8s), and multi-model serving with KServe.

- **Recommends products** — Two-Tower neural CF model trained on 29.6 GB of reviews
- **Summarizes reviews** — Mistral-7B fine-tuned with QLoRA via FSDP
- **Answers product questions** — RAG over 384-dim review embeddings in Milvus

---

## Pipeline Diagram

```mermaid
flowchart TB
    classDef stage fill:#1a1a2e,color:#fff,stroke:#e94560,stroke-width:2px,font-weight:bold
    classDef data fill:#0f3460,color:#fff,stroke:#16213e
    classDef spark fill:#e94560,color:#fff,stroke:#c81e4e
    classDef feast fill:#533483,color:#fff,stroke:#3d2566
    classDef train fill:#f08a5d,color:#fff,stroke:#c96d48
    classDef serve fill:#00b894,color:#fff,stroke:#00876a
    classDef registry fill:#6c5ce7,color:#fff,stroke:#5240c4
    classDef infra fill:#2d3436,color:#dfe6e9,stroke:#636e72,stroke-dasharray:5
    classDef rapids fill:#76b900,color:#fff,stroke:#5a8f00

    subgraph S1["STAGE 1: DATA INGESTION"]
        direction LR
        HF["Amazon Reviews 2023\n233M reviews, 49GB\n(HuggingFace)"]
        S3raw["Object Storage\nS3 / MinIO\n(Raw Parquet)"]
        HF -->|"download.py"| S3raw
    end

    subgraph S2["STAGE 2: SPARK PREPROCESSING"]
        direction TB
        subgraph SparkOp["Kubeflow Spark Operator"]
            direction LR
            subgraph JobA["Job A: Feature Engineering"]
                A1["groupBy / agg / join\nUser features\nItem features\nInteraction features"]
            end
            subgraph JobB["Job B: Text Preprocessing"]
                B1["Clean, dedup, regex\nBuild instruction prompts\nTrain / val / test split"]
            end
            subgraph JobC["Job C: Embedding Generation"]
                C1["sentence-transformer\nvia pandas UDF\n384-dim vectors"]
            end
        end
        subgraph RapidsOpt["Optional: RAPIDS GPU Acceleration"]
            direction LR
            RA["Job A on GPU\n10-30x speedup\nspark-rapids plugin"]
        end
    end

    subgraph S3["STAGE 3: FEAST FEATURE STORE"]
        direction LR
        Offline["Offline Store\nS3 / Parquet"]
        Online["Online Store\nRedis"]
        Vector["Vector Store\nMilvus"]
        Offline -->|"feast materialize"| Online
    end

    subgraph S4["STAGE 4: DISTRIBUTED TRAINING"]
        direction LR
        subgraph RecTrain["4a: Recommendation Model"]
            Rec["Two-Tower Neural CF\nPyTorch DDP\n2 nodes x 2 GPUs"]
        end
        subgraph LLMTrain["4b: LLM Fine-Tuning"]
            LLM["Mistral-7B + QLoRA\nDDP on K8s\n2 nodes x 2 GPUs"]
        end
    end

    subgraph S5["STAGE 5: MODEL REGISTRY"]
        direction LR
        RecModel["smartshop-rec-model\nv1.0"]
        LLMModel["smartshop-llm-adapter\nv1.0 (LoRA weights)"]
    end

    subgraph S6["STAGE 6: SERVING (KServe)"]
        direction LR
        subgraph EP1["/recommend"]
            Recommend["Feast online lookup\nModel inference\nTop-N items"]
        end
        subgraph EP2["/summarize"]
            Summarize["LLM inference\nvia vLLM\nSummary + sentiment"]
        end
        subgraph EP3["/ask"]
            Ask["Feast vector search\nRetrieve context\nLLM answer (RAG)"]
        end
    end

    Gradio["Gradio Demo UI"]

    S3raw --> JobA
    S3raw --> JobB
    S3raw --> JobC
    JobA -.->|"Optional GPU path"| RA
    A1 -->|"Parquet"| Offline
    B1 -->|"JSONL"| S3llm["Object Storage\n(LLM training data)"]
    C1 -->|"Vectors"| Vector
    Offline -->|"S3 parquet (direct read)"| Rec
    S3llm --> LLM
    Rec --> RecModel
    LLM --> LLMModel
    RecModel --> Recommend
    LLMModel --> Summarize
    LLMModel --> Ask
    Online -->|"get_online_features()"| Recommend
    Vector -->|"similarity_search()"| Ask
    Recommend --> Gradio
    Summarize --> Gradio
    Ask --> Gradio

    class S1,S2,S3,S4,S5,S6 stage
    class HF,S3raw,S3llm data
    class JobA,JobB,JobC,A1,B1,C1,SparkOp spark
    class Offline,Online,Vector feast
    class RecTrain,Rec,LLMTrain,LLM train
    class RecModel,LLMModel registry
    class EP1,EP2,EP3,Recommend,Summarize,Ask,Gradio serve
    class RapidsOpt,RA rapids
```

---

## Infrastructure Diagram

```mermaid
flowchart LR
    classDef platform fill:#ee0000,color:#fff,stroke:#b30000,font-weight:bold
    classDef component fill:#2d3436,color:#dfe6e9,stroke:#636e72
    classDef gpu fill:#76b900,color:#fff,stroke:#5a8f00

    subgraph OCP["Red Hat OpenShift AI"]
        direction TB
        subgraph Operators["Operators"]
            SparkOp2["Kubeflow\nSpark Operator"]
            Trainer["Kubeflow\nTrainer"]
            KServe2["KServe"]
            Kueue["Kueue"]
        end
        subgraph Storage["Storage & State"]
            MinIO["MinIO\n(S3)"]
            Redis2["Redis\n(Online Store)"]
            Milvus2["Milvus\n(Vector Store)"]
            ModelReg["Model\nRegistry"]
        end
        subgraph Compute["Compute"]
            CPU["CPU Nodes\nSpark drivers · Rec DDP"]
            GPU["GPU Nodes\nLLM FSDP · vLLM · RAPIDS"]
        end
    end

    SparkOp2 --> CPU
    SparkOp2 -.->|"RAPIDS"| GPU
    Trainer --> CPU
    Trainer --> GPU
    KServe2 --> GPU

    class OCP platform
    class SparkOp2,Trainer,KServe2,Kueue,MinIO,Redis2,Milvus2,ModelReg,CPU component
    class GPU gpu
```

---

## Data Flow Summary

| Stage | Input | Processing | Output | Accelerator |
|---|---|---|---|---|
| 1. Ingest | HuggingFace dataset | `download.py` | S3 raw Parquet | — |
| 2a. Features | Raw reviews | Spark groupBy/agg/join | Feast offline store (Parquet) | CPU or GPU (RAPIDS) |
| 2b. Text prep | Raw reviews | Spark filter/dedup + UDFs | JSONL on S3 | CPU |
| 2c. Embeddings | Raw reviews | sentence-transformer UDF | Feast vector store (Milvus) | CPU (UDF) |
| 3. Feast | Offline Parquet | `feast materialize` | Redis online store | — |
| 4a. Rec model | S3 interactions | PyTorch DDP (2 nodes x 2 GPUs) | Model Registry | GPU |
| 4b. LLM | JSONL training data | QLoRA DDP (2 nodes x 2 GPUs) | Model Registry | GPU |
| 6. Serving | User requests | KServe + vLLM | JSON responses | GPU |

---

## Why Each Component

| Component | Role | Why It's Needed |
|---|---|---|
| **Spark** | ETL, feature eng, embeddings | 29.6 GB of reviews can't be processed in pandas |
| **RAPIDS** | GPU-accelerated Spark | 4% speedup on feature engineering — same PySpark code, zero changes; see `feast/BFV-DESIGN.md §5` |
| **Feast** | Feature store + vector store | Train-serve skew prevention; vector store for RAG |
| **Kubeflow Trainer** | Distributed training orchestration | DDP rec model + QLoRA LLM fine-tuning via TrainJob CRD |
| **KServe** | Model serving | Three autoscaling endpoints |
| **Milvus** | Vector similarity search | RAG retrieval over 5M review embeddings |

---

## Key References

- [Feast GenAI Integration](https://docs.feast.dev/getting-started/genai)
- [Sovereign AI with Kubeflow Trainer + Feast](https://redhat.com/en/blog/sovereign-ai-architecture-scaling-distributed-training-kubeflow-trainer-and-feast-red-hat-openshift-ai)
- [Fine-tune RAG with Feast + Kubeflow Trainer](https://developers.redhat.com/articles/2025/12/17/fine-tune-rag-model-feast-kubeflow-trainer)
- [Red Hat AI Quickstart Product Recommender](https://developers.redhat.com/articles/2026/01/20/ai-quickstart-product-recommender-openshift-ai)
- [Amazon Reviews 2023 Dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
