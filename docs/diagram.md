# SmartShop AI -- Architecture Diagram

```mermaid
flowchart TB
    %% ── Styles ──────────────────────────────────────────────────────────
    classDef stage fill:#1a1a2e,color:#fff,stroke:#e94560,stroke-width:2px,font-weight:bold
    classDef data fill:#0f3460,color:#fff,stroke:#16213e
    classDef spark fill:#e94560,color:#fff,stroke:#c81e4e
    classDef feast fill:#533483,color:#fff,stroke:#3d2566
    classDef train fill:#f08a5d,color:#fff,stroke:#c96d48
    classDef serve fill:#00b894,color:#fff,stroke:#00876a
    classDef registry fill:#6c5ce7,color:#fff,stroke:#5240c4
    classDef infra fill:#2d3436,color:#dfe6e9,stroke:#636e72,stroke-dasharray:5
    classDef rapids fill:#76b900,color:#fff,stroke:#5a8f00

    %% ── Stage 1: Data Ingestion ─────────────────────────────────────────
    subgraph S1["STAGE 1: DATA INGESTION"]
        direction LR
        HF["Amazon Reviews 2023\n233M reviews, 49GB\n(HuggingFace)"]
        S3raw["Object Storage\nS3 / MinIO\n(Raw Parquet)"]
        HF -->|"download.py"| S3raw
    end

    %% ── Stage 2: Spark Preprocessing ────────────────────────────────────
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

    %% ── Stage 3: Feast Feature Store ────────────────────────────────────
    subgraph S3["STAGE 3: FEAST FEATURE STORE"]
        direction LR
        Offline["Offline Store\nS3 / Parquet"]
        Online["Online Store\nRedis"]
        Vector["Vector Store\nMilvus"]

        Offline -->|"feast materialize"| Online
    end

    %% ── Stage 4: Distributed Training ───────────────────────────────────
    subgraph S4["STAGE 4: DISTRIBUTED TRAINING"]
        direction LR

        subgraph RecTrain["4a: Recommendation Model"]
            Rec["Two-Tower Neural CF\nPyTorch DDP\n4 workers on K8s"]
        end

        subgraph LLMTrain["4b: LLM Fine-Tuning"]
            LLM["Mistral-7B + QLoRA\nFSDP multi-node\n2 nodes x 4 GPUs"]
        end

        subgraph SlurmCluster["Slurm Cluster (via Kueue / Slinky)"]
            Slurm["Gang scheduling\nTopology-aware placement\nHPC GPU nodes"]
        end
    end

    %% ── Stage 5: Model Registry ─────────────────────────────────────────
    subgraph S5["STAGE 5: MODEL REGISTRY"]
        direction LR
        RecModel["smartshop-rec-model\nv1.0"]
        LLMModel["smartshop-llm-adapter\nv1.0 (LoRA weights)"]
    end

    %% ── Stage 6: Serving ────────────────────────────────────────────────
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

    %% ── Demo UI ─────────────────────────────────────────────────────────
    Gradio["Gradio Demo UI"]

    %% ── Data Flow ───────────────────────────────────────────────────────
    S3raw --> JobA
    S3raw --> JobB
    S3raw --> JobC

    JobA -.->|"Optional GPU path"| RA

    A1 -->|"Parquet"| Offline
    B1 -->|"JSONL"| S3llm["Object Storage\n(LLM training data)"]
    C1 -->|"Vectors"| Vector

    Offline -->|"get_historical_features()"| Rec
    S3llm --> LLM
    LLM --> Slurm

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

    %% ── Apply styles ────────────────────────────────────────────────────
    class S1 stage
    class S2 stage
    class S3 stage
    class S4 stage
    class S5 stage
    class S6 stage

    class HF,S3raw,S3llm data
    class JobA,JobB,JobC,A1,B1,C1,SparkOp spark
    class Offline,Online,Vector feast
    class RecTrain,Rec,LLMTrain,LLM train
    class SlurmCluster,Slurm infra
    class RecModel,LLMModel registry
    class EP1,EP2,EP3,Recommend,Summarize,Ask,Gradio serve
    class RapidsOpt,RA rapids
```

## Infrastructure & Orchestration

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
            CPU["CPU Nodes\nSpark drivers\nRec training (DDP)"]
            GPU["GPU Nodes\nLLM training (FSDP)\nLLM serving (vLLM)\nSpark RAPIDS (optional)"]
        end

        subgraph External["External Integration"]
            SlurmExt["Slurm HPC Cluster\n(via Slinky)"]
        end
    end

    SparkOp2 --> CPU
    SparkOp2 -.->|"RAPIDS"| GPU
    Trainer --> CPU
    Trainer --> Kueue --> SlurmExt
    KServe2 --> GPU

    class OCP platform
    class SparkOp2,Trainer,KServe2,Kueue component
    class MinIO,Redis2,Milvus2,ModelReg component
    class CPU component
    class GPU gpu
    class SlurmExt component
```

## Data Flow Summary

| Stage | Input | Processing | Output | Accelerator |
|---|---|---|---|---|
| 1. Ingest | HuggingFace dataset | `download.py` | S3 raw Parquet | -- |
| 2a. Features | Raw reviews + metadata | Spark groupBy/agg/join | Feast offline store | CPU or GPU (RAPIDS) |
| 2b. Text prep | Raw reviews | Spark filter/dedup + UDFs | JSONL on S3 | CPU |
| 2c. Embeddings | Raw reviews | sentence-transformer UDF | Feast vector store | GPU (model) |
| 3. Feast | Offline Parquet | `feast materialize` | Redis online store | -- |
| 4a. Rec model | Feast features | PyTorch DDP (4 workers) | Model Registry | GPU |
| 4b. LLM | JSONL training data | FSDP QLoRA (8 GPUs) | Model Registry | GPU |
| 6. Serving | User requests | KServe + vLLM | JSON responses | GPU |
