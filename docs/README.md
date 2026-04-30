# SmartShop AI — Documentation

Red Hat Summit 2026 demo: distributed ML at scale on Red Hat OpenShift AI.

---

## Navigation

### Start here
| Doc | What it covers |
|---|---|
| [Architecture](architecture.md) | Pipeline overview, Mermaid diagrams, data flow table, component rationale |
| [Demo Script](demo/SCRIPT.md) | Summit walkthrough — URLs, talking points, fallback commands |

### Setup (follow in order)
| Doc | What it covers |
|---|---|
| [01 Prerequisites](setup/01-prerequisites.md) | What the demo does, component overview, storage layout, local tooling |
| [02 Cluster](setup/02-cluster.md) | Credentials, RHOAI verify, namespace, Slurm, Spark Operator, MinIO, Redis, Milvus |
| [03 Feast](setup/03-feast.md) | Feast Feature Store — build image, ConfigMaps, BFV apply + materialize |
| [04 Pipeline](setup/04-pipeline.md) | MLflow, container images, Spark ETL (data download + Jobs A/B/C + RAPIDS) |
| [05 Training](setup/05-training.md) | Kubeflow TrainJob — DDP rec model + QLoRA LLM fine-tuning |
| [06 Serving](setup/06-serving.md) | KServe InferenceServices, quick reference, pipeline dependencies |
| [07 Automation](setup/07-automation.md) | Full deploy script, manifest inventory, known issues, teardown |

### Feast deep-dives
| Doc | What it covers |
|---|---|
| [BFV Design](feast/BFV-DESIGN.md) | `@batch_feature_view` architecture, benchmarks (CPU 686s / RAPIDS 658s), runbook, troubleshooting |
| [Internals](feast/INTERNALS.md) | ODH fork investigation, SparkOfflineStore/SparkComputeEngine source analysis |

### Observability
| Doc | What it covers |
|---|---|
| [Monitoring](observability/MONITORING.md) | Grafana, DCGM GPU metrics, Spark Prometheus, Redis exporter, MLflow, collection runbook |

### Roadmap
| Doc | What it covers |
|---|---|
| [Tech Debt](TECH-DEBT.md) | Platform gaps, productisation roadmap, top 5 RHOAI investments |

---

## Key Scripts

| Script | When to run |
|---|---|
| `scripts/wait-and-materialize.sh` | Benchmark CPU vs RAPIDS Feast materialization |
| `scripts/load_embeddings_to_milvus.py` | Load Job C embedding parquet → Milvus (after Spark job completes) |
| `scripts/fetch_training_features.py` | Pre-compute training dataset via `get_historical_features()` |

## Assets

All screenshots are in [`docs/assets/`](assets/) — 51 images covering executor logs, Grafana dashboards, Spark History, RedisInsight, and RHOAI UI.
