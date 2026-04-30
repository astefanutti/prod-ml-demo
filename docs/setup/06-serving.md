## 13. Deploy Inference Services

> **Prerequisite:** Trained models at `s3://smartshop-models/` (step 12).
> Rec server image built via BuildConfig or pushed to quay.io.

### Deploy all serving resources (one command)

```bash
set -a && source .env && set +a
envsubst < infrastructure/openshift/serving-runtimes.yaml | oc apply -f -
```

This single manifest creates:
- `smartshop-s3-serving` Secret (S3 creds for adapter download)
- `smartshop-serving-sa` ServiceAccount
- `smartshop-vllm-runtime` ServingRuntime (vLLM + LoRA)
- `smartshop-rec-runtime` ServingRuntime (FastAPI)
- `smartshop-rec` InferenceService (recommendation)
- `smartshop-llm` InferenceService (Mistral-7B + LoRA)
- `smartshop-rag` InferenceService (RAG Q&A)
- `smartshop-serving-metrics` ServiceMonitor (Prometheus scraping)

### ServingRuntimes

| Runtime | Image | Purpose |
|---------|-------|---------|
| `smartshop-vllm-runtime` | `vllm/vllm-openai:v0.13.0` | LLM serving with LoRA support, OpenAI-compatible API |
| `smartshop-rec-runtime` | `quay.io/abdhumal/smartshop-rec-server:latest` | Recommendation model (FastAPI + Prometheus metrics) |

### InferenceServices

| Service | Mode | GPU | Model source |
|---------|------|-----|-------------|
| `smartshop-rec` | RawDeployment | — | `s3://smartshop-models/recommendation` via KServe storage initializer |
| `smartshop-llm` | RawDeployment | 1x GPU | Base: HF download by vLLM. Adapter: `s3://smartshop-models/llm-checkpoints/checkpoint-200/` via init container |
| `smartshop-rag` | RawDeployment | — | No model download — connects to Milvus + LLM endpoint |

### Startup timeline

| Service | Expected time | Bottleneck |
|---------|-------------|-----------|
| `smartshop-rec` | ~30s | S3 model download (~100MB) |
| `smartshop-llm` | ~5 min | vLLM image pull (19GB, cached after first pull) + HF model download (15GB) + CUDA graph capture |
| `smartshop-rag` | ~15s | No model to download |

### Verify

```bash
oc get inferenceservice -n smartshop -w
# NAME            URL                                                       READY
# smartshop-rec   http://smartshop-rec-predictor.smartshop.svc.cluster.local   True
# smartshop-llm   http://smartshop-llm-predictor.smartshop.svc.cluster.local   True
# smartshop-rag   http://smartshop-rag-predictor.smartshop.svc.cluster.local   True
```

### Smoke tests

```bash
# Recommendation (internal — via oc exec)
oc exec deploy/smartshop-rec-predictor -n smartshop -c kserve-container -- \
  curl -s http://localhost:8000/v1/models/smartshop-rec:predict \
  -X POST -H "Content-Type: application/json" \
  -d '{"user_id": "42"}'

# LLM — OpenAI-compatible chat completion
oc exec deploy/smartshop-llm-predictor -n smartshop -c kserve-container -- \
  curl -s http://localhost:8000/v1/chat/completions \
  -X POST -H "Content-Type: application/json" \
  -d '{"model":"smartshop-llm","messages":[{"role":"user","content":"Summarize: Great product, fast shipping, works perfectly."}],"max_tokens":64}'

# LLM — LoRA adapter
oc exec deploy/smartshop-llm-predictor -n smartshop -c kserve-container -- \
  curl -s http://localhost:8000/v1/models
# Should list both "smartshop-llm" (base) and "smartshop-qa" (LoRA adapter)
```

### Prometheus metrics

All serving endpoints expose `/metrics/` with:
- `smartshop_rec_requests_total` / `smartshop_rec_request_duration_seconds`
- `smartshop_llm_requests_total` / `smartshop_llm_request_duration_seconds` / `smartshop_llm_tokens_generated`
- `smartshop_rag_requests_total` / `smartshop_rag_request_duration_seconds`

The `smartshop-serving-metrics` ServiceMonitor scrapes all three via the KServe predictor services.

### Grafana dashboard

The **SmartShop Inference Metrics** dashboard (`uid: smartshop-inference`) shows:
- Request rate by endpoint
- Error rate
- Recommendation latency (p50/p95/p99)
- RAG latency breakdown (retrieval vs LLM)
- LLM latency by endpoint
- Candidates scored, sources retrieved, tokens generated

Uses `ocp-thanos` datasource (Thanos Querier port 9091 — includes user-workload metrics).

### OpenShift-specific fixes (applied in manifest)

| Issue | Fix in `serving-runtimes.yaml` |
|-------|-------------------------------|
| `PermissionError: /.triton` | emptyDir mount at `/.triton` + `TRITON_CACHE_DIR=/tmp/triton_cache` |
| `PermissionError: /.config` | emptyDir mount at `/.config` + `XDG_CONFIG_HOME=/tmp/config` |
| `PermissionError: /.cache` | emptyDir mount at `/.cache` + `XDG_CACHE_HOME=/tmp/xdg_cache` |
| Storage initializer OOM on 15GB model | RawDeployment mode — vLLM pulls base model from HF directly |
| RHOAI vLLM image auth failure | Use public `vllm/vllm-openai:v0.13.0` instead |

---

## Quick Reference

| Component | Endpoint | Credentials |
|---|---|---|
| MinIO Console | `https://minio-console-smartshop.apps.<cluster>/` | Secret: `minio-root-user` |
| MinIO S3 (internal) | `http://minio.smartshop.svc.cluster.local:9000` | Secret: `smartshop-credentials` |
| Redis | `redis.smartshop.svc.cluster.local:6379` | Secret: `redis-credentials` |
| RedisInsight UI | `https://redisinsight-smartshop.apps.<cluster>/` | — |
| Milvus gRPC | `milvus.smartshop.svc.cluster.local:19530` | — |
| Attu (Milvus UI) | `https://attu-smartshop.apps.<cluster>/` | — |
| Feast UI | `https://feast-smartshop-feast-ui-smartshop.apps.<cluster>/` | — |
| MLflow UI | `https://mlflow-redhat-ods-applications.apps.<cluster>/` | — |
| Grafana | `https://grafana-smartshop.apps.<cluster>/` | admin / smartshop2026 |
| Spark History | `https://spark-history-smartshop.apps.<cluster>/` | — |

| Bucket | Purpose |
|---|---|
| `smartshop-raw` | Raw Amazon Reviews dataset |
| `smartshop-features` | Feast offline feature store (Parquet) |
| `smartshop-models` | Trained model artifacts + checkpoints |
| `smartshop-embeddings` | Product/user embedding vectors |
| `milvus` | Milvus vector index segments |
| `smartshop-spark-logs` | Spark event logs (History Server) |

## Pipeline Dependencies

| Phase | Command | Produces | Required by |
|---|---|---|---|
| 0 — Credentials | `make setup-secrets` | Kubernetes Secrets | Everything |
| 1 — Infra | `bash scripts/apply-all.sh infra` | Namespace, RBAC, monitoring | Phases 2–8 |
| 2 — Images | `bash scripts/apply-all.sh images` | Container images | Phases 3–8 |
| 3 — Dataset | `bash scripts/apply-all.sh data` | Raw data in `s3://smartshop-raw/` | Phase 4 |
| 4 — Spark ETL | `bash scripts/apply-all.sh spark` | Features in `s3://smartshop-features/` | Phase 5, 6 |
| 5 — Feast | `bash scripts/apply-all.sh feast` | Features in Redis, embeddings in Milvus | Phase 6, 7 |
| 6 — Training | `bash scripts/apply-all.sh training` | Models in `s3://smartshop-models/` | Phase 7 |
| 7 — Serving | `bash scripts/apply-all.sh serving` | 3 InferenceService endpoints | Phase 8 |
| 8 — Demo UI | `make demo` | Live Gradio UI | — |
