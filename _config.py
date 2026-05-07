"""SmartShop demo — shared configuration.

Single source of truth for all pipeline notebooks.
Usage: `from _config import *` in each notebook's config cell.
"""

import os

# ── Namespace ────────────────────────────────────────────────────────
NAMESPACE = os.environ.get("NAMESPACE", "smartshop")

# ── S3 / MinIO ──────────────────────────────────────────────────────
S3_ENDPOINT = os.environ.get(
    "S3_ENDPOINT_URL",
    os.environ.get(
        "AWS_ENDPOINT_URL_S3",
        f"http://minio.{NAMESPACE}.svc.cluster.local:9000",
    ),
).rstrip("/")
MINIO_ENDPOINT = S3_ENDPOINT  # alias (NB2/NB4 convention)

AWS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET = os.environ.get("AWS_SECRET_ACCESS_KEY", "")

S3_RAW_BUCKET = os.environ.get("S3_RAW_BUCKET", "smartshop-raw")
S3_FEATURES_BUCKET = os.environ.get("S3_FEATURES_BUCKET", "smartshop-features")
S3_MODELS_BUCKET = os.environ.get("S3_MODELS_BUCKET", "smartshop-models")

# ── Redis ────────────────────────────────────────────────────────────
REDIS_HOST = os.environ.get("REDIS_HOST", f"redis.{NAMESPACE}.svc.cluster.local")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")

# ── Milvus ───────────────────────────────────────────────────────────
MILVUS_HOST = os.environ.get("MILVUS_HOST", f"milvus.{NAMESPACE}.svc.cluster.local")
MILVUS_PORT = int(os.environ.get("MILVUS_PORT", "19530"))

# ── Feast ────────────────────────────────────────────────────────────
FEAST_REGISTRY = f"feast-smartshop-feast-registry.{NAMESPACE}.svc.cluster.local:443"
FEAST_TLS_CERT = os.environ.get(
    "FEAST_TLS_CERT", "/etc/pki/tls/custom-certs/ca-bundle.crt"
)
# Auto-mounted by RHOAI when FeatureStore CR is attached to the workbench.
# Read-only client config with remote endpoints for offline/online/registry.
FEAST_CLIENT_CONFIG = f"/opt/app-root/src/feast-config/{NAMESPACE}"
FEAST_FEATURE_REPO_ON_POD = "/feast-data/smartshop/feast/feature_repo"
MATERIALIZE_START = "2016-01-01T00:00:00"

# ── K8s Secret names ────────────────────────────────────────────────
S3_CREDENTIALS_SECRET = "smartshop-credentials"
MLFLOW_SECRET = "smartshop-mlflow-token"
HF_SECRET = "hf-credentials"

# ── Container images ────────────────────────────────────────────────
# All images are defined in .env.example and overridable via env vars.
INTERNAL_REGISTRY = "image-registry.openshift-image-registry.svc:5000"
REGISTRY = os.environ.get("REGISTRY", "quay.io/abdhumal")
IMAGE_TAG = os.environ.get("IMAGE_TAG", "v3")

# Feast
FEAST_SERVER_IMAGE = os.environ.get(
    "FEAST_SERVER_IMAGE", f"{REGISTRY}/smartshop-feast-spark-server:{IMAGE_TAG}"
)
FEAST_SPARK_EXECUTOR_IMAGE = os.environ.get(
    "FEAST_SPARK_EXECUTOR_IMAGE", f"{REGISTRY}/smartshop-feast-spark-executor:{IMAGE_TAG}"
)
FEAST_SPARK_EXECUTOR_RAPIDS_IMAGE = os.environ.get(
    "FEAST_SPARK_EXECUTOR_RAPIDS_IMAGE",
    f"{REGISTRY}/smartshop-feast-spark-executor-rapids:{IMAGE_TAG}",
)

# Spark / batch
SPARK_JOBS_IMAGE = os.environ.get(
    "SPARK_JOBS_IMAGE", f"{REGISTRY}/smartshop-spark-jobs:latest"
)
SPARK_EXECUTOR_EMBEDDINGS_IMAGE = os.environ.get(
    "SPARK_EXECUTOR_EMBEDDINGS_IMAGE",
    f"{REGISTRY}/smartshop-feast-spark-executor-embeddings:latest",
)

# Training
REC_TRAINER_IMAGE = os.environ.get(
    "REC_TRAINER_IMAGE", f"{REGISTRY}/smartshop-rec-trainer:latest"
)
LLM_TRAINER_IMAGE = os.environ.get(
    "LLM_TRAINER_IMAGE", f"{REGISTRY}/smartshop-llm-trainer:latest"
)

# Serving
REC_SERVER_IMAGE = os.environ.get(
    "REC_SERVER_IMAGE", f"{REGISTRY}/smartshop-rec-server:latest"
)
RAG_SERVER_IMAGE = os.environ.get(
    "RAG_SERVER_IMAGE", f"{REGISTRY}/smartshop-rag-server:latest"
)
VLLM_IMAGE = os.environ.get("VLLM_IMAGE", "vllm/vllm-openai:v0.13.0")
DEMO_UI_IMAGE = os.environ.get(
    "DEMO_UI_IMAGE", f"{REGISTRY}/smartshop-demo-ui:latest"
)

# Observability
GRAFANA_IMAGE = os.environ.get("GRAFANA_IMAGE", "grafana/grafana-oss:11.6.0")
REDIS_EXPORTER_IMAGE = os.environ.get(
    "REDIS_EXPORTER_IMAGE", "oliver006/redis_exporter:latest"
)

# ── MLflow ──────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    "https://mlflow.redhat-ods-applications.svc.cluster.local:8443/mlflow",
)

# ── Models ──────────────────────────────────────────────────────────
LLM_BASE_MODEL = os.environ.get(
    "LLM_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"
)
EMBED_MODEL = os.environ.get(
    "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
MODEL_PATH = os.environ.get(
    "MODEL_PATH", f"s3://{S3_MODELS_BUCKET}/recommendation/best_model.pt"
)
LLM_ADAPTER_PATH = os.environ.get(
    "LLM_ADAPTER_PATH", f"s3://{S3_MODELS_BUCKET}/llm-adapter"
)
VLLM_RUNTIME = "smartshop-vllm-runtime"

# ── Storage class ──────────────────────────────────────────────────
STORAGE_CLASS = os.environ.get("STORAGE_CLASS", "nfs-csi")


def validate():
    """Initialize K8s config and validate environment. Call once per notebook."""
    from kubernetes import config as k8s_config

    issues = []
    try:
        k8s_config.load_incluster_config()
    except Exception:
        issues.append("Not running in-cluster (load_incluster_config failed)")

    if not AWS_KEY:
        issues.append("AWS_ACCESS_KEY_ID not set (check DataConnection or env)")
    if not AWS_SECRET:
        issues.append("AWS_SECRET_ACCESS_KEY not set (check DataConnection or env)")

    if issues:
        print("⚠ Configuration issues:")
        for i in issues:
            print(f"  • {i}")
        return False

    print(f"Namespace:  {NAMESPACE}")
    print(f"S3:         {S3_ENDPOINT}")
    print(f"Redis:      {REDIS_HOST}:{REDIS_PORT}")
    print(f"Milvus:     {MILVUS_HOST}:{MILVUS_PORT}")
    print(f"Feast:      {FEAST_REGISTRY}")
    print("Config OK ✓")
    return True
