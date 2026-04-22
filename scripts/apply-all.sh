#!/usr/bin/env bash
# apply-all.sh — Full cluster state reproduction for SmartShop AI demo.
#
# Applies every manifest in dependency order from a clean namespace.
# Idempotent: safe to re-run after partial failures.
#
# USAGE:
#   set -a; source .env; set +a
#   bash scripts/apply-all.sh [PHASE]
#
# PHASES (run individually or all at once):
#   infra         Namespace, RBAC, secrets, storage, monitoring
#   images        BuildConfigs + ImageStreams (triggers builds)
#   data          Data download job (HuggingFace → MinIO)
#   observability Grafana, redis_exporter, Spark metrics ConfigMap, collect script
#   spark         Submit all 3 Spark ETL jobs (feature engineering, text preprocessing)
#   feast         feast apply inside pod + redis secret patch
#   training      TrainingRuntime + submit rec TrainJob (after spark completes)
#   serving       Apply all 3 InferenceServices
#   notebook      Create notebook ConfigMap + submit papermill runner job
#   all           Everything above in order (default)
#
# PREREQUISITES:
#   oc login <cluster>
#   All vars in .env filled (especially QUAY_TOKEN, HF_TOKEN, PROMETHEUS_TOKEN)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:-all}"

source "$REPO_ROOT/.env" 2>/dev/null || true

log() { echo ""; echo "══ $* ══════════════════════════════════════════════════"; }
ok()  { echo "  ✅ $*"; }
run() { echo "  → $*"; eval "$*"; }

render() {
  python3 "$REPO_ROOT/scripts/render_yaml.py" "$1" "${@:2}"
}

apply() {
  render "$@" | oc apply -f -
}

# ── Phase: infra ──────────────────────────────────────────────────────────────
phase_infra() {
  log "Phase: infra"

  # Namespace (idempotent)
  oc get namespace "$NAMESPACE" &>/dev/null || oc create namespace "$NAMESPACE"
  ok "namespace $NAMESPACE"

  # Secrets — smartshop-credentials
  oc create secret generic smartshop-credentials \
    --from-literal=AWS_ACCESS_KEY_ID="$MINIO_ACCESS_KEY" \
    --from-literal=AWS_SECRET_ACCESS_KEY="$MINIO_SECRET_KEY" \
    --from-literal=AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
    --from-literal=REDIS_HOST="$REDIS_HOST" \
    --from-literal=REDIS_PORT="$REDIS_PORT" \
    --from-literal=REDIS_PASSWORD="$REDIS_PASSWORD" \
    --from-literal=MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
    --from-literal=MINIO_ENDPOINT="$MINIO_ENDPOINT" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
  ok "secret smartshop-credentials"

  # HuggingFace secret
  oc create secret generic hf-credentials \
    --from-literal=token="$HF_TOKEN" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
  ok "secret hf-credentials"

  # Quay push secret
  oc create secret docker-registry "$QUAY_PUSH_SECRET" \
    --docker-server=quay.io \
    --docker-username="$QUAY_USER" \
    --docker-password="$QUAY_TOKEN" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
  ok "secret $QUAY_PUSH_SECRET"

  # Feast Redis secret (used by Feast operator)
  oc create secret generic feast-redis-secret \
    --from-literal=redis="type: redis
connection_string: \"${REDIS_HOST}:${REDIS_PORT},password=${REDIS_PASSWORD}\"" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
  ok "secret feast-redis-secret"

  # Feast S3 credentials
  oc create secret generic feast-s3-credentials \
    --from-literal=AWS_ACCESS_KEY_ID="$MINIO_ACCESS_KEY" \
    --from-literal=AWS_SECRET_ACCESS_KEY="$MINIO_SECRET_KEY" \
    --from-literal=AWS_ENDPOINT_URL_S3="$MINIO_ENDPOINT" \
    --from-literal=AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
  ok "secret feast-s3-credentials"

  # MLflow token secret (MLFLOW_TRACKING_TOKEN must be pre-generated from RHOAI SA)
  # Generate: TOKEN=$(oc create token <sa> -n redhat-ods-applications --duration=8760h)
  # Then set MLFLOW_TRACKING_TOKEN=<token> in .env
  if [[ -n "${MLFLOW_TRACKING_TOKEN:-}" ]]; then
    oc create secret generic smartshop-mlflow-token \
      --from-literal=MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
      --from-literal=MLFLOW_TRACKING_TOKEN="$MLFLOW_TRACKING_TOKEN" \
      --from-literal=MLFLOW_TRACKING_INSECURE_TLS="${MLFLOW_TRACKING_INSECURE_TLS:-true}" \
      --from-literal=MLFLOW_WORKSPACE="$NAMESPACE" \
      -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
    ok "secret smartshop-mlflow-token"
  else
    echo "  ⚠  MLFLOW_TRACKING_TOKEN not set — creating empty mlflow token secret (tracking disabled)"
    oc create secret generic smartshop-mlflow-token \
      --from-literal=MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-}" \
      --from-literal=MLFLOW_TRACKING_TOKEN="" \
      --from-literal=MLFLOW_TRACKING_INSECURE_TLS="true" \
      --from-literal=MLFLOW_WORKSPACE="$NAMESPACE" \
      -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
    ok "secret smartshop-mlflow-token (no token — MLflow disabled)"
  fi

  # Spark RBAC (ServiceAccount + Role + RoleBinding)
  oc create serviceaccount spark -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
  oc create role spark-role \
    --verb=create,get,list,watch,delete,patch,update,deletecollection \
    --resource=pods,services,configmaps,persistentvolumeclaims,endpoints \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
  oc create role spark-role-logs \
    --verb=get,list,watch \
    --resource=pods/log,pods/exec \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
  oc create rolebinding spark-role-binding \
    --role=spark-role --serviceaccount="$NAMESPACE:spark" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
  oc create rolebinding spark-role-logs-binding \
    --role=spark-role-logs --serviceaccount="$NAMESPACE:spark" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
  ok "spark RBAC"

  # User-workload monitoring (cluster-admin required, once)
  apply "$REPO_ROOT/infrastructure/openshift/user-workload-monitoring.yaml" || \
    echo "  ⚠ user-workload-monitoring: needs cluster-admin — skip if already enabled"

  # MinIO: create smartshop-spark-logs bucket + events/ prefix (idempotent)
  MINIO_POD=$(oc get pod -n "$NAMESPACE" -l app=minio \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
  if [[ -n "$MINIO_POD" ]]; then
    oc exec -n "$NAMESPACE" "$MINIO_POD" -- /bin/sh -c "
      mc alias set local http://localhost:9000 \${MINIO_ROOT_USER} \${MINIO_ROOT_PASSWORD} >/dev/null 2>&1
      mc mb local/smartshop-spark-logs 2>/dev/null || true
      printf '' | mc pipe local/smartshop-spark-logs/events/.keep >/dev/null 2>&1 || true
      echo 'spark-logs bucket ready'
    " && ok "MinIO smartshop-spark-logs/events/ bucket ready"
  else
    echo "  ⚠  MinIO pod not found — create smartshop-spark-logs/events/ bucket manually"
  fi
}

# ── Phase: images ─────────────────────────────────────────────────────────────
phase_images() {
  log "Phase: images"
  apply "$REPO_ROOT/infrastructure/openshift/imagestreams.yaml"
  ok "ImageStreams"

  apply "$REPO_ROOT/infrastructure/openshift/buildconfigs.yaml"
  ok "BuildConfigs applied — builds auto-start from git"
  echo "  Monitor: oc get builds -n $NAMESPACE -w"
  echo "  All 6 must Complete before proceeding:"
  echo "    spark-jobs, spark-jobs-rapids, feast-spark-server, rec-trainer, llm-trainer, rec-server"
}

# ── Phase: data ───────────────────────────────────────────────────────────────
phase_data() {
  log "Phase: data"
  # Upload S3A JARs to MinIO (required by Spark before ETL)
  apply "$REPO_ROOT/infrastructure/openshift/upload-spark-jars-job.yaml"
  ok "upload-spark-jars job submitted"
  echo "  Wait: oc wait job/upload-spark-jars -n $NAMESPACE --for=condition=Complete --timeout=600s"

  # Full dataset download (HuggingFace → MinIO sharded parquet)
  apply "$REPO_ROOT/infrastructure/openshift/data-download-job.yaml"
  ok "data-download job submitted"
  # Job name comes from the YAML metadata.name field
  JOB_NAME=$(grep 'name: smartshop-data-download' "$REPO_ROOT/infrastructure/openshift/data-download-job.yaml" | awk '{print $2}' | head -1)
  echo "  Wait: oc logs -n $NAMESPACE job/${JOB_NAME} -f"
}

# ── Phase: observability ──────────────────────────────────────────────────────
phase_observability() {
  log "Phase: observability"

  apply "$REPO_ROOT/infrastructure/openshift/spark-metrics-configmap.yaml"
  ok "spark-metrics-config ConfigMap"

  apply "$REPO_ROOT/infrastructure/openshift/redis-exporter.yaml"
  ok "redis-exporter Deployment + ServiceMonitor"

  apply "$REPO_ROOT/infrastructure/openshift/grafana.yaml"
  ok "Grafana Deployment + route"
  echo "  URL: https://grafana-$NAMESPACE.apps.$OC_CLUSTER_DOMAIN"

  # Embed collect script into ConfigMap
  oc create configmap smartshop-collect-script \
    --from-file=collect-run-metrics.sh="$REPO_ROOT/scripts/collect-run-metrics.sh" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
  ok "smartshop-collect-script ConfigMap (collect-run-metrics.sh)"
}

# ── Phase: spark ──────────────────────────────────────────────────────────────
phase_spark() {
  log "Phase: spark"

  # Spark History Server (event log UI)
  apply "$REPO_ROOT/infrastructure/openshift/spark-history-server.yaml"
  ok "Spark History Server Deployment + Service + Route"

  # Spark metrics ConfigMap (shared by all SparkApps)
  apply "$REPO_ROOT/infrastructure/openshift/spark-metrics-configmap.yaml"
  ok "spark-metrics-config ConfigMap"

  # Refresh script ConfigMaps (always sync from repo — these are hot-reloaded by pods)
  oc create configmap smartshop-feature-engineering-script \
    --from-file=feature_engineering.py="$REPO_ROOT/spark/feature_engineering.py" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
  ok "feature_engineering script ConfigMap"

  oc create configmap smartshop-text-preprocessing-script \
    --from-file=text_preprocessing.py="$REPO_ROOT/spark/text_preprocessing.py" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
  ok "text_preprocessing script ConfigMap"

  oc create configmap smartshop-embedding-script \
    --from-file=embedding_generation.py="$REPO_ROOT/spark/embedding_generation.py" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
  ok "embedding_generation script ConfigMap"

  # Fixed mlflow_metrics.py (Python 3.8 compat + MLflow auth guard)
  # Mounted into RAPIDS/cpu-baseline driver/executor pods to override the baked-in version
  oc create configmap spark-mlflow-metrics-script \
    --from-file=mlflow_metrics.py="$REPO_ROOT/spark/utils/mlflow_metrics.py" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
  ok "spark-mlflow-metrics-script ConfigMap (mlflow_metrics.py)"

  # GPU discovery script
  oc create configmap smartshop-gpu-discovery-script \
    --from-literal=getGpusResources.sh='#!/usr/bin/env bash
echo "[{\"name\": \"gpu\", \"addresses\": [\"$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)\"]}]"' \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
  ok "GPU discovery script ConfigMap"

  # RAPIDS GPU job (8 executors, ~140M rows — sharded dirs only)
  render "$REPO_ROOT/infrastructure/openshift/spark-application-rapids.yaml" \
    SPARK_EXECUTOR_INSTANCES=8 SPARK_EXECUTOR_CORES=4 \
    SPARK_EXECUTOR_MEMORY=16g SPARK_DRIVER_MEMORY=8g | oc apply -f -
  ok "RAPIDS SparkApp submitted (8 executors, full 140M row dataset)"

  # CPU baseline (8 executors — apples-to-apples comparison)
  render "$REPO_ROOT/infrastructure/openshift/spark-application-cpu-baseline.yaml" \
    SPARK_EXECUTOR_INSTANCES=8 SPARK_EXECUTOR_CORES=4 \
    SPARK_EXECUTOR_MEMORY=16g SPARK_DRIVER_MEMORY=8g | oc apply -f -
  ok "CPU baseline SparkApp submitted (8 executors)"

  # Text preprocessing (4 executors)
  render "$REPO_ROOT/infrastructure/openshift/spark-application-text-preprocessing.yaml" \
    SPARK_EXECUTOR_INSTANCES=4 SPARK_EXECUTOR_CORES=4 \
    SPARK_EXECUTOR_MEMORY=8g | oc apply -f -
  ok "Text preprocessing SparkApp submitted"

  echo ""
  echo "  Monitor all jobs: oc get sparkapplication -n $NAMESPACE -w"
  echo "  Get metrics after completion:"
  echo "    oc logs -n $NAMESPACE smartshop-feature-engineering-rapids-driver | grep METRIC"
  echo "    oc logs -n $NAMESPACE smartshop-feature-engineering-cpu-baseline-driver | grep METRIC"
}

# ── Phase: feast ──────────────────────────────────────────────────────────────
phase_feast() {
  log "Phase: feast"

  # Apply Feast Spark engine ConfigMap + Secret
  apply "$REPO_ROOT/infrastructure/openshift/feast-spark-engine.yaml"
  ok "feast-spark-engine ConfigMap + feast-spark-config Secret"

  # Apply FeatureStore CR (Feast operator deploys the pod)
  apply "$REPO_ROOT/infrastructure/openshift/feast-operator.yaml"
  ok "FeatureStore CR applied — waiting for pod to be Ready..."

  # Wait up to 5 min for the feast pod to be 4/4 Running
  echo "  Waiting for feast pod (4/4 containers)..."
  timeout 300 bash -c "
    until oc get pod -n $NAMESPACE -l 'feast.dev/name=smartshop-feast' \
        -o jsonpath='{.items[0].status.containerStatuses[*].ready}' 2>/dev/null | \
        grep -q 'true true true true'; do
      sleep 10
    done
  " || { echo "  ⚠  Feast pod not ready after 5 min — check: oc describe pod -n $NAMESPACE -l feast.dev/name=smartshop-feast"; return 1; }
  ok "Feast pod ready"

  FEAST_POD=$(oc get pod -n "$NAMESPACE" -l "feast.dev/name=smartshop-feast" \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

  # Patch Redis secret with correct password (Feast operator may reset it)
  REDIS_CONN_B64=$(python3 -c "
import base64
s = 'type: redis\nconnection_string: \"${REDIS_HOST}:${REDIS_PORT},password=${REDIS_PASSWORD}\"'
print(base64.b64encode(s.encode()).decode())
")
  oc patch secret feast-redis-secret -n "$NAMESPACE" \
    --type='json' \
    -p="[{\"op\":\"replace\",\"path\":\"/data/redis\",\"value\":\"$REDIS_CONN_B64\"}]"
  ok "feast-redis-secret patched with Redis password"

  # Run feast apply via the offline container (has pyspark + SparkSource + AWS creds).
  # Path: /feast-data/smartshop/feast/feature_repo (featureRepoPath=feast/feature_repo in CR).
  oc exec -n "$NAMESPACE" "$FEAST_POD" -c offline -- bash -c "
    export AWS_ACCESS_KEY_ID=${MINIO_ACCESS_KEY:-minio}
    export AWS_SECRET_ACCESS_KEY=${MINIO_SECRET_KEY:-minio123}
    cd /feast-data/smartshop/feast/feature_repo
    feast apply 2>&1
  "
  ok "feast apply — feature views registered"

  echo ""
  echo "  ⚠  Run feast materialize AFTER Spark ETL jobs complete:"
  echo "  bash scripts/wait-and-materialize.sh"
  echo "  OR manually: oc exec -n $NAMESPACE $FEAST_POD -c offline -- bash -c '"
  echo "    export AWS_ACCESS_KEY_ID=\$MINIO_ACCESS_KEY"
  echo "    export AWS_SECRET_ACCESS_KEY=\$MINIO_SECRET_KEY"
  echo "    cd /feast-data/smartshop/feast/feature_repo"
  echo "    feast materialize-incremental \$(date -u +%Y-%m-%dT%H:%M:%S)'"
}

# ── Phase: training ───────────────────────────────────────────────────────────
phase_training() {
  log "Phase: training"

  # Apply TrainingRuntime + ClusterTrainingRuntime (without TrainJobs)
  python3 -c "
content = open('$REPO_ROOT/infrastructure/openshift/trainjobs.yaml').read()
import re, os
# Load env
env = {}
for line in open('$REPO_ROOT/.env'):
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, _, v = line.partition('=')
        env[k.strip()] = v.strip().strip('\"').strip(\"'\")

def render(tmpl):
    result = re.sub(r'\\\$\{(\w+)(?::-(.*?))?\}',
                    lambda m: env.get(m.group(1), m.group(2) or m.group(0)), tmpl)
    return re.sub(r'\\\$(\w+)', lambda m: env.get(m.group(1), m.group(0)), result)

for doc in content.split('---'):
    doc = doc.strip()
    if 'TrainingRuntime' in doc and 'kind: TrainJob' not in doc:
        print('---'); print(render(doc))
" | oc apply -f -
  ok "TrainingRuntime + ClusterTrainingRuntime applied"

  # Render and save rec TrainJob for manual submission after Spark completes
  render "$REPO_ROOT/infrastructure/openshift/trainjobs.yaml" \
    > /tmp/trainjobs-rendered.yaml
  python3 -c "
content = open('/tmp/trainjobs-rendered.yaml').read()
for doc in content.split('---'):
    if 'kind: TrainJob' in doc and 'smartshop-rec-train' in doc:
        print('---'); print(doc.strip())
" > /tmp/rec-trainjob.yaml
  ok "rec-trainjob.yaml saved to /tmp/rec-trainjob.yaml"

  echo ""
  echo "  ⚠  Submit TrainJob AFTER feast materialize completes:"
  echo "    oc apply -f /tmp/rec-trainjob.yaml"
  echo "    oc get trainjob smartshop-rec-train -n $NAMESPACE -w"
}

# ── Phase: serving ────────────────────────────────────────────────────────────
phase_serving() {
  log "Phase: serving"
  apply "$REPO_ROOT/infrastructure/openshift/inferenceservices.yaml"
  ok "InferenceServices applied"
  echo "  Monitor: oc get inferenceservice -n $NAMESPACE -w"
  echo "  Ready when READY=True for all 3: smartshop-rec, smartshop-llm, smartshop-rag"
}

# ── Phase: notebook ───────────────────────────────────────────────────────────
phase_notebook() {
  log "Phase: notebook"

  # Create/update notebook ConfigMap (< 1MB — fits fine)
  oc create configmap smartshop-metrics-notebook \
    --from-file=metrics_analysis.ipynb="$REPO_ROOT/notebooks/metrics_analysis.ipynb" \
    -n "$NAMESPACE" --dry-run=client -o yaml | oc apply -f -
  ok "smartshop-metrics-notebook ConfigMap"

  # Delete previous run if exists (Job names must be unique)
  oc delete job smartshop-notebook-runner -n "$NAMESPACE" --ignore-not-found=true

  # Submit papermill job
  render "$REPO_ROOT/infrastructure/openshift/notebook-runner-job.yaml" | oc apply -f -
  ok "notebook-runner Job submitted"

  echo ""
  echo "  Watch: oc logs -n $NAMESPACE job/smartshop-notebook-runner -f"
  echo "  Outputs uploaded to: s3://smartshop-models/notebooks/"
  echo "  Download:"
  echo "    aws s3 cp s3://smartshop-models/notebooks/ ./notebooks/output/ \\"
  echo "      --recursive --endpoint-url $MINIO_ENDPOINT_EXTERNAL"
}

# ── Main ──────────────────────────────────────────────────────────────────────
case "$PHASE" in
  infra)         phase_infra ;;
  images)        phase_images ;;
  data)          phase_data ;;
  observability) phase_observability ;;
  spark)         phase_spark ;;
  feast)         phase_feast ;;
  training)      phase_training ;;
  serving)       phase_serving ;;
  notebook)      phase_notebook ;;
  all)
    phase_infra
    phase_images
    phase_data
    phase_observability
    phase_spark
    phase_feast
    phase_training
    phase_serving
    phase_notebook
    ;;
  *)
    echo "Unknown phase: $PHASE"
    echo "Valid: infra images data observability spark feast training serving notebook all"
    exit 1
    ;;
esac

echo ""
echo "✅  Phase '$PHASE' complete."
