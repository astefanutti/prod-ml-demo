#!/usr/bin/env bash
# Deploy Milvus standalone for SmartShop demo
# Usage: ./deploy.sh [namespace] [minio-route-host]
#   minio-route-host defaults to the cluster-local service (works for in-cluster bucket creation)
set -euo pipefail

NAMESPACE=${1:-smartshop}
CHART_VERSION="4.2.58"
MINIO_ENDPOINT=${2:-"http://minio.${NAMESPACE}.svc.cluster.local:9000"}

# Load .env from repo root if not already set in environment
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../../.env"
if [ -f "${ENV_FILE}" ] && [ -z "${MINIO_ACCESS_KEY}" ]; then
  # shellcheck disable=SC1090
  set -a; . "${ENV_FILE}"; set +a
fi

: "${MINIO_ACCESS_KEY:?'MINIO_ACCESS_KEY not set. Copy .env.example to .env and fill in values.'}"
: "${MINIO_SECRET_KEY:?'MINIO_SECRET_KEY not set. Copy .env.example to .env and fill in values.'}"

echo "==> Adding Milvus Helm repo..."
helm repo add milvus https://zilliztech.github.io/milvus-helm/ --force-update
helm repo update milvus

echo "==> Pre-creating milvus standalone PVC (nfs-csi, required before Helm install)..."
oc apply -n "${NAMESPACE}" -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: milvus
  namespace: ${NAMESPACE}
  annotations:
    helm.sh/resource-policy: keep
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: nfs-csi
  resources:
    requests:
      storage: 50Gi
EOF

echo "==> Creating milvus bucket in MinIO..."
AWS_ACCESS_KEY_ID="${MINIO_ACCESS_KEY}" AWS_SECRET_ACCESS_KEY="${MINIO_SECRET_KEY}" \
  aws s3 mb s3://milvus \
  --endpoint-url "${MINIO_ENDPOINT}" \
  --no-verify-ssl 2>/dev/null || echo "    bucket already exists, skipping"

echo "==> Creating milvus ServiceAccount with anyuid SCC..."
oc create sa milvus -n "${NAMESPACE}" 2>/dev/null || echo "    SA already exists"
oc adm policy add-scc-to-user anyuid -z milvus -n "${NAMESPACE}"

echo "==> Installing Milvus chart v${CHART_VERSION}..."
helm upgrade --install milvus milvus/milvus \
  --namespace "${NAMESPACE}" \
  --version "${CHART_VERSION}" \
  -f "$(dirname "$0")/values.yaml"

echo "==> Patching etcd StatefulSet to use milvus SA (anyuid SCC)..."
oc patch statefulset milvus-etcd -n "${NAMESPACE}" \
  --type=merge \
  -p '{"spec":{"template":{"spec":{"serviceAccountName":"milvus"}}}}'

echo "==> Patching standalone Deployment to use milvus SA (anyuid SCC)..."
oc patch deployment milvus-standalone -n "${NAMESPACE}" \
  --type=merge \
  -p '{"spec":{"template":{"spec":{"serviceAccountName":"milvus"}}}}'

echo "==> Blanking injected MinIO env vars that conflict with Milvus config..."
oc set env deployment/milvus-standalone -n "${NAMESPACE}" \
  MINIO_ADDRESS=minio.smartshop.svc.cluster.local:9000 \
  MINIO_PORT="" \
  MINIO_SERVICE_HOST="" \
  MINIO_SERVICE_PORT="" \
  MINIO_SERVICE_PORT_API=""

echo "==> Waiting for etcd to become ready (60s init delay)..."
oc rollout status statefulset/milvus-etcd -n "${NAMESPACE}" --timeout=3m

echo "==> Waiting for standalone to become ready..."
oc rollout status deployment/milvus-standalone -n "${NAMESPACE}" --timeout=5m

echo ""
echo "Done. Milvus is ready."
echo "  Internal gRPC endpoint: milvus.${NAMESPACE}.svc.cluster.local:19530"
echo "  Internal REST endpoint: milvus.${NAMESPACE}.svc.cluster.local:9091"
