#!/usr/bin/env bash
# Deploy Milvus standalone for SmartShop demo
# Usage: ./deploy.sh [namespace]
set -euo pipefail

NAMESPACE=${1:-smartshop}
CHART_VERSION="4.2.58"
MINIO_ENDPOINT="https://minio-s3-${NAMESPACE}.apps.oai-kft-ibm.ibm.rh-ods.com"

echo "==> Adding Milvus Helm repo..."
helm repo add milvus https://zilliztech.github.io/milvus-helm/ --force-update
helm repo update milvus

echo "==> Creating milvus bucket in MinIO..."
AWS_ACCESS_KEY_ID=minio AWS_SECRET_ACCESS_KEY=minio123 \
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
