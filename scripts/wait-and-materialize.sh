#!/usr/bin/env bash
# wait-and-materialize.sh
#
# Waits for the two GPU/CPU feature-engineering SparkApps to finish,
# then runs feast materialize-incremental via the feast pod.
# Text preprocessing and embedding jobs run independently and are
# NOT waited on here (they feed a separate pipeline step).
#
# USAGE:
#   source .env
#   bash scripts/wait-and-materialize.sh
#
# Pipeline:
#   1. Poll SparkApps (GPU + CPU baseline) until COMPLETED / FAILED
#   2. Print [METRIC] lines from driver pods
#   3. feast apply  (idempotent — ensures registry is current)
#   4. feast materialize-incremental → Redis
#   5. Verify Redis key count
#   6. Collect metrics bundles (RAPIDS, CPU, feast)
set -euo pipefail

NAMESPACE="${NAMESPACE:-smartshop}"

# SparkApps to wait for.
# NOTE: feature-engineering.py writes to smartshop-features/ (Parquet) — NOT to processed/reviews/.
# processed/reviews/ is produced by preprocess_reviews_full.py (adds event_timestamp column).
# Feast BFV materialize reads from processed/reviews/, not from smartshop-features/.
# text-preprocessing and embedding run independently (different output path/consumer).
APPS=(
  "smartshop-feature-engineering-rapids"
  "smartshop-feature-engineering-cpu-baseline"
)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 1. Wait for SparkApps to complete ─────────────────────────────────────────

log "Waiting for SparkApps to complete..."
while true; do
  all_done=true
  for app in "${APPS[@]}"; do
    status=$(oc get sparkapplication "$app" -n "$NAMESPACE" \
      -o jsonpath='{.status.applicationState.state}' 2>/dev/null || echo "UNKNOWN")
    log "  $app → $status"
    if [[ "$status" != "COMPLETED" && "$status" != "FAILED" ]]; then
      all_done=false
    fi
  done

  if $all_done; then
    log "SparkApps finished."
    break
  fi
  sleep 30
done

# ── 2. Print key metrics from each driver ─────────────────────────────────────

log ""
log "══ SparkApp Metrics ════════════════════════════════════"
for app in "${APPS[@]}"; do
  status=$(oc get sparkapplication "$app" -n "$NAMESPACE" \
    -o jsonpath='{.status.applicationState.state}' 2>/dev/null || echo "UNKNOWN")
  log "  [$status] $app"
  DRIVER_POD="${app}-driver"
  if oc get pod "$DRIVER_POD" -n "$NAMESPACE" &>/dev/null; then
    oc logs "$DRIVER_POD" -n "$NAMESPACE" 2>/dev/null | \
      grep "\[METRIC\]" | sed 's/^/      /' || true
  fi
done
log "════════════════════════════════════════════════════════"

for app in "${APPS[@]}"; do
  status=$(oc get sparkapplication "$app" -n "$NAMESPACE" \
    -o jsonpath='{.status.applicationState.state}' 2>/dev/null)
  if [[ "$status" == "FAILED" ]]; then
    log "ERROR: $app FAILED — check logs before materializing."
    log "  oc logs -n $NAMESPACE ${app}-driver | tail -50"
    exit 1
  fi
done

# ── 3. feast apply + feast materialize-incremental ────────────────────────────

FEAST_POD=$(oc get pod -n "$NAMESPACE" -l "feast.dev/name=smartshop-feast" \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

if [[ -z "$FEAST_POD" ]]; then
  log "ERROR: feast pod not found in namespace $NAMESPACE"
  exit 1
fi

log "Running feast apply via $FEAST_POD ..."
oc exec -n "$NAMESPACE" "$FEAST_POD" -c offline -- bash -c "
  export AWS_ACCESS_KEY_ID=${MINIO_ACCESS_KEY:-minio}
  export AWS_SECRET_ACCESS_KEY=${MINIO_SECRET_KEY:-minio123}
  cd /feast-data/smartshop/feast/feature_repo
  feast apply 2>&1
" | tail -5

log "Running feast materialize-incremental via $FEAST_POD ..."
MATL_START=$(date +%s)

oc exec -n "$NAMESPACE" "$FEAST_POD" -c offline -- bash -c "
  export AWS_ACCESS_KEY_ID=${MINIO_ACCESS_KEY:-minio}
  export AWS_SECRET_ACCESS_KEY=${MINIO_SECRET_KEY:-minio123}
  cd /feast-data/smartshop/feast/feature_repo
  feast materialize-incremental \"$(date -u +%Y-%m-%dT%H:%M:%S)\" 2>&1
"

MATL_ELAPSED=$(( $(date +%s) - MATL_START ))
log "feast materialize-incremental done in ${MATL_ELAPSED}s"

# ── 4. Verify Redis has feature keys ──────────────────────────────────────────

log "Verifying Redis feature keys..."
KEY_COUNT=$(oc exec -n "$NAMESPACE" "$FEAST_POD" -c online -- python3 -c "
import redis, os
r = redis.Redis(
    host='redis.${NAMESPACE}.svc.cluster.local',
    port=6379,
    password=os.environ.get('REDIS_PASSWORD', '${REDIS_PASSWORD:-smartshop-redis-2026}'),
    decode_responses=True
)
print(r.dbsize())
" 2>/dev/null || echo "0")

log "Redis keys: $KEY_COUNT"
if [[ "$KEY_COUNT" -gt 0 ]]; then
  log "✅ Feast materialization verified — $KEY_COUNT keys in Redis"
else
  log "⚠️  Redis appears empty — check feast materialize output above"
fi

# ── 5. Collect metrics bundles ────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log "Collecting RAPIDS metrics bundle..."
RUN_TYPE=rapids APP_NAME=smartshop-feature-engineering-rapids \
  bash "${SCRIPT_DIR}/collect-run-metrics.sh" || true

log "Collecting CPU baseline metrics bundle..."
RUN_TYPE=cpu APP_NAME=smartshop-feature-engineering-cpu-baseline \
  bash "${SCRIPT_DIR}/collect-run-metrics.sh" || true

log "Collecting feast metrics bundle..."
RUN_TYPE=feast bash "${SCRIPT_DIR}/collect-run-metrics.sh" || true

log ""
log "══ All done ════════════════════════════════════════════"
log "  Phase 5: Submit TrainJob"
log "    oc apply -f infrastructure/openshift/trainjobs.yaml"
log "  Phase 7: Run metrics notebook"
log "    jupyter notebook notebooks/metrics_analysis.ipynb"
log "═══════════════════════════════════════════════════════"
