#!/usr/bin/env bash
# wait-and-materialize.sh
#
# Runs a full, timed feast materialize via the Feast pod using distributed
# Spark (k8s:// mode).  Supports CPU and RAPIDS profiles via SPARK_PROFILE env.
#
# USAGE:
#   source .env
#   SPARK_PROFILE=cpu   bash scripts/wait-and-materialize.sh   # CPU baseline
#   SPARK_PROFILE=rapids bash scripts/wait-and-materialize.sh  # RAPIDS GPU
#
# What this does:
#   1. Resolve the Feast offline-store pod (label feast.dev/name=smartshop-feast)
#   2. Switch feast-spark-engine ConfigMap to the chosen profile (cpu or rapids)
#   3. Flush Redis so the key count is a clean signal
#   4. Run `feast apply` (idempotent — syncs registry)
#   5. Run `feast materialize <start> <end>` with nohup, stream logs, wait for exit
#   6. Report wall-clock time and final Redis key count
#
# Prerequisities:
#   - Feast pod running with image that has pyspark (feast-spark-server)
#   - feast-spark-driver Service applied:
#       oc apply -f infrastructure/openshift/feast-spark-driver-svc.yaml
#   - spark SA RBAC applied:
#       oc apply -f infrastructure/smartshop/spark-rbac.yaml
#   - ConfigMaps applied:
#       oc apply -f infrastructure/openshift/feast-spark-engine.yaml
#       oc apply -f infrastructure/openshift/feast-spark-engine-cpu.yaml
#       oc apply -f infrastructure/openshift/feast-spark-engine-rapids.yaml
#
# Benchmark dates used in Summit demo:
#   START=2014-01-01T00:00:00
#   END=2018-12-31T23:59:59
#
set -euo pipefail

NAMESPACE="${NAMESPACE:-smartshop}"
SPARK_PROFILE="${SPARK_PROFILE:-cpu}"        # cpu | rapids
START_DATE="${START_DATE:-2014-01-01T00:00:00}"
END_DATE="${END_DATE:-2018-12-31T23:59:59}"
REDIS_PASSWORD="${REDIS_PASSWORD:-smartshop-redis-2026}"
FEATURE_REPO="/feast-data/smartshop/feast/feature_repo"

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { log "ERROR: $*"; exit 1; }

# ── 0. Validate profile ───────────────────────────────────────────────────────
[[ "$SPARK_PROFILE" == "cpu" || "$SPARK_PROFILE" == "rapids" ]] \
  || die "SPARK_PROFILE must be 'cpu' or 'rapids', got: $SPARK_PROFILE"

# ── 1. Resolve Feast offline-store pod ───────────────────────────────────────
log "Looking up Feast offline pod (feast.dev/name=smartshop-feast) ..."
FEAST_POD=$(oc get pod -n "$NAMESPACE" \
  -l "feast.dev/name=smartshop-feast" \
  --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)

[[ -n "$FEAST_POD" ]] || die "No running feast pod found in namespace $NAMESPACE"
log "Using pod: $FEAST_POD"

# ── 2. Switch ConfigMap batchEngine to the desired profile ───────────────────
PROFILE_CM="feast-spark-engine-${SPARK_PROFILE}"
log "Patching feast-spark-engine ConfigMap → pointing to $PROFILE_CM ..."
oc patch configmap feast-spark-engine -n "$NAMESPACE" \
  --type merge \
  -p "{\"data\":{\"batch_engine\": \"$(
    oc get configmap "$PROFILE_CM" -n "$NAMESPACE" \
      -o jsonpath='{.data.batch_engine}'
  )\"}}" \
  || die "Failed to patch feast-spark-engine"

# Bounce the feast deployment to pick up the new ConfigMap value
log "Restarting feast deployment to reload ConfigMap ..."
oc rollout restart deployment/feast-smartshop-feast -n "$NAMESPACE"
oc rollout status  deployment/feast-smartshop-feast -n "$NAMESPACE" --timeout=120s

# Re-resolve pod after restart
FEAST_POD=$(oc get pod -n "$NAMESPACE" \
  -l "feast.dev/name=smartshop-feast" \
  --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
[[ -n "$FEAST_POD" ]] || die "Feast pod not running after rollout"
log "Using pod (post-restart): $FEAST_POD"

# ── 3. Flush Redis ────────────────────────────────────────────────────────────
log "Flushing Redis (clean key count baseline) ..."
REDIS_POD=$(oc get pod -n "$NAMESPACE" -l "app=redis" \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
[[ -n "$REDIS_POD" ]] || die "Redis pod not found"

oc exec -n "$NAMESPACE" "$REDIS_POD" -- \
  redis-cli -a "$REDIS_PASSWORD" FLUSHALL
log "Redis flushed."

# ── 4. feast apply ────────────────────────────────────────────────────────────
log "Running feast apply ..."
oc exec -n "$NAMESPACE" "$FEAST_POD" -c offline -- bash -c "
  export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-minio}
  export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-minio123}
  cd ${FEATURE_REPO}
  feast apply 2>&1
" | tail -10

# ── 5. feast materialize (timed) ─────────────────────────────────────────────
log "Starting feast materialize [${SPARK_PROFILE}] ${START_DATE} → ${END_DATE} ..."
WALL_START=$(date +%s)

oc exec -n "$NAMESPACE" "$FEAST_POD" -c offline -- bash -c "
  export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-minio}
  export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-minio123}
  cd ${FEATURE_REPO}
  feast materialize '${START_DATE}' '${END_DATE}' 2>&1
"
RC=$?

WALL_END=$(date +%s)
ELAPSED=$(( WALL_END - WALL_START ))

# ── 6. Report ─────────────────────────────────────────────────────────────────
log ""
log "══════════════════════════════════════════════════"
log "  Profile  : ${SPARK_PROFILE}"
log "  Exit code: ${RC}"
log "  Wall time: ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)"

KEY_COUNT=$(oc exec -n "$NAMESPACE" "$REDIS_POD" -- \
  redis-cli -a "$REDIS_PASSWORD" DBSIZE 2>/dev/null | awk '{print $1}' || echo "unknown")
log "  Redis keys: ${KEY_COUNT}"

if [[ "$RC" -eq 0 && "$KEY_COUNT" -gt 26000000 ]]; then
  log "  ✅ PASS — materialization complete"
else
  log "  ⚠️  WARN — RC=${RC}, keys=${KEY_COUNT} (expected ~26,493,202)"
fi
log "══════════════════════════════════════════════════"
