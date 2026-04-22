#!/usr/bin/env bash
# collect-run-metrics.sh — Post-run metrics collection for Spark RAPIDS and Slurm FSDP jobs.
#
# Scrapes four sources and bundles them into a single JSON artifact uploaded to MinIO:
#   1. Spark REST API  — stage durations, task counts, shuffle bytes, executor metrics
#   2. Spark SQL API   — query plans with GPU/CPU operator breakdown
#   3. DCGM Prometheus — GPU utilization, framebuffer memory, NVLink bandwidth
#   4. MLflow REST API — logged metrics + params for cross-run comparison
#
# OUTPUT:
#   s3://smartshop-models/metrics/<RUN_ID>_metrics_bundle.json
#   Printed summary to stdout for direct capture in CI/CD or Jupyter
#
# USAGE (local):
#   source .env
#   RUN_TYPE=rapids APP_NAME=smartshop-feature-engineering-rapids \
#     bash scripts/collect-run-metrics.sh
#
#   RUN_TYPE=cpu APP_NAME=smartshop-feature-engineering-cpu-baseline \
#     bash scripts/collect-run-metrics.sh
#
#   RUN_TYPE=slurm TRAINJOB_NAME=smartshop-llm-finetune \
#     bash scripts/collect-run-metrics.sh
#
# REQUIRED ENV:
#   NAMESPACE, MINIO_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
#   S3_MODELS_BUCKET, MLFLOW_TRACKING_URI, OC_CLUSTER_DOMAIN
set -euo pipefail

RUN_TYPE="${RUN_TYPE:-rapids}"           # rapids | cpu | slurm | feast
APP_NAME="${APP_NAME:-}"
TRAINJOB_NAME="${TRAINJOB_NAME:-}"
NAMESPACE="${NAMESPACE:-smartshop}"
S3_MODELS_BUCKET="${S3_MODELS_BUCKET:-smartshop-models}"
MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://minio.${NAMESPACE}.svc.cluster.local:9000}"
MLFLOW_URI="${MLFLOW_TRACKING_URI:-}"
RUN_ID="$(date +%Y%m%d-%H%M%S)-${RUN_TYPE}"
OUTPUT_FILE="/tmp/metrics_bundle_${RUN_ID}.json"

# ── Helpers ────────────────────────────────────────────────────────────────────

log() { echo "[collect-metrics] $*"; }

s3_upload() {
    local local_file="$1" s3_key="$2"
    AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
    AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
    aws s3 cp "${local_file}" "s3://${S3_MODELS_BUCKET}/metrics/${s3_key}" \
        --endpoint-url "${MINIO_ENDPOINT}" \
        --no-verify-ssl \
        --quiet && log "Uploaded → s3://${S3_MODELS_BUCKET}/metrics/${s3_key}"
}

json_merge() {
    # Merge two JSON objects (requires python3)
    python3 -c "
import json, sys
a = json.load(open('$1'))
b = json.loads('$2')
a.update(b)
print(json.dumps(a, indent=2))
" > /tmp/_merge_tmp.json && mv /tmp/_merge_tmp.json "$1"
}

# ── 1. Spark REST API ──────────────────────────────────────────────────────────

collect_spark_rest() {
    local app_name="$1"
    log "Collecting Spark REST metrics for ${app_name}..."

    # Find driver pod
    DRIVER_POD=$(oc get pod -n "${NAMESPACE}" \
        -l "spark-app-name=${app_name},spark-role=driver" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [[ -z "${DRIVER_POD}" ]]; then
        log "  WARN: driver pod not found for ${app_name} — checking completed pods"
        # Try all pods with matching label (completed jobs are Completed state)
        DRIVER_POD=$(oc get pod -n "${NAMESPACE}" \
            -l "spark-role=driver" \
            --field-selector=status.phase=Succeeded \
            -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null || echo "")
    fi

    if [[ -z "${DRIVER_POD}" ]]; then
        log "  WARN: no driver pod found — skipping Spark REST collection"
        echo '{"spark_rest": "unavailable"}' > /tmp/spark_rest.json
        return
    fi

    log "  Driver pod: ${DRIVER_POD}"

    # Port-forward Spark UI in background
    oc port-forward -n "${NAMESPACE}" "${DRIVER_POD}" 14040:4040 &>/dev/null &
    PF_PID=$!
    sleep 3

    SPARK_BASE="http://localhost:14040"

    # Get application info
    APP_INFO=$(curl -sf "${SPARK_BASE}/api/v1/applications" 2>/dev/null || echo "[]")
    APP_ID=$(echo "${APP_INFO}" | python3 -c "import json,sys; apps=json.load(sys.stdin); print(apps[0]['id'] if apps else '')" 2>/dev/null || echo "")

    if [[ -z "${APP_ID}" ]]; then
        log "  WARN: Spark UI not responding — application may have stopped"
        kill ${PF_PID} 2>/dev/null || true
        echo '{"spark_rest": "ui_not_available"}' > /tmp/spark_rest.json
        return
    fi

    log "  App ID: ${APP_ID}"

    # Stages
    STAGES=$(curl -sf "${SPARK_BASE}/api/v1/applications/${APP_ID}/stages" 2>/dev/null || echo "[]")
    # SQL executions
    SQL=$(curl -sf "${SPARK_BASE}/api/v1/applications/${APP_ID}/sql" 2>/dev/null || echo "[]")
    # Executors
    EXECUTORS=$(curl -sf "${SPARK_BASE}/api/v1/applications/${APP_ID}/executors" 2>/dev/null || echo "[]")
    # Environment (Spark config)
    ENV=$(curl -sf "${SPARK_BASE}/api/v1/applications/${APP_ID}/environment" 2>/dev/null || echo "{}")

    python3 - <<PYEOF > /tmp/spark_rest.json
import json, sys

stages = json.loads("""${STAGES}""")
executors = json.loads("""${EXECUTORS}""")
sql = json.loads("""${SQL}""")
env_data = json.loads("""${ENV}""")

# Summarize stages
stage_summary = []
total_task_ms, total_shuffle_read, total_shuffle_write = 0, 0, 0
for s in stages:
    entry = {
        "stage_id": s.get("stageId"),
        "name": s.get("name", "")[:80],
        "status": s.get("status"),
        "duration_ms": s.get("executorRunTime", 0),
        "tasks_complete": s.get("numCompleteTasks", 0),
        "tasks_failed": s.get("numFailedTasks", 0),
        "shuffle_read_bytes": s.get("shuffleReadBytes", 0),
        "shuffle_write_bytes": s.get("shuffleWriteBytes", 0),
        "input_bytes": s.get("inputBytes", 0),
        "output_bytes": s.get("outputBytes", 0),
    }
    total_task_ms += entry["duration_ms"]
    total_shuffle_read += entry["shuffle_read_bytes"]
    total_shuffle_write += entry["shuffle_write_bytes"]
    stage_summary.append(entry)

# Executor aggregate
executor_agg = {
    "count": len([e for e in executors if e.get("id") != "driver"]),
    "total_gc_time_ms": sum(e.get("totalGCTime", 0) for e in executors),
    "total_task_time_ms": sum(e.get("totalDuration", 0) for e in executors),
    "max_memory_used_bytes": max((e.get("memoryUsed", 0) for e in executors), default=0),
    "total_input_bytes": sum(e.get("totalInputBytes", 0) for e in executors),
    "total_shuffle_read": sum(e.get("totalShuffleRead", 0) for e in executors),
    "total_shuffle_write": sum(e.get("totalShuffleWrite", 0) for e in executors),
}

# SQL plan — detect GPU operators
gpu_ops = cpu_fallback_ops = 0
for q in sql:
    plan = q.get("physicalPlanDescription", "")
    # GPU ops contain "Gpu" prefix in RAPIDS plan (GpuHashAggregateExec, etc.)
    gpu_ops += plan.count("Gpu")
    # CPU fallbacks show "!Exec<" in RAPIDS explain output
    cpu_fallback_ops += plan.count("!Exec<")

result = {
    "spark_rest": {
        "app_id": "${APP_ID}",
        "stage_count": len(stage_summary),
        "total_task_time_ms": total_task_ms,
        "total_shuffle_read_bytes": total_shuffle_read,
        "total_shuffle_write_bytes": total_shuffle_write,
        "stages": stage_summary,
        "executor_aggregate": executor_agg,
        "sql_gpu_operator_count": gpu_ops,
        "sql_cpu_fallback_count": cpu_fallback_ops,
        "sql_gpu_coverage_pct": round(
            100.0 * gpu_ops / (gpu_ops + cpu_fallback_ops), 1
        ) if (gpu_ops + cpu_fallback_ops) > 0 else None,
    }
}
print(json.dumps(result, indent=2))
PYEOF

    kill ${PF_PID} 2>/dev/null || true
    log "  Spark REST: $(python3 -c "import json; d=json.load(open('/tmp/spark_rest.json')); s=d['spark_rest']; print(f\"{s.get('stage_count',0)} stages, GPU ops={s.get('sql_gpu_operator_count',0)}, CPU fallback={s.get('sql_cpu_fallback_count',0)}\")")"
}

# ── 2. DCGM GPU Metrics via Prometheus ────────────────────────────────────────

collect_dcgm_metrics() {
    log "Collecting DCGM GPU metrics..."

    # OCP monitoring Prometheus — internal URL (requires token)
    PROM_TOKEN=$(oc sa get-token prometheus-k8s -n openshift-monitoring 2>/dev/null || \
                 oc serviceaccounts get-token prometheus-k8s -n openshift-monitoring 2>/dev/null || echo "")
    PROM_URL="https://prometheus-k8s.openshift-monitoring.svc:9091"

    query_prometheus() {
        local metric="$1"
        local label_filter="${2:-}"
        curl -sk -H "Authorization: Bearer ${PROM_TOKEN}" \
            "${PROM_URL}/api/v1/query?query=${metric}${label_filter}" 2>/dev/null || echo '{"data":{"result":[]}}'
    }

    python3 - <<PYEOF > /tmp/dcgm_metrics.json
import json, subprocess, os

prom_token = """${PROM_TOKEN}"""
prom_url = "${PROM_URL}"

def query(metric):
    import urllib.request, urllib.parse, ssl
    url = f"{prom_url}/api/v1/query?query={urllib.parse.quote(metric)}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {prom_token}"})
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"data": {"result": []}, "error": str(e)}

def extract(resp):
    results = resp.get("data", {}).get("result", [])
    return [{"gpu": r["metric"].get("gpu", r["metric"].get("UUID", "?")),
             "hostname": r["metric"].get("Hostname", "?"),
             "value": float(r["value"][1]) if r.get("value") else None}
            for r in results]

metrics = {
    "gpu_utilization_pct":       extract(query("DCGM_FI_DEV_GPU_UTIL")),
    "gpu_memory_used_mb":        extract(query("DCGM_FI_DEV_FB_USED")),
    "gpu_memory_total_mb":       extract(query("DCGM_FI_DEV_FB_FREE + DCGM_FI_DEV_FB_USED")),
    "gpu_dram_active_ratio":     extract(query("DCGM_FI_PROF_DRAM_ACTIVE")),
    "gpu_sm_active_ratio":       extract(query("DCGM_FI_PROF_SM_ACTIVE")),
    "gpu_nvlink_bandwidth_mbps": extract(query("rate(DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL[1m])")),
    "gpu_power_usage_watts":     extract(query("DCGM_FI_DEV_POWER_USAGE")),
    "gpu_temperature_c":         extract(query("DCGM_FI_DEV_GPU_TEMP")),
}

# Compute averages across all GPUs for a single summary number per metric
summary = {}
for key, readings in metrics.items():
    vals = [r["value"] for r in readings if r["value"] is not None]
    if vals:
        summary[f"{key}_avg"] = round(sum(vals) / len(vals), 2)
        summary[f"{key}_max"] = round(max(vals), 2)

print(json.dumps({"dcgm": {"per_gpu": metrics, "summary": summary}}, indent=2))
PYEOF

    log "  DCGM: $(python3 -c "import json; d=json.load(open('/tmp/dcgm_metrics.json')); s=d['dcgm']['summary']; util=s.get('gpu_utilization_pct_avg','?'); mem=s.get('gpu_memory_used_mb_max','?'); print(f'avg GPU util={util}%, max GPU mem={mem}MB')")"
}

# ── 3. MLflow — fetch run metrics from RHOAI instance ─────────────────────────

collect_mlflow_metrics() {
    log "Collecting MLflow run metrics..."

    python3 - <<PYEOF > /tmp/mlflow_metrics.json
import json, os

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
if not tracking_uri:
    print(json.dumps({"mlflow": "MLFLOW_TRACKING_URI not set"}))
    exit(0)

try:
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    exp_name = "smartshop-feature-engineering"
    try:
        exp = client.get_experiment_by_name(exp_name)
    except Exception:
        exp = None

    if not exp:
        print(json.dumps({"mlflow": f"experiment '{exp_name}' not found"}))
        exit(0)

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=10,
    )

    run_summaries = []
    for run in runs:
        run_summaries.append({
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time_ms": run.info.start_time,
            "end_time_ms": run.info.end_time,
            "duration_s": round((run.info.end_time - run.info.start_time) / 1000, 1)
                          if run.info.end_time and run.info.start_time else None,
            "params": dict(run.data.params),
            "metrics": {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in run.data.metrics.items()},
            "tags": {k: v for k, v in run.data.tags.items()
                     if not k.startswith("mlflow.")},
        })

    # Compute GPU speedup if we have both cpu and rapids runs
    rapids_run = next((r for r in run_summaries if "rapids" in (r.get("run_name") or "").lower()), None)
    cpu_run    = next((r for r in run_summaries if "cpu" in (r.get("run_name") or "").lower()), None)
    speedup = None
    if rapids_run and cpu_run:
        t_rapids = rapids_run["metrics"].get("total_elapsed_s")
        t_cpu    = cpu_run["metrics"].get("total_elapsed_s")
        if t_rapids and t_cpu and t_rapids > 0:
            speedup = round(t_cpu / t_rapids, 2)

    result = {
        "mlflow": {
            "experiment": exp_name,
            "experiment_id": exp.experiment_id,
            "run_count": len(run_summaries),
            "gpu_vs_cpu_speedup": speedup,
            "runs": run_summaries,
        }
    }
    print(json.dumps(result, indent=2))
except ImportError:
    print(json.dumps({"mlflow": "mlflow package not installed in this environment"}))
except Exception as e:
    print(json.dumps({"mlflow": f"error: {e}"}))
PYEOF

    SPEEDUP=$(python3 -c "import json; d=json.load(open('/tmp/mlflow_metrics.json')); print(d.get('mlflow',{}).get('gpu_vs_cpu_speedup','?'))")
    log "  MLflow: GPU vs CPU speedup=${SPEEDUP}×"
}

# ── 4. Slurm / FSDP TrainJob metrics ──────────────────────────────────────────

collect_slurm_metrics() {
    local job_name="$1"
    log "Collecting Slurm/FSDP TrainJob metrics for ${job_name}..."

    python3 - <<PYEOF > /tmp/slurm_metrics.json
import json, subprocess

job_name = "${job_name}"
namespace = "${NAMESPACE}"

def oc(*args):
    result = subprocess.run(["oc"] + list(args), capture_output=True, text=True)
    return result.stdout.strip()

# Get TrainJob status
trainjob_json = oc("get", "trainjob", job_name, "-n", namespace, "-o", "json")
try:
    tj = json.loads(trainjob_json)
    status = tj.get("status", {})
    conditions = status.get("conditions", [])
    start_time = tj.get("metadata", {}).get("creationTimestamp")
    completion_time = next((c.get("lastTransitionTime") for c in conditions
                            if c.get("type") == "Complete" and c.get("status") == "True"), None)
except Exception:
    tj = {}; status = {}; start_time = None; completion_time = None

# Parse NCCL debug logs for bandwidth metrics
# Look for lines like: "NCCL INFO AllReduce: coll 0 nchunks 1 nsteps 4 ... algorithm TREE proto LL128"
nccl_bandwidth_entries = []
worker_pods = oc("get", "pod", "-n", namespace, "-l", f"training.kubeflow.org/job-name={job_name}",
                 "-o", "jsonpath={.items[*].metadata.name}").split()

for pod in worker_pods[:4]:  # cap to avoid log overload
    logs = oc("logs", "-n", namespace, pod, "--tail=500")
    for line in logs.splitlines():
        if "Avg bus bandwidth" in line or "busBw" in line:
            nccl_bandwidth_entries.append({"pod": pod, "line": line.strip()})
        if "[METRIC]" in line:
            nccl_bandwidth_entries.append({"pod": pod, "metric_line": line.strip()})

result = {
    "slurm_fsdp": {
        "job_name": job_name,
        "start_time": start_time,
        "completion_time": completion_time,
        "conditions": conditions,
        "worker_pods": worker_pods,
        "nccl_entries": nccl_bandwidth_entries[:50],
    }
}
print(json.dumps(result, indent=2))
PYEOF

    log "  Slurm FSDP: $(python3 -c "import json; d=json.load(open('/tmp/slurm_metrics.json')); s=d['slurm_fsdp']; print(f\"pods={len(s.get('worker_pods',[]))}, nccl_entries={len(s.get('nccl_entries',[]))}\")")"
}

# ── 5. Feast materialization metrics ──────────────────────────────────────────

collect_feast_metrics() {
    log "Collecting Feast metrics..."

    FEAST_POD=$(oc get pod -n "${NAMESPACE}" -l "feast.dev/name=smartshop-feast" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    python3 - <<PYEOF > /tmp/feast_metrics.json
import json, subprocess

feast_pod = "${FEAST_POD}"
namespace = "${NAMESPACE}"

if not feast_pod:
    print(json.dumps({"feast": "pod not found"}))
    exit(0)

def oc(*args):
    r = subprocess.run(["oc"] + list(args), capture_output=True, text=True)
    return r.stdout.strip()

# Scrape Feast server metrics endpoint (if exposed)
feast_metrics_raw = oc("exec", "-n", namespace, feast_pod, "--",
                       "curl", "-sf", "http://localhost:6566/metrics")

# Parse materialization stats from pod logs
logs = oc("logs", "-n", namespace, feast_pod, "--tail=500")
mat_lines = [l for l in logs.splitlines() if "materialize" in l.lower() or "written" in l.lower()]

print(json.dumps({
    "feast": {
        "pod": feast_pod,
        "raw_metrics": feast_metrics_raw[:2000] if feast_metrics_raw else None,
        "materialization_log_lines": mat_lines[:30],
    }
}, indent=2))
PYEOF

    log "  Feast: done"
}

# ── Bundle and upload ──────────────────────────────────────────────────────────

bundle_and_upload() {
    log "Bundling metrics into ${OUTPUT_FILE}..."

    python3 - <<PYEOF
import json, os, datetime

bundle = {
    "bundle_id": "${RUN_ID}",
    "run_type": "${RUN_TYPE}",
    "collected_at": datetime.datetime.utcnow().isoformat() + "Z",
    "cluster": {
        "domain": os.environ.get("OC_CLUSTER_DOMAIN", ""),
        "namespace": "${NAMESPACE}",
    },
}

for fname in ["/tmp/spark_rest.json", "/tmp/dcgm_metrics.json",
              "/tmp/mlflow_metrics.json", "/tmp/slurm_metrics.json",
              "/tmp/feast_metrics.json"]:
    if os.path.exists(fname):
        try:
            bundle.update(json.load(open(fname)))
        except Exception:
            pass

with open("${OUTPUT_FILE}", "w") as f:
    json.dump(bundle, f, indent=2)

print(json.dumps(bundle, indent=2))
PYEOF

    s3_upload "${OUTPUT_FILE}" "${RUN_ID}_metrics_bundle.json"
    log "Bundle complete: s3://${S3_MODELS_BUCKET}/metrics/${RUN_ID}_metrics_bundle.json"
}

# ── Print comparison summary ───────────────────────────────────────────────────

print_summary() {
    log ""
    log "════════════════════════════════════════════════════════════"
    log " SmartShop Observability Bundle — ${RUN_TYPE} run"
    log "════════════════════════════════════════════════════════════"
    python3 - <<PYEOF
import json, os

bundle = json.load(open("${OUTPUT_FILE}"))

spark = bundle.get("spark_rest", {})
dcgm  = bundle.get("dcgm", {}).get("summary", {})
ml    = bundle.get("mlflow", {})
fsdp  = bundle.get("slurm_fsdp", {})

print(f"\n  Spark")
print(f"    Stages         : {spark.get('stage_count', '?')}")
print(f"    Total task ms  : {spark.get('total_task_time_ms', '?'):,}" if isinstance(spark.get('total_task_time_ms'), int) else f"    Total task ms  : {spark.get('total_task_time_ms', '?')}")
print(f"    Shuffle read   : {spark.get('total_shuffle_read_bytes', 0) // 1_048_576} MB")
print(f"    Shuffle write  : {spark.get('total_shuffle_write_bytes', 0) // 1_048_576} MB")
print(f"    GPU operators  : {spark.get('sql_gpu_operator_count', 'N/A')}")
print(f"    CPU fallbacks  : {spark.get('sql_cpu_fallback_count', 'N/A')}")
print(f"    GPU coverage % : {spark.get('sql_gpu_coverage_pct', 'N/A')}")

print(f"\n  GPU (DCGM)")
print(f"    Avg utilization: {dcgm.get('gpu_utilization_pct_avg', '?')} %")
print(f"    Max mem used   : {dcgm.get('gpu_memory_used_mb_max', '?')} MB")
print(f"    Avg power      : {dcgm.get('gpu_power_usage_watts_avg', '?')} W")
print(f"    SM active avg  : {dcgm.get('gpu_sm_active_ratio_avg', '?')}")

print(f"\n  MLflow")
speedup = ml.get("gpu_vs_cpu_speedup")
print(f"    GPU speedup    : {speedup}× vs CPU" if speedup else "    GPU speedup    : (run CPU baseline to compute)")
print(f"    Runs logged    : {ml.get('run_count', '?')}")

if fsdp:
    print(f"\n  FSDP / Slurm")
    print(f"    Worker pods    : {len(fsdp.get('worker_pods', []))}")
    print(f"    NCCL entries   : {len(fsdp.get('nccl_entries', []))}")

print(f"\n  Bundle: s3://${S3_MODELS_BUCKET}/metrics/${RUN_ID}_metrics_bundle.json")
PYEOF
    log "════════════════════════════════════════════════════════════"
}

# ── Main ───────────────────────────────────────────────────────────────────────

main() {
    log "Starting metrics collection | run_type=${RUN_TYPE} | id=${RUN_ID}"

    case "${RUN_TYPE}" in
        rapids)
            collect_spark_rest "${APP_NAME:-smartshop-feature-engineering-rapids}"
            collect_dcgm_metrics
            collect_mlflow_metrics
            ;;
        cpu)
            collect_spark_rest "${APP_NAME:-smartshop-feature-engineering-cpu-baseline}"
            collect_dcgm_metrics
            collect_mlflow_metrics
            ;;
        slurm)
            collect_slurm_metrics "${TRAINJOB_NAME:-smartshop-llm-finetune}"
            collect_dcgm_metrics
            collect_mlflow_metrics
            ;;
        feast)
            collect_feast_metrics
            collect_mlflow_metrics
            ;;
        all)
            collect_spark_rest "${APP_NAME:-smartshop-feature-engineering-rapids}"
            collect_dcgm_metrics
            collect_mlflow_metrics
            collect_slurm_metrics "${TRAINJOB_NAME:-smartshop-llm-finetune}"
            collect_feast_metrics
            ;;
        *)
            echo "Unknown RUN_TYPE: ${RUN_TYPE}. Use: rapids | cpu | slurm | feast | all"
            exit 1
            ;;
    esac

    bundle_and_upload
    print_summary
}

main "$@"
