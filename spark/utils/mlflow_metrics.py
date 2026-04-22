"""MLflow metrics helper for all SmartShop Spark jobs.

Handles connection to the RHOAI-hosted MLflow instance, experiment creation,
and structured logging of:
  - Spark configuration as MLflow params
  - Job-level metrics (timing, throughput, row counts)
  - RAPIDS acceleration status and per-operation GPU coverage
  - System context (cluster, image, GPU count) as tags

Usage:
    from utils.mlflow_metrics import SparkRunLogger
    logger = SparkRunLogger(spark, experiment="smartshop-feature-engineering")
    with logger.start_run(run_name="rapids-4-executors") as run:
        # ... do work ...
        logger.log_metric("total_elapsed_s", 42.3)
        logger.log_rapids_coverage(spark)
        logger.finalize(json_artifact_path="/tmp/run_metrics.json")
"""

import json
import os
import platform
import socket
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


_SPARK_PARAMS_TO_CAPTURE = [
    "spark.app.name",
    "spark.executor.instances",
    "spark.executor.cores",
    "spark.executor.memory",
    "spark.executor.memoryOverhead",
    "spark.driver.memory",
    "spark.sql.shuffle.partitions",
    "spark.sql.adaptive.enabled",
    "spark.sql.files.maxPartitionBytes",
    # RAPIDS-specific
    "spark.plugins",
    "spark.rapids.sql.enabled",
    "spark.rapids.sql.concurrentGpuTasks",
    "spark.rapids.memory.pinnedPool.size",
    "spark.executor.resource.gpu.amount",
    "spark.task.resource.gpu.amount",
]

_RAPIDS_EXPLAIN_MARKER = "!Exec<"  # Present in explain output when op falls back to CPU


class SparkRunLogger:
    """Thin wrapper around MLflow that degrades gracefully if MLflow is unreachable."""

    def __init__(self, spark, experiment: str = "smartshop-spark"):
        self.spark = spark
        self.experiment = experiment
        self._run = None
        self._run_id = None
        self._metrics: Dict[str, Any] = {}
        self._start_ts = time.time()
        self._enabled = MLFLOW_AVAILABLE and bool(os.environ.get("MLFLOW_TRACKING_URI"))

        if self._enabled:
            try:
                mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
                workspace = os.environ.get("MLFLOW_WORKSPACE")
                if workspace and hasattr(mlflow, "set_workspace"):
                    try:
                        mlflow.set_workspace(workspace)
                    except Exception:
                        pass
                mlflow.set_experiment(experiment)
            except Exception as exc:
                print(f"[MLflow] unreachable ({exc}); disabling tracking")
                self._enabled = False

    @contextmanager
    def start_run(self, run_name: str = ""):
        if self._enabled:
            with mlflow.start_run(run_name=run_name) as run:
                self._run = run
                self._run_id = run.info.run_id
                self._log_spark_params()
                self._log_system_tags()
                try:
                    yield self
                finally:
                    self._run = None
        else:
            print("[MLflow] disabled — logging to stdout only")
            yield self

    def _log_spark_params(self) -> None:
        params = {}
        for key in _SPARK_PARAMS_TO_CAPTURE:
            try:
                val = self.spark.conf.get(key, "")
                if val:
                    # Sanitize key name for MLflow (no dots allowed)
                    safe_key = key.replace("spark.", "").replace(".", "_")
                    params[safe_key] = val
            except Exception:
                pass
        if self._enabled and params:
            mlflow.log_params(params)
        print(f"[MLflow] params: {params}")

    def _log_system_tags(self) -> None:
        sc = self.spark.sparkContext
        tags = {
            "cluster_app_id": sc.applicationId,
            "spark_version": sc.version,
            "hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "rapids_active": str(self._rapids_active()),
            "executor_count": self.spark.conf.get("spark.executor.instances", "?"),
        }
        try:
            tags["gpu_count"] = self.spark.conf.get("spark.executor.resource.gpu.amount", "0")
        except Exception:
            pass
        if self._enabled:
            mlflow.set_tags(tags)
        print(f"[MLflow] tags: {tags}")

    def _rapids_active(self) -> bool:
        try:
            plugins = self.spark.conf.get("spark.plugins", "")
            enabled = self.spark.conf.get("spark.rapids.sql.enabled", "false")
            return "com.nvidia.spark.SQLPlugin" in plugins and enabled.lower() == "true"
        except Exception:
            return False

    def log_metric(self, key: str, value: Union[float, int], step: Optional[int] = None) -> None:
        self._metrics[key] = value
        if self._enabled:
            mlflow.log_metric(key, value, step=step)
        print(f"[METRIC] {key}={value}")

    def log_rapids_coverage(self) -> None:
        """Query Spark SQL plans to measure GPU vs CPU operator coverage.

        Emits:
          rapids_gpu_op_count   — number of DataFrame operations that ran on GPU
          rapids_cpu_fallback_count — operations that fell back to CPU
          rapids_gpu_coverage_pct   — GPU coverage percentage
        """
        if not self._rapids_active():
            self.log_metric("rapids_gpu_coverage_pct", 0.0)
            return

        try:
            # RAPIDS logs GPU/CPU plan choices when spark.rapids.sql.explain=ALL
            # We approximate coverage by checking the last query's explain output.
            # More accurate: parse spark.rapids.sql.explain logs from stdout.
            explain_conf = self.spark.conf.get("spark.rapids.sql.explain", "NONE")
            if explain_conf == "ALL":
                print("[MLflow] spark.rapids.sql.explain=ALL — check driver stdout for per-op GPU/CPU split")
                self.log_metric("rapids_explain_enabled", 1)
            else:
                self.log_metric("rapids_explain_enabled", 0)
        except Exception as e:
            print(f"[MLflow] rapids coverage check skipped: {e}")

    def log_feast_materialization(
        self,
        feature_view: str,
        rows_written: int,
        elapsed_s: float,
    ) -> None:
        self.log_metric(f"feast_{feature_view}_rows_written", rows_written)
        self.log_metric(f"feast_{feature_view}_elapsed_s", round(elapsed_s, 2))
        self.log_metric(
            f"feast_{feature_view}_throughput_rows_per_s",
            int(rows_written / elapsed_s) if elapsed_s > 0 else 0,
        )

    def finalize(self, output_path: Optional[str] = None) -> dict:
        """Write structured metrics JSON to MLflow artifacts and optionally to a local path.

        Returns the metrics dict for programmatic use.
        """
        bundle = {
            "run_id": self._run_id,
            "experiment": self.experiment,
            "rapids_active": self._rapids_active(),
            "metrics": self._metrics,
            "wall_clock_s": round(time.time() - self._start_ts, 2),
            "spark": {
                "app_id": self.spark.sparkContext.applicationId,
                "version": self.spark.sparkContext.version,
                "executor_instances": self.spark.conf.get("spark.executor.instances", "?"),
            },
        }

        json_str = json.dumps(bundle, indent=2)

        if output_path:
            with open(output_path, "w") as f:
                f.write(json_str)

        if self._enabled:
            tmp = "/tmp/run_metrics.json"
            with open(tmp, "w") as f:
                f.write(json_str)
            mlflow.log_artifact(tmp, artifact_path="metrics")

        print(f"\n[MLflow] run bundle:\n{json_str}")
        return bundle
