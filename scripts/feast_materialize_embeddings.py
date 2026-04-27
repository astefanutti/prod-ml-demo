#!/usr/bin/env python3
"""
Feast materialize wrapper for review_embeddings → Milvus.

Pre-creates the Spark session with all batch_engine configs applied BEFORE
Feast's SparkComputeEngine calls SparkSession.getActiveSession(). This ensures
configs like spark.sql.sources.useV1SourceList are set on the active session,
working around the bug in feast/infra/compute_engines/spark/utils.py where
getActiveSession() returns the existing session without applying spark_config.

Bug fix tracked at: https://github.com/feast-dev/feast/pull/XXXX
Patched in: feast/infra/compute_engines/spark/utils.py (PR #6317 fork)

Usage (from inside the Feast pod offline container):
  python3 /tmp/feast_materialize_embeddings.py \
      --config /tmp/feast-milvus-repo/feature_store.yaml \
      --start  2023-06-01T00:00:00 \
      --end    2023-06-08T00:00:00
"""

import argparse
import os
import sys
import yaml

from datetime import datetime, timezone

# ── Monkey-patch feast.infra.compute_engines.spark.utils ─────────────────────
# The site-packages copy may be read-only in the container.  Load the patched
# version from /tmp/feast_spark_utils_patch.py (uploaded via oc exec) BEFORE
# anything imports the module.
_PATCH = "/tmp/feast_spark_utils_patch.py"
if os.path.exists(_PATCH):
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("feast.infra.compute_engines.spark.utils", _PATCH)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    sys.modules["feast.infra.compute_engines.spark.utils"] = _mod
    print(f"[patch] feast.infra.compute_engines.spark.utils loaded from {_PATCH}")

from pyspark import SparkConf
from pyspark.sql import SparkSession

from feast import FeatureStore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to feature_store.yaml directory")
    p.add_argument("--start", required=True, help="Materialize start timestamp (ISO 8601)")
    p.add_argument("--end", required=True, help="Materialize end timestamp (ISO 8601)")
    return p.parse_args()


def load_batch_engine_conf(config_dir: str) -> dict:
    """Extract batch_engine spark configs from the feature_store.yaml."""
    yaml_path = os.path.join(config_dir, "feature_store.yaml")
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    engine = cfg.get("batch_engine", {})
    # All keys in batch_engine are Spark conf keys (flat dict)
    return {k: str(v) for k, v in engine.items() if k not in ("type", "partitions")}


def pre_create_spark_session(spark_conf: dict) -> SparkSession:
    """
    Build the SparkSession explicitly so all configs are applied upfront.
    Feast's utils.py calls SparkSession.getActiveSession() — if we create
    the session here first, it returns this already-configured session.
    """
    master = spark_conf.pop("spark.master", "local[*]")
    conf = SparkConf().setMaster(master).setAll(list(spark_conf.items()))
    session = SparkSession.builder.config(conf=conf).getOrCreate()

    # Force-apply SQL configs that getOrCreate() may have dropped on a reused session
    _SQL_PREFIXES = ("spark.sql.", "spark.hadoop.")
    for k, v in spark_conf.items():
        if any(k.startswith(p) for p in _SQL_PREFIXES):
            try:
                session.conf.set(k, v)
            except Exception:
                pass

    print(f"SparkSession created — master: {session.sparkContext.master}")
    v1_list = session.conf.get("spark.sql.sources.useV1SourceList", "<not set>")
    print(f"  spark.sql.sources.useV1SourceList = {v1_list}")
    return session


def main():
    args = parse_args()

    print(f"Loading batch_engine config from {args.config}")
    spark_conf = load_batch_engine_conf(args.config)

    print("Pre-creating SparkSession with full batch_engine config...")
    spark = pre_create_spark_session(spark_conf)  # noqa: F841 — must stay alive

    start_ts = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_ts = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    print(f"Opening FeatureStore at {args.config}")
    store = FeatureStore(repo_path=args.config)

    print(f"Running materialize {start_ts} → {end_ts} for review_embeddings")
    store.materialize(
        start_date=start_ts,
        end_date=end_ts,
        feature_views=["review_embeddings"],
    )

    print("Materialize complete.")


if __name__ == "__main__":
    main()
