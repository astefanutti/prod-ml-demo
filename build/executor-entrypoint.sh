#!/bin/bash
# Spark k8s executor entrypoint — mirrors official Spark 3.5.3 Docker image pattern.
#
# The k8s driver sets CMD=["executor"] and populates executor config via env vars:
#   SPARK_DRIVER_URL, SPARK_EXECUTOR_ID, SPARK_EXECUTOR_POD_IP, SPARK_EXECUTOR_POD_NAME,
#   SPARK_EXECUTOR_CORES, SPARK_EXECUTOR_MEMORY, SPARK_APPLICATION_ID,
#   SPARK_RESOURCE_PROFILE_ID, SPARK_JAVA_OPT_0..N
#
# Key differences from spark-class approach:
#   - Calls java directly (avoids spark.launcher.Main overhead and $@ arg-passing pitfalls)
#   - Uses KubernetesExecutorBackend (official k8s class, handles pod lifecycle + GenerateExecID)
#   - Passes --podName for proper executor tracking by the driver
#   - Collects SPARK_JAVA_OPT_* into an array for safe handling of opts with spaces

set -e

SPARK_HOME=$(python3 -c 'import pyspark, os; print(os.path.dirname(pyspark.__file__))')
export SPARK_HOME
export PATH="${SPARK_HOME}/bin:${PATH}"

case "$1" in
  executor)
    # Collect SPARK_JAVA_OPT_0..N into an array (same as official entrypoint.sh)
    declare -a JAVA_OPTS_ARRAY=()
    for i in $(seq 0 50); do
      var="SPARK_JAVA_OPT_${i}"
      val="${!var}"
      [ -n "$val" ] && JAVA_OPTS_ARRAY+=("$val")
    done

    SPARK_CLASSPATH="${SPARK_HOME}/conf:${SPARK_HOME}/jars/*"

    exec "${JAVA_HOME}/bin/java" \
      "${JAVA_OPTS_ARRAY[@]}" \
      -Xms"${SPARK_EXECUTOR_MEMORY}" \
      -Xmx"${SPARK_EXECUTOR_MEMORY}" \
      -cp "${SPARK_CLASSPATH}" \
      org.apache.spark.scheduler.cluster.k8s.KubernetesExecutorBackend \
      --driver-url     "${SPARK_DRIVER_URL}" \
      --executor-id    "${SPARK_EXECUTOR_ID}" \
      --cores          "${SPARK_EXECUTOR_CORES}" \
      --app-id         "${SPARK_APPLICATION_ID}" \
      --hostname       "${SPARK_EXECUTOR_POD_IP}" \
      --resourceProfileId "${SPARK_RESOURCE_PROFILE_ID:-0}" \
      --podName        "${SPARK_EXECUTOR_POD_NAME}"
    ;;
  *)
    exec "$@"
    ;;
esac
