# Optional: GPU-Accelerated Spark with RAPIDS

## Overview

The [RAPIDS Accelerator for Apache Spark](https://nvidia.github.io/spark-rapids/) is a Spark plugin that offloads DataFrame and Spark SQL operations to GPUs via NVIDIA cuDF. It intercepts physical plan nodes at execution time and replaces them with GPU-accelerated equivalents -- existing PySpark code runs **unchanged**.

For SmartShop AI, this means GPU acceleration across the **entire pipeline** -- not just training and inference, but data preprocessing too.

## How It Works

```
PySpark Code (unchanged)
        │
        ▼
Spark Catalyst Optimizer
        │
        ▼
Physical Plan
        │
    ┌───┴───┐
    │ RAPIDS │  spark-rapids plugin intercepts plan nodes
    │ Plugin │  and replaces supported ops with GPU kernels
    └───┬───┘
        │
        ▼
GPU Execution (cuDF)      ← groupBy, join, agg, filter, sort
CPU Fallback              ← Python UDFs, unsupported ops
```

The `rapids-4-spark` JAR is added to the Spark classpath and activated via `spark.plugins=com.nvidia.spark.SQLPlugin`. No changes to application code are required for DataFrame/SQL operations.

## Per-Job Analysis

### Job A: Feature Engineering -- High Benefit

**Operations**: groupBy, agg, join, window functions over 233M rows across 3 categories.

This is the ideal RAPIDS workload. The entire job is DataFrame operations:
- `groupBy("user_id").agg(avg, count, collect_set)` for user features
- `groupBy("parent_asin").agg(avg, count)` for item features
- Large join between reviews and metadata tables
- Window functions for interaction features

Expected speedup: **10-30x** on shuffle-heavy aggregation workloads. This is the recommended job for RAPIDS.

### Job B: Text Preprocessing -- Moderate Benefit

**Operations**: filter, dedup, regex, plus Python UDFs for prompt formatting.

DataFrame operations (filter nulls, drop duplicates, regex cleaning) run on GPU, but the `build_summarization_prompt` and `build_qa_prompt` Python UDFs fall back to CPU. The net benefit depends on the ratio of DataFrame ops to UDF execution time.

Expected speedup: **2-5x overall** (DataFrame portion is fast, but UDFs remain CPU-bound).

### Job C: Embedding Generation -- Low Incremental Benefit

**Operations**: lightweight DataFrame ops around a pandas UDF running sentence-transformer inference.

The GPU is already saturated by the sentence-transformer model inside the pandas UDF. The surrounding DataFrame operations (select, filter) are minimal. Adding RAPIDS here would compete for GPU memory with the embedding model.

Expected speedup: **Negligible** -- skip RAPIDS for this job.

## Summary

| Job | Primary Operations | RAPIDS Benefit | Recommendation |
|---|---|---|---|
| A: Feature Engineering | groupBy, agg, join | High (10-30x) | Use RAPIDS |
| B: Text Preprocessing | filter, dedup, regex + Python UDFs | Moderate (2-5x) | Optional |
| C: Embedding Generation | pandas UDF (sentence-transformer) | Low | Skip |

## Infrastructure Requirements

- **GPU nodes**: Spark executors need NVIDIA GPUs (T4, A10G, or better)
- **Container image**: Based on `nvcr.io/nvidia/spark:3.5.0-rapids24.10` (includes CUDA runtime + rapids-4-spark JAR)
- **Device plugin**: NVIDIA GPU device plugin deployed on OpenShift (standard for OpenShift AI clusters)

## Usage

### Build the RAPIDS-enabled image

```bash
make build-image-spark-rapids
```

### Submit Job A with RAPIDS

```bash
make spark-features-rapids
```

Or apply the SparkApplication directly:

```bash
kubectl apply -f infrastructure/openshift/spark-application-rapids.yaml -n smartshop
```

## Key Spark Configuration

| Property | Value | Purpose |
|---|---|---|
| `spark.plugins` | `com.nvidia.spark.SQLPlugin` | Activate RAPIDS plugin |
| `spark.rapids.sql.enabled` | `true` | Enable GPU execution |
| `spark.rapids.sql.concurrentGpuTasks` | `2` | GPU task parallelism per executor |
| `spark.executor.resource.gpu.amount` | `1` | One GPU per executor |
| `spark.task.resource.gpu.amount` | `0.5` | Two tasks share each GPU |
| `spark.rapids.memory.pinnedPool.size` | `2g` | Pinned memory for host-to-GPU transfers |
| `spark.sql.files.maxPartitionBytes` | `512m` | Larger partitions to amortize GPU kernel launch |

## Demo Narrative

RAPIDS fits the SmartShop AI story as a natural extension:

> "We're running GPU acceleration across the entire ML lifecycle. Training uses DDP and FSDP on GPUs. Inference uses vLLM on GPUs. And now even our Spark ETL pipelines run on GPUs -- processing 233M reviews through the same hardware, with zero code changes."

This reinforces the OpenShift AI platform message: heterogeneous workloads (ETL, training, inference) all managed on the same GPU-enabled infrastructure.
