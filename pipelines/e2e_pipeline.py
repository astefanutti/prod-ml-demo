"""Kubeflow Pipeline: End-to-end SmartShop AI workflow.

Orchestrates the full ML lifecycle:
  1. Spark preprocessing (features + text + embeddings)
  2. Feast materialization
  3. Distributed training (rec model DDP + LLM FSDP)
  4. Model registration
  5. Deployment to KServe

Canonical S3 bucket layout:
  smartshop-raw/raw/              - Amazon Reviews raw parquet
  smartshop-raw/raw/metadata/     - product metadata
  smartshop-features/             - engineered user/item features
  smartshop-features/llm_data/    - instruction-tuning JSONL for LLM
  smartshop-embeddings/review_embeddings/ - sentence embeddings for Feast vector store
  smartshop-models/recommendation/- trained Two-Tower checkpoint
  smartshop-models/llm-adapter/   - QLoRA LoRA adapter weights

Images resolve from ImageStream short names inside the smartshop namespace
(lookupPolicy.local: true). Override via env vars SPARK_JOBS_IMAGE,
REC_TRAINER_IMAGE, LLM_TRAINER_IMAGE if using quay.io images.

Usage:
    python pipelines/e2e_pipeline.py  # Compile to YAML
    # Then upload to Kubeflow Pipelines UI or submit via SDK
"""

import os

from kfp import dsl

_SPARK_IMAGE = os.environ.get("SPARK_JOBS_IMAGE", "spark-jobs:latest")
_REC_TRAINER_IMAGE = os.environ.get("REC_TRAINER_IMAGE", "rec-trainer:latest")
_LLM_TRAINER_IMAGE = os.environ.get("LLM_TRAINER_IMAGE", "llm-trainer:latest")


@dsl.component(
    base_image=_SPARK_IMAGE,
    packages_to_install=["kfp"],
)
def spark_feature_engineering(
    input_path: str,
    output_path: str,
    metadata_path: str,
) -> str:
    """Run Spark feature engineering job."""
    import subprocess

    cmd = [
        "spark-submit",
        "--master", "local[*]",
        "feature_engineering.py",
        "--input", input_path,
        "--output", output_path,
        "--metadata-input", metadata_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(result.stdout)
    return output_path


@dsl.component(
    base_image=_SPARK_IMAGE,
    packages_to_install=["kfp"],
)
def spark_text_preprocessing(
    input_path: str,
    output_path: str,
) -> str:
    """Run Spark text preprocessing for LLM fine-tuning data."""
    import subprocess

    cmd = [
        "spark-submit",
        "--master", "local[*]",
        "text_preprocessing.py",
        "--input", input_path,
        "--output", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(result.stdout)
    return output_path


@dsl.component(
    base_image=_SPARK_IMAGE,
    packages_to_install=["kfp"],
)
def spark_embedding_generation(
    input_path: str,
    output_path: str,
) -> str:
    """Run Spark embedding generation for vector store."""
    import subprocess

    cmd = [
        "spark-submit",
        "--master", "local[*]",
        "embedding_generation.py",
        "--input", input_path,
        "--output", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(result.stdout)
    return output_path


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["feast[redis]"],
)
def feast_materialize(feast_repo_path: str) -> str:
    """Apply and materialize Feast feature views."""
    import subprocess
    from datetime import datetime

    subprocess.run(["feast", "apply"], cwd=feast_repo_path, check=True)

    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    subprocess.run(
        ["feast", "materialize-incremental", now],
        cwd=feast_repo_path,
        check=True,
    )
    return "materialized"


@dsl.component(
    base_image=_REC_TRAINER_IMAGE,
    packages_to_install=["kfp"],
)
def train_recommendation_model(
    data_dir: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 1024,
) -> str:
    """Train the Two-Tower recommendation model with DDP."""
    import subprocess

    cmd = [
        "torchrun",
        "--nproc_per_node=4",
        "training/recommendation/train.py",
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(result.stdout)
    return output_dir


@dsl.component(
    base_image=_LLM_TRAINER_IMAGE,
    packages_to_install=["kfp"],
)
def finetune_llm(
    data_dir: str,
    output_dir: str,
    epochs: int = 3,
    sample: bool = False,
) -> str:
    """Fine-tune Mistral-7B with QLoRA."""
    import subprocess

    cmd = [
        "python",
        "training/llm/finetune.py",
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--epochs", str(epochs),
    ]
    if sample:
        cmd.append("--sample")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(result.stdout)
    return output_dir


@dsl.pipeline(
    name="SmartShop AI E2E Pipeline",
    description="End-to-end ML pipeline: Spark preprocessing -> Feast -> Training -> Serving",
)
def smartshop_pipeline(
    raw_data_path: str = "s3://smartshop-raw/raw/",
    metadata_path: str = "s3://smartshop-raw/raw/metadata/",
    features_path: str = "s3://smartshop-features/",
    llm_data_path: str = "s3://smartshop-features/llm_data/",
    embeddings_path: str = "s3://smartshop-embeddings/review_embeddings/",
    rec_model_path: str = "s3://smartshop-models/recommendation/",
    llm_adapter_path: str = "s3://smartshop-models/llm-adapter/",
    feast_repo_path: str = "feast/feature_repo",
    sample: bool = False,
):
    # Stage 1: Spark preprocessing (all three jobs run in parallel)
    features_task = spark_feature_engineering(
        input_path=raw_data_path,
        output_path=features_path,
        metadata_path=metadata_path,
    )

    text_task = spark_text_preprocessing(
        input_path=raw_data_path,
        output_path=llm_data_path,
    )

    embeddings_task = spark_embedding_generation(
        input_path=raw_data_path,
        output_path=embeddings_path,
    )

    # Stage 2: Feast materialization (after features + embeddings are ready)
    feast_task = feast_materialize(feast_repo_path=feast_repo_path)
    feast_task.after(features_task, embeddings_task)

    # Stage 3: Training (after Feast is materialized)
    rec_task = train_recommendation_model(
        data_dir=features_path,
        output_dir=rec_model_path,
    )
    rec_task.after(feast_task)

    llm_task = finetune_llm(
        data_dir=llm_data_path,
        output_dir=llm_adapter_path,
        sample=sample,
    )
    llm_task.after(feast_task)


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=smartshop_pipeline,
        package_path="pipelines/smartshop_pipeline.yaml",
    )
    print("Pipeline compiled to pipelines/smartshop_pipeline.yaml")
