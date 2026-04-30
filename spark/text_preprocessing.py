"""Spark Job B: Text Preprocessing for LLM Fine-Tuning.

Processes raw reviews into instruction-tuning format for Mistral-7B:
  - Cleans and deduplicates review text
  - Creates instruction-format examples (summarization + sentiment)
  - Splits into train/val/test sets at scale

Outputs JSONL files ready for fine-tuning.

Usage:
    spark-submit spark/text_preprocessing.py \
        --input s3a://smartshop-raw/ \
        --output s3a://smartshop-features/llm_data/
"""

import argparse
import json
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

try:
    from spark.utils.mlflow_metrics import SparkRunLogger
except ImportError:
    SparkRunLogger = None


def _log_metric(key, value, unit=""):
    tag = f"[METRIC] {key}={value}"
    if unit:
        tag += f" {unit}"
    print(tag)


def create_spark_session(app_name: str = "SmartShop-TextPreprocessing") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


@F.udf(T.StringType())
def build_summarization_prompt(title, text, rating):
    """Build an instruction-tuning example for review summarization."""
    if not text or len(str(text)) < 50:
        return None
    try:
        r = int(float(rating)) if rating is not None else 3
    except (ValueError, TypeError):
        r = 3
    sentiment = "positive" if r >= 4 else "negative" if r <= 2 else "neutral"
    example = {
        "instruction": (
            "Summarize the following product review in 1-2 sentences. "
            "Include the overall sentiment (positive/negative/neutral)."
        ),
        "input": f"Product: {title or 'Unknown'}\nReview: {text}",
        "output": "",
        "metadata": {"rating": r, "sentiment": sentiment},
    }
    return json.dumps(example)


@F.udf(T.StringType())
def build_qa_prompt(title, text, rating):
    """Build an instruction-tuning example for product Q&A."""
    if not text or len(str(text)) < 100:
        return None
    try:
        r = int(float(rating)) if rating is not None else 3
    except (ValueError, TypeError):
        r = 3
    example = {
        "instruction": (
            "Based on the following product review, answer questions about the product. "
            "Be helpful, accurate, and concise."
        ),
        "input": f"Product: {title or 'Unknown'}\nReview ({r}/5 stars): {text}",
        "output": "",
        "metadata": {"rating": r},
    }
    return json.dumps(example)


def clean_reviews(df):
    """Clean and deduplicate review text."""
    return (
        df
        # Remove empty/very short reviews
        .filter(F.length(F.col("text")) >= 50)
        # Remove duplicate reviews (same user + same text)
        .dropDuplicates(["user_id", "text"])
        # Clean text
        .withColumn("text", F.regexp_replace("text", r"<[^>]+>", ""))  # strip HTML
        .withColumn("text", F.regexp_replace("text", r"\s+", " "))  # normalize whitespace
        .withColumn("text", F.trim("text"))
        # Ensure title exists
        .withColumn("title", F.coalesce(F.col("title"), F.lit("Unknown Product")))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input path for raw reviews")
    parser.add_argument("--output", required=True, help="Output path for JSONL files")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Max examples to generate (for sample mode)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.9, help="Train split ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.05, help="Validation split ratio"
    )
    args = parser.parse_args()

    job_start = time.time()
    spark = create_spark_session()

    mlflow_logger = SparkRunLogger(spark, experiment="smartshop-text-preprocessing") if SparkRunLogger else None
    _run_ctx = mlflow_logger.start_run(run_name="text-preprocessing") if mlflow_logger else None
    if _run_ctx:
        _run_ctx.__enter__()

    def _metric(key, value, unit=""):
        _log_metric(key, value, unit)
        if mlflow_logger:
            mlflow_logger.log_metric(key, value)

    print(f"Reading reviews from {args.input}")
    t = time.time()
    input_paths = [p.strip() for p in args.input.split(",") if p.strip()]
    df = spark.read.parquet(*input_paths)

    # Normalize columns
    if "asin" in df.columns and "parent_asin" not in df.columns:
        df = df.withColumnRenamed("asin", "parent_asin")

    # Guard: ensure text and rating columns exist
    if "text" not in df.columns:
        df = df.withColumn("text", F.lit(""))
    if "rating" not in df.columns:
        df = df.withColumn("rating", F.lit(3))
    if "title" not in df.columns:
        df = df.withColumn("title", F.lit("Unknown Product"))

    total_input = df.count()
    _metric("read_elapsed_s", round(time.time() - t, 2), "s")
    _metric("total_input_reviews", total_input)

    print("Cleaning and deduplicating...")
    df = clean_reviews(df)

    if args.max_examples:
        df = df.limit(args.max_examples)

    clean_count = df.count()
    _metric("clean_reviews", clean_count)

    # Build instruction-tuning examples
    print("Building summarization prompts...")
    summarization_df = (
        df.withColumn(
            "jsonl", build_summarization_prompt(F.col("title"), F.col("text"), F.col("rating"))
        )
        .filter(F.col("jsonl").isNotNull())
        .select("jsonl")
    )

    print("Building Q&A prompts...")
    qa_df = (
        df.withColumn(
            "jsonl", build_qa_prompt(F.col("title"), F.col("text"), F.col("rating"))
        )
        .filter(F.col("jsonl").isNotNull())
        .select("jsonl")
    )

    # Combine all examples
    all_examples = summarization_df.union(qa_df)

    # Split into train/val/test
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    train_df, val_df, test_df = all_examples.randomSplit(
        [args.train_ratio, args.val_ratio, test_ratio],
        seed=42,
    )

    # Write JSONL
    print("Writing train/val/test splits...")
    t = time.time()
    train_df.write.mode("overwrite").text(f"{args.output}/train")
    val_df.write.mode("overwrite").text(f"{args.output}/val")
    test_df.write.mode("overwrite").text(f"{args.output}/test")

    n_train = train_df.count()
    n_val = val_df.count()
    n_test = test_df.count()
    n_total = n_train + n_val + n_test
    write_elapsed = time.time() - t
    job_elapsed = time.time() - job_start

    _metric("train_examples", n_train)
    _metric("val_examples", n_val)
    _metric("test_examples", n_test)
    _metric("total_examples", n_total)
    _metric("write_elapsed_s", round(write_elapsed, 2), "s")
    _metric("total_elapsed_s", round(job_elapsed, 2), "s")
    _metric("throughput_rows_per_s", int(total_input / job_elapsed) if job_elapsed > 0 else 0, "rows/s")

    print(f"  Train: {n_train:,} examples")
    print(f"  Val:   {n_val:,} examples")
    print(f"  Test:  {n_test:,} examples")

    if mlflow_logger:
        mlflow_logger.finalize(output_path="/tmp/text_preprocessing_metrics.json")
    if _run_ctx:
        _run_ctx.__exit__(None, None, None)

    print(
        f"\n{'='*60}\n"
        f"Text preprocessing complete.\n"
        f"  Input reviews : {total_input:,}\n"
        f"  Clean reviews : {clean_count:,}\n"
        f"  Total examples: {n_total:,} (train/val/test)\n"
        f"  Elapsed       : {job_elapsed:.1f}s\n"
        f"{'='*60}"
    )
    spark.stop()


if __name__ == "__main__":
    main()
