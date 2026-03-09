"""Spark Job B: Text Preprocessing for LLM Fine-Tuning.

Processes raw reviews into instruction-tuning format for Mistral-7B:
  - Cleans and deduplicates review text
  - Creates instruction-format examples (summarization + sentiment)
  - Splits into train/val/test sets at scale

Outputs JSONL files ready for fine-tuning.

Usage:
    spark-submit spark/text_preprocessing.py \
        --input s3a://smartshop/raw/ \
        --output s3a://smartshop/llm_data/
"""

import argparse
import json

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


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

    sentiment = "positive" if int(rating) >= 4 else "negative" if int(rating) <= 2 else "neutral"

    example = {
        "instruction": (
            "Summarize the following product review in 1-2 sentences. "
            "Include the overall sentiment (positive/negative/neutral)."
        ),
        "input": f"Product: {title}\nReview: {text}",
        "output": "",  # Will be filled by the model during training
        "metadata": {"rating": int(rating), "sentiment": sentiment},
    }
    return json.dumps(example)


@F.udf(T.StringType())
def build_qa_prompt(title, text, rating):
    """Build an instruction-tuning example for product Q&A."""
    if not text or len(str(text)) < 100:
        return None

    example = {
        "instruction": (
            "Based on the following product review, answer questions about the product. "
            "Be helpful, accurate, and concise."
        ),
        "input": f"Product: {title}\nReview ({rating}/5 stars): {text}",
        "output": "",
        "metadata": {"rating": int(rating)},
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

    spark = create_spark_session()

    print(f"Reading reviews from {args.input}")
    df = spark.read.parquet(args.input)

    # Normalize columns
    if "asin" in df.columns and "parent_asin" not in df.columns:
        df = df.withColumnRenamed("asin", "parent_asin")

    print("Cleaning and deduplicating...")
    df = clean_reviews(df)

    if args.max_examples:
        df = df.limit(args.max_examples)

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
    train_df.write.text(f"{args.output}/train", mode="overwrite")
    val_df.write.text(f"{args.output}/val", mode="overwrite")
    test_df.write.text(f"{args.output}/test", mode="overwrite")

    print(f"  Train: {train_df.count():,} examples")
    print(f"  Val:   {val_df.count():,} examples")
    print(f"  Test:  {test_df.count():,} examples")

    spark.stop()
    print("Text preprocessing complete.")


if __name__ == "__main__":
    main()
