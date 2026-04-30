"""Spark Job C: Embedding Generation for Feast Vector Store.

Generates sentence embeddings for reviews using a sentence-transformer model
via Spark UDFs. Outputs embeddings as Parquet files for Feast vector store.

Usage:
    spark-submit spark/embedding_generation.py \
        --input s3a://smartshop-raw/ \
        --output s3a://smartshop-embeddings/review_embeddings/
"""

import argparse
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

try:
    from spark.utils.mlflow_metrics import SparkRunLogger
except ImportError:
    SparkRunLogger = None


EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _metric(key, value, unit=""):
    tag = f"[METRIC] {key}={value}"
    if unit:
        tag += f" {unit}"
    print(tag)


def create_spark_session(app_name: str = "SmartShop-Embeddings") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )


def make_embedding_udf():
    """Create a Spark UDF that generates embeddings using sentence-transformers.

    The model is loaded once per executor via a broadcast-like pattern.
    """

    @F.pandas_udf(T.ArrayType(T.FloatType()))
    def embed_texts(texts):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        # Convert to list, handle nulls
        text_list = texts.fillna("").tolist()
        embeddings = model.encode(text_list, batch_size=64, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    return embed_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input path for raw reviews")
    parser.add_argument("--output", required=True, help="Output path for embeddings")
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=None,
        help="Max reviews to embed (for sample mode)",
    )
    args = parser.parse_args()

    job_start = time.time()
    spark = create_spark_session()

    mlflow_logger = SparkRunLogger(spark, experiment="smartshop-embeddings") if SparkRunLogger else None
    _run_ctx = mlflow_logger.start_run(run_name="embedding-generation") if mlflow_logger else None
    if _run_ctx:
        _run_ctx.__enter__()

    print(f"Reading reviews from {args.input}")
    t = time.time()
    # Support comma-separated S3 paths (Electronics, Books, Home_and_Kitchen shards)
    input_paths = [p.strip() for p in args.input.split(",") if p.strip()]
    df = spark.read.parquet(*input_paths)

    # Normalize columns
    if "asin" in df.columns and "parent_asin" not in df.columns:
        df = df.withColumnRenamed("asin", "parent_asin")

    # Prepare text for embedding (combine title + review text)
    df = df.withColumn(
        "embed_text",
        F.concat_ws(
            " ",
            F.coalesce(F.col("title"), F.lit("")),
            F.coalesce(F.col("text"), F.lit("")),
        ),
    ).filter(F.length(F.col("embed_text")) >= 20)

    if args.max_reviews:
        df = df.limit(args.max_reviews)

    # Select relevant columns
    df = df.select(
        F.col("parent_asin").alias("item_id"),
        F.col("user_id"),
        F.col("rating"),
        F.col("embed_text"),
        F.col("title").alias("review_title"),
        F.monotonically_increasing_id().alias("review_id"),
    )

    # Generate embeddings
    print(f"Generating embeddings with {EMBEDDING_MODEL}...")
    embed_udf = make_embedding_udf()
    df = df.withColumn("embedding", embed_udf(F.col("embed_text")))

    # Add event timestamp for Feast
    df = df.withColumn("event_timestamp", F.current_timestamp())

    # Select final columns
    result = df.select(
        "review_id",
        "item_id",
        "user_id",
        "rating",
        "review_title",
        "embed_text",
        "embedding",
        "event_timestamp",
    )

    _metric("model", EMBEDDING_MODEL)
    _metric("embedding_dim", EMBEDDING_DIM)

    print("Writing embeddings...")
    t_write = time.time()
    result.write.mode("overwrite").parquet(args.output)

    total_written = result.count()
    job_elapsed = time.time() - job_start
    write_elapsed = time.time() - t_write

    _metric("total_embeddings", total_written)
    _metric("write_elapsed_s", round(write_elapsed, 2), "s")
    _metric("total_elapsed_s", round(job_elapsed, 2), "s")
    _metric("throughput_embeddings_per_s", int(total_written / job_elapsed) if job_elapsed > 0 else 0, "embeddings/s")

    if mlflow_logger:
        mlflow_logger.log_metric("total_embeddings", total_written)
        mlflow_logger.log_metric("total_elapsed_s", round(job_elapsed, 2))
        mlflow_logger.finalize(output_path="/tmp/embedding_metrics.json")
    if _run_ctx:
        _run_ctx.__exit__(None, None, None)

    print(
        f"\n{'='*60}\n"
        f"Embedding generation complete.\n"
        f"  Total embeddings : {total_written:,}\n"
        f"  Elapsed          : {job_elapsed:.1f}s\n"
        f"  Output           : {args.output}\n"
        f"{'='*60}"
    )
    spark.stop()


if __name__ == "__main__":
    main()
