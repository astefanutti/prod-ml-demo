"""Spark Job C: Embedding Generation for Feast Vector Store.

Generates sentence embeddings for reviews using a sentence-transformer model
via Spark UDFs. Outputs embeddings as Parquet files for Feast vector store.

Usage:
    spark-submit spark/embedding_generation.py \
        --input s3a://smartshop/raw/ \
        --output s3a://smartshop/features/review_embeddings/
"""

import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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

    spark = create_spark_session()

    print(f"Reading reviews from {args.input}")
    df = spark.read.parquet(args.input)

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

    print("Writing embeddings...")
    result.write.parquet(args.output, mode="overwrite")

    print(f"Wrote {result.count():,} embeddings to {args.output}")
    spark.stop()
    print("Embedding generation complete.")


if __name__ == "__main__":
    main()
