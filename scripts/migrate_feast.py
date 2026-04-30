"""Migrate review embeddings from old Milvus collection to Feast-managed collection.

Reads batches from smartshop_review_embeddings_old (SparkApp-written, wrong key format),
writes through Feast's write_to_online_store() so entity keys are properly serialized.
Uses cursor-based pagination (PK ordering) to avoid Milvus offset limits.
"""

import os
import time

import pandas as pd
from pymilvus import MilvusClient
from feast import FeatureStore
from feast.repo_config import load_repo_config

BATCH_SIZE = 2000
TARGET_ROWS = int(os.environ.get("MIGRATE_ROWS", "1000000"))
OLD_COLLECTION = "smartshop_review_embeddings_old"
FIELDS = ["review_id_pk", "review_id", "item_id", "rating", "embed_text",
          "review_title", "user_id", "embedding"]

client = MilvusClient(uri="http://milvus.smartshop.svc.cluster.local:19530")

feast_repo = os.environ.get("FEAST_REPO_PATH", "feast/feature_repo")
feast_config = os.environ.get("FEAST_CONFIG", "feature_store_serving.yaml")
config_path = os.path.join(feast_repo, feast_config)
repo_config = load_repo_config(repo_path=feast_repo, fs_yaml_file=config_path)
store = FeatureStore(config=repo_config)

print(f"Migrating up to {TARGET_ROWS:,} rows from {OLD_COLLECTION} through Feast SDK")

total = 0
cursor = ""
t0 = time.time()

while total < TARGET_ROWS:
    filt = f'review_id_pk > "{cursor}"' if cursor else 'review_id_pk != ""'
    rows = client.query(
        OLD_COLLECTION,
        filter=filt,
        limit=BATCH_SIZE,
        output_fields=FIELDS,
    )
    if not rows:
        print("No more rows")
        break

    cursor = max(r["review_id_pk"] for r in rows)

    records = []
    for r in rows:
        records.append({
            "review_id": r["review_id"],
            "item_id": r["item_id"],
            "user_id": r["user_id"],
            "rating": float(r["rating"]),
            "review_title": r["review_title"],
            "embed_text": r["embed_text"],
            "embedding": r["embedding"],
            "event_timestamp": pd.Timestamp.now(),
        })
    df = pd.DataFrame(records)
    store.write_to_online_store("review_embeddings", df)

    total += len(rows)
    elapsed = time.time() - t0
    rate = total / elapsed if elapsed > 0 else 0
    print(f"  Migrated {total:,} rows ({rate:.0f} rows/s)")

elapsed = time.time() - t0
print(f"\nDone: {total:,} rows migrated in {elapsed:.0f}s")

stats = client.get_collection_stats("smartshop_review_embeddings")
print(f"New collection stats: {stats}")
