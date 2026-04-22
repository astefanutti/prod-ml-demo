"""Download Amazon Reviews 2023 dataset from HuggingFace.

Streams JSONL directly from the Hub using the `json` dataset type — no local cache
of the full category file (Electronics alone is ~22GB). Only the rows we need land
on disk as Parquet.

File layout in McAuley-Lab/Amazon-Reviews-2023:
  raw/review_categories/{Category}.jsonl       ← reviews (streamed)
  raw_meta_{Category}/full-*.parquet           ← metadata (Parquet shards, much smaller)

Usage:
    python data/download.py --mode sample   # ~333K reviews per category, ~500MB total
    python data/download.py --mode full     # Full category files (49GB+, needs cluster)
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
HF_TOKEN = os.environ.get("HF_TOKEN")

CATEGORIES = [
    "Electronics",
    "Books",
    "Home_and_Kitchen",
]

RAW_DIR = Path("data/raw")
SAMPLE_DIR = Path("data/sample")

SAMPLE_PER_CATEGORY = 333_334   # ~1M reviews total across 3 categories
META_SAMPLE = 50_000


def download_category(category: str, mode: str, output_dir: Path) -> None:
    print(f"Downloading {category} reviews ({mode} mode)...")

    hf_path = f"hf://datasets/{DATASET_NAME}/raw/review_categories/{category}.jsonl"

    if mode == "sample":
        # Stream line-by-line — no full file download, stays memory-efficient
        ds = load_dataset(
            "json",
            data_files={"full": hf_path},
            split="full",
            streaming=True,
            token=HF_TOKEN,
        )
        rows = []
        for i, row in enumerate(ds):
            if i >= SAMPLE_PER_CATEGORY:
                break
            rows.append(row)
        df = pd.DataFrame(rows)
    else:
        # Full download — requires ~22GB free per category; use on cluster only
        ds = load_dataset(
            "json",
            data_files={"full": hf_path},
            split="full",
            token=HF_TOKEN,
        )
        df = ds.to_pandas()

    out = output_dir / f"{category}.parquet"
    df.to_parquet(out, index=False)
    print(f"  Saved {len(df):,} reviews → {out}")


def download_metadata(category: str, mode: str, output_dir: Path) -> None:
    print(f"Downloading {category} metadata ({mode} mode)...")

    from huggingface_hub import hf_hub_download, list_repo_files

    # Metadata already available as Parquet shards (much smaller than review JSONL)
    all_files = list(list_repo_files(DATASET_NAME, repo_type="dataset", token=HF_TOKEN))
    shards = sorted([
        f for f in all_files
        if f.startswith(f"raw_meta_{category}/") and f.endswith(".parquet")
    ])

    frames = []
    collected = 0
    for shard in shards:
        local = hf_hub_download(
            repo_id=DATASET_NAME, filename=shard,
            repo_type="dataset", token=HF_TOKEN,
        )
        df = pd.read_parquet(local)
        if mode == "sample":
            remaining = META_SAMPLE - collected
            df = df.head(remaining)
        frames.append(df)
        collected += len(df)
        if mode == "sample" and collected >= META_SAMPLE:
            break

    result = pd.concat(frames, ignore_index=True)
    out = output_dir / f"{category}_meta.parquet"
    result.to_parquet(out, index=False)
    print(f"  Saved {len(result):,} items → {out}")


def main():
    parser = argparse.ArgumentParser(description="Download Amazon Reviews 2023 dataset")
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    args = parser.parse_args()

    output_dir = SAMPLE_DIR if args.mode == "sample" else RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for category in args.categories:
        download_category(category, args.mode, output_dir)
        download_metadata(category, args.mode, output_dir)

    print(f"\nDone! Data saved to {output_dir}/")


if __name__ == "__main__":
    main()
