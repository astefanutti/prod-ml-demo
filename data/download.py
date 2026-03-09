"""Download Amazon Reviews 2023 dataset from HuggingFace.

Supports downloading a small sample (~1M reviews) for local testing
or the full dataset (~233M reviews) for production runs.

Usage:
    python data/download.py --mode sample   # ~1M reviews, ~500MB
    python data/download.py --mode full     # Full dataset, ~49GB
"""

import argparse
import os
from pathlib import Path

from datasets import load_dataset

# Categories to include (balances scale with diversity)
CATEGORIES = [
    "Electronics",
    "Books",
    "Home_and_Kitchen",
]

DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
RAW_DIR = Path("data/raw")
SAMPLE_DIR = Path("data/sample")


def download_category(category: str, mode: str, output_dir: Path) -> Path:
    """Download a single category of reviews."""
    print(f"Downloading {category} reviews ({mode} mode)...")

    subset_name = f"raw_review_{category}"
    if mode == "sample":
        ds = load_dataset(
            DATASET_NAME,
            subset_name,
            split="full",
            streaming=True,
            trust_remote_code=True,
        )
        # Take first 333K per category (~1M total across 3 categories)
        rows = []
        for i, row in enumerate(ds):
            if i >= 333_334:
                break
            rows.append(row)

        import pandas as pd

        df = pd.DataFrame(rows)
        out_path = output_dir / f"{category}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved {len(df)} reviews to {out_path}")
    else:
        ds = load_dataset(
            DATASET_NAME,
            subset_name,
            split="full",
            trust_remote_code=True,
        )
        out_path = output_dir / f"{category}.parquet"
        ds.to_parquet(out_path)
        print(f"  Saved {len(ds)} reviews to {out_path}")

    return out_path


def download_metadata(category: str, mode: str, output_dir: Path) -> Path:
    """Download product metadata for a category."""
    print(f"Downloading {category} metadata ({mode} mode)...")

    subset_name = f"raw_meta_{category}"
    if mode == "sample":
        ds = load_dataset(
            DATASET_NAME,
            subset_name,
            split="full",
            streaming=True,
            trust_remote_code=True,
        )
        rows = []
        for i, row in enumerate(ds):
            if i >= 50_000:
                break
            rows.append(row)

        import pandas as pd

        df = pd.DataFrame(rows)
        out_path = output_dir / f"{category}_meta.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved {len(df)} items to {out_path}")
    else:
        ds = load_dataset(
            DATASET_NAME,
            subset_name,
            split="full",
            trust_remote_code=True,
        )
        out_path = output_dir / f"{category}_meta.parquet"
        ds.to_parquet(out_path)
        print(f"  Saved {len(ds)} items to {out_path}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download Amazon Reviews 2023 dataset")
    parser.add_argument(
        "--mode",
        choices=["sample", "full"],
        default="sample",
        help="Download mode: 'sample' (~1M reviews) or 'full' (~233M reviews)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=CATEGORIES,
        help="Categories to download",
    )
    args = parser.parse_args()

    output_dir = SAMPLE_DIR if args.mode == "sample" else RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for category in args.categories:
        download_category(category, args.mode, output_dir)
        download_metadata(category, args.mode, output_dir)

    print(f"\nDone! Data saved to {output_dir}/")


if __name__ == "__main__":
    main()
