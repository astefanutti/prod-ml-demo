"""Distributed training script for Two-Tower recommendation model.

Uses PyTorch DDP (DistributedDataParallel) for multi-GPU training.

Feature retrieval — two modes controlled by --use-feast flag:

  --use-feast (default when FEAST_REPO_PATH is set):
    feast.get_historical_features() with SparkOfflineStore (local[*]).
    Rank-0 runs point-in-time correct Spark join against SparkSource Parquet
    in MinIO, saves result to /tmp/training_data.parquet, all ranks load it.
    Requires: pyspark==4.0.0, feast[spark], FEAST_REPO_PATH env var,
              FEAST_REGISTRY_TYPE=remote, FEAST_REGISTRY_PATH=<svc>:6570

  --no-feast:
    Direct pd.read_parquet from s3://smartshop-features/ (legacy path).
    Faster startup, no Feast dependency, no point-in-time correctness.

Usage (single node):
    torchrun --nproc_per_node=4 training/recommendation/train.py --use-feast

Usage (multi-node via Kubeflow Trainer):
    Launched automatically by TrainJob with TrainingRuntime: pytorch-ddp
"""

import argparse
import io
import os
import time

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import mlflow
import mlflow.pytorch

try:
    import fsspec
    HAS_FSSPEC = True
except ImportError:
    HAS_FSSPEC = False

try:
    from feast import FeatureStore
    HAS_FEAST = True
except ImportError:
    HAS_FEAST = False

from model import TwoTowerModel

# Feature columns served by Feast feature views
_USER_FEAT_COLS = [
    "user_avg_rating",
    "user_review_count",
    "user_unique_items",
    "user_avg_review_length",
    "user_category_count",
    "user_tenure_days",
]
_ITEM_FEAT_COLS = [
    "item_avg_rating",
    "item_rating_stddev",
    "item_review_count",
    "item_total_helpful_votes",
    "item_avg_review_length",
    "item_price",
]
_FEAST_FEATURES = (
    [f"user_features:{c}" for c in _USER_FEAT_COLS]
    + [f"item_features:{c}" for c in _ITEM_FEAT_COLS]
)


def _is_s3(path: str) -> bool:
    return path.startswith("s3://") or path.startswith("s3a://")


def _save_checkpoint(state: dict, path: str) -> None:
    """Save a PyTorch checkpoint to local or S3 path."""
    if _is_s3(path) and HAS_FSSPEC:
        buf = io.BytesIO()
        torch.save(state, buf)
        buf.seek(0)
        endpoint = os.environ.get("AWS_ENDPOINT_URL_S3", os.environ.get("S3_ENDPOINT", ""))
        storage_opts = {}
        if endpoint:
            storage_opts = {
                "key": os.environ.get("AWS_ACCESS_KEY_ID", ""),
                "secret": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                "client_kwargs": {"endpoint_url": endpoint},
            }
        with fsspec.open(path, "wb", **storage_opts) as f:
            f.write(buf.read())
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)


def _load_features_via_feast(interactions_df: pd.DataFrame, feast_repo_path: str) -> pd.DataFrame:
    """Point-in-time correct feature retrieval via Feast SparkOfflineStore.

    Rank-0 only. Calls store.get_historical_features() which triggers a
    SparkSession (local[*]) to join interactions against SparkSource Parquet
    in MinIO. Returns a merged DataFrame with all feature columns.

    Registry is read from the feast server's gRPC endpoint via remote registry
    (FEAST_REGISTRY_TYPE=remote, FEAST_REGISTRY_PATH=<svc>:6570).
    """
    if not HAS_FEAST:
        raise RuntimeError("feast not installed — run: pip install feast[spark]==0.62.0")

    store = FeatureStore(repo_path=feast_repo_path)

    # entity_df: user_id + item_id + UTC-aware event_timestamp for point-in-time join
    entity_df = interactions_df[["user_id", "item_id", "event_timestamp", "label"]].copy()
    entity_df["event_timestamp"] = pd.to_datetime(
        entity_df["event_timestamp"], unit="ms", utc=True
    )

    print(f"[Feast] get_historical_features for {len(entity_df):,} interactions...")
    t0 = time.time()
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=_FEAST_FEATURES,
    ).to_df()
    print(f"[Feast] retrieved {len(training_df):,} rows in {time.time()-t0:.1f}s")

    # Fill missing features (entities with no feature row in offline store)
    for col in _USER_FEAT_COLS + _ITEM_FEAT_COLS:
        if col not in training_df.columns:
            training_df[col] = 0.0
        else:
            training_df[col] = training_df[col].fillna(0.0)

    return training_df


def _load_features_direct(
    interactions_path: str,
    user_features_path: str,
    item_features_path: str,
    storage_options: dict = None,
) -> pd.DataFrame:
    """Direct pd.read_parquet path — no Feast, no point-in-time correctness.

    Legacy fallback for quick iteration or environments without pyspark.
    Feature join is done in pandas: interactions LEFT JOIN user_features
    LEFT JOIN item_features on user_id / item_id.
    """
    kw = {"storage_options": storage_options} if storage_options else {}
    interactions = pd.read_parquet(interactions_path, **kw)
    user_feats = pd.read_parquet(user_features_path, **kw)
    item_feats = pd.read_parquet(item_features_path, **kw)

    user_feats = user_feats[["user_id"] + _USER_FEAT_COLS].drop_duplicates("user_id")
    item_feats = item_feats.rename(columns={"parent_asin": "item_id"})
    item_cols_avail = [c for c in _ITEM_FEAT_COLS if c in item_feats.columns]
    item_feats = item_feats[["item_id"] + item_cols_avail].drop_duplicates("item_id")

    df = interactions.merge(user_feats, on="user_id", how="left")
    df = df.merge(item_feats, on="item_id", how="left")
    for col in _USER_FEAT_COLS + _ITEM_FEAT_COLS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)
    return df


class RecommendationDataset(Dataset):
    """Dataset wrapping a pre-loaded DataFrame with user-item features."""

    def __init__(self, data: pd.DataFrame):
        self.data = data

        # Build ID mappings
        all_users = self.data["user_id"].unique()
        all_items = self.data["item_id"].unique()
        self.user_to_idx = {uid: i for i, uid in enumerate(all_users)}
        self.item_to_idx = {iid: i for i, iid in enumerate(all_items)}
        self.num_users = len(all_users)
        self.num_items = len(all_items)
        self.user_feat_cols = _USER_FEAT_COLS
        self.item_feat_cols = _ITEM_FEAT_COLS

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user_idx = self.user_to_idx.get(row["user_id"], 0)
        item_idx = self.item_to_idx.get(row["item_id"], 0)
        user_feats = torch.tensor(
            [row[c] for c in self.user_feat_cols], dtype=torch.float32
        )
        item_feats = torch.tensor(
            [row[c] for c in self.item_feat_cols], dtype=torch.float32
        )
        label = torch.tensor(row["label"], dtype=torch.float32)
        return (
            torch.tensor(user_idx, dtype=torch.long),
            user_feats,
            torch.tensor(item_idx, dtype=torch.long),
            item_feats,
            label,
        )


def setup_distributed():
    """Initialize DDP process group."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, local_rank, dist.get_world_size()
    return 0, 0, 1


def train_epoch(model, dataloader, optimizer, criterion, device, rank):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for user_ids, user_feats, item_ids, item_feats, labels in dataloader:
        user_ids = user_ids.to(device)
        user_feats = user_feats.to(device)
        item_ids = item_ids.to(device)
        item_feats = item_feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(user_ids, user_feats, item_ids, item_feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for user_ids, user_feats, item_ids, item_feats, labels in dataloader:
        user_ids = user_ids.to(device)
        user_feats = user_feats.to(device)
        item_ids = item_ids.to(device)
        item_feats = item_feats.to(device)
        labels = labels.to(device)

        logits = model(user_ids, user_feats, item_ids, item_feats)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed", help="Feature data dir (s3:// or local). Used in --no-feast mode.")
    parser.add_argument("--output-dir", default="models/recommendation", help="Model output dir (s3:// or local)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Sample N interactions for fast iteration (default: use all)")
    feast_group = parser.add_mutually_exclusive_group()
    feast_group.add_argument("--use-feast", dest="use_feast", action="store_true",
                             default=bool(os.environ.get("FEAST_REPO_PATH")),
                             help="Use feast.get_historical_features() via SparkOfflineStore")
    feast_group.add_argument("--no-feast", dest="use_feast", action="store_false",
                             help="Read Parquet directly from --data-dir (legacy)")
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    job_start = time.time()

    feast_repo_path = os.environ.get("FEAST_REPO_PATH", "")
    use_feast = args.use_feast and bool(feast_repo_path)

    if rank == 0:
        print(f"Training with {world_size} processes on {device}")
        print(f"Feature source: {'Feast SparkOfflineStore' if use_feast else 'direct Parquet'}")
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        if os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "").lower() in ("true", "1"):
            os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        mlflow.set_experiment("smartshop-rec-training")
        run = mlflow.start_run(
            run_name=f"two-tower-ddp-{world_size}gpu-{args.epochs}ep"
        )
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "world_size": world_size,
            "device": str(device),
            "data_dir": args.data_dir,
            "feature_source": "feast_spark" if use_feast else "direct_parquet",
        })

    # ---- Feature loading -------------------------------------------------------
    # Rank-0 loads training data (Feast or direct), saves to /tmp for other ranks.
    # dist.barrier() ensures rank-0 finishes before others read /tmp.
    _STAGING = "/tmp/smartshop_training_data.parquet"

    storage_options = None
    if _is_s3(args.data_dir):
        endpoint = os.environ.get("AWS_ENDPOINT_URL_S3", os.environ.get("S3_ENDPOINT", ""))
        if endpoint:
            storage_options = {
                "key": os.environ.get("AWS_ACCESS_KEY_ID", ""),
                "secret": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                "client_kwargs": {"endpoint_url": endpoint},
            }

    if rank == 0:
        if use_feast:
            # Load interactions only (needed as entity_df for the Feast join)
            kw = {"storage_options": storage_options} if storage_options else {}
            interactions_df = pd.read_parquet(f"{args.data_dir}/interactions", **kw)
            if args.max_rows and len(interactions_df) > args.max_rows:
                interactions_df = interactions_df.sample(n=args.max_rows, random_state=42)
                print(f"Sampled {args.max_rows:,} / {len(interactions_df):,} interactions")
            df = _load_features_via_feast(interactions_df, feast_repo_path)
        else:
            df = _load_features_direct(
                interactions_path=f"{args.data_dir}/interactions",
                user_features_path=f"{args.data_dir}/user_features",
                item_features_path=f"{args.data_dir}/item_features",
                storage_options=storage_options,
            )
        df.to_parquet(_STAGING, index=False)
        print(f"[rank-0] saved {len(df):,} rows to {_STAGING}")

    if world_size > 1:
        dist.barrier()  # wait for rank-0 to finish writing

    df = pd.read_parquet(_STAGING)
    dataset = RecommendationDataset(df)

    # Train/val split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    if rank == 0:
        print(f"Users: {dataset.num_users:,}, Items: {dataset.num_items:,}")
        print(f"Train: {train_size:,}, Val: {val_size:,}")
        mlflow.log_params({
            "num_users": dataset.num_users,
            "num_items": dataset.num_items,
            "train_size": train_size,
            "val_size": val_size,
        })

    # Distributed sampler
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = TwoTowerModel(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        user_feat_dim=len(dataset.user_feat_cols),
        item_feat_dim=len(dataset.item_feat_cols),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, rank)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if rank == 0:
            elapsed = time.time() - job_start
            throughput = int(len(train_dataset) / (elapsed / (epoch + 1)))
            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Throughput: {throughput:,} samples/s"
            )
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "throughput_samples_per_s": throughput,
                    "elapsed_s": round(elapsed, 1),
                },
                step=epoch + 1,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = f"{args.output_dir}/best_model.pt"
                raw_model = model.module if hasattr(model, "module") else model
                state = {
                    "model_state_dict": raw_model.state_dict(),
                    "num_users": dataset.num_users,
                    "num_items": dataset.num_items,
                    "user_to_idx": dataset.user_to_idx,
                    "item_to_idx": dataset.item_to_idx,
                    "user_feat_dim": len(dataset.user_feat_cols),
                    "item_feat_dim": len(dataset.item_feat_cols),
                    "embed_dim": args.embed_dim,
                    "hidden_dim": args.hidden_dim,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "epoch": epoch + 1,
                }
                _save_checkpoint(state, ckpt_path)
                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch + 1)
                print(f"  Saved best model → {ckpt_path} (val_loss={val_loss:.4f})")

    if world_size > 1:
        dist.destroy_process_group()

    if rank == 0:
        total_time = time.time() - job_start
        mlflow.log_metrics({
            "best_val_loss": best_val_loss,
            "total_training_time_s": round(total_time, 1),
        })
        mlflow.end_run()
        print(
            f"\n{'='*60}\n"
            f"Training complete.\n"
            f"  Best val loss : {best_val_loss:.4f}\n"
            f"  Total time    : {total_time:.1f}s\n"
            f"  Model saved   : {args.output_dir}/best_model.pt\n"
            f"{'='*60}"
        )


if __name__ == "__main__":
    main()
