"""Distributed training script for Two-Tower recommendation model.

Uses PyTorch DDP (DistributedDataParallel) for multi-GPU training.
Reads training data from Feast offline store.

Usage (single node):
    torchrun --nproc_per_node=4 training/recommendation/train.py

Usage (multi-node via Kubeflow Trainer):
    Launched automatically by TrainJob with TrainingRuntime: pytorch-ddp
"""

import argparse
import os

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from model import TwoTowerModel


class RecommendationDataset(Dataset):
    """Dataset that loads user-item interactions with Feast features."""

    def __init__(self, interactions_path: str, user_features_path: str, item_features_path: str):
        self.interactions = pd.read_parquet(interactions_path)
        user_feats = pd.read_parquet(user_features_path)
        item_feats = pd.read_parquet(item_features_path)

        # Build ID mappings
        all_users = self.interactions["user_id"].unique()
        all_items = self.interactions["item_id"].unique()
        self.user_to_idx = {uid: i for i, uid in enumerate(all_users)}
        self.item_to_idx = {iid: i for i, iid in enumerate(all_items)}
        self.num_users = len(all_users)
        self.num_items = len(all_items)

        # Numeric feature columns
        self.user_feat_cols = [
            "user_avg_rating",
            "user_review_count",
            "user_unique_items",
            "user_avg_review_length",
            "user_category_count",
            "user_tenure_days",
        ]
        self.item_feat_cols = [
            "item_avg_rating",
            "item_rating_stddev",
            "item_review_count",
            "item_total_helpful_votes",
            "item_avg_review_length",
            "item_price",
        ]

        # Merge features into interactions
        user_feats = user_feats[["user_id"] + self.user_feat_cols].drop_duplicates("user_id")
        item_feats_renamed = item_feats.rename(columns={"parent_asin": "item_id"})
        item_feat_cols_available = [c for c in self.item_feat_cols if c in item_feats_renamed.columns]
        item_feats_renamed = item_feats_renamed[["item_id"] + item_feat_cols_available].drop_duplicates("item_id")

        self.data = self.interactions.merge(user_feats, on="user_id", how="left")
        self.data = self.data.merge(item_feats_renamed, on="item_id", how="left")

        # Fill NaN features with 0
        for col in self.user_feat_cols + item_feat_cols_available:
            self.data[col] = self.data[col].fillna(0.0)

        # Pad missing item feature columns
        for col in self.item_feat_cols:
            if col not in self.data.columns:
                self.data[col] = 0.0

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
    parser.add_argument("--data-dir", default="data/processed", help="Feature data directory")
    parser.add_argument("--output-dir", default="models/recommendation", help="Model output dir")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"Training with {world_size} processes on {device}")

    # Load dataset
    dataset = RecommendationDataset(
        interactions_path=f"{args.data_dir}/interactions",
        user_features_path=f"{args.data_dir}/user_features",
        item_features_path=f"{args.data_dir}/item_features",
    )

    # Train/val split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    if rank == 0:
        print(f"Users: {dataset.num_users:,}, Items: {dataset.num_items:,}")
        print(f"Train: {train_size:,}, Val: {val_size:,}")

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
            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(args.output_dir, exist_ok=True)
                raw_model = model.module if hasattr(model, "module") else model
                torch.save(
                    {
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
                    },
                    f"{args.output_dir}/best_model.pt",
                )
                print(f"  Saved best model (val_loss={val_loss:.4f})")

    if world_size > 1:
        dist.destroy_process_group()

    if rank == 0:
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
