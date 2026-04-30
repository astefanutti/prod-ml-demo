"""Two-Tower Neural Collaborative Filtering model for product recommendations.

Architecture:
  - User tower: Encodes user features into a dense embedding
  - Item tower: Encodes item features into a dense embedding
  - Similarity: Dot product between user and item embeddings
  - Loss: Binary cross-entropy (positive = rating >= 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    """Encodes user features into a dense embedding."""

    def __init__(
        self,
        num_users: int,
        user_feat_dim: int = 6,
        embed_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + user_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, user_ids: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        x = torch.cat([user_emb, user_features], dim=-1)
        return F.normalize(self.fc(x), p=2, dim=-1)


class ItemTower(nn.Module):
    """Encodes item features into a dense embedding."""

    def __init__(
        self,
        num_items: int,
        item_feat_dim: int = 6,
        embed_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + item_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, item_ids: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        item_emb = self.item_embedding(item_ids)
        x = torch.cat([item_emb, item_features], dim=-1)
        return F.normalize(self.fc(x), p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """Two-Tower recommendation model.

    Computes similarity between user and item embeddings via dot product.
    Trained with binary cross-entropy loss where positive = rating >= 4.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_feat_dim: int = 6,
        item_feat_dim: int = 6,
        embed_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.user_tower = UserTower(num_users, user_feat_dim, embed_dim, hidden_dim)
        self.item_tower = ItemTower(num_items, item_feat_dim, embed_dim, hidden_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        user_ids: torch.Tensor,
        user_features: torch.Tensor,
        item_ids: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        user_emb = self.user_tower(user_ids, user_features)
        item_emb = self.item_tower(item_ids, item_features)
        logits = (user_emb * item_emb).sum(dim=-1) / self.temperature
        return logits

    def get_user_embedding(
        self, user_ids: torch.Tensor, user_features: torch.Tensor
    ) -> torch.Tensor:
        return self.user_tower(user_ids, user_features)

    def get_item_embedding(
        self, item_ids: torch.Tensor, item_features: torch.Tensor
    ) -> torch.Tensor:
        return self.item_tower(item_ids, item_features)
