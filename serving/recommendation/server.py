"""KServe-compatible recommendation server.

Loads the Two-Tower model and Feast online features to serve
real-time product recommendations.

Endpoints:
    POST /v1/models/smartshop-rec:predict
    Body: {"user_id": "...", "candidate_items": ["ASIN1", "ASIN2", ...], "top_k": 10}
    Response: {"recommendations": [{"item_id": "...", "score": 0.95}, ...]}
"""

import os
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI
from feast import FeatureStore
from pydantic import BaseModel

# Add parent to path for model import
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../training/recommendation"))
from model import TwoTowerModel

app = FastAPI(title="SmartShop Recommendation Service")

# Globals (loaded on startup)
model: Optional[TwoTowerModel] = None
feast_store: Optional[FeatureStore] = None
checkpoint: Optional[dict] = None


class RecommendRequest(BaseModel):
    user_id: str
    candidate_items: list[str] = []
    top_k: int = 10


class RecommendResponse(BaseModel):
    recommendations: list[dict]


@app.on_event("startup")
def load_model():
    global model, feast_store, checkpoint

    model_path = os.environ.get("MODEL_PATH", "models/recommendation/best_model.pt")
    feast_repo = os.environ.get("FEAST_REPO_PATH", "feast/feature_repo")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = TwoTowerModel(
        num_users=checkpoint["num_users"],
        num_items=checkpoint["num_items"],
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    feast_store = FeatureStore(repo_path=feast_repo)
    print(f"Model loaded from {model_path}")


@app.post("/v1/models/smartshop-rec:predict", response_model=RecommendResponse)
def predict(request: RecommendRequest):
    user_to_idx = checkpoint["user_to_idx"]
    item_to_idx = checkpoint["item_to_idx"]

    # Resolve user
    user_idx = user_to_idx.get(request.user_id, 0)

    # Get user features from Feast online store
    try:
        user_features = feast_store.get_online_features(
            features=[
                "user_features:user_avg_rating",
                "user_features:user_review_count",
                "user_features:user_unique_items",
                "user_features:user_avg_review_length",
                "user_features:user_category_count",
                "user_features:user_tenure_days",
            ],
            entity_rows=[{"user_id": request.user_id}],
        ).to_dict()
        user_feat_values = [
            user_features.get("user_avg_rating", [0.0])[0] or 0.0,
            user_features.get("user_review_count", [0])[0] or 0,
            user_features.get("user_unique_items", [0])[0] or 0,
            user_features.get("user_avg_review_length", [0.0])[0] or 0.0,
            user_features.get("user_category_count", [0])[0] or 0,
            user_features.get("user_tenure_days", [0])[0] or 0,
        ]
    except Exception:
        user_feat_values = [0.0] * 6

    user_ids_t = torch.tensor([user_idx], dtype=torch.long)
    user_feats_t = torch.tensor([user_feat_values], dtype=torch.float32)

    # Score candidate items
    candidates = request.candidate_items or list(item_to_idx.keys())[:100]
    scores = []

    for item_id in candidates:
        item_idx = item_to_idx.get(item_id, 0)

        # Get item features from Feast
        try:
            item_features = feast_store.get_online_features(
                features=[
                    "item_features:item_avg_rating",
                    "item_features:item_rating_stddev",
                    "item_features:item_review_count",
                    "item_features:item_total_helpful_votes",
                    "item_features:item_avg_review_length",
                    "item_features:item_price",
                ],
                entity_rows=[{"item_id": item_id}],
            ).to_dict()
            item_feat_values = [
                item_features.get("item_avg_rating", [0.0])[0] or 0.0,
                item_features.get("item_rating_stddev", [0.0])[0] or 0.0,
                item_features.get("item_review_count", [0])[0] or 0,
                item_features.get("item_total_helpful_votes", [0])[0] or 0,
                item_features.get("item_avg_review_length", [0.0])[0] or 0.0,
                item_features.get("item_price", [0.0])[0] or 0.0,
            ]
        except Exception:
            item_feat_values = [0.0] * 6

        # Pad to 8 features (model expects item_feat_dim=8)
        item_feat_values.extend([0.0, 0.0])

        item_ids_t = torch.tensor([item_idx], dtype=torch.long)
        item_feats_t = torch.tensor([item_feat_values], dtype=torch.float32)

        with torch.no_grad():
            logit = model(user_ids_t, user_feats_t, item_ids_t, item_feats_t)
            score = torch.sigmoid(logit).item()

        scores.append({"item_id": item_id, "score": round(score, 4)})

    # Sort and return top-K
    scores.sort(key=lambda x: x["score"], reverse=True)
    return RecommendResponse(recommendations=scores[: request.top_k])


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}
