"""KServe-compatible recommendation server.

Loads the Two-Tower model and Feast online features to serve
real-time product recommendations.

Endpoints:
    POST /v1/models/smartshop-rec:predict
    Body: {"user_id": "...", "candidate_items": ["ASIN1", "ASIN2", ...], "top_k": 10}
    Response: {"recommendations": [{"item_id": "...", "score": 0.95}, ...]}
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI
from feast import FeatureStore
from pydantic import BaseModel

from training.recommendation.model import TwoTowerModel

_model: Optional[TwoTowerModel] = None
_feast_store: Optional[FeatureStore] = None
_checkpoint: Optional[dict] = None

USER_FEAT_COLS = [
    "user_features:user_avg_rating",
    "user_features:user_review_count",
    "user_features:user_unique_items",
    "user_features:user_avg_review_length",
    "user_features:user_category_count",
    "user_features:user_tenure_days",
]
ITEM_FEAT_COLS = [
    "item_features:item_avg_rating",
    "item_features:item_rating_stddev",
    "item_features:item_review_count",
    "item_features:item_total_helpful_votes",
    "item_features:item_avg_review_length",
    "item_features:item_price",
]
USER_FEAT_KEYS = [f.split(":")[1] for f in USER_FEAT_COLS]
ITEM_FEAT_KEYS = [f.split(":")[1] for f in ITEM_FEAT_COLS]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _feast_store, _checkpoint

    model_path = os.environ.get("MODEL_PATH", "models/recommendation/best_model.pt")
    feast_repo = os.environ.get("FEAST_REPO_PATH", "feast/feature_repo")

    _checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    _model = TwoTowerModel(
        num_users=_checkpoint["num_users"],
        num_items=_checkpoint["num_items"],
        user_feat_dim=_checkpoint["user_feat_dim"],
        item_feat_dim=_checkpoint["item_feat_dim"],
        embed_dim=_checkpoint["embed_dim"],
        hidden_dim=_checkpoint["hidden_dim"],
    )
    _model.load_state_dict(_checkpoint["model_state_dict"])
    _model.eval()

    _feast_store = FeatureStore(repo_path=feast_repo)
    print(f"Model loaded from {model_path} "
          f"(user_feat_dim={_checkpoint['user_feat_dim']}, "
          f"item_feat_dim={_checkpoint['item_feat_dim']})")
    yield


app = FastAPI(title="SmartShop Recommendation Service", lifespan=lifespan)


class RecommendRequest(BaseModel):
    user_id: str
    candidate_items: list[str] = []
    top_k: int = 10


class RecommendResponse(BaseModel):
    recommendations: list[dict]


def _feat_row(d: dict, keys: list[str], n: int) -> list[float]:
    """Extract feature values from a Feast result dict, defaulting to 0."""
    return [float(d.get(k, [0.0])[0] or 0.0) for k in keys[:n]]


@app.post("/v1/models/smartshop-rec:predict", response_model=RecommendResponse)
def predict(request: RecommendRequest):
    user_to_idx = _checkpoint["user_to_idx"]
    item_to_idx = _checkpoint["item_to_idx"]
    n_user = _checkpoint["user_feat_dim"]
    n_item = _checkpoint["item_feat_dim"]

    user_idx = user_to_idx.get(request.user_id, 0)

    # User features — single Feast call
    try:
        user_result = _feast_store.get_online_features(
            features=USER_FEAT_COLS[:n_user],
            entity_rows=[{"user_id": request.user_id}],
        ).to_dict()
        user_feat_values = _feat_row(user_result, USER_FEAT_KEYS, n_user)
    except Exception:
        user_feat_values = [0.0] * n_user

    user_ids_t = torch.tensor([user_idx], dtype=torch.long)
    user_feats_t = torch.tensor([user_feat_values], dtype=torch.float32)

    candidates = request.candidate_items or list(item_to_idx.keys())[:100]

    # Batch all candidate item lookups in a single Feast call
    entity_rows = [{"item_id": iid} for iid in candidates]
    try:
        item_result = _feast_store.get_online_features(
            features=ITEM_FEAT_COLS[:n_item],
            entity_rows=entity_rows,
        ).to_dict()
        # item_result values are lists aligned to entity_rows order
        item_feats_batch = [
            [float(item_result.get(k, [0.0] * len(candidates))[i] or 0.0)
             for k in ITEM_FEAT_KEYS[:n_item]]
            for i in range(len(candidates))
        ]
    except Exception:
        item_feats_batch = [[0.0] * n_item] * len(candidates)

    # Score all candidates in one forward pass
    item_indices = [item_to_idx.get(iid, 0) for iid in candidates]
    item_ids_t = torch.tensor(item_indices, dtype=torch.long)
    item_feats_t = torch.tensor(item_feats_batch, dtype=torch.float32)
    user_ids_exp = user_ids_t.expand(len(candidates))
    user_feats_exp = user_feats_t.expand(len(candidates), -1)

    with torch.no_grad():
        logits = _model(user_ids_exp, user_feats_exp, item_ids_t, item_feats_t)
        scores_t = torch.sigmoid(logits).tolist()

    scores = [
        {"item_id": iid, "score": round(s, 4)}
        for iid, s in zip(candidates, scores_t)
    ]
    scores.sort(key=lambda x: x["score"], reverse=True)
    return RecommendResponse(recommendations=scores[: request.top_k])


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": _model is not None}
