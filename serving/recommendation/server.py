"""KServe-compatible recommendation server.

Loads the Two-Tower model with ID mappings from the training checkpoint
and serves real-time product recommendations.

Endpoints:
    POST /v1/models/smartshop-rec:predict
    Body: {"user_id": "...", "candidate_items": ["ASIN1", "ASIN2", ...], "top_k": 10}
    Response: {"recommendations": [{"item_id": "...", "score": 0.95}, ...]}
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
import torch.nn as nn
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel

REQUEST_DURATION = Histogram(
    "smartshop_rec_request_duration_seconds", "Predict latency",
    buckets=[.005, .01, .025, .05, .1, .25, .5, 1, 2.5],
)
REQUESTS_TOTAL = Counter(
    "smartshop_rec_requests_total", "Total prediction requests", ["status"],
)
CANDIDATES_SCORED = Histogram(
    "smartshop_rec_candidates_scored", "Items scored per request",
    buckets=[10, 25, 50, 100, 200, 500],
)


class TwoTower(nn.Module):
    """Matches the architecture produced by 01_training_rec.ipynb."""

    def __init__(self, n_users: int, n_items: int, embed_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        u = self.user_mlp(self.user_embed(users))
        i = self.item_mlp(self.item_embed(items))
        return (u * i).sum(dim=1)


_model: Optional[TwoTower] = None
_user_to_idx: dict = {}
_item_to_idx: dict = {}
_idx_to_item: dict = {}
_all_item_ids: list = []
_all_item_embeddings: Optional[torch.Tensor] = None
_all_item_indices: Optional[torch.Tensor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _user_to_idx, _item_to_idx, _idx_to_item, _all_item_ids
    global _all_item_embeddings, _all_item_indices

    model_path = os.environ.get("MODEL_PATH", "models/recommendation/best_model.pt")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        n_users = checkpoint["n_users"]
        n_items = checkpoint["n_items"]
        embed_dim = checkpoint.get("embed_dim", 64)
        hidden_dim = checkpoint.get("hidden_dim", 256)
        _user_to_idx = checkpoint.get("user_to_idx", {})
        _item_to_idx = checkpoint.get("item_to_idx", {})
    else:
        state_dict = checkpoint
        n_users = state_dict["user_embed.weight"].shape[0]
        n_items = state_dict["item_embed.weight"].shape[0]
        embed_dim = state_dict["user_embed.weight"].shape[1]
        hidden_dim = state_dict["user_mlp.0.weight"].shape[0]

    _idx_to_item = {v: k for k, v in _item_to_idx.items()}
    _all_item_ids = list(_item_to_idx.keys())

    _model = TwoTower(n_users, n_items, embed_dim, hidden_dim)
    _model.load_state_dict(state_dict)
    _model.eval()

    with torch.no_grad():
        _all_item_indices = torch.arange(n_items, dtype=torch.long)
        _all_item_embeddings = _model.item_mlp(_model.item_embed(_all_item_indices))

    print(f"Model loaded: {n_users} users, {n_items} items, "
          f"embed_dim={embed_dim}, hidden_dim={hidden_dim}, "
          f"mappings={'yes' if _user_to_idx else 'no'}, "
          f"item_embeddings_precomputed={_all_item_embeddings.shape}")
    yield


app = FastAPI(title="SmartShop Recommendation Service", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


class RecommendRequest(BaseModel):
    user_id: str
    candidate_items: list[str] = []
    top_k: int = 10


class RecommendResponse(BaseModel):
    recommendations: list[dict]


@app.post("/v1/models/smartshop-rec:predict", response_model=RecommendResponse)
def predict(request: RecommendRequest):
    t0 = time.perf_counter()
    try:
        if _user_to_idx:
            user_idx = _user_to_idx.get(request.user_id, 0)
        else:
            user_idx = hash(request.user_id) % _model.user_embed.num_embeddings

        with torch.no_grad():
            user_t = torch.tensor([user_idx], dtype=torch.long)
            user_emb = _model.user_mlp(_model.user_embed(user_t))

            if request.candidate_items:
                idxs = [_item_to_idx.get(iid, 0) for iid in request.candidate_items]
                item_t = torch.tensor(idxs, dtype=torch.long)
                item_embs = _model.item_mlp(_model.item_embed(item_t))
                scores_t = torch.sigmoid((user_emb * item_embs).sum(dim=1))
                item_ids = request.candidate_items
            else:
                scores_t = torch.sigmoid((user_emb * _all_item_embeddings).sum(dim=1))
                item_ids = None

            top_k = min(request.top_k, scores_t.shape[0])
            top_scores, top_indices = torch.topk(scores_t, top_k)

        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            iid = item_ids[idx] if item_ids else _idx_to_item.get(idx, str(idx))
            results.append({"item_id": iid, "score": round(score, 4)})

        CANDIDATES_SCORED.observe(scores_t.shape[0])
        REQUESTS_TOTAL.labels(status="ok").inc()
        return RecommendResponse(recommendations=results)
    except Exception as e:
        REQUESTS_TOTAL.labels(status="error").inc()
        raise e
    finally:
        REQUEST_DURATION.observe(time.perf_counter() - t0)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "has_mappings": bool(_user_to_idx),
    }
