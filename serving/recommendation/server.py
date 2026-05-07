"""KServe-compatible recommendation server.

Loads the Two-Tower model with ID mappings from the training checkpoint
and serves real-time product recommendations. Enriches results with
product metadata (title, brand, category, rating, price) from Feast
and user profile features for persona context.

Endpoints:
    POST /v1/models/smartshop-rec:predict
    POST /v1/models/smartshop-rec:user-profile
    GET  /health
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as tnf
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
    """Two-Tower with optional category-aware embeddings.

    Matches the training architecture: user/item embeddings are concatenated
    with a shared category embedding before the MLP towers. Falls back to
    the legacy embed-only architecture when n_categories=0.
    """

    def __init__(self, n_users: int, n_items: int, embed_dim: int = 64,
                 hidden_dim: int = 256, n_categories: int = 0):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        self.has_cat = n_categories > 0
        if self.has_cat:
            self.cat_embed = nn.Embedding(n_categories, embed_dim // 4)
            mlp_in = embed_dim + embed_dim // 4
        else:
            mlp_in = embed_dim
        self.user_mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim),
        )

    def user_tower(self, users: torch.Tensor, cats: Optional[torch.Tensor] = None) -> torch.Tensor:
        e = self.user_embed(users)
        if self.has_cat and cats is not None:
            e = torch.cat([e, self.cat_embed(cats)], dim=-1)
        return self.user_mlp(e)

    def item_tower(self, items: torch.Tensor, cats: Optional[torch.Tensor] = None) -> torch.Tensor:
        e = self.item_embed(items)
        if self.has_cat and cats is not None:
            e = torch.cat([e, self.cat_embed(cats)], dim=-1)
        return self.item_mlp(e)

    def forward(self, users: torch.Tensor, items: torch.Tensor,
                user_cats=None, item_cats=None, labels=None) -> torch.Tensor:
        u = self.user_tower(users, user_cats)
        i = self.item_tower(items, item_cats)
        return (u * i).sum(dim=1)


_model: Optional[TwoTower] = None
_user_to_idx: dict = {}
_item_to_idx: dict = {}
_idx_to_item: dict = {}
_cat_to_idx: dict = {}
_item_id_to_cat: Optional[torch.Tensor] = None
_user_id_to_cat: Optional[torch.Tensor] = None
_all_item_ids: list = []
_all_item_embeddings: Optional[torch.Tensor] = None
_all_item_indices: Optional[torch.Tensor] = None
_feast_store = None

ITEM_DISPLAY_FEATURES = [
    "item_metadata:item_title",
    "item_metadata:item_brand",
    "item_metadata:item_category",
    "item_features:item_avg_rating",
    "item_features:item_review_count",
    "item_features:item_rating_stddev",
    "item_features:item_total_helpful_votes",
    "item_metadata:item_price",
]

USER_PROFILE_FEATURES = [
    "user_features:user_avg_rating",
    "user_features:user_review_count",
    "user_features:user_primary_category",
    "user_features:user_tenure_days",
]

_OVER_FETCH_FACTOR = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _user_to_idx, _item_to_idx, _idx_to_item, _all_item_ids
    global _all_item_embeddings, _all_item_indices, _feast_store
    global _cat_to_idx, _item_id_to_cat, _user_id_to_cat

    feast_config = os.environ.get("FEAST_CONFIG_PATH")
    if feast_config:
        try:
            from feast import FeatureStore
            from feast.repo_config import load_repo_config
            import pathlib
            import tempfile

            cfg_path = pathlib.Path(feast_config)
            raw_yaml = cfg_path.read_text()
            expanded = os.path.expandvars(raw_yaml)
            if expanded != raw_yaml:
                resolved_dir = pathlib.Path("/tmp/feast-config")
                resolved_dir.mkdir(parents=True, exist_ok=True)
                resolved_path = resolved_dir / cfg_path.name
                resolved_path.write_text(expanded)
                print(f"Feast config: expanded env vars → {resolved_path}")
            else:
                resolved_path = cfg_path

            repo_config = load_repo_config(
                repo_path=str(resolved_path.parent), fs_yaml_file=str(resolved_path),
            )
            _feast_store = FeatureStore(config=repo_config)
            print(f"Feast connected: {feast_config}")
        except Exception as e:
            print(f"Feast init failed ({e}), serving without enrichment")
            _feast_store = None
    else:
        print("No FEAST_CONFIG_PATH set, serving without product metadata enrichment")

    model_path = os.environ.get("MODEL_PATH", "models/recommendation/best_model.pt")

    if model_path.startswith("s3://"):
        import fsspec
        fs, _ = fsspec.core.url_to_fs(
            model_path,
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3"),
        )
        if fs.isdir(model_path):
            model_path = model_path.rstrip("/") + "/best_model.pt"
        local_path = "/tmp/best_model.pt"
        print(f"Downloading {model_path} -> {local_path}")
        fs.get(model_path, local_path)
        model_path = local_path

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        n_users = checkpoint.get("n_users", checkpoint.get("num_users"))
        n_items = checkpoint.get("n_items", checkpoint.get("num_items"))
        embed_dim = checkpoint.get("embed_dim", 64)
        hidden_dim = checkpoint.get("hidden_dim", 256)
        n_categories = checkpoint.get("n_categories", 0)
        _user_to_idx = checkpoint.get("user_to_idx", {})
        _item_to_idx = checkpoint.get("item_to_idx", {})
        _cat_to_idx = checkpoint.get("cat_to_idx", {})
        raw_item_cat = checkpoint.get("item_id_to_cat")
        raw_user_cat = checkpoint.get("user_id_to_cat")
    else:
        state_dict = checkpoint
        n_users = state_dict["user_embed.weight"].shape[0]
        n_items = state_dict["item_embed.weight"].shape[0]
        embed_dim = state_dict["user_embed.weight"].shape[1]
        hidden_dim = state_dict["user_mlp.0.weight"].shape[0]
        n_categories = 0
        raw_item_cat = None
        raw_user_cat = None

    _idx_to_item = {v: k for k, v in _item_to_idx.items()}
    _all_item_ids = list(_item_to_idx.keys())

    if raw_item_cat is not None:
        _item_id_to_cat = torch.tensor(raw_item_cat, dtype=torch.long) if not isinstance(raw_item_cat, torch.Tensor) else raw_item_cat
    if raw_user_cat is not None:
        _user_id_to_cat = torch.tensor(raw_user_cat, dtype=torch.long) if not isinstance(raw_user_cat, torch.Tensor) else raw_user_cat

    _model = TwoTower(n_users, n_items, embed_dim, hidden_dim, n_categories)
    _model.load_state_dict(state_dict)
    _model.eval()

    with torch.no_grad():
        _all_item_indices = torch.arange(n_items, dtype=torch.long)
        item_cats = _item_id_to_cat if _item_id_to_cat is not None else None
        _all_item_embeddings = tnf.normalize(
            _model.item_tower(_all_item_indices, item_cats), p=2, dim=-1,
        )

    print(f"Model loaded: {n_users} users, {n_items} items, "
          f"n_categories={n_categories}, embed_dim={embed_dim}, hidden_dim={hidden_dim}, "
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
    num_scored: int = 0


class UserProfileRequest(BaseModel):
    user_id: str


class UserProfileResponse(BaseModel):
    user_id: str
    profile: dict


def _enrich_with_feast(results: list[dict]) -> list[dict]:
    """Look up product metadata from Feast for recommended item_ids."""
    if not _feast_store or not results:
        return results
    try:
        entity_rows = [{"item_id": r["item_id"]} for r in results]
        features = _feast_store.get_online_features(
            features=ITEM_DISPLAY_FEATURES, entity_rows=entity_rows,
        ).to_dict()
        for i, rec in enumerate(results):
            title = features.get("item_title", [None] * len(results))[i]
            rec["title"] = title if title else ""
            brand = features.get("item_brand", [None] * len(results))[i]
            rec["brand"] = brand if brand else ""
            cat = features.get("item_category", [None] * len(results))[i]
            rec["category"] = cat if cat else ""
            rec["avg_rating"] = features.get("item_avg_rating", [None] * len(results))[i]
            rec["review_count"] = features.get("item_review_count", [None] * len(results))[i]
            price = features.get("item_price", [None] * len(results))[i]
            rec["price"] = round(price, 2) if price else None
            rec["rating_stddev"] = features.get("item_rating_stddev", [None] * len(results))[i]
            helpful = features.get("item_total_helpful_votes", [None] * len(results))[i]
            rec["helpful_votes"] = int(helpful) if helpful else None
            rec["has_metadata"] = bool(title)
    except Exception as e:
        print(f"Feast enrichment failed: {e}")
    return results


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
            user_cat = None
            if _user_id_to_cat is not None:
                user_cat = _user_id_to_cat[user_idx].unsqueeze(0)
            user_emb = tnf.normalize(
                _model.user_tower(user_t, user_cat), p=2, dim=-1,
            )

            if request.candidate_items:
                idxs = [_item_to_idx.get(iid, 0) for iid in request.candidate_items]
                item_t = torch.tensor(idxs, dtype=torch.long)
                item_cats = _item_id_to_cat[item_t] if _item_id_to_cat is not None else None
                item_embs = tnf.normalize(
                    _model.item_tower(item_t, item_cats), p=2, dim=-1,
                )
                scores_t = torch.sigmoid((user_emb * item_embs).sum(dim=1))
                item_ids = request.candidate_items
                num_scored = len(item_ids)
            else:
                scores_t = torch.sigmoid((user_emb * _all_item_embeddings).sum(dim=1))
                item_ids = None
                num_scored = scores_t.shape[0]

            fetch_k = min(request.top_k * _OVER_FETCH_FACTOR, scores_t.shape[0])
            top_scores, top_indices = torch.topk(scores_t, fetch_k)

        raw_results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            iid = item_ids[idx] if item_ids else _idx_to_item.get(idx, str(idx))
            raw_results.append({"item_id": iid, "score": round(score, 4)})

        enriched = _enrich_with_feast(raw_results)

        # R2: prefer items with metadata, fill remainder with un-enriched items
        with_meta = [r for r in enriched if r.get("has_metadata")]
        without_meta = [r for r in enriched if not r.get("has_metadata")]
        final = with_meta[:request.top_k]
        if len(final) < request.top_k:
            final.extend(without_meta[:request.top_k - len(final)])

        CANDIDATES_SCORED.observe(num_scored)
        REQUESTS_TOTAL.labels(status="ok").inc()
        return RecommendResponse(recommendations=final, num_scored=num_scored)
    except Exception as e:
        REQUESTS_TOTAL.labels(status="error").inc()
        raise e
    finally:
        REQUEST_DURATION.observe(time.perf_counter() - t0)


@app.post("/v1/models/smartshop-rec:user-profile", response_model=UserProfileResponse)
def user_profile(request: UserProfileRequest):
    """Return Feast user features for persona context display."""
    profile = {}
    if _feast_store:
        try:
            features = _feast_store.get_online_features(
                features=USER_PROFILE_FEATURES,
                entity_rows=[{"user_id": request.user_id}],
            ).to_dict()
            profile = {
                "avg_rating": features.get("user_avg_rating", [None])[0],
                "review_count": features.get("user_review_count", [None])[0],
                "primary_category": features.get("user_primary_category", [None])[0],
                "tenure_days": features.get("user_tenure_days", [None])[0],
            }
        except Exception as e:
            print(f"User profile lookup failed: {e}")
    return UserProfileResponse(user_id=request.user_id, profile=profile)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "has_mappings": bool(_user_to_idx),
        "num_items": len(_all_item_ids),
        "feast_connected": _feast_store is not None,
    }
