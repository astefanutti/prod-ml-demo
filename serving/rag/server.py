"""RAG (Retrieval-Augmented Generation) server for product Q&A.

Uses Feast vector store for similarity search over review embeddings,
then passes retrieved context to the LLM for answer generation.

Usage:
    python serving/rag/server.py
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import requests
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

REQUEST_DURATION = Histogram(
    "smartshop_rag_request_duration_seconds", "Total ask latency",
    buckets=[.1, .25, .5, 1, 2.5, 5, 10, 30],
)
RETRIEVAL_DURATION = Histogram(
    "smartshop_rag_retrieval_duration_seconds", "Feast/Milvus vector search time",
    buckets=[.01, .025, .05, .1, .25, .5, 1, 2.5],
)
LLM_DURATION = Histogram(
    "smartshop_rag_llm_duration_seconds", "LLM generation time",
    buckets=[.1, .25, .5, 1, 2.5, 5, 10, 30],
)
SOURCES_RETRIEVED = Histogram(
    "smartshop_rag_sources_retrieved", "Sources returned per request",
    buckets=[0, 1, 3, 5, 10, 20],
)
REQUESTS_TOTAL = Counter(
    "smartshop_rag_requests_total", "Total RAG requests", ["status"],
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embed_model: Optional[SentenceTransformer] = None
_feast_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embed_model, _feast_store
    from feast import FeatureStore
    from feast.repo_config import load_repo_config

    _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    feast_repo = os.environ.get("FEAST_REPO_PATH", "feast/feature_repo")
    feast_config = os.environ.get("FEAST_CONFIG", "feature_store_serving.yaml")
    config_path = os.path.join(feast_repo, feast_config)
    repo_config = load_repo_config(repo_path=feast_repo, fs_yaml_file=config_path)
    _feast_store = FeatureStore(config=repo_config)
    online_type = getattr(repo_config.online_store, "type", "unknown")
    print(f"RAG server ready — online store: {online_type}")
    yield


app = FastAPI(title="SmartShop Product Q&A (RAG)", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


class AskRequest(BaseModel):
    question: str
    product_id: str = ""
    top_k: int = 5
    max_length: int = 512


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


def retrieve_similar_reviews(question: str, product_id: str, top_k: int) -> list[dict]:
    """Retrieve similar reviews using Feast vector store (Milvus)."""
    query_embedding = _embed_model.encode(question).tolist()

    try:
        result_df = _feast_store.retrieve_online_documents_v2(
            features=[
                "review_embeddings:embedding",
                "review_embeddings:embed_text",
                "review_embeddings:item_id",
                "review_embeddings:rating",
                "review_embeddings:review_title",
            ],
            query=query_embedding,
            top_k=top_k,
            distance_metric="COSINE",
        ).to_df()

        # Optional: filter to a specific product if provided
        if product_id and "review_embeddings__item_id" in result_df.columns:
            product_df = result_df[result_df["review_embeddings__item_id"] == product_id]
            if not product_df.empty:
                result_df = product_df

        sources = []
        for _, row in result_df.iterrows():
            sources.append({
                "item_id": row.get("review_embeddings__item_id", ""),
                "text": row.get("review_embeddings__embed_text", ""),
                "rating": row.get("review_embeddings__rating", 0),
                "title": row.get("review_embeddings__review_title", ""),
            })
        return sources
    except Exception as e:
        print(f"Vector search error: {e}")
        return []


def generate_answer(question: str, context: str) -> str:
    """Generate an answer using the LLM service."""
    llm_url = os.environ.get(
        "LLM_URL", "http://localhost:8001/v1/completions"
    )

    prompt = (
        f"[INST] Based on the following product reviews, answer the user's question. "
        f"Be helpful, accurate, and concise. If the reviews don't contain enough "
        f"information, say so.\n\n"
        f"Reviews:\n{context}\n\n"
        f"Question: {question} [/INST]"
    )

    try:
        response = requests.post(
            llm_url,
            json={
                "model": os.environ.get("LLM_MODEL", "smartshop-llm"),
                "prompt": prompt,
                "max_tokens": 512,
                "temperature": 0.3,
            },
            timeout=30,
        )
        if response.ok:
            return response.json().get("choices", [{}])[0].get("text", "").strip()
    except Exception as e:
        print(f"LLM service error: {e}")

    return "I'm sorry, I couldn't generate an answer. The LLM service may be unavailable."


@app.post("/v1/ask", response_model=AskResponse)
def ask(request: AskRequest):
    t0 = time.perf_counter()
    try:
        t_ret = time.perf_counter()
        sources = retrieve_similar_reviews(request.question, request.product_id, request.top_k)
        RETRIEVAL_DURATION.observe(time.perf_counter() - t_ret)
        SOURCES_RETRIEVED.observe(len(sources))

        if sources:
            context_parts = []
            for i, src in enumerate(sources, 1):
                rating = src.get("rating", "?")
                title = src.get("title", "")
                text = src.get("text", "")[:500]
                prefix = f"Review {i} ({rating}/5 stars)"
                if title:
                    prefix += f" — {title[:60]}"
                context_parts.append(f"{prefix}: {text}")
            context = "\n\n".join(context_parts)
        else:
            context = "No relevant reviews found."

        t_llm = time.perf_counter()
        answer = generate_answer(request.question, context)
        LLM_DURATION.observe(time.perf_counter() - t_llm)

        REQUESTS_TOTAL.labels(status="ok").inc()
        return AskResponse(answer=answer, sources=sources)
    except Exception as e:
        REQUESTS_TOTAL.labels(status="error").inc()
        raise e
    finally:
        REQUEST_DURATION.observe(time.perf_counter() - t0)


@app.get("/health")
def health():
    return {"status": "healthy", "embed_model_loaded": _embed_model is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
