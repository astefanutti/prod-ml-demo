"""RAG (Retrieval-Augmented Generation) server for product Q&A.

Uses Feast vector store for similarity search over review embeddings,
then passes retrieved context to the LLM for answer generation.

Usage:
    python serving/rag/server.py
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embed_model: Optional[SentenceTransformer] = None
_feast_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embed_model, _feast_store
    from feast import FeatureStore

    _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    feast_repo = os.environ.get("FEAST_REPO_PATH", "feast/feature_repo")
    _feast_store = FeatureStore(repo_path=feast_repo)
    print("RAG server ready")
    yield


app = FastAPI(title="SmartShop Product Q&A (RAG)", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    product_id: str = ""
    top_k: int = 5
    max_length: int = 512


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


def retrieve_similar_reviews(question: str, product_id: str, top_k: int) -> list[dict]:
    """Retrieve similar reviews using Feast vector store."""
    query_embedding = _embed_model.encode(question).tolist()

    try:
        results = _feast_store.retrieve_online_documents(
            feature="review_embeddings:embedding",
            query=query_embedding,
            top_k=top_k,
        )

        sources = []
        if results and hasattr(results, "to_dict"):
            result_dict = results.to_dict()
            for i in range(len(result_dict.get("review_id", []))):
                sources.append({
                    "review_id": result_dict.get("review_id", [None])[i],
                    "item_id": result_dict.get("item_id", [""])[i],
                    "text": result_dict.get("embed_text", [""])[i],
                    "rating": result_dict.get("rating", [0])[i],
                })
        return sources
    except Exception as e:
        print(f"Vector search fallback (Feast not available): {e}")
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
    # Step 1: Retrieve similar reviews
    sources = retrieve_similar_reviews(request.question, request.product_id, request.top_k)

    # Step 2: Build context from retrieved reviews
    if sources:
        context_parts = []
        for i, src in enumerate(sources, 1):
            rating = src.get("rating", "?")
            text = src.get("text", "")[:500]  # Truncate long reviews
            context_parts.append(f"Review {i} ({rating}/5 stars): {text}")
        context = "\n\n".join(context_parts)
    else:
        context = "No relevant reviews found."

    # Step 3: Generate answer
    answer = generate_answer(request.question, context)

    return AskResponse(answer=answer, sources=sources)


@app.get("/health")
def health():
    return {"status": "healthy", "embed_model_loaded": _embed_model is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
