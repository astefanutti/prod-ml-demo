"""LLM serving wrapper for review summarization.

In production this runs as a KServe InferenceService with vLLM runtime
(OpenAI-compatible at /v1/completions). This module is a lightweight
wrapper for local integration testing.

Usage (local testing):
    python serving/llm/server.py --model-path models/llm-adapter
"""

import argparse
import os
import time
from contextlib import asynccontextmanager

import fsspec
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel

REQUEST_DURATION = Histogram(
    "smartshop_llm_request_duration_seconds", "Generation latency",
    ["endpoint"],
    buckets=[.1, .25, .5, 1, 2.5, 5, 10, 30],
)
TOKENS_GENERATED = Histogram(
    "smartshop_llm_tokens_generated", "Output tokens per request",
    ["endpoint"],
    buckets=[10, 25, 50, 100, 200, 512],
)
REQUESTS_TOTAL = Counter(
    "smartshop_llm_requests_total", "Total LLM requests", ["endpoint", "status"],
)

_model = None
_tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer

    model_path = os.environ.get("MODEL_PATH", "models/llm-adapter")
    base_model = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {base_model}")
    _tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    _tokenizer.pad_token = _tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    # Check existence via fsspec so S3 paths work too
    fs, _ = fsspec.core.url_to_fs(model_path)
    if fs.exists(model_path):
        print(f"Loading LoRA adapter: {model_path}")
        _model = PeftModel.from_pretrained(base, model_path)
    else:
        print(f"Adapter not found at {model_path}, using base model")
        _model = base

    _model.eval()
    print("Model loaded successfully")
    yield


app = FastAPI(title="SmartShop Review Summarizer", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


class SummarizeRequest(BaseModel):
    product_name: str
    review_text: str
    max_length: int = 256


class SummarizeResponse(BaseModel):
    summary: str
    sentiment: str


class CompletionRequest(BaseModel):
    """OpenAI-compatible request schema — used by rag/server.py and vLLM passthrough."""
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.3


class CompletionResponse(BaseModel):
    choices: list[dict]


def _generate(prompt: str, max_new_tokens: int, temperature: float) -> str:
    import torch
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
        )
    return _tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


@app.post("/v1/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    t0 = time.perf_counter()
    try:
        prompt = (
            f"[INST] Summarize the following product review in 1-2 sentences. "
            f"Include the overall sentiment (positive/negative/neutral).\n\n"
            f"Product: {request.product_name}\n"
            f"Review: {request.review_text} [/INST]"
        )

        response = _generate(prompt, request.max_length, temperature=0.3)
        TOKENS_GENERATED.labels(endpoint="summarize").observe(len(response.split()))
        response_lower = response.lower()
        sentiment = (
            "positive" if "positive" in response_lower
            else "negative" if "negative" in response_lower
            else "neutral"
        )
        REQUESTS_TOTAL.labels(endpoint="summarize", status="ok").inc()
        return SummarizeResponse(summary=response.strip(), sentiment=sentiment)
    except Exception as e:
        REQUESTS_TOTAL.labels(endpoint="summarize", status="error").inc()
        raise e
    finally:
        REQUEST_DURATION.labels(endpoint="summarize").observe(time.perf_counter() - t0)


@app.post("/v1/completions", response_model=CompletionResponse)
def completions(request: CompletionRequest):
    """OpenAI-compatible endpoint — called by rag/server.py and integration tests."""
    t0 = time.perf_counter()
    try:
        text = _generate(request.prompt, request.max_tokens, request.temperature)
        TOKENS_GENERATED.labels(endpoint="completions").observe(len(text.split()))
        REQUESTS_TOTAL.labels(endpoint="completions", status="ok").inc()
        return CompletionResponse(choices=[{"text": text, "finish_reason": "stop"}])
    except Exception as e:
        REQUESTS_TOTAL.labels(endpoint="completions", status="error").inc()
        raise e
    finally:
        REQUEST_DURATION.labels(endpoint="completions").observe(time.perf_counter() - t0)


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": _model is not None}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/llm-adapter")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    os.environ["MODEL_PATH"] = args.model_path
    uvicorn.run(app, host="0.0.0.0", port=args.port)
