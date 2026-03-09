"""LLM serving configuration for review summarization.

In production, this is served via vLLM with the LoRA adapter.
This module provides a lightweight wrapper for testing and the demo UI.

Production deployment uses KServe InferenceService with vLLM runtime.
See infrastructure/openshift/inferenceservices.yaml for the K8s config.

Usage (local testing):
    python serving/llm/server.py --model-path models/llm-adapter
"""

import argparse
import os

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="SmartShop Review Summarizer")

# Globals
model = None
tokenizer = None


class SummarizeRequest(BaseModel):
    product_name: str
    review_text: str
    max_length: int = 256


class SummarizeResponse(BaseModel):
    summary: str
    sentiment: str


@app.on_event("startup")
def load_model():
    global model, tokenizer

    model_path = os.environ.get("MODEL_PATH", "models/llm-adapter")
    base_model = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    if os.path.exists(model_path):
        print(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(base, model_path)
    else:
        print("No adapter found, using base model")
        model = base

    model.eval()
    print("Model loaded successfully")


@app.post("/v1/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    prompt = (
        f"[INST] Summarize the following product review in 1-2 sentences. "
        f"Include the overall sentiment (positive/negative/neutral).\n\n"
        f"Product: {request.product_name}\n"
        f"Review: {request.review_text} [/INST]"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with __import__("torch").no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_length,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Simple sentiment extraction
    response_lower = response.lower()
    if "positive" in response_lower:
        sentiment = "positive"
    elif "negative" in response_lower:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return SummarizeResponse(summary=response.strip(), sentiment=sentiment)


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/llm-adapter")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    os.environ["MODEL_PATH"] = args.model_path
    uvicorn.run(app, host="0.0.0.0", port=args.port)
