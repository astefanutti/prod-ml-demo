"""Gradio demo UI for SmartShop AI.

Provides three tabs:
  1. Product Recommendations - Get personalized product suggestions
  2. Review Summarizer - Summarize and analyze product reviews
  3. Product Q&A (RAG) - Ask questions about products using review knowledge

Usage:
    python demo/app.py
    # or
    make demo
"""

import os

import gradio as gr
import requests

# Service endpoints (configurable via environment)
RECOMMEND_URL = os.environ.get("RECOMMEND_URL", "http://localhost:8000/v1/models/smartshop-rec:predict")
SUMMARIZE_URL = os.environ.get("SUMMARIZE_URL", "http://localhost:8001/v1/summarize")
RAG_URL = os.environ.get("RAG_URL", "http://localhost:8002/v1/ask")


def get_recommendations(user_id: str, top_k: int = 10) -> str:
    """Get product recommendations for a user."""
    try:
        response = requests.post(
            RECOMMEND_URL,
            json={"user_id": user_id, "top_k": int(top_k)},
            timeout=10,
        )
        if response.ok:
            recs = response.json()["recommendations"]
            lines = [f"Top {len(recs)} Recommendations for {user_id}:\n"]
            for i, rec in enumerate(recs, 1):
                lines.append(f"  {i}. {rec['item_id']} (score: {rec['score']})")
            return "\n".join(lines)
        return f"Error: {response.status_code} - {response.text}"
    except requests.ConnectionError:
        return "Error: Recommendation service not available. Start with: make serve-rec"
    except Exception as e:
        return f"Error: {e}"


def summarize_review(product_name: str, review_text: str) -> str:
    """Summarize a product review."""
    try:
        response = requests.post(
            SUMMARIZE_URL,
            json={"product_name": product_name, "review_text": review_text},
            timeout=30,
        )
        if response.ok:
            result = response.json()
            return f"Summary: {result['summary']}\nSentiment: {result['sentiment']}"
        return f"Error: {response.status_code} - {response.text}"
    except requests.ConnectionError:
        return "Error: LLM service not available. Start with: make serve-llm"
    except Exception as e:
        return f"Error: {e}"


def ask_question(question: str, product_id: str = "") -> str:
    """Ask a question about a product using RAG."""
    try:
        response = requests.post(
            RAG_URL,
            json={"question": question, "product_id": product_id},
            timeout=30,
        )
        if response.ok:
            result = response.json()
            answer = result["answer"]
            sources = result.get("sources", [])
            output = f"Answer: {answer}\n"
            if sources:
                output += f"\nBased on {len(sources)} reviews:"
                for src in sources[:3]:
                    rating = src.get("rating", "?")
                    text = src.get("text", "")[:200]
                    output += f"\n  - ({rating}/5) {text}..."
            return output
        return f"Error: {response.status_code} - {response.text}"
    except requests.ConnectionError:
        return "Error: RAG service not available. Start with: make serve-rag"
    except Exception as e:
        return f"Error: {e}"


# Build Gradio UI
with gr.Blocks(title="SmartShop AI Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SmartShop AI - Production ML at Scale Demo")
    gr.Markdown(
        "Demonstrating PyTorch Distributed, Kubeflow Trainer, Spark, Feast, "
        "and Slurm on Red Hat OpenShift AI"
    )

    with gr.Tab("Recommendations"):
        gr.Markdown("### Get personalized product recommendations")
        gr.Markdown("Uses a Two-Tower model trained with DDP + Feast online features")
        with gr.Row():
            user_input = gr.Textbox(
                label="User ID",
                placeholder="Enter a user ID (e.g., AEXAMPLEUSER123)",
            )
            top_k_input = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Top K")
        rec_output = gr.Textbox(label="Recommendations", lines=12)
        rec_btn = gr.Button("Get Recommendations", variant="primary")
        rec_btn.click(get_recommendations, inputs=[user_input, top_k_input], outputs=rec_output)

    with gr.Tab("Review Summarizer"):
        gr.Markdown("### Summarize product reviews with fine-tuned Mistral-7B")
        gr.Markdown("Uses QLoRA fine-tuning with FSDP on Slurm")
        product_input = gr.Textbox(label="Product Name", placeholder="e.g., Sony WH-1000XM5")
        review_input = gr.Textbox(
            label="Review Text",
            placeholder="Paste a product review here...",
            lines=6,
        )
        summary_output = gr.Textbox(label="Summary & Sentiment", lines=4)
        summary_btn = gr.Button("Summarize", variant="primary")
        summary_btn.click(
            summarize_review, inputs=[product_input, review_input], outputs=summary_output
        )

    with gr.Tab("Product Q&A (RAG)"):
        gr.Markdown("### Ask questions about products")
        gr.Markdown("Uses Feast vector store for retrieval + Mistral-7B for generation")
        question_input = gr.Textbox(
            label="Question",
            placeholder="e.g., Is this laptop good for gaming?",
        )
        product_id_input = gr.Textbox(
            label="Product ID (optional)",
            placeholder="ASIN to filter by",
        )
        qa_output = gr.Textbox(label="Answer", lines=8)
        qa_btn = gr.Button("Ask", variant="primary")
        qa_btn.click(
            ask_question, inputs=[question_input, product_id_input], outputs=qa_output
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
