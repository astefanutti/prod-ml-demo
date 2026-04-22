"""Gradio demo UI for SmartShop AI — Red Hat Summit 2026.

Tabs:
  1. Recommendations    — Two-Tower model via KServe + Feast online features
  2. Review Summarizer  — Fine-tuned Mistral-7B (FSDP on Slurm) via KServe
  3. Product Q&A (RAG)  — Milvus + Mistral-7B for product knowledge retrieval
  4. Platform Metrics   — Live GPU/Redis/pipeline metrics + RAPIDS speedup proof

Usage:
    python demo/app.py
    # or
    make demo
"""

import json
import os
import ssl
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

import gradio as gr
import requests

# ── Service endpoints ─────────────────────────────────────────────────────────
RECOMMEND_URL = os.environ.get("RECOMMEND_URL", "http://localhost:8000/v1/models/smartshop-rec:predict")
SUMMARIZE_URL = os.environ.get("SUMMARIZE_URL", "http://localhost:8001/v1/summarize")
RAG_URL       = os.environ.get("RAG_URL", "http://localhost:8002/v1/ask")

# ── Observability endpoints ────────────────────────────────────────────────────
PROM_URL       = os.environ.get("PROMETHEUS_URL", "https://thanos-querier.openshift-monitoring.svc.cluster.local:9091")
PROM_TOKEN     = os.environ.get("PROMETHEUS_TOKEN", "")
MLFLOW_URI     = os.environ.get("MLFLOW_TRACKING_URI", "")
GRAFANA_URL    = os.environ.get("GRAFANA_URL", "")   # e.g. https://grafana-smartshop.apps.<cluster>
NAMESPACE      = os.environ.get("NAMESPACE", "smartshop")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _prom_instant(promql: str) -> list[dict]:
    """Single-point Prometheus query. Returns [] on error."""
    if not PROM_URL:
        return []
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    headers = {"Authorization": f"Bearer {PROM_TOKEN}"} if PROM_TOKEN else {}
    url = f"{PROM_URL}/api/v1/query?query={urllib.parse.quote(promql)}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=5) as r:
            data = json.loads(r.read())
        return data.get("data", {}).get("result", [])
    except Exception:
        return []


def _prom_range(promql: str, minutes: int = 30, step: str = "30s") -> list[dict]:
    """Range Prometheus query. Returns [] on error."""
    if not PROM_URL:
        return []
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    headers = {"Authorization": f"Bearer {PROM_TOKEN}"} if PROM_TOKEN else {}
    end   = datetime.utcnow()
    start = end - timedelta(minutes=minutes)
    url = (
        f"{PROM_URL}/api/v1/query_range"
        f"?query={urllib.parse.quote(promql)}"
        f"&start={start.timestamp():.0f}"
        f"&end={end.timestamp():.0f}"
        f"&step={step}"
    )
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=8) as r:
            data = json.loads(r.read())
        return data.get("data", {}).get("result", [])
    except Exception:
        return []


def _val(results: list[dict], default=None):
    """Extract first scalar value from Prometheus instant result."""
    try:
        return float(results[0]["value"][1])
    except Exception:
        return default


def _mlflow_speedup() -> float | None:
    """Fetch GPU vs CPU speedup ratio from MLflow. Cached for 60 seconds."""
    if not MLFLOW_URI or not _mlflow_speedup._cached_at or (time.time() - _mlflow_speedup._cached_at > 60):
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_URI)
            client = mlflow.tracking.MlflowClient()
            exp = client.get_experiment_by_name("smartshop-feature-engineering")
            if not exp:
                return None
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=10,
            )
            rapids = next((r for r in runs if r.data.tags.get("rapids_active") == "True" and r.info.status == "FINISHED"), None)
            cpu    = next((r for r in runs if r.data.tags.get("rapids_active") == "False" and r.info.status == "FINISHED"), None)
            if rapids and cpu:
                t_rapids = rapids.data.metrics.get("total_elapsed_s")
                t_cpu    = cpu.data.metrics.get("total_elapsed_s")
                if t_rapids and t_cpu and t_rapids > 0:
                    speedup = round(t_cpu / t_rapids, 1)
                    _mlflow_speedup._value     = speedup
                    _mlflow_speedup._cached_at = time.time()
                    return speedup
        except Exception:
            pass
    return getattr(_mlflow_speedup, "_value", None)

_mlflow_speedup._cached_at = None
_mlflow_speedup._value     = None


# ── Per-request latency tracker ───────────────────────────────────────────────
_last_request_ms: float = 0.0


# ── Service call handlers ─────────────────────────────────────────────────────

def get_recommendations(user_id: str, top_k: int = 10) -> tuple[str, str]:
    global _last_request_ms
    t0 = time.time()
    try:
        response = requests.post(
            RECOMMEND_URL,
            json={"user_id": user_id, "top_k": int(top_k)},
            timeout=10,
        )
        elapsed_ms = (time.time() - t0) * 1000
        _last_request_ms = elapsed_ms
        if response.ok:
            recs = response.json()["recommendations"]
            lines = [f"**Top {len(recs)} Recommendations for `{user_id}`**\n"]
            for i, rec in enumerate(recs, 1):
                lines.append(f"{i}. `{rec['item_id']}` — score: **{rec['score']:.3f}**")
            return "\n".join(lines), f"✅ {elapsed_ms:.0f} ms"
        return f"❌ {response.status_code}: {response.text}", ""
    except requests.ConnectionError:
        return "❌ Recommendation service not available — deploy KServe InferenceService first.", ""
    except Exception as e:
        return f"❌ {e}", ""


def summarize_review(product_name: str, review_text: str) -> str:
    try:
        response = requests.post(
            SUMMARIZE_URL,
            json={"product_name": product_name, "review_text": review_text},
            timeout=30,
        )
        if response.ok:
            result = response.json()
            return f"**Summary:** {result['summary']}\n\n**Sentiment:** {result['sentiment']}"
        return f"❌ {response.status_code}: {response.text}"
    except requests.ConnectionError:
        return "❌ LLM service not available — deploy KServe InferenceService first."
    except Exception as e:
        return f"❌ {e}"


def ask_question(question: str, product_id: str = "") -> str:
    try:
        response = requests.post(
            RAG_URL,
            json={"question": question, "product_id": product_id},
            timeout=30,
        )
        if response.ok:
            result = response.json()
            answer  = result["answer"]
            sources = result.get("sources", [])
            output  = f"**Answer:** {answer}\n"
            if sources:
                output += f"\n*Based on {len(sources)} reviews:*"
                for src in sources[:3]:
                    output += f"\n- ({src.get('rating', '?')}/5) {src.get('text', '')[:200]}…"
            return output
        return f"❌ {response.status_code}: {response.text}"
    except requests.ConnectionError:
        return "❌ RAG service not available — deploy KServe InferenceService first."
    except Exception as e:
        return f"❌ {e}"


# ── Metrics refresh functions ─────────────────────────────────────────────────

def _service_badge(url: str, name: str, timeout: float = 2.0) -> str:
    try:
        r = requests.get(url.replace("/v1/models/smartshop-rec:predict", "/v2/health/ready")
                             .replace("/v1/summarize", "/health")
                             .replace("/v1/ask", "/health"),
                         timeout=timeout)
        ok = r.status_code < 400
    except Exception:
        ok = False
    icon = "🟢" if ok else "🔴"
    return f"{icon} {name}"


def fetch_pipeline_status() -> str:
    services = [
        (RECOMMEND_URL, "Rec model (KServe)"),
        (SUMMARIZE_URL, "LLM summarizer (KServe)"),
        (RAG_URL,       "RAG (KServe)"),
    ]
    # Check Feast and Redis via Prometheus (simpler than direct socket)
    feast_up  = bool(_prom_instant(f'up{{namespace="{NAMESPACE}", job=~".*feast.*"}}'))
    redis_up  = bool(_prom_instant(f'redis_up{{namespace="{NAMESPACE}"}}'))

    lines = ["### Service Status\n"]
    for url, name in services:
        lines.append(_service_badge(url, name))
    lines.append(f"{'🟢' if feast_up else '🔴'} Feast feature server")
    lines.append(f"{'🟢' if redis_up else '🔴'} Redis online store")
    return "\n".join(lines)


def fetch_headline_metrics() -> str:
    gpu_util  = _val(_prom_instant("avg(DCGM_FI_DEV_GPU_UTIL)"))
    gpu_mem   = _val(_prom_instant("avg(DCGM_FI_DEV_FB_USED)"))
    redis_ops = _val(_prom_instant(f'rate(redis_commands_processed_total{{namespace="{NAMESPACE}"}}[1m])'))
    redis_hit = _val(_prom_instant(
        f'rate(redis_keyspace_hits_total{{namespace="{NAMESPACE}"}}[5m]) / '
        f'(rate(redis_keyspace_hits_total{{namespace="{NAMESPACE}"}}[5m]) + '
        f'rate(redis_keyspace_misses_total{{namespace="{NAMESPACE}"}}[5m]))'
    ))
    redis_keys = _val(_prom_instant(f'sum(redis_db_keys{{namespace="{NAMESPACE}"}})'))
    speedup    = _mlflow_speedup()

    lines = ["### Live Platform Metrics\n"]

    # RAPIDS headline
    if speedup:
        lines.append(f"⚡ **RAPIDS GPU Speedup: {speedup}× faster** than CPU feature engineering")
    else:
        lines.append("⚡ RAPIDS GPU speedup: _run A/B comparison to compute_")

    lines.append("")

    # GPU
    gpu_str = f"{gpu_util:.0f}%" if gpu_util is not None else "_no DCGM data_"
    mem_str = f"{gpu_mem:.0f} MB" if gpu_mem is not None else "—"
    lines.append(f"🖥️  **GPU utilization:** {gpu_str}  |  **FB memory:** {mem_str}")

    # Redis
    ops_str   = f"{redis_ops:,.0f} ops/sec" if redis_ops is not None else "_deploy redis-exporter_"
    hit_str   = f"{redis_hit*100:.1f}%" if redis_hit is not None else "—"
    keys_str  = f"{redis_keys:,.0f} keys" if redis_keys is not None else "—"
    lines.append(f"📦  **Redis ops:** {ops_str}  |  **hit ratio:** {hit_str}  |  **feature keys:** {keys_str}")

    # Last request
    if _last_request_ms > 0:
        lines.append(f"⏱️  **Last recommendation latency:** {_last_request_ms:.0f} ms")

    return "\n".join(lines)


def fetch_gpu_plot():
    """Return a matplotlib figure for GPU utilization (last 30 min)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors='#94a3b8')
        ax.spines[:].set_color('#334155')
        ax.yaxis.label.set_color('#94a3b8')
        ax.title.set_color('#e2e8f0')

    # GPU Utilization
    results = _prom_range("avg(DCGM_FI_DEV_GPU_UTIL)", minutes=30)
    if results:
        for series in results[:8]:  # cap at 8 GPUs
            pts = series.get("values", [])
            if pts:
                ts  = [datetime.fromtimestamp(float(t)) for t, _ in pts]
                val = [float(v) for _, v in pts]
                axes[0].plot(ts, val, linewidth=1.2, alpha=0.8)
        axes[0].set_ylim(0, 105)
        axes[0].axhline(80, color='#f97316', linestyle='--', alpha=0.5, linewidth=0.8)
    else:
        axes[0].text(0.5, 0.5, "No DCGM data\n(check prometheus token)", color='#64748b',
                     ha='center', transform=axes[0].transAxes, fontsize=10)
    axes[0].set_title("GPU Utilization % (all GPUs)", fontsize=11)
    axes[0].set_ylabel("%")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()

    # Redis ops/sec
    ns = NAMESPACE
    redis_results = _prom_range(
        f'rate(redis_commands_processed_total{{namespace="{ns}"}}[1m])', minutes=30
    )
    if redis_results:
        pts = redis_results[0].get("values", [])
        ts  = [datetime.fromtimestamp(float(t)) for t, _ in pts]
        val = [float(v) for _, v in pts]
        axes[1].plot(ts, val, color='#22d3ee', linewidth=1.5)
        axes[1].fill_between(ts, val, alpha=0.15, color='#22d3ee')
    else:
        axes[1].text(0.5, 0.5, "No redis_exporter data\n(deploy redis-exporter.yaml)", color='#64748b',
                     ha='center', transform=axes[1].transAxes, fontsize=10)
    axes[1].set_title("Redis Feature Store — ops/sec", fontsize=11)
    axes[1].set_ylabel("ops/sec")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    plt.tight_layout(pad=1.5)
    return fig


def grafana_panels_html() -> str:
    """Return HTML with embedded Grafana panel iframes, or a link if URL is set."""
    if not GRAFANA_URL:
        return (
            "<div style='padding:16px; color:#64748b; font-size:13px;'>"
            "ℹ️ Set <code>GRAFANA_URL</code> env var to embed live Grafana panels here.<br>"
            "Deploy: <code>envsubst &lt; infrastructure/openshift/grafana.yaml | oc apply -f -</code>"
            "</div>"
        )
    base = GRAFANA_URL.rstrip("/")
    panels = [
        (f"{base}/d-solo/smartshop-gpu/smartshop-gpu-performance?orgId=1&panelId=1&refresh=10s&theme=dark",
         "GPU Utilization %"),
        (f"{base}/d-solo/smartshop-gpu/smartshop-gpu-performance?orgId=1&panelId=3&refresh=10s&theme=dark",
         "SM Active Ratio"),
        (f"{base}/d-solo/smartshop-redis/smartshop-redis-feature-store?orgId=1&panelId=1&refresh=10s&theme=dark",
         "Redis ops/sec"),
        (f"{base}/d-solo/smartshop-redis/smartshop-redis-feature-store?orgId=1&panelId=2&refresh=10s&theme=dark",
         "Cache Hit Ratio"),
    ]
    iframes = ""
    for url, title in panels:
        iframes += (
            f'<div style="display:inline-block; margin:6px;">'
            f'<p style="color:#94a3b8;font-size:11px;margin:0 0 4px 0;">{title}</p>'
            f'<iframe src="{url}" width="380" height="200" frameborder="0" '
            f'style="border-radius:8px; border:1px solid #334155;"></iframe>'
            f'</div>'
        )
    links = (
        f'<div style="margin-top:12px; font-size:12px; color:#64748b;">'
        f'Full dashboards: '
        f'<a href="{base}/d/smartshop-gpu" target="_blank" style="color:#38bdf8;">GPU Performance</a> · '
        f'<a href="{base}/d/smartshop-redis" target="_blank" style="color:#38bdf8;">Redis Feature Store</a> · '
        f'<a href="{base}/d/smartshop-spark" target="_blank" style="color:#38bdf8;">Spark Executors</a>'
        f'</div>'
    )
    return (
        f'<div style="background:#0f172a; padding:12px; border-radius:10px;">'
        f'{iframes}{links}'
        f'</div>'
    )


# ── UI ────────────────────────────────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    body_background_fill="#0f172a",
    body_background_fill_dark="#0f172a",
    block_background_fill="#1e293b",
    block_background_fill_dark="#1e293b",
    block_border_color="#334155",
    block_title_text_color="#e2e8f0",
    body_text_color="#cbd5e1",
    button_primary_background_fill="#2563eb",
    button_primary_text_color="white",
    input_background_fill="#0f172a",
    input_border_color="#334155",
    input_placeholder_color="#475569",
)

with gr.Blocks(
    title="SmartShop AI — Production ML at Scale",
    theme=THEME,
    css="""
    .badge { font-size:11px; padding:2px 8px; border-radius:99px; display:inline-block; }
    .metric-card { background:#1e293b; border:1px solid #334155; border-radius:10px; padding:12px; }
    .speedup-badge { font-size:28px; font-weight:700; color:#f97316; }
    footer { display:none !important; }
    """,
) as demo:

    gr.HTML("""
    <div style="background:linear-gradient(135deg,#1e3a5f,#0f172a); padding:24px 28px; border-radius:12px; margin-bottom:8px;">
      <div style="display:flex; align-items:center; gap:12px;">
        <img src="https://www.redhat.com/rhdc/managed-files/Logo-Red_Hat-A-Standard-RGB.svg"
             height="36" style="filter:brightness(0) invert(1);" onerror="this.style.display='none'"/>
        <div>
          <h1 style="color:#e2e8f0; margin:0; font-size:22px;">SmartShop AI</h1>
          <p style="color:#94a3b8; margin:0; font-size:13px;">
            PyTorch DDP · FSDP on Slurm · Spark RAPIDS · Kubeflow · Feast · KServe
            on <strong style="color:#ef4444;">Red Hat OpenShift AI</strong>
          </p>
        </div>
      </div>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Recommendations ─────────────────────────────────────────────
        with gr.Tab("🛍️  Recommendations"):
            gr.Markdown(
                "**Two-Tower model** trained with PyTorch DDP (4× A100) · "
                "Online features served from **Feast → Redis** in &lt;1 ms · "
                "Deployed on **KServe**"
            )
            with gr.Row():
                with gr.Column(scale=2):
                    user_input = gr.Textbox(
                        label="User ID",
                        placeholder="e.g. AEXAMPLEUSER123",
                        lines=1,
                    )
                    top_k_input = gr.Slider(minimum=1, maximum=50, value=10, step=1,
                                            label="Top K recommendations")
                    rec_btn = gr.Button("Get Recommendations", variant="primary", size="lg")
                with gr.Column(scale=3):
                    rec_output    = gr.Markdown(label="Recommendations")
                    latency_badge = gr.Markdown("")

            rec_btn.click(
                get_recommendations,
                inputs=[user_input, top_k_input],
                outputs=[rec_output, latency_badge],
            )

        # ── Tab 2: Review Summarizer ───────────────────────────────────────────
        with gr.Tab("📝  Review Summarizer"):
            gr.Markdown(
                "**Mistral-7B-Instruct** fine-tuned with QLoRA + **FSDP on Slurm** "
                "(2 nodes × 4 A100) · Deployed on **KServe**"
            )
            with gr.Row():
                with gr.Column():
                    product_input = gr.Textbox(label="Product Name",
                                               placeholder="e.g. Sony WH-1000XM5")
                    review_input  = gr.Textbox(label="Review Text",
                                               placeholder="Paste a product review here...",
                                               lines=7)
                    summary_btn   = gr.Button("Summarize", variant="primary")
                with gr.Column():
                    summary_output = gr.Markdown(label="Summary & Sentiment")
            summary_btn.click(summarize_review, inputs=[product_input, review_input],
                              outputs=summary_output)

        # ── Tab 3: Product Q&A (RAG) ───────────────────────────────────────────
        with gr.Tab("💬  Product Q&A (RAG)"):
            gr.Markdown(
                "**Milvus** vector store (review embeddings) + **Mistral-7B** generation · "
                "Feast provides context features · Deployed on **KServe**"
            )
            with gr.Row():
                with gr.Column():
                    question_input   = gr.Textbox(label="Question",
                                                  placeholder="Is this laptop good for gaming?")
                    product_id_input = gr.Textbox(label="Product ID (optional, ASIN)",
                                                  placeholder="B09XXXXX")
                    qa_btn           = gr.Button("Ask", variant="primary")
                with gr.Column():
                    qa_output = gr.Markdown(label="Answer")
            qa_btn.click(ask_question, inputs=[question_input, product_id_input],
                         outputs=qa_output)

        # ── Tab 4: Platform Metrics ────────────────────────────────────────────
        with gr.Tab("📊  Platform Metrics"):
            gr.Markdown(
                "Live observability from **DCGM/Prometheus** · **Redis Exporter** · "
                "**MLflow** — auto-refreshes every 15 seconds."
            )

            with gr.Row():
                with gr.Column(scale=3):
                    headline_md = gr.Markdown(fetch_headline_metrics())
                with gr.Column(scale=2):
                    status_md   = gr.Markdown(fetch_pipeline_status())

            gpu_plot = gr.Plot(fetch_gpu_plot(), label="GPU Utilization & Redis ops/sec (last 30 min)")

            gr.Markdown("### Live Grafana Dashboards")
            grafana_html = gr.HTML(grafana_panels_html())

            with gr.Row():
                refresh_btn = gr.Button("🔄  Refresh Now", variant="secondary", size="sm")
                gr.Markdown(
                    "_Auto-refreshes every 15s. "
                    "[Full Grafana →](" + (GRAFANA_URL or "#") + ")_",
                    elem_classes=["metric-card"],
                )

            def _refresh_all():
                return (
                    fetch_headline_metrics(),
                    fetch_pipeline_status(),
                    fetch_gpu_plot(),
                    grafana_panels_html(),
                )

            refresh_btn.click(
                _refresh_all,
                outputs=[headline_md, status_md, gpu_plot, grafana_html],
            )

            # Auto-refresh every 15 seconds while the tab is visible
            demo.load(
                _refresh_all,
                outputs=[headline_md, status_md, gpu_plot, grafana_html],
                every=15,
            )

        # ── Tab 5: Pipeline Architecture ──────────────────────────────────────
        with gr.Tab("🏗️  Architecture"):
            gr.Markdown("""
## SmartShop AI — Production ML Pipeline on Red Hat OpenShift AI

```
Amazon Reviews 2023 (571M)
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Data Ingestion (Kubernetes Job)                                  │
│  HuggingFace Hub → streaming → MinIO S3                          │
└───────────────────────┬───────────────────────────────────────────┘
                        │  s3://smartshop-raw/
                        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Feature Engineering  (Kubeflow Spark Operator)                   │
│                                                                   │
│  CPU path:   spark-jobs image  →  Parquet features               │
│  GPU path:   RAPIDS plugin     →  same code, N× faster ⚡        │
│              4× A100-80GB, CUDA 13.0, rapids-4-spark 26.02.2     │
└───────────────────────┬───────────────────────────────────────────┘
                        │  s3://smartshop-features/
                        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Feast Feature Store  (RHOAI Feast Operator)                      │
│  Offline: MinIO Parquet (dask)  →  materialize  →  Redis online  │
└──────────────┬────────────────────────────────────────────────────┘
               │  online features <1ms
               ▼
┌──────────────────────────────┐   ┌────────────────────────────────┐
│  Rec Model Training          │   │  LLM Fine-Tuning               │
│  PyTorch DDP (4× A100, 1 node│   │  FSDP + QLoRA                  │
│  Kubeflow TrainingOperator   │   │  2 nodes × 4 A100              │
│  → s3://smartshop-models/    │   │  Kubeflow → Slurm dispatch     │
└──────────────┬───────────────┘   └──────────────┬─────────────────┘
               │                                  │
               ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  KServe InferenceServices                                           │
│  smartshop-rec  ·  smartshop-llm  ·  smartshop-rag                 │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
                            This Gradio UI
```

### Observability Stack
| Layer | Tool | Metrics |
|---|---|---|
| GPU / RAPIDS | DCGM → Prometheus | Utilization, NVLink BW, power, FB memory |
| Spark internals | PrometheusServlet | Heap, GC, shuffle bytes, BlockManager |
| Feature store | redis_exporter | ops/sec, hit ratio, keys, memory |
| Training runs | MLflow (RHOAI) | Loss curves, throughput, GPU speedup |
| Dashboards | Grafana | GPU / Redis / Spark panels |
| Analysis | `notebooks/metrics_analysis.ipynb` | Publication charts |
""")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
