"""SmartShop AI — Red Hat Summit 2026 Demo UI.

Production ML at Scale: PyTorch DDP, QLoRA/FSDP, Spark RAPIDS, Kubeflow,
Feast, KServe on Red Hat OpenShift AI.

Usage:
    source .venv/bin/activate && python demo/app.py
"""

import hashlib
_original_md5 = hashlib.md5
def _fips_md5(*args, **kwargs):
    kwargs.setdefault("usedforsecurity", False)
    return _original_md5(*args, **kwargs)
hashlib.md5 = _fips_md5

import json
import os
import ssl
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

import gradio as gr
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
SESSION = requests.Session()
SESSION.verify = False

# ── Config ────────────────────────────────────────────────────────────────────

RECOMMEND_URL = os.environ.get(
    "RECOMMEND_URL",
    "http://localhost:8000/v1/models/smartshop-rec:predict",
)
LLM_URL = os.environ.get(
    "LLM_URL",
    os.environ.get("SUMMARIZE_URL", "http://localhost:8001/v1/completions"),
)
RAG_URL = os.environ.get(
    "RAG_URL", "http://localhost:8002/v1/ask"
)
PROM_URL = os.environ.get("PROMETHEUS_URL", "")
PROM_TOKEN = os.environ.get("PROMETHEUS_TOKEN", "")
GRAFANA_URL = os.environ.get("GRAFANA_URL", "")
NAMESPACE = os.environ.get("NAMESPACE", "smartshop")

# ── Session Stats (always works, no Prometheus needed) ────────────────────────

_session_start = time.time()
_session_stats: dict[str, list[float]] = {"rec": [], "llm": [], "rag": []}


def _record(service: str, ms: float):
    _session_stats[service].append(ms)


def _stats_summary() -> dict:
    all_latencies = sum(_session_stats.values(), [])
    total = len(all_latencies)
    avg = sum(all_latencies) / total if total else 0
    best = min(all_latencies) if all_latencies else 0
    worst = max(all_latencies) if all_latencies else 0
    uptime = int(time.time() - _session_start)
    return {
        "total": total,
        "avg": avg,
        "best": best,
        "worst": worst,
        "uptime_min": uptime // 60,
        "rec": len(_session_stats["rec"]),
        "llm": len(_session_stats["llm"]),
        "rag": len(_session_stats["rag"]),
    }


# ── Shopper Personas ──────────────────────────────────────────────────────────

PERSONAS = [
    ("Tech Enthusiast", "AE2222HXKLMAEK4SG56OW23V3LTA"),
    ("Book Lover", "AE222ARFTTB3OUTW2RXSHTO3DBZQ"),
    ("Fitness Buff", "AE222BGFMVEFSSH4ECHR5FTL4Z4Q"),
    ("Home Chef", "AE222CDSB6TPQEN2W5PHQCAKERRQ"),
    ("Music Fan", "AE222CLD7MMLOFP37THNLRRPBGZA"),
]

PERSONA_CONTEXT = {
    "Tech Enthusiast": "Frequent electronics buyer · avg rating 4.2 · 47 past purchases",
    "Book Lover": "Avid reader · prefers non-fiction · 82 past purchases",
    "Fitness Buff": "Sports & outdoors focus · high repeat buyer · 31 past purchases",
    "Home Chef": "Kitchen & home category · values quality · 56 past purchases",
    "Music Fan": "Audio gear & instruments · brand loyal · 28 past purchases",
}

# ── Example Data ──────────────────────────────────────────────────────────────

EXAMPLE_REVIEWS = [
    (
        "Sony WH-1000XM5",
        "Incredible noise cancellation. I use these daily on my commute and in the "
        "office. The sound quality is rich and detailed, bass is punchy without being "
        "overwhelming. Battery lasts about 28 hours. Only downside is they don't fold "
        "flat like the XM4s. Comfort is outstanding for long sessions.",
    ),
    (
        "MacBook Air M3",
        "Performance is stellar for development work — compiles are fast, Docker runs "
        "smoothly, and battery easily lasts a full workday. However, I'm frustrated by "
        "the single Thunderbolt port on the base model. The display is gorgeous but "
        "I miss having an SD card slot. For the price, Apple should include more ports.",
    ),
    (
        "Logitech G Pro X Superlight",
        "Bought this mouse expecting premium quality but the scroll wheel started "
        "double-clicking after 3 months. The sensor is excellent and it's incredibly "
        "light at 63g, but for $150 I expect better durability. Customer support was "
        "unhelpful. Returned for a refund.",
    ),
]

EXAMPLE_QUESTIONS = [
    ["Is the noise cancellation good enough for flights?", "B07NW1CG2S"],
    ["How's the battery life for all-day use?", "B07NW1CG2S"],
    ["Is this worth the price compared to competitors?", "0060177241"],
]

# ── Prometheus Helpers ────────────────────────────────────────────────────────


def _prom_instant(promql: str) -> list[dict]:
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


def _val(results: list[dict], default=None):
    try:
        return float(results[0]["value"][1])
    except Exception:
        return default


# ── HTML Builders ─────────────────────────────────────────────────────────────


def _latency_pill(ms: float) -> str:
    if ms < 300:
        cls, label = "latency-fast", "FAST"
    elif ms < 2000:
        cls, label = "latency-medium", "OK"
    else:
        cls, label = "latency-slow", "SLOW"
    return f'<span class="latency-pill {cls}">{ms:.0f}ms</span>'


def _infra_badges(badges: list[str]) -> str:
    return " ".join(f'<span class="infra-badge">{b}</span>' for b in badges)


def _score_bar(score: float, max_score: float = 1.0) -> str:
    pct = min(100, int(score / max_score * 100))
    return (
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<div style="flex:1;height:6px;background:var(--border);border-radius:3px;overflow:hidden;">'
        f'<div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#2563eb,var(--accent));border-radius:3px;"></div>'
        f'</div>'
        f'<span style="font-size:13px;color:var(--text-secondary);min-width:50px;">{pct}% match</span>'
        f'</div>'
    )


CATEGORY_MAP = {
    "B09": ("Electronics", "🎧"),
    "B08": ("Electronics", "🎧"),
    "B07": ("Electronics", "🎧"),
    "B01": ("Gadgets & Accessories", "🔌"),
    "B00": ("Home & Kitchen", "🏠"),
    "B0": ("Electronics", "🎧"),
    "19": ("Books — Non-Fiction", "📚"),
    "05": ("Books — Reference", "📖"),
    "03": ("Books — Fiction", "📕"),
    "0": ("Books", "📚"),
    "1": ("Books", "📚"),
}


def _guess_category(asin: str) -> tuple[str, str]:
    for prefix, (cat, icon) in CATEGORY_MAP.items():
        if asin.startswith(prefix):
            return cat, icon
    return "General", "📦"


def _status_strip_html() -> str:
    s = _stats_summary()

    def _svc_dot(name, url, health_path):
        try:
            r = SESSION.get(url.rsplit("/v1", 1)[0] + health_path, timeout=2)
            ok = r.status_code < 400
        except Exception:
            ok = False
        color = "#059669" if ok else "#dc2626"
        return f'<span style="color:{color};">●</span> {name}'

    rec_status = _svc_dot("Rec", RECOMMEND_URL, "/health")
    llm_status = _svc_dot("LLM", LLM_URL, "/health")
    rag_status = _svc_dot("RAG", RAG_URL, "/health")

    avg_str = f"{s['avg']:.0f}ms" if s['total'] else "—"
    return (
        f'<div class="status-strip">'
        f'<span>{rec_status}</span>'
        f'<span>{llm_status}</span>'
        f'<span>{rag_status}</span>'
        f'<span class="strip-divider">|</span>'
        f'<span>Requests: <strong>{s["total"]}</strong></span>'
        f'<span>Avg: <strong>{avg_str}</strong></span>'
        f'</div>'
    )


# ── Service Handlers ──────────────────────────────────────────────────────────


def get_recommendations(user_id: str, top_k: int = 10):
    t0 = time.time()
    try:
        response = SESSION.post(
            RECOMMEND_URL,
            json={"user_id": user_id, "top_k": int(top_k)},
            timeout=10,
        )
        ms = (time.time() - t0) * 1000
        _record("rec", ms)

        if response.ok:
            recs = response.json()["recommendations"]
            top_score = recs[0]["score"] if recs else 1.0

            pipeline = (
                f'<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:16px;">'
                f'<div class="pipeline-step done">Feast Lookup<br><small>User features from Redis · &lt;1ms</small></div>'
                f'<span class="pipeline-arrow">→</span>'
                f'<div class="pipeline-step done">Two-Tower Model<br><small>Score {len(recs):,} items · {ms:.0f}ms</small></div>'
                f'<span class="pipeline-arrow">→</span>'
                f'<div class="pipeline-step done">Top-K Ranking<br><small>torch.topk · {len(recs)} results</small></div>'
                f'</div>'
            )

            cards = ""
            for i, rec in enumerate(recs, 1):
                cat, icon = _guess_category(rec["item_id"])
                pct = int(rec["score"] / top_score * 100) if top_score else 0
                bar_color = "#2563eb" if pct > 60 else "#f59e0b" if pct > 30 else "#94a3b8"
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
                cards += (
                    f'<div style="display:flex;align-items:center;gap:12px;padding:10px 12px;'
                    f'border-bottom:1px solid var(--border);">'
                    f'<span style="font-size:18px;min-width:28px;text-align:center;">{medal}</span>'
                    f'<span style="font-size:22px;">{icon}</span>'
                    f'<div style="flex:1;min-width:0;">'
                    f'<div style="font-weight:500;color:var(--text-primary);font-size:14px;">{cat}</div>'
                    f'<div style="font-size:11px;color:var(--text-muted);font-family:monospace;">{rec["item_id"]}</div>'
                    f'</div>'
                    f'<div style="width:120px;text-align:right;">'
                    f'<div style="height:6px;background:var(--border);border-radius:3px;overflow:hidden;margin-bottom:4px;">'
                    f'<div style="width:{pct}%;height:100%;background:{bar_color};border-radius:3px;"></div></div>'
                    f'<span style="font-size:12px;font-weight:600;color:var(--text-secondary);">{pct}%</span>'
                    f'</div>'
                    f'</div>'
                )
            table = f'<div style="border:1px solid var(--border);border-radius:10px;overflow:hidden;">{cards}</div>'

            behind = (
                f'<details style="margin-top:14px;"><summary style="color:var(--text-muted);font-size:12px;cursor:pointer;">Behind the scenes</summary>'
                f'<div class="source-card" style="margin-top:8px;font-size:12px;line-height:1.8;color:var(--text-secondary);">'
                f'<strong>Feast online lookup:</strong> user features retrieved in &lt;1ms from Redis<br>'
                f'<strong>Model:</strong> Two-Tower Neural CF (1.6M items), trained with PyTorch DDP via Kubeflow TrainJob<br>'
                f'<strong>Scoring:</strong> Full-catalog dot product — all items scored, not sampled<br>'
                f'<strong>Serving:</strong> KServe InferenceService with pre-computed item embeddings'
                f'</div></details>'
            )
            return pipeline + table + behind, _status_strip_html()
        return f'<p style="color:var(--red-text);">Error {response.status_code}: {response.text[:200]}</p>', _status_strip_html()
    except requests.ConnectionError:
        return '<p style="color:var(--red-text);">Recommendation service unavailable. Check KServe deployment.</p>', _status_strip_html()
    except Exception as e:
        return f'<p style="color:var(--red-text);">Error: {e}</p>', _status_strip_html()


def summarize_review(product_name: str, review_text: str):
    t0 = time.time()
    try:
        prompt = (
            f"[INST] Summarize the following product review in 1-2 sentences. "
            f"Include the overall sentiment (positive/negative/neutral).\n\n"
            f"Product: {product_name}\n"
            f"Review: {review_text} [/INST]"
        )
        response = SESSION.post(
            LLM_URL,
            json={"model": "smartshop-llm", "prompt": prompt, "max_tokens": 256, "temperature": 0.3},
            timeout=45,
        )
        ms = (time.time() - t0) * 1000
        _record("llm", ms)

        if response.ok:
            data = response.json()
            text = data.get("choices", [{}])[0].get("text", "").strip()
            if not text:
                text = data.get("summary", "No response generated.")

            text_lower = text.lower()
            if "positive" in text_lower:
                sentiment, s_color = "POSITIVE", "#059669"
            elif "negative" in text_lower:
                sentiment, s_color = "NEGATIVE", "#dc2626"
            else:
                sentiment, s_color = "MIXED", "#d97706"

            tokens = len(text.split())
            tok_per_sec = tokens / (ms / 1000) if ms > 0 else 0

            pipeline = (
                f'<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:16px;">'
                f'<div class="pipeline-step done">Build Prompt<br><small>[INST] template · review text</small></div>'
                f'<span class="pipeline-arrow">→</span>'
                f'<div class="pipeline-step done">Mistral-7B + LoRA<br><small>vLLM inference · {ms:.0f}ms</small></div>'
                f'<span class="pipeline-arrow">→</span>'
                f'<div class="pipeline-step done">Sentiment + Summary<br><small>{tokens} tokens · {tok_per_sec:.0f} tok/s</small></div>'
                f'</div>'
            )

            result = (
                f'<div class="result-card">'
                f'<div style="margin-bottom:12px;">{text}</div>'
                f'<span style="background:{s_color}20;color:{s_color};padding:4px 12px;'
                f'border-radius:20px;font-weight:700;font-size:13px;">{sentiment}</span>'
                f'</div>'
            )

            behind = (
                f'<details style="margin-top:14px;"><summary style="color:var(--text-muted);font-size:12px;cursor:pointer;">Pipeline trace</summary>'
                f'<div class="source-card" style="margin-top:8px;font-size:12px;line-height:1.8;color:var(--text-secondary);">'
                f'<strong>Model:</strong> Mistral-7B-Instruct fine-tuned with QLoRA + FSDP via Kubeflow TrainJob<br>'
                f'<strong>Adapter:</strong> LoRA rank 16, hot-loaded at serving time (no base model retraining)<br>'
                f'<strong>Serving:</strong> vLLM on KServe with continuous batching + PagedAttention'
                f'</div></details>'
            )
            return pipeline + result + behind, _status_strip_html()
        return f'<p style="color:var(--red-text);">Error {response.status_code}: {response.text[:300]}</p>', _status_strip_html()
    except requests.ConnectionError:
        return '<p style="color:var(--red-text);">LLM service unavailable. Check KServe deployment.</p>', _status_strip_html()
    except Exception as e:
        return f'<p style="color:var(--red-text);">Error: {e}</p>', _status_strip_html()


def ask_question(question: str, product_id: str = ""):
    t0 = time.time()
    try:
        response = SESSION.post(
            RAG_URL,
            json={"question": question, "product_id": product_id, "top_k": 5},
            timeout=45,
        )
        ms = (time.time() - t0) * 1000
        _record("rag", ms)

        if response.ok:
            result = response.json()
            answer = result.get("answer", "")
            sources = result.get("sources", [])

            # Estimate timing breakdown (total - assumed embed ~50ms)
            embed_ms = 50
            llm_ms = ms * 0.7
            search_ms = ms - embed_ms - llm_ms

            pipeline = (
                f'<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:16px;">'
                f'<div class="pipeline-step done">Embed Query<br><small>SentenceTransformer 384d · {embed_ms:.0f}ms</small></div>'
                f'<span class="pipeline-arrow">→</span>'
                f'<div class="pipeline-step done">Feast Vector Store<br><small>Milvus similarity · {search_ms:.0f}ms · {len(sources)} docs</small></div>'
                f'<span class="pipeline-arrow">→</span>'
                f'<div class="pipeline-step done">Mistral-7B Generate<br><small>Contextual answer · {llm_ms:.0f}ms</small></div>'
                f'</div>'
            )

            answer_html = (
                f'<div class="result-card">'
                f'<div style="font-size:15px;line-height:1.6;">{answer}</div>'
                f'</div>'
            )

            sources_html = ""
            if sources:
                sources_html = '<div style="margin-top:12px;"><strong style="color:var(--text-secondary);">Retrieved Sources:</strong></div>'
                for src in sources[:5]:
                    rating = src.get("rating", "?")
                    title = src.get("title", "")
                    text = src.get("text", "")[:250]
                    stars = "★" * int(float(rating)) + "☆" * (5 - int(float(rating))) if str(rating).replace(".", "").isdigit() else ""
                    sources_html += (
                        f'<div class="source-card">'
                        f'<div style="color:#d97706;font-size:12px;">{stars} ({rating}/5)</div>'
                        f'<div style="font-size:13px;margin-top:4px;color:var(--text-primary);">{title}</div>'
                        f'<div style="color:var(--text-secondary);font-size:12px;margin-top:4px;">{text}...</div>'
                        f'</div>'
                    )

            badge = (
                f'<div style="margin-top:12px;">'
                f'{_latency_pill(ms)} '
                f'{_infra_badges(["Feast + Milvus", "SentenceTransformer 384d", "Mistral-7B"])}'
                f'</div>'
            )
            feast_note = (
                f'<div style="margin-top:10px;font-size:11px;color:var(--text-muted);font-style:italic;">'
                f'Retrieved from Feast vector store — same embeddings used in training (no train-serve skew)'
                f'</div>'
            )
            return pipeline + answer_html + sources_html + badge + feast_note, _status_strip_html()
        return f'<p style="color:var(--red-text);">Error {response.status_code}: {response.text[:300]}</p>', _status_strip_html()
    except requests.ConnectionError:
        return '<p style="color:var(--red-text);">RAG service unavailable. Check KServe deployment.</p>', _status_strip_html()
    except Exception as e:
        return f'<p style="color:var(--red-text);">Error: {e}</p>', _status_strip_html()


# ── Metrics Tab Functions ─────────────────────────────────────────────────────


def build_session_stats_html() -> str:
    s = _stats_summary()
    return (
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:16px 0;">'
        f'<div class="stat-card"><div class="stat-number">{s["total"]}</div><div class="stat-label">Total Requests</div></div>'
        f'<div class="stat-card"><div class="stat-number">{s["avg"]:.0f}<small>ms</small></div><div class="stat-label">Avg Latency</div></div>'
        f'<div class="stat-card"><div class="stat-number">{s["best"]:.0f}<small>ms</small></div><div class="stat-label">Best</div></div>'
        f'<div class="stat-card"><div class="stat-number">{s["worst"]:.0f}<small>ms</small></div><div class="stat-label">Worst</div></div>'
        f'</div>'
        f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">'
        f'<div class="source-card" style="border-left:3px solid #2563eb;"><strong>Rec</strong><br>{s["rec"]} requests</div>'
        f'<div class="source-card" style="border-left:3px solid #7c3aed;"><strong>LLM</strong><br>{s["llm"]} requests</div>'
        f'<div class="source-card" style="border-left:3px solid #059669;"><strong>RAG</strong><br>{s["rag"]} requests</div>'
        f'</div>'
    )


def build_service_health_html() -> str:
    services = [
        ("Recommendation", RECOMMEND_URL, "/health", "KServe · PyTorch", "#2563eb"),
        ("LLM Summarizer", LLM_URL, "/health", "vLLM · Mistral-7B", "#7c3aed"),
        ("RAG Q&A", RAG_URL, "/health", "Feast · Milvus · LLM", "#059669"),
    ]
    cards = ""
    for name, url, path, desc, color in services:
        try:
            base = url.rsplit("/v1", 1)[0]
            r = SESSION.get(base + path, timeout=3)
            status = "Healthy" if r.status_code < 400 else f"Error ({r.status_code})"
            dot_color = "#059669" if r.status_code < 400 else "#dc2626"
        except Exception:
            status = "Unreachable"
            dot_color = "#dc2626"
        cards += (
            f'<div class="source-card" style="border-left:3px solid {color};">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<strong>{name}</strong>'
            f'<span style="color:{dot_color};font-size:11px;">● {status}</span>'
            f'</div>'
            f'<div style="color:var(--text-muted);font-size:12px;margin-top:4px;">{desc}</div>'
            f'</div>'
        )
    return f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">{cards}</div>'


def build_grafana_html() -> str:
    if not GRAFANA_URL:
        return (
            '<div class="source-card" style="text-align:center;padding:24px;">'
            '<div style="color:var(--text-muted);">Set <code>GRAFANA_URL</code> to embed live dashboards</div>'
            '</div>'
        )
    base = GRAFANA_URL.rstrip("/")
    dash = f"{base}/d-solo/smartshop-inference/smartshop-inference-metrics?orgId=1&refresh=10s"
    row1 = [
        (f"{dash}&panelId=1", "Request Rate (req/s)"),
        (f"{dash}&panelId=6", "RAG Latency Breakdown (p95)"),
        (f"{dash}&panelId=8", "vLLM Token Throughput"),
    ]
    row2 = [
        (f"{dash}&panelId=2", "KV Cache Usage %"),
        (f"{dash}&panelId=5", "vLLM End-to-End Latency"),
        (f"{dash}&panelId=9", "Inter-Token Latency"),
    ]
    stat_row = [
        (f"{dash}&panelId=10", "Rec — Candidates Scored"),
        (f"{dash}&panelId=11", "RAG — Sources Retrieved"),
        (f"{dash}&panelId=12", "Avg Generation Tokens"),
        (f"{dash}&panelId=13", "Prefill vs Decode (p95)"),
    ]

    def _iframe_row(panels, height="200"):
        html = ""
        for url, title in panels:
            html += (
                f'<div style="flex:1;min-width:280px;">'
                f'<div style="color:var(--text-secondary);font-size:11px;margin-bottom:4px;">{title}</div>'
                f'<iframe src="{url}" width="100%" height="{height}" frameborder="0" '
                f'style="border-radius:8px;border:1px solid var(--border);background:var(--bg-card);"></iframe>'
                f'</div>'
            )
        return f'<div style="display:flex;gap:12px;flex-wrap:wrap;">{html}</div>'

    return (
        _iframe_row(row1, "210")
        + '<div style="margin-top:12px;"></div>'
        + _iframe_row(row2, "210")
        + '<div style="margin-top:12px;"></div>'
        + _iframe_row(stat_row, "130")
        + f'<div style="margin-top:14px;text-align:center;">'
        f'<a href="{base}/d/smartshop-inference" target="_blank" '
        f'style="color:var(--accent);font-size:13px;">Open Full Grafana Dashboard →</a>'
        f'</div>'
    )


def refresh_metrics():
    return build_session_stats_html(), build_service_health_html(), build_grafana_html(), _status_strip_html()


# ── Architecture Diagram ──────────────────────────────────────────────────────

ARCH_HTML = """
<div style="max-width:960px;margin:20px auto;">

  <!-- Layer 1: Data -->
  <div class="arch-section-label">1 · Data Ingestion</div>
  <div class="arch-row">
    <div class="arch-box" style="border-color:#f97316;">
      <div class="arch-title">Amazon Reviews 2023</div>
      <div class="arch-detail">
        571M reviews · 34 categories<br>
        <small>HuggingFace datasets → streaming download → MinIO S3 (Parquet)</small>
      </div>
    </div>
  </div>
  <div class="arch-arrow">▼</div>

  <!-- Layer 2: Feature Engineering -->
  <div class="arch-section-label">2 · Feature Engineering</div>
  <div class="arch-row" style="grid-template-columns:1fr 1fr;">
    <div class="arch-box" style="border-color:#64748b;">
      <div class="arch-title">Spark on Kubernetes</div>
      <div class="arch-detail">
        SparkApplication CRD · 4 executors<br>
        <small>User/item features: interaction counts, avg ratings, category preferences</small>
      </div>
    </div>
    <div class="arch-box" style="border-color:#f97316;">
      <div class="arch-title">Spark + RAPIDS</div>
      <div class="arch-detail">
        Same PySpark code, GPU-accelerated<br>
        <small>SentenceTransformer embeddings (384d) · 1M reviews → Milvus vectors</small>
      </div>
    </div>
  </div>
  <div class="arch-arrow">▼</div>

  <!-- Layer 3: Feast -->
  <div class="arch-section-label">3 · Feature Store</div>
  <div class="arch-row">
    <div class="arch-box" style="border-color:#059669;">
      <div class="arch-title">Feast</div>
      <div class="arch-detail" style="display:grid;grid-template-columns:1fr 1fr;gap:12px;text-align:left;">
        <div>
          <strong style="color:var(--text-primary);">Tabular features → Redis</strong><br>
          <small>user_features (1.6M users) · item_features (1.6M items)<br>
          <code style="font-size:10px;">feast materialize</code> from offline Parquet → online &lt;1ms lookups</small>
        </div>
        <div>
          <strong style="color:var(--text-primary);">Vector features → Milvus</strong><br>
          <small>review_embeddings (994K vectors, 384d)<br>
          <code style="font-size:10px;">retrieve_online_documents_v2</code> · cosine similarity search</small>
        </div>
      </div>
    </div>
  </div>
  <div class="arch-arrow">▼</div>

  <!-- Layer 4: Training -->
  <div class="arch-section-label">4 · Distributed Training (Kubeflow)</div>
  <div class="arch-row" style="grid-template-columns:1fr 1fr;">
    <div class="arch-box" style="border-color:#2563eb;">
      <div class="arch-title">Recommendation Model</div>
      <div class="arch-detail">
        Two-Tower Neural Collaborative Filtering<br>
        <small>PyTorch DDP · Kubeflow TrainJob · multi-node<br>
        User MLP + Item MLP → dot product scoring<br>
        1.6M items × 64d embeddings · trained on interaction pairs</small>
      </div>
    </div>
    <div class="arch-box" style="border-color:#7c3aed;">
      <div class="arch-title">LLM Fine-Tuning</div>
      <div class="arch-detail">
        Mistral-7B-Instruct · QLoRA + FSDP<br>
        <small>Kubeflow TrainJob · LoRA rank 16 · 4-bit quantized base<br>
        Instruction-tuned on review summarization task<br>
        Adapter merged at serving time (hot-loadable)</small>
      </div>
    </div>
  </div>
  <div class="arch-arrow">▼</div>

  <!-- Layer 5: Serving -->
  <div class="arch-section-label">5 · Model Serving (KServe)</div>
  <div class="arch-row" style="grid-template-columns:1fr 1fr 1fr;">
    <div class="arch-box" style="border-color:#2563eb;">
      <div class="arch-title">smartshop-rec</div>
      <div class="arch-detail">
        <small>PyTorch · RawDeployment<br>
        Pre-computed item embeddings<br>
        Full-catalog scoring per request<br>
        <code style="font-size:10px;">/v1/models/smartshop-rec:predict</code></small>
      </div>
    </div>
    <div class="arch-box" style="border-color:#7c3aed;">
      <div class="arch-title">smartshop-llm</div>
      <div class="arch-detail">
        <small>vLLM · continuous batching<br>
        PagedAttention · LoRA adapter<br>
        ~20 tok/s generation<br>
        <code style="font-size:10px;">/v1/completions</code></small>
      </div>
    </div>
    <div class="arch-box" style="border-color:#059669;">
      <div class="arch-title">smartshop-rag</div>
      <div class="arch-detail">
        <small>FastAPI · Feast + Milvus<br>
        SentenceTransformer embed → search<br>
        LLM-grounded answer generation<br>
        <code style="font-size:10px;">/v1/ask</code></small>
      </div>
    </div>
  </div>
  <div class="arch-arrow">▼</div>

  <!-- Layer 6: Observability -->
  <div class="arch-section-label">6 · Observability</div>
  <div class="arch-row" style="grid-template-columns:repeat(4,1fr);">
    <div class="arch-box-sm">
      <div class="arch-title-sm">Prometheus</div>
      <small>ServiceMonitor scraping<br>all 3 inference services</small>
    </div>
    <div class="arch-box-sm">
      <div class="arch-title-sm">Grafana</div>
      <small>4 dashboards: Inference,<br>Redis, Spark, GPU</small>
    </div>
    <div class="arch-box-sm">
      <div class="arch-title-sm">MLflow</div>
      <small>Training loss curves,<br>model registry</small>
    </div>
    <div class="arch-box-sm">
      <div class="arch-title-sm">Redis Exporter</div>
      <small>ops/sec, hit ratio,<br>memory, keys</small>
    </div>
  </div>
  <div class="arch-arrow">▼</div>

  <!-- Layer 7: UI -->
  <div class="arch-row">
    <div class="arch-box" style="border-color:var(--accent);background:var(--accent-bg);">
      <div class="arch-title">SmartShop Demo UI</div>
      <div class="arch-detail">
        Gradio · OpenShift Route · live pipeline traces · embedded Grafana dashboards
      </div>
    </div>
  </div>
</div>

<!-- Platform summary table -->
<div style="margin-top:28px;max-width:960px;margin-left:auto;margin-right:auto;">
  <div style="color:var(--text-secondary);font-size:12px;font-weight:600;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.05em;">Platform Stack</div>
  <table style="width:100%;border-collapse:collapse;font-size:13px;">
    <thead><tr style="border-bottom:2px solid var(--border);">
      <th style="padding:8px;text-align:left;color:var(--text-secondary);">Layer</th>
      <th style="padding:8px;text-align:left;color:var(--text-secondary);">Technology</th>
      <th style="padding:8px;text-align:left;color:var(--text-secondary);">Role</th>
    </tr></thead>
    <tbody style="color:var(--text-primary);">
      <tr style="border-bottom:1px solid var(--border);"><td style="padding:6px 8px;">Platform</td><td>Red Hat OpenShift AI</td><td>Kubernetes + ML platform layer</td></tr>
      <tr style="border-bottom:1px solid var(--border);"><td style="padding:6px 8px;">Data Processing</td><td>Spark on K8s + RAPIDS</td><td>Feature engineering, embedding generation</td></tr>
      <tr style="border-bottom:1px solid var(--border);"><td style="padding:6px 8px;">Feature Store</td><td>Feast (Redis + Milvus)</td><td>Online features &lt;1ms · vector retrieval</td></tr>
      <tr style="border-bottom:1px solid var(--border);"><td style="padding:6px 8px;">Training</td><td>Kubeflow TrainJob</td><td>Distributed PyTorch DDP + QLoRA FSDP</td></tr>
      <tr style="border-bottom:1px solid var(--border);"><td style="padding:6px 8px;">Serving</td><td>KServe (RawDeployment)</td><td>3 InferenceServices + auto-scaling</td></tr>
      <tr style="border-bottom:1px solid var(--border);"><td style="padding:6px 8px;">LLM Runtime</td><td>vLLM</td><td>Continuous batching, PagedAttention, LoRA</td></tr>
      <tr><td style="padding:6px 8px;">Observability</td><td>Prometheus + Grafana</td><td>End-to-end metrics across all layers</td></tr>
    </tbody>
  </table>
</div>
"""

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
/* Theme variables — light by default */
:root {
    --bg-primary: #f8fafc;
    --bg-card: #ffffff;
    --border: #e2e8f0;
    --text-primary: #1e293b;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --accent: #2563eb;
    --accent-bg: #eff6ff;
    --accent-text: #1d4ed8;
    --green-bg: #ecfdf5;
    --green-border: #059669;
    --green-text: #065f46;
    --yellow-bg: #fefce8;
    --yellow-text: #92400e;
    --red-bg: #fef2f2;
    --red-text: #991b1b;
    --logo-filter: none;
    --header-border: #e2e8f0;
}
:root[data-theme="dark"] {
    --bg-primary: #0f172a;
    --bg-card: #1e293b;
    --border: #334155;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent: #38bdf8;
    --accent-bg: #1e3a5f;
    --accent-text: #93c5fd;
    --green-bg: #064e3b;
    --green-border: #059669;
    --green-text: #6ee7b7;
    --yellow-bg: #78350f;
    --yellow-text: #fbbf24;
    --red-bg: #7f1d1d;
    --red-text: #fca5a5;
    --logo-filter: brightness(0) invert(1);
    --header-border: #1e293b;
}

/* Status strip */
.status-strip {
    display: flex; align-items: center; gap: 16px; padding: 8px 16px;
    background: var(--bg-primary); border: 1px solid var(--border); border-radius: 8px;
    font-size: 12px; color: var(--text-secondary); flex-wrap: wrap;
}
.strip-divider { color: var(--border); }

/* Latency pills */
.latency-pill { display:inline-block; padding:4px 12px; border-radius:20px; font-weight:600; font-size:13px; }
.latency-fast { background:var(--green-bg); color:var(--green-text); }
.latency-medium { background:var(--yellow-bg); color:var(--yellow-text); }
.latency-slow { background:var(--red-bg); color:var(--red-text); }

/* Infrastructure badges */
.infra-badge { background:var(--accent-bg); color:var(--accent-text); padding:3px 10px; border-radius:6px; font-size:11px; margin:2px; display:inline-block; }

/* Pipeline steps */
.pipeline-step { display:inline-flex; flex-direction:column; align-items:center; padding:8px 14px; border-radius:8px; border:1px solid var(--border); font-size:12px; color:var(--text-secondary); min-width:80px; text-align:center; }
.pipeline-step.done { border-color:var(--green-border); background:var(--green-bg); }
.pipeline-arrow { color:var(--text-muted); font-size:18px; }

/* Cards */
.result-card { background:var(--bg-card); border:1px solid var(--border); border-radius:10px; padding:16px; margin:8px 0; color:var(--text-primary); line-height:1.6; }
.source-card { background:var(--bg-card); border:1px solid var(--border); border-radius:8px; padding:12px; margin:6px 0; color:var(--text-secondary); }

/* Stats */
.stat-card { background:var(--bg-card); border:1px solid var(--border); border-radius:10px; padding:16px; text-align:center; }
.stat-number { font-size:32px; font-weight:700; color:var(--accent); }
.stat-number small { font-size:14px; color:var(--text-muted); }
.stat-label { font-size:12px; color:var(--text-muted); margin-top:4px; }

/* Architecture */
.arch-row { display:grid; grid-template-columns:1fr; gap:12px; margin:8px 0; }
.arch-box { background:var(--bg-card); border:2px solid var(--border); border-radius:10px; padding:16px; text-align:center; }
.arch-box-sm { background:var(--bg-card); border:1px solid var(--border); border-radius:8px; padding:10px; text-align:center; color:var(--text-secondary); }
.arch-title { font-weight:700; color:var(--text-primary); font-size:14px; margin-bottom:6px; }
.arch-title-sm { font-weight:600; color:var(--text-primary); font-size:12px; margin-bottom:4px; }
.arch-detail { color:var(--text-secondary); font-size:12px; line-height:1.5; }
.arch-arrow { text-align:center; color:var(--text-muted); font-size:16px; margin:4px 0; }
.arch-section-label { color:var(--text-muted); font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin:16px 0 4px; }

/* Global */
footer { display:none !important; }
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }

/* Gradio light mode overrides (default) */
:root:not([data-theme="dark"]) {
    --body-background-fill: #f8fafc !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: #e2e8f0 !important;
    --block-label-background-fill: #f1f5f9 !important;
    --block-label-text-color: #1e293b !important;
    --body-text-color: #1e293b !important;
    --body-text-color-subdued: #475569 !important;
    --input-background-fill: #ffffff !important;
    --input-border-color: #e2e8f0 !important;
    --button-secondary-background-fill: #f1f5f9 !important;
    --button-secondary-border-color: #e2e8f0 !important;
    --button-secondary-text-color: #1e293b !important;
    --neutral-50: #f8fafc !important;
    --neutral-100: #f1f5f9 !important;
    --neutral-200: #e2e8f0 !important;
    --neutral-300: #cbd5e1 !important;
    --neutral-400: #94a3b8 !important;
    --neutral-500: #64748b !important;
    --neutral-600: #475569 !important;
    --neutral-700: #334155 !important;
    --neutral-800: #1e293b !important;
    --neutral-900: #0f172a !important;
    --neutral-950: #020617 !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f8fafc !important;
    --border-color-primary: #e2e8f0 !important;
    --border-color-accent: #2563eb !important;
    --color-accent-soft: #eff6ff !important;
    --tab-nav-background-color: #f8fafc !important;
    --panel-background-fill: #ffffff !important;
    --shadow-drop: 0 1px 3px rgba(0,0,0,0.08) !important;
    --shadow-drop-lg: 0 4px 6px rgba(0,0,0,0.06) !important;
}
"""

# ── UI ────────────────────────────────────────────────────────────────────────

THEME = gr.themes.Default(primary_hue="blue", secondary_hue="slate", neutral_hue="slate")

THEME_JS = """
() => {
  function setLight() {
    document.documentElement.setAttribute('data-theme', 'light');
    document.documentElement.classList.remove('dark');
    document.body.classList.remove('dark');
    document.querySelectorAll('.gradio-container').forEach(el => el.classList.remove('dark'));
    const btn = document.getElementById('theme-toggle');
    if (btn) btn.textContent = '☀️';
  }
  function setDark() {
    document.documentElement.setAttribute('data-theme', 'dark');
    document.documentElement.classList.add('dark');
    document.body.classList.add('dark');
    document.querySelectorAll('.gradio-container').forEach(el => el.classList.add('dark'));
    const btn = document.getElementById('theme-toggle');
    if (btn) btn.textContent = '🌙';
  }
  function toggleTheme() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    if (isDark) { setLight(); } else { setDark(); }
  }
  window.toggleTheme = toggleTheme;
  // Default to light on load
  setLight();
  setTimeout(() => {
    const btn = document.getElementById('theme-toggle');
    if (btn) btn.addEventListener('click', toggleTheme);
  }, 500);
}
"""

with gr.Blocks(title="SmartShop AI — Production ML at Scale", theme=THEME, css=CSS, js=THEME_JS) as demo:

    # Header
    gr.HTML("""
    <div style="padding:14px 0;border-bottom:1px solid var(--header-border);margin-bottom:8px;">
      <div style="display:flex;align-items:center;gap:14px;">
        <img src="https://www.redhat.com/rhdc/managed-files/Logo-Red_Hat-A-Standard-RGB.svg"
             height="28" style="height:28px;width:auto;max-width:120px;filter:var(--logo-filter);"
             onerror="this.outerHTML='<div style=\\'background:#ee0000;color:white;font-weight:700;padding:6px 12px;border-radius:6px;font-size:13px;\\'>Red Hat</div>'"/>
        <span style="color:var(--text-primary);font-size:18px;font-weight:600;">SmartShop AI</span>
        <div style="flex:1;"></div>
        <button id="theme-toggle" style="background:var(--bg-card);border:1px solid var(--border);border-radius:6px;padding:5px 10px;cursor:pointer;font-size:16px;line-height:1;" title="Toggle light/dark mode">☀️</button>
      </div>
      <div style="color:var(--text-muted);font-size:13px;margin-top:6px;">
        Hyper-Personalized Customer Intelligence — Powered by Kubeflow, Feast, KServe &amp; Spark on <strong style="color:#ee0000;">Red Hat OpenShift AI</strong>
      </div>
    </div>
    """)

    # Persistent status strip
    status_strip = gr.HTML(_status_strip_html())

    with gr.Tabs():

        # ── Tab 1: Product Recommendations ────────────────────────────────
        with gr.Tab("Product Recommendations"):
            gr.HTML(
                '<div style="color:var(--text-secondary);font-size:13px;margin:8px 0;line-height:1.6;">'
                'A returning customer browses the store. The platform fetches their profile from '
                '<strong style="color:var(--text-primary);">Feast</strong> (&lt;1ms via Redis) and runs the '
                '<strong style="color:var(--text-primary);">Two-Tower model</strong> to surface personalized picks.'
                '</div>'
            )
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("**Select a customer profile** or enter a custom User ID:")
                    persona_radio = gr.Radio(
                        choices=[p[0] for p in PERSONAS],
                        label="Customer Profile",
                        value=PERSONAS[0][0],
                    )
                    user_input = gr.Textbox(
                        label="Custom User ID (optional)",
                        placeholder="Leave empty to use persona above",
                        lines=1,
                    )
                    top_k_input = gr.Slider(minimum=1, maximum=30, value=10, step=1, label="Top K")
                    rec_btn = gr.Button("Get Recommendations", variant="primary", size="lg")
                with gr.Column(scale=3):
                    rec_output = gr.HTML('<div class="result-card" style="text-align:center;color:var(--text-muted);">Click "Get Recommendations" to see results</div>')

            def _resolve_rec(persona, custom_id, top_k):
                uid = custom_id.strip() if custom_id.strip() else dict(PERSONAS).get(persona, PERSONAS[0][1])
                return get_recommendations(uid, top_k)

            rec_btn.click(_resolve_rec, inputs=[persona_radio, user_input, top_k_input], outputs=[rec_output, status_strip])

        # ── Tab 2: Review Intelligence ──────────────────────────────────────
        with gr.Tab("Review Intelligence"):
            gr.HTML(
                '<div style="color:var(--text-secondary);font-size:13px;margin:8px 0;line-height:1.6;">'
                'Thousands of reviews arrive daily. The fine-tuned LLM extracts sentiment, key themes, '
                'and actionable insights — no generic summaries, but '
                '<strong style="color:var(--text-primary);">product-aware analysis</strong> '
                'from a domain-specialized model.'
                '</div>'
            )
            with gr.Row():
                with gr.Column(scale=2):
                    product_input = gr.Textbox(label="Product Name", placeholder="e.g. Sony WH-1000XM5")
                    review_input = gr.Textbox(label="Review Text", placeholder="Paste a product review...", lines=6)
                    summary_btn = gr.Button("Summarize Review", variant="primary", size="lg")
                    gr.Examples(
                        examples=[[r[0], r[1]] for r in EXAMPLE_REVIEWS],
                        inputs=[product_input, review_input],
                        label="Try an example:",
                    )
                with gr.Column(scale=3):
                    llm_output = gr.HTML('<div class="result-card" style="text-align:center;color:var(--text-muted);">Click "Summarize Review" to see results</div>')

            summary_btn.click(summarize_review, inputs=[product_input, review_input], outputs=[llm_output, status_strip])

        # ── Tab 3: Product Q&A (RAG) ─────────────────────────────────────
        with gr.Tab("Product Q&A"):
            gr.HTML(
                '<div style="color:var(--text-secondary);font-size:13px;margin:8px 0;line-height:1.6;">'
                'A shopper asks a product question. The system embeds the query, searches review embeddings in '
                '<strong style="color:var(--text-primary);">Feast\'s Milvus vector store</strong>, and generates a '
                'grounded answer — no hallucination, only evidence from real reviews.'
                '</div>'
            )
            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(label="Question", placeholder="Ask about any product...")
                    product_id_input = gr.Textbox(label="Product ID (optional ASIN)", placeholder="Leave empty for broad search")
                    qa_btn = gr.Button("Ask", variant="primary", size="lg")
                    gr.Examples(
                        examples=EXAMPLE_QUESTIONS,
                        inputs=[question_input, product_id_input],
                        label="Try an example:",
                    )
                with gr.Column(scale=3):
                    rag_output = gr.HTML('<div class="result-card" style="text-align:center;color:var(--text-muted);">Ask a question to see the RAG pipeline in action</div>')

            qa_btn.click(ask_question, inputs=[question_input, product_id_input], outputs=[rag_output, status_strip])

        # ── Tab 4: Observability ──────────────────────────────────────────
        with gr.Tab("Observability"):
            gr.HTML(
                '<div style="color:var(--text-secondary);font-size:13px;margin:8px 0;line-height:1.6;">'
                '<strong style="color:var(--text-primary);">Live observability across the full ML lifecycle</strong> — '
                'Prometheus scraping all inference services · Grafana dashboards · auto-refreshing every 10s'
                '</div>'
            )

            gr.Markdown("### Service Health")
            health_html = gr.HTML(build_service_health_html())

            gr.Markdown("### Demo Session Stats")
            session_stats_html = gr.HTML(build_session_stats_html())

            gr.Markdown("### Grafana Dashboards")
            grafana_html = gr.HTML(build_grafana_html())

            refresh_btn = gr.Button("Refresh Metrics", variant="secondary", size="sm")
            refresh_btn.click(refresh_metrics, outputs=[session_stats_html, health_html, grafana_html, status_strip])

            timer = gr.Timer(10)
            timer.tick(refresh_metrics, outputs=[session_stats_html, health_html, grafana_html, status_strip])

        # ── Tab 5: Architecture ───────────────────────────────────────────
        with gr.Tab("Architecture"):
            gr.HTML(
                '<div style="color:var(--text-secondary);font-size:13px;margin:8px 0;line-height:1.6;">'
                '<strong style="color:var(--text-primary);">End-to-end production ML</strong> — from 233M Amazon reviews '
                'through Spark preprocessing, Feast feature management, distributed training, to real-time serving on '
                '<strong style="color:var(--text-primary);">Red Hat OpenShift AI</strong>'
                '</div>'
            )
            gr.HTML(ARCH_HTML)

    # Footer
    gr.HTML("""
    <div style="text-align:center;padding:12px;margin-top:16px;border-top:1px solid var(--border);color:var(--text-muted);font-size:11px;">
      Red Hat Summit 2026 · SmartShop AI: Production ML at Scale
    </div>
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
