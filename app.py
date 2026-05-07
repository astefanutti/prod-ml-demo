"""SmartShop AI — Red Hat Summit 2026 Demo UI.

Production ML at Scale: PyTorch DDP, LoRA/FSDP, Spark, Kubeflow,
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
    ("Tech Enthusiast", "AHSV5AUFONH7QMMUPF7M6FUJRJ6Q_1"),
    ("Book Lover", "AGSP6LSQK32SQEJO3YVVNACPWMSQ"),
    ("Home Enthusiast", "AEIIRIHLIYKQGI7ZOCIJTRDF5NPQ"),
    ("Photography & Audio Fan", "AE224EFKNL3XBZDNWNYDVRMJKG2Q"),
    ("DIY & Outdoors Pro", "AE2225K3KY4D3KKSN6I2AHBVR4QQ"),
]

PERSONA_CONTEXT = {
    "Tech Enthusiast": "Tech & Computing · Electronics, Computers, Phones",
    "Book Lover": "Books & Media · Fiction, Non-Fiction, Kindle",
    "Home Enthusiast": "Home & Living · Kitchen, Garden, Decor",
    "Photography & Audio Fan": "Photography & Audio · Cameras, Headphones, Car Audio",
    "DIY & Outdoors Pro": "DIY & Outdoors · Tools, Sports, Automotive",
}

RECOMMEND_PROFILE_URL = os.environ.get(
    "RECOMMEND_PROFILE_URL",
    RECOMMEND_URL.replace(":predict", ":user-profile"),
)

# ── Example Data ──────────────────────────────────────────────────────────────

EXAMPLE_ASINS = [
    ["B07SM135LS"],   # Tech & Computing — Digi-Tatoo MacBook Decal
    ["0316769177"],   # Books & Media — The Tipping Point
    ["B07848ZT9T"],   # Photography & Audio — ANNKE 4K Security DVR
    ["B08GG5KD6F"],   # DIY & Outdoors — Fitbit Charge 4 Bands
    ["B00XVMYGC2"],   # Home & Living — Cordless Phone Battery
]

EXAMPLE_QUESTIONS = [
    ["Does this laptop skin leave residue when removed?", "B07SM135LS"],
    ["Is this book good for business beginners?", "0316769177"],
    ["Is this security camera good for outdoor use?", "B07848ZT9T"],
    ["Is this band comfortable for daily wear?", "B08GG5KD6F"],
    ["How long does this battery last compared to OEM?", "B00XVMYGC2"],
]

EXAMPLE_CLASSIFY = [
    ["Absolutely love this! Build quality is excellent and battery lasts all day. Best purchase I've made this year.", "B07SM135LS"],
    ["The book started strong but the second half felt repetitive. Some good insights but not worth the price.", "0316769177"],
    ["Terrible experience. Stopped working after 2 weeks, customer support was unresponsive. Avoid.", "B07848ZT9T"],
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
    "B00": ("Home & Living", "🏠"),
    "B0": ("Tech & Computing", "💻"),
    "19": ("Books & Media", "📚"),
    "05": ("Books & Media", "📖"),
    "03": ("Books & Media", "📕"),
    "0": ("Books & Media", "📚"),
    "1": ("Books & Media", "📚"),
}

SUPER_CATEGORY_ICONS = {
    "Tech & Computing": "💻",
    "Books & Media": "📚",
    "Home & Living": "🏠",
    "Photography & Audio": "📷",
    "DIY & Outdoors": "🔧",
}


def _guess_category(asin: str) -> tuple[str, str]:
    for prefix, (cat, icon) in CATEGORY_MAP.items():
        if asin.startswith(prefix):
            return cat, icon
    return "General", "📦"


def _star_rating_html(rating: float) -> str:
    if not rating:
        return ""
    full = int(rating)
    half = 1 if rating - full >= 0.25 else 0
    empty = 5 - full - half
    stars = "★" * full + ("½" if half else "") + "☆" * empty
    return (
        f'<span style="color:#f59e0b;font-size:13px;letter-spacing:-1px;">{stars}</span>'
        f'<span style="font-size:12px;color:var(--text-muted);margin-left:4px;">{rating:.1f}</span>'
    )


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


def get_recommendations(user_id: str, top_k: int = 10, preferred_category: str = ""):
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
            resp_json = response.json()
            recs = resp_json["recommendations"]
            num_scored = resp_json.get("num_scored", len(recs))

            if preferred_category:
                pref_lower = preferred_category.lower()
                for r in recs:
                    cat = (r.get("category") or "").lower()
                    if pref_lower in cat or cat in pref_lower:
                        r["_boosted"] = True
                        r["score"] = min(r["score"] * 1.15, 1.0)
                recs.sort(key=lambda x: x["score"], reverse=True)
            top_score = recs[0]["score"] if recs else 1.0

            pipeline = (
                f'<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:16px;">'
                f'<div class="pipeline-step done">Feast Lookup<br><small>User + item features from Redis · &lt;1ms</small></div>'
                f'<span class="pipeline-arrow">→</span>'
                f'<div class="pipeline-step done">Two-Tower Model<br><small>Score {num_scored:,} items · {ms:.0f}ms</small></div>'
                f'<span class="pipeline-arrow">→</span>'
                f'<div class="pipeline-step done">Metadata Filter + Top-K<br><small>{len(recs)} enriched results</small></div>'
                f'</div>'
            )

            cards = ""
            for i, rec in enumerate(recs, 1):
                title = rec.get("title") or ""
                brand = rec.get("brand") or ""
                category = rec.get("category") or ""
                avg_rating = rec.get("avg_rating")
                review_count = rec.get("review_count")
                price = rec.get("price")

                if title:
                    icon = SUPER_CATEGORY_ICONS.get(category, "📦")
                    if icon == "📦":
                        icon = {"Electronics": "💻", "Computers": "💻", "Books": "📚",
                                "Amazon Home": "🏠", "Home": "🏠", "Camera": "📷",
                                "Cell Phones": "📱", "Tools": "🔧", "Sports": "🔧",
                                "Automotive": "🔧"}.get(
                            (category or "").split(" & ")[0].split()[0], "📦"
                        )
                else:
                    title = f"ASIN: {rec['item_id']}"
                    _, icon = _guess_category(rec["item_id"])

                display_title = title[:70] + "…" if len(title) > 70 else title

                meta_parts = []
                if brand:
                    meta_parts.append(brand)
                if category and category not in ("All Electronics",):
                    meta_parts.append(f'<span style="background:var(--border);padding:1px 6px;border-radius:8px;font-size:10px;">{category}</span>')

                stars_html = _star_rating_html(avg_rating) if avg_rating else ""
                review_count_html = f'<span style="font-size:11px;color:var(--text-muted);">({review_count:,})</span>' if review_count else ""
                price_html = f'<span style="font-size:13px;font-weight:600;color:var(--text-primary);">${price:.2f}</span>' if price else ""

                pct = int(rec["score"] * 100)
                bar_color = "#2563eb" if pct > 60 else "#f59e0b" if pct > 30 else "#94a3b8"
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"

                asin = rec["item_id"]
                boosted = rec.get("_boosted")
                why = f'{"⚡ " if boosted else ""}Category: {category}' if category else ''
                cards += (
                    f'<div style="display:flex;align-items:center;gap:12px;padding:10px 12px;'
                    f'border-bottom:1px solid var(--border);">'
                    f'<span style="font-size:18px;min-width:28px;text-align:center;">{medal}</span>'
                    f'<span style="font-size:22px;">{icon}</span>'
                    f'<div style="flex:1;min-width:0;">'
                    f'<div style="font-weight:500;color:var(--text-primary);font-size:14px;">{display_title}</div>'
                    f'<div style="font-size:11px;color:var(--text-muted);margin-top:3px;display:flex;gap:6px;align-items:center;flex-wrap:wrap;">'
                    f'{" · ".join(meta_parts)}'
                    f'</div>'
                    f'<div style="font-size:11px;margin-top:2px;display:flex;gap:8px;align-items:center;">'
                    f'{stars_html} {review_count_html}'
                    f'{"" if not price_html else "  ·  " + price_html}'
                    f'</div>'
                    f'<div style="margin-top:4px;display:flex;gap:6px;">'
                    f'<button onclick="navigateToTab(1, \'{asin}\')" class="crosstab-btn">Analyze Reviews</button>'
                    f'<button onclick="navigateToTab(2, \'{asin}\')" class="crosstab-btn">Ask Q&A</button>'
                    f'</div>'
                    f'</div>'
                    f'<div style="width:80px;text-align:right;">'
                    f'<span style="font-size:13px;font-weight:600;color:{bar_color};">{pct}%</span>'
                    f'<div style="font-size:9px;color:var(--text-muted);margin-top:2px;">{why}</div>'
                    f'</div>'
                    f'</div>'
                )
            table = f'<div style="border:1px solid var(--border);border-radius:10px;overflow:hidden;">{cards}</div>'

            behind = (
                f'<details style="margin-top:14px;"><summary style="color:var(--text-muted);font-size:12px;cursor:pointer;">Behind the scenes</summary>'
                f'<div class="source-card" style="margin-top:8px;font-size:12px;line-height:1.8;color:var(--text-secondary);">'
                f'<strong>Feast online lookup:</strong> user + item features from Redis (&lt;1ms)<br>'
                f'<strong>Model:</strong> Two-Tower Neural CF ({num_scored:,} items) with category-aware embeddings (5 super-categories), '
                f'trained with HF Trainer + DDP via Kubeflow TrainJob (2:1 negative sampling, eval each epoch, early stopping)<br>'
                f'<strong>Scoring:</strong> Full-catalog dot product — all items scored, top-100 over-fetched, filtered to items with metadata<br>'
                f'<strong>Serving:</strong> KServe InferenceService with pre-computed item embeddings + Feast enrichment'
                f'</div></details>'
            )
            return pipeline + table + behind, _status_strip_html()
        return f'<p style="color:var(--red-text);">Error {response.status_code}: {response.text[:200]}</p>', _status_strip_html()
    except requests.ConnectionError:
        return '<p style="color:var(--red-text);">Recommendation service unavailable. Check KServe deployment.</p>', _status_strip_html()
    except Exception as e:
        return f'<p style="color:var(--red-text);">Error: {e}</p>', _status_strip_html()


def _fetch_product_meta(product_id: str) -> dict:
    """Fetch product metadata dict from the rec server's Feast enrichment."""
    if not product_id.strip():
        return {}
    try:
        resp = SESSION.post(
            RECOMMEND_URL,
            json={"user_id": "dummy", "candidate_items": [product_id.strip()], "top_k": 1},
            timeout=5,
        )
        if resp.ok:
            recs = resp.json().get("recommendations", [])
            if recs:
                return recs[0]
    except Exception:
        pass
    return {}


def _fetch_persona_affinity(product_id: str) -> list[dict]:
    """Score this product against each persona using the Two-Tower model."""
    results = []
    for name, uid in PERSONAS:
        try:
            resp = SESSION.post(
                RECOMMEND_URL,
                json={"user_id": uid, "candidate_items": [product_id.strip()], "top_k": 1},
                timeout=5,
            )
            if resp.ok:
                recs = resp.json().get("recommendations", [])
                score = recs[0]["score"] if recs else 0
            else:
                score = 0
        except Exception:
            score = 0
        ctx = PERSONA_CONTEXT.get(name, "")
        cat_key = ctx.split(" · ")[0] if " · " in ctx else ""
        results.append({"name": name, "score": score, "icon": SUPER_CATEGORY_ICONS.get(cat_key, "📦")})
    return results


def _fetch_related_products(product_id: str, top_k: int = 5) -> list[dict]:
    """Fetch top-scored items in the same category as the given product."""
    try:
        meta = _fetch_product_meta(product_id)
        category = meta.get("category", "")
        resp = SESSION.post(
            RECOMMEND_URL,
            json={"user_id": "dummy", "top_k": top_k + 10},
            timeout=8,
        )
        if resp.ok:
            recs = resp.json().get("recommendations", [])
            same_cat = [
                r for r in recs
                if r.get("item_id") != product_id.strip()
                and r.get("title")
                and (not category or category.lower() in (r.get("category") or "").lower()
                     or (r.get("category") or "").lower() in category.lower())
            ]
            if same_cat:
                return same_cat[:top_k]
            return [r for r in recs if r.get("item_id") != product_id.strip() and r.get("title")][:top_k]
    except Exception:
        pass
    return []


def _fetch_reviews(product_id: str, product_title: str = "", top_k: int = 20) -> list[dict]:
    """Retrieve reviews via dual-search: product-relevant + negative sentiment."""
    seen_texts: set[str] = set()
    all_sources: list[dict] = []

    def _dedup_add(sources):
        for s in sources:
            if isinstance(s, str):
                s = {"text": s}
            if not isinstance(s, dict):
                continue
            key = s.get("text", "")[:80]
            if key and key not in seen_texts:
                seen_texts.add(key)
                all_sources.append(s)

    query = product_title if product_title else product_id.strip()

    try:
        resp = SESSION.post(
            RAG_URL,
            json={"question": f"{query} review", "product_id": product_id.strip(), "top_k": top_k},
            timeout=30,
        )
        if resp.ok:
            _dedup_add(resp.json().get("sources", []))
    except Exception:
        pass

    try:
        resp = SESSION.post(
            RAG_URL,
            json={
                "question": f"{query} disappointing problems issues not worth",
                "product_id": product_id.strip(),
                "top_k": top_k // 2,
            },
            timeout=20,
        )
        if resp.ok:
            _dedup_add(resp.json().get("sources", []))
    except Exception:
        pass

    return all_sources


def _llm_summarize_reviews(product_meta: dict, reviews: list[dict]) -> str:
    """Generate a TL;DR summary using the fine-tuned LoRA adapter.

    Uses the exact prompt format the LoRA was trained on so the adapter
    actually adds value over the base model.
    """
    title = product_meta.get("title", "Unknown")
    capped = reviews[:8]
    combined = " | ".join(
        (r.get("text", "") if isinstance(r, dict) else str(r))[:200]
        for r in capped
    )
    prompt = (
        "[INST] Summarize the following product review in 1-2 sentences. "
        "Include the overall sentiment (positive/negative/neutral).\n\n"
        f"Product: {title}\n"
        f"Review: {combined[:1500]}\n[/INST]"
    )
    try:
        resp = SESSION.post(
            LLM_URL,
            json={"model": "smartshop-qa", "prompt": prompt,
                  "max_tokens": 200, "temperature": 0.3},
            timeout=30,
        )
        if resp.ok:
            return resp.json().get("choices", [{}])[0].get("text", "").strip()
    except Exception:
        pass
    return ""


def _llm_product_intelligence(product_meta: dict, reviews: list[dict]) -> dict:
    """Call LLM for per-review sentiment classification + structured product analysis."""
    capped = reviews[:12]
    reviews_block = "\n\n".join(
        f"Review {i+1}: {(r.get('text', '') if isinstance(r, dict) else str(r))[:250]}"
        for i, r in enumerate(capped)
    )
    title = product_meta.get("title", "Unknown")
    category = product_meta.get("category", "")
    price = product_meta.get("price")
    price_str = f"${price:.2f}" if price else "N/A"
    avg_rating = product_meta.get("avg_rating", "N/A")

    prompt = (
        '[INST] You are a product intelligence analyst. Analyze each review for '
        f'"{title}" (Category: {category}, Price: {price_str}, Avg Rating: {avg_rating}).\n\n'
        'Respond ONLY with valid JSON (no markdown, no extra text):\n'
        '{"per_review": [{"sentiment": "positive|neutral|negative"}, ...],\n'
        ' "verdict": "<conditional buy/skip recommendation>",\n'
        ' "pros": ["pro1", "pro2", "pro3"], "cons": ["con1", "con2", "con3"],\n'
        ' "best_for": ["audience1", "audience2"],\n'
        ' "not_for": ["audience1", "audience2"],\n'
        ' "aspects": {"aspect_name": "positive|neutral|negative|mixed", ...},\n'
        ' "themes": ["theme1", "theme2", "theme3"]}\n\n'
        f'Reviews:\n{reviews_block}\n[/INST]'
    )
    for model_name in ("smartshop-qa", "smartshop-llm"):
        try:
            resp = SESSION.post(
                LLM_URL,
                json={"model": model_name, "prompt": prompt, "max_tokens": 600, "temperature": 0.2},
                timeout=60,
            )
            if resp.ok:
                raw = resp.json().get("choices", [{}])[0].get("text", "").strip()
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    result = json.loads(raw[start:end])
                    result["_model_used"] = model_name
                    return result
        except Exception:
            continue
    return {}


def _aggregate_sentiment(analysis: dict) -> dict:
    """Aggregate per-review LLM sentiment into counts and percentages."""
    per_review = analysis.get("per_review", [])
    def _sent(r):
        if isinstance(r, dict):
            return r.get("sentiment", "")
        return str(r)
    pos = sum(1 for r in per_review if _sent(r) == "positive")
    neu = sum(1 for r in per_review if _sent(r) == "neutral")
    neg = sum(1 for r in per_review if _sent(r) == "negative")
    total = pos + neu + neg or 1
    return {
        "positive": pos, "neutral": neu, "negative": neg, "total": total,
        "pos_pct": int(pos / total * 100),
        "neu_pct": int(neu / total * 100),
        "neg_pct": int(neg / total * 100),
    }


def _classify_single_review(review_text: str, product_asin: str,
                             aggregate_state: dict | None) -> tuple[str, str]:
    """Classify a single review using the fine-tuned LLM. Works standalone."""
    t0 = time.time()
    product_context = ""
    if aggregate_state and aggregate_state.get("title"):
        product_context = (
            f'Product: {aggregate_state["title"]} ({aggregate_state.get("category", "")})\n'
            f'Aggregate sentiment: {aggregate_state.get("pos_pct", "?")}% positive\n'
        )
    elif product_asin and product_asin.strip():
        meta = _fetch_product_meta(product_asin.strip())
        if meta.get("title"):
            product_context = f'Product: {meta["title"]} ({meta.get("category", "")})\n'

    prompt = (
        '[INST] You are a review analyst. Classify this review and respond ONLY with valid JSON.\n\n'
        f'{product_context}'
        f'Review: {review_text[:500]}\n\n'
        'Required JSON: {{"sentiment": "positive|mixed|negative", '
        '"aspects": {{"aspect": "positive|negative", ...}}, '
        '"context_note": "brief note on how this compares to typical reviews"}}\n[/INST]'
    )
    model_used = "smartshop-llm"
    try:
        parsed = {}
        for model_name in ("smartshop-qa", "smartshop-llm"):
            try:
                resp = SESSION.post(
                    LLM_URL,
                    json={"model": model_name, "prompt": prompt, "max_tokens": 300, "temperature": 0.2},
                    timeout=30,
                )
                if resp.ok:
                    raw = resp.json().get("choices", [{}])[0].get("text", "").strip()
                    start = raw.find("{")
                    end = raw.rfind("}") + 1
                    if start >= 0 and end > start:
                        parsed = json.loads(raw[start:end])
                        model_used = model_name
                        break
            except Exception:
                continue

        ms = (time.time() - t0) * 1000
        _record("llm", ms)

        if True:
            sentiment = parsed.get("sentiment", "unknown")
            s_colors = {"positive": "#059669", "mixed": "#d97706", "negative": "#dc2626"}
            s_color = s_colors.get(sentiment, "#94a3b8")
            s_emoji = {"positive": "👍", "mixed": "🤔", "negative": "👎"}.get(sentiment, "❓")

            model_label = "Fine-tuned LoRA" if model_used == "smartshop-qa" else "Base Mistral-7B"

            header = (
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
                f'<span style="font-size:32px;">{s_emoji}</span>'
                f'<div>'
                f'<span style="background:{s_color}20;color:{s_color};padding:4px 14px;'
                f'border-radius:20px;font-weight:700;font-size:15px;">{sentiment.upper()}</span>'
                f'<div style="font-size:10px;color:var(--text-muted);margin-top:4px;">'
                f'{model_label} · {ms:.0f}ms</div>'
                f'</div></div>'
            )

            raw_aspects = parsed.get("aspects", {})
            if isinstance(raw_aspects, list):
                normalized = {}
                for a in raw_aspects:
                    if isinstance(a, dict):
                        name = a.get("aspect", a.get("name", ""))
                        val = a.get("sentiment", a.get("value", "mixed"))
                        if name:
                            normalized[name] = val
                    elif isinstance(a, str):
                        normalized[a] = "mixed"
                raw_aspects = normalized
            elif not isinstance(raw_aspects, dict):
                raw_aspects = {}

            _NEG_WORDS = {"bad", "poor", "terrible", "awful", "worst", "broken", "slow", "dim", "cheap", "flimsy", "fails", "disappointing"}
            _POS_WORDS = {"great", "excellent", "good", "amazing", "fantastic", "fast", "solid", "love", "perfect", "best", "comfortable", "smooth"}

            def _infer_sentiment(val: str) -> str:
                if val in ("positive", "negative", "mixed", "neutral"):
                    return val
                vl = val.lower()
                if any(w in vl for w in _NEG_WORDS):
                    return "negative"
                if any(w in vl for w in _POS_WORDS):
                    return "positive"
                return "mixed"

            aspects_html = ""
            if raw_aspects:
                tags = ""
                for aspect, val in list(raw_aspects.items())[:6]:
                    sent = _infer_sentiment(str(val))
                    a_color = s_colors.get(sent, "#94a3b8")
                    icon = "✓" if sent == "positive" else "✗" if sent == "negative" else "◐"
                    display = f"{aspect}: {val}" if val not in ("positive", "negative", "mixed", "neutral") else aspect
                    tags += (
                        f'<span style="display:inline-block;background:{a_color}10;color:{a_color};'
                        f'border:1px solid {a_color}30;padding:3px 10px;border-radius:14px;'
                        f'font-size:12px;margin:2px 4px 2px 0;">{icon} {display.replace("_", " ")}</span>'
                    )
                aspects_html = (
                    f'<div style="margin-top:10px;">'
                    f'<div style="font-weight:600;font-size:12px;margin-bottom:6px;color:var(--text-primary);">Aspects</div>'
                    f'<div style="display:flex;flex-wrap:wrap;">{tags}</div></div>'
                )

            context_note = parsed.get("context_note", "")
            context_html = ""
            if context_note:
                context_html = (
                    f'<div style="margin-top:10px;background:var(--accent-bg);border-left:3px solid var(--accent);'
                    f'border-radius:0 8px 8px 0;padding:8px 12px;font-size:12px;'
                    f'color:var(--text-secondary);font-style:italic;">{context_note}</div>'
                )

            product_title = ""
            if aggregate_state and aggregate_state.get("title"):
                product_title = aggregate_state["title"]
            elif product_asin and product_asin.strip():
                m = _fetch_product_meta(product_asin.strip())
                product_title = m.get("title", "")
            summary_prompt = (
                "[INST] Summarize the following product review in 1-2 sentences. "
                "Include the overall sentiment (positive/negative/neutral).\n\n"
                f"Product: {product_title or 'Unknown'}\n"
                f"Review: {review_text[:500]}\n[/INST]"
            )
            lora_summary = ""
            try:
                sr = SESSION.post(
                    LLM_URL,
                    json={"model": "smartshop-qa", "prompt": summary_prompt,
                          "max_tokens": 150, "temperature": 0.3},
                    timeout=20,
                )
                if sr.ok:
                    lora_summary = sr.json().get("choices", [{}])[0].get("text", "").strip()
            except Exception:
                pass

            summary_card = ""
            if lora_summary:
                summary_card = (
                    f'<div style="margin-top:12px;background:linear-gradient(135deg,#ede9fe 0%,#e0e7ff 100%);'
                    f'border:1px solid #c4b5fd;border-radius:8px;padding:10px 14px;">'
                    f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">'
                    f'<span style="font-size:14px;">🧠</span>'
                    f'<span style="font-weight:700;font-size:12px;color:#5b21b6;">LoRA Summary</span>'
                    f'<span style="background:#7c3aed;color:white;padding:1px 6px;border-radius:8px;'
                    f'font-size:9px;font-weight:600;">FINE-TUNED</span>'
                    f'</div>'
                    f'<div style="font-size:12px;line-height:1.5;color:#374151;">{lora_summary}</div>'
                    f'</div>'
                )

            card = (
                f'<div class="result-card" style="padding:14px;">'
                f'{header}{aspects_html}{context_html}{summary_card}'
                f'</div>'
            )
            return card, _status_strip_html()
        return f'<p style="color:var(--red-text);">LLM error: {resp.status_code}</p>', _status_strip_html()
    except requests.ConnectionError:
        return '<p style="color:var(--red-text);">LLM service unavailable.</p>', _status_strip_html()
    except Exception as e:
        return f'<p style="color:var(--red-text);">Error: {e}</p>', _status_strip_html()


def _render_intelligence_card(product_id: str, meta: dict, sentiment: dict,
                              analysis: dict, num_reviews: int, total_ms: float,
                              persona_scores: list[dict] = None,
                              similar_items: list[dict] = None,
                              lora_summary: str = "") -> str:
    """Build the rich HTML intelligence card."""
    title = meta.get("title") or f"ASIN: {product_id}"
    brand = meta.get("brand") or ""
    category = meta.get("category") or ""
    price = meta.get("price")
    avg_rating = meta.get("avg_rating") or sentiment.get("avg_rating")

    _, icon = _guess_category(product_id)
    stars = _star_rating_html(avg_rating) if avg_rating else ""
    price_html = f'<span style="font-size:16px;font-weight:700;color:var(--text-primary);">${price:.2f}</span>' if price else ""
    brand_cat = " · ".join(filter(None, [brand, category]))

    header = (
        f'<div style="display:flex;gap:16px;align-items:center;padding:16px;'
        f'background:var(--bg-card);border:1px solid var(--border);border-radius:12px;margin-bottom:16px;">'
        f'<div style="font-size:36px;">{icon}</div>'
        f'<div style="flex:1;">'
        f'<div style="font-weight:700;font-size:17px;color:var(--text-primary);">{title}</div>'
        f'<div style="font-size:12px;color:var(--text-muted);margin-top:3px;">{brand_cat}</div>'
        f'<div style="margin-top:4px;display:flex;gap:8px;align-items:center;">{stars}</div>'
        f'</div>'
        f'<div style="text-align:right;">{price_html}'
        f'<div style="font-size:10px;color:var(--text-muted);margin-top:2px;">ASIN: {product_id}</div>'
        f'</div></div>'
    )

    pos_pct = sentiment["pos_pct"]
    neu_pct = sentiment["neu_pct"]
    neg_pct = sentiment["neg_pct"]
    sentiment_bar = (
        f'<div style="margin-bottom:16px;">'
        f'<div style="font-weight:600;font-size:13px;color:var(--text-primary);margin-bottom:8px;">'
        f'Sentiment Breakdown <span style="font-weight:400;color:var(--text-muted);">({num_reviews} reviews analyzed)</span></div>'
        f'<div style="display:flex;height:10px;border-radius:5px;overflow:hidden;margin-bottom:6px;">'
        f'<div style="width:{pos_pct}%;background:#059669;" title="Positive {pos_pct}%"></div>'
        f'<div style="width:{neu_pct}%;background:#d97706;" title="Neutral {neu_pct}%"></div>'
        f'<div style="width:{neg_pct}%;background:#dc2626;" title="Negative {neg_pct}%"></div>'
        f'</div>'
        f'<div style="display:flex;gap:16px;font-size:12px;">'
        f'<span style="color:#059669;">● Positive {pos_pct}% ({sentiment["positive"]})</span>'
        f'<span style="color:#d97706;">● Neutral {neu_pct}% ({sentiment["neutral"]})</span>'
        f'<span style="color:#dc2626;">● Negative {neg_pct}% ({sentiment["negative"]})</span>'
        f'</div></div>'
    )

    pros = analysis.get("pros", [])
    cons = analysis.get("cons", [])
    pros_html = "".join(
        f'<div style="padding:6px 0;border-bottom:1px solid var(--border);font-size:13px;">'
        f'<span style="color:#059669;margin-right:6px;">✓</span>{p}</div>'
        for p in pros[:5]
    ) if pros else '<div style="color:var(--text-muted);font-size:13px;">—</div>'
    cons_html = "".join(
        f'<div style="padding:6px 0;border-bottom:1px solid var(--border);font-size:13px;">'
        f'<span style="color:#dc2626;margin-right:6px;">✗</span>{c}</div>'
        for c in cons[:5]
    ) if cons else '<div style="color:var(--text-muted);font-size:13px;">—</div>'

    pros_cons = (
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;">'
        f'<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:14px;">'
        f'<div style="font-weight:600;font-size:13px;color:#059669;margin-bottom:8px;">Top Pros</div>'
        f'{pros_html}</div>'
        f'<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:14px;">'
        f'<div style="font-weight:600;font-size:13px;color:#dc2626;margin-bottom:8px;">Top Cons</div>'
        f'{cons_html}</div></div>'
    )

    best_for = analysis.get("best_for", [])
    not_for = analysis.get("not_for", [])
    audience_html = ""
    if best_for or not_for:
        bf_badges = " · ".join(best_for[:4]) if best_for else "—"
        nf_badges = " · ".join(not_for[:4]) if not_for else "—"
        audience_html = (
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;">'
            f'<div style="font-size:13px;"><span style="font-weight:600;color:#059669;">Best for:</span> {bf_badges}</div>'
            f'<div style="font-size:13px;"><span style="font-weight:600;color:#dc2626;">Not for:</span> {nf_badges}</div>'
            f'</div>'
        )

    raw_aspects = analysis.get("aspects", {})
    if isinstance(raw_aspects, list):
        normalized = {}
        for a in raw_aspects:
            if isinstance(a, dict):
                name = a.get("aspect", a.get("name", ""))
                val = a.get("value", a.get("sentiment", "mixed"))
                if name:
                    normalized[name] = val
            elif isinstance(a, str):
                normalized[a] = "mixed"
        raw_aspects = normalized
    elif not isinstance(raw_aspects, dict):
        raw_aspects = {}
    aspects_html = ""
    if raw_aspects:
        aspect_colors = {"positive": "#059669", "negative": "#dc2626", "neutral": "#d97706", "mixed": "#6366f1"}
        aspect_rows = ""
        for asp, val in list(raw_aspects.items())[:6]:
            a_color = aspect_colors.get(val, "#94a3b8")
            bar_w = {"positive": 85, "negative": 30, "neutral": 55, "mixed": 50}.get(val, 50)
            aspect_rows += (
                f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0;font-size:12px;">'
                f'<span style="min-width:120px;color:var(--text-secondary);">{asp.replace("_", " ")}</span>'
                f'<div style="flex:1;height:6px;background:var(--border);border-radius:3px;">'
                f'<div style="width:{bar_w}%;height:100%;background:{a_color};border-radius:3px;"></div></div>'
                f'<span style="color:{a_color};font-weight:600;min-width:65px;font-size:11px;">{val}</span>'
                f'</div>'
            )
        aspects_html = (
            f'<div style="margin-bottom:16px;">'
            f'<div style="font-weight:600;font-size:13px;color:var(--text-primary);margin-bottom:8px;">Aspect Breakdown</div>'
            f'{aspect_rows}</div>'
        )

    themes = analysis.get("themes", [])
    themes_html = ""
    if themes:
        badges = " ".join(
            f'<span style="background:var(--accent-bg);color:var(--accent-text);'
            f'padding:4px 12px;border-radius:20px;font-size:12px;display:inline-block;margin:3px;">{t}</span>'
            for t in themes[:6]
        )
        themes_html = (
            f'<div style="margin-bottom:16px;">'
            f'<div style="font-weight:600;font-size:13px;color:var(--text-primary);margin-bottom:8px;">Key Themes</div>'
            f'<div>{badges}</div></div>'
        )

    # Polarization + credibility signals
    stddev = meta.get("rating_stddev")
    helpful = meta.get("helpful_votes")
    signals_html = ""
    signal_parts = []
    if stddev is not None:
        try:
            std_val = float(stddev)
            if std_val > 1.5:
                signal_parts.append('<span style="background:#fef3c7;color:#92400e;padding:3px 10px;border-radius:12px;font-size:11px;">⚡ Opinions vary widely</span>')
            elif std_val < 0.5:
                signal_parts.append('<span style="background:#d1fae5;color:#065f46;padding:3px 10px;border-radius:12px;font-size:11px;">✓ Consistent quality</span>')
        except (ValueError, TypeError):
            pass
    if helpful is not None:
        try:
            hv = int(helpful)
            if hv > 50:
                signal_parts.append(f'<span style="background:#dbeafe;color:#1e40af;padding:3px 10px;border-radius:12px;font-size:11px;">👍 {hv:,} helpful votes</span>')
            elif hv > 0:
                signal_parts.append(f'<span style="background:#f3f4f6;color:#6b7280;padding:3px 10px;border-radius:12px;font-size:11px;">👍 {hv} helpful votes</span>')
        except (ValueError, TypeError):
            pass
    if signal_parts:
        signals_html = f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;">{" ".join(signal_parts)}</div>'

    verdict = analysis.get("verdict", "")
    verdict_html = ""
    if verdict:
        verdict_html = (
            f'<div style="background:var(--accent-bg);border-left:3px solid var(--accent);'
            f'border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:16px;font-size:14px;'
            f'color:var(--text-primary);font-style:italic;">"{verdict}"</div>'
        )

    summary_html = ""
    if lora_summary:
        summary_html = (
            f'<div style="background:linear-gradient(135deg,#ede9fe 0%,#e0e7ff 100%);'
            f'border:1px solid #c4b5fd;border-radius:10px;padding:14px 16px;margin-bottom:16px;">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
            f'<span style="font-size:18px;">🧠</span>'
            f'<span style="font-weight:700;font-size:13px;color:#5b21b6;">TL;DR Summary</span>'
            f'<span style="background:#7c3aed;color:white;padding:2px 8px;border-radius:10px;'
            f'font-size:10px;font-weight:600;">Fine-tuned LoRA</span>'
            f'</div>'
            f'<div style="font-size:13px;line-height:1.6;color:#374151;">{lora_summary}</div>'
            f'</div>'
        )

    has_affinity = bool(persona_scores)
    model_used = analysis.get("_model_used", "smartshop-llm")
    llm_label = "Fine-tuned LoRA" if model_used == "smartshop-qa" else "Mistral-7B"
    has_summary = bool(lora_summary)
    pipeline = (
        f'<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:16px;">'
        f'<div class="pipeline-step done">Milvus Search<br><small>Feast vector store · {num_reviews} reviews</small></div>'
        f'<span class="pipeline-arrow">→</span>'
        f'<div class="pipeline-step done">Feast Metadata<br><small>Redis · product enrichment</small></div>'
        f'<span class="pipeline-arrow">→</span>'
        f'<div class="pipeline-step done">{llm_label}<br><small>vLLM · structured extraction</small></div>'
        f'<span class="pipeline-arrow">→</span>'
        f'<div class="pipeline-step {"done" if has_summary else ""}">LoRA Adapter<br><small>Fine-tuned · summarization</small></div>'
        f'<span class="pipeline-arrow">→</span>'
        f'<div class="pipeline-step {"done" if has_affinity else ""}">Two-Tower Model<br><small>Persona affinity · {total_ms:.0f}ms</small></div>'
        f'</div>'
    )

    behind = (
        f'<details><summary style="color:var(--text-muted);font-size:12px;cursor:pointer;">Pipeline trace</summary>'
        f'<div class="source-card" style="margin-top:8px;font-size:12px;line-height:1.8;color:var(--text-secondary);">'
        f'<strong>Review retrieval:</strong> SentenceTransformer 384d embedding → Feast retrieve_online_documents_v2 → Milvus cosine search<br>'
        f'<strong>Metadata:</strong> Feast get_online_features → Redis (&lt;1ms) → title, brand, category, price<br>'
        f'<strong>LLM (base):</strong> Mistral-7B-Instruct on vLLM · structured JSON extraction (pros/cons/themes/verdict)<br>'
        f'<strong>LLM (LoRA):</strong> Fine-tuned adapter · review summarization (trained on 1.4M Amazon reviews)<br>'
        f'<strong>Serving:</strong> KServe InferenceService · continuous batching + PagedAttention'
        f'</div></details>'
    )

    # Model affinity section
    affinity_html = ""
    if persona_scores:
        bars = ""
        for ps in sorted(persona_scores, key=lambda x: x["score"], reverse=True):
            pct = int(ps["score"] * 100)
            bar_color = "#2563eb" if pct > 60 else "#f59e0b" if pct > 30 else "#94a3b8"
            bars += (
                f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;font-size:12px;">'
                f'<span style="min-width:20px;">{ps["icon"]}</span>'
                f'<span style="min-width:140px;color:var(--text-secondary);">{ps["name"]}</span>'
                f'<div style="flex:1;height:8px;background:var(--border);border-radius:4px;">'
                f'<div style="width:{pct}%;height:100%;background:{bar_color};border-radius:4px;"></div></div>'
                f'<span style="min-width:35px;text-align:right;font-weight:600;color:{bar_color};">{pct}%</span>'
                f'</div>'
            )
        affinity_html = (
            f'<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:14px;margin-bottom:16px;">'
            f'<div style="font-weight:600;font-size:13px;color:var(--text-primary);margin-bottom:10px;">'
            f'🧠 Two-Tower Model Affinity <span style="font-weight:400;color:var(--text-muted);font-size:11px;">'
            f'— How well does this product match each persona?</span></div>'
            f'{bars}'
            f'<div style="font-size:10px;color:var(--text-muted);margin-top:8px;">'
            f'Scored by the trained Two-Tower Neural CF model · category-aware embeddings · Feast + Redis</div>'
            f'</div>'
        )

    similar_html = ""
    if similar_items:
        sim_cards = ""
        for si in similar_items[:4]:
            st = si.get("title", "")[:50]
            sc = int(si.get("score", 0) * 100)
            si_cat = si.get("category", "")
            si_icon = SUPER_CATEGORY_ICONS.get(si_cat, "📦")
            si_asin = si.get("item_id", "")
            sim_cards += (
                f'<div style="padding:8px 10px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px;">'
                f'<span>{si_icon}</span>'
                f'<div style="flex:1;min-width:0;">'
                f'<div style="font-size:12px;font-weight:500;color:var(--text-primary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{st}</div>'
                f'<div style="font-size:10px;color:var(--text-muted);">{si_cat}</div>'
                f'</div>'
                f'<span style="font-size:12px;font-weight:600;color:#2563eb;">{sc}%</span>'
                f'<button onclick="navigateToTab(1, \'{si_asin}\')" class="crosstab-btn" style="font-size:9px;">Analyze</button>'
                f'</div>'
            )
        similar_html = (
            f'<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:10px;overflow:hidden;margin-bottom:16px;">'
            f'<div style="font-weight:600;font-size:13px;color:var(--text-primary);padding:12px 14px;border-bottom:1px solid var(--border);">'
            f'🔗 Related Products <span style="font-weight:400;color:var(--text-muted);font-size:11px;">— top-scored in same category</span></div>'
            f'{sim_cards}</div>'
        )

    return pipeline + header + signals_html + sentiment_bar + summary_html + verdict_html + pros_cons + audience_html + aspects_html + themes_html + affinity_html + similar_html + behind


_last_aggregate_state: dict = {}

def analyze_product_reviews(product_id: str):
    """Aggregate review intelligence for a product ASIN using per-review LLM classification."""
    global _last_aggregate_state
    product_id = product_id.strip()
    if not product_id:
        return '<p style="color:var(--red-text);">Please enter a product ASIN.</p>', _status_strip_html()

    t0 = time.time()
    try:
        meta = _fetch_product_meta(product_id)
        product_title = meta.get("title", "")
        sources = _fetch_reviews(product_id, product_title=product_title, top_k=20)

        if not sources:
            return (
                f'<div class="result-card" style="text-align:center;color:var(--text-muted);">'
                f'No reviews found in Milvus for ASIN <strong>{product_id}</strong>.<br>'
                f'<small>Ensure embeddings have been materialized for this product.</small></div>',
                _status_strip_html(),
            )

        analysis = _llm_product_intelligence(meta, sources)
        lora_summary = _llm_summarize_reviews(meta, sources)
        sentiment = _aggregate_sentiment(analysis)

        ratings = [float(s.get("rating", 0)) for s in sources if s.get("rating")]
        sentiment["avg_rating"] = sum(ratings) / len(ratings) if ratings else meta.get("avg_rating", 0)

        persona_scores = _fetch_persona_affinity(product_id)
        similar_items = _fetch_related_products(product_id)

        ms = (time.time() - t0) * 1000
        _record("llm", ms)

        _last_aggregate_state = {
            "asin": product_id, "title": product_title,
            "category": meta.get("category", ""),
            "pos_pct": sentiment.get("pos_pct"),
        }

        card = _render_intelligence_card(
            product_id, meta, sentiment, analysis, len(sources), ms,
            persona_scores=persona_scores, similar_items=similar_items,
            lora_summary=lora_summary,
        )
        return card, _status_strip_html()

    except requests.ConnectionError:
        return '<p style="color:var(--red-text);">Service unavailable. Check KServe deployments.</p>', _status_strip_html()
    except Exception as e:
        return f'<p style="color:var(--red-text);">Error: {e}</p>', _status_strip_html()


def _build_product_card(product_id: str, meta: dict) -> str:
    """Render a product context card from metadata dict (used by Product Q&A tab)."""
    title = meta.get("title") or ""
    brand = meta.get("brand") or ""
    category = meta.get("category") or ""
    avg_rating = meta.get("avg_rating")
    review_count = meta.get("review_count")
    price = meta.get("price")

    if not (title or avg_rating or review_count or price):
        return ""

    display_title = title if title else f"ASIN: {product_id.strip()}"
    stars = _star_rating_html(avg_rating) if avg_rating else ""
    rc_html = f'<span style="font-size:12px;color:var(--text-muted);">({review_count:,} reviews)</span>' if review_count else ""
    price_html = f'<span style="font-size:16px;font-weight:700;color:var(--text-primary);">${price:.2f}</span>' if price else ""
    brand_html = f'<span style="font-size:12px;color:var(--text-muted);">{brand}</span>' if brand else ""
    cat_html = (
        f'<span style="background:var(--accent-bg);color:var(--accent-text);'
        f'padding:2px 8px;border-radius:10px;font-size:11px;">{category}</span>'
    ) if category else ""

    return (
        f'<div style="display:flex;gap:16px;align-items:center;padding:14px 16px;'
        f'background:var(--bg-card);border:1px solid var(--border);border-radius:10px;margin-bottom:14px;">'
        f'<div style="font-size:32px;">📦</div>'
        f'<div style="flex:1;">'
        f'<div style="font-weight:600;font-size:15px;color:var(--text-primary);">{display_title}</div>'
        f'<div style="display:flex;gap:8px;align-items:center;margin-top:4px;flex-wrap:wrap;">'
        f'{brand_html} {cat_html}'
        f'</div>'
        f'<div style="display:flex;gap:12px;align-items:center;margin-top:6px;">'
        f'{stars} {rc_html}'
        f'</div>'
        f'</div>'
        f'<div style="text-align:right;">'
        f'{price_html}'
        f'<div style="font-size:10px;color:var(--text-muted);margin-top:2px;">ASIN: {product_id}</div>'
        f'</div>'
        f'</div>'
    )


def ask_question(question: str, product_id: str = ""):
    t0 = time.time()
    try:
        meta = _fetch_product_meta(product_id) if product_id.strip() else {}
        product_card = _build_product_card(product_id, meta) if meta else ""

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

            pipeline = (
                f'<div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:16px;">'
                f'<div class="pipeline-step done">Embed Query<br><small>SentenceTransformer 384d</small></div>'
                f'<span class="pipeline-arrow">→</span>'
                f'<div class="pipeline-step done">Feast + Vector Store<br><small>Milvus similarity · {len(sources)} docs</small></div>'
                f'<span class="pipeline-arrow">→</span>'
                f'<div class="pipeline-step done">Mistral-7B Generate<br><small>Contextual answer · {ms:.0f}ms total</small></div>'
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
            return product_card + pipeline + answer_html + sources_html + badge + feast_note, _status_strip_html()
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
        ("LLM Analyzer", LLM_URL, "/health", "vLLM · Mistral-7B", "#7c3aed"),
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
        233M reviews · 33 categories<br>
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
        <small>User/item features: interaction counts, avg ratings, category preferences<br>
        LLM training data: review → summarization instruction pairs</small>
      </div>
    </div>
    <div class="arch-box" style="border-color:#f97316;">
      <div class="arch-title">Embedding Pipeline</div>
      <div class="arch-detail">
        SentenceTransformer (all-MiniLM-L6-v2 · 384d)<br>
        <small>Review text → dense vectors → Feast materialization → Milvus</small>
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
          <small>user_features · item_features · item_metadata<br>
          <code style="font-size:10px;">feast materialize</code> → online &lt;1ms lookups</small>
        </div>
        <div>
          <strong style="color:var(--text-primary);">Vector features → Milvus</strong><br>
          <small>review_embeddings (384d vectors)<br>
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
        <small>PyTorch DDP · Kubeflow TrainJob · 4 GPUs<br>
        4M users · 1.8M items → dot product scoring<br>
        5 super-categories · category-aware embeddings · 64d</small>
      </div>
    </div>
    <div class="arch-box" style="border-color:#7c3aed;">
      <div class="arch-title">LLM Fine-Tuning</div>
      <div class="arch-detail">
        Mistral-7B-Instruct · LoRA + FSDP<br>
        <small>Kubeflow TrainJob · LoRA rank 16 · 4 GPUs<br>
        Trained on 1.4M Amazon reviews · summarization task<br>
        Adapter hot-loaded at serving time via vLLM LoRA support</small>
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
      <tr style="border-bottom:1px solid var(--border);"><td style="padding:6px 8px;">Data Processing</td><td>Spark on K8s</td><td>Feature engineering, training data generation</td></tr>
      <tr style="border-bottom:1px solid var(--border);"><td style="padding:6px 8px;">Feature Store</td><td>Feast (Redis + Milvus)</td><td>Online features &lt;1ms · vector retrieval</td></tr>
      <tr style="border-bottom:1px solid var(--border);"><td style="padding:6px 8px;">Training</td><td>Kubeflow TrainJob</td><td>Distributed PyTorch DDP + LoRA FSDP</td></tr>
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

/* Cross-tab buttons */
.crosstab-btn { font-size:10px; padding:3px 8px; border:1px solid var(--border); border-radius:6px; background:var(--bg-card); color:var(--accent); cursor:pointer; transition:all 0.15s; }
.crosstab-btn:hover { background:var(--accent); color:white; border-color:var(--accent); }

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
  window.navigateToTab = function(tabIdx, asin) {
    const tabs = document.querySelectorAll('.tabs > .tab-nav > button');
    if (tabs[tabIdx]) tabs[tabIdx].click();
    setTimeout(() => {
      const inputs = document.querySelectorAll('textarea, input[type="text"]');
      for (const inp of inputs) {
        const label = inp.closest('.block')?.querySelector('label, span');
        if (label && /ASIN|Product ID/i.test(label.textContent)) {
          inp.value = asin;
          inp.dispatchEvent(new Event('input', {bubbles: true}));
          break;
        }
      }
    }, 300);
  };
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
                '<strong style="color:var(--text-primary);">Feast</strong> (&lt;1ms via Redis), runs the '
                '<strong style="color:var(--text-primary);">Two-Tower model</strong> with category-aware embeddings '
                'across 5 super-categories, and surfaces personalized picks scored against the full catalog.'
                '</div>'
            )
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("**Select a shopper persona** or enter any User ID to see per-user results:")
                    persona_radio = gr.Radio(
                        choices=[p[0] for p in PERSONAS],
                        label="Shopper Persona",
                        value=PERSONAS[0][0],
                    )
                    persona_info = gr.HTML(
                        f'<div style="font-size:12px;color:var(--text-muted);padding:4px 0;">'
                        f'{PERSONA_CONTEXT.get(PERSONAS[0][0], "")}</div>'
                    )
                    user_input = gr.Textbox(
                        label="User ID (optional — overrides persona)",
                        placeholder="Leave empty to use persona above",
                        lines=1,
                    )
                    _unique_uids = list(dict.fromkeys(uid for _, uid in PERSONAS))
                    gr.Examples(
                        examples=[[uid] for uid in _unique_uids],
                        inputs=[user_input],
                        label="Try a user ID:",
                    )
                    top_k_input = gr.Slider(minimum=1, maximum=30, value=10, step=1, label="Top K")
                    rec_btn = gr.Button("Get Recommendations", variant="primary", size="lg")
                with gr.Column(scale=3):
                    rec_output = gr.HTML('<div class="result-card" style="text-align:center;color:var(--text-muted);">Click "Get Recommendations" to see results</div>')

            def _update_persona_info(persona):
                ctx = PERSONA_CONTEXT.get(persona, "")
                cat_key = ctx.split(" · ")[0] if " · " in ctx else ""
                icon = SUPER_CATEGORY_ICONS.get(cat_key, "📦")
                subcats = ctx.split(" · ")[1] if " · " in ctx else ctx

                profile_html = ""
                uid = dict(PERSONAS).get(persona)
                if uid:
                    try:
                        resp = SESSION.post(RECOMMEND_PROFILE_URL, json={"user_id": uid}, timeout=3)
                        if resp.ok:
                            p = resp.json().get("profile", {})
                            parts = []
                            if p.get("review_count"):
                                parts.append(f'<strong>{p["review_count"]:,}</strong> reviews')
                            if p.get("avg_rating"):
                                parts.append(f'avg ★ <strong>{p["avg_rating"]:.1f}</strong>')
                            if p.get("tenure_days"):
                                parts.append(f'<strong>{p["tenure_days"]:,}</strong>d tenure')
                            if parts:
                                profile_html = f'<div style="font-size:11px;color:var(--text-muted);margin-top:4px;">{" · ".join(parts)}</div>'
                    except Exception:
                        pass

                return (
                    f'<div style="display:flex;gap:10px;align-items:center;padding:8px 12px;'
                    f'background:var(--bg-card);border:1px solid var(--border);border-radius:10px;">'
                    f'<span style="font-size:28px;">{icon}</span>'
                    f'<div>'
                    f'<div style="font-weight:600;font-size:13px;color:var(--text-primary);">{cat_key}</div>'
                    f'<div style="font-size:11px;color:var(--text-secondary);">{subcats}</div>'
                    f'{profile_html}'
                    f'<div style="font-size:10px;color:var(--text-muted);margin-top:2px;">'
                    f'<code>{uid}</code></div>'
                    f'</div></div>'
                )

            persona_radio.change(_update_persona_info, inputs=[persona_radio], outputs=[persona_info])

            def _resolve_rec(persona, custom_id, top_k):
                uid = custom_id.strip() if custom_id.strip() else dict(PERSONAS).get(persona, PERSONAS[0][1])
                ctx = PERSONA_CONTEXT.get(persona, "")
                pref_cat = ctx.split(" · ")[0] if " · " in ctx else ""
                return get_recommendations(uid, top_k, preferred_category=pref_cat)

            rec_btn.click(_resolve_rec, inputs=[persona_radio, user_input, top_k_input], outputs=[rec_output, status_strip])

        # ── Tab 2: Review Intelligence ──────────────────────────────────────
        with gr.Tab("Review Intelligence"):
            gr.HTML(
                '<div style="color:var(--text-secondary);font-size:13px;margin:8px 0;line-height:1.6;">'
                'Enter a product ASIN for <strong style="color:var(--text-primary);">aggregate intelligence</strong> '
                '(reviews from Feast → Milvus, analysed by Mistral-7B), or paste a single review for '
                '<strong style="color:var(--text-primary);">real-time classification</strong>. '
                'The <strong style="color:#7c3aed;">fine-tuned LoRA adapter</strong> generates TL;DR summaries '
                '(trained on 1.4M product reviews) while the base model handles structured extraction.'
                '</div>'
            )
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    gr.HTML(
                        '<div style="font-weight:600;font-size:13px;color:var(--text-primary);margin-bottom:6px;">'
                        '🔍 Live Review Classifier</div>'
                    )
                    classify_review_text = gr.Textbox(
                        label="Review Text",
                        placeholder="Great product! The battery life is amazing but the screen is dim...",
                        lines=4,
                    )
                    review_asin_input = gr.Textbox(
                        label="Product ASIN",
                        placeholder="e.g. B07NW1CG2S",
                        lines=1,
                    )
                    with gr.Row():
                        classify_btn = gr.Button("Classify Review", variant="secondary", size="lg")
                        analyze_btn = gr.Button("Analyze All Reviews", variant="primary", size="lg")
                    gr.Examples(
                        examples=EXAMPLE_CLASSIFY,
                        inputs=[classify_review_text, review_asin_input],
                        label="Try a sample review:",
                    )
                    classify_output = gr.HTML(
                        '<div class="result-card" style="text-align:center;color:var(--text-muted);">'
                        'Paste a review and click "Classify Review" for real-time sentiment analysis</div>'
                    )
                with gr.Column(scale=3):
                    llm_output = gr.HTML(
                        '<div class="result-card" style="text-align:center;color:var(--text-muted);">'
                        'Enter a product ASIN and click "Analyze All Reviews" to see aggregate intelligence</div>'
                    )

            analyze_btn.click(analyze_product_reviews, inputs=[review_asin_input], outputs=[llm_output, status_strip])

            def _do_classify(review_text, asin_input):
                if not review_text or not review_text.strip():
                    return '<p style="color:var(--red-text);">Please enter review text.</p>', _status_strip_html()
                return _classify_single_review(review_text, asin_input, _last_aggregate_state)

            classify_btn.click(
                _do_classify,
                inputs=[classify_review_text, review_asin_input],
                outputs=[classify_output, status_strip],
            )

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
