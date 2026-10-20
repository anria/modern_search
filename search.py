#!/usr/bin/env python3
"""
search.py — standalone search page over /data/cases.pkl.

Run with:  uv run search.py
Serves on: http://127.0.0.1:8766

uv add numpy
uv add pandas
uv add rank_bm25
uv add sentence-transformers
"""

import json
import os
import pickle
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_DENSE_MODEL = True
except ImportError:
    HAS_DENSE_MODEL = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
try: 
    # This is local - on codespaces it takes env from the Secrets in the Settings
    load_dotenv()
except:
    pass

PORT = os.getenv("SEARCH_PORT")
if isinstance(PORT, str):
    PORT = int(PORT)
HOST = os.getenv("HOST")

BASE_DIR = Path(__file__).resolve().parent
CASES_FILE = BASE_DIR / "data" / "cases.pkl"

# Fields exposed in results
DISPLAY_FIELDS = ["case_name", "title", "summary", "court_name",
                  "state", "judge_ruling", "case_date", "url"]

# ---------------------------------------------------------------------------
# Load the pickle into a DataFrame
# ---------------------------------------------------------------------------
def load_cases() -> pd.DataFrame:
    if not CASES_FILE.exists():
        print(f"[search] WARNING: {CASES_FILE} not found — using empty corpus")
        return pd.DataFrame(columns=DISPLAY_FIELDS + ["text",
                            "dv_case_text", "dv_general_text", "id"])

    with open(CASES_FILE, "rb") as f:
        records = pickle.load(f)

    if isinstance(records, dict):
        records = [records]
    df = pd.DataFrame(records)

    # Ensure every display column exists, even if the pickle is sparse
    for col in DISPLAY_FIELDS + ["text", "dv_case_text", "dv_general_text", "id"]:
        if col not in df.columns:
            df[col] = ""

    # String-ify the text fields for search; leave vectors as-is
    for col in ["case_name", "title", "summary", "text", "court_name",
                "state", "judge_ruling", "case_date", "url", "id"]:
        df[col] = df[col].fillna("").astype(str)

    print(f"[search] Loaded {len(df)} cases from {CASES_FILE}")
    return df


CASES_DF = load_cases()

# ---------------------------------------------------------------------------
# Search corpus: one searchable string per row
# ---------------------------------------------------------------------------
def build_corpus_text(df: pd.DataFrame) -> list:
    return [
        f"{row.title} {row.summary} {row.text}".strip()
        for row in df.itertuples(index=False)
    ]

CORPUS = build_corpus_text(CASES_DF) if len(CASES_DF) else []
DOC_IDS = list(CASES_DF["id"]) if len(CASES_DF) else []

# ---------------------------------------------------------------------------
# BM25 (built lazily)
# ---------------------------------------------------------------------------
_bm25 = None
_bm25_lock = threading.Lock()

def tokenize(text: str):
    return re.findall(r"[\w$,]+", text.lower())

def get_bm25():
    global _bm25
    if not HAS_BM25 or not CORPUS:
        return None
    if _bm25 is None:
        with _bm25_lock:
            if _bm25 is None:
                _bm25 = BM25Okapi([tokenize(d) for d in CORPUS], k1=1.5, b=0.75)
    return _bm25

# ---------------------------------------------------------------------------
# Dense search over the precomputed vectors in the pickle
# ---------------------------------------------------------------------------
# Choose which stored vector to use: "dv_case_text" or "dv_general_text"
DENSE_FIELD = "dv_case_text"

_doc_matrix = None
_doc_matrix_lock = threading.Lock()

def get_doc_matrix() -> np.ndarray | None:
    """Stack dv_* vectors into a single (N, d) matrix, L2-normalised."""
    global _doc_matrix
    if _doc_matrix is None:
        with _doc_matrix_lock:
            if _doc_matrix is None:
                vecs = CASES_DF[DENSE_FIELD].tolist()
                if not vecs:
                    return None
                arr = np.array([np.asarray(v, dtype=float) for v in vecs])
                # Normalise so dot product = cosine similarity
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                _doc_matrix = arr / norms
    return _doc_matrix

# Query embedding model — used to embed the user's query into the same space
_embed_model = None
_embed_lock = threading.Lock()

def get_embed_model():
    global _embed_model
    if not HAS_DENSE_MODEL:
        return None
    if _embed_model is None:
        with _embed_lock:
            if _embed_model is None:
                _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model

def embed_query(query: str) -> np.ndarray | None:
    model = get_embed_model()
    if model is None:
        return None
    emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    return emb

# ---------------------------------------------------------------------------
# Search implementations
# ---------------------------------------------------------------------------
def plain_search(query: str, k: int = 10):
    q = query.lower()
    hits = []
    for i, doc in enumerate(CORPUS):
        if q in doc.lower():
            hits.append((DOC_IDS[i], i, 1.0))
    return hits[:k]


def bm25_search(query: str, k: int = 10):
    bm25 = get_bm25()
    if bm25 is None:
        return []
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return [(DOC_IDS[i], i, float(scores[i]))
            for i in order[:k] if scores[i] > 0]


def dense_search(query: str, k: int = 10):
    mat = get_doc_matrix()
    q_vec = embed_query(query)
    if mat is None or q_vec is None:
        return []
    sims = mat @ q_vec
    order = np.argsort(-sims)[:k]
    return [(DOC_IDS[i], int(i), float(sims[i])) for i in order]


def hybrid_search(query: str, k: int = 10, alpha: float = 0.7):
    bm_hits = {i: s for _, i, s in bm25_search(query, k=len(CORPUS))}
    dn_hits = {i: s for _, i, s in dense_search(query, k=len(CORPUS))}

    def norm(d):
        if not d:
            return {}
        vals = list(d.values())
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {i: 0.5 for i in d}
        return {i: (s - lo) / (hi - lo) for i, s in d.items()}

    bm_n, dn_n = norm(bm_hits), norm(dn_hits)
    blended = {i: alpha * dn_n.get(i, 0.0) + (1 - alpha) * bm_n.get(i, 0.0)
               for i in set(bm_n) | set(dn_n)}
    order = sorted(blended.items(), key=lambda x: -x[1])[:k]
    return [(DOC_IDS[i], i, s) for i, s in order]


def all_search(query: str, k: int = 10):
    seen = {}
    for hits in (plain_search(query, k), bm25_search(query, k),
                 dense_search(query, k), hybrid_search(query, k)):
        for doc_id, idx, score in hits:
            if idx not in seen or score > seen[idx][1]:
                seen[idx] = (doc_id, score)
    order = sorted(seen.items(), key=lambda x: -x[1][1])[:k]
    return [(doc_id, idx, s) for idx, (doc_id, s) in order]


SEARCH_MODES = {
    "plain":  ("plain",  "Plain Search",
               "Case-insensitive substring match across title, summary, and full text. Fast and exact, but does not understand synonyms."),
    "bm25":   ("bm25",   "BM25",
               "Lexical ranking using term frequency and inverse document frequency. Strong on exact names, citations, and statutes; weak on paraphrases."),
    "dense":  ("dense",  "Dense Vectors",
               "Cosine similarity between your query and the precomputed dv_case_text vector for each case. Captures semantic meaning, but depends on the embedding model used to build the corpus."),
    "hybrid": ("hybrid", "BM25 + Dense Vectors",
               "Weighted blend (70% dense, 30% BM25). Balances semantic similarity with exact-term matching."),
    "all":    ("all",    "All Combined",
               "Union of all modes, deduplicated by best score. Maximises recall at the cost of precision."),
}

# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
BASE_CSS = """
:root {
    --bg: #1e293b; --bg-elevated: #334155; --bg-surface: #0f172a;
    --border: #475569; --border-accent: #00d4ff;
    --text: #e2e8f0; --text-muted: #94a3b8;
    --accent: #00d4ff; --accent-hover: #38bdf8;
    --accent-glow: rgba(0, 212, 255, 0.35);
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    font-size: 14px; line-height: 1.55;
    background: var(--bg); color: var(--text); min-height: 100vh;
}
a { color: var(--accent); text-decoration: none; }
a:hover { color: var(--accent-hover); }
header.banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-bottom: 1px solid var(--border-accent);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    padding: 12px 24px; display: flex; align-items: center;
    justify-content: space-between; flex-wrap: wrap; gap: 12px;
    position: sticky; top: 0; z-index: 100;
}
.banner-title { color: var(--accent); font-size: 20px; font-weight: 700;
    letter-spacing: 0.4px; text-shadow: 0 0 12px var(--accent-glow); }
main { max-width: 1000px; margin: 0 auto; padding: 28px 24px; }
.search-bar { display: flex; gap: 10px; margin-bottom: 20px; }
.search-bar input[type="text"] {
    flex: 1; background: var(--bg-surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 14px; color: var(--text);
    font-size: 14px; font-family: inherit; outline: none;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.search-bar input[type="text"]:focus { border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(0,212,255,0.15); }
.search-bar select {
    background: var(--bg-surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 14px; color: var(--text);
    font-size: 14px; font-family: inherit; cursor: pointer; outline: none;
}
.search-bar button {
    background: var(--accent); color: #0f172a; border: none;
    border-radius: 8px; padding: 10px 22px; cursor: pointer;
    font-size: 14px; font-weight: 600; transition: background 0.15s ease,
    box-shadow 0.15s ease;
}
.search-bar button:hover { background: var(--accent-hover);
    box-shadow: 0 0 14px var(--accent-glow); }
.summary {
    background: var(--bg-elevated); border: 1px solid var(--border);
    border-left: 3px solid var(--accent); border-radius: 8px;
    padding: 14px 18px; margin-bottom: 20px;
}
.summary-title { color: var(--accent); font-weight: 700; font-size: 13px;
    text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 4px; }
.summary-text { color: var(--text); font-size: 13px; }
.results { list-style: none; }
.result {
    background: var(--bg-elevated); border: 1px solid var(--border);
    border-left: 3px solid var(--accent); border-radius: 8px;
    padding: 14px 18px; margin-bottom: 12px;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.result:hover { box-shadow: 0 0 14px rgba(0,212,255,0.15); }
.result-id { color: var(--text-muted); font-size: 11px;
    font-family: ui-monospace, monospace; margin-bottom: 4px; }
.result-title { color: var(--accent); font-weight: 600; font-size: 15px;
    margin-bottom: 4px; }
.result-meta { color: var(--text-muted); font-size: 12px; margin-bottom: 6px; }
.result-text { color: var(--text); font-size: 14px; white-space: pre-wrap;
    word-break: break-word; }
.result-score { color: var(--accent); font-size: 12px; margin-top: 6px; }
.empty { text-align: center; padding: 60px 20px; color: var(--text-muted); }
.warn { color: #fbbf24; font-size: 12px; margin-top: 8px; }
.banner-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.banner-left svg { 
    width: 38px;
    height: 38px;
    filter: drop-shadow(0 0 6px var(--accent-glow));
}

.banner-right {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}
"""

SEARCH_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Search — Case Results</title>
<style>__BASE_CSS__</style>
</head>
<body>
<header class="banner">
    <div class="banner-left">
        <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="50" r="45" fill="#0f172a" stroke="#00d4ff" stroke-width="3"/>
            <text x="50" y="68" font-size="46" text-anchor="middle"
                  fill="#00d4ff" font-family="-apple-system, Arial, sans-serif" font-weight="700">AI</text>
            <circle cx="30" cy="40" r="7" fill="#00d4ff"/>
            <circle cx="70" cy="40" r="7" fill="#00d4ff"/>
            <path d="M35 65 Q50 74 65 65" stroke="#00d4ff" stroke-width="3" fill="none"/>
        </svg>

        <span class="banner-title">AI Search — Case Corpus</span>
    </div>
    
</header>
<main>
    <form class="search-bar" method="get" action="/">
        <input type="text" name="q" placeholder="Search cases…"
               value="__QUERY__" autofocus>
        <select name="mode">
            <option value="plain"  __SEL_PLAIN__>plain search</option>
            <option value="bm25"   __SEL_BM25__>bm25</option>
            <option value="dense"  __SEL_DENSE__>Dense Vectors</option>
            <option value="hybrid" __SEL_HYBRID__>bm25 + Dense Vectors</option>
            <option value="all"    __SEL_ALL__>All combined</option>
        </select>
        <button type="submit">Search</button>
    </form>
    <div class="summary">
        <div class="summary-title">__MODE_LABEL__</div>
        <div class="summary-text">__MODE_DESC__</div>
    </div>
    <ul class="results">
        __RESULTS__
    </ul>
    <p class="warn">__WARN__</p>
</main>
</body>
</html>
"""

def render_result(df_row) -> str:
    idx = df_row.name
    title = df_row["case_name"] or df_row["title"] or f"Case {df_row['id']}"
    summary = df_row["summary"] or df_row["text"][:300]
    meta = " · ".join(filter(None, [
        df_row["court_name"], df_row["state"],
        df_row["case_date"], df_row["judge_ruling"],
    ]))
    return (
        f'<li class="result">'
        f'<div class="result-id">{escape(df_row["id"])}</div>'
        f'<div class="result-title">{escape(title)}</div>'
        f'<div class="result-meta">{escape(meta)}</div>'
        f'<div class="result-text">{escape(summary[:400])}</div>'
        f'<div class="result-score">__SCORE__</div>'
        f'</li>'
    )


def render_page(query: str, mode: str, results: list, warn: str = "") -> str:
    mode_key, mode_label, mode_desc = SEARCH_MODES.get(
        mode, ("plain", "Plain Search", SEARCH_MODES["plain"][2]))

    if not query:
        results_html = '<div class="empty">Enter a query above and press Search.</div>'
    elif not results:
        results_html = '<div class="empty">No cases found.</div>'
    else:
        parts = []
        for doc_id, idx, score in results:
            row = CASES_DF.iloc[idx]
            title = row["case_name"] or row["title"] or f"Case {row['id']}"
            summary = row["summary"] or row["text"][:300]
            judge_ruling = row["judge_ruling"] or ""
            meta = " · ".join(filter(None, [
                row["court_name"], row["state"], row["case_date"], 
            ]))
            parts.append(
                f'<li class="result">'
                # f'<div class="result-id">{escape(row["id"])}</div>'
                f'<div class="result-title"><u><b>Case: </b></u>{escape(title)}</div>'
                f'<div class="result-meta"><u><b>Court: </b></u>{escape(meta)}</div>'
                f'<br><div class="result-text"><u><b>Summary: </b></u>{escape(summary[:400])}</div>'
                f'<br><div class="result-text"><u><b>Ruling: </b></u>{escape(judge_ruling[:800])}</div>'
                f'<div class="result-score">score: {score:.4f} - <span class="result-id">{escape(row["id"])}</span></div>'
                f'</li>'
            )
        results_html = "".join(parts)

    sel = {k: "selected" if k == mode_key else "" for k in SEARCH_MODES}
    html = SEARCH_HTML
    html = html.replace("__BASE_CSS__", BASE_CSS)
    html = html.replace("__QUERY__", escape(query))
    html = html.replace("__SEL_PLAIN__", sel["plain"])
    html = html.replace("__SEL_BM25__", sel["bm25"])
    html = html.replace("__SEL_DENSE__", sel["dense"])
    html = html.replace("__SEL_HYBRID__", sel["hybrid"])
    html = html.replace("__SEL_ALL__", sel["all"])
    html = html.replace("__MODE_LABEL__", mode_label)
    html = html.replace("__MODE_DESC__", mode_desc)
    html = html.replace("__RESULTS__", results_html)
    html = html.replace("__WARN__", warn)
    return html


def escape(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[search] {self.address_string()} - {fmt % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        query = qs.get("q", [""])[0].strip()
        mode = qs.get("mode", ["plain"])[0]
        warn = ""

        if query:
            if mode == "bm25" and not HAS_BM25:
                warn = "rank_bm25 not installed — run: uv add rank_bm25"
            elif mode in ("dense", "hybrid") and not HAS_DENSE_MODEL:
                warn = "sentence-transformers not installed — run: uv add sentence-transformers"
            elif mode == "hybrid" and not HAS_BM25:
                warn = "BM25 dependency missing"

            try:
                if mode == "bm25":
                    results = bm25_search(query)
                elif mode == "dense":
                    results = dense_search(query)
                elif mode == "hybrid":
                    results = hybrid_search(query)
                elif mode == "all":
                    results = all_search(query)
                else:
                    results = plain_search(query)
            except Exception as e:
                results = []
                warn = f"Search error: {e}"
        else:
            results = []

        self._send_html(render_page(query, mode, results, warn))

    def _send_html(self, html: str, status: int = 200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[search] Serving on http://{HOST}:{PORT}")
    print(f"[search] Corpus: {len(CASES_DF)} cases")
    print(f"[search] BM25:   {'available' if HAS_BM25 else 'missing (uv add rank_bm25)'}")
    print(f"[search] Dense:  {'available' if HAS_DENSE_MODEL else 'missing (uv add sentence-transformers)'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[search] Shutting down…")
        server.shutdown()


if __name__ == "__main__":
    main()