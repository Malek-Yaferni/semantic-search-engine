"""Core search logic for a BEIR-style dataset:
semantic search (Sentence Transformers + FAISS) vs keyword search (BM25).

Expected layout:
    data/raw/corpus/corpus.jsonl
    data/raw/queries/queries.jsonl
    data/raw/qrels/qrels.json
    data/processed/          <- cached FAISS indexes are written here
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Selectable embedding models. Some models expect prefixes on queries/documents.
MODELS = {
    "all-MiniLM-L6-v2": {},
    "BAAI/bge-small-en-v1.5": {
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "intfloat/e5-small-v2": {"query_prefix": "query: ", "doc_prefix": "passage: "},
}

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "it", "its", "that", "this", "with", "as", "by", "be", "can", "from", "at",
    "how", "do", "does", "what", "why", "i", "my", "we", "you", "so", "than",
}


# ----------------------------------------------------------------- loading
def _find(folder: str, preferred: str) -> Path:
    """Use the preferred file name, else the first file in the folder."""
    p = RAW / folder / preferred
    if p.exists():
        return p
    files = sorted(f for f in (RAW / folder).glob("*") if f.is_file())
    if not files:
        raise FileNotFoundError(f"No files found in {RAW / folder}")
    return files[0]


def _first(d: dict, keys: tuple, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_qrels() -> dict[str, dict[str, int]]:
    """Returns {query_id: {doc_id: relevance}} (only relevance > 0)."""
    path = _find("qrels", "qrels.json")
    raw = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:  # JSON Lines
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]

    qrels: dict[str, dict[str, int]] = {}

    def add(q, d, s=1):
        if q is None or d is None:
            return
        if int(float(s)) > 0:
            qrels.setdefault(str(q), {})[str(d)] = int(float(s))

    if isinstance(data, dict) and isinstance(data.get("query-id"), list):
        # column-oriented: {"query-id": [...], "corpus-id": [...], "score": [...]}
        scores = data.get("score", [1] * len(data["query-id"]))
        for q, d, s in zip(data["query-id"], data["corpus-id"], scores):
            add(q, d, s)
    elif isinstance(data, dict):
        for q, v in data.items():
            if isinstance(v, dict):
                for d, s in v.items():
                    add(q, d, s)
            elif isinstance(v, list):
                for d in v:
                    add(q, d, 1)
    else:  # list of records
        for r in data:
            add(
                _first(r, ("query-id", "query_id", "qid", "query")),
                _first(r, ("corpus-id", "corpus_id", "doc_id", "docid")),
                _first(r, ("score", "relevance", "rel"), 1),
            )
    return qrels


def load_queries() -> dict[str, str]:
    path = _find("queries", "queries.jsonl")
    return {
        str(_first(r, ("_id", "id", "query_id"))): str(_first(r, ("text", "query"), ""))
        for r in read_jsonl(path)
    }


def load_corpus(max_docs: int | None = None, must_keep: set[str] | None = None) -> list[dict]:
    """Normalises records to {"id", "title", "content"}.
    If max_docs is set, documents referenced in qrels (must_keep) are always kept."""
    path = _find("corpus", "corpus.jsonl")
    docs = [
        {
            "id": str(_first(r, ("_id", "id", "doc_id"))),
            "title": str(_first(r, ("title",), "") or ""),
            "content": str(_first(r, ("text", "content", "body"), "") or ""),
        }
        for r in read_jsonl(path)
    ]
    if max_docs and len(docs) > max_docs:
        must_keep = must_keep or set()
        keep = [d for d in docs if d["id"] in must_keep]
        rest = [d for d in docs if d["id"] not in must_keep]
        docs = keep + rest[: max(0, max_docs - len(keep))]
    return docs


def doc_text(doc: dict) -> str:
    return f"{doc['title']}. {doc['content']}" if doc["title"] else doc["content"]


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


# ------------------------------------------------------------------ engines
class SemanticSearch:
    """Embeds documents once, stores them in FAISS (cached on disk), searches by meaning."""

    def __init__(self, docs: list[dict], model_name: str = MODEL_NAME):
        self.docs = docs
        self.model_name = model_name
        cfg = MODELS.get(model_name, {})
        self.query_prefix = cfg.get("query_prefix", "")
        self.doc_prefix = cfg.get("doc_prefix", "")

        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = min(self.model.max_seq_length, 384)  # faster on CPU

        PROCESSED.mkdir(parents=True, exist_ok=True)
        key = hashlib.md5(
            (model_name + "|".join(d["id"] for d in docs)).encode()
        ).hexdigest()[:10]
        index_path = PROCESSED / f"faiss_{key}.index"

        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        else:
            emb = self.model.encode(
                [self.doc_prefix + doc_text(d) for d in docs],
                batch_size=64,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            ).astype("float32")
            # Normalised vectors: inner product == cosine similarity
            self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(emb)
            faiss.write_index(self.index, str(index_path))

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = self.model.encode(
            [self.query_prefix + query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, ids = self.index.search(q, k)
        return [
            {**self.docs[i], "score": float(s)}
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]


class KeywordSearch:
    """Traditional lexical retrieval with BM25."""

    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(doc_text(d)) for d in docs])

    def search(self, query: str, k: int = 5) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [
            {**self.docs[i], "score": float(scores[i])} for i in top if scores[i] > 0
        ]


class HybridSearch:
    """Combines semantic and BM25 rankings with Reciprocal Rank Fusion (RRF).

    Each document scores sum(1 / (rrf_k + rank)) over the rankings it appears in,
    so documents ranked well by BOTH methods rise to the top. RRF works on ranks,
    so the incompatible score scales (cosine vs BM25) don't matter.
    """

    def __init__(self, semantic: SemanticSearch, keyword: KeywordSearch,
                 rrf_k: int = 60, depth: int = 50):
        self.semantic, self.keyword = semantic, keyword
        self.rrf_k, self.depth = rrf_k, depth

    def search(self, query: str, k: int = 5) -> list[dict]:
        fused: dict[str, float] = {}
        by_id: dict[str, dict] = {}
        for engine in (self.semantic, self.keyword):
            for rank, r in enumerate(engine.search(query, self.depth), start=1):
                fused[r["id"]] = fused.get(r["id"], 0.0) + 1.0 / (self.rrf_k + rank)
                by_id[r["id"]] = r
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [{**by_id[i], "score": fused[i]} for i in top]


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000