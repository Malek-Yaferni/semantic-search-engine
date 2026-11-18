"""Compare BM25, semantic and hybrid search on the labelled queries.

Usage:
  python evaluate.py
  python evaluate.py --models all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5 intfloat/e5-small-v2
  python evaluate.py --max-queries 100 --max-docs 20000
"""
import argparse
import json
import math

from search_engine import (
    MODEL_NAME, PROCESSED, HybridSearch, KeywordSearch, SemanticSearch,
    load_corpus, load_qrels, load_queries,
)

K = 10


def metrics(ranked_ids: list[str], rel: dict[str, int]) -> dict:
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked_ids[:K]))
    ideal = sorted(rel.values(), reverse=True)[:K]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    hits = [d in rel for d in ranked_ids[:K]]
    rr = next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
    return {
        "nDCG@10": dcg / idcg if idcg else 0.0,
        "MRR@10": rr,
        "Recall@10": sum(hits) / len(rel) if rel else 0.0,
        "P@5": sum(hits[:5]) / 5,
    }


def evaluate(engine, qids, queries, qrels) -> dict:
    rows = [metrics([r["id"] for r in engine.search(queries[q], K)], qrels[q]) for q in qids]
    return {m: sum(r[m] for r in rows) / len(rows) for m in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[MODEL_NAME])
    ap.add_argument("--max-queries", type=int, default=200)
    ap.add_argument("--max-docs", type=int, default=0)
    args = ap.parse_args()

    qrels, queries = load_qrels(), load_queries()
    must_keep = {d for rel in qrels.values() for d in rel}
    docs = load_corpus(args.max_docs or None, must_keep)
    qids = [q for q in qrels if q in queries][: args.max_queries]
    print(f"{len(docs):,} docs · evaluating on {len(qids)} queries\n")

    keyword = KeywordSearch(docs)
    results = {"BM25": evaluate(keyword, qids, queries, qrels)}
    for name in args.models:
        sem = SemanticSearch(docs, name)
        results[f"Semantic · {name}"] = evaluate(sem, qids, queries, qrels)
        results[f"Hybrid   · {name}"] = evaluate(HybridSearch(sem, keyword), qids, queries, qrels)

    cols = list(next(iter(results.values())))
    width = max(len(n) for n in results) + 2
    print(f"{'Method':<{width}}" + "".join(f"{c:>11}" for c in cols))
    for name, res in results.items():
        print(f"{name:<{width}}" + "".join(f"{res[c]:>11.3f}" for c in cols))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "eval_results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to data/processed/eval_results.json")


if __name__ == "__main__":
    main()