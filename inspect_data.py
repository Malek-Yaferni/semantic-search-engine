"""Run this first to check the dataset is read correctly:  python inspect_data.py"""
from search_engine import load_corpus, load_queries, load_qrels

corpus, queries, qrels = load_corpus(), load_queries(), load_qrels()
print(f"Corpus : {len(corpus):,} documents")
print(f"Queries: {len(queries):,}")
print(f"Qrels  : {len(qrels):,} queries with labels, "
      f"{sum(len(v) for v in qrels.values()):,} relevant pairs")
print("\nSample doc  :", {k: str(v)[:100] for k, v in corpus[0].items()})
q = next(iter(queries))
print("Sample query:", q, "->", queries[q])
print("Its qrels   :", qrels.get(q))