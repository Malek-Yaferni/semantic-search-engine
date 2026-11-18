import os
import random

import streamlit as st

from search_engine import (
    MODEL_NAME, MODELS, HybridSearch, KeywordSearch, SemanticSearch,
    load_corpus, load_qrels, load_queries, timed,
)

st.set_page_config(page_title="Semantic Search Engine", page_icon="🔎", layout="wide")

# For very large corpora: MAX_DOCS=20000 streamlit run app.py
MAX_DOCS = int(os.getenv("MAX_DOCS", "0")) or None


@st.cache_resource(show_spinner="Loading dataset and building BM25 index...")
def get_data():
    qrels, queries = load_qrels(), load_queries()
    must_keep = {d for rel in qrels.values() for d in rel}
    docs = load_corpus(MAX_DOCS, must_keep)
    return docs, queries, qrels, KeywordSearch(docs)


@st.cache_resource(show_spinner="Loading model and FAISS index (first run per model can take minutes)...")
def get_semantic(model_name: str):
    docs, *_ = get_data()
    return SemanticSearch(docs, model_name)


docs, queries, qrels, keyword = get_data()
text_to_qid = {t: q for q, t in queries.items()}

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Embedding model", list(MODELS), index=list(MODELS).index(MODEL_NAME))
    k = st.slider("Results to show", 1, 10, 5)
    st.divider()

    st.header("Sample queries")
    st.caption("From the dataset, so relevant answers are known (✅).")
    if "samples" not in st.session_state:
        labelled = [q for q in qrels if q in queries]
        st.session_state["samples"] = random.sample(labelled, min(8, len(labelled)))
    for qid in st.session_state["samples"]:
        if st.button(queries[qid][:80], key=f"s_{qid}", use_container_width=True):
            st.session_state["query"] = queries[qid]
    if st.button("🔄 New samples"):
        del st.session_state["samples"]
        st.rerun()

semantic = get_semantic(model_name)
hybrid = HybridSearch(semantic, keyword)

st.title("🔎 Semantic vs Keyword vs Hybrid Search")
st.caption(
    f"{len(docs):,} documents · embeddings: {model_name} · "
    "vector index: FAISS · keyword baseline: BM25 · hybrid: Reciprocal Rank Fusion"
)

query = st.text_input("Search", key="query", placeholder="Type your own query...")


def render(results, latency_ms, relevant, empty_msg):
    st.caption(f"⏱ {latency_ms:.1f} ms")
    if relevant:
        found = sum(r["id"] in relevant for r in results)
        st.caption(f"✅ {found} of the top {len(results)} are labelled relevant")
    if not results:
        st.info(empty_msg)
        return
    for rank, r in enumerate(results, 1):
        mark = "✅ " if r["id"] in relevant else ""
        with st.container(border=True):
            st.markdown(f"**{rank}. {mark}{r['title'] or r['id']}**")
            body = r["content"]
            if len(body) > 250:
                st.write(body[:250] + "...")
                with st.expander("Read full abstract"):
                    st.write(body)
            else:
                st.write(body)
            st.caption(f"id: {r['id']} · score: {r['score']:.4f}")


if query.strip():
    qid = text_to_qid.get(query.strip())
    relevant = set(qrels.get(qid, {})) if qid else set()

    sem, sem_ms = timed(semantic.search, query, k)
    kw, kw_ms = timed(keyword.search, query, k)
    hyb, hyb_ms = timed(hybrid.search, query, k)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🧠 Semantic")
        render(sem, sem_ms, relevant, "No results.")
    with c2:
        st.subheader("🔤 Keyword (BM25)")
        render(kw, kw_ms, relevant, "No documents share words with this query.")
    with c3:
        st.subheader("⚡ Hybrid (RRF)")
        render(hyb, hyb_ms, relevant, "No results.")
    st.caption("Scores are not comparable across columns (cosine vs BM25 vs RRF); compare the rankings.")
else:
    st.info("Type a query or pick a sample from the sidebar.")