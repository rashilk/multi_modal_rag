# app/streamlit_app.py
import sys, os, pathlib
# make project root importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import streamlit as st
from qa.qa import answer_question
from vectorstore.retriever import cosine_search

st.set_page_config(page_title="Multi-modal RAG — Big Air Lab", layout="wide")

st.title("Multi-modal RAG — Big Air Lab (Demo)")
st.markdown("Ask questions about the uploaded IMF Qatar report. Uses local embeddings + retrieval. OpenAI optional for final answer if you have an API key.")

with st.sidebar:
    st.header("Quick actions")
    uploaded = st.file_uploader("(Optional) Upload new PDF to ingest", type=["pdf"])
    use_openai = st.checkbox("Prefer OpenAI for answer (requires OPENAI_API_KEY)", value=False)
    k = st.slider("Top-k retrieval", min_value=1, max_value=8, value=3)
    st.markdown("---")
    st.caption("Workflow: ingestion → chunking → embeddings → index → QA. If you already ran ingestion & built index, upload is optional.")

if uploaded:
    st.warning("Upload feature is for demo only. Re-ingestion in the UI isn't implemented in this demo. Please run ingestion scripts in terminal if you want to index a new file.")

q = st.text_input("Ask a question about the document", placeholder="e.g. What is Qatar's projected GDP growth for 2024-25?")
ask = st.button("Get Answer")

if ask and not q.strip():
    st.error("Please enter a question.")
elif ask:
    with st.spinner("Retrieving..."):
        # Run retrieval and QA
        try:
            # first show retrieval hits
            hits = cosine_search(q, k=k)
        except Exception as e:
            st.error(f"Retrieval failed: {e}")
            hits = []

        st.subheader("Top retrieved snippets")
        for i, h in enumerate(hits):
            meta = h.get("meta", {})
            score = h.get("score", 0.0)
            st.markdown(f"**{i+1}. (page {meta.get('page')}) — score: {score:.4f}**")
            snippet = meta.get("text", "")
            st.write(snippet)

        st.subheader("Answer")
        try:
            res = answer_question(q, k=k, prefer_openai=use_openai)
            st.success(res["answer"])
            st.markdown("**Sources used (top-k):**")
            for i, s in enumerate(res["sources"]):
                st.write(f"{i+1}. page {s['meta']['page']} (score={s['score']:.4f})")
        except Exception as e:
            st.error(f"QA failed: {e}")

st.markdown("---")
st.caption("Demo built for the Big Air Lab assignment. Code: ingestion, chunking, embeddings, index, retrieval, QA, and this Streamlit UI.")


