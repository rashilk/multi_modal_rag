# qa/qa.py
import os
import json
import traceback

# ensure project root on path
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from vectorstore.retriever import cosine_search
import numpy as np

# Try to import OpenAI client if available
OPENAI_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

def build_prompt(question, hits):
    """
    Build a concise prompt containing the top-k contexts with page citations.
    """
    ctxs = []
    for i, h in enumerate(hits):
        page = h["meta"].get("page")
        text = h["meta"].get("text", "")
        ctxs.append(f"Source {i+1} (page {page}):\n{text}\n")
    context_block = "\n\n".join(ctxs)

    prompt = (
        "You are an assistant that answers questions using ONLY the provided sources.\n"
        "If the answer cannot be found in the sources, say 'Not found in the document.'\n\n"
        "Use the context below and then answer the question.\n\n"
        "CONTEXT:\n"
        f"{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely, and at the end list the sources you referenced like: (page X)."
    )
    return prompt

def call_openai_chat(prompt, model="gpt-4o-mini", max_tokens=400):
    """
    Call OpenAI ChatCompletion (wrapped) — returns answer text.
    If OpenAI raises any error (invalid key, quota), return None.
    """
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"system","content":"You are a concise, factual assistant."},
                      {"role":"user","content":prompt}],
            max_tokens=max_tokens,
            temperature=0.0
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        # print stack for debugging but return None so fallback can run
        print("OpenAI call failed:", e)
        # uncomment next line if you want full trace in terminal
        # traceback.print_exc()
        return None

def simple_extractive_answer(hits):
    """
    Minimal fallback: stitch top 2-3 hits and return as answer with citations.
    """
    parts = []
    for i, h in enumerate(hits[:3]):
        page = h["meta"].get("page")
        text = h["meta"].get("text", "")
        snippet = text.strip().replace("\n", " ")
        if len(snippet) > 800:
            snippet = snippet[:800] + "..."
        parts.append(f"(page {page}) {snippet}")
    return " ".join(parts)

def answer_question(question, k=5, prefer_openai=False):
    # 1) retrieve
    hits = cosine_search(question, k=k)
    # 2) build prompt
    prompt = build_prompt(question, hits)
    # 3) try OpenAI if available and requested
    if OPENAI_AVAILABLE and prefer_openai:
        ans = call_openai_chat(prompt)
        if ans:
            return {"answer": ans, "sources": hits}
    # 4) try OpenAI if available (even if not preferred)
    if OPENAI_AVAILABLE:
        ans = call_openai_chat(prompt)
        if ans:
            return {"answer": ans, "sources": hits}
    # 5) fallback to extractive
    fallback = simple_extractive_answer(hits)
    return {"answer": fallback, "sources": hits}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", "-q", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--openai", action="store_true", help="Try OpenAI for final answer")
    args = parser.parse_args()
    out = answer_question(args.q, k=args.k, prefer_openai=args.openai)
    print("\n=== Answer ===\n")
    print(out["answer"])
    print("\n=== Sources ===\n")
    for i, s in enumerate(out["sources"]):
        print(f"{i+1}. page {s['meta']['page']} (score={s['score']:.4f})")
