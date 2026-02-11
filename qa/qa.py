# qa/qa.py
import os
import sys
import pathlib
import traceback

# Ensure project root is on path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from vectorstore.retriever import cosine_search

# OpenAI optional
OPENAI_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# =========================
# PROMPT BUILDER (STRICT)
# =========================
def build_prompt(question, hits):
    if not hits:
        return None

    context_chunks = []
    for i, h in enumerate(hits):
        page = h["meta"].get("page", "N/A")
        text = h["meta"].get("text", "").strip()
        context_chunks.append(f"Source {i+1} (page {page}):\n{text}")

    context_block = "\n\n".join(context_chunks)

    SYSTEM_PROMPT = """
You are a strict document-based question answering assistant.

RULES (MANDATORY):
- Answer ONLY using the provided context.
- Do NOT use outside knowledge.
- Do NOT guess or infer.
- If the answer is NOT explicitly present, reply exactly:
  "The answer is not available in the provided documents."
- Keep the answer concise and factual.
"""

    prompt = f"""
{SYSTEM_PROMPT}

====================
CONTEXT:
{context_block}
====================

QUESTION:
{question}

ANSWER:
"""

    return prompt


# =========================
# OPENAI CALL (GUARDED)
# =========================
def call_openai_chat(prompt, model="gpt-4o-mini", max_tokens=300):
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None

        openai.api_key = api_key

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer strictly from provided context. No hallucination."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )

        return response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print("OpenAI error:", e)
        return None


# =========================
# FALLBACK (EXTRACTIVE)
# =========================
def extractive_fallback(hits):
    parts = []
    for h in hits[:3]:
        page = h["meta"].get("page", "N/A")
        text = h["meta"].get("text", "").strip().replace("\n", " ")
        parts.append(f"(page {page}) {text[:500]}")
    return " ".join(parts)


# =========================
# MAIN QA PIPELINE
# =========================
def answer_question(question, k=5, prefer_openai=False):
    # 1. Retrieve
    hits = cosine_search(question, k=k)

    # Guard 1: No retrieval
    if not hits:
        return {
            "answer": "The answer is not available in the provided documents.",
            "sources": []
        }

    # Guard 2: Similarity threshold
    SIMILARITY_THRESHOLD = 0.75
    filtered_hits = [h for h in hits if h.get("score", 0) >= SIMILARITY_THRESHOLD]

    if not filtered_hits:
        return {
            "answer": "The answer is not available in the provided documents.",
            "sources": []
        }

    # 2. Build strict prompt
    prompt = build_prompt(question, filtered_hits)

    # 3. LLM answer
    if OPENAI_AVAILABLE and prefer_openai:
        answer = call_openai_chat(prompt)
        if answer:
            return {"answer": answer, "sources": filtered_hits}

    if OPENAI_AVAILABLE:
        answer = call_openai_chat(prompt)
        if answer:
            return {"answer": answer, "sources": filtered_hits}

    # 4. Fallback extractive
    fallback = extractive_fallback(filtered_hits)
    return {"answer": fallback, "sources": filtered_hits}


# =========================
# CLI TESTING
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--q", "-q", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--openai", action="store_true")
    args = parser.parse_args()

    result = answer_question(args.q, k=args.k, prefer_openai=args.openai)

    print("\n=== ANSWER ===\n")
    print(result["answer"])

    print("\n=== SOURCES ===\n")
    for i, s in enumerate(result["sources"], 1):
        print(f"{i}. page {s['meta'].get('page')} | score={s['score']:.4f}")
