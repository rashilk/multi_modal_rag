# vectorstore/retriever.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import json
from typing import List, Dict
from embeddings.embedder import embed_text  # uses fastembed or chosen embedder

_INDEX_PATH = "ingested/index.npy"
_META_PATH = "ingested/metadata.json"

def load_index(index_path=_INDEX_PATH, meta_path=_META_PATH):
    mat = np.load(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metas = json.load(f)
    return mat, metas

def cosine_search(query: str, k: int = 5):
    mat, metas = load_index()
    qv = np.array(embed_text(query), dtype=np.float32)
    # normalize query
    qv_norm = np.linalg.norm(qv)
    if qv_norm == 0:
        qv_norm = 1e-10
    qv = qv / qv_norm
    # compute cosine similarities using dot product (mat rows are normalized)
    sims = mat.dot(qv)
    # top k
    idxs = np.argsort(-sims)[:k]
    results = []
    for idx in idxs:
        results.append({
            "score": float(sims[idx]),
            "meta": metas[idx]
        })
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", "-q", required=True, help="Query string")
    parser.add_argument("--k", type=int, default=5, help="Top k")
    args = parser.parse_args()
    res = cosine_search(args.q, k=args.k)
    for i, r in enumerate(res):
        print(f"\n--- Rank {i+1} (score={r['score']:.4f}) ---")
        print(f"page: {r['meta']['page']} type: {r['meta']['type']}")
        print(r['meta']['text'][:600].replace("\n"," "))
