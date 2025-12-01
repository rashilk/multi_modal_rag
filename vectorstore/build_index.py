# vectorstore/build_index.py
import json
import os
import numpy as np

def load_embeddings(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_index(embeddings_path="ingested/embeddings_test.json", out_index="ingested/index.npy", out_meta="ingested/metadata.json"):
    data = load_embeddings(embeddings_path)
    vectors = []
    metas = []
    for item in data:
        vec = item.get("embedding")
        if vec is None:
            continue
        vectors.append(np.array(vec, dtype=np.float32))
        metas.append({
            "chunk_id": item.get("chunk_id"),
            "page": item.get("page"),
            "type": item.get("type"),
            "text": item.get("text")[:800]  # store a snippet for quick preview
        })
    if not vectors:
        raise ValueError("No vectors found in embeddings file.")
    mat = np.vstack(vectors)  # shape (N, D)
    # normalize rows for cosine similarity
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    mat = mat / norms
    # save
    os.makedirs(os.path.dirname(out_index), exist_ok=True)
    np.save(out_index, mat)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metas, f, indent=2, ensure_ascii=False)
    print(f"Saved index -> {out_index} (shape: {mat.shape})")
    print(f"Saved metadata -> {out_meta}")
    return mat, metas

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", "-e", default="ingested/embeddings_test.json")
    parser.add_argument("--index", "-i", default="ingested/index.npy")
    parser.add_argument("--meta", "-m", default="ingested/metadata.json")
    args = parser.parse_args()
    build_index(args.emb, args.index, args.meta)
