# embeddings/embedder.py

import json
import os
from fastembed import TextEmbedding

model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

def load_chunks(chunks_path):
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)

def embed_text(text):
    # Fastembed returns a generator → convert to list
    emb = list(model.embed([text]))[0]
    return emb.tolist()

def create_embeddings(chunks_path, out_path="ingested/embeddings.json"):
    chunks = load_chunks(chunks_path)
    embedded = []

    for idx, ch in enumerate(chunks):
        content = ch.get("text", "")
        if not content.strip():
            continue

        vector = embed_text(content)

        embedded.append({
            "chunk_id": idx,
            "page": ch.get("page"),
            "type": ch.get("type"),
            "text": content,
            "embedding": vector
        })

        print(f"Embedded {idx+1}/{len(chunks)}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(embedded, f, indent=2)

    print(f"\nSaved embeddings → {out_path}")
    return embedded

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", "-c", required=True)
    parser.add_argument("--out", "-o", default="ingested/embeddings.json")
    args = parser.parse_args()

    create_embeddings(args.chunks, args.out)
