# utils/chunker.py
import json
import os
import re
from typing import List, Dict

def read_ingested_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def words_count(s: str) -> int:
    return len(s.split())

def chunk_text(text: str, page_number: int, chunk_size_words: int = 300) -> List[Dict]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size_words):
        chunk_words = words[i:i+chunk_size_words]
        chunk_text = " ".join(chunk_words).strip()
        if not chunk_text:
            continue
        chunks.append({
            "page": page_number,
            "type": "text",
            "text": chunk_text,
            "word_count": len(chunk_words)
        })
    return chunks

def chunk_table_like_text(table_text: str, page_number: int) -> Dict:
    # keep tables as single chunk (if any table text)
    return {
        "page": page_number,
        "type": "table",
        "text": table_text,
        "word_count": words_count(table_text)
    }

def chunk_images(imgs: List[Dict], page_number: int) -> List[Dict]:
    chunks = []
    for idx, img in enumerate(imgs):
        ocr = img.get("ocr_text", "") or ""
        # short crop for snippet
        snippet = (ocr[:1000] + "...") if len(ocr) > 1000 else ocr
        chunks.append({
            "page": page_number,
            "type": "image",
            "image_path": img.get("img_path"),
            "text": snippet,
            "full_ocr_text": ocr,
            "word_count": words_count(ocr)
        })
    return chunks

def create_chunks_from_ingested(ingested_json_path: str,
                                out_chunks_path: str = "ingested/chunks.json",
                                chunk_size_words: int = 300):
    data = read_ingested_json(ingested_json_path)
    all_chunks = []
    for p in data:
        pno = p.get("page_number")
        text = p.get("text", "") or ""
        images = p.get("images", []) or []

        # Chunk main text
        text_chunks = chunk_text(text, pno, chunk_size_words=chunk_size_words)
        all_chunks.extend(text_chunks)

        # Images (OCR text) as separate chunks
        img_chunks = chunk_images(images, pno)
        all_chunks.extend(img_chunks)

    # Save chunks
    os.makedirs(os.path.dirname(out_chunks_path), exist_ok=True)
    with open(out_chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Created {len(all_chunks)} chunks -> {out_chunks_path}")
    return all_chunks

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingested", "-i", default="ingested/pages.json", help="Path to ingested JSON")
    parser.add_argument("--out", "-o", default="ingested/chunks.json", help="Output chunks JSON")
    parser.add_argument("--size", "-s", type=int, default=300, help="Chunk size in words")
    args = parser.parse_args()
    create_chunks_from_ingested(args.ingested, out_chunks_path=args.out, chunk_size_words=args.size)
