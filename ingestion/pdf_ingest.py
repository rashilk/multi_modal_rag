# ingestion/pdf_ingest.py
import fitz  # pymupdf
import os
import json
from ocr_utils import ocr_image_file

import argparse

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_image_bytes(image_bytes, out_path):
    with open(out_path, "wb") as f:
        f.write(image_bytes)

def extract_images_from_page(doc, page_num, out_dir):
    page = doc.load_page(page_num)
    image_list = page.get_images(full=True)
    saved = []
    for img_index, img_meta in enumerate(image_list):
        xref = img_meta[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        ext = base_image.get("ext", "png")
        img_name = f"page{page_num+1}_img{img_index}.{ext}"
        img_path = os.path.join(out_dir, img_name)
        save_image_bytes(image_bytes, img_path)
        ocr_text = ocr_image_file(img_path)
        saved.append({
            "img_path": img_path,
            "ocr_text": ocr_text,
            "xref": xref,
            "ext": ext
        })
    return saved

def extract_pdf(pdf_path, out_dir="ingested", save_json="ingested/pages.json", max_pages=None):
    ensure_dir(out_dir)
    doc = fitz.open(pdf_path)
    total = doc.page_count
    if max_pages:
        total = min(total, max_pages)
    pages_data = []
    for pno in range(total):
        page = doc.load_page(pno)
        text = page.get_text("text") or ""
        images_dir = os.path.join(out_dir, "images")
        ensure_dir(images_dir)
        images = extract_images_from_page(doc, pno, images_dir)
        page_record = {
            "page_number": pno+1,
            "text": text.strip(),
            "images": images
        }
        pages_data.append(page_record)
        print(f"Extracted page {pno+1}/{total}  |  chars:{len(text)}  |  images:{len(images)}")
    ensure_dir(os.path.dirname(save_json))
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(pages_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved ingestion output -> {save_json}")
    return pages_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF ingestion script (text + images + OCR)")
    parser.add_argument("--pdf", "-p", required=True, help="Path to PDF file")
    parser.add_argument("--out", "-o", default="ingested", help="Output folder")
    parser.add_argument("--json", "-j", default="ingested/pages.json", help="Output JSON file")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit number of pages")
    args = parser.parse_args()
    extract_pdf(
        pdf_path=args.pdf,
        out_dir=args.out,
        save_json=args.json,
        max_pages=args.max_pages
    )
