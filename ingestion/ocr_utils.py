# ingestion/ocr_utils.py
import pytesseract
from PIL import Image, ImageOps, ImageFilter
import os
import sys

def configure_tesseract():
    if sys.platform.startswith("win"):
        possible = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ]
        for p in possible:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p

def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    img = image.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    w, h = img.size
    if w < 800:
        img = img.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
    return img

def ocr_image_file(img_path: str, lang="eng"):
    configure_tesseract()
    try:
        img = Image.open(img_path)
        processed = preprocess_image_for_ocr(img)
        return pytesseract.image_to_string(processed, lang=lang)
    except Exception as e:
        return f"[OCR ERROR] {e}"
