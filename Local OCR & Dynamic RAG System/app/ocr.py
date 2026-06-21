"""
OCR module — extracts text from uploaded documents entirely locally,
no external APIs.

Strategy:
- Images (PNG/JPG/etc): run straight through Tesseract OCR (ben+eng).
- PDFs: try extracting embedded digital text first (PyMuPDF) — this is
  instant and 100% accurate for "born-digital" PDFs (e.g. exported from
  Word). Only if that yields near-empty text (a strong signal the PDF is
  actually a scan with no text layer) do we fall back to rendering each
  page as an image and running Tesseract OCR on it. This avoids paying
  the OCR accuracy/speed cost on documents that don't need it.
"""

import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

TESSERACT_LANGS = "ben+eng"
MIN_DIGITAL_TEXT_CHARS = 20  # below this, assume the PDF page is a scan


def _ocr_image(image: Image.Image) -> str:
    return pytesseract.image_to_string(image, lang=TESSERACT_LANGS)


def extract_text_from_image_bytes(file_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(file_bytes))
    text = _ocr_image(image)
    return {"text": text, "method": "ocr", "pages": 1}


def extract_text_from_pdf_bytes(file_bytes: bytes) -> dict:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page_texts = []
    methods_used = set()

    for page in doc:
        digital_text = page.get_text().strip()

        if len(digital_text) >= MIN_DIGITAL_TEXT_CHARS:
            page_texts.append(digital_text)
            methods_used.add("digital")
        else:
            # No usable embedded text — render the page as an image and OCR it
            pix = page.get_pixmap(dpi=300)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            page_texts.append(_ocr_image(image))
            methods_used.add("ocr")

    doc.close()
    return {
        "text": "\n\n".join(page_texts),
        "method": "+".join(sorted(methods_used)) or "none",
        "pages": len(page_texts),
    }


def extract_text(file_bytes: bytes, filename: str) -> dict:
    """Routes to the right extractor based on file extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf_bytes(file_bytes)
    elif lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")):
        return extract_text_from_image_bytes(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
