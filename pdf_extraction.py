"""PDF ingestion, type detection, and text extraction."""

import base64
import io
from typing import Literal

import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image

from config import OCR_DPI, SAMPLE_PAGES, SCANNED_THRESHOLD
from llm import extract_text_from_page_image

RoutingMode = Literal["text", "scanned"]


def detect_pdf_type(pdf_bytes: bytes) -> tuple[RoutingMode, int, float]:
    """
    Detect whether PDF is text-based or scanned/image-based.
    Returns (mode, page_count, avg_chars_per_page).
    """
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page_count = len(pdf.pages)
        pages_to_sample = min(SAMPLE_PAGES, page_count)
        char_counts = []

        for i in range(pages_to_sample):
            text = pdf.pages[i].extract_text() or ""
            char_counts.append(len(text.strip()))

        avg_chars = sum(char_counts) / max(pages_to_sample, 1)
        mode: RoutingMode = "scanned" if avg_chars < SCANNED_THRESHOLD else "text"
        return mode, page_count, avg_chars


def extract_text_pdf(pdf_bytes: bytes) -> str:
    """Extract full text from a normal text-based PDF."""
    parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text.strip())
    return "\n\n".join(parts)


def _image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85)
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


def extract_scanned_pdf(pdf_bytes: bytes) -> str:
    """Extract text from scanned PDF using MiniMax-M3 vision page by page."""
    images = convert_from_bytes(pdf_bytes, dpi=OCR_DPI, fmt="jpeg")
    page_texts = []

    for idx, image in enumerate(images, start=1):
        b64 = _image_to_base64(image)
        text = extract_text_from_page_image(b64, idx)
        page_texts.append(text.strip())

    return "\n\n".join(page_texts)


def extract_document(pdf_bytes: bytes, routing_mode: RoutingMode) -> str:
    """Route extraction based on detected PDF type."""
    if routing_mode == "scanned":
        return extract_scanned_pdf(pdf_bytes)
    return extract_text_pdf(pdf_bytes)


def is_solar_related(text: str) -> bool:
    """Heuristic check for solar/tender-related content."""
    keywords = [
        "solar", "pv", "photovoltaic", "mwp", "mwh", "seci", "mnre",
        "tender", "rfp", "epc", "inverter", "module", "grid",
        "renewable", "rooftop", "ground mount", "bid", "nit",
    ]
    lower = text.lower()
    matches = sum(1 for kw in keywords if kw in lower)
    return len(text.strip()) > 200 and matches >= 2
