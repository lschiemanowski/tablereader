# pyright: basic
"""Typed wrapper around the parts of pypdfium2 we use (pypdfium2 ships incomplete type information)."""

from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.version as pdfium_version
from PIL import Image

RENDERER: str = f"pypdfium2 {pdfium_version.PYPDFIUM_INFO} / pdfium {pdfium_version.PDFIUM_INFO}"


def render(path: Path, page: int, scale: float) -> tuple[Image.Image, tuple[float, float]]:
    """Page `page` (from 1) as displayed, as RGB at `scale` pixels per point, and its displayed size in points."""
    pdf = pdfium.PdfDocument(path)
    try:
        if not 1 <= page <= len(pdf):
            raise ValueError(f"{path} has no page {page}")
        pdf_page = pdf[page - 1]
        try:
            width, height = pdf_page.get_size()
            image = pdf_page.render(scale=scale).to_pil()  # pyright: ignore[reportArgumentType]  (annotated int, takes float).convert("RGB")
        finally:
            pdf_page.close()
    finally:
        pdf.close()
    return image, (float(width), float(height))


def page_count(path: Path) -> int:
    pdf = pdfium.PdfDocument(path)
    try:
        return len(pdf)
    finally:
        pdf.close()
