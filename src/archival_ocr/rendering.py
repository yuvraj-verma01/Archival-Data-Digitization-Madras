"""PDF rendering helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import fitz
import numpy as np


def render_page(pdf_path: Path, page_number: int, dpi: int) -> np.ndarray:
    """Render a 1-based PDF page to a BGR numpy image."""
    with fitz.open(pdf_path) as document:
        if page_number < 1 or page_number > document.page_count:
            raise ValueError(
                f"Requested page {page_number} but document only has {document.page_count} pages."
            )
        page = document[page_number - 1]
        scale = dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
        if pixmap.n == 4:
            return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
