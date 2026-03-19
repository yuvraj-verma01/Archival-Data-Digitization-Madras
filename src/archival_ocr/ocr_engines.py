"""OCR backends used by the pipeline."""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

from .models import OCRCellResult, OCRToken
from .schema import NUMERIC_FIELDS, RATE_FIELDS
from .utils import NOISE_TOKENS, clean_whitespace, is_placeholder_blank, normalize_basic_text


def configure_tesseract() -> None:
    command = shutil.which("tesseract")
    fallback = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if command:
        pytesseract.pytesseract.tesseract_cmd = command
    elif fallback.exists():
        pytesseract.pytesseract.tesseract_cmd = str(fallback)


def _prepare_image(image: np.ndarray, numeric: bool) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
    gray = cv2.resize(gray, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
    if numeric:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return gray


def _cell_config(field_name: str) -> str:
    if field_name in NUMERIC_FIELDS:
        return "--psm 7 -c tessedit_char_whitelist=0123456789.,-"
    return "--psm 7"


def tesseract_tokens(image: np.ndarray) -> list[OCRToken]:
    configure_tesseract()
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    data = pytesseract.image_to_data(gray, config="--psm 6", output_type=pytesseract.Output.DICT)
    tokens: list[OCRToken] = []
    for index, text in enumerate(data["text"]):
        text = clean_whitespace(text)
        if not text:
            continue
        confidence_raw = data["conf"][index]
        confidence = float(confidence_raw) if confidence_raw != "-1" else -1.0
        if confidence < 15:
            continue
        tokens.append(
            OCRToken(
                text=text,
                confidence=confidence / 100.0,
                x=int(data["left"][index]),
                y=int(data["top"][index]),
                width=int(data["width"][index]),
                height=int(data["height"][index]),
            )
        )
    return tokens


def ocr_cell_tesseract(image: np.ndarray, field_name: str) -> OCRCellResult:
    configure_tesseract()
    prepared = _prepare_image(image, numeric=field_name in NUMERIC_FIELDS)
    pil_image = Image.fromarray(prepared)
    config = _cell_config(field_name)
    data = pytesseract.image_to_data(pil_image, config=config, output_type=pytesseract.Output.DICT)
    parts: list[str] = []
    confidences: list[float] = []
    for index, raw_text in enumerate(data["text"]):
        text = clean_whitespace(raw_text)
        if not text:
            continue
        if is_placeholder_blank(text):
            continue
        conf_raw = data["conf"][index]
        conf = float(conf_raw) if conf_raw != "-1" else -1.0
        if conf < 10:
            continue
        parts.append(text)
        confidences.append(conf / 100.0)
    if field_name in NUMERIC_FIELDS:
        text = "".join(parts)
    else:
        text = normalize_basic_text(" ".join(parts))
    mean_conf = sum(confidences) / len(confidences) if confidences else None
    return OCRCellResult(text=text, confidence=mean_conf, engine="tesseract")


class OCRBackendChain:
    """Backend chain with a guarded Paddle attempt and stable Tesseract fallback."""

    def __init__(
        self,
        backend_name: str,
        workspace_root: Path,
        allow_paddle: bool = False,
    ) -> None:
        self.backend_name = backend_name
        self.workspace_root = workspace_root
        self.allow_paddle = allow_paddle
        self._paddle = None
        self._paddle_ready = False
        self._paddle_attempted = False

    def _init_paddle(self) -> None:
        if self._paddle_ready or self._paddle_attempted or not self.allow_paddle:
            return
        self._paddle_attempted = True
        try:
            os.environ.setdefault(
                "PADDLE_PDX_CACHE_HOME",
                str((self.workspace_root / ".paddlex_cache").resolve()),
            )
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import PaddleOCR

            self._paddle = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang="en",
            )
            self._paddle_ready = True
            logging.info("PaddleOCR initialized successfully.")
        except Exception as exc:
            self._paddle_ready = False
            self._paddle = None
            logging.warning("PaddleOCR unavailable, using Tesseract fallback: %s", exc)

    def table_tokens(self, image: np.ndarray) -> tuple[list[OCRToken], str]:
        if self.backend_name in {"auto", "paddle"}:
            self._init_paddle()
        if self._paddle_ready:
            try:
                result = self._paddle.predict(image)
                page_result = result[0]
                tokens: list[OCRToken] = []
                for text, score, points in zip(
                    page_result["rec_texts"],
                    page_result["rec_scores"],
                    page_result["dt_polys"],
                ):
                    xs = [int(point[0]) for point in points]
                    ys = [int(point[1]) for point in points]
                    tokens.append(
                        OCRToken(
                            text=clean_whitespace(str(text)),
                            confidence=float(score),
                            x=min(xs),
                            y=min(ys),
                            width=max(xs) - min(xs),
                            height=max(ys) - min(ys),
                        )
                    )
                return tokens, "paddle"
            except Exception as exc:
                logging.warning("Paddle table OCR failed, switching to Tesseract: %s", exc)
        return tesseract_tokens(image), "tesseract"

    def cell_text(self, image: np.ndarray, field_name: str) -> OCRCellResult:
        if self.backend_name in {"auto", "paddle"}:
            self._init_paddle()
        if self._paddle_ready:
            try:
                result = self._paddle.predict(image)
                page_result = result[0]
                texts = [normalize_basic_text(str(item)) for item in page_result["rec_texts"]]
                scores = [float(item) for item in page_result["rec_scores"]]
                if field_name in NUMERIC_FIELDS:
                    text = "".join(texts)
                else:
                    text = " ".join(texts)
                if text:
                    return OCRCellResult(
                        text=text,
                        confidence=(sum(scores) / len(scores)) if scores else None,
                        engine="paddle",
                    )
            except Exception as exc:
                logging.warning("Paddle cell OCR failed, switching to Tesseract: %s", exc)
        return ocr_cell_tesseract(image, field_name)


def normalize_joined_tokens(field_name: str, texts: list[str]) -> str:
    cleaned = [
        clean_whitespace(text)
        for text in texts
        if clean_whitespace(text) and not is_placeholder_blank(text)
    ]
    if field_name in NUMERIC_FIELDS:
        cleaned = [text for text in cleaned if any(character.isdigit() for character in text)]
        return "".join(cleaned)
    filtered: list[str] = []
    for text in cleaned:
        lowered = text.lower()
        if lowered in NOISE_TOKENS:
            continue
        filtered.append(text)
    if field_name in RATE_FIELDS:
        return normalize_basic_text(" ".join(filtered))
    return normalize_basic_text(" ".join(filtered))


def parse_integer_candidate(text: str, upper_bound: int) -> int | None:
    digits = re.findall(r"\d+", text)
    if len(digits) != 1:
        return None
    value = int(digits[0])
    if 1 <= value <= upper_bound:
        return value
    return None
