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
from .preprocess import prepare_cell_variants
from .schema import LONG_TEXT_FIELDS, NUMERIC_FIELDS, RATE_FIELDS
from .utils import NOISE_TOKENS, clean_whitespace, is_placeholder_blank, normalize_basic_text


def configure_tesseract() -> None:
    command = shutil.which("tesseract")
    fallback = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if command:
        pytesseract.pytesseract.tesseract_cmd = command
    elif fallback.exists():
        pytesseract.pytesseract.tesseract_cmd = str(fallback)


def _pad_and_resize(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
    gray = cv2.resize(gray, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
    return gray


def _cell_configs(field_name: str) -> list[str]:
    if field_name in NUMERIC_FIELDS:
        return ["--psm 7 -c tessedit_char_whitelist=0123456789.,-"]
    if field_name in RATE_FIELDS:
        return ["--psm 7", "--psm 6"]
    if field_name in LONG_TEXT_FIELDS:
        return ["--psm 7", "--psm 6"]
    return ["--psm 7"]


def _clean_cell_text(field_name: str, parts: list[str]) -> str:
    if field_name in NUMERIC_FIELDS:
        return "".join(parts)
    return normalize_basic_text(" ".join(parts))


def score_candidate(field_name: str, text: str, confidence: float | None) -> float:
    cleaned = clean_whitespace(text)
    if not cleaned or is_placeholder_blank(cleaned):
        return -10.0

    score = confidence or 0.0
    if field_name in NUMERIC_FIELDS:
        normalized = cleaned.replace(" ", "")
        if re.fullmatch(r"[-0-9.,;:]+", normalized):
            score += 1.5
        if any(character.isalpha() for character in normalized):
            score -= 1.5
        score += min(sum(character.isdigit() for character in normalized), 8) * 0.04
        if re.fullmatch(r"-?\d{1,3}(,\d{3})+[.;:]?", normalized):
            score += 0.8
        if re.fullmatch(r"-?\d{1,3}(\.\d{3})+[.;:]?", normalized):
            score -= 0.4
        if re.fullmatch(r"-?0[,.]\d{3}[.;:]?", normalized):
            score -= 1.2
    elif field_name in RATE_FIELDS:
        if re.search(r"do\.?|cent|per|ann|as\.?|a\.|p\.|rs\.?", cleaned, re.I):
            score += 1.0
        if sum(character.isalpha() for character in cleaned) < 2:
            score -= 0.6
    elif field_name == "article":
        alpha_count = sum(character.isalpha() for character in cleaned)
        score += min(alpha_count, 20) * 0.04
        if re.search(r"[A-Za-z]{3,}", cleaned):
            score += 0.8
        titlecase_words = len(re.findall(r"\b[A-Z][a-z]{2,}\b", cleaned))
        score += min(titlecase_words, 3) * 0.25
        if re.match(r"[A-Z]", cleaned):
            score += 0.15
        if re.match(r"[a-z]", cleaned):
            score -= 0.2
        if re.match(r"[^A-Za-z]", cleaned):
            score -= 0.35
        noise_chars = sum(character in "|~{}[]<>_=+" for character in cleaned)
        score -= min(noise_chars * 0.2, 1.0)
        if any(not character.isascii() for character in cleaned):
            score -= 0.15
        if cleaned.startswith(("|", ".", ",")):
            score -= 0.4
    else:
        if re.fullmatch(r"[A-Za-z0-9 .,';:()-]+", cleaned):
            score += 0.3
    return score


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


def _ocr_cell_tesseract_candidates(image: np.ndarray, field_name: str) -> list[OCRCellResult]:
    configure_tesseract()
    candidates: list[OCRCellResult] = []
    for variant_name, variant_image in prepare_cell_variants(image, field_name):
        prepared = _pad_and_resize(variant_image)
        pil_image = Image.fromarray(prepared)
        for config in _cell_configs(field_name):
            data = pytesseract.image_to_data(
                pil_image,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
            parts: list[str] = []
            confidences: list[float] = []
            for index, raw_text in enumerate(data["text"]):
                text = clean_whitespace(raw_text)
                if not text or is_placeholder_blank(text):
                    continue
                conf_raw = data["conf"][index]
                conf = float(conf_raw) if conf_raw != "-1" else -1.0
                if conf < 10:
                    continue
                parts.append(text)
                confidences.append(conf / 100.0)
            candidates.append(
                OCRCellResult(
                    text=_clean_cell_text(field_name, parts),
                    confidence=(sum(confidences) / len(confidences)) if confidences else None,
                    engine=f"tesseract:{variant_name}",
                )
            )
    return candidates


def _filter_nonempty_candidates(candidates: list[OCRCellResult]) -> list[OCRCellResult]:
    return [
        candidate
        for candidate in candidates
        if clean_whitespace(candidate.text) and not is_placeholder_blank(candidate.text)
    ]


class OCRBackendChain:
    """Backend chain with optional Paddle/Tesseract ensembling."""

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

    def _paddle_tokens(self, image: np.ndarray) -> list[OCRToken]:
        if not self._paddle_ready:
            return []
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
        return tokens

    def table_tokens(self, image: np.ndarray) -> tuple[list[OCRToken], str]:
        tesseract_result = tesseract_tokens(image)
        if self.backend_name in {"auto", "paddle"}:
            self._init_paddle()
        if self._paddle_ready:
            try:
                paddle_result = self._paddle_tokens(image)
                if self.backend_name == "paddle":
                    return paddle_result, "paddle"
                if len(paddle_result) >= len(tesseract_result):
                    return paddle_result, "paddle"
            except Exception as exc:
                logging.warning("Paddle table OCR failed, switching to Tesseract: %s", exc)
        return tesseract_result, "tesseract"

    def _paddle_cell_candidate(self, image: np.ndarray, field_name: str) -> OCRCellResult | None:
        if not self._paddle_ready:
            return None
        try:
            result = self._paddle.predict(image)
            page_result = result[0]
            texts = [
                normalize_basic_text(str(item))
                for item in page_result["rec_texts"]
                if clean_whitespace(str(item))
            ]
            if not texts:
                return None
            if field_name in NUMERIC_FIELDS:
                text = "".join(texts)
            else:
                text = " ".join(texts)
            scores = [float(item) for item in page_result["rec_scores"]]
            return OCRCellResult(
                text=text,
                confidence=(sum(scores) / len(scores)) if scores else None,
                engine="paddle",
            )
        except Exception as exc:
            logging.warning("Paddle cell OCR failed, switching to Tesseract: %s", exc)
            return None

    def cell_text(self, image: np.ndarray, field_name: str, ensemble: bool = False) -> OCRCellResult:
        if self.backend_name in {"auto", "paddle"}:
            self._init_paddle()

        candidates = _ocr_cell_tesseract_candidates(image, field_name)
        if self._paddle_ready and (ensemble or self.backend_name == "paddle"):
            paddle_candidate = self._paddle_cell_candidate(image, field_name)
            if paddle_candidate is not None:
                candidates.append(paddle_candidate)

        nonempty = _filter_nonempty_candidates(candidates)
        if not nonempty:
            return OCRCellResult(text="", confidence=None, engine="none")

        best = max(
            nonempty,
            key=lambda candidate: score_candidate(field_name, candidate.text, candidate.confidence),
        )
        return best


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
