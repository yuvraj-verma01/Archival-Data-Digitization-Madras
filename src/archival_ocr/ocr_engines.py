"""OCR backends used by the pipeline."""

from __future__ import annotations

import base64
import logging
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

try:
    import requests
except Exception:
    requests = None

try:
    from rapidfuzz import fuzz, process
except Exception:
    fuzz = None
    process = None

from .models import OCRCellResult, OCRToken
from .preprocess import prepare_cell_variants
from .schema import LONG_TEXT_FIELDS, NUMERIC_FIELDS, RATE_FIELDS
from .utils import NOISE_TOKENS, clean_whitespace, is_placeholder_blank, normalize_basic_text


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
DEEPSEEK_MODEL = os.environ.get("OLLAMA_DEEPSEEK_OCR_MODEL", "deepseek-ocr")
GLM_MODEL = os.environ.get("OLLAMA_GLM_OCR_MODEL", "glm-ocr")
OLLAMA_TIMEOUT = 60

_PADDLE_INSTANCE = None
_PADDLE_READY = False
_PADDLE_ATTEMPTED = False
_OLLAMA_UNAVAILABLE_MODELS: set[str] = set()


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


def _ensure_paddle_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _paddle_init_kwargs() -> dict[str, object]:
    # Windows CPU inference is more stable here without the MKLDNN/HPI/CINN path.
    return {
        "lang": "en",
        "device": "cpu",
        "enable_hpi": False,
        "enable_mkldnn": False,
        "enable_cinn": False,
    }


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


def _field_type_for(field_name: str) -> str | None:
    if field_name == "article":
        return "text"
    if field_name in NUMERIC_FIELDS:
        return "numeric"
    return None


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
        alpha_words = re.findall(r"[A-Za-z]+", cleaned)
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
        if len(alpha_words) > 5:
            score -= min((len(alpha_words) - 5) * 0.35, 2.0)
        if len(cleaned) > 40:
            score -= min((len(cleaned) - 40) * 0.04, 2.0)
    else:
        if re.fullmatch(r"[A-Za-z0-9 .,';:()-]+", cleaned):
            score += 0.3
    return score


def run_tesseract(image: np.ndarray, config: str) -> str:
    configure_tesseract()
    prepared = _pad_and_resize(image)
    pil_image = Image.fromarray(prepared)
    data = pytesseract.image_to_data(
        pil_image,
        config=config,
        output_type=pytesseract.Output.DICT,
    )
    parts: list[str] = []
    for index, raw_text in enumerate(data["text"]):
        text = clean_whitespace(raw_text)
        if not text or is_placeholder_blank(text):
            continue
        conf_raw = data["conf"][index]
        conf = float(conf_raw) if conf_raw != "-1" else -1.0
        if conf < 10:
            continue
        parts.append(text)
    return normalize_basic_text(" ".join(parts))


def _init_global_paddle() -> None:
    global _PADDLE_INSTANCE, _PADDLE_READY, _PADDLE_ATTEMPTED
    if _PADDLE_READY or _PADDLE_ATTEMPTED:
        return
    _PADDLE_ATTEMPTED = True
    try:
        os.environ.setdefault(
            "PADDLE_PDX_CACHE_HOME",
            str((Path.cwd() / ".paddlex_cache").resolve()),
        )
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        _PADDLE_INSTANCE = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            **_paddle_init_kwargs(),
        )
        _PADDLE_READY = True
        logging.info("PaddleOCR initialized successfully.")
    except Exception as exc:
        _PADDLE_INSTANCE = None
        _PADDLE_READY = False
        logging.warning("PaddleOCR unavailable, using Tesseract-only fallback: %s", exc)


def run_paddleocr(image: np.ndarray) -> str:
    _init_global_paddle()
    if not _PADDLE_READY or _PADDLE_INSTANCE is None:
        return ""
    try:
        result = _PADDLE_INSTANCE.predict(_ensure_paddle_image(image))
        page_result = result[0]
        texts = [
            normalize_basic_text(str(item))
            for item in page_result["rec_texts"]
            if clean_whitespace(str(item))
        ]
        return normalize_basic_text(" ".join(texts))
    except Exception:
        return ""


def _encode_image(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Failed to encode OCR cell image.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _strip_ollama_text(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        lines = [line for line in value.splitlines() if not line.startswith("```")]
        value = "\n".join(lines).strip()
    value = value.strip().strip('"').strip("'")
    return normalize_basic_text(value)


def _ollama_prompt(field_type: str, model_name: str) -> str:
    if field_type == "numeric":
        return (
            "Read this single OCR cell from a 19th century printed trade table. "
            "Return only the numeric content exactly as seen. "
            "If the cell is blank, return an empty string."
        )
    return (
        "Read this single OCR cell from an 1850s Madras Presidency export table. "
        "It contains a commodity article name. Return only the article text. "
        "If the cell is blank, return an empty string."
    )


def _run_ollama_ocr(image: np.ndarray, field_type: str, model_name: str) -> str:
    if requests is None or model_name in _OLLAMA_UNAVAILABLE_MODELS:
        return ""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": _ollama_prompt(field_type, model_name),
                "images": [_encode_image(image)],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "num_predict": 128,
                },
            },
            timeout=OLLAMA_TIMEOUT,
        )
        if response.status_code >= 400:
            _OLLAMA_UNAVAILABLE_MODELS.add(model_name)
            return ""
        return _strip_ollama_text(response.json().get("response", ""))
    except Exception:
        _OLLAMA_UNAVAILABLE_MODELS.add(model_name)
        return ""


def run_deepseek_ocr(image: np.ndarray, field_type: str) -> str:
    return _run_ollama_ocr(image, field_type, DEEPSEEK_MODEL)


def run_glm_ocr(image: np.ndarray, field_type: str) -> str:
    return _run_ollama_ocr(image, field_type, GLM_MODEL)


def _normalize_numeric_vote(text: str) -> str:
    cleaned = clean_whitespace(text).strip(".,;:")
    cleaned = cleaned.replace(" ", "").replace(",", "")
    if not any(character.isdigit() for character in cleaned):
        return ""
    return cleaned


def _pick_text_result(results: dict[str, str], known_articles: set[str]) -> tuple[str, str, float]:
    best_engine = "none"
    best_value = ""
    best_score = 0.0
    for engine_name in ("tesseract", "paddle", "deepseek", "glm"):
        candidate = clean_whitespace(results.get(engine_name, ""))
        if not candidate:
            continue
        score = 0.0
        snapped = candidate
        if known_articles and process is not None and fuzz is not None:
            match = process.extractOne(candidate, known_articles, scorer=fuzz.WRatio)
            if match:
                snapped = str(match[0]) if match[1] > 85 else candidate
                score = float(match[1]) / 100.0
        else:
            score = max(score_candidate("article", candidate, None) / 5.0, 0.0)
        if score > best_score:
            best_engine = engine_name
            best_value = snapped
            best_score = score
    return best_value, best_engine, best_score


def ensemble_ocr(
    image: np.ndarray,
    field_type: str,
    known_articles: set[str],
    use_paddle: bool,
    use_deepseek: bool,
    use_glm: bool,
) -> dict[str, object]:
    tesseract_configs = {
        "numeric": "--psm 7 -c tessedit_char_whitelist=0123456789.,-",
        "text": "--psm 7",
    }
    all_results: dict[str, str] = {
        "tesseract": run_tesseract(image, tesseract_configs.get(field_type, "--psm 7")),
    }
    if use_paddle:
        all_results["paddle"] = run_paddleocr(image)
    if use_deepseek:
        all_results["deepseek"] = run_deepseek_ocr(image, field_type)
    if use_glm:
        all_results["glm"] = run_glm_ocr(image, field_type)

    nonempty = {engine: value for engine, value in all_results.items() if clean_whitespace(value)}
    if not nonempty:
        return {"value": "", "engine": "none", "all": all_results, "confidence": 0.0}

    if field_type == "numeric":
        normalized = {
            engine: _normalize_numeric_vote(value)
            for engine, value in nonempty.items()
        }
        normalized = {engine: value for engine, value in normalized.items() if value}
        if not normalized:
            return {"value": "", "engine": "none", "all": all_results, "confidence": 0.0}
        counts = Counter(normalized.values())
        winning_count = max(counts.values())
        winning_values = {value for value, count in counts.items() if count == winning_count}
        for preferred_engine in ("tesseract", "paddle", "glm", "deepseek"):
            if normalized.get(preferred_engine) in winning_values:
                return {
                    "value": normalized[preferred_engine],
                    "engine": preferred_engine,
                    "all": all_results,
                    "confidence": winning_count / len(normalized),
                }
        first_engine = next(iter(normalized))
        return {
            "value": normalized[first_engine],
            "engine": first_engine,
            "all": all_results,
            "confidence": winning_count / len(normalized),
        }

    value, engine, confidence = _pick_text_result(nonempty, known_articles)
    return {"value": value, "engine": engine, "all": all_results, "confidence": confidence}


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
        use_deepseek: bool = True,
        use_glm: bool = True,
        known_articles: set[str] | None = None,
    ) -> None:
        self.backend_name = backend_name
        self.workspace_root = workspace_root
        self.allow_paddle = allow_paddle
        self.use_deepseek = use_deepseek
        self.use_glm = use_glm
        self.known_articles = known_articles or set()
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
                **_paddle_init_kwargs(),
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
        result = self._paddle.predict(_ensure_paddle_image(image))
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
            result = self._paddle.predict(_ensure_paddle_image(image))
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

    def cell_text(
        self,
        image: np.ndarray,
        field_name: str,
        ensemble: bool = False,
    ) -> tuple[OCRCellResult, dict[str, str]]:
        field_type = _field_type_for(field_name)
        if field_type is not None:
            result = ensemble_ocr(
                image=image,
                field_type=field_type,
                known_articles=self.known_articles,
                use_paddle=self.allow_paddle,
                use_deepseek=self.use_deepseek,
                use_glm=self.use_glm,
            )
            return (
                OCRCellResult(
                    text=str(result["value"]),
                    confidence=float(result["confidence"]) if result["confidence"] is not None else None,
                    engine=str(result["engine"]),
                ),
                dict(result["all"]),
            )

        if self.backend_name in {"auto", "paddle"}:
            self._init_paddle()

        candidates = _ocr_cell_tesseract_candidates(image, field_name)
        if self._paddle_ready and (ensemble or self.backend_name == "paddle"):
            paddle_candidate = self._paddle_cell_candidate(image, field_name)
            if paddle_candidate is not None:
                candidates.append(paddle_candidate)

        nonempty = _filter_nonempty_candidates(candidates)
        if not nonempty:
            return OCRCellResult(text="", confidence=None, engine="none"), {"tesseract": ""}

        best = max(
            nonempty,
            key=lambda candidate: score_candidate(field_name, candidate.text, candidate.confidence),
        )
        all_outputs = {
            candidate.engine.split(":", 1)[0]: candidate.text
            for candidate in nonempty
        }
        return best, all_outputs


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
