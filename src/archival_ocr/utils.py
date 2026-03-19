"""Small utility helpers for file IO, logging, and text cleanup."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import cv2


NOISE_TOKENS = {
    "|",
    "||",
    ".",
    ":",
    ";",
    ",",
    "_",
    "-",
    "+",
    "ve",
    "vee",
    "wee",
    "web",
    "ae",
    "te",
    "on",
    "en",
    "rn",
    "tn",
    "ws",
    "us",
    "po",
    "boe",
    "bee",
    "bes",
    "ves",
    "webte",
}


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_placeholder_blank(text: str) -> bool:
    """
    Treat cells that are only leader dots or dash-like placeholders as blank.
    Embedded dots inside real content such as "Shoes..." should not be blanked.
    """
    value = clean_whitespace(text)
    if not value:
        return True
    normalized = re.sub(r"\s+", "", value)
    return bool(re.fullmatch(r"[.\-–—_:;|]{2,}", normalized))


def normalize_basic_text(text: str) -> str:
    value = clean_whitespace(text)
    value = value.replace("Da.", "Do.")
    value = value.replace("Da,", "Do.")
    value = value.replace("Do,", "Do.")
    value = value.replace("Vent.", "Cent.")
    value = value.replace("Vent", "Cent")
    return value.strip()


def safe_token_text(text: str) -> str:
    return clean_whitespace(text).strip()
