"""Field cleaning, validation, and confidence flagging."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .schema import NUMERIC_FIELDS, RATE_FIELDS
from .utils import clean_whitespace, is_placeholder_blank, normalize_basic_text


def clean_numeric_value(raw_text: str) -> tuple[Any, bool]:
    text = clean_whitespace(raw_text)
    if not text:
        return pd.NA, False
    if is_placeholder_blank(text):
        return pd.NA, False
    normalized = (
        text.replace(" ", "")
        .replace(",", "")
        .replace("—", "-")
        .replace("−", "-")
        .replace("|", "")
    )
    normalized = normalized.strip(".,;:")
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+", normalized):
        normalized = normalized.replace(".", "")
    elif normalized.count(".") == 1:
        head, tail = normalized.split(".")
        if head.lstrip("-").isdigit() and tail.isdigit() and len(tail) == 3:
            normalized = f"{head}{tail}"
    if not any(character.isdigit() for character in normalized):
        return pd.NA, True
    if re.fullmatch(r"-?\d+", normalized):
        return int(normalized), False
    if re.fullmatch(r"-?\d+\.\d+", normalized):
        return float(normalized), False
    return pd.NA, True


def clean_text_field(field_name: str, raw_text: str) -> str:
    text = normalize_basic_text(raw_text)
    if not text:
        return ""
    if is_placeholder_blank(text):
        return ""
    if field_name in RATE_FIELDS:
        return text
    if not re.search(r"[A-Za-z0-9]", text):
        return ""
    return text


def finalize_record(
    raw_record: dict[str, Any],
    expected_row_no: int,
) -> dict[str, Any]:
    record = raw_record.copy()
    issues: list[str] = []

    record["row_no"] = expected_row_no

    for field_name in (
        "article",
        "unit",
        "foreign_duty_rate",
        "indian_duty_rate",
        "remarks",
    ):
        record[field_name] = clean_text_field(field_name, str(record.get(field_name, "")))

    for field_name in NUMERIC_FIELDS - {"row_no"}:
        value, suspicious = clean_numeric_value(str(record.get(field_name, "")))
        record[field_name] = value
        if suspicious:
            issues.append(f"{field_name}_non_numeric")

    mean_confidence = raw_record.get("_mean_confidence")
    if mean_confidence is not None and mean_confidence < 0.55:
        issues.append("low_ocr_confidence")

    if record.get("article", "").lower() in {"", "do", "do."} and expected_row_no == 1:
        issues.append("first_article_unresolved")

    record["confidence_flag"] = "ok" if not issues else f"review:{';'.join(sorted(set(issues)))}"
    record.pop("_mean_confidence", None)
    return record
