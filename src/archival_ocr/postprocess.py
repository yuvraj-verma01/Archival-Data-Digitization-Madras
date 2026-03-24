"""Field cleaning, validation, and confidence flagging."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
try:
    from rapidfuzz import fuzz, process
except Exception:
    fuzz = None
    process = None

try:
    from known_articles import KNOWN_ARTICLES
except Exception:
    KNOWN_ARTICLES = set()

try:
    from llm_validator import validate_weak_rows
except Exception:
    def validate_weak_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return records

from .schema import NUMERIC_FIELDS, RATE_FIELDS
from .utils import clean_whitespace, is_placeholder_blank, normalize_basic_text

CATEGORY_LABELS = {
    "Apparel",
    "Cattle",
    "Cotton Goods",
    "Drugs",
    "Dyes",
    "Fruits and Nuts",
    "Grain",
    "Horns",
    "Manufactured Metals",
    "Naval Stores",
    "Oils",
    "Precious Stones",
    "Provisions",
    "Seeds",
    "Spices",
    "Spirits",
    "Timber and Planks",
    "Woods",
    "Woollens",
    "Metals",
}
ARTICLE_CATEGORY_BLOCKLIST = CATEGORY_LABELS

DITTO_VALUES = {"do", "do.", "per do", "per do."}
LEADING_ARTICLE_TRIM_CHARS = set('{}[]\\("')
TRAILING_ARTICLE_PUNCTUATION = "',.; °"
ANNAS_FIELDS = ("foreign_duty_as", "indian_duty_as")
PIES_FIELDS = ("foreign_duty_p", "indian_duty_p")
UNIT_DITTO_VALUES = {'"', "“", "”"}
UNIT_NORMALIZATION = {
    "No": "No.",
    "No]": "No.",
    "No,": "No.",
    "Cwt": "Cwt.",
    "Cwtl": "Cwt.",
    "Cwt.": "Cwt.",
    "Ibs": "lbs.",
    "Ibs.": "lbs.",
    "lbs": "lbs.",
    "lbs.": "lbs.",
}


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
    if field_name == "article":
        return clean_article_field(text)
    if field_name == "unit":
        return clean_unit_field(text)
    if field_name in RATE_FIELDS:
        return normalize_rate_field(field_name, text)
    if not re.search(r"[A-Za-z0-9]", text):
        return ""
    return text


def clean_article_field(raw_text: str) -> str:
    text = normalize_basic_text(raw_text)
    if not text or is_placeholder_blank(text):
        return ""
    while text and text[0] in LEADING_ARTICLE_TRIM_CHARS:
        text = text[1:].lstrip()
    while text.endswith("..."):
        text = text[:-3].rstrip()
    text = text.rstrip(TRAILING_ARTICLE_PUNCTUATION)
    if not re.search(r"[A-Za-z0-9]", text):
        return ""
    return text


def snap_known_article(article: str) -> str:
    cleaned = clean_article_field(article)
    if not cleaned or cleaned in CATEGORY_LABELS or not KNOWN_ARTICLES or process is None or fuzz is None:
        return cleaned
    match = process.extractOne(cleaned, KNOWN_ARTICLES, scorer=fuzz.WRatio)
    if not match:
        return cleaned
    candidate, score, _ = match
    if score > 85:
        return str(candidate)
    return cleaned


def clean_unit_field(raw_text: str) -> str:
    text = normalize_basic_text(raw_text)
    text = text.strip()
    if text in UNIT_DITTO_VALUES:
        return '"'
    normalized = UNIT_NORMALIZATION.get(text, text)
    if normalized in UNIT_DITTO_VALUES:
        return '"'
    if not normalized:
        return ""
    if not re.search(r"[A-Za-z0-9\"]", normalized):
        return ""
    return normalized


def normalize_rate_field(field_name: str, raw_text: str) -> str:
    text = normalize_basic_text(raw_text)
    normalized = re.sub(r"[^a-z0-9.]+", " ", text.lower()).strip()
    if normalized in DITTO_VALUES:
        return "Do."
    if (
        field_name == "foreign_duty_rate"
        and "cent" in normalized
        and len(normalized.split()) <= 3
    ):
        return "at 3 per Cent."
    if (
        "3" in normalized
        and "cent" in normalized
        and ("at" in normalized or "per" in normalized)
        and len(normalized.split()) <= 6
    ):
        return "at 3 per Cent."
    if (
        "3" in normalized
        and "rs" in normalized
        and ("md" in normalized or "in" in normalized)
        and len(normalized.split()) <= 8
    ):
        return "at 3 Rs. per In. Md."
    return text


def resolve_ditto(records: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    last_seen = {field: "" for field in fields}
    for record in records:
        for field in fields:
            value = clean_whitespace(str(record.get(field, "")))
            normalized = value.lower()
            if normalized in DITTO_VALUES:
                record[field] = last_seen[field]
                continue
            if value:
                last_seen[field] = value
                record[field] = value
    return records


def _range_issue(field_name: str, value: Any) -> str | None:
    if pd.isna(value):
        return None
    if field_name in ANNAS_FIELDS and value > 15:
        return "annas_out_of_range"
    if field_name in PIES_FIELDS and value > 3:
        return "pies_out_of_range"
    return None


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


def finalize_records(raw_records: list[dict[str, Any]], use_llm: bool = True) -> list[dict[str, Any]]:
    cleaned_records: list[dict[str, Any]] = []

    for raw_record in raw_records:
        expected_row_no = int(raw_record.get("row_no", 0))
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

        record["_issues"] = issues
        cleaned_records.append(record)

    resolve_ditto(cleaned_records, ["foreign_duty_rate", "indian_duty_rate"])

    finalized_records: list[dict[str, Any]] = []
    for record in cleaned_records:
        article = str(record.get("article", ""))
        if article in ARTICLE_CATEGORY_BLOCKLIST:
            continue
        snapped_article = snap_known_article(article)
        if snapped_article:
            record["article"] = snapped_article
            article = snapped_article

        issues = list(record.pop("_issues", []))
        for field_name in ANNAS_FIELDS + PIES_FIELDS:
            range_issue = _range_issue(field_name, record.get(field_name))
            if range_issue:
                issues.append(range_issue)

        mean_confidence = record.get("_mean_confidence")
        if mean_confidence is not None and mean_confidence < 0.55:
            issues.append("low_ocr_confidence")

        if article.lower() in {"", "do", "do."} and int(record["row_no"]) == 1:
            issues.append("first_article_unresolved")

        record["confidence_flag"] = "ok" if not issues else f"review:{';'.join(sorted(set(issues)))}"
        record.pop("_mean_confidence", None)
        finalized_records.append(record)

    return postprocess(finalized_records, use_llm=use_llm)


def postprocess(records: list[dict[str, Any]], use_llm: bool = True) -> list[dict[str, Any]]:
    for record in records:
        record.setdefault("llm_notes", None)
    if not use_llm:
        return records
    return validate_weak_rows(records)
