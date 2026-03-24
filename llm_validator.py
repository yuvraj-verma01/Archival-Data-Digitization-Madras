from __future__ import annotations

import json
import os
from typing import Any

try:
    import requests
except Exception:
    requests = None

try:
    from known_articles import KNOWN_ARTICLES as IMPORTED_KNOWN_ARTICLES
except Exception:
    IMPORTED_KNOWN_ARTICLES = set()


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
KNOWN_ARTICLES = set(IMPORTED_KNOWN_ARTICLES)
TIMEOUT = 60
BATCH_SIZE = 6

SYSTEM_PROMPT = """
You validate OCR-extracted rows from a historical Madras Presidency export table.
You will receive a JSON array of flagged rows. For each row return a JSON array with:
- row_no
- article_suggestion: corrected commodity name if the article looks garbled, else null
- duty_rate_ok: true if foreign_duty_rate and indian_duty_rate are blank or match
  "at 3 per Cent." / "at 2 Annas per I. Md." / "at 3 Rs. per In. Md.", else false
- notes: one short sentence if anything looks wrong, else null
Return only valid JSON. No explanation outside the JSON.
""".strip()


def _append_flag(record: dict[str, Any], flag: str) -> None:
    current = str(record.get("confidence_flag", "") or "")
    if not current:
        record["confidence_flag"] = flag
        return
    parts = current.split(";")
    if flag not in parts:
        record["confidence_flag"] = f"{current};{flag}"


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    content = (text or "").strip()
    if content.startswith("```"):
        lines = [line for line in content.splitlines() if not line.startswith("```")]
        content = "\n".join(lines).strip()
    if content.startswith("[") and content.endswith("]"):
        return json.loads(content)

    start = content.find("[")
    if start == -1:
        raise ValueError("No JSON array found in model output")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return json.loads(content[start : index + 1])

    raise ValueError("Could not extract complete JSON array from model output")


def _build_prompt(payload: list[dict[str, Any]]) -> str:
    known_articles_text = json.dumps(sorted(KNOWN_ARTICLES), ensure_ascii=False)
    rows_text = json.dumps(payload, ensure_ascii=False)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Known article names for reference:\n{known_articles_text}\n\n"
        f"Rows to validate:\n{rows_text}"
    )


def _call_ollama(prompt: str) -> list[dict[str, Any]]:
    if requests is None:
        raise RuntimeError("requests is unavailable")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 1.0,
                "num_predict": 768,
            },
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    raw = (response.json().get("response") or "").strip()
    return _extract_json_array(raw)


def validate_weak_rows(records: list[dict]) -> list[dict]:
    for record in records:
        record.setdefault("llm_notes", None)

    weak = [r for r in records if r.get("confidence_flag", "ok") != "ok"]
    if not weak:
        return records

    result_map: dict[str, dict[str, Any]] = {}
    for start in range(0, len(weak), BATCH_SIZE):
        payload = [
            {
                "row_no": r["row_no"],
                "article": r["article"],
                "foreign_duty_rate": r["foreign_duty_rate"],
                "indian_duty_rate": r["indian_duty_rate"],
                "total_value": r["total_value"],
            }
            for r in weak[start : start + BATCH_SIZE]
        ]

        try:
            results = _call_ollama(_build_prompt(payload))
        except Exception:
            continue

        for item in results:
            if "row_no" in item:
                result_map[str(item["row_no"])] = item

    if not result_map:
        return records

    for record in records:
        feedback = result_map.get(str(record["row_no"]))
        if feedback is None:
            continue

        record["llm_notes"] = feedback.get("notes")
        suggestion = feedback.get("article_suggestion")
        if suggestion:
            record["article"] = suggestion
            _append_flag(record, "llm_corrected_article")
        if feedback.get("duty_rate_ok") is False:
            _append_flag(record, "llm_flagged_duty_rate")

    return records
