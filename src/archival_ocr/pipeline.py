"""Main orchestration for the archival OCR workflow."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .exporter import export_records
from .layouts import SPREAD_LAYOUTS
from .models import PageArtifacts
from .ocr_engines import OCRBackendChain, normalize_joined_tokens, score_candidate
from .postprocess import finalize_record
from .preprocess import cell_content_features, preprocess_page
from .rendering import render_page
from .schema import CELL_PRIMARY_FIELDS, LEFT_PAGE_FIELDS, OUTPUT_COLUMNS, RIGHT_PAGE_FIELDS
from .table_detection import (
    build_row_bound_map,
    build_row_center_map,
    draw_grid_overlay,
    draw_table_overlay,
    extract_crop,
    resolve_table_bbox,
)
from .utils import ensure_dir, normalize_basic_text, save_image, write_json


def _column_pixels(layout, table_width: int, field_name: str) -> tuple[int, int]:
    start_ratio, end_ratio = layout.columns[field_name]
    return int(table_width * start_ratio), int(table_width * end_ratio)


def _clip_box(
    artifacts: PageArtifacts,
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> tuple[int, int, int, int]:
    width = artifacts.table_image.shape[1]
    height = artifacts.table_image.shape[0]
    left = min(max(0, left), width - 1)
    right = max(left + 1, min(width, right))
    top = min(max(0, top), height - 1)
    bottom = max(top + 1, min(height, bottom))
    return left, right, top, bottom


def _cell_box(layout, artifacts: PageArtifacts, row_no: int, field_name: str) -> tuple[int, int, int, int]:
    if layout.side == "left" and field_name == "article":
        x0, _ = _column_pixels(layout, artifacts.table_image.shape[1], "article")
        _, x1 = _column_pixels(layout, artifacts.table_image.shape[1], "unit")
    else:
        x0, x1 = _column_pixels(layout, artifacts.table_image.shape[1], field_name)
    y0, y1 = artifacts.row_bounds[row_no]
    width = x1 - x0
    height = y1 - y0
    pad_x = max(1, int(width * 0.03))
    pad_y = max(1, int(height * 0.08))
    if field_name in CELL_PRIMARY_FIELDS:
        pad_x = max(1, int(width * 0.02))
        pad_y = max(1, int(height * 0.06))
    if layout.side == "left" and field_name == "article":
        pad_x = max(1, int(width * 0.015))
    return _clip_box(
        artifacts,
        x0 + pad_x,
        x1 - pad_x,
        y0 + pad_y,
        y1 - pad_y,
    )


def _token_box(layout, artifacts: PageArtifacts, row_no: int, field_name: str) -> tuple[int, int, int, int]:
    x0, x1 = _column_pixels(layout, artifacts.table_image.shape[1], field_name)
    y0, y1 = artifacts.row_bounds[row_no]
    width = x1 - x0
    height = y1 - y0
    pad_x = max(1, int(width * 0.03))
    pad_y = max(1, int(height * 0.08))
    if field_name in CELL_PRIMARY_FIELDS:
        pad_x = max(1, int(width * 0.02))
        pad_y = max(1, int(height * 0.06))
    return _clip_box(
        artifacts,
        x0 + pad_x,
        x1 - pad_x,
        y0 + pad_y,
        y1 - pad_y,
    )


def _candidate_boxes(
    artifacts: PageArtifacts,
    base_box: tuple[int, int, int, int],
    field_name: str,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    left, right, top, bottom = base_box
    width = right - left
    height = bottom - top
    dx = max(2, int(width * (0.06 if field_name == "article" else 0.04)))
    dy = max(2, int(height * (0.18 if field_name == "article" else 0.12)))
    variants = [
        ("base", base_box),
        ("wider", _clip_box(artifacts, left - dx, right + dx, top, bottom)),
        ("taller", _clip_box(artifacts, left, right, top - dy, bottom + dy)),
        ("wide_tall", _clip_box(artifacts, left - dx, right + dx, top - dy, bottom + dy)),
        ("shift_left", _clip_box(artifacts, left - dx, right - dx, top, bottom)),
        ("shift_right", _clip_box(artifacts, left + dx, right + dx, top, bottom)),
    ]
    deduped: list[tuple[str, tuple[int, int, int, int]]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for variant_name, box in variants:
        if box in seen:
            continue
        seen.add(box)
        deduped.append((variant_name, box))
    return deduped


def _retry_threshold(field_name: str) -> float:
    if field_name == "article":
        return 2.2
    if field_name in {"foreign_value_rs", "indian_value_rs", "total_value"}:
        return 2.5
    return 0.9


def _should_retry_field(field_name: str) -> bool:
    return field_name in {
        "article",
        "unit",
        "foreign_value_rs",
        "indian_value_rs",
        "total_value",
    }


def _needs_retry(field_name: str, text: str, confidence: float | None) -> bool:
    if not text:
        return True
    return score_candidate(field_name, text, confidence) < _retry_threshold(field_name)


def _split_article_and_unit(article_text: str, unit_text: str) -> tuple[str, str]:
    article = normalize_basic_text(article_text)
    unit = normalize_basic_text(unit_text)
    valid_unit_pattern = re.compile(
        r"^(No\.?|Nos\.?|Pairs?|Pieces?|Pkgs?\.?|Bags?\.?|Lbs?\.?|Yds?\.?|Cwts?\.?)$",
        re.IGNORECASE,
    )
    if unit and not valid_unit_pattern.fullmatch(unit):
        unit = ""
    if not article:
        return "", unit

    if unit:
        lowered_article = article.lower()
        lowered_unit = unit.lower()
        if lowered_article.endswith(lowered_unit):
            article = article[: len(article) - len(unit)].rstrip(" ,;:-")
            return article, unit

    match = re.search(
        r"\b(No\.?|Nos\.?|Pairs?|Pieces?|Pkgs?\.?|Bags?\.?|Lbs?\.?|Yds?\.?|Cwts?\.?)$",
        article,
        re.IGNORECASE,
    )
    if match:
        inferred_unit = normalize_basic_text(match.group(0))
        stripped_article = article[: match.start()].rstrip(" ,;:-")
        if stripped_article:
            return stripped_article, unit or inferred_unit

    return article, unit


def _assign_tokens_to_field(tokens, left: int, right: int, top: int, bottom: int, field_name: str) -> tuple[str, list[float]]:
    chosen = []
    confidences = []
    for token in tokens:
        if token.y_center < top or token.y_center > bottom:
            continue
        if token.x_center < left or token.x_center > right:
            continue
        chosen.append(token)
        confidences.append(token.confidence)
    chosen.sort(key=lambda token: (token.y, token.x))
    return normalize_joined_tokens(field_name, [token.text for token in chosen]), confidences


def _prefer_token_text(
    field_name: str,
    cell_text: str,
    cell_confidence: float | None,
    token_text: str,
    token_confidences: list[float],
) -> bool:
    if not token_text:
        return False
    token_confidence = (sum(token_confidences) / len(token_confidences)) if token_confidences else None
    token_score = score_candidate(field_name, token_text, token_confidence)
    cell_score = score_candidate(field_name, cell_text, cell_confidence)
    if (
        field_name == "article"
        and token_confidence is not None
        and token_confidence >= 0.8
        and token_score >= 1.8
        and cell_score < (token_score + 0.6)
    ):
        return True
    return token_score >= cell_score


def _cell_text(
    artifacts: PageArtifacts,
    layout,
    row_no: int,
    field_name: str,
    backend_chain: OCRBackendChain,
    *,
    ensemble: bool,
) -> tuple[str, float | None, str | None, list[int]]:
    base_box = _cell_box(layout, artifacts, row_no, field_name)
    variants = _candidate_boxes(artifacts, base_box, field_name)
    best_candidate: tuple[float, str, float | None, str | None, list[int]] | None = None
    blank_signals: list[dict[str, float | int | bool]] = []

    for index, (variant_name, box) in enumerate(variants):
        left, right, top, bottom = box
        cell_gray = artifacts.table_gray[top:bottom, left:right]
        cell_binary = artifacts.table_binary[top:bottom, left:right]
        signal = cell_content_features(cell_binary)
        blank_signals.append(signal)
        if signal["likely_blank"]:
            continue

        result = backend_chain.cell_text(cell_gray, field_name, ensemble=ensemble)
        candidate_score = score_candidate(field_name, result.text, result.confidence)
        candidate = (
            candidate_score,
            result.text,
            result.confidence,
            f"{result.engine}@{variant_name}",
            [left, right, top, bottom],
        )
        if best_candidate is None or candidate_score > best_candidate[0]:
            best_candidate = candidate

        if index == 0 and not _needs_retry(field_name, result.text, result.confidence):
            break

    if best_candidate is not None:
        _, text, confidence, engine, crop_box = best_candidate
        return text, confidence, engine, crop_box

    if blank_signals and all(bool(signal["likely_blank"]) for signal in blank_signals):
        return "", None, "blank_classifier", list(base_box)

    return "", None, None, list(base_box)


def _page_artifacts(
    pdf_path: Path,
    page_number: int,
    dpi: int,
    layout,
    expected_rows: int,
    backend_chain: OCRBackendChain,
    output_dirs: dict[str, Path],
) -> PageArtifacts:
    logging.info("Rendering page %s at %s DPI", page_number, dpi)
    image = render_page(pdf_path, page_number, dpi=dpi)
    save_image(output_dirs["rendered"] / f"page_{page_number:03d}.png", image)

    gray, binary = preprocess_page(image)
    save_image(output_dirs["preprocessed"] / f"page_{page_number:03d}_gray.png", gray)
    save_image(output_dirs["preprocessed"] / f"page_{page_number:03d}_binary.png", binary)

    table_bbox = resolve_table_bbox(binary, layout)
    save_image(output_dirs["debug"] / f"page_{page_number:03d}_table_box.png", draw_table_overlay(image, table_bbox))

    table_image = extract_crop(image, table_bbox)
    table_gray = extract_crop(gray, table_bbox)
    table_binary = extract_crop(binary, table_bbox)
    save_image(output_dirs["tables"] / f"page_{page_number:03d}_table.png", table_image)

    tokens, token_engine = backend_chain.table_tokens(table_gray)
    logging.info("Page %s OCR token pass used %s and found %s tokens", page_number, token_engine, len(tokens))

    row_centers, row_anchors = build_row_center_map(
        tokens=tokens,
        layout=layout,
        table_width=table_image.shape[1],
        table_height=table_image.shape[0],
        expected_rows=expected_rows,
    )
    row_bounds = build_row_bound_map(row_centers, table_image.shape[0])
    save_image(
        output_dirs["debug"] / f"page_{page_number:03d}_grid.png",
        draw_grid_overlay(table_image, row_bounds, layout.columns),
    )

    return PageArtifacts(
        page_number=page_number,
        image=image,
        gray=gray,
        binary=binary,
        table_bbox=table_bbox,
        table_image=table_image,
        table_gray=table_gray,
        table_binary=table_binary,
        tokens=tokens,
        row_centers=row_centers,
        row_bounds=row_bounds,
        row_anchors=row_anchors,
    )


def run_pipeline(
    pdf_path: Path,
    outdir: Path,
    start_page: int,
    dpi: int,
    expected_rows: int | None,
    layout_name: str,
    ocr_backend: str,
) -> dict:
    layout_bundle = SPREAD_LAYOUTS[layout_name]
    expected_rows = expected_rows or layout_bundle["expected_rows"]
    if start_page < 1:
        raise ValueError("--start-page must be a 1-based page number.")

    output_dirs = {
        "rendered": ensure_dir(outdir / "intermediate" / "rendered"),
        "preprocessed": ensure_dir(outdir / "intermediate" / "preprocessed"),
        "tables": ensure_dir(outdir / "intermediate" / "tables"),
        "debug": ensure_dir(outdir / "debug"),
    }

    backend_chain = OCRBackendChain(
        backend_name=ocr_backend,
        workspace_root=outdir,
        allow_paddle=(ocr_backend in {"auto", "paddle"}),
    )

    left_artifacts = _page_artifacts(
        pdf_path=pdf_path,
        page_number=start_page,
        dpi=dpi,
        layout=layout_bundle["left"],
        expected_rows=expected_rows,
        backend_chain=backend_chain,
        output_dirs=output_dirs,
    )
    right_artifacts = _page_artifacts(
        pdf_path=pdf_path,
        page_number=start_page + 1,
        dpi=dpi,
        layout=layout_bundle["right"],
        expected_rows=expected_rows,
        backend_chain=backend_chain,
        output_dirs=output_dirs,
    )

    records: list[dict] = []
    debug_rows: list[dict] = []

    for row_no in range(1, expected_rows + 1):
        raw_record = {
            "source_file": pdf_path.name,
            "page_number": f"{start_page}-{start_page + 1}",
            "row_no": row_no,
            "article": "",
            "unit": "",
            "foreign_quantity": "",
            "foreign_value_rs": "",
            "foreign_duty_rate": "",
            "foreign_duty_rs": "",
            "foreign_duty_as": "",
            "foreign_duty_p": "",
            "indian_quantity": "",
            "indian_value_rs": "",
            "indian_duty_rate": "",
            "indian_duty_rs": "",
            "indian_duty_as": "",
            "indian_duty_p": "",
            "total_value": "",
            "remarks": "",
            "raw_ocr_text": "",
            "confidence_flag": "",
            "_mean_confidence": None,
        }
        cell_debug = {}
        confidences: list[float] = []

        for artifacts, layout, field_names in (
            (left_artifacts, layout_bundle["left"], LEFT_PAGE_FIELDS),
            (right_artifacts, layout_bundle["right"], RIGHT_PAGE_FIELDS),
        ):
            for field_name in field_names:
                if field_name == "row_no":
                    continue

                if field_name in CELL_PRIMARY_FIELDS:
                    left, right, top, bottom = _token_box(layout, artifacts, row_no, field_name)
                    token_text, token_confidences = _assign_tokens_to_field(
                        artifacts.tokens,
                        left,
                        right,
                        top,
                        bottom,
                        field_name,
                    )
                    final_text = token_text
                    field_conf = (sum(token_confidences) / len(token_confidences)) if token_confidences else None
                    engine_used = "table_tokens" if token_text else None
                    crop_box = [left, right, top, bottom]
                    if not final_text or _needs_retry(field_name, final_text, field_conf):
                        fallback_text, fallback_conf, fallback_engine, fallback_box = _cell_text(
                            artifacts,
                            layout,
                            row_no,
                            field_name,
                            backend_chain,
                            ensemble=True,
                        )
                        if fallback_text and not _prefer_token_text(
                            field_name,
                            fallback_text,
                            fallback_conf,
                            token_text,
                            token_confidences,
                        ):
                            final_text = fallback_text
                            field_conf = fallback_conf
                            engine_used = fallback_engine
                            crop_box = fallback_box
                        elif not final_text:
                            engine_used = fallback_engine or engine_used
                else:
                    left, right, top, bottom = _token_box(layout, artifacts, row_no, field_name)
                    token_text, token_confidences = _assign_tokens_to_field(
                        artifacts.tokens,
                        left,
                        right,
                        top,
                        bottom,
                        field_name,
                    )
                    final_text = token_text
                    field_conf = (sum(token_confidences) / len(token_confidences)) if token_confidences else None
                    engine_used = "table_tokens"
                    crop_box = [left, right, top, bottom]
                    if _should_retry_field(field_name) and (
                        not final_text or _needs_retry(field_name, final_text, field_conf)
                    ):
                        fallback_text, fallback_conf, fallback_engine, crop_box = _cell_text(
                            artifacts,
                            layout,
                            row_no,
                            field_name,
                            backend_chain,
                            ensemble=False,
                        )
                        if fallback_text and score_candidate(field_name, fallback_text, fallback_conf) >= score_candidate(
                            field_name,
                            final_text,
                            field_conf,
                        ):
                            final_text = fallback_text
                            field_conf = fallback_conf
                            engine_used = fallback_engine or engine_used

                raw_record[field_name] = final_text
                cell_debug[field_name] = {
                    "text": final_text,
                    "confidence": field_conf,
                    "engine": engine_used,
                    "crop_box": crop_box,
                }
                if field_conf is not None:
                    confidences.append(field_conf)

        article_text, unit_text = _split_article_and_unit(
            str(raw_record.get("article", "")),
            str(raw_record.get("unit", "")),
        )
        if article_text != raw_record.get("article", ""):
            raw_record["article"] = article_text
            if "article" in cell_debug:
                cell_debug["article"]["text"] = article_text
                if cell_debug["article"].get("engine"):
                    cell_debug["article"]["engine"] = f"{cell_debug['article']['engine']}+row_split"
        if unit_text != raw_record.get("unit", ""):
            raw_record["unit"] = unit_text
            if "unit" in cell_debug:
                cell_debug["unit"]["text"] = unit_text

        raw_record["_mean_confidence"] = (sum(confidences) / len(confidences)) if confidences else None
        raw_record["raw_ocr_text"] = " | ".join(
            f"{field}={raw_record.get(field, '')}"
            for field in OUTPUT_COLUMNS
            if field not in {"source_file", "page_number", "confidence_flag", "raw_ocr_text"}
        )
        finalized = finalize_record(raw_record, expected_row_no=row_no)
        records.append(finalized)
        debug_rows.append(
            {
                "row_no": row_no,
                "cells": cell_debug,
                "left_row_anchor": left_artifacts.row_anchors.get(row_no),
                "right_row_anchor": right_artifacts.row_anchors.get(row_no),
                "mean_confidence": raw_record["_mean_confidence"],
            }
        )

    csv_path, xlsx_path = export_records(records, outdir)
    write_json(outdir / "debug" / "ocr_debug.json", debug_rows)

    return {
        "rows": len(records),
        "csv_path": str(csv_path),
        "xlsx_path": str(xlsx_path),
    }
