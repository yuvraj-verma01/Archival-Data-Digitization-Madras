"""Main orchestration for the archival OCR workflow."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .exporter import export_records
from .layouts import SPREAD_LAYOUTS
from .models import PageArtifacts
from .ocr_engines import OCRBackendChain, normalize_joined_tokens
from .postprocess import finalize_record
from .preprocess import preprocess_page
from .rendering import render_page
from .schema import LEFT_PAGE_FIELDS, OUTPUT_COLUMNS, RIGHT_PAGE_FIELDS
from .table_detection import (
    build_row_bounds,
    detect_table_bbox,
    draw_grid_overlay,
    draw_table_overlay,
    extract_crop,
    fit_row_centers,
)
from .utils import ensure_dir, save_image, write_json


def _column_pixels(layout, table_width: int, field_name: str) -> tuple[int, int]:
    start_ratio, end_ratio = layout.columns[field_name]
    return int(table_width * start_ratio), int(table_width * end_ratio)


def _ink_ratio(binary_crop: np.ndarray) -> float:
    if binary_crop.size == 0:
        return 0.0
    return float(np.count_nonzero(binary_crop)) / float(binary_crop.size)


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


def _fallback_cell_text(
    artifacts: PageArtifacts,
    layout,
    row_bounds: tuple[int, int],
    field_name: str,
    backend_chain: OCRBackendChain,
) -> tuple[str, float | None, str | None]:
    x0, x1 = _column_pixels(layout, artifacts.table_image.shape[1], field_name)
    y0, y1 = row_bounds
    cell_gray = artifacts.table_gray[y0:y1, x0:x1]
    cell_binary = artifacts.table_binary[y0:y1, x0:x1]
    if _ink_ratio(cell_binary) < 0.008:
        return "", None, None
    if cell_gray.size == 0:
        return "", None, None
    result = backend_chain.cell_text(cell_gray, field_name)
    return result.text, result.confidence, result.engine


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

    table_bbox = detect_table_bbox(binary)
    save_image(output_dirs["debug"] / f"page_{page_number:03d}_table_box.png", draw_table_overlay(image, table_bbox))

    table_image = extract_crop(image, table_bbox)
    table_gray = extract_crop(gray, table_bbox)
    table_binary = extract_crop(binary, table_bbox)
    save_image(output_dirs["tables"] / f"page_{page_number:03d}_table.png", table_image)

    tokens, token_engine = backend_chain.table_tokens(table_gray)
    logging.info("Page %s OCR token pass used %s and found %s tokens", page_number, token_engine, len(tokens))

    row_centers = fit_row_centers(
        tokens=tokens,
        layout=layout,
        table_width=table_image.shape[1],
        table_height=table_image.shape[0],
        expected_rows=expected_rows,
    )
    row_bounds = build_row_bounds(row_centers, table_image.shape[0])
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

    for row_index in range(expected_rows):
        row_no = row_index + 1
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
            top, bottom = artifacts.row_bounds[row_index]
            for field_name in field_names:
                if field_name == "row_no":
                    continue
                x0, x1 = _column_pixels(layout, artifacts.table_image.shape[1], field_name)
                token_text, token_confidences = _assign_tokens_to_field(
                    artifacts.tokens,
                    x0,
                    x1,
                    top,
                    bottom,
                    field_name,
                )
                engine_used = "table_tokens"
                final_text = token_text
                field_conf = (sum(token_confidences) / len(token_confidences)) if token_confidences else None
                if not final_text:
                    fallback_text, fallback_conf, fallback_engine = _fallback_cell_text(
                        artifacts,
                        layout,
                        (top, bottom),
                        field_name,
                        backend_chain,
                    )
                    if fallback_text:
                        final_text = fallback_text
                        field_conf = fallback_conf
                        engine_used = fallback_engine or engine_used
                raw_record[field_name] = final_text
                cell_debug[field_name] = {
                    "text": final_text,
                    "confidence": field_conf,
                    "engine": engine_used,
                    "row_bounds": [top, bottom],
                    "column_bounds": [x0, x1],
                }
                if field_conf is not None:
                    confidences.append(field_conf)

        raw_record["_mean_confidence"] = (sum(confidences) / len(confidences)) if confidences else None
        raw_record["raw_ocr_text"] = " | ".join(
            f"{field}={raw_record.get(field, '')}"
            for field in OUTPUT_COLUMNS
            if field not in {"source_file", "page_number", "confidence_flag", "raw_ocr_text"}
        )
        finalized = finalize_record(raw_record, expected_row_no=row_no)
        records.append(finalized)
        debug_rows.append({"row_no": row_no, "cells": cell_debug, "mean_confidence": raw_record["_mean_confidence"]})

    csv_path, xlsx_path = export_records(records, outdir)
    write_json(outdir / "debug" / "ocr_debug.json", debug_rows)

    return {
        "rows": len(records),
        "csv_path": str(csv_path),
        "xlsx_path": str(xlsx_path),
    }
