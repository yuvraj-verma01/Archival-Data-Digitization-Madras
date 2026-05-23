from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from archival_ocr.models import BoundingBox, OCRToken
from archival_ocr.ocr_engines import configure_tesseract, tesseract_tokens
from archival_ocr.preprocess import (
    cell_content_features,
    denoise,
    enhance_contrast,
    prepare_cell_variants,
    preprocess_page,
    trim_cell_border,
)
from archival_ocr.rendering import render_page
from archival_ocr.table_detection import build_row_bound_map, detect_table_bbox, extract_crop
from archival_ocr.utils import clean_whitespace, ensure_dir, normalize_basic_text, setup_logging

import pytesseract


@dataclass(frozen=True)
class PageColumn:
    index: int
    left: int
    right: int
    header: str
    field_name: str
    field_kind: str

    @property
    def width(self) -> int:
        return self.right - self.left


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the scanned import tables into an Excel workbook.")
    parser.add_argument(
        "--input-dir",
        default="raw/Import",
        help="Directory containing the import PDFs. Default: raw/Import",
    )
    parser.add_argument(
        "--outdir",
        default="output/import_extraction",
        help="Directory where the Excel workbook and debug files will be written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Render DPI for OCR. Default: 300",
    )
    return parser.parse_args()


def _sanitize_sheet_name(name: str) -> str:
    cleaned = re.sub(r"[\[\]\*:/\\?]", "_", name)
    return cleaned[:31]


def _normalize_header_text(text: str) -> str:
    value = normalize_basic_text(text)
    value = value.replace("Rupoes", "Rupees")
    value = value.replace("Rupees.", "Rupees")
    value = value.replace("Extemal", "External")
    value = value.replace("Externa! ", "External ")
    return clean_whitespace(value)


def _group_positions(values: list[int], tolerance: int) -> list[int]:
    if not values:
        return []
    values = sorted(values)
    groups: list[list[int]] = [[values[0]]]
    for value in values[1:]:
        if abs(value - groups[-1][-1]) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [int(round(sum(group) / len(group))) for group in groups]


def _detect_import_table_bbox(binary_image: np.ndarray) -> BoundingBox:
    height, width = binary_image.shape[:2]
    trim_y = int(height * 0.05)
    trim_x = int(width * 0.05)
    inner = binary_image[trim_y : height - trim_y, trim_x : width - trim_x]

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, inner.shape[0] // 18)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, inner.shape[1] // 20), 1))
    vertical = cv2.morphologyEx(inner, cv2.MORPH_OPEN, vertical_kernel)
    horizontal = cv2.morphologyEx(inner, cv2.MORPH_OPEN, horizontal_kernel)
    grid = cv2.add(vertical, horizontal)
    grid = cv2.dilate(
        grid,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, inner.shape[1] // 40), max(15, inner.shape[0] // 40))),
        iterations=1,
    )

    points = cv2.findNonZero(grid)
    if points is None:
        return detect_table_bbox(binary_image)

    x, y, w, h = cv2.boundingRect(points)
    return BoundingBox(
        x=max(0, x + trim_x - 12),
        y=max(0, y + trim_y - 12),
        width=min(width, w + 24),
        height=min(height, h + 24),
    )


def _detect_columns(table_binary: np.ndarray) -> list[tuple[int, int]]:
    height, width = table_binary.shape[:2]
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, height // 12)))
    vertical = cv2.morphologyEx(table_binary, cv2.MORPH_OPEN, vertical_kernel)
    projection = vertical.sum(axis=0) / 255.0
    threshold = max(np.percentile(projection, 90), float(projection.max()) * 0.12)
    hit_indices = np.flatnonzero(projection >= threshold)
    if hit_indices.size == 0:
        return []

    runs: list[list[int]] = [[int(hit_indices[0])]]
    for index in hit_indices[1:]:
        current = int(index)
        if current - runs[-1][-1] <= 8:
            runs[-1].append(current)
        else:
            runs.append([current])

    peaks = [int(round(sum(run) / len(run))) for run in runs]
    boundaries = _group_positions(peaks, tolerance=max(12, width // 100))
    intervals: list[tuple[int, int]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        if (right - left) < max(18, width // 160):
            continue
        intervals.append((left, right))
    return intervals


def _expected_total_import_row_start(pdf_path: Path, page_number: int) -> int | None:
    stem = pdf_path.stem
    page_maps = {
        "186465_TotalImports_1": {
            1: 1,
            2: 1,
            3: 1,
            4: 48,
            5: 48,
            6: 95,
            7: 95,
            8: 142,
            9: 142,
        },
        "186465_TotalImports_2": {
            1: 189,
            2: 189,
            3: 236,
            4: 236,
            5: 236,
            6: 283,
            7: 283,
            8: 330,
            9: 330,
        },
    }
    return page_maps.get(stem, {}).get(page_number)


def _horizontal_rule_peaks(table_binary: np.ndarray) -> list[float]:
    height, width = table_binary.shape[:2]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, width // 18), 1))
    horizontal = cv2.morphologyEx(table_binary, cv2.MORPH_OPEN, horizontal_kernel)
    projection = horizontal.sum(axis=1) / 255.0
    threshold = max(np.percentile(projection, 92), float(projection.max()) * 0.10)
    hit_indices = np.flatnonzero(projection >= threshold)
    if hit_indices.size == 0:
        return []

    runs: list[list[int]] = [[int(hit_indices[0])]]
    for index in hit_indices[1:]:
        current = int(index)
        if current - runs[-1][-1] <= 4:
            runs[-1].append(current)
        else:
            runs.append([current])
    return [sum(run) / len(run) for run in runs]


def _fallback_row_centers_from_rules(
    table_binary: np.ndarray,
    row_start: int,
    expected_rows: int = 47,
) -> tuple[dict[int, float], None]:
    height = table_binary.shape[0]
    peaks = _horizontal_rule_peaks(table_binary)
    header_peaks = [peak for peak in peaks if peak < height * 0.35]
    body_bottom_peaks = [peak for peak in peaks if peak > height * 0.70]

    if header_peaks and body_bottom_peaks:
        header_bottom = max(header_peaks)
        body_bottom = min(body_bottom_peaks)
        step = (body_bottom - header_bottom) / float(expected_rows + 3.1)
        first_center = header_bottom + (2.5 * step)
    else:
        first_center = height * 0.21
        step = (height * 0.70) / float(max(expected_rows - 1, 1))

    row_centers = {
        row_no: first_center + (step * offset)
        for offset, row_no in enumerate(range(row_start, row_start + expected_rows))
    }
    return row_centers, None


def _row_centers_are_plausible(row_centers: dict[int, float], table_height: int) -> bool:
    if not row_centers:
        return False
    ordered = [center for _, center in sorted(row_centers.items())]
    if ordered[0] < table_height * 0.10:
        return False
    if ordered[-1] > table_height * 1.02:
        return False
    return True


def _row_tokens_from_strip(strip_gray: np.ndarray) -> list[tuple[int, float, float]]:
    enhanced = enhance_contrast(strip_gray)
    thresholded = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    scaled = cv2.copyMakeBorder(thresholded, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
    scaled = cv2.resize(scaled, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    candidates: list[tuple[int, float, float]] = []
    for psm in (6, 11):
        data = pytesseract.image_to_data(
            scaled,
            config=f"--psm {psm} -c tessedit_char_whitelist=0123456789",
            output_type=pytesseract.Output.DICT,
        )
        for index, raw_text in enumerate(data["text"]):
            text = clean_whitespace(raw_text)
            if not text:
                continue
            digits = "".join(character for character in text if character.isdigit())
            if not digits:
                continue
            row_no = int(digits)
            if row_no <= 0 or row_no > 600:
                continue
            confidence_raw = data["conf"][index]
            confidence = float(confidence_raw) if confidence_raw != "-1" else -1.0
            if confidence < 0:
                continue
            y_center = (float(data["top"][index]) + (float(data["height"][index]) / 2.0) - 8.0) / 2.0
            candidates.append((row_no, y_center, confidence / 100.0))
    return candidates


def _fit_row_model(
    candidates: list[tuple[int, float, float]],
) -> tuple[float, float, list[tuple[int, float, float]], float] | None:
    best_model: tuple[float, float, list[tuple[int, float, float]], float] | None = None
    for index, (row_a, y_a, _) in enumerate(candidates):
        for row_b, y_b, _ in candidates[index + 1 :]:
            if row_b <= row_a or y_b <= y_a:
                continue
            step = (y_b - y_a) / float(row_b - row_a)
            if step < 25.0 or step > 65.0:
                continue
            intercept = y_a - (step * row_a)
            tolerance = max(8.0, step * 0.35)
            inliers: list[tuple[int, float, float]] = []
            score = 0.0
            for row_no, y_center, confidence in candidates:
                error = abs(((step * row_no) + intercept) - y_center)
                if error > tolerance:
                    continue
                inliers.append((row_no, y_center, confidence))
                score += 1.0 + confidence - ((error / tolerance) * 0.25)
            unique_rows = {row_no for row_no, _, _ in inliers}
            if len(unique_rows) < 8:
                continue
            best_unique_count = len({row_no for row_no, _, _ in best_model[2]}) if best_model is not None else -1
            if best_model is None or (len(unique_rows), score) > (best_unique_count, best_model[3]):
                best_model = (step, intercept, inliers, score)
    if best_model is None:
        return None

    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row_no, y_center, confidence in best_model[2]:
        grouped[row_no].append((confidence, y_center))
    rows = np.array(sorted(grouped), dtype=np.float32)
    y_values = np.array(
        [
            max(values, key=lambda item: item[0])[1]
            for _, values in sorted(grouped.items())
        ],
        dtype=np.float32,
    )
    if rows.size >= 2:
        refined_step, refined_intercept = np.polyfit(rows, y_values, 1)
        step = float(refined_step)
        intercept = float(refined_intercept)
    else:
        step, intercept = best_model[0], best_model[1]
    inlier_list = [
        (row_no, y_center, confidence)
        for row_no, y_center, confidence in candidates
        if abs(((step * row_no) + intercept) - y_center) <= max(8.0, step * 0.35)
    ]
    return step, intercept, inlier_list, best_model[3]


def _infer_row_centers(
    table_gray: np.ndarray,
    intervals: list[tuple[int, int]],
    table_binary: np.ndarray | None = None,
    expected_row_start: int | None = None,
) -> tuple[dict[int, float], tuple[int, int] | None]:
    _, table_width = table_gray.shape[:2]
    best: tuple[float, float, list[tuple[int, float, float]], float, int, int] | None = None

    for left, right in intervals:
        if (right - left) > max(90, int(table_width * 0.12)):
            continue
        strip = table_gray[:, max(0, left - 4) : min(table_width, right + 4)]
        candidates = sorted(_row_tokens_from_strip(strip), key=lambda item: item[1])
        if len(candidates) < 8:
            continue
        model = _fit_row_model(candidates)
        if model is None:
            continue
        step, intercept, inliers, score = model
        unique_rows = {row_no for row_no, _, _ in inliers}
        best_unique_count = len({row_no for row_no, _, _ in best[2]}) if best is not None else -1
        best_score = best[3] if best is not None else -1.0
        if best is None or (len(unique_rows), score) > (best_unique_count, best_score):
            best = (step, intercept, inliers, score, left, right)

    if best is None:
        if table_binary is not None and expected_row_start is not None:
            logging.warning("Unable to infer row positions from OCR; using ruled-grid fallback.")
            return _fallback_row_centers_from_rules(table_binary, expected_row_start)
        raise RuntimeError("Unable to infer row positions from the numbered strip.")

    step, intercept, inliers, _, row_left, row_right = best
    visible_rows = sorted({row_no for row_no, _, _ in inliers})
    row_start = expected_row_start
    if row_start is None:
        row_start = ((min(visible_rows) - 1) // 47) * 47 + 1
    row_end = row_start + 46
    row_centers = {
        row_no: (step * row_no) + intercept
        for row_no in range(row_start, row_end + 1)
    }
    if not _row_centers_are_plausible(row_centers, table_gray.shape[0]):
        if table_binary is not None and expected_row_start is not None:
            logging.warning("OCR row model was out of bounds; using ruled-grid fallback.")
            return _fallback_row_centers_from_rules(table_binary, expected_row_start)
        raise RuntimeError("Inferred row positions are outside the table bounds.")
    return row_centers, (row_left, row_right)


def _join_tokens_in_box(
    tokens: list[OCRToken],
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> str:
    chosen = [
        token
        for token in tokens
        if left <= token.x_center <= right and top <= token.y_center <= bottom
    ]
    chosen.sort(key=lambda token: (token.y, token.x))
    return _normalize_header_text(" ".join(token.text for token in chosen))


def _best_article_and_unit_columns(intervals: list[tuple[int, int]], table_width: int) -> tuple[int, int | None]:
    candidates = [
        (index, right - left)
        for index, (left, right) in enumerate(intervals)
        if left < (table_width * 0.55) and index != 0
    ]
    if not candidates:
        return 1, None
    article_idx = max(candidates, key=lambda item: item[1])[0]
    unit_idx: int | None = None
    if article_idx + 1 < len(intervals):
        unit_width = intervals[article_idx + 1][1] - intervals[article_idx + 1][0]
        if unit_width <= int(table_width * 0.10):
            unit_idx = article_idx + 1
    return article_idx, unit_idx


def _classify_header(
    header: str,
    index: int,
    article_idx: int,
    unit_idx: int | None,
    row_index: int | None,
) -> tuple[str, str]:
    header_lower = header.lower()
    if row_index is not None and index == row_index:
        return "row_no", "numeric"
    if index == article_idx or "article" in header_lower:
        return "article", "text"
    if unit_idx is not None and index == unit_idx:
        return "unit", "text"
    if "remark" in header_lower:
        return "remarks", "text"
    if "rates of duty" in header_lower or "chargeable" in header_lower:
        return "duty_rate", "rate"
    if "gross amount of duty" in header_lower:
        return "duty_amount", "numeric"
    if header_lower in {"a.", "a", "as.", "annas"}:
        return "duty_as", "numeric"
    if header_lower in {"p.", "p", "pies"}:
        return "duty_p", "numeric"
    if "number" in header_lower:
        return f"column_{index:02d}", "numeric"
    if "quantity" in header_lower or "value" in header_lower or "rs" in header_lower:
        return f"column_{index:02d}", "numeric"
    if header_lower:
        return f"column_{index:02d}", "numeric"
    return f"column_{index:02d}", "numeric"


def _infer_columns(
    tokens: list[OCRToken],
    intervals: list[tuple[int, int]],
    first_row_top: int,
    table_width: int,
    row_interval: tuple[int, int] | None,
) -> list[PageColumn]:
    article_idx, unit_idx = _best_article_and_unit_columns(intervals, table_width)
    row_index = intervals.index(row_interval) if row_interval in intervals else None
    columns: list[PageColumn] = []
    for index, (left, right) in enumerate(intervals):
        header = _join_tokens_in_box(tokens, left, right, 0, max(10, first_row_top - 2))
        field_name, field_kind = _classify_header(header, index, article_idx, unit_idx, row_index)
        columns.append(
            PageColumn(
                index=index,
                left=left,
                right=right,
                header=header,
                field_name=field_name,
                field_kind=field_kind,
            )
        )
    return columns


def _ocr_cell(image: np.ndarray, field_kind: str) -> str:
    crop = trim_cell_border(image)
    if crop.size == 0:
        return ""
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()
    gray = enhance_contrast(gray)
    gray = denoise(gray)
    blank_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    if bool(cell_content_features(blank_mask)["likely_blank"]):
        return ""

    if field_kind == "numeric":
        configs = ["--psm 7 -c tessedit_char_whitelist=0123456789.,-;:"]
    elif field_kind == "rate":
        configs = ["--psm 7", "--psm 6"]
    else:
        configs = ["--psm 7", "--psm 6"]

    best_text = ""
    best_score = -1.0
    for _, variant in prepare_cell_variants(gray, "foreign_value_rs" if field_kind == "numeric" else "article"):
        for config in configs:
            data = pytesseract.image_to_data(
                variant,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
            pieces: list[str] = []
            confidences: list[float] = []
            for index, raw_text in enumerate(data["text"]):
                text = clean_whitespace(raw_text)
                if not text:
                    continue
                conf_raw = data["conf"][index]
                conf = float(conf_raw) if conf_raw != "-1" else -1.0
                if conf < 10:
                    continue
                pieces.append(text)
                confidences.append(conf / 100.0)
            value = _normalize_header_text(" ".join(pieces))
            score = (sum(confidences) / len(confidences)) if confidences else 0.0
            if field_kind == "numeric":
                digits = sum(character.isdigit() for character in value)
                score += digits * 0.02
            elif field_kind == "text":
                score += min(sum(character.isalpha() for character in value), 12) * 0.01
            if value and score >= best_score:
                best_text = value
                best_score = score
    return best_text


def _page_kind(tokens: list[OCRToken], first_row_top: int, table_width: int) -> tuple[str, str]:
    title = _join_tokens_in_box(tokens, 0, table_width, 0, max(10, first_row_top // 2))
    role_text = _join_tokens_in_box(tokens, 0, table_width, max(0, first_row_top // 5), max(15, first_row_top - 5))
    role_text = _normalize_header_text(role_text or title)
    role_lower = role_text.lower()
    if "indian" in role_lower or "home ports" in role_lower:
        return "indian", title
    if "foreign" in role_lower or "external ports" in role_lower:
        return "foreign", title
    return "unknown", title


def _extract_page(
    pdf_path: Path,
    page_number: int,
    dpi: int,
    debug_dir: Path,
) -> tuple[list[dict], dict]:
    image = render_page(pdf_path, page_number, dpi)
    gray, binary = preprocess_page(image)
    table_bbox = _detect_import_table_bbox(binary)

    table_image = extract_crop(image, table_bbox)
    table_gray = extract_crop(gray, table_bbox)
    table_binary = extract_crop(binary, table_bbox)
    intervals = _detect_columns(table_binary)
    if not intervals:
        raise RuntimeError("Unable to detect table columns on the page.")
    tokens = tesseract_tokens(table_gray)
    row_centers, row_interval = _infer_row_centers(
        table_gray,
        intervals,
        table_binary=table_binary,
        expected_row_start=_expected_total_import_row_start(pdf_path, page_number),
    )
    row_bounds = build_row_bound_map(row_centers, table_gray.shape[0])
    first_row_top = row_bounds[min(row_bounds)][0]
    columns = _infer_columns(tokens, intervals, first_row_top, table_gray.shape[1], row_interval)
    page_kind, title = _page_kind(tokens, first_row_top, table_gray.shape[1])

    overlay = cv2.cvtColor(table_gray, cv2.COLOR_GRAY2BGR)
    for row_no, (top, bottom) in row_bounds.items():
        cv2.line(overlay, (0, top), (overlay.shape[1], top), (0, 255, 0), 1)
        cv2.putText(overlay, str(row_no), (5, max(12, top + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 128, 0), 1)
    for column in columns:
        cv2.line(overlay, (column.left, 0), (column.left, overlay.shape[0]), (255, 0, 0), 1)
        cv2.putText(
            overlay,
            column.field_name,
            (column.left + 2, 14 + (12 * (column.index % 2))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 0, 0),
            1,
        )
    debug_path = debug_dir / f"{pdf_path.stem}_page_{page_number:03d}_grid.png"
    cv2.imwrite(str(debug_path), overlay)

    rows: list[dict] = []
    for row_no, (top, bottom) in sorted(row_bounds.items()):
        record = {
            "source_file": pdf_path.name,
            "page_number": page_number,
            "page_kind": page_kind,
            "title": title,
            "row_group_start": min(row_bounds),
            "row_group_end": max(row_bounds),
            "row_no": row_no,
            "article": "",
            "unit": "",
        }
        for column in columns:
            if column.field_name == "row_no":
                continue
            text = _join_tokens_in_box(tokens, column.left, column.right, top, bottom)
            if column.field_name in {"article", "unit"}:
                record[column.field_name] = text
            else:
                record[f"col_{column.index:02d}"] = text
        rows.append(record)

    header_record = {
        "source_file": pdf_path.name,
        "page_number": page_number,
        "page_kind": page_kind,
        "title": title,
        "row_no_min": min(row_bounds),
        "row_no_max": max(row_bounds),
        "row_group_start": min(row_bounds),
        "row_group_end": max(row_bounds),
        "table_bbox_x": table_bbox.x,
        "table_bbox_y": table_bbox.y,
        "table_bbox_width": table_bbox.width,
        "table_bbox_height": table_bbox.height,
        "row_strip_left": row_interval[0] if row_interval else "",
        "row_strip_right": row_interval[1] if row_interval else "",
        "debug_grid": str(debug_path),
    }
    for column in columns:
        header_record[f"col_{column.index:02d}_header"] = column.header
        header_record[f"col_{column.index:02d}_field"] = column.field_name

    logging.info(
        "Processed %s page %s: %s rows (%s-%s), %s columns, kind=%s",
        pdf_path.name,
        page_number,
        len(rows),
        min(row_bounds),
        max(row_bounds),
        len(columns),
        page_kind,
    )
    return rows, header_record


def _write_workbook(rows: list[dict], headers: list[dict], outdir: Path) -> Path:
    rows_df = pd.DataFrame(rows).fillna("")
    headers_df = pd.DataFrame(headers).fillna("")

    workbook_path = outdir / "import_tables.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        rows_df.to_excel(writer, index=False, sheet_name="all_rows")
        headers_df.to_excel(writer, index=False, sheet_name="page_headers")

        for pdf_name, subset in rows_df.groupby("source_file", sort=False):
            subset.to_excel(writer, index=False, sheet_name=_sanitize_sheet_name(Path(pdf_name).stem))

    return workbook_path


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    outdir = ensure_dir(Path(args.outdir).resolve())
    debug_dir = ensure_dir(outdir / "debug")
    setup_logging(outdir / "pipeline.log")
    configure_tesseract()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {input_dir}")

    rows: list[dict] = []
    headers: list[dict] = []
    for pdf_path in pdf_paths:
        import fitz  # local import keeps the top-level imports smaller for the CLI path

        with fitz.open(pdf_path) as pdf:
            page_count = pdf.page_count
        logging.info("Starting %s (%s pages)", pdf_path.name, page_count)
        for page_number in range(1, page_count + 1):
            try:
                page_rows, page_header = _extract_page(
                    pdf_path=pdf_path,
                    page_number=page_number,
                    dpi=args.dpi,
                    debug_dir=debug_dir,
                )
                rows.extend(page_rows)
                headers.append(page_header)
            except Exception as exc:
                logging.exception("Failed to process %s page %s", pdf_path.name, page_number)
                headers.append(
                    {
                        "source_file": pdf_path.name,
                        "page_number": page_number,
                        "page_kind": "error",
                        "title": "",
                        "error": str(exc),
                    }
                )

    workbook_path = _write_workbook(rows, headers, outdir)
    print(f"Workbook: {workbook_path}")
    print(f"Rows: {len(rows)}")
    print(f"Pages: {len(headers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
