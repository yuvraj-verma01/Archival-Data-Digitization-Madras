"""Template-driven table detection and row alignment."""

from __future__ import annotations

import logging
from statistics import median

import cv2
import numpy as np

from .layouts import PageLayout
from .models import BoundingBox, OCRToken
from .ocr_engines import parse_integer_candidate


def detect_table_bbox(binary_image: np.ndarray) -> BoundingBox:
    """
    Fallback detector for the main ruled table.
    """
    height, width = binary_image.shape
    trim_y = int(height * 0.05)
    trim_x = int(width * 0.05)
    inner = binary_image[trim_y : height - trim_y, trim_x : width - trim_x]

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(20, inner.shape[0] // 30))
    )
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(20, inner.shape[1] // 20), 1)
    )
    vertical = cv2.morphologyEx(inner, cv2.MORPH_OPEN, vertical_kernel)
    horizontal = cv2.morphologyEx(inner, cv2.MORPH_OPEN, horizontal_kernel)
    grid = cv2.add(vertical, horizontal)

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[int, int, int, int, int]] = []
    for contour in contours:
        x, y, width_box, height_box = cv2.boundingRect(contour)
        area = width_box * height_box
        if area >= inner.shape[0] * inner.shape[1] * 0.02:
            candidates.append((area, x, y, width_box, height_box))
    if not candidates:
        raise RuntimeError("Unable to detect a table region in the page image.")

    _, x, y, width_box, height_box = max(candidates)
    return BoundingBox(
        x=x + trim_x,
        y=y + trim_y,
        width=width_box,
        height=height_box,
    )


def template_table_bbox(image_shape: tuple[int, int], layout: PageLayout) -> BoundingBox:
    height, width = image_shape[:2]
    x_ratio, y_ratio, width_ratio, height_ratio = layout.table_bbox_ratio
    return BoundingBox(
        x=int(width * x_ratio),
        y=int(height * y_ratio),
        width=int(width * width_ratio),
        height=int(height * height_ratio),
    )


def resolve_table_bbox(binary_image: np.ndarray, layout: PageLayout) -> BoundingBox:
    """
    Use the fixed template for this archival format and fall back to rule detection
    if the template would be invalid.
    """
    template_bbox = template_table_bbox(binary_image.shape, layout)
    if template_bbox.width > 0 and template_bbox.height > 0:
        return template_bbox
    return detect_table_bbox(binary_image)


def extract_crop(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    return image[bbox.y : bbox.y2, bbox.x : bbox.x2]


def draw_table_overlay(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    overlay = image.copy()
    cv2.rectangle(overlay, (bbox.x, bbox.y), (bbox.x2, bbox.y2), (0, 0, 255), 3)
    return overlay


def collect_row_number_candidates(
    tokens: list[OCRToken],
    layout: PageLayout,
    table_width: int,
    table_height: int,
    expected_rows: int,
) -> dict[int, float]:
    row_x0 = int(layout.row_number_ratio[0] * table_width)
    row_x1 = int(layout.row_number_ratio[1] * table_width)
    grouped: dict[int, list[tuple[float, float]]] = {}
    for token in tokens:
        if token.x_center < row_x0 or token.x_center > row_x1:
            continue
        if token.y_center < table_height * layout.fallback_body_top_ratio * 0.75:
            continue
        row_number = parse_integer_candidate(token.text, expected_rows)
        if row_number is None:
            continue
        grouped.setdefault(row_number, []).append((token.confidence, token.y_center))

    anchors: dict[int, float] = {}
    for row_number, candidates in grouped.items():
        candidates.sort(key=lambda item: item[0], reverse=True)
        anchors[row_number] = candidates[0][1]
    return anchors


def _fallback_row_center_map(layout: PageLayout, table_height: int, expected_rows: int) -> dict[int, float]:
    top = table_height * layout.fallback_body_top_ratio
    bottom = table_height * layout.fallback_body_bottom_ratio
    step = (bottom - top) / max(expected_rows - 1, 1)
    return {row_number: top + (step * (row_number - 1)) for row_number in range(1, expected_rows + 1)}


def build_row_center_map(
    tokens: list[OCRToken],
    layout: PageLayout,
    table_width: int,
    table_height: int,
    expected_rows: int,
) -> tuple[dict[int, float], dict[int, float]]:
    """
    Build a row-number keyed center map using a page-specific line fit over OCR'd
    row-number anchors. This keeps the left and right pages aligned by row number
    without over-trusting individual bad anchors.
    """
    anchors = collect_row_number_candidates(tokens, layout, table_width, table_height, expected_rows)
    if len(anchors) < 4:
        logging.warning(
            "Row-number OCR was sparse on the %s page; using the template row grid.",
            layout.side,
        )
        return _fallback_row_center_map(layout, table_height, expected_rows), anchors

    best_score = None
    best_step = None
    best_intercept = None
    step_guess = table_height / max(expected_rows + 2, 1)
    min_step = max(12.0, step_guess * 0.65)
    max_step = max(min_step + 0.5, step_guess * 1.35)
    min_intercept = table_height * 0.04
    max_intercept = table_height * 0.24

    for step in np.arange(min_step, max_step + 0.1, 0.5):
        for intercept in np.arange(min_intercept, max_intercept + 0.1, 0.5):
            score = 0
            for row_number, y_center in anchors.items():
                predicted_row = round((y_center - intercept) / step)
                if predicted_row == row_number:
                    score += 3
                elif abs(predicted_row - row_number) == 1:
                    score += 1
            if best_score is None or score > best_score:
                best_score = score
                best_step = step
                best_intercept = intercept

    if best_step is None or best_intercept is None:
        logging.warning(
            "Unable to fit a row grid on the %s page; using the template row grid.",
            layout.side,
        )
        return _fallback_row_center_map(layout, table_height, expected_rows), anchors

    centers = {
        row_number: best_intercept + (best_step * row_number)
        for row_number in range(1, expected_rows + 1)
    }
    return centers, anchors


def build_row_bound_map(row_centers: dict[int, float], table_height: int) -> dict[int, tuple[int, int]]:
    ordered_rows = sorted(row_centers.items())
    centers = [center for _, center in ordered_rows]
    if len(centers) == 1:
        center = int(centers[0])
        row_number = ordered_rows[0][0]
        return {row_number: (max(0, center - 10), min(table_height, center + 10))}

    gaps = [centers[index + 1] - centers[index] for index in range(len(centers) - 1)]
    median_gap = median(gaps)
    bounds: dict[int, tuple[int, int]] = {}
    for index, (row_number, center) in enumerate(ordered_rows):
        if index == 0:
            top = int(max(0, center - (median_gap / 2)))
        else:
            top = int((ordered_rows[index - 1][1] + center) / 2)
        if index == len(ordered_rows) - 1:
            bottom = int(min(table_height, center + (median_gap / 2)))
        else:
            bottom = int((center + ordered_rows[index + 1][1]) / 2)
        bounds[row_number] = (max(0, top), min(table_height, bottom))
    return bounds


def draw_grid_overlay(
    table_image: np.ndarray,
    row_bounds: dict[int, tuple[int, int]],
    columns: dict[str, tuple[float, float]],
) -> np.ndarray:
    overlay = table_image.copy()
    height, width = overlay.shape[:2]
    for _, (top, bottom) in sorted(row_bounds.items()):
        cv2.line(overlay, (0, top), (width, top), (0, 255, 0), 1)
        cv2.line(overlay, (0, bottom), (width, bottom), (0, 255, 0), 1)
    for label, (start_ratio, end_ratio) in columns.items():
        x = int(width * start_ratio)
        cv2.line(overlay, (x, 0), (x, height), (255, 0, 0), 1)
        cv2.putText(
            overlay,
            label,
            (x + 2, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
        end_x = int(width * end_ratio)
        cv2.line(overlay, (end_x, 0), (end_x, height), (255, 0, 0), 1)
    return overlay
