"""Table detection and row-grid estimation."""

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
    Detect the main ruled table by combining vertical and horizontal line masks.
    The search trims page borders first so the scan frame does not win.
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


def extract_crop(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    return image[bbox.y : bbox.y2, bbox.x : bbox.x2]


def draw_table_overlay(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    overlay = image.copy()
    cv2.rectangle(overlay, (bbox.x, bbox.y), (bbox.x2, bbox.y2), (0, 0, 255), 3)
    return overlay


def fit_row_centers(
    tokens: list[OCRToken],
    layout: PageLayout,
    table_width: int,
    table_height: int,
    expected_rows: int,
) -> list[float]:
    """
    Fit a near-linear row grid from OCR row-number tokens.
    The archival tables are almost evenly spaced, so a robust line fit is more
    stable than trusting every OCR token or every visible rule.
    """
    row_x0 = int(layout.row_number_ratio[0] * table_width)
    row_x1 = int(layout.row_number_ratio[1] * table_width)
    candidates: list[tuple[int, float]] = []
    for token in tokens:
        if token.x_center < row_x0 or token.x_center > row_x1:
            continue
        if token.y_center < table_height * layout.fallback_body_top_ratio * 0.75:
            continue
        value = parse_integer_candidate(token.text, expected_rows)
        if value is None:
            continue
        candidates.append((value, token.y_center))

    if len(candidates) < 6:
        logging.warning(
            "Row-number OCR was sparse on the %s page; using the layout fallback grid.",
            layout.side,
        )
        top = table_height * layout.fallback_body_top_ratio
        bottom = table_height * layout.fallback_body_bottom_ratio
        step = (bottom - top) / max(expected_rows - 1, 1)
        return [top + (step * index) for index in range(expected_rows)]

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
            for parsed_row, y_center in candidates:
                predicted_row = round((y_center - intercept) / step)
                if predicted_row == parsed_row:
                    score += 3
                elif abs(predicted_row - parsed_row) == 1:
                    score += 1
            if best_score is None or score > best_score:
                best_score = score
                best_step = step
                best_intercept = intercept

    if best_step is None or best_intercept is None:
        logging.warning(
            "Unable to fit an OCR-driven row grid on the %s page; using the layout fallback grid.",
            layout.side,
        )
        top = table_height * layout.fallback_body_top_ratio
        bottom = table_height * layout.fallback_body_bottom_ratio
        step = (bottom - top) / max(expected_rows - 1, 1)
        return [top + (step * index) for index in range(expected_rows)]

    return [
        best_intercept + (best_step * row_number)
        for row_number in range(1, expected_rows + 1)
    ]


def build_row_bounds(row_centers: list[float], table_height: int) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    if len(row_centers) == 1:
        center = int(row_centers[0])
        return [(max(0, center - 10), min(table_height, center + 10))]
    gaps = [row_centers[index + 1] - row_centers[index] for index in range(len(row_centers) - 1)]
    median_gap = median(gaps)
    for index, center in enumerate(row_centers):
        if index == 0:
            top = int(max(0, center - (median_gap / 2)))
        else:
            top = int((row_centers[index - 1] + center) / 2)
        if index == len(row_centers) - 1:
            bottom = int(min(table_height, center + (median_gap / 2)))
        else:
            bottom = int((center + row_centers[index + 1]) / 2)
        bounds.append((max(0, top), min(table_height, bottom)))
    return bounds


def draw_grid_overlay(
    table_image: np.ndarray,
    row_bounds: list[tuple[int, int]],
    columns: dict[str, tuple[float, float]],
) -> np.ndarray:
    overlay = table_image.copy()
    height, width = overlay.shape[:2]
    for top, bottom in row_bounds:
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
