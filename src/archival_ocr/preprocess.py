"""Image preprocessing for archival scans and cell OCR."""

from __future__ import annotations

import cv2
import numpy as np

from .schema import LONG_TEXT_FIELDS, NUMERIC_FIELDS, RATE_FIELDS


def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    return clahe.apply(gray)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, 12, 7, 21)


def deskew_if_needed(gray: np.ndarray) -> np.ndarray:
    """
    The sample pages are already close to upright.
    Keep the hook explicit, but avoid a rotation that can distort table cells.
    """
    return gray


def binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )


def preprocess_page(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = to_grayscale(image)
    gray = deskew_if_needed(gray)
    gray = enhance_contrast(gray)
    gray = denoise(gray)
    binary = binarize(gray)
    return gray, binary


def trim_cell_border(gray: np.ndarray) -> np.ndarray:
    if gray.size == 0:
        return gray
    height, width = gray.shape[:2]
    margin_y = max(1, int(height * 0.08))
    margin_x = max(1, int(width * 0.04))
    if height <= (margin_y * 2) or width <= (margin_x * 2):
        return gray
    return gray[margin_y : height - margin_y, margin_x : width - margin_x]


def threshold_to_white_background(gray: np.ndarray) -> np.ndarray:
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def adaptive_to_white_background(gray: np.ndarray) -> np.ndarray:
    inverted = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        11,
    )
    return 255 - inverted


def remove_table_rules(binary_white: np.ndarray) -> np.ndarray:
    """
    Remove long vertical or horizontal rules from a cell crop.
    Input is black text on a white background.
    """
    if binary_white.size == 0:
        return binary_white
    inverted = 255 - binary_white
    height, width = inverted.shape[:2]
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, height // 2)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, width // 2), 1))
    vertical = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, vertical_kernel)
    horizontal = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horizontal_kernel)
    cleaned = cv2.subtract(inverted, cv2.add(vertical, horizontal))
    return 255 - cleaned


def cell_content_features(binary_crop: np.ndarray) -> dict[str, float | int | bool]:
    """
    Estimate whether a crop is likely blank before OCR.
    The page-level binary uses white ink on a black background.
    """
    if binary_crop.size == 0:
        return {
            "ink_density": 0.0,
            "component_count": 0,
            "avg_stroke_width": 0.0,
            "max_component_ratio": 0.0,
            "likely_blank": True,
        }

    mask = (binary_crop > 0).astype(np.uint8)
    ink_density = float(mask.mean())
    component_count = 0
    avg_stroke_width = 0.0
    max_component_ratio = 0.0

    if np.any(mask):
        min_component_area = max(4, int(mask.size * 0.00015))
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = [
            int(stat[cv2.CC_STAT_AREA])
            for stat in stats[1:]
            if int(stat[cv2.CC_STAT_AREA]) >= min_component_area
        ]
        component_count = len(areas)
        if areas:
            max_component_ratio = max(areas) / float(mask.size)
        distances = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        positive = distances[distances > 0]
        if positive.size:
            avg_stroke_width = float(positive.mean() * 2.0)

    likely_blank = (
        ink_density < 0.0015
        and component_count <= 2
        and max_component_ratio < 0.0015
    ) or (
        ink_density < 0.0040
        and component_count <= 4
        and avg_stroke_width < 1.25
        and max_component_ratio < 0.0035
    )

    return {
        "ink_density": ink_density,
        "component_count": component_count,
        "avg_stroke_width": avg_stroke_width,
        "max_component_ratio": max_component_ratio,
        "likely_blank": likely_blank,
    }


def prepare_cell_variants(gray: np.ndarray, field_name: str) -> list[tuple[str, np.ndarray]]:
    """
    Build a compact preprocessing ensemble tailored to the column type.
    """
    base = trim_cell_border(gray)
    if base.size == 0:
        return [("raw", gray)]

    contrast = enhance_contrast(base)
    softened = cv2.GaussianBlur(contrast, (3, 3), 0)
    denoised = denoise(contrast)

    variants: list[tuple[str, np.ndarray]] = []
    if field_name in NUMERIC_FIELDS:
        variants.extend(
            [
                ("numeric_binary", remove_table_rules(threshold_to_white_background(softened))),
                ("numeric_adaptive", remove_table_rules(adaptive_to_white_background(contrast))),
                ("numeric_gray", contrast),
            ]
        )
    elif field_name in RATE_FIELDS:
        variants.extend(
            [
                ("rate_gray", denoised),
                ("rate_binary", threshold_to_white_background(contrast)),
            ]
        )
    elif field_name in LONG_TEXT_FIELDS:
        variants.extend(
            [
                ("text_gray", denoised),
                ("text_binary", remove_table_rules(threshold_to_white_background(denoised))),
                ("text_adaptive", remove_table_rules(adaptive_to_white_background(contrast))),
            ]
        )
    else:
        variants.extend(
            [
                ("default_binary", remove_table_rules(threshold_to_white_background(contrast))),
                ("default_gray", contrast),
            ]
        )

    return variants
