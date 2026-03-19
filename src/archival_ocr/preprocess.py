"""Image preprocessing for archival scans."""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    return clahe.apply(gray)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, 12, 7, 21)


def deskew_if_needed(gray: np.ndarray) -> np.ndarray:
    """
    The sample pages are close to upright already.
    This hook stays explicit without applying a rotation that could distort
    narrow archival columns in version 1.
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
