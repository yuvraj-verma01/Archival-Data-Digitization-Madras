"""Shared dataclasses used across the pipeline."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


@dataclass
class OCRToken:
    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int

    @property
    def x_center(self) -> float:
        return self.x + (self.width / 2.0)

    @property
    def y_center(self) -> float:
        return self.y + (self.height / 2.0)


@dataclass
class OCRCellResult:
    text: str
    confidence: float | None
    engine: str


@dataclass
class PageArtifacts:
    page_number: int
    image: np.ndarray
    gray: np.ndarray
    binary: np.ndarray
    table_bbox: BoundingBox
    table_image: np.ndarray
    table_gray: np.ndarray
    table_binary: np.ndarray
    tokens: list[OCRToken]
    row_centers: list[float]
    row_bounds: list[tuple[int, int]]
