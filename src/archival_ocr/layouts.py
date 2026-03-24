"""Sample-oriented layout presets for the Madras archival tables."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PageLayout:
    side: str
    table_bbox_ratio: tuple[float, float, float, float]
    fallback_body_top_ratio: float
    fallback_body_bottom_ratio: float
    row_number_ratio: tuple[float, float]
    columns: dict[str, tuple[float, float]]


MADRAS_EXPORTS_LEFT = PageLayout(
    side="left",
    table_bbox_ratio=(0.07295, 0.15308, 0.67191, 0.75114),
    fallback_body_top_ratio=0.188,
    fallback_body_bottom_ratio=0.982,
    row_number_ratio=(0.0, 0.075),
    columns={
        "row_no": (0.0, 0.05639),  # measured at 300 DPI on page 1 from the full-page debug grid
        "article": (0.05639, 0.37193),  # measured at 300 DPI on page 1 from the full-page debug grid
        "unit": (0.37193, 0.44511),  # measured at 300 DPI on page 1 from the full-page debug grid
        "foreign_quantity": (0.44511, 0.58188),  # measured at 300 DPI on page 1 from the full-page debug grid
        "foreign_value_rs": (0.58188, 0.71506),  # measured at 300 DPI on page 1 from the full-page debug grid
        "foreign_duty_rate": (0.71506, 0.90342),  # measured at 300 DPI on page 1 from the full-page debug grid
        "foreign_duty_rs": (0.90342, 0.95861),  # measured at 300 DPI on page 1 from the full-page debug grid
        "foreign_duty_as": (0.95861, 0.97900),  # measured at 300 DPI on page 1 from the full-page debug grid
        "foreign_duty_p": (0.97900, 0.99760),  # measured at 300 DPI on page 1 from the full-page debug grid
    },
)

MADRAS_EXPORTS_RIGHT = PageLayout(
    side="right",
    table_bbox_ratio=(0.09714, 0.17474, 0.69972, 0.71864),
    fallback_body_top_ratio=0.165,
    fallback_body_bottom_ratio=0.982,
    row_number_ratio=(0.0, 0.04),
    columns={
        "row_no": (0.0, 0.03571),  # measured at 300 DPI on page 2 from the full-page debug grid
        "indian_quantity": (0.03571, 0.14516),  # measured at 300 DPI on page 2 from the full-page debug grid
        "indian_value_rs": (0.14516, 0.25576),  # measured at 300 DPI on page 2 from the full-page debug grid
        "indian_duty_rate": (0.25576, 0.47005),  # measured at 300 DPI on page 2 from the full-page debug grid
        "indian_duty_rs": (0.47005, 0.60138),  # measured at 300 DPI on page 2 from the full-page debug grid
        "indian_duty_as": (0.60138, 0.66912),  # measured at 300 DPI on page 2 from the full-page debug grid
        "indian_duty_p": (0.66912, 0.73675),  # measured at 300 DPI on page 2 from the full-page debug grid
        # The ruled right page has a narrow "Total Merchandize, Foreign and Indian Ports"
        # value column followed by a separate rightmost remarks column.
        "total_value": (0.73675, 0.85369),  # measured at 300 DPI on page 2 from the full-page debug grid
        "remarks": (0.85369, 0.98500),  # measured at 300 DPI on page 2 from the full-page debug grid
    },
)

SPREAD_LAYOUTS = {
    "madras_exports_v1": {
        "left": MADRAS_EXPORTS_LEFT,
        "right": MADRAS_EXPORTS_RIGHT,
        "expected_rows": 46,
    }
}
