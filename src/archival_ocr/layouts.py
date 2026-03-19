"""Sample-oriented layout presets for the Madras archival tables."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PageLayout:
    side: str
    fallback_body_top_ratio: float
    fallback_body_bottom_ratio: float
    row_number_ratio: tuple[float, float]
    columns: dict[str, tuple[float, float]]


MADRAS_EXPORTS_LEFT = PageLayout(
    side="left",
    fallback_body_top_ratio=0.19,
    fallback_body_bottom_ratio=0.985,
    row_number_ratio=(0.0, 0.075),
    columns={
        "row_no": (0.0, 0.075),
        "article": (0.075, 0.383),
        "unit": (0.383, 0.417),
        "foreign_quantity": (0.417, 0.538),
        "foreign_value_rs": (0.538, 0.676),
        "foreign_duty_rate": (0.676, 0.890),
        "foreign_duty_rs": (0.890, 0.952),
        "foreign_duty_as": (0.952, 0.975),
        "foreign_duty_p": (0.975, 0.998),
    },
)

MADRAS_EXPORTS_RIGHT = PageLayout(
    side="right",
    fallback_body_top_ratio=0.17,
    fallback_body_bottom_ratio=0.985,
    row_number_ratio=(0.0, 0.04),
    columns={
        "row_no": (0.0, 0.04),
        "indian_quantity": (0.04, 0.138),
        "indian_value_rs": (0.138, 0.243),
        "indian_duty_rate": (0.243, 0.445),
        "indian_duty_rs": (0.445, 0.573),
        "indian_duty_as": (0.573, 0.639),
        "indian_duty_p": (0.639, 0.703),
        # The ruled right page has a narrow "Total Merchandize, Foreign and Indian Ports"
        # value column followed by a separate rightmost remarks column.
        "total_value": (0.735, 0.852),
        "remarks": (0.852, 0.985),
    },
)

SPREAD_LAYOUTS = {
    "madras_exports_v1": {
        "left": MADRAS_EXPORTS_LEFT,
        "right": MADRAS_EXPORTS_RIGHT,
        "expected_rows": 46,
    }
}
