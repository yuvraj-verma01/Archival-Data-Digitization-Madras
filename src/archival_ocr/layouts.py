"""Sample-oriented layout presets for the Madras archival tables."""

from dataclasses import dataclass, replace


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
    "madras_exports_p1_p2": {
        "left": MADRAS_EXPORTS_LEFT,
        "right": MADRAS_EXPORTS_RIGHT,
        "expected_rows": 46,
        "row_start": 1,
    },
    "madras_exports_p3_p4": {
        "left": replace(
            MADRAS_EXPORTS_LEFT,
            table_bbox_ratio=(0.15518, 0.18757, 0.74688, 0.72007),
        ),
        "right": replace(
            MADRAS_EXPORTS_RIGHT,
            table_bbox_ratio=(0.11890, 0.15279, 0.75655, 0.68900),
        ),
        "expected_rows": 47,
        "row_start": 47,
    },
    "madras_exports_p5_p6": {
        "left": replace(
            MADRAS_EXPORTS_LEFT,
            table_bbox_ratio=(0.13100, 0.17474, 0.75816, 0.72406),
        ),
        "right": replace(
            MADRAS_EXPORTS_RIGHT,
            table_bbox_ratio=(0.12495, 0.26767, 0.74204, 0.62771),
        ),
        "expected_rows": 47,
        "row_start": 94,
    },
    "madras_exports_p7_p8": {
        "left": replace(
            MADRAS_EXPORTS_LEFT,
            table_bbox_ratio=(0.11165, 0.15365, 0.68077, 0.75086),
        ),
        "right": replace(
            MADRAS_EXPORTS_RIGHT,
            table_bbox_ratio=(0.34301, 0.18358, 0.58565, 0.71351),
        ),
        "expected_rows": 47,
        "row_start": 141,
    },
    "madras_exports_p9_p10": {
        "left": replace(
            MADRAS_EXPORTS_LEFT,
            table_bbox_ratio=(0.10480, 0.18358, 0.72511, 0.51596),
            fallback_body_bottom_ratio=0.760,
        ),
        "right": replace(
            MADRAS_EXPORTS_RIGHT,
            table_bbox_ratio=(0.08384, 0.14994, 0.74728, 0.54618),
            fallback_body_bottom_ratio=0.760,
        ),
        "expected_rows": 15,
        "row_start": 188,
    },
    "madras_exports_v1": {
        "left": MADRAS_EXPORTS_LEFT,
        "right": MADRAS_EXPORTS_RIGHT,
        "expected_rows": 46,
        "row_start": 1,
    },
}

SPREAD_LAYOUT_BY_START_PAGE = {
    1: "madras_exports_p1_p2",
    3: "madras_exports_p3_p4",
    5: "madras_exports_p5_p6",
    7: "madras_exports_p7_p8",
    9: "madras_exports_p9_p10",
}

SPREAD_LAYOUT_BY_START_PAGE_186566 = {
    1: "madras_exports_186566_p1_p2",
    3: "madras_exports_186566_p3_p4",
    5: "madras_exports_186566_p5_p6",
    7: "madras_exports_186566_p7_p8",
    9: "madras_exports_186566_p9_p10",
}

for start_page, base_layout_name in SPREAD_LAYOUT_BY_START_PAGE.items():
    SPREAD_LAYOUTS[f"madras_exports_186566_p{start_page}_p{start_page + 1}"] = SPREAD_LAYOUTS[base_layout_name]


def resolve_spread_layout(layout_name: str, start_page: int) -> dict:
    if layout_name == "madras_exports_auto":
        resolved_name = SPREAD_LAYOUT_BY_START_PAGE.get(start_page)
        if resolved_name is None:
            raise KeyError(
                f"No auto layout preset is defined for start page {start_page}. "
                f"Available starts: {sorted(SPREAD_LAYOUT_BY_START_PAGE)}"
            )
        return SPREAD_LAYOUTS[resolved_name]
    if layout_name == "madras_exports_186566_auto":
        resolved_name = SPREAD_LAYOUT_BY_START_PAGE_186566.get(start_page)
        if resolved_name is None:
            raise KeyError(
                f"No 186566 auto layout preset is defined for start page {start_page}. "
                f"Available starts: {sorted(SPREAD_LAYOUT_BY_START_PAGE_186566)}"
            )
        return SPREAD_LAYOUTS[resolved_name]
    return SPREAD_LAYOUTS[layout_name]
