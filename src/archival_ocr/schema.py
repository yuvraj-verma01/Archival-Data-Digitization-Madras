"""Schema and field group definitions."""

OUTPUT_COLUMNS = [
    "source_file",
    "page_number",
    "row_no",
    "article",
    "unit",
    "foreign_quantity",
    "foreign_value_rs",
    "foreign_duty_rate",
    "foreign_duty_rs",
    "foreign_duty_as",
    "foreign_duty_p",
    "indian_quantity",
    "indian_value_rs",
    "indian_duty_rate",
    "indian_duty_rs",
    "indian_duty_as",
    "indian_duty_p",
    "total_value",
    "remarks",
    "raw_ocr_text",
    "llm_notes",
    "confidence_flag",
]

LEFT_PAGE_FIELDS = [
    "row_no",
    "article",
    "unit",
    "foreign_quantity",
    "foreign_value_rs",
    "foreign_duty_rate",
    "foreign_duty_rs",
    "foreign_duty_as",
    "foreign_duty_p",
]

RIGHT_PAGE_FIELDS = [
    "row_no",
    "indian_quantity",
    "indian_value_rs",
    "indian_duty_rate",
    "indian_duty_rs",
    "indian_duty_as",
    "indian_duty_p",
    "total_value",
    "remarks",
]

NUMERIC_FIELDS = {
    "row_no",
    "foreign_quantity",
    "foreign_value_rs",
    "foreign_duty_rs",
    "foreign_duty_as",
    "foreign_duty_p",
    "indian_quantity",
    "indian_value_rs",
    "indian_duty_rs",
    "indian_duty_as",
    "indian_duty_p",
    "total_value",
}

TEXT_FIELDS = {
    "article",
    "unit",
    "foreign_duty_rate",
    "indian_duty_rate",
    "remarks",
}

RATE_FIELDS = {"foreign_duty_rate", "indian_duty_rate"}

CELL_PRIMARY_FIELDS = {
    "article",
    "foreign_value_rs",
    "indian_value_rs",
    "total_value",
}

LONG_TEXT_FIELDS = {"article", "remarks"}
