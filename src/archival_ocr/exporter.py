"""CSV/XLSX export helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import OUTPUT_COLUMNS


def export_records(records: list[dict], outdir: Path) -> tuple[Path, Path]:
    dataframe = pd.DataFrame(records)
    dataframe = dataframe.reindex(columns=OUTPUT_COLUMNS)
    csv_path = outdir / "table_output.csv"
    xlsx_path = outdir / "table_output.xlsx"
    dataframe.to_csv(csv_path, index=False, na_rep="")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="table")
    return csv_path, xlsx_path
