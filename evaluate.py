from __future__ import annotations

import argparse
from typing import Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare OCR output against ground truth.")
    parser.add_argument("--truth", required=True, help="Path to the ground-truth CSV.")
    parser.add_argument("--pred", required=True, help="Path to the predicted CSV.")
    return parser.parse_args()


def normalize_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def comparable_fields(truth: pd.DataFrame, pred: pd.DataFrame) -> list[str]:
    return [column for column in truth.columns if column != "row_no" and column in pred.columns]


def row_differences(row: pd.Series, fields: Iterable[str]) -> list[tuple[str, str, str]]:
    differences: list[tuple[str, str, str]] = []
    for field in fields:
        expected = normalize_value(row[f"{field}_truth"])
        actual = normalize_value(row[f"{field}_pred"])
        if expected != actual:
            differences.append((field, expected, actual))
    return differences


def main() -> int:
    args = parse_args()
    truth = pd.read_csv(args.truth, dtype=str).fillna("")
    pred = pd.read_csv(args.pred, dtype=str).fillna("")

    if "row_no" not in truth.columns or "row_no" not in pred.columns:
        raise ValueError("Both CSV files must contain a row_no column.")

    fields = comparable_fields(truth, pred)
    merged = truth.merge(pred[["row_no"] + fields], on="row_no", how="inner", suffixes=("_truth", "_pred"))

    row_results: list[tuple[str, list[tuple[str, str, str]]]] = []
    fully_correct_rows = 0
    for _, row in merged.iterrows():
        differences = row_differences(row, fields)
        row_results.append((normalize_value(row["row_no"]), differences))
        if not differences:
            fully_correct_rows += 1

    print(f"Joined rows: {len(merged)}")
    print(f"Rows fully correct: {fully_correct_rows}")
    print("Per-field exact matches:")
    for field in fields:
        count = sum(
            normalize_value(row[f"{field}_truth"]) == normalize_value(row[f"{field}_pred"])
            for _, row in merged.iterrows()
        )
        print(f"  {field}: {count}/{len(merged)}")

    print("Differing rows:")
    differing = False
    for row_no, differences in row_results:
        if not differences:
            continue
        differing = True
        print(f"  row {row_no}")
        for field, expected, actual in differences:
            print(f"    {field}")
            print(f"      expected: {expected}")
            print(f"      actual:   {actual}")
    if not differing:
        print("  none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
