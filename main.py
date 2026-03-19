from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from archival_ocr.pipeline import run_pipeline
from archival_ocr.utils import ensure_dir, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one archival Madras trade table spread into CSV/XLSX."
    )
    parser.add_argument("--input", required=True, help="Path to the scanned PDF.")
    parser.add_argument("--outdir", required=True, help="Directory where outputs will be written.")
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="1-based starting page for the two-page table spread. Default: 1",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Render DPI for PDF pages. Higher values improve OCR but cost more time. Default: 300",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=None,
        help="Override the layout default row count. Default: preset value (46)",
    )
    parser.add_argument(
        "--layout",
        default="madras_exports_v1",
        help="Layout preset to use. Default: madras_exports_v1",
    )
    parser.add_argument(
        "--ocr-backend",
        choices=["auto", "tesseract", "paddle"],
        default="auto",
        help="OCR backend preference. 'auto' tries Paddle first and falls back to Tesseract.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.input).resolve()
    outdir = ensure_dir(Path(args.outdir).resolve())
    setup_logging(outdir / "pipeline.log")

    if not pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    result = run_pipeline(
        pdf_path=pdf_path,
        outdir=outdir,
        start_page=args.start_page,
        dpi=args.dpi,
        expected_rows=args.expected_rows,
        layout_name=args.layout,
        ocr_backend=args.ocr_backend,
    )

    print(f"Rows exported: {result['rows']}")
    print(f"CSV: {result['csv_path']}")
    print(f"XLSX: {result['xlsx_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
