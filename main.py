from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fitz
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from archival_ocr.exporter import export_records
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
        default="madras_exports_auto",
        help="Layout preset to use. Default: madras_exports_auto",
    )
    parser.add_argument(
        "--ocr-backend",
        choices=["auto", "tesseract", "paddle"],
        default="auto",
        help="OCR backend preference. 'auto' tries Paddle first and falls back to Tesseract.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save full-page debug grid overlays for the left and right pages.",
    )
    parser.add_argument(
        "--no-paddle",
        action="store_true",
        help="Exclude PaddleOCR from the optional cell-level OCR ensemble.",
    )
    parser.add_argument(
        "--no-glm",
        action="store_true",
        help="Exclude the local GLM OCR Ollama model from the optional ensemble.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the local Ollama-based weak-row validator.",
    )
    parser.add_argument(
        "--all-spreads",
        action="store_true",
        help="Run extraction across every two-page spread in the PDF and write a combined output.",
    )
    return parser.parse_args()


def _spread_start_pages(pdf_path: Path) -> list[int]:
    with fitz.open(pdf_path) as document:
        page_count = document.page_count
    return list(range(1, page_count + 1, 2))


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.input).resolve()
    outdir = ensure_dir(Path(args.outdir).resolve())
    setup_logging(outdir / "pipeline.log")

    if not pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    if args.all_spreads:
        spread_dirs = []
        combined_frames = []
        for start_page in _spread_start_pages(pdf_path):
            spread_outdir = ensure_dir(outdir / "spreads" / f"p{start_page:03d}_p{start_page + 1:03d}")
            result = run_pipeline(
                pdf_path=pdf_path,
                outdir=spread_outdir,
                start_page=start_page,
                dpi=args.dpi,
                expected_rows=args.expected_rows,
                layout_name=args.layout,
                ocr_backend=args.ocr_backend,
                debug=args.debug,
                use_paddle=(not args.no_paddle),
                use_glm=(not args.no_glm),
                use_llm=(not args.no_llm),
            )
            spread_dirs.append(spread_outdir)
            combined_frames.append(pd.read_csv(result["csv_path"]))
            print(f"Completed spread {start_page}-{start_page + 1}: {result['rows']} rows")

        combined_records = pd.concat(combined_frames, ignore_index=True).to_dict(orient="records")
        csv_path, xlsx_path = export_records(combined_records, outdir)
        print(f"Combined spreads: {len(spread_dirs)}")
        print(f"CSV: {csv_path}")
        print(f"XLSX: {xlsx_path}")
    else:
        result = run_pipeline(
            pdf_path=pdf_path,
            outdir=outdir,
            start_page=args.start_page,
            dpi=args.dpi,
            expected_rows=args.expected_rows,
            layout_name=args.layout,
            ocr_backend=args.ocr_backend,
            debug=args.debug,
            use_paddle=(not args.no_paddle),
            use_glm=(not args.no_glm),
            use_llm=(not args.no_llm),
        )

        print(f"Rows exported: {result['rows']}")
        print(f"CSV: {result['csv_path']}")
        print(f"XLSX: {result['xlsx_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
