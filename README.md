# Archival OCR Prototype V1

This repository contains a first-pass OCR pipeline for digitizing one archival Madras port trade table spread from a scanned PDF into Excel-ready rows.

## Repository Note

This repo is prepared for GitHub without archival data.

- No source PDFs are included
- No extracted CSV/XLSX outputs are included
- `raw/` and `output/` are kept as local-only working directories

## What It Does

- Renders two PDF pages at high resolution with `PyMuPDF`
- Preprocesses the scans with grayscale conversion, contrast enhancement, denoising, and binarization
- Detects the main ruled table region with `OpenCV`
- Uses OCR word boxes to estimate row positions from the row-number strip
- Maps OCR text into a fixed research schema
- Writes debug images, a row-level OCR debug JSON file, a CSV, and an XLSX
- Flags suspicious rows for manual review instead of forcing uncertain text into numeric cells

## Current Scope

Version 1 is intentionally format-specific.

- It assumes the sample Madras export table layout
- It expects a two-page spread where:
  - the left page contains `article/unit/foreign`
  - the right page contains `indian/totals/remarks`
- It defaults to the first spread in the sample PDF (`--start-page 1`)

This is not a generic all-table extractor.

## Setup

1. Install Python 3.12 or later.
2. Install Tesseract OCR on the machine and ensure `tesseract.exe` is available.
   On this machine it is installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`.
3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## How To Run

```bash
python main.py --input "path/to/local/sample.pdf" --outdir output
```

If you want to follow the repository convention, place your local PDF inside `raw/` and point `--input` to it.

Useful optional arguments:

- `--start-page 1`
- `--dpi 300`
- `--expected-rows 46`
- `--ocr-backend auto`

## Pipeline Stages

1. PDF rendering
2. Image preprocessing
3. Table-region detection
4. OCR token extraction
5. Row-grid fitting from row-number tokens
6. Field assignment by fixed column ranges
7. Cell-level OCR retry for unresolved cells
8. Numeric cleaning and review-flagging
9. CSV/XLSX export

## Output Structure

Running the command creates:

- `output/intermediate/rendered/`
- `output/intermediate/preprocessed/`
- `output/intermediate/tables/`
- `output/debug/`
- `output/table_output.csv`
- `output/table_output.xlsx`
- `output/pipeline.log`

The debug directory includes:

- table-box overlays
- row/column grid overlays
- `ocr_debug.json` with per-row cell text, confidence, and engine notes

## Notes On OCR Backends

- `auto` tries `PaddleOCR` first and falls back to `Tesseract`
- In restricted environments, `PaddleOCR` may fail because it needs writable model cache paths and sometimes first-run model downloads
- The pipeline still runs with `Tesseract`, which is the stable fallback for this prototype

## Limitations

- The row grid is tuned for this archival layout, not arbitrary tables
- OCR quality on faint or broken cells is still imperfect
- Multi-line headers are not reconstructed semantically in v1
- `Do.` values are preserved, but no separate normalized article column is exported yet
- Some sparse numeric columns may remain blank and be flagged for manual review
- The sample schema is assembled from a page pair, so `page_number` is stored as a spread string such as `1-2`
