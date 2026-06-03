# Architecture Reuse: From Form Parser to Spec-Region Detector

The document-understanding architecture built in this repo — the **Financial
Form Text Extractor** — was generalised and reused in a sibling project,
[**G-GEN**](https://github.com/joehiggi/g-gen), to ingest CNC
**product-specification PDFs** and turn them into G-code.

This note records the lineage so the two capstone artifacts stay legible as one
system of patterns.

## What carried over

| SIADS-699 (this repo) | G-GEN reuse |
|---|---|
| PDF/PNG/JPEG scanned forms | Product-spec PDFs (digital, scanned, or table) |
| YOLOv8 detects `header` / `body` / `footer` | YOLOv8 detects `title_block` / `dimension_table` / `drawing` / `notes` / `tolerance_block` / `material_block` |
| Tesseract 5 OCR on the body region | Tesseract 5 OCR on text regions (drawings skipped as pictorial) |
| Structured text → PostgreSQL | Structured `ProductSpec` → prompt → RAG + LLM → G-code |
| `models/yolov8-run/` training kit | `models/spec-region-run/` training kit (same layout) |
| Great Lakes SLURM `batch_job.sh` | Same SLURM kit + a Kaggle T4 notebook |
| Label Studio hand-labelled data | Synthetic, perfectly-labelled generator (no manual annotation) |

## The same training kit shape

G-GEN's `models/spec-region-run/` mirrors this repo's `models/yolov8-run/`:

- `src/train.py` — the reproducible YOLOv8 fine-tune CLI, ported nearly verbatim
- `src/<dataset>.yaml` — dataset config with the region class names
- `src/batch_job.sh` — SLURM GPU job
- `runs/` + exported `best.pt` — artifacts (git-ignored)

The chief difference: where this project labelled real scans in Label Studio,
G-GEN synthesises spec sheets with known bounding boxes
(`scripts/generate_spec_dataset.py`), because engineering spec layouts are
regular enough to generate. Real Label-Studio exports still drop into the same
`images/` + `labels/` split when available.

## Where to look in G-GEN

- `backend/specparse/` — the PDF → regions → OCR → structured-spec pipeline
- `models/spec-region-run/` — the YOLOv8 training kit (this repo's pattern)
- `docs/spec-pdf-pipeline.md` — full pipeline write-up and the 699 mapping
