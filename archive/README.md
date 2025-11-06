# Archive

This directory contains old, deprecated, and duplicate files that are no longer actively used in the project.

## Structure

### deprecated_sql/
Contains SQL files that have been deprecated and replaced by newer versions:
- `create_metadata_table.sql.deprecated` - Old metadata table schema
- `create_parquet_ocr_tables.sql.deprecated` - Old OCR table schema

These files are kept for reference but should not be used in the current project.

### duplicate_scripts/
Contains duplicate or platform-specific versions of files that have been consolidated:
- `visualize_ocr_boxes_windows.py` - Windows-specific version (replaced by cross-platform version in `scripts/ocr/`)
- `yolov8n.pt` - Duplicate model file (primary copy is in `models/`)

## Archive Date
Files archived on: 2025-11-04

## Restoration
If you need to restore any files from this archive, you can copy them back to their original locations. However, be aware that they may be outdated or incompatible with the current codebase.
