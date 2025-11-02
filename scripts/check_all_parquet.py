"""Check all parquet files for metadata"""
import pandas as pd
from pathlib import Path

parquet_dir = Path("/workspace/data/raw")
parquet_files = sorted(parquet_dir.glob("*.parquet"))

print(f"Found {len(parquet_files)} parquet files\n")
print("=" * 70)

total_rows = 0
label_counts = {}

for pf in parquet_files:
    df = pd.read_parquet(pf)
    rows = len(df)
    total_rows += rows

    # Count labels
    for label, count in df["label"].value_counts().items():
        label_counts[label] = label_counts.get(label, 0) + count

    # Get average image size
    first_image = df["image"].iloc[0]
    if isinstance(first_image, dict) and "bytes" in first_image:
        avg_size = (
            df["image"].apply(lambda x: len(x["bytes"]) if isinstance(x, dict) else len(x)).mean()
        )
    else:
        avg_size = df["image"].apply(len).mean()

    print(f"File: {pf.name}")
    print(f"  Rows: {rows:,}")
    print(f"  Labels: {sorted(df['label'].unique())}")
    print(f"  Avg image size: {avg_size:,.0f} bytes")
    print()

print("=" * 70)
print(f"Total rows across all files: {total_rows:,}")
print(f"\nOverall label distribution:")
for label in sorted(label_counts.keys()):
    print(f"  Label {label}: {label_counts[label]:,} images")
