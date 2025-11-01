"""
Load parquet file metadata into PostgreSQL database

This script extracts metadata (without loading actual images) from parquet files
and stores it in the database for efficient querying and dataset management.
"""

import pandas as pd
import os
from pathlib import Path
from sqlalchemy import create_engine, text
from tqdm import tqdm
import time

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:123@db:5432/postgres')

def get_image_size(image_data):
    """Extract image size from the image data structure"""
    if isinstance(image_data, dict):
        if 'bytes' in image_data:
            return len(image_data['bytes'])
    return len(image_data) if image_data else 0

def load_metadata_from_parquet(parquet_dir='/workspace/data/raw'):
    """Load metadata from all parquet files into database"""

    print("="*70)
    print("Loading Parquet Metadata to PostgreSQL")
    print("="*70)

    # Connect to database
    print(f"\nConnecting to database: {DATABASE_URL.split('@')[1]}")
    engine = create_engine(DATABASE_URL)

    # Create table if it doesn't exist
    print("Creating document_metadata table...")
    with open('/workspace/scripts/create_metadata_table.sql', 'r') as f:
        sql = f.read()

    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()
    print("✓ Table created/verified")

    # Find all parquet files
    parquet_path = Path(parquet_dir)
    parquet_files = sorted(parquet_path.glob('*.parquet'))

    if not parquet_files:
        print(f"\n❌ No parquet files found in {parquet_dir}")
        return

    print(f"\nFound {len(parquet_files)} parquet files")
    print()

    total_rows_loaded = 0
    start_time = time.time()

    for pf in parquet_files:
        print(f"Processing: {pf.name}")

        # Read parquet file
        df = pd.read_parquet(pf)

        # Extract metadata (without loading full images)
        metadata_records = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Extracting metadata"):
            metadata_records.append({
                'parquet_file': pf.name,
                'row_index': idx,
                'label': row['label'],
                'image_size_bytes': get_image_size(row['image']),
            })

        # Create DataFrame from metadata
        metadata_df = pd.DataFrame(metadata_records)

        # Load to database (replace duplicates)
        try:
            # Delete existing records for this parquet file first
            with engine.connect() as conn:
                conn.execute(
                    text("DELETE FROM document_metadata WHERE parquet_file = :pf"),
                    {"pf": pf.name}
                )
                conn.commit()

            # Insert new records
            metadata_df.to_sql(
                'document_metadata',
                engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=1000
            )

            total_rows_loaded += len(metadata_df)
            print(f"  ✓ Loaded {len(metadata_df):,} records")
            print()

        except Exception as e:
            print(f"  ❌ Error loading {pf.name}: {e}")
            continue

    elapsed = time.time() - start_time

    # Print summary
    print("="*70)
    print("Summary")
    print("="*70)
    print(f"Total records loaded: {total_rows_loaded:,}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print(f"Records per second: {total_rows_loaded/elapsed:.0f}")

    # Query statistics
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) as total FROM document_metadata"))
        total = result.scalar()

        result = conn.execute(text("""
            SELECT label, COUNT(*) as count
            FROM document_metadata
            GROUP BY label
            ORDER BY label
        """))
        label_dist = result.fetchall()

        result = conn.execute(text("""
            SELECT
                AVG(image_size_bytes) as avg_size,
                MIN(image_size_bytes) as min_size,
                MAX(image_size_bytes) as max_size
            FROM document_metadata
        """))
        size_stats = result.fetchone()

    print(f"\nDatabase Statistics:")
    print(f"  Total records: {total:,}")
    print(f"\n  Label distribution:")
    for label, count in label_dist:
        print(f"    Label {label}: {count:,} images ({count/total*100:.1f}%)")

    print(f"\n  Image size statistics:")
    print(f"    Average: {size_stats[0]:,.0f} bytes ({size_stats[0]/1024:.1f} KB)")
    print(f"    Min: {size_stats[1]:,} bytes")
    print(f"    Max: {size_stats[2]:,} bytes")

    print("\n✓ Metadata loading complete!")
    print("\nYou can now query the data efficiently using SQL:")
    print("  SELECT * FROM document_metadata WHERE label = 5 LIMIT 10;")
    print("  SELECT label, COUNT(*) FROM document_metadata GROUP BY label;")


if __name__ == "__main__":
    load_metadata_from_parquet()
