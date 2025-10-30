# This file loads all raw parquet data files into the raw_documents table
import pandas as pd
from sqlalchemy import create_engine
import os
import glob
from pathlib import Path

print("Current directory:", os.getcwd())

# Define the data directory path
data_dir = '/workspace/data/raw'

# Check if directory exists
if not os.path.exists(data_dir):
    print(f"Data directory not found: {data_dir}")
    print("Available directories:")
    for root, dirs, files in os.walk('/workspace'):
        if 'data' in dirs or any('parquet' in f for f in files):
            print(f"  {root}")
    exit(1)

# Find all parquet files
parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
print(f"Found {len(parquet_files)} parquet files:")
for file in parquet_files:
    print(f"  {file}")

if not parquet_files:
    print("No parquet files found!")
    exit(1)

# Create database connection
engine = create_engine('postgresql://postgres:123@localhost:5432/postgres')

# Load each parquet file and append to database
total_rows = 0
for i, file_path in enumerate(parquet_files):
    try:
        print(f"\nProcessing file {i+1}/{len(parquet_files)}: {os.path.basename(file_path)}")
        
        # Read parquet file
        df = pd.read_parquet(file_path)
        print(f"  Loaded {len(df)} rows")

        # Show columns for first file
        if i == 0:
            print(f"  Columns: {df.columns.tolist()}")
            print(f"  Image column type: {type(df['image'].iloc[0])}")

        # Extract bytes from the image dictionary
        print(f"  Extracting image bytes from dictionary...")
        df['image'] = df['image'].apply(lambda x: x['bytes'] if isinstance(x, dict) else x)

        # Load to documents table (not raw_documents, we need the table with BYTEA column)
        df.to_sql('documents', engine, if_exists='append', index=False, method='multi')
        
        total_rows += len(df)
        print(f"  Successfully loaded {len(df)} rows to database")
        
    except Exception as e:
        print(f"  Error processing {file_path}: {str(e)}")
        continue

print(f"\n=== SUMMARY ===")
print(f"Total files processed: {len(parquet_files)}")
print(f"Total rows loaded: {total_rows}")
print("Data loading completed!")

# Verify the data in database
try:
    result = pd.read_sql("SELECT COUNT(*) as total_rows FROM documents", engine)
    print(f"Total rows in database: {result['total_rows'].iloc[0]}")
except Exception as e:
    print(f"Error verifying data: {e}")