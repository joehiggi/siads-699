"""Test database connection and query OCR results"""

import os
from sqlalchemy import create_engine, text
import pandas as pd


def test_connection():
    """Test database connection and show OCR results"""

    # Connect to database
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:123@db:5432/postgres")

    print("=" * 70)
    print("Testing Database Connection\n" + "=" * 70)
    print(f"\nConnecting to: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'database'}")

    try:
        engine = create_engine(DATABASE_URL)

        with engine.connect() as conn:
            # Test connection
            conn.execute(text("SELECT 1"))
            print("✓ Database connection successful\n")

            # Check if OCR tables exist
            result = conn.execute(
                text(
                    """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name LIKE 'parquet%'
                ORDER BY table_name
            """
                )
            )
            tables = result.fetchall()

            if tables:
                print("OCR Tables found:")
                for table in tables:
                    print(f"  - {table[0]}")
                print()
            else:
                print("⚠ No OCR tables found. Run ocr_processor.py to create them.\n")
                return

            # Count OCR results
            result = conn.execute(text("SELECT COUNT(*) FROM parquet_ocr_results"))
            count = result.scalar()
            print(f"Total OCR results in database: {count:,}\n")

            if count > 0:
                # Show recent results
                print("Recent OCR Results:")
                print("-" * 70)

                df = pd.read_sql(
                    text(
                        """
                    SELECT
                        parquet_file,
                        row_index,
                        label,
                        ocr_engine,
                        tesseract_confidence,
                        tesseract_word_count,
                        yolo_region_count,
                        processing_status,
                        processed_at
                    FROM parquet_ocr_results
                    ORDER BY processed_at DESC
                    LIMIT 10
                """
                    ),
                    conn,
                )

                print(df.to_string(index=False))
                print()

                # Show sample text
                result = conn.execute(
                    text(
                        """
                    SELECT
                        parquet_file,
                        row_index,
                        LEFT(tesseract_full_text, 100) as text_sample
                    FROM parquet_ocr_results
                    WHERE tesseract_full_text IS NOT NULL
                    AND LENGTH(tesseract_full_text) > 0
                    LIMIT 3
                """
                    )
                )
                samples = result.fetchall()

                if samples:
                    print("\nSample Extracted Text:")
                    print("-" * 70)
                    for file, idx, text in samples:
                        print(f"\n{file}[{idx}]:\n  {text}...")
                print()

                # Statistics by label
                result = conn.execute(
                    text(
                        """
                    SELECT
                        label,
                        COUNT(*) as count,
                        AVG(tesseract_confidence) as avg_confidence
                    FROM parquet_ocr_results
                    WHERE ocr_engine = 'tesseract'
                    GROUP BY label
                    ORDER BY label
                """
                    )
                )
                stats = result.fetchall()

                if stats:
                    print("Statistics by Label:")
                    print("-" * 70)
                    print(f"{'Label':<10} {'Count':<10} {'Avg Confidence':<15}")
                    print("-" * 70)
                    for label, count, conf in stats:
                        conf_str = f"{conf:.1f}%" if conf else "N/A"
                        print(f"{label:<10} {count:<10} {conf_str:<15}")
                    print()

            print("=" * 70)
            print("✓ Database test complete\n")
            print("Useful queries:")
            print("  SELECT * FROM parquet_ocr_summary;")
            print("  SELECT * FROM parquet_ocr_words WHERE confidence > 80;")
            print("  SELECT * FROM parquet_yolo_regions;")
            print("=" * 70)

    except Exception as e:
        print(f"❌ Database error: {e}\n")
        print("Make sure:")
        print("  1. Docker containers are running: docker-compose up -d")
        print("  2. Database is initialized")
        print("  3. DATABASE_URL environment variable is set correctly")


if __name__ == "__main__":
    test_connection()
