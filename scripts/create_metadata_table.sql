-- Create document_metadata table for storing parquet file metadata
-- This allows efficient querying and filtering without loading all images

CREATE TABLE IF NOT EXISTS document_metadata (
    id SERIAL PRIMARY KEY,
    parquet_file VARCHAR(255) NOT NULL,  -- Name of the parquet file
    row_index INTEGER NOT NULL,          -- Row index within the parquet file
    label INTEGER NOT NULL,              -- Image classification label
    image_size_bytes INTEGER,            -- Size of image in bytes
    image_path TEXT,                     -- Original path if available
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(parquet_file, row_index)      -- Ensure no duplicates
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_metadata_label ON document_metadata(label);
CREATE INDEX IF NOT EXISTS idx_metadata_parquet_file ON document_metadata(parquet_file);
CREATE INDEX IF NOT EXISTS idx_metadata_size ON document_metadata(image_size_bytes);

COMMENT ON TABLE document_metadata IS 'Metadata for images stored in parquet files - enables efficient querying without loading all images';
COMMENT ON COLUMN document_metadata.parquet_file IS 'Name of parquet file containing the image';
COMMENT ON COLUMN document_metadata.row_index IS 'Row index within the parquet file (0-based)';
COMMENT ON COLUMN document_metadata.label IS 'Image classification label';
COMMENT ON COLUMN document_metadata.image_size_bytes IS 'Size of image data in bytes';
