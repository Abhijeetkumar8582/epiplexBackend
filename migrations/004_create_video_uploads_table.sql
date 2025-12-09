-- Create video_uploads table
CREATE TABLE video_uploads (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Essential fields
    name            VARCHAR(255) NOT NULL,
    source_type     VARCHAR(50) NOT NULL DEFAULT 'upload',
    video_url       TEXT NOT NULL,
    original_input  TEXT NOT NULL,
    status          VARCHAR(50) NOT NULL DEFAULT 'uploaded',
    
    -- Video tech metadata
    video_length_seconds  DOUBLE PRECISION,
    video_size_bytes      BIGINT,
    mime_type            VARCHAR(100),
    resolution_width      INTEGER,
    resolution_height     INTEGER,
    fps                   DOUBLE PRECISION,
    
    -- Timestamps
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Link to job_status
    job_id          VARCHAR(255),
    
    -- Video file number
    video_file_number VARCHAR(50) UNIQUE
);

-- Indexes for faster queries
CREATE INDEX idx_video_uploads_user_id ON video_uploads (user_id);
CREATE INDEX idx_video_uploads_status ON video_uploads (status);
CREATE INDEX idx_video_uploads_created_at ON video_uploads (created_at DESC);
CREATE INDEX idx_video_uploads_job_id ON video_uploads (job_id);
CREATE INDEX idx_video_uploads_video_file_number ON video_uploads (video_file_number);

-- Foreign key constraint ensures referential integrity
ALTER TABLE video_uploads 
ADD CONSTRAINT fk_video_uploads_user 
FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

-- Add comment to table
COMMENT ON TABLE video_uploads IS 'Stores video upload metadata and processing information';
COMMENT ON COLUMN video_uploads.source_type IS 'upload or url';
COMMENT ON COLUMN video_uploads.status IS 'uploaded, processing, completed, failed, cancelled';
COMMENT ON COLUMN video_uploads.user_id IS 'Foreign key to users table - ensures data integrity';
COMMENT ON COLUMN video_uploads.video_file_number IS 'Unique video file number (e.g., VF-2024-0001)';

