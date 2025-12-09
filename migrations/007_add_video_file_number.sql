-- Migration: Add video_file_number to video_uploads and gpt_response to frame_analyses
-- Date: 2024-01-15

-- Add video_file_number column to video_uploads table
ALTER TABLE video_uploads
ADD COLUMN IF NOT EXISTS video_file_number VARCHAR(50) UNIQUE;

-- Create index on video_file_number for fast lookups
CREATE INDEX IF NOT EXISTS idx_video_uploads_video_file_number ON video_uploads (video_file_number);

-- Add gpt_response column to frame_analyses table (JSONB for storing full GPT response)
ALTER TABLE frame_analyses
ADD COLUMN IF NOT EXISTS gpt_response JSONB;

-- Create GIN index on gpt_response for efficient JSON queries
CREATE INDEX IF NOT EXISTS idx_frame_analyses_gpt_response ON frame_analyses USING GIN (gpt_response);

-- Add comment for documentation
COMMENT ON COLUMN video_uploads.video_file_number IS 'Unique video file number (e.g., VF-2024-001)';
COMMENT ON COLUMN frame_analyses.gpt_response IS 'Full GPT response JSON stored for each frame analysis';

