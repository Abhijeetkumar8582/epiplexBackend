-- Create frame_analyses table for storing frame-by-frame analysis results
CREATE TABLE frame_analyses (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id            UUID NOT NULL REFERENCES video_uploads(id) ON DELETE CASCADE,
    
    -- Frame metadata
    timestamp           DOUBLE PRECISION NOT NULL,
    frame_number        INTEGER,
    image_path          TEXT NOT NULL,
    
    -- Analysis results
    description         TEXT,
    ocr_text            TEXT,
    gpt_response        JSONB,
    
    -- Processing metadata
    processing_time_ms  INTEGER,
    
    -- Timestamps
    created_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for faster queries
CREATE INDEX idx_frame_analyses_video_id ON frame_analyses (video_id);
CREATE INDEX idx_frame_analyses_timestamp ON frame_analyses (video_id, timestamp);
CREATE INDEX idx_frame_analyses_created_at ON frame_analyses (created_at DESC);
CREATE INDEX idx_frame_analyses_gpt_response ON frame_analyses USING GIN (gpt_response);

-- Foreign key constraint ensures referential integrity
ALTER TABLE frame_analyses 
ADD CONSTRAINT fk_frame_analyses_video_upload 
FOREIGN KEY (video_id) REFERENCES video_uploads(id) ON DELETE CASCADE;

-- Add comments
COMMENT ON TABLE frame_analyses IS 'Stores frame-by-frame analysis results from video processing';
COMMENT ON COLUMN frame_analyses.timestamp IS 'Timestamp in seconds from video start';
COMMENT ON COLUMN frame_analyses.description IS 'GPT-generated description/caption of the frame';
COMMENT ON COLUMN frame_analyses.ocr_text IS 'Extracted OCR text from the frame';
COMMENT ON COLUMN frame_analyses.gpt_response IS 'Full GPT API response stored as JSONB';
COMMENT ON COLUMN frame_analyses.video_id IS 'Foreign key to video_uploads table - ensures data integrity';

