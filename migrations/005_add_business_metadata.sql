-- Add business/functional metadata fields to video_uploads table

-- Add application_name column
ALTER TABLE video_uploads 
ADD COLUMN IF NOT EXISTS application_name VARCHAR(100);

-- Add tags column (JSONB for array storage)
ALTER TABLE video_uploads 
ADD COLUMN IF NOT EXISTS tags JSONB;

-- Add language_code column
ALTER TABLE video_uploads 
ADD COLUMN IF NOT EXISTS language_code VARCHAR(10);

-- Add priority column
ALTER TABLE video_uploads 
ADD COLUMN IF NOT EXISTS priority VARCHAR(20) DEFAULT 'normal';

-- Add soft delete columns
ALTER TABLE video_uploads 
ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE video_uploads 
ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP WITH TIME ZONE;

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_video_uploads_application_name ON video_uploads (application_name);
CREATE INDEX IF NOT EXISTS idx_video_uploads_language_code ON video_uploads (language_code);
CREATE INDEX IF NOT EXISTS idx_video_uploads_priority ON video_uploads (priority);
CREATE INDEX IF NOT EXISTS idx_video_uploads_is_deleted ON video_uploads (is_deleted);
CREATE INDEX IF NOT EXISTS idx_video_uploads_tags ON video_uploads USING GIN (tags);

-- Add comments
COMMENT ON COLUMN video_uploads.application_name IS 'Application name (e.g., SAP, Salesforce)';
COMMENT ON COLUMN video_uploads.tags IS 'Tags as JSON array (e.g., ["HR", "Payroll", "Onboarding"])';
COMMENT ON COLUMN video_uploads.language_code IS 'Language code (e.g., en, hi)';
COMMENT ON COLUMN video_uploads.priority IS 'Priority level: normal, high';
COMMENT ON COLUMN video_uploads.is_deleted IS 'Soft delete flag';
COMMENT ON COLUMN video_uploads.deleted_at IS 'Timestamp when video was soft deleted';

