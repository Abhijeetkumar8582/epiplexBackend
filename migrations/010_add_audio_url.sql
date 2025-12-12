-- Migration: Add audio_url column to video_uploads table
-- Date: 2024
-- Description: Adds audio_url column to store path to extracted audio files

-- Add audio_url column to video_uploads table
IF NOT EXISTS (
    SELECT 1 
    FROM sys.columns 
    WHERE object_id = OBJECT_ID('video_uploads') 
    AND name = 'audio_url'
)
BEGIN
    ALTER TABLE video_uploads 
    ADD audio_url NVARCHAR(MAX) NULL;
    
    PRINT 'Column audio_url added to video_uploads table';
END
ELSE
BEGIN
    PRINT 'Column audio_url already exists in video_uploads table';
END
GO
