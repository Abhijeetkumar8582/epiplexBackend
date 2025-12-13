-- Create video_summaries table for storing batch summaries of frame analyses
-- SQL Server compatible syntax

-- Drop table if exists (for re-running migration)
IF OBJECT_ID('video_summaries', 'U') IS NOT NULL
    DROP TABLE video_summaries;
GO

-- Create the table
CREATE TABLE video_summaries (
    id                  UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
    video_id            UNIQUEIDENTIFIER NOT NULL,
    batch_number        INT NOT NULL,
    batch_start_frame   INT NOT NULL,
    batch_end_frame     INT NOT NULL,
    total_frames_in_batch INT NOT NULL,
    summary_text        NVARCHAR(MAX) NOT NULL,
    summary_metadata    NVARCHAR(MAX) NULL,
    processing_time_ms  INT NULL,
    model_used          NVARCHAR(50) NULL DEFAULT 'gpt-4o-mini',
    created_at          DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    -- Foreign key constraint
    CONSTRAINT FK_video_summaries_video_uploads 
        FOREIGN KEY (video_id) 
        REFERENCES video_uploads(id) 
        ON DELETE CASCADE
);
GO

-- Create indexes
CREATE INDEX idx_video_summaries_video_id 
    ON video_summaries (video_id);
GO

CREATE INDEX idx_video_summaries_batch_number 
    ON video_summaries (video_id, batch_number);
GO

-- Add table description
EXEC sp_addextendedproperty 
    @name = N'MS_Description', 
    @value = N'Stores batch summaries of frame analyses for videos', 
    @level0type = N'SCHEMA', @level0name = N'dbo', 
    @level1type = N'TABLE', @level1name = N'video_summaries';
GO
