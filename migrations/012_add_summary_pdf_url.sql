-- Add summary_pdf_url column to video_uploads table
-- SQL Server compatible syntax

-- Check if column already exists
IF NOT EXISTS (
    SELECT 1 
    FROM sys.columns 
    WHERE object_id = OBJECT_ID('video_uploads') 
    AND name = 'summary_pdf_url'
)
BEGIN
    ALTER TABLE video_uploads
    ADD summary_pdf_url NVARCHAR(MAX) NULL;
    
    -- Add comment
    EXEC sp_addextendedproperty 
        @name = N'MS_Description', 
        @value = N'Path to generated summary PDF file containing frame images and summaries', 
        @level0type = N'SCHEMA', @level0name = N'dbo', 
        @level1type = N'TABLE', @level1name = N'video_uploads',
        @level2type = N'COLUMN', @level2name = N'summary_pdf_url';
END
GO

