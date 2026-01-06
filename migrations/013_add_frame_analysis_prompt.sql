-- Add frame_analysis_prompt column to users table
-- SQL Server compatible syntax

-- Check if column already exists
IF NOT EXISTS (
    SELECT 1 
    FROM sys.columns 
    WHERE object_id = OBJECT_ID('users') 
    AND name = 'frame_analysis_prompt'
)
BEGIN
    ALTER TABLE users
    ADD frame_analysis_prompt NVARCHAR(MAX) NULL;
    
    -- Add comment
    EXEC sp_addextendedproperty 
        @name = N'MS_Description', 
        @value = N'Custom GPT prompt for frame analysis. If set, this prompt will be used instead of the default prompt.txt file for analyzing video frames.', 
        @level0type = N'SCHEMA', @level0name = N'dbo', 
        @level1type = N'TABLE', @level1name = N'users',
        @level2type = N'COLUMN', @level2name = N'frame_analysis_prompt';
END
GO

