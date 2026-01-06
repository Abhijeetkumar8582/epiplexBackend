-- Add openai_api_key column to users table
-- SQL Server compatible syntax

-- Check if column already exists
IF NOT EXISTS (
    SELECT 1 
    FROM sys.columns 
    WHERE object_id = OBJECT_ID('users') 
    AND name = 'openai_api_key'
)
BEGIN
    ALTER TABLE users
    ADD openai_api_key NVARCHAR(500) NULL;
    
    -- Add comment
    EXEC sp_addextendedproperty 
        @name = N'MS_Description', 
        @value = N'User''s personal OpenAI API key. If set, this key will be used instead of the system default for GPT API calls.', 
        @level0type = N'SCHEMA', @level0name = N'dbo', 
        @level1type = N'TABLE', @level1name = N'users',
        @level2type = N'COLUMN', @level2name = N'openai_api_key';
END
GO

