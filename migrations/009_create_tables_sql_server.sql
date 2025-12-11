-- ============================================================================
-- SQL Server Migration Script
-- Creates all tables for Epiplex application in SQL Server
-- Database: Druid_AbhijeetKumar
-- ============================================================================

USE Druid_AbhijeetKumar;
GO

-- ============================================================================
-- 1. USERS TABLE
-- ============================================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'users' AND schema_id = SCHEMA_ID('dbo'))
BEGIN
    CREATE TABLE dbo.users (
        id              UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
        full_name       NVARCHAR(150) NOT NULL,
        email           NVARCHAR(255) NOT NULL,
        password_hash   NVARCHAR(255) NULL,  -- Nullable for OAuth users
        google_id       NVARCHAR(255) NULL,  -- For Google OAuth
        provider        NVARCHAR(50) NOT NULL DEFAULT 'email',  -- 'email' or 'google'
        role            NVARCHAR(50) NOT NULL DEFAULT 'user',
        is_active       BIT NOT NULL DEFAULT 1,
        last_login_at   DATETIME2 NULL,
        created_at      DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        updated_at      DATETIME2 NOT NULL DEFAULT GETUTCDATE()
    );

    -- Create unique constraint on email
    CREATE UNIQUE INDEX idx_users_email ON dbo.users (email);

    -- Create unique index on google_id (if using Google OAuth)
    CREATE UNIQUE INDEX idx_users_google_id ON dbo.users (google_id) WHERE google_id IS NOT NULL;

    PRINT 'Table [users] created successfully.';
END
ELSE
BEGIN
    PRINT 'Table [users] already exists.';
END
GO

-- ============================================================================
-- 2. USER SESSIONS TABLE
-- ============================================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'user_sessions' AND schema_id = SCHEMA_ID('dbo'))
BEGIN
    CREATE TABLE dbo.user_sessions (
        id              UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
        user_id         UNIQUEIDENTIFIER NOT NULL,
        session_token   NVARCHAR(255) NOT NULL,
        ip_address      NVARCHAR(45) NULL,  -- IPv6 max length
        user_agent      NVARCHAR(MAX) NULL,
        created_at      DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        expires_at      DATETIME2 NOT NULL,
        
        -- Foreign key constraint
        CONSTRAINT FK_user_sessions_user_id 
            FOREIGN KEY (user_id) 
            REFERENCES dbo.users(id) 
            ON DELETE CASCADE
    );

    -- Indexes for user_sessions
    CREATE INDEX idx_user_sessions_user_id ON dbo.user_sessions (user_id);
    CREATE UNIQUE INDEX idx_user_sessions_token ON dbo.user_sessions (session_token);

    PRINT 'Table [user_sessions] created successfully.';
END
ELSE
BEGIN
    PRINT 'Table [user_sessions] already exists.';
END
GO

-- ============================================================================
-- 3. USER ACTIVITY LOGS TABLE
-- ============================================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'user_activity_logs' AND schema_id = SCHEMA_ID('dbo'))
BEGIN
    CREATE TABLE dbo.user_activity_logs (
        id              BIGINT IDENTITY(1,1) PRIMARY KEY,
        user_id         UNIQUEIDENTIFIER NOT NULL,
        action          NVARCHAR(100) NOT NULL,
        description     NVARCHAR(MAX) NULL,
        metadata        NVARCHAR(MAX) NULL,  -- JSON stored as NVARCHAR(MAX)
        ip_address      NVARCHAR(45) NULL,  -- IPv6 max length
        created_at      DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        
        -- Foreign key constraint
        CONSTRAINT FK_user_activity_logs_user_id 
            FOREIGN KEY (user_id) 
            REFERENCES dbo.users(id) 
            ON DELETE CASCADE
    );

    -- Indexes for user_activity_logs
    CREATE INDEX idx_user_activity_user_id ON dbo.user_activity_logs (user_id);
    CREATE INDEX idx_user_activity_action ON dbo.user_activity_logs (action);

    PRINT 'Table [user_activity_logs] created successfully.';
END
ELSE
BEGIN
    PRINT 'Table [user_activity_logs] already exists.';
END
GO

-- ============================================================================
-- 4. VIDEO UPLOADS TABLE
-- ============================================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'video_uploads' AND schema_id = SCHEMA_ID('dbo'))
BEGIN
    CREATE TABLE dbo.video_uploads (
        id                      UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
        user_id                 UNIQUEIDENTIFIER NOT NULL,
        
        -- Essential fields
        name                    NVARCHAR(255) NOT NULL,
        source_type             NVARCHAR(50) NOT NULL DEFAULT 'upload',
        video_url               NVARCHAR(MAX) NOT NULL,
        original_input          NVARCHAR(MAX) NOT NULL,
        status                  NVARCHAR(50) NOT NULL DEFAULT 'uploaded',
        
        -- Video tech metadata
        video_length_seconds    FLOAT NULL,
        video_size_bytes        BIGINT NULL,
        mime_type               NVARCHAR(100) NULL,
        resolution_width        INT NULL,
        resolution_height       INT NULL,
        fps                     FLOAT NULL,
        
        -- Business/Functional metadata
        application_name        NVARCHAR(100) NULL,
        tags                    NVARCHAR(MAX) NULL,  -- JSON array stored as string
        language_code           NVARCHAR(10) NULL,
        priority                NVARCHAR(20) NULL DEFAULT 'normal',
        
        -- Soft delete support
        is_deleted              BIT NOT NULL DEFAULT 0,
        deleted_at              DATETIME2 NULL,
        
        -- Timestamps
        created_at              DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        updated_at              DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        
        -- Link to job_status
        job_id                  NVARCHAR(255) NULL,
        
        -- Video file number
        video_file_number       NVARCHAR(50) NULL,
        
        -- Foreign key constraint
        CONSTRAINT FK_video_uploads_user_id 
            FOREIGN KEY (user_id) 
            REFERENCES dbo.users(id) 
            ON DELETE CASCADE
    );

    -- Indexes
    CREATE INDEX idx_video_uploads_user_id ON dbo.video_uploads (user_id);
    CREATE INDEX idx_video_uploads_status ON dbo.video_uploads (status);
    CREATE INDEX idx_video_uploads_is_deleted ON dbo.video_uploads (is_deleted);
    CREATE INDEX idx_video_uploads_job_id ON dbo.video_uploads (job_id);
    CREATE UNIQUE INDEX idx_video_uploads_video_file_number ON dbo.video_uploads (video_file_number) WHERE video_file_number IS NOT NULL;

    PRINT 'Table [video_uploads] created successfully.';
END
ELSE
BEGIN
    PRINT 'Table [video_uploads] already exists.';
END
GO

-- ============================================================================
-- 5. FRAME ANALYSES TABLE
-- ============================================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'frame_analyses' AND schema_id = SCHEMA_ID('dbo'))
BEGIN
    CREATE TABLE dbo.frame_analyses (
        id                  UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
        video_id            UNIQUEIDENTIFIER NOT NULL,
        
        -- Frame metadata
        timestamp           FLOAT NOT NULL,
        frame_number        INT NULL,
        image_path          NVARCHAR(MAX) NOT NULL,
        
        -- Analysis results
        description         NVARCHAR(MAX) NULL,
        ocr_text            NVARCHAR(MAX) NULL,
        gpt_response        NVARCHAR(MAX) NULL,  -- JSON stored as string
        
        -- Processing metadata
        processing_time_ms  INT NULL,
        
        -- Timestamps
        created_at          DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
        
        -- Foreign key constraint
        CONSTRAINT FK_frame_analyses_video_id 
            FOREIGN KEY (video_id) 
            REFERENCES dbo.video_uploads(id) 
            ON DELETE CASCADE
    );

    -- Indexes
    CREATE INDEX idx_frame_analyses_video_id ON dbo.frame_analyses (video_id);
    CREATE INDEX idx_frame_analyses_timestamp ON dbo.frame_analyses (timestamp);

    PRINT 'Table [frame_analyses] created successfully.';
END
ELSE
BEGIN
    PRINT 'Table [frame_analyses] already exists.';
END
GO

-- ============================================================================
-- 6. JOB STATUS TABLE
-- ============================================================================
IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'job_status' AND schema_id = SCHEMA_ID('dbo'))
BEGIN
    CREATE TABLE dbo.job_status (
        job_id          NVARCHAR(255) PRIMARY KEY,
        status          NVARCHAR(50) NOT NULL,
        progress        INT DEFAULT 0,
        message         NVARCHAR(MAX) NULL,
        current_step    NVARCHAR(255) NULL,
        step_progress   NVARCHAR(MAX) NULL,  -- JSON stored as string
        output_files    NVARCHAR(MAX) NULL,  -- JSON stored as string
        transcript      NVARCHAR(MAX) NULL,
        frame_analyses  NVARCHAR(MAX) NULL,  -- JSON stored as string
        error           NVARCHAR(MAX) NULL,
        created_at      DATETIME2 DEFAULT GETUTCDATE(),
        updated_at      DATETIME2 DEFAULT GETUTCDATE()
    );

    CREATE INDEX idx_job_status_status ON dbo.job_status (status);

    PRINT 'Table [job_status] created successfully.';
END
ELSE
BEGIN
    PRINT 'Table [job_status] already exists.';
END
GO

-- ============================================================================
-- VERIFICATION
-- ============================================================================
PRINT '';
PRINT '=== TABLE CREATION SUMMARY ===';
SELECT 
    t.name AS TableName,
    CASE WHEN t.name IS NOT NULL THEN 'EXISTS' ELSE 'MISSING' END AS Status
FROM sys.tables t
WHERE t.name IN ('users', 'user_sessions', 'user_activity_logs', 'video_uploads', 'frame_analyses', 'job_status')
    AND t.schema_id = SCHEMA_ID('dbo')
ORDER BY t.name;

PRINT '';
PRINT 'All tables created successfully!';
GO

