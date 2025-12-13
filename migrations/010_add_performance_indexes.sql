-- Performance optimization indexes
-- Add composite indexes for common query patterns

-- Activity logs: user_id + created_at (for paginated queries)
CREATE INDEX IF NOT EXISTS idx_user_activity_user_created 
ON user_activity_logs (user_id, created_at DESC);

-- Activity logs: user_id + action + created_at (for filtered queries)
CREATE INDEX IF NOT EXISTS idx_user_activity_user_action_created 
ON user_activity_logs (user_id, action, created_at DESC);

-- Video uploads: user_id + status + created_at (for filtered pagination)
CREATE INDEX IF NOT EXISTS idx_video_uploads_user_status_created 
ON video_uploads (user_id, status, created_at DESC);

-- Video uploads: user_id + is_deleted + updated_at (for active videos)
CREATE INDEX IF NOT EXISTS idx_video_uploads_user_deleted_updated 
ON video_uploads (user_id, is_deleted, updated_at DESC);

-- Frame analyses: video_id + timestamp (for ordered frame queries)
CREATE INDEX IF NOT EXISTS idx_frame_analyses_video_timestamp 
ON frame_analyses (video_id, timestamp);

-- Activity logs: created_at (for date range queries)
CREATE INDEX IF NOT EXISTS idx_user_activity_created_at 
ON user_activity_logs (created_at DESC);

