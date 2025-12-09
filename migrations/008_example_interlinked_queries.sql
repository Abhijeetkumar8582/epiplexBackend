-- Example SQL queries showing how to fetch interlinked data
-- These queries demonstrate the relationships between tables

-- 1. Get all video uploads with their user information
SELECT 
    vu.id,
    vu.video_file_number,
    vu.name,
    vu.status,
    vu.created_at,
    u.full_name as user_name,
    u.email as user_email
FROM video_uploads vu
INNER JOIN users u ON vu.user_id = u.id
WHERE vu.is_deleted = false
ORDER BY vu.created_at DESC;

-- 2. Get all frame analyses for a specific video with video details
SELECT 
    fa.id as frame_id,
    fa.timestamp,
    fa.frame_number,
    fa.description,
    fa.ocr_text,
    fa.gpt_response,
    fa.image_path,
    fa.processing_time_ms,
    fa.created_at,
    vu.video_file_number,
    vu.name as video_name,
    vu.status as video_status
FROM frame_analyses fa
INNER JOIN video_uploads vu ON fa.video_id = vu.id
WHERE vu.video_file_number = 'VF-2024-0001'
ORDER BY fa.timestamp ASC;

-- 3. Get all GPT responses for a video file number
SELECT 
    fa.gpt_response,
    fa.timestamp,
    fa.frame_number,
    fa.description,
    fa.ocr_text
FROM frame_analyses fa
INNER JOIN video_uploads vu ON fa.video_id = vu.id
WHERE vu.video_file_number = 'VF-2024-0001'
  AND fa.gpt_response IS NOT NULL
ORDER BY fa.timestamp ASC;

-- 4. Get user's video uploads with frame analysis count
SELECT 
    vu.id,
    vu.video_file_number,
    vu.name,
    vu.status,
    vu.created_at,
    COUNT(fa.id) as total_frames_analyzed
FROM video_uploads vu
LEFT JOIN frame_analyses fa ON vu.id = fa.video_id
WHERE vu.user_id = 'USER_UUID_HERE'
  AND vu.is_deleted = false
GROUP BY vu.id, vu.video_file_number, vu.name, vu.status, vu.created_at
ORDER BY vu.created_at DESC;

-- 5. Get all activity logs for a user with video upload details
SELECT 
    ual.id,
    ual.action,
    ual.description,
    ual.metadata,
    ual.created_at,
    vu.video_file_number,
    vu.name as video_name
FROM user_activity_logs ual
LEFT JOIN video_uploads vu ON (ual.metadata->>'upload_id')::uuid = vu.id
WHERE ual.user_id = 'USER_UUID_HERE'
ORDER BY ual.created_at DESC
LIMIT 50;

-- 6. Get video upload with all related data (user, frames, activities)
SELECT 
    vu.id as video_id,
    vu.video_file_number,
    vu.name,
    vu.status,
    u.full_name as uploaded_by,
    u.email as user_email,
    COUNT(DISTINCT fa.id) as total_frames,
    COUNT(DISTINCT ual.id) as total_activities
FROM video_uploads vu
INNER JOIN users u ON vu.user_id = u.id
LEFT JOIN frame_analyses fa ON vu.id = fa.video_id
LEFT JOIN user_activity_logs ual ON ual.user_id = u.id 
    AND (ual.metadata->>'upload_id')::uuid = vu.id
WHERE vu.video_file_number = 'VF-2024-0001'
GROUP BY vu.id, vu.video_file_number, vu.name, vu.status, u.full_name, u.email;

-- 7. Get frames with GPT responses for a specific video (using video_file_number)
SELECT 
    fa.id,
    fa.timestamp,
    fa.frame_number,
    fa.description,
    fa.ocr_text,
    fa.gpt_response->>'description' as gpt_description,
    fa.gpt_response->>'ocr_text' as gpt_ocr,
    fa.gpt_response->>'processing_time_ms' as gpt_processing_time,
    fa.image_path,
    vu.video_file_number,
    vu.name as video_name
FROM frame_analyses fa
INNER JOIN video_uploads vu ON fa.video_id = vu.id
WHERE vu.video_file_number = 'VF-2024-0001'
  AND fa.gpt_response IS NOT NULL
ORDER BY fa.timestamp;

-- 8. Get all videos with their latest frame analysis
SELECT 
    vu.id,
    vu.video_file_number,
    vu.name,
    vu.status,
    fa.timestamp as latest_frame_timestamp,
    fa.description as latest_frame_description,
    fa.created_at as latest_frame_created_at
FROM video_uploads vu
LEFT JOIN LATERAL (
    SELECT timestamp, description, created_at
    FROM frame_analyses
    WHERE video_id = vu.id
    ORDER BY timestamp DESC
    LIMIT 1
) fa ON true
WHERE vu.is_deleted = false
ORDER BY vu.created_at DESC;

-- 9. Get user statistics with video and frame counts
SELECT 
    u.id,
    u.full_name,
    u.email,
    COUNT(DISTINCT vu.id) as total_videos,
    COUNT(DISTINCT fa.id) as total_frames_analyzed,
    COUNT(DISTINCT ual.id) as total_activities
FROM users u
LEFT JOIN video_uploads vu ON u.id = vu.user_id AND vu.is_deleted = false
LEFT JOIN frame_analyses fa ON vu.id = fa.video_id
LEFT JOIN user_activity_logs ual ON u.id = ual.user_id
WHERE u.id = 'USER_UUID_HERE'
GROUP BY u.id, u.full_name, u.email;

-- 10. Get all GPT responses grouped by video
SELECT 
    vu.video_file_number,
    vu.name as video_name,
    COUNT(fa.id) as total_responses,
    AVG((fa.gpt_response->>'processing_time_ms')::integer) as avg_processing_time_ms,
    MIN(fa.timestamp) as first_frame_timestamp,
    MAX(fa.timestamp) as last_frame_timestamp
FROM video_uploads vu
INNER JOIN frame_analyses fa ON vu.id = fa.video_id
WHERE fa.gpt_response IS NOT NULL
  AND vu.is_deleted = false
GROUP BY vu.video_file_number, vu.name
ORDER BY total_responses DESC;

