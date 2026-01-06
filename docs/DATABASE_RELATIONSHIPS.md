# Database Relationships and Interlinked Tables

This document describes the SQL table relationships and how to fetch/push data using interlinked tables.

## Table Relationships

### Entity Relationship Diagram

```
users (1) ──< (N) user_sessions
users (1) ──< (N) user_activity_logs
users (1) ──< (N) video_uploads
video_uploads (1) ──< (N) frame_analyses
```

### Foreign Key Relationships

1. **user_sessions.user_id** → **users.id** (ON DELETE CASCADE)
2. **user_activity_logs.user_id** → **users.id** (ON DELETE CASCADE)
3. **video_uploads.user_id** → **users.id** (ON DELETE CASCADE)
4. **frame_analyses.video_id** → **video_uploads.id** (ON DELETE CASCADE)

## SQLAlchemy Relationships

All relationships are properly defined in `app/database.py`:

- `User.sessions` - One-to-many relationship with UserSession
- `User.activity_logs` - One-to-many relationship with UserActivityLog
- `User.video_uploads` - One-to-many relationship with VideoUpload
- `VideoUpload.user` - Many-to-one relationship with User
- `VideoUpload.frame_analyses` - One-to-many relationship with FrameAnalysis
- `FrameAnalysis.video_upload` - Many-to-one relationship with VideoUpload

## Example SQL Queries

See `migrations/008_example_interlinked_queries.sql` for comprehensive examples.

### Key Queries:

1. **Get video uploads with user info:**
```sql
SELECT vu.*, u.full_name, u.email
FROM video_uploads vu
INNER JOIN users u ON vu.user_id = u.id;
```

2. **Get frame analyses with video details:**
```sql
SELECT fa.*, vu.video_file_number, vu.name
FROM frame_analyses fa
INNER JOIN video_uploads vu ON fa.video_id = vu.id
WHERE vu.video_file_number = 'VF-2024-0001';
```

3. **Get all GPT responses for a video:**
```sql
SELECT fa.gpt_response, fa.timestamp
FROM frame_analyses fa
INNER JOIN video_uploads vu ON fa.video_id = vu.id
WHERE vu.video_file_number = 'VF-2024-0001'
  AND fa.gpt_response IS NOT NULL;
```

## Using SQLAlchemy ORM Relationships

### Fetching Related Data

```python
# Get user with all video uploads
user = await db.get(User, user_id)
videos = user.video_uploads  # Automatically loaded via relationship

# Get video with all frame analyses
video = await db.get(VideoUpload, video_id)
frames = video.frame_analyses  # Automatically loaded via relationship

# Get frame with video details
frame = await db.get(FrameAnalysis, frame_id)
video = frame.video_upload  # Automatically loaded via relationship
```

### Eager Loading (JOIN)

```python
from sqlalchemy.orm import selectinload

# Load user with all related data
result = await db.execute(
    select(User)
    .options(
        selectinload(User.video_uploads),
        selectinload(User.activity_logs)
    )
    .where(User.id == user_id)
)
user = result.scalar_one()

# Access related data
for video in user.video_uploads:
    print(f"Video: {video.name}")
    for frame in video.frame_analyses:
        print(f"Frame: {frame.timestamp}")
```

## Data Integrity

All foreign keys have `ON DELETE CASCADE` to ensure:
- When a user is deleted, all their sessions, activity logs, and video uploads are deleted
- When a video upload is deleted, all its frame analyses are deleted
- Referential integrity is maintained at the database level

## Indexes for Performance

All foreign key columns are indexed for fast joins:
- `idx_user_sessions_user_id`
- `idx_user_activity_user_id`
- `idx_video_uploads_user_id`
- `idx_frame_analyses_video_id`
- `idx_frame_analyses_timestamp` (composite index on video_id, timestamp)

## Production Ready

✅ All tables properly interlinked with foreign keys
✅ SQLAlchemy relationships defined for ORM access
✅ Cascade deletes configured for data integrity
✅ Indexes on all foreign keys for performance
✅ Example SQL queries provided for direct SQL access
✅ No mocking - uses real GPT service

