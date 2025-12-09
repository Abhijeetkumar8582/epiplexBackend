# Database Migrations

This directory contains SQL migration files for setting up the database schema.

## Setup Instructions

### 1. Create PostgreSQL Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE epiplex;

# Exit psql
\q
```

### 2. Run Migrations

Run migrations in order:

```bash
# Connect to your database and run the migrations
psql -U postgres -d epiplex -f migrations/001_create_users_tables.sql
psql -U postgres -d epiplex -f migrations/003_add_oauth_support.sql
psql -U postgres -d epiplex -f migrations/004_create_video_uploads_table.sql
psql -U postgres -d epiplex -f migrations/005_add_business_metadata.sql
psql -U postgres -d epiplex -f migrations/006_create_frame_analyses_table.sql
psql -U postgres -d epiplex -f migrations/007_add_video_file_number.sql
```

Or using psql interactively:

```bash
psql -U postgres -d epiplex
\i migrations/001_create_users_tables.sql
\i migrations/003_add_oauth_support.sql
\i migrations/004_create_video_uploads_table.sql
```

### 3. Verify Tables

```bash
psql -U postgres -d epiplex -c "\dt"
```

You should see:
- `users`
- `user_sessions`
- `user_activity_logs`
- `job_status` (created by SQLAlchemy)

## Migration Files

- `001_create_users_tables.sql` - Creates users, user_sessions, and user_activity_logs tables
- `002_example_queries.sql` - Example SQL queries for reference
- `003_add_oauth_support.sql` - Adds Google OAuth support (google_id, provider columns)
- `004_create_video_uploads_table.sql` - Creates video_uploads table for storing video upload metadata
- `005_add_business_metadata.sql` - Adds business/functional metadata fields (application_name, tags, language_code, priority, soft delete)
- `006_create_frame_analyses_table.sql` - Creates frame_analyses table for storing frame-by-frame analysis results
- `007_add_video_file_number.sql` - Adds video_file_number to video_uploads and gpt_response to frame_analyses
- `008_example_interlinked_queries.sql` - Example SQL queries showing how to fetch interlinked data using relationships

## Environment Variables

Make sure your `.env` file has the correct database URL:

```env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/epiplex
```

## Notes

- The migration assumes the `pgcrypto` extension is available (enabled by default in PostgreSQL 13+)
- UUIDs are generated using `gen_random_uuid()`
- All timestamps use `TIMESTAMP WITH TIME ZONE` for proper timezone handling
- Migration 003 makes `password_hash` nullable to support OAuth users who don't have passwords
