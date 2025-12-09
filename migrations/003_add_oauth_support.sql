-- Add OAuth support to users table
-- This migration adds Google OAuth fields and makes password_hash nullable

-- Add google_id column (for Google OAuth users)
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS google_id VARCHAR(255) UNIQUE;

-- Add provider column (to track auth method: 'email' or 'google')
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS provider VARCHAR(50) NOT NULL DEFAULT 'email';

-- Make password_hash nullable (OAuth users don't have passwords)
ALTER TABLE users 
ALTER COLUMN password_hash DROP NOT NULL;

-- Create index on google_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_users_google_id ON users (google_id);

-- Update existing users to have provider='email' if not set
UPDATE users 
SET provider = 'email' 
WHERE provider IS NULL OR provider = '';

