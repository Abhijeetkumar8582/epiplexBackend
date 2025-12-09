-- Example INSERT: Create a user with password hash
-- Note: In production, use bcrypt or similar to hash passwords
INSERT INTO users (full_name, email, password_hash, role)
VALUES (
    'John Doe',
    'john.doe@example.com',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYqBWVHxkd0',  -- Example bcrypt hash
    'user'
);

-- Example SELECT: Get user by email (for login)
SELECT 
    id,
    full_name,
    email,
    password_hash,
    role,
    is_active,
    last_login_at,
    created_at,
    updated_at
FROM users
WHERE email = 'john.doe@example.com';

-- Example SELECT: Get user by id
SELECT 
    id,
    full_name,
    email,
    password_hash,
    role,
    is_active,
    last_login_at,
    created_at,
    updated_at
FROM users
WHERE id = '123e4567-e89b-12d3-a456-426614174000'::UUID;

-- Example SELECT: Get last 20 activity logs for a user
SELECT 
    id,
    user_id,
    action,
    description,
    metadata,
    ip_address,
    created_at
FROM user_activity_logs
WHERE user_id = '123e4567-e89b-12d3-a456-426614174000'::UUID
ORDER BY created_at DESC
LIMIT 20;

