# Authentication System Documentation

This document explains how authentication works in the application, including password hashing and JWT token implementation.

## Overview

The application uses:
- **Bcrypt** for password hashing (via `passlib`)
- **JWT (JSON Web Tokens)** for authentication
- **HS256** algorithm for JWT signing

## Password Hashing

### Implementation

Passwords are hashed using **bcrypt** through the `passlib` library:

```python
# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Hash password during signup
password_hash = AuthService.hash_password(password)  # Returns bcrypt hash

# Verify password during login
is_valid = AuthService.verify_password(plain_password, hashed_password)
```

### Security Features

- **Bcrypt** is a strong, adaptive hashing algorithm
- Automatically includes salt in the hash
- Computationally expensive to prevent brute-force attacks
- Passwords are **never stored in plain text**

### Flow

1. **Signup**: User provides password → Hashed with bcrypt → Stored in database
2. **Login**: User provides password → Hashed password retrieved from DB → Compared using bcrypt verify

## JWT Token Authentication

### Implementation

JWT tokens are created and verified using `python-jose`:

```python
# Create JWT token
access_token = AuthService.create_access_token(
    data={"sub": str(user.id), "email": user.email, "role": user.role},
    expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
)

# Verify JWT token
payload = AuthService.verify_token(token)
```

### Token Structure

JWT tokens contain:
- **sub**: User ID (subject)
- **email**: User's email address
- **role**: User's role (user/admin)
- **exp**: Expiration timestamp

### Configuration

```python
# In config.py
SECRET_KEY: str = "your-secret-key-change-in-production"
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
```

**⚠️ Important**: Change `SECRET_KEY` in production! Generate a secure key:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Authentication Flow

### 1. User Signup

```
POST /api/auth/signup
Body: { "full_name": "...", "email": "...", "password": "..." }

Process:
1. Validate input
2. Check if email exists
3. Hash password with bcrypt
4. Create user in database
5. Return user info (NO token - user must login)
```

### 2. User Login

```
POST /api/auth/login
Body: { "email": "...", "password": "..." }

Process:
1. Find user by email
2. Verify password using bcrypt
3. Create JWT access token
4. Create session record (for tracking)
5. Update last_login_at
6. Return JWT token + user info

Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "session_token": "...",  // For tracking only, not used for auth
  "token_type": "bearer",
  "user": { ... },
  "expires_at": "2024-01-01T00:00:00Z"
}
```

### 3. Using JWT Token

```
GET /api/auth/me
Headers: Authorization: Bearer <access_token>

Process:
1. Extract token from Authorization header
2. Verify JWT token signature
3. Decode token payload
4. Get user from database using user_id
5. Return user info
```

## Protected Endpoints

All protected endpoints use the `get_current_user` dependency:

```python
@app.get("/api/uploads")
async def get_uploads(
    current_user: User = Depends(get_current_user),  # Requires JWT token
    ...
):
    # Only accessible with valid JWT token
    ...
```

### How It Works

1. Client sends request with `Authorization: Bearer <token>` header
2. FastAPI's `HTTPBearer` extracts the token
3. `get_current_user` dependency:
   - Verifies JWT token signature
   - Decodes token payload
   - Retrieves user from database
   - Returns user object
4. If token is invalid/expired → Returns 401 Unauthorized

## Security Best Practices

### ✅ Implemented

- Passwords are hashed with bcrypt (never stored in plain text)
- JWT tokens are signed with a secret key
- Tokens have expiration time (7 days default)
- User accounts can be deactivated (`is_active` flag)
- Rate limiting on auth endpoints
- Activity logging for security auditing

### 🔒 Recommendations

1. **Change SECRET_KEY in production**
   ```env
   SECRET_KEY=your-very-secure-random-key-here
   ```

2. **Use HTTPS in production**
   - JWT tokens should only be sent over HTTPS
   - Prevents token interception

3. **Implement refresh tokens** (optional)
   - Short-lived access tokens (15-30 min)
   - Long-lived refresh tokens (7-30 days)
   - Reduces risk if access token is compromised

4. **Token rotation** (optional)
   - Rotate tokens on each use
   - Invalidate old tokens

5. **Rate limiting**
   - Already implemented on login/signup endpoints
   - Prevents brute-force attacks

## Example Usage

### Frontend Implementation

```javascript
// Login
const response = await fetch('/api/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ email, password })
});

const { access_token, user } = await response.json();

// Store token
localStorage.setItem('access_token', access_token);

// Use token for authenticated requests
const userData = await fetch('/api/auth/me', {
  headers: {
    'Authorization': `Bearer ${access_token}`
  }
});
```

### Python Client Example

```python
import requests

# Login
response = requests.post('http://localhost:8000/api/auth/login', json={
    'email': 'user@example.com',
    'password': 'password123'
})

data = response.json()
access_token = data['access_token']

# Use token
headers = {'Authorization': f'Bearer {access_token}'}
user_info = requests.get('http://localhost:8000/api/auth/me', headers=headers)
```

## Troubleshooting

### "Invalid authentication credentials"
- Token is expired → User needs to login again
- Token is malformed → Check token format
- Secret key mismatch → Ensure SECRET_KEY matches

### "User not found"
- Token is valid but user was deleted
- User ID in token doesn't exist in database

### "User account is inactive"
- User account has been deactivated
- Contact administrator

## Dependencies

- `bcrypt==4.1.2` - Password hashing
- `passlib[bcrypt]==1.7.4` - Password hashing library
- `python-jose[cryptography]==3.3.0` - JWT token handling

## Summary

✅ **Password Hashing**: Implemented with bcrypt  
✅ **JWT Tokens**: Implemented with python-jose  
✅ **Token Verification**: Automatic via FastAPI dependencies  
✅ **Security**: Rate limiting, activity logging, account status checks  

The authentication system is production-ready and follows security best practices!

