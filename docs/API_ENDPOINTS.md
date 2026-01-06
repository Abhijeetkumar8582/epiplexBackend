# API Endpoints Documentation

Complete list of all API endpoints in the Video Processing API.

**Base URL**: `http://localhost:8000` (or your server URL)

**Authentication**: Most endpoints require JWT Bearer token in the `Authorization` header:
```
Authorization: Bearer <your_access_token>
```

---

## 📋 Table of Contents

1. [Health & Status](#health--status)
2. [Authentication](#authentication)
3. [Video Upload & Management](#video-upload--management)
4. [Frame Analysis](#frame-analysis)
5. [Activity Logs](#activity-logs)
6. [Document & GPT Responses](#document--gpt-responses)

---

## Health & Status

### GET `/`
**Description**: Root endpoint - API information  
**Authentication**: ❌ No  
**Response**: API version and status

```json
{
  "message": "Video Processing API",
  "version": "1.0.0",
  "status": "operational"
}
```

---

### GET `/health`
**Description**: Basic health check  
**Authentication**: ❌ No  
**Response**: Health status

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "database": "connected"
}
```

---

### GET `/api/health`
**Description**: Detailed health check with service status  
**Authentication**: ❌ No  
**Response**: Detailed health information

```json
{
  "status": "healthy",
  "services": {
    "database": "operational",
    "openai": "configured"
  }
}
```

---

## Authentication

### POST `/api/auth/signup`
**Description**: Register a new user  
**Authentication**: ❌ No  
**Rate Limit**: 5/minute  
**Request Body**:
```json
{
  "full_name": "John Doe",
  "email": "john@example.com",
  "password": "securepassword123"
}
```
**Response**: User details and registration confirmation

---

### POST `/api/auth/login`
**Description**: Login user and get access token  
**Authentication**: ❌ No  
**Rate Limit**: 10/minute  
**Request Body**:
```json
{
  "email": "john@example.com",
  "password": "securepassword123"
}
```
**Response**: Access token, session token, user info, and expiration

---

### GET `/api/auth/me`
**Description**: Get current authenticated user information  
**Authentication**: ✅ Required  
**Response**: Current user details

---

### GET `/api/auth/google`
**Description**: Initiate Google OAuth flow - redirects to Google  
**Authentication**: ❌ No  
**Rate Limit**: 10/minute  
**Query Parameters**:
- `redirect_uri` (optional): Frontend callback URL

**Response**: Redirects to Google OAuth consent page

---

### GET `/api/auth/google/callback`
**Description**: Handle Google OAuth callback  
**Authentication**: ❌ No  
**Query Parameters**:
- `code`: Authorization code from Google
- `state` (optional): State token
- `error` (optional): Error from Google

**Response**: Redirects to frontend with tokens

---

### POST `/api/auth/google/token`
**Description**: Exchange Google OAuth code for tokens (alternative to callback)  
**Authentication**: ❌ No  
**Rate Limit**: 10/minute  
**Query Parameters**:
- `code`: Authorization code from Google

**Response**: Access token, session token, and user info

---

## Video Upload & Management

### POST `/api/upload`
**Description**: Upload video file and start processing  
**Authentication**: ✅ Required  
**Rate Limit**: Configurable (default: 10/minute)  
**Request**: `multipart/form-data`
- `file`: Video file (MP4, AVI, MOV, MKV, WEBM)
- `name` (optional): Video name
- `application_name` (optional): Application name (e.g., SAP, Salesforce)
- `tags` (optional): Comma-separated tags
- `language_code` (optional): Language code (e.g., en, hi)
- `priority` (optional): Priority level (normal, high) - default: normal

**Response**: Video upload details with unique video file number

---

### GET `/api/uploads`
**Description**: Get paginated list of user's video uploads with filtering  
**Authentication**: ✅ Required  
**Query Parameters**:
- `page` (default: 1): Page number
- `page_size` (default: 20, max: 100): Items per page
- `status` (optional): Filter by status (uploaded, processing, completed, failed, cancelled)
- `include_deleted` (default: false): Include soft-deleted uploads
- `application_name` (optional): Filter by application name
- `language_code` (optional): Filter by language code
- `priority` (optional): Filter by priority (normal, high)
- `tags` (optional): Filter by tags (comma-separated)

**Response**: Paginated list of video uploads

---

### GET `/api/videos/panel`
**Description**: Get all videos for panel/list view with frame statistics  
**Authentication**: ✅ Required  
**Query Parameters**:
- `page` (default: 1): Page number
- `page_size` (default: 20, max: 100): Items per page
- `status` (optional): Filter by status
- `application_name` (optional): Filter by application name
- `language_code` (optional): Filter by language code
- `priority` (optional): Filter by priority
- `tags` (optional): Filter by tags (comma-separated)
- `sort_by` (default: updated_at): Sort field (updated_at, created_at, name, status)
- `sort_order` (default: desc): Sort order (asc, desc)

**Response**: Videos with frame analysis statistics (total_frames, frames_with_gpt)

---

### GET `/api/uploads/{upload_id}`
**Description**: Get specific video upload by ID  
**Authentication**: ✅ Required  
**Path Parameters**:
- `upload_id`: UUID of the video upload

**Response**: Video upload details

---

### PATCH `/api/uploads/{upload_id}`
**Description**: Update video upload metadata  
**Authentication**: ✅ Required  
**Path Parameters**:
- `upload_id`: UUID of the video upload

**Request Body** (all fields optional):
```json
{
  "name": "Updated Video Name",
  "status": "completed",
  "application_name": "SAP",
  "tags": ["HR", "Payroll"],
  "language_code": "en",
  "priority": "high"
}
```

**Response**: Updated video upload details

---

### DELETE `/api/uploads/{upload_id}`
**Description**: Delete a video upload (soft delete by default)  
**Authentication**: ✅ Required  
**Path Parameters**:
- `upload_id`: UUID of the video upload

**Query Parameters**:
- `permanent` (default: false): Permanently delete (hard delete)

**Response**: Success message

---

### POST `/api/uploads/{upload_id}/restore`
**Description**: Restore a soft-deleted video upload  
**Authentication**: ✅ Required  
**Path Parameters**:
- `upload_id`: UUID of the video upload

**Response**: Restored video upload details

---

## Frame Analysis

### GET `/api/videos/{video_id}/frames`
**Description**: Get frame analyses for a video  
**Authentication**: ✅ Required  
**Path Parameters**:
- `video_id`: UUID of the video

**Query Parameters**:
- `limit` (optional, max: 1000): Maximum number of frames to return
- `offset` (default: 0): Number of frames to skip

**Response**: List of frame analyses with descriptions, OCR text, and GPT responses

---

## Activity Logs

### GET `/api/activity-logs`
**Description**: Get paginated activity logs for the current user with filtering  
**Authentication**: ✅ Required  
**Query Parameters**:
- `page` (default: 1): Page number
- `page_size` (default: 20, max: 100): Items per page
- `action` (optional): Filter by action type (e.g., LOGIN, UPLOAD_VIDEO)
- `start_date` (optional): Start date (YYYY-MM-DD or ISO format)
- `end_date` (optional): End date (YYYY-MM-DD or ISO format)
- `search` (optional): Search in descriptions

**Response**: Paginated list of activity logs

---

### GET `/api/activity-logs/{log_id}`
**Description**: Get a specific activity log by ID  
**Authentication**: ✅ Required  
**Path Parameters**:
- `log_id`: Activity log ID

**Response**: Activity log details

---

### GET `/api/activity-logs/stats`
**Description**: Get activity statistics for the current user  
**Authentication**: ✅ Required  
**Query Parameters**:
- `days` (default: 30, max: 365): Number of days to include in statistics

**Response**: Activity statistics including total activities, activities by action, and recent activities

---

### GET `/api/activity-logs/actions`
**Description**: Get list of available action types for the current user  
**Authentication**: ✅ Required  
**Response**: List of distinct action types

---

## Document & GPT Responses

### GET `/api/videos/file-number/{video_file_number}/gpt-responses`
**Description**: Get all GPT responses for a video by video file number  
**Authentication**: ✅ Required  
**Path Parameters**:
- `video_file_number`: Video file number (e.g., VF-2024-0001)

**Response**: List of all frame analyses with GPT responses for the video

---

### GET `/api/videos/file-number/{video_file_number}/document`
**Description**: Get complete document/data for a video file number  
**Authentication**: ✅ Required  
**Path Parameters**:
- `video_file_number`: Video file number (e.g., VF-2024-0001)

**Response**: Complete document data including:
- Video metadata
- All frame analyses with GPT responses
- Summary statistics (total frames, frames with GPT, processing times, etc.)

---

## Job Status & Downloads

### GET `/api/status/{job_id}`
**Description**: Get processing status for a job  
**Authentication**: ❌ No (public endpoint)  
**Path Parameters**:
- `job_id`: Job ID (UUID)

**Response**: Job status with progress, current step, and step progress

---

### GET `/api/download/{job_id}`
**Description**: Download generated document in specified format  
**Authentication**: ✅ Required  
**Path Parameters**:
- `job_id`: Job ID (UUID)

**Query Parameters**:
- `format` (default: docx): Document format (docx, pdf, html)

**Response**: File download

---

## 📝 Notes

### Authentication
- Most endpoints require JWT Bearer token authentication
- Token is obtained from `/api/auth/login` or Google OAuth endpoints
- Token expires after configured time (default: 7 days)

### Rate Limiting
- Some endpoints have rate limiting to prevent abuse
- Rate limits are configurable in settings
- Exceeding rate limit returns 429 Too Many Requests

### Error Responses
All endpoints return standard HTTP status codes:
- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `429`: Too Many Requests
- `500`: Internal Server Error

### Pagination
Paginated endpoints return:
```json
{
  "items": [...],
  "total": 100,
  "page": 1,
  "page_size": 20,
  "has_more": true
}
```

### Video File Numbers
- Each uploaded video gets a unique file number: `VF-YYYY-NNNN`
- Example: `VF-2024-0001`, `VF-2024-0002`
- Used to fetch all GPT responses and document data

---

## 🔗 Interactive API Documentation

When running in debug mode, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These provide interactive API documentation with the ability to test endpoints directly.

