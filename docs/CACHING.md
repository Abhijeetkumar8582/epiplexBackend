# Caching Implementation

This document describes the caching strategy implemented in the Epiplex application for improved performance.

## Overview

The application implements a two-tier caching strategy:
1. **Backend Caching**: In-memory caching using `cachetools` with TTL-based expiration
2. **Frontend Caching**: Client-side caching using in-memory Map with configurable expiration

## Backend Caching

### Cache Service

Location: `backend/app/services/cache_service.py`

The `CacheService` provides:
- **TTL-based expiration**: Automatic cache invalidation after time-to-live expires
- **Type-specific caches**: Different cache instances for different data types
- **User-specific keys**: Cache keys include user ID for security
- **Pattern-based invalidation**: Clear cache entries matching patterns
- **Automatic cleanup**: Expired entries are automatically removed

### Cache Types and TTLs

| Cache Type | TTL | Max Size | Description |
|------------|-----|----------|-------------|
| `video_panel` | 60s | 100 | Video list with statistics |
| `document_data` | 300s | 50 | Complete document data (frames, metadata) |
| `video_upload` | 120s | 200 | Individual video upload details |
| `activity_stats` | 180s | 50 | Activity log statistics |
| `user_data` | 300s | 100 | User profile data |
| `dashboard_stats` | 180s | 50 | Dashboard statistics |
| `default` | 120s | 100 | Default cache settings |

### Cached Endpoints

The following endpoints are cached:

1. **GET `/api/videos/panel`**
   - Cache type: `video_panel`
   - TTL: 60 seconds
   - Cache key includes: user_id, page, page_size, filters, sort parameters

2. **GET `/api/videos/file-number/{video_file_number}/document`**
   - Cache type: `document_data`
   - TTL: 300 seconds (5 minutes)
   - Only cached if status is "completed"
   - Cache key includes: user_id, video_file_number

3. **GET `/api/activity-logs/stats`**
   - Cache type: `activity_stats`
   - TTL: 180 seconds (3 minutes)
   - Cache key includes: user_id, days parameter

### Cache Invalidation

Cache is automatically invalidated on:

- **Video Updates**: When video metadata is updated via `PATCH /api/uploads/{upload_id}`
- **Video Deletion**: When videos are deleted (soft or hard delete)
- **Bulk Operations**: When bulk delete operations occur
- **Video Upload**: When new videos are uploaded
- **Processing Completion**: When video processing completes and status changes to "completed"
- **User-specific**: All user-related cache entries can be invalidated via `invalidate_user_cache()`
- **Video-specific**: All video-related cache entries can be invalidated via `invalidate_video_cache()`

### Usage Example

```python
from app.services.cache_service import CacheService

# Generate cache key
cache_key = CacheService._generate_cache_key(
    prefix="video_panel",
    user_id=current_user.id,
    page=page,
    status=status
)

# Check cache
cached_response = CacheService.get(cache_key, "video_panel")
if cached_response is not None:
    return cached_response

# ... fetch data from database ...

# Store in cache
CacheService.set(cache_key, response, "video_panel")
```

## Frontend Caching

### Cache Utility

Location: `frontend/lib/dataCache.js`

The frontend cache provides:
- **In-memory storage**: Using JavaScript Map
- **Automatic expiration**: Entries expire based on TTL
- **Periodic cleanup**: Expired entries are cleaned up every minute
- **Pattern-based clearing**: Clear cache entries matching patterns

### Cache Durations

| Data Type | Duration | Description |
|-----------|----------|-------------|
| Dashboard Stats | 5 minutes | Dashboard statistics |
| Video List | 2 minutes | Video panel/list data |
| Document Data | 10 minutes | Complete document data |
| User Data | 5 minutes | User profile information |
| Default | 2 minutes | Default cache duration |

### Cache Invalidation

Frontend cache is invalidated when:
- Videos are uploaded
- Videos are deleted
- Videos are updated
- Bulk operations occur
- User navigates between pages (selective clearing)

## Performance Benefits

### Backend Caching Benefits

1. **Reduced Database Load**: Frequently accessed data is served from memory
2. **Faster Response Times**: Cached responses are returned immediately
3. **Better Scalability**: Reduces database connection pool usage
4. **Cost Savings**: Fewer database queries mean lower infrastructure costs

### Frontend Caching Benefits

1. **Instant Navigation**: Cached data loads immediately when navigating between pages
2. **Reduced API Calls**: Fewer requests to the backend
3. **Better UX**: Faster page loads and smoother user experience
4. **Offline Resilience**: Cached data available even with network issues

## Cache Strategy

### When to Cache

- ✅ Frequently accessed data (video lists, document data)
- ✅ Expensive queries (joins, aggregations, statistics)
- ✅ Relatively stable data (completed documents)
- ✅ User-specific data (personalized content)

### When NOT to Cache

- ❌ Real-time data (processing status, live updates)
- ❌ Frequently changing data (unless short TTL)
- ❌ Sensitive operations (authentication, authorization)
- ❌ Large binary data (files, images)

## Configuration

### Backend Configuration

Cache settings can be configured in `backend/app/config.py`:

```python
CACHE_ENABLED: bool = True
CACHE_DEFAULT_TTL: int = 120  # seconds
```

### Frontend Configuration

Cache durations are defined in `frontend/lib/dataCache.js`:

```javascript
const CACHE_DURATION = {
  DASHBOARD_STATS: 5 * 60 * 1000,  // 5 minutes
  VIDEO_LIST: 2 * 60 * 1000,       // 2 minutes
  DOCUMENT_DATA: 10 * 60 * 1000,   // 10 minutes
  // ...
};
```

## Monitoring

Cache performance can be monitored through:
- Application logs (cache hits/misses are logged)
- Response time metrics
- Database query reduction

## Future Enhancements

Potential improvements:
1. **Redis Support**: Distributed caching for multi-instance deployments
2. **Cache Metrics**: Detailed cache hit/miss statistics
3. **Adaptive TTL**: Dynamic TTL based on data change frequency
4. **Cache Warming**: Pre-populate cache with frequently accessed data
5. **Cache Compression**: Compress large cached responses

## Troubleshooting

### Cache Not Working

1. Check if `CACHE_ENABLED` is `True` in settings
2. Verify cache keys are generated correctly
3. Check cache TTL hasn't expired
4. Review logs for cache hit/miss information

### Stale Data

1. Verify cache invalidation is triggered on mutations
2. Check TTL values are appropriate
3. Ensure cache keys include all relevant parameters
4. Clear cache manually if needed: `CacheService.clear()`

### Memory Issues

1. Reduce `CACHE_MAXSIZE` for cache types
2. Reduce TTL values to expire entries faster
3. Monitor cache sizes in logs
4. Consider implementing cache eviction policies

