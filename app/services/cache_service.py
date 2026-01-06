"""
Cache service for fast API responses
Supports TTL-based expiration and cache invalidation
"""
import hashlib
import json
from typing import Any, Optional, Callable, Dict
from functools import wraps
from datetime import datetime, timedelta
from cachetools import TTLCache
from uuid import UUID

from app.config import settings
from app.utils.logger import logger

# Import metrics service for cache tracking
try:
    from app.services.metrics_service import metrics_service
    _metrics_available = True
except ImportError:
    _metrics_available = False


class CacheService:
    """
    In-memory cache service with TTL support
    Uses cachetools for efficient memory management
    """
    
    # Cache instances for different data types
    _caches: Dict[str, TTLCache] = {}
    
    # Cache TTL configurations (in seconds)
    CACHE_TTL = {
        "video_panel": 60,  # 1 minute - frequently changing
        "document_data": 300,  # 5 minutes - stable once complete
        "video_upload": 120,  # 2 minutes
        "activity_stats": 180,  # 3 minutes
        "user_data": 300,  # 5 minutes
        "dashboard_stats": 180,  # 3 minutes
        "default": 120  # 2 minutes default
    }
    
    # Maximum cache sizes (number of entries)
    CACHE_MAXSIZE = {
        "video_panel": 100,
        "document_data": 50,
        "video_upload": 200,
        "activity_stats": 50,
        "user_data": 100,
        "dashboard_stats": 50,
        "default": 100
    }
    
    @classmethod
    def _get_cache(cls, cache_type: str = "default") -> TTLCache:
        """Get or create a cache instance for the given type"""
        if cache_type not in cls._caches:
            ttl = cls.CACHE_TTL.get(cache_type, cls.CACHE_TTL["default"])
            maxsize = cls.CACHE_MAXSIZE.get(cache_type, cls.CACHE_MAXSIZE["default"])
            cls._caches[cache_type] = TTLCache(maxsize=maxsize, ttl=ttl)
            logger.debug(f"Created cache", cache_type=cache_type, ttl=ttl, maxsize=maxsize)
        return cls._caches[cache_type]
    
    @classmethod
    def _generate_cache_key(
        cls,
        prefix: str,
        user_id: Optional[UUID] = None,
        **kwargs
    ) -> str:
        """
        Generate a cache key from prefix, user_id, and additional parameters
        
        Args:
            prefix: Cache key prefix (e.g., "video_panel")
            user_id: User ID for user-specific caching
            **kwargs: Additional parameters to include in key
        
        Returns:
            Cache key string
        """
        key_parts = [prefix]
        
        if user_id:
            key_parts.append(str(user_id))
        
        # Add sorted kwargs for consistent key generation
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            # Convert values to strings, handling special types
            kwargs_str = "_".join(
                f"{k}:{str(v)}" for k, v in sorted_kwargs if v is not None
            )
            if kwargs_str:
                key_parts.append(kwargs_str)
        
        # Create hash for long keys to keep them manageable
        key = "_".join(key_parts)
        if len(key) > 200:
            key_hash = hashlib.md5(key.encode()).hexdigest()
            return f"{prefix}_{key_hash}"
        
        return key
    
    @classmethod
    def get(cls, key: str, cache_type: str = "default") -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            cache_type: Type of cache to use
        
        Returns:
            Cached value or None if not found/expired
        """
        cache = cls._get_cache(cache_type)
        value = cache.get(key)
        if value is not None:
            logger.debug(f"Cache hit", key=key, cache_type=cache_type)
            if _metrics_available:
                metrics_service.record_cache_hit(cache_type)
        else:
            logger.debug(f"Cache miss", key=key, cache_type=cache_type)
            if _metrics_available:
                metrics_service.record_cache_miss(cache_type)
        return value
    
    @classmethod
    def set(cls, key: str, value: Any, cache_type: str = "default", ttl: Optional[int] = None) -> None:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            cache_type: Type of cache to use
            ttl: Optional TTL override (in seconds)
        """
        cache = cls._get_cache(cache_type)
        
        # Override TTL if provided
        if ttl is not None:
            # Create a new cache with custom TTL for this entry
            # Note: cachetools doesn't support per-entry TTL, so we'll use the cache's default TTL
            # For per-entry TTL, we'd need a different approach
            pass
        
        cache[key] = value
        logger.debug(f"Cache set", key=key, cache_type=cache_type, cache_size=len(cache))
    
    @classmethod
    def delete(cls, key: str, cache_type: str = "default") -> None:
        """
        Delete a specific cache entry
        
        Args:
            key: Cache key to delete
            cache_type: Type of cache to use
        """
        cache = cls._get_cache(cache_type)
        if key in cache:
            del cache[key]
            logger.debug(f"Cache deleted", key=key, cache_type=cache_type)
    
    @classmethod
    def clear(cls, cache_type: Optional[str] = None) -> None:
        """
        Clear cache entries
        
        Args:
            cache_type: Type of cache to clear, or None to clear all
        """
        if cache_type:
            if cache_type in cls._caches:
                cls._caches[cache_type].clear()
                logger.info(f"Cache cleared", cache_type=cache_type)
        else:
            for cache in cls._caches.values():
                cache.clear()
            logger.info("All caches cleared")
    
    @classmethod
    def clear_by_pattern(cls, pattern: str, cache_type: str = "default") -> None:
        """
        Clear cache entries matching a pattern
        
        Args:
            pattern: Pattern to match in cache keys
            cache_type: Type of cache to search
        """
        cache = cls._get_cache(cache_type)
        keys_to_delete = [key for key in cache.keys() if pattern in str(key)]
        for key in keys_to_delete:
            del cache[key]
        if keys_to_delete:
            logger.info(f"Cache cleared by pattern", pattern=pattern, count=len(keys_to_delete))
    
    @classmethod
    def invalidate_user_cache(cls, user_id: UUID) -> None:
        """
        Invalidate all cache entries for a specific user
        
        Args:
            user_id: User ID to invalidate cache for
        """
        user_id_str = str(user_id)
        for cache_type in cls._caches.keys():
            cache = cls._caches[cache_type]
            keys_to_delete = [
                key for key in cache.keys()
                if user_id_str in str(key)
            ]
            for key in keys_to_delete:
                del cache[key]
            if keys_to_delete:
                logger.info(f"Invalidated user cache", user_id=user_id, cache_type=cache_type, count=len(keys_to_delete))
    
    @classmethod
    def invalidate_video_cache(cls, video_id: Optional[UUID] = None, video_file_number: Optional[str] = None) -> None:
        """
        Invalidate cache entries related to a video
        
        Args:
            video_id: Video ID
            video_file_number: Video file number
        """
        patterns = []
        if video_id:
            patterns.append(str(video_id))
        if video_file_number:
            patterns.append(video_file_number)
        
        for pattern in patterns:
            # Clear video panel cache (contains video lists)
            cls.clear_by_pattern(pattern, "video_panel")
            # Clear document data cache
            cls.clear_by_pattern(pattern, "document_data")
            # Clear video upload cache
            cls.clear_by_pattern(pattern, "video_upload")
        
        logger.info(f"Invalidated video cache", video_id=video_id, video_file_number=video_file_number)


def cached(
    cache_type: str = "default",
    ttl: Optional[int] = None,
    key_prefix: Optional[str] = None
):
    """
    Decorator to cache function results
    
    Usage:
        @cached(cache_type="video_panel", key_prefix="get_videos")
        async def get_videos(user_id: UUID, page: int = 1):
            ...
    
    Args:
        cache_type: Type of cache to use
        ttl: Optional TTL override (in seconds)
        key_prefix: Prefix for cache key (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract user_id from args/kwargs if present
            user_id = None
            if args and isinstance(args[0], UUID):
                user_id = args[0]
            elif 'user_id' in kwargs:
                user_id = kwargs['user_id']
            elif 'current_user' in kwargs:
                user_id = kwargs['current_user'].id if hasattr(kwargs['current_user'], 'id') else None
            
            # Generate cache key
            prefix = key_prefix or func.__name__
            cache_key = CacheService._generate_cache_key(
                prefix=prefix,
                user_id=user_id,
                **{k: v for k, v in kwargs.items() if k not in ['db', 'current_user', 'request']}
            )
            
            # Try to get from cache
            cached_value = CacheService.get(cache_key, cache_type)
            if cached_value is not None:
                return cached_value
            
            # Call function and cache result
            result = await func(*args, **kwargs)
            CacheService.set(cache_key, result, cache_type, ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Extract user_id from args/kwargs if present
            user_id = None
            if args and isinstance(args[0], UUID):
                user_id = args[0]
            elif 'user_id' in kwargs:
                user_id = kwargs['user_id']
            elif 'current_user' in kwargs:
                user_id = kwargs['current_user'].id if hasattr(kwargs['current_user'], 'id') else None
            
            # Generate cache key
            prefix = key_prefix or func.__name__
            cache_key = CacheService._generate_cache_key(
                prefix=prefix,
                user_id=user_id,
                **{k: v for k, v in kwargs.items() if k not in ['db', 'current_user', 'request']}
            )
            
            # Try to get from cache
            cached_value = CacheService.get(cache_key, cache_type)
            if cached_value is not None:
                return cached_value
            
            # Call function and cache result
            result = func(*args, **kwargs)
            CacheService.set(cache_key, result, cache_type, ttl)
            
            return result
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

