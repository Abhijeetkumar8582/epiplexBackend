"""
Metrics Service for Performance Monitoring
Tracks response times, request counts, percentiles, and cache statistics
"""
import time
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
from threading import Lock
from datetime import datetime, timezone, timedelta

from app.config import settings
from app.utils.logger import logger


class MetricsService:
    """
    Thread-safe metrics collection service
    Tracks response times, request counts, and calculates percentiles
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MetricsService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._lock = Lock()
        
        # Configuration
        self.retention_hours = getattr(settings, 'METRICS_RETENTION_HOURS', 24)
        self.percentiles = getattr(settings, 'METRICS_PERCENTILES', [0.5, 0.75, 0.9, 0.95, 0.99])
        self.enabled = getattr(settings, 'METRICS_ENABLED', True)
        
        # Response time tracking: endpoint -> deque of (timestamp, duration_ms)
        self._response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Request counts: endpoint -> {status_code: count}
        self._request_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        # Cache statistics: cache_type -> {hits: int, misses: int}
        self._cache_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'hits': 0, 'misses': 0})
        
        # Slow queries: list of (timestamp, query, duration_ms, endpoint)
        self._slow_queries: deque = deque(maxlen=100)
        
        # Cleanup old data periodically
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # 1 hour
    
    def record_response_time(self, endpoint: str, method: str, duration_ms: float, status_code: int):
        """Record response time for an endpoint"""
        if not self.enabled:
            return
        
        key = f"{method} {endpoint}"
        timestamp = time.time()
        
        with self._lock:
            self._response_times[key].append((timestamp, duration_ms))
            self._request_counts[key][status_code] += 1
            
            # Cleanup old data periodically
            if time.time() - self._last_cleanup > self._cleanup_interval:
                self._cleanup_old_data()
    
    def record_cache_hit(self, cache_type: str):
        """Record a cache hit"""
        if not self.enabled:
            return
        
        with self._lock:
            self._cache_stats[cache_type]['hits'] += 1
    
    def record_cache_miss(self, cache_type: str):
        """Record a cache miss"""
        if not self.enabled:
            return
        
        with self._lock:
            self._cache_stats[cache_type]['misses'] += 1
    
    def record_slow_query(self, query: str, duration_ms: float, endpoint: Optional[str] = None):
        """Record a slow database query"""
        if not self.enabled:
            return
        
        with self._lock:
            self._slow_queries.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'query': query[:500],  # Limit query length
                'duration_ms': round(duration_ms, 2),
                'endpoint': endpoint or 'unknown'
            })
    
    def _cleanup_old_data(self):
        """Remove data older than retention period"""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        for endpoint in list(self._response_times.keys()):
            times = self._response_times[endpoint]
            # Remove old entries
            while times and times[0][0] < cutoff_time:
                times.popleft()
            
            # Remove empty entries
            if not times:
                del self._response_times[endpoint]
        
        self._last_cleanup = time.time()
    
    def get_response_time_stats(self, endpoint: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Get response time statistics (percentiles, avg, max)
        
        Args:
            endpoint: Optional endpoint filter (e.g., "GET /api/videos/panel")
        
        Returns:
            Dictionary mapping endpoint to statistics
        """
        if not self.enabled:
            return {}
        
        stats = {}
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        with self._lock:
            endpoints = [endpoint] if endpoint else self._response_times.keys()
            
            for ep in endpoints:
                if ep not in self._response_times:
                    continue
                
                # Filter recent data
                recent_times = [
                    duration for timestamp, duration in self._response_times[ep]
                    if timestamp >= cutoff_time
                ]
                
                if not recent_times:
                    continue
                
                # Calculate statistics
                sorted_times = sorted(recent_times)
                n = len(sorted_times)
                
                percentile_values = {}
                for p in self.percentiles:
                    index = int(n * p)
                    if index >= n:
                        index = n - 1
                    percentile_key = f"p{int(p * 100)}"
                    percentile_values[percentile_key] = sorted_times[index]
                
                stats[ep] = {
                    **percentile_values,
                    'avg': sum(recent_times) / n,
                    'max': max(recent_times),
                    'min': min(recent_times),
                    'count': n
                }
        
        return stats
    
    def get_request_counts(self) -> Dict[str, Dict[int, int]]:
        """Get request counts by endpoint and status code"""
        if not self.enabled:
            return {}
        
        with self._lock:
            return {
                endpoint: dict(counts)
                for endpoint, counts in self._request_counts.items()
            }
    
    def get_cache_stats(self) -> Dict[str, Dict[str, any]]:
        """Get cache statistics with hit rates"""
        if not self.enabled:
            return {}
        
        with self._lock:
            stats = {}
            for cache_type, data in self._cache_stats.items():
                hits = data['hits']
                misses = data['misses']
                total = hits + misses
                hit_rate = hits / total if total > 0 else 0.0
                
                stats[cache_type] = {
                    'hits': hits,
                    'misses': misses,
                    'total': total,
                    'hit_rate': round(hit_rate, 4)
                }
            
            return stats
    
    def get_slow_queries(self, limit: int = 100) -> List[Dict[str, any]]:
        """Get list of slow queries"""
        if not self.enabled:
            return []
        
        with self._lock:
            return list(self._slow_queries)[-limit:]
    
    def get_top_endpoints(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get top endpoints by request count"""
        if not self.enabled:
            return []
        
        with self._lock:
            endpoint_totals = []
            for endpoint, counts in self._request_counts.items():
                total = sum(counts.values())
                endpoint_totals.append((endpoint, total))
            
            # Sort by total count descending
            endpoint_totals.sort(key=lambda x: x[1], reverse=True)
            return endpoint_totals[:limit]
    
    def get_error_rates(self) -> Dict[str, float]:
        """Get error rates (4xx + 5xx) per endpoint"""
        if not self.enabled:
            return {}
        
        with self._lock:
            error_rates = {}
            for endpoint, counts in self._request_counts.items():
                total = sum(counts.values())
                if total == 0:
                    continue
                
                errors = sum(
                    count for status, count in counts.items()
                    if 400 <= status < 600
                )
                error_rates[endpoint] = round(errors / total, 4) if total > 0 else 0.0
            
            return error_rates
    
    def reset(self):
        """Reset all metrics (for testing)"""
        with self._lock:
            self._response_times.clear()
            self._request_counts.clear()
            self._cache_stats.clear()
            self._slow_queries.clear()


# Singleton instance
metrics_service = MetricsService()

