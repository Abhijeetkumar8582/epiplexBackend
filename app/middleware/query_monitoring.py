"""
Database Query Monitoring Middleware
Tracks database query execution times and detects slow queries
"""
from sqlalchemy import event
from sqlalchemy.engine import Engine
from typing import Dict, Optional
import time
import re

from app.config import settings
from app.utils.logger import logger
from app.services.metrics_service import metrics_service


class QueryMonitoring:
    """
    Monitor database queries for performance issues
    Tracks execution times and logs slow queries
    """
    
    def __init__(self):
        self.slow_threshold_ms = getattr(settings, 'SLOW_QUERY_THRESHOLD_MS', 500)
        self.enabled = getattr(settings, 'METRICS_ENABLED', True)
        self._query_context: Dict[int, Dict] = {}
        self._current_endpoint: Optional[str] = None
    
    def set_current_endpoint(self, endpoint: str):
        """Set the current endpoint context for query tracking"""
        self._current_endpoint = endpoint
    
    def clear_current_endpoint(self):
        """Clear the current endpoint context"""
        self._current_endpoint = None
    
    def setup_event_listeners(self):
        """Setup SQLAlchemy event listeners"""
        if not self.enabled:
            return
        
        @event.listens_for(Engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Record query start time"""
            conn.info.setdefault('query_start_time', []).append(time.time())
            conn.info.setdefault('query_statement', []).append(statement)
        
        @event.listens_for(Engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Record query end time and log if slow"""
            if not conn.info.get('query_start_time'):
                return
            
            start_time = conn.info['query_start_time'].pop()
            duration_ms = (time.time() - start_time) * 1000
            
            # Skip health check queries
            if self._is_health_check_query(statement):
                return
            
            # Record slow queries
            if duration_ms > self.slow_threshold_ms:
                query_text = self._sanitize_query(statement)
                endpoint = self._current_endpoint or 'unknown'
                
                logger.warning(
                    "Slow database query detected",
                    query=query_text[:200],  # Limit length
                    duration_ms=round(duration_ms, 2),
                    endpoint=endpoint,
                    threshold_ms=self.slow_threshold_ms
                )
                
                # Record in metrics service
                metrics_service.record_slow_query(
                    query=query_text,
                    duration_ms=duration_ms,
                    endpoint=endpoint
                )
            
            # Log very slow queries (>5s) as errors
            if duration_ms > 5000:
                logger.error(
                    "Very slow database query",
                    query=self._sanitize_query(statement)[:200],
                    duration_ms=round(duration_ms, 2),
                    endpoint=endpoint
                )
    
    def _is_health_check_query(self, statement: str) -> bool:
        """Check if query is a health check query"""
        health_check_patterns = [
            r'SELECT\s+1',
            r'SELECT\s+CAST\(SERVERPROPERTY',
            r'SELECT\s+schema_name\(\)',
            r'SELECT\s+CAST\(',
            r'SELECT\s+1\s+FROM\s+fn_listextendedproperty',
            r'SELECT.*INFORMATION_SCHEMA.*TABLES'
        ]
        
        statement_upper = statement.upper().strip()
        for pattern in health_check_patterns:
            if re.match(pattern, statement_upper, re.IGNORECASE):
                return True
        return False
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitize query for logging (remove sensitive data)"""
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Truncate very long queries
        if len(query) > 500:
            query = query[:500] + '...'
        
        return query


# Global instance
query_monitor = QueryMonitoring()

# Setup event listeners on import
query_monitor.setup_event_listeners()

