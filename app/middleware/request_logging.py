"""
Request/Response Logging Middleware
Logs all API requests with timing, request IDs, and integrates with activity logs
"""
import time
import uuid
from typing import Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.config import settings
from app.utils.logger import logger
from app.services.activity_service import ActivityService
from app.services.auth_service import AuthService
from app.services.metrics_service import metrics_service
from app.middleware.query_monitoring import query_monitor


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all API requests with:
    - Unique request ID
    - Request timing
    - Slow request detection
    - Activity log integration for authenticated requests
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.skip_paths = getattr(settings, 'REQUEST_LOGGING_SKIP_PATHS', [
            "/health",
            "/api/health",
            "/docs",
            "/redoc",
            "/openapi.json"
        ])
        self.slow_threshold_ms = getattr(settings, 'REQUEST_LOGGING_SLOW_THRESHOLD_MS', 1000)
        self.log_to_activity = getattr(settings, 'REQUEST_LOGGING_LOG_TO_ACTIVITY', True)
        self.enabled = getattr(settings, 'REQUEST_LOGGING_ENABLED', True)
    
    def _should_skip(self, path: str) -> bool:
        """Check if path should be skipped from logging"""
        if not self.enabled:
            return True
        return any(path.startswith(skip_path) for skip_path in self.skip_paths)
    
    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from JWT token if available"""
        try:
            auth_header = request.headers.get("authorization", "")
            if not auth_header.startswith("Bearer "):
                return None
            
            token = auth_header.replace("Bearer ", "")
            payload = AuthService.verify_token(token)
            user_id = payload.get("sub")
            return str(user_id) if user_id else None
        except Exception:
            # Token invalid or not present - that's okay
            return None
    
    async def dispatch(self, request: Request, call_next):
        """Process request and log details"""
        # Skip logging for health checks and docs
        if self._should_skip(request.url.path):
            return await call_next(request)
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Extract user ID if authenticated
        user_id = self._extract_user_id(request)
        
        # Start timing
        start_time = time.time()
        
        # Extract request details
        method = request.method
        path = request.url.path
        query_params = str(request.url.query) if request.url.query else None
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Set query monitoring context
        endpoint_key = f"{method} {path}"
        query_monitor.set_current_endpoint(endpoint_key)
        
        # Log request start
        logger.info(
            "API request started",
            request_id=request_id,
            method=method,
            path=path,
            user_id=user_id,
            client_ip=client_ip
        )
        
        # Process request
        status_code = 500
        error_occurred = False
        error_message = None
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Calculate response time
            duration_ms = (time.time() - start_time) * 1000
            is_slow = duration_ms > self.slow_threshold_ms
            
            # Record metrics
            metrics_service.record_response_time(
                endpoint=path,
                method=method,
                duration_ms=duration_ms,
                status_code=status_code
            )
            
            # Clear query monitoring context
            query_monitor.clear_current_endpoint()
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            # Log response with appropriate level
            if status_code >= 500:
                logger.error(
                    "API request completed",
                    request_id=request_id,
                    method=method,
                    path=path,
                    status_code=status_code,
                    duration_ms=round(duration_ms, 2),
                    is_slow=is_slow,
                    user_id=user_id,
                    client_ip=client_ip
                )
            elif status_code >= 400:
                logger.warning(
                    "API request completed",
                    request_id=request_id,
                    method=method,
                    path=path,
                    status_code=status_code,
                    duration_ms=round(duration_ms, 2),
                    is_slow=is_slow,
                    user_id=user_id,
                    client_ip=client_ip
                )
            elif is_slow:
                logger.warning(
                    "API request completed (slow)",
                    request_id=request_id,
                    method=method,
                    path=path,
                    status_code=status_code,
                    duration_ms=round(duration_ms, 2),
                    is_slow=is_slow,
                    user_id=user_id,
                    client_ip=client_ip
                )
            else:
                logger.info(
                    "API request completed",
                    request_id=request_id,
                    method=method,
                    path=path,
                    status_code=status_code,
                    duration_ms=round(duration_ms, 2),
                    is_slow=is_slow,
                    user_id=user_id,
                    client_ip=client_ip
                )
            
            # Log to activity log for authenticated requests
            if self.log_to_activity and user_id:
                try:
                    from uuid import UUID
                    user_uuid = UUID(user_id)
                    
                    # Build description
                    description = f"API Request: {method} {path}"
                    if query_params:
                        description += f"?{query_params[:50]}"  # Limit query string length
                    description += f" ({status_code}) - {round(duration_ms, 0)}ms"
                    
                    # Build metadata
                    metadata = {
                        "request_id": request_id,
                        "method": method,
                        "path": path,
                        "status_code": status_code,
                        "response_time_ms": round(duration_ms, 2),
                        "is_slow": is_slow,
                        "user_agent": user_agent,
                        "client_ip": client_ip
                    }
                    if query_params:
                        metadata["query_params"] = query_params[:200]  # Limit length
                    
                    # Log to activity (non-blocking)
                    await ActivityService.log_activity(
                        db=None,  # Use separate session
                        user_id=user_uuid,
                        action="API_REQUEST",
                        description=description,
                        metadata=metadata,
                        ip_address=client_ip
                    )
                except Exception as e:
                    # Don't fail request if activity logging fails
                    logger.warning(
                        "Failed to log API request to activity log",
                        request_id=request_id,
                        error=str(e)
                    )
            
            return response
            
        except Exception as e:
            # Calculate duration even on error
            duration_ms = (time.time() - start_time) * 1000
            error_occurred = True
            error_message = str(e)
            
            # Record metrics for failed request
            metrics_service.record_response_time(
                endpoint=path,
                method=method,
                duration_ms=duration_ms,
                status_code=500  # Error status
            )
            
            # Clear query monitoring context
            query_monitor.clear_current_endpoint()
            
            # Log error
            logger.error(
                "API request failed",
                request_id=request_id,
                method=method,
                path=path,
                status_code=status_code,
                duration_ms=round(duration_ms, 2),
                error=error_message,
                user_id=user_id,
                client_ip=client_ip,
                exc_info=True
            )
            
            # Re-raise exception to let error handler process it
            raise

