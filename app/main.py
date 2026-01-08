from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, Request, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.exceptions import HTTPException as StarletteHTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
from pathlib import Path
import uuid
import asyncio
import aiofiles
from contextlib import asynccontextmanager
from typing import Optional, List
from datetime import timedelta, datetime
from uuid import UUID

from app.config import settings
from app.database import init_db, get_db, AsyncSession, User
from app.services.video_processor import VideoProcessor
from app.services.document_generator import DocumentGenerator
from app.services.job_service import JobService
from app.services.auth_service import AuthService
from app.services.activity_service import ActivityService
from app.services.google_oauth_service import GoogleOAuthService
from app.services.video_upload_service import VideoUploadService
from app.services.video_metadata_service import VideoMetadataService
from app.services.frame_analysis_service import FrameAnalysisService
from app.services.cache_service import CacheService
from app.services.metrics_service import metrics_service
from app.services.system_monitor import system_monitor
from app.services.queue_worker_service import queue_worker
from app.models import (
    UserSignup, UserLogin, SignupResponse, LoginResponse, UserResponse,
    VideoUploadCreate, VideoUploadResponse, VideoUploadListResponse, VideoUploadUpdate, BulkDeleteRequest,
    FrameAnalysisResponse, FrameAnalysisListResponse,
    ActivityLogResponse, ActivityLogListResponse, ActivityLogStatsResponse,
    DocumentResponse, VideoMetadata, FrameData,
    VideoPanelItem, VideoPanelResponse
)
from app.models.gpt_response_schemas import GPTResponseListResponse, GPTResponseItem
from app.utils.logger import configure_logging, logger
from app.utils.validators import validate_file, validate_file_size
from app.middleware.error_handler import (
    validation_exception_handler,
    http_exception_handler,
    general_exception_handler
)
from app.middleware.request_logging import RequestLoggingMiddleware

# Security
security = HTTPBearer()

# Configure logging
configure_logging()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Directories
UPLOAD_DIR = settings.UPLOAD_DIR
OUTPUT_DIR = settings.OUTPUT_DIR
FRAMES_DIR = settings.FRAMES_DIR
AUDIO_DIR = settings.AUDIO_DIR
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
FRAMES_DIR.mkdir(exist_ok=True, parents=True)
AUDIO_DIR.mkdir(exist_ok=True, parents=True)

# Initialize services
video_processor = VideoProcessor()
document_generator = DocumentGenerator()
frame_analysis_service = FrameAnalysisService(max_workers=settings.FRAME_ANALYSIS_WORKERS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting application", version=settings.API_VERSION)
    await init_db()
    logger.info("Database initialized")
    
    # Start system monitoring
    from app.services.system_monitor import system_monitor
    if getattr(settings, 'METRICS_ENABLED', True):
        system_monitor.start_background_monitoring()
    
    # Start queue worker
    queue_worker.start()
    logger.info("Queue worker started")
    
    yield
    # Shutdown
    logger.info("Shutting down application")
    queue_worker.stop()


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Request logging middleware (should be first to capture all requests)
app.add_middleware(RequestLoggingMiddleware)

# Compression middleware (should be added before CORS)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Error handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Video Processing API",
        "version": settings.API_VERSION,
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "database": "connected"
    }


@app.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint
    Returns metrics in Prometheus format
    """
    if not getattr(settings, 'METRICS_ENABLED', True):
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(content="# Metrics disabled\n", media_type="text/plain")
    
    from fastapi.responses import PlainTextResponse
    
    lines = []
    
    # HTTP Request Metrics
    request_counts = metrics_service.get_request_counts()
    for endpoint, counts in request_counts.items():
        # Parse method and path from endpoint key (format: "METHOD /path")
        parts = endpoint.split(' ', 1)
        method = parts[0] if len(parts) > 0 else 'GET'
        path = parts[1] if len(parts) > 1 else endpoint
        
        # Escape quotes in path for Prometheus
        path_escaped = path.replace('"', '\\"')
        
        for status_code, count in counts.items():
            lines.append(
                f'http_requests_total{{method="{method}",endpoint="{path_escaped}",status="{status_code}"}} {count}'
            )
    
    # Response Time Histogram (simplified - using percentiles)
    response_times = metrics_service.get_response_time_stats()
    for endpoint, stats in response_times.items():
        # Parse method and path from endpoint key
        parts = endpoint.split(' ', 1)
        method = parts[0] if len(parts) > 0 else 'GET'
        path = parts[1] if len(parts) > 1 else endpoint
        
        # Escape quotes in path for Prometheus
        path_escaped = path.replace('"', '\\"')
        
        # Convert to seconds for Prometheus
        for percentile, value in stats.items():
            if percentile.startswith('p'):
                lines.append(
                    f'http_request_duration_seconds{{method="{method}",endpoint="{path_escaped}",quantile="{percentile}"}} {value / 1000.0}'
                )
        
        # Average and max
        if 'avg' in stats:
            lines.append(
                f'http_request_duration_seconds_sum{{method="{method}",endpoint="{path_escaped}"}} {stats["avg"] * stats.get("count", 1) / 1000.0}'
            )
            lines.append(
                f'http_request_duration_seconds_count{{method="{method}",endpoint="{path_escaped}"}} {stats.get("count", 0)}'
            )
    
    # Cache Metrics
    cache_stats = metrics_service.get_cache_stats()
    for cache_type, stats in cache_stats.items():
        lines.append(f'cache_hits_total{{cache_type="{cache_type}"}} {stats["hits"]}')
        lines.append(f'cache_misses_total{{cache_type="{cache_type}"}} {stats["misses"]}')
        lines.append(f'cache_hit_rate{{cache_type="{cache_type}"}} {stats["hit_rate"]}')
    
    # System Resource Metrics
    current_metrics = system_monitor.get_current_metrics()
    if current_metrics:
        memory = current_metrics.get('memory', {})
        cpu = current_metrics.get('cpu', {})
        
        if memory:
            lines.append(f'system_memory_process_bytes {memory.get("process_mb", 0) * 1024 * 1024}')
            lines.append(f'system_memory_available_bytes {memory.get("system_available_gb", 0) * 1024 * 1024 * 1024}')
            lines.append(f'system_memory_used_percent {memory.get("system_used_percent", 0)}')
        
        if cpu:
            lines.append(f'system_cpu_process_percent {cpu.get("process_percent", 0)}')
            lines.append(f'system_cpu_system_percent {cpu.get("system_percent", 0)}')
    
    # Slow Queries Count
    slow_queries = metrics_service.get_slow_queries(limit=1)
    lines.append(f'slow_queries_total {len(slow_queries)}')
    
    # Add help and type comments
    prometheus_output = """# HELP http_requests_total Total number of HTTP requests
# TYPE http_requests_total counter
# HELP http_request_duration_seconds HTTP request duration in seconds
# TYPE http_request_duration_seconds histogram
# HELP cache_hits_total Total number of cache hits
# TYPE cache_hits_total counter
# HELP cache_misses_total Total number of cache misses
# TYPE cache_misses_total counter
# HELP cache_hit_rate Cache hit rate (0-1)
# TYPE cache_hit_rate gauge
# HELP system_memory_process_bytes Process memory usage in bytes
# TYPE system_memory_process_bytes gauge
# HELP system_memory_available_bytes Available system memory in bytes
# TYPE system_memory_available_bytes gauge
# HELP system_memory_used_percent System memory used percentage
# TYPE system_memory_used_percent gauge
# HELP system_cpu_process_percent Process CPU usage percentage
# TYPE system_cpu_process_percent gauge
# HELP system_cpu_system_percent System CPU usage percentage
# TYPE system_cpu_system_percent gauge
# HELP slow_queries_total Total number of slow queries detected
# TYPE slow_queries_total gauge
""" + "\n".join(lines)
    
    return PlainTextResponse(content=prometheus_output, media_type="text/plain")


@app.get("/api/health")
async def api_health(db: AsyncSession = Depends(get_db)):
    """Detailed health check with actual service tests"""
    import time
    import shutil
    from datetime import datetime, timezone
    from sqlalchemy import text, select, func
    from app.database import JobStatus
    
    overall_status = "healthy"
    services = {}
    
    # Test Database Connectivity
    db_status = "operational"
    db_response_time_ms = 0
    db_error = None
    try:
        start_time = time.time()
        result = await db.execute(text("SELECT 1"))
        result.scalar()  # Consume result
        db_response_time_ms = (time.time() - start_time) * 1000
        
        if db_response_time_ms > 5000:  # >5s is degraded
            db_status = "degraded"
            overall_status = "degraded"
    except Exception as e:
        db_status = "down"
        db_error = str(e)
        overall_status = "unhealthy"
        logger.error("Database health check failed", error=str(e))
    
    services["database"] = {
        "status": db_status,
        "response_time_ms": round(db_response_time_ms, 2),
        "error": db_error
    }
    
    # Check OpenAI API
    openai_status = "not_configured"
    openai_error = None
    if settings.OPENAI_API_KEY:
        openai_status = "configured"
        # Optionally test connectivity (lightweight check)
        try:
            # Just verify key format, don't make actual API call
            if len(settings.OPENAI_API_KEY) > 20 and settings.OPENAI_API_KEY.startswith("sk-"):
                openai_status = "configured"
            else:
                openai_status = "error"
                openai_error = "Invalid API key format"
        except Exception as e:
            openai_status = "error"
            openai_error = str(e)
    
    services["openai"] = {
        "status": openai_status,
        "error": openai_error
    }
    
    # Check Disk Space
    disk_status = "ok"
    available_gb = 0
    used_percent = 0
    disk_error = None
    try:
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(exist_ok=True, parents=True)
        
        # Get disk usage
        total, used, free = shutil.disk_usage(upload_dir)
        available_gb = free / (1024 ** 3)  # Convert to GB
        used_percent = (used / total) * 100
        free_percent = (free / total) * 100
        
        if free_percent < settings.HEALTH_CHECK_DISK_CRITICAL_THRESHOLD:
            disk_status = "critical"
            overall_status = "unhealthy"
        elif free_percent < settings.HEALTH_CHECK_DISK_WARNING_THRESHOLD:
            disk_status = "warning"
            if overall_status == "healthy":
                overall_status = "degraded"
    except Exception as e:
        disk_status = "error"
        disk_error = str(e)
        logger.error("Disk space check failed", error=str(e))
    
    services["disk_space"] = {
        "status": disk_status,
        "available_gb": round(available_gb, 2),
        "used_percent": round(used_percent, 2),
        "free_percent": round(100 - used_percent, 2),
        "threshold_warning": settings.HEALTH_CHECK_DISK_WARNING_THRESHOLD,
        "threshold_critical": settings.HEALTH_CHECK_DISK_CRITICAL_THRESHOLD,
        "error": disk_error
    }
    
    # Check Background Jobs
    jobs_status = "operational"
    active_jobs = 0
    failed_jobs = 0
    pending_jobs = 0
    jobs_error = None
    try:
        # Count jobs by status
        active_query = select(func.count(JobStatus.job_id)).where(
            JobStatus.status.in_(["processing", "running"])
        )
        active_result = await db.execute(active_query)
        active_jobs = active_result.scalar() or 0
        
        failed_query = select(func.count(JobStatus.job_id)).where(
            JobStatus.status == "failed"
        )
        failed_result = await db.execute(failed_query)
        failed_jobs = failed_result.scalar() or 0
        
        pending_query = select(func.count(JobStatus.job_id)).where(
            JobStatus.status.in_(["pending", "queued"])
        )
        pending_result = await db.execute(pending_query)
        pending_jobs = pending_result.scalar() or 0
        
        # If too many failed jobs, mark as degraded
        if failed_jobs > 10:
            jobs_status = "degraded"
            if overall_status == "healthy":
                overall_status = "degraded"
    except Exception as e:
        jobs_status = "error"
        jobs_error = str(e)
        logger.error("Background jobs check failed", error=str(e))
    
    services["background_jobs"] = {
        "status": jobs_status,
        "active": active_jobs,
        "failed": failed_jobs,
        "pending": pending_jobs,
        "error": jobs_error
    }
    
    return {
        "status": overall_status,
        "version": settings.API_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": services
    }


@app.get("/api/metrics")
async def get_performance_metrics(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get performance metrics in JSON format for dashboard
    Requires authentication
    """
    # Verify authentication
    try:
        payload = AuthService.verify_token(credentials.credentials)
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    if not getattr(settings, 'METRICS_ENABLED', True):
        raise HTTPException(status_code=404, detail="Metrics disabled")
    
    # Get response time statistics
    response_times = metrics_service.get_response_time_stats()
    
    # Get slow queries
    slow_queries = metrics_service.get_slow_queries(limit=100)
    
    # Get system resources
    system_resources = system_monitor.get_current_metrics()
    
    # Get cache statistics
    cache_stats = metrics_service.get_cache_stats()
    
    # Get top endpoints
    top_endpoints = metrics_service.get_top_endpoints(limit=10)
    
    # Get error rates
    error_rates = metrics_service.get_error_rates()
    
    # Import timezone here to avoid circular import issues
    from datetime import timezone
    
    return {
        "response_times": response_times,
        "slow_queries": slow_queries,
        "system_resources": {
            "memory": system_resources.get("memory", {}),
            "cpu": system_resources.get("cpu", {}),
            "disk_io": system_resources.get("disk_io", {}),
            "network_io": system_resources.get("network_io", {}),
            "process": system_resources.get("process", {})
        },
        "cache_stats": cache_stats,
        "top_endpoints": [
            {"endpoint": endpoint, "total_requests": count}
            for endpoint, count in top_endpoints
        ],
        "error_rates": error_rates,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# Authentication dependencies
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Get current authenticated user from JWT token"""
    token = credentials.credentials
    payload = AuthService.verify_token(token)
    
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    try:
        # Convert user_id to UUID - handle both string and UUID formats
        if isinstance(user_id, str):
            user_uuid = uuid.UUID(user_id)
        else:
            user_uuid = user_id
        
        user = await AuthService.get_user_by_id(db, user_uuid)
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        
        if not user.is_active:
            raise HTTPException(status_code=403, detail="User account is inactive")
        
        return user
    except (ValueError, TypeError) as e:
        logger.error("Invalid user ID format in token", user_id=user_id, error=str(e))
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except Exception as e:
        logger.error("Error getting current user", error=str(e), exc_info=True)
        raise HTTPException(status_code=401, detail="Failed to authenticate user")


def get_client_ip(request: Request) -> Optional[str]:
    """Get client IP address from request"""
    if request.client:
        return request.client.host
    return None


def get_user_agent(request: Request) -> Optional[str]:
    """Get user agent from request"""
    return request.headers.get("user-agent")


# Authentication endpoints
@app.post("/api/auth/signup", response_model=SignupResponse)
@limiter.limit("5/minute")
async def signup(
    request: Request,
    user_data: UserSignup,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user"""
    try:
        # Create user
        user = await AuthService.create_user(
            db=db,
            full_name=user_data.full_name,
            email=user_data.email,
            password=user_data.password
        )
        
        # Log activity (non-blocking - uses separate session, won't affect signup)
        await ActivityService.log_activity(
            db=None,  # Use separate session to avoid transaction conflicts
            user_id=user.id,
            action="SIGNUP",
            description=f"User {user.email} registered",
            ip_address=get_client_ip(request)
        )
        
        return SignupResponse(
            message="User registered successfully",
            user=UserResponse.model_validate(user)
        )
    except HTTPException:
        raise
    except Exception as e:
        error_detail = str(e)
        logger.error("Signup error", error=error_detail, exc_info=True)
        # Return more detailed error in debug mode
        if settings.DEBUG:
            raise HTTPException(status_code=500, detail=f"Failed to register user: {error_detail}")
        else:
            raise HTTPException(status_code=500, detail="Failed to register user")


@app.post("/api/auth/login", response_model=LoginResponse)
@limiter.limit("10/minute")
async def login(
    request: Request,
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """Login user and create session"""
    try:
        # Authenticate user
        user = await AuthService.authenticate_user(
            db=db,
            email=credentials.email,
            password=credentials.password
        )
        
        if not user:
            logger.warning("Login failed - incorrect credentials", email=credentials.email)
            raise HTTPException(status_code=401, detail="Incorrect email or password")
        
        logger.info("User authenticated successfully", user_id=str(user.id), email=user.email)
        
        # Update last login (non-blocking - if it fails, don't fail the login)
        try:
            await AuthService.update_last_login(db, user.id)
        except Exception as login_update_error:
            logger.warning("Failed to update last login", error=str(login_update_error), user_id=str(user.id))
        
        # Create access token
        try:
            access_token = AuthService.create_access_token(
                data={"sub": str(user.id), "email": user.email, "role": user.role},
                expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
            )
        except Exception as token_error:
            logger.error("Failed to create access token", error=str(token_error), user_id=str(user.id))
            raise HTTPException(status_code=500, detail="Failed to create access token")
        
        # Create session (non-blocking - if it fails, still allow login but log the error)
        session = None
        try:
            session = await AuthService.create_session(
                db=db,
                user_id=user.id,
                ip_address=get_client_ip(request),
                user_agent=get_user_agent(request)
            )
        except Exception as session_error:
            logger.error("Failed to create session", error=str(session_error), user_id=str(user.id), exc_info=True)
            # Create a temporary session token for response
            session_token = AuthService.generate_session_token()
            expires_at = datetime.utcnow() + timedelta(days=7)
            # Don't fail login if session creation fails
        
        # Log activity (non-blocking - uses separate session, won't affect login)
        # Pass None for db to use separate session
        await ActivityService.log_activity(
            db=None,  # Use separate session to avoid transaction conflicts
            user_id=user.id,
            action="LOGIN",
            description=f"User {user.email} logged in",
            ip_address=get_client_ip(request)
        )
        
        # Return response
        if session:
            return LoginResponse(
                access_token=access_token,
                session_token=session.session_token,
                user=UserResponse.model_validate(user),
                expires_at=session.expires_at
            )
        else:
            # Fallback if session creation failed
            return LoginResponse(
                access_token=access_token,
                session_token=AuthService.generate_session_token(),
                user=UserResponse.model_validate(user),
                expires_at=datetime.utcnow() + timedelta(days=7)
            )
    except HTTPException:
        raise
    except Exception as e:
        error_detail = str(e)
        logger.error("Login error", error=error_detail, email=credentials.email, exc_info=True)
        # Return more detailed error in debug mode
        if settings.DEBUG:
            raise HTTPException(status_code=500, detail=f"Failed to login: {error_detail}")
        else:
            raise HTTPException(status_code=500, detail="Failed to login")


@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user information"""
    # Create response but mask the API key for security
    user_dict = {
        "id": current_user.id,
        "full_name": current_user.full_name,
        "email": current_user.email,
        "role": current_user.role,
        "is_active": current_user.is_active,
        "last_login_at": current_user.last_login_at,
        "frame_analysis_prompt": current_user.frame_analysis_prompt,
        "openai_api_key": None,  # Never expose the actual API key
        "created_at": current_user.created_at,
        "updated_at": current_user.updated_at
    }
    return UserResponse.model_validate(user_dict)


# Google OAuth endpoints
@app.get("/api/auth/google")
@limiter.limit("10/minute")
async def google_oauth_start(
    request: Request,
    redirect_uri: Optional[str] = Query(None)
):
    """Initiate Google OAuth flow - redirects to Google"""
    try:
        # Store redirect_uri in state if provided (for frontend callback)
        state = None
        if redirect_uri:
            import base64
            state = base64.urlsafe_b64encode(redirect_uri.encode()).decode()
        
        auth_url, state_token = GoogleOAuthService.get_authorization_url(state)
        
        # If redirect_uri was provided, combine it with state_token
        if redirect_uri:
            # Store the state_token with the redirect_uri
            # In production, you might want to use a session or cache for this
            return RedirectResponse(url=auth_url)
        
        return RedirectResponse(url=auth_url)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Google OAuth start error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to initiate Google OAuth")


@app.get("/api/auth/google/callback")
async def google_oauth_callback(
    request: Request,
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Handle Google OAuth callback"""
    try:
        # Check for errors from Google
        if error:
            logger.error("Google OAuth error", error=error)
            # Redirect to frontend with error
            frontend_url = settings.CORS_ORIGINS[0] if settings.CORS_ORIGINS else "http://localhost:3000"
            return RedirectResponse(
                url=f"{frontend_url}/auth?error=oauth_failed&message={error}"
            )
        
        if not code:
            raise HTTPException(status_code=400, detail="Authorization code not provided")
        
        # Authenticate with Google
        user = await GoogleOAuthService.authenticate_with_google(db, code)
        
        if not user.is_active:
            raise HTTPException(status_code=403, detail="User account is inactive")
        
        # Update last login
        await AuthService.update_last_login(db, user.id)
        
        # Create access token
        access_token = AuthService.create_access_token(
            data={"sub": str(user.id), "email": user.email, "role": user.role},
            expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        # Create session
        session = await AuthService.create_session(
            db=db,
            user_id=user.id,
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request)
        )
        
        # Log activity
        await ActivityService.log_activity(
            db=db,
            user_id=user.id,
            action="LOGIN_GOOGLE",
            description=f"User {user.email} logged in with Google",
            metadata={"provider": "google"},
            ip_address=get_client_ip(request)
        )
        
        # Determine redirect URL - use the frontend callback URL
        frontend_url = settings.CORS_ORIGINS[0] if settings.CORS_ORIGINS else "http://localhost:3000"
        
        # Redirect to frontend Google OAuth callback with tokens
        redirect_url = f"{frontend_url}/auth/google/callback?token={access_token}&session={session.session_token}"
        
        return RedirectResponse(url=redirect_url)
        
    except HTTPException as e:
        # Re-raise HTTP exceptions with their original status codes
        frontend_url = settings.CORS_ORIGINS[0] if settings.CORS_ORIGINS else "http://localhost:3000"
        error_message = e.detail if hasattr(e, 'detail') else "Authentication failed"
        return RedirectResponse(
            url=f"{frontend_url}/auth?error=oauth_failed&message={error_message}"
        )
    except Exception as e:
        logger.error("Google OAuth callback error", 
                    error=str(e), 
                    error_type=type(e).__name__,
                    exc_info=True)
        frontend_url = settings.CORS_ORIGINS[0] if settings.CORS_ORIGINS else "http://localhost:3000"
        error_message = "Authentication failed"
        if "exchange" in str(e).lower() or "token" in str(e).lower():
            error_message = "Failed to exchange authorization code"
        elif "user" in str(e).lower() or "create" in str(e).lower():
            error_message = "Failed to create or retrieve user account"
        return RedirectResponse(
            url=f"{frontend_url}/auth?error=oauth_failed&message={error_message}"
        )


@app.post("/api/auth/google/token")
@limiter.limit("10/minute")
async def google_oauth_token_exchange(
    request: Request,
    code: str = Query(...),
    db: AsyncSession = Depends(get_db)
):
    """Exchange Google OAuth code for tokens (alternative to callback redirect)"""
    try:
        # Authenticate with Google
        user = await GoogleOAuthService.authenticate_with_google(db, code)
        
        if not user.is_active:
            raise HTTPException(status_code=403, detail="User account is inactive")
        
        # Update last login
        await AuthService.update_last_login(db, user.id)
        
        # Create access token
        access_token = AuthService.create_access_token(
            data={"sub": str(user.id), "email": user.email, "role": user.role},
            expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        # Create session
        session = await AuthService.create_session(
            db=db,
            user_id=user.id,
            ip_address=get_client_ip(request),
            user_agent=get_user_agent(request)
        )
        
        # Log activity
        await ActivityService.log_activity(
            db=db,
            user_id=user.id,
            action="LOGIN_GOOGLE",
            description=f"User {user.email} logged in with Google",
            metadata={"provider": "google"},
            ip_address=get_client_ip(request)
        )
        
        # Create user response without exposing sensitive data
        user_dict = {
            "id": user.id,
            "full_name": user.full_name,
            "email": user.email,
            "role": user.role,
            "is_active": user.is_active,
            "last_login_at": user.last_login_at,
            "frame_analysis_prompt": user.frame_analysis_prompt,
            "openai_api_key": None,  # Never expose the actual API key
            "created_at": user.created_at,
            "updated_at": user.updated_at
        }
        
        return LoginResponse(
            access_token=access_token,
            session_token=session.session_token,
            user=UserResponse.model_validate(user_dict),
            expires_at=session.expires_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Google OAuth token exchange error", 
                    error=str(e), 
                    error_type=type(e).__name__,
                    exc_info=True)
        # Provide more specific error message if possible
        error_detail = "Failed to authenticate with Google"
        if "exchange" in str(e).lower() or "token" in str(e).lower():
            error_detail = "Failed to exchange authorization code with Google"
        elif "user" in str(e).lower() or "create" in str(e).lower():
            error_detail = "Failed to create or retrieve user account"
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/api/upload", response_model=VideoUploadResponse)
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def upload_video(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: Optional[str] = None,
    application_name: Optional[str] = None,
    tags: Optional[str] = None,  # Comma-separated string or JSON array
    language_code: Optional[str] = None,
    priority: Optional[str] = "normal",
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Upload video file and start processing"""
    try:
        # IMPORTANT: Check OpenAI API key BEFORE accepting file upload
        # This prevents unnecessary file storage if key is missing
        await db.refresh(current_user)
        
        # Check user's encrypted API key
        has_user_key = False
        if current_user.openai_api_key:
            try:
                from app.utils.encryption import EncryptionService
                decrypted_key = EncryptionService.decrypt(current_user.openai_api_key)
                has_user_key = decrypted_key is not None and decrypted_key.strip() != ""
            except Exception as e:
                logger.warning("Failed to decrypt user API key", user_id=str(current_user.id), error=str(e))
        
        # Check system key
        has_system_key = settings.OPENAI_API_KEY is not None and settings.OPENAI_API_KEY.strip() != ""
        
        if not has_user_key and not has_system_key:
            raise HTTPException(
                status_code=400,
                detail="OpenAI API key is required to process videos. Please add your API key in Settings before uploading videos."
            )
        
        # Validate file (only after key check passes)
        validate_file(file)
        await validate_file_size(file)
        
        # Generate unique ID for this processing job
        job_id = str(uuid.uuid4())
        
        # Use provided name or default to filename
        video_name = name or file.filename or "Untitled Video"
        
        # Parse tags if provided (comma-separated string or JSON array)
        tags_list = None
        if tags:
            try:
                # Try parsing as JSON array first
                import json
                tags_list = json.loads(tags)
                if not isinstance(tags_list, list):
                    tags_list = [t.strip() for t in tags.split(',')]
            except (json.JSONDecodeError, ValueError):
                # If not JSON, treat as comma-separated string
                tags_list = [t.strip() for t in tags.split(',')]
        
        # Save uploaded file using streaming (much faster for large files)
        file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
        file_size_bytes = 0
        async with aiofiles.open(file_path, "wb") as f:
            # Stream file in chunks instead of loading entire file into memory
            chunk_size = 1024 * 1024  # 1MB chunks
            while chunk := await file.read(chunk_size):
                await f.write(chunk)
                file_size_bytes += len(chunk)
        
        file_size_mb = file_size_bytes / (1024 * 1024)
        logger.info("File uploaded", job_id=job_id, filename=file.filename, size_mb=round(file_size_mb, 2))
        
        # Create minimal metadata (just file size for now, extract full metadata in background)
        # Get mime type from extension
        from pathlib import Path
        extension = Path(file_path).suffix.lower()
        mime_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
            '.flv': 'video/x-flv',
            '.wmv': 'video/x-ms-wmv',
            '.m4v': 'video/x-m4v'
        }
        mime_type = mime_types.get(extension, 'video/unknown')
        
        minimal_metadata = {
            "video_size_bytes": file_size_bytes,
            "mime_type": mime_type,
            "video_length_seconds": None,
            "resolution_width": None,
            "resolution_height": None,
            "fps": None
        }
        
        # Create video upload record with minimal metadata
        video_upload = await VideoUploadService.create_upload(
            db=db,
            user_id=current_user.id,
            name=video_name,
            source_type="upload",
            video_url=str(file_path),
            original_input=file.filename or "unknown",
            status="uploaded",
            job_id=job_id,
            metadata=minimal_metadata,
            application_name=application_name,
            tags=tags_list,
            language_code=language_code,
            priority=priority or "normal"
        )
        
        # Keep status as "uploaded" - queue worker will pick it up
        # Don't start processing immediately - let queue worker handle it
        logger.info("Video uploaded and added to queue", 
                   upload_id=str(video_upload.id),
                   video_file_number=video_upload.video_file_number,
                   job_id=job_id)
        
        # Log activity in background (non-blocking)
        background_tasks.add_task(
            log_upload_activity,
            current_user.id,
            video_name,
            str(video_upload.id),
            job_id,
            file.filename,
            file_size_bytes,
            get_client_ip(request)
        )
        
        # Extract full metadata in background and update record
        background_tasks.add_task(
            extract_and_update_metadata,
            str(file_path),
            video_upload.id
        )
        
        # Invalidate user's video panel cache since a new video was added
        CacheService.invalidate_user_cache(current_user.id)
        
        return VideoUploadResponse.model_validate(video_upload)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload error", error=str(e), exc_info=True)
        error_detail = str(e) if settings.DEBUG else "Failed to upload video"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/api/queue/status")
async def get_queue_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get queue status and statistics
    
    Returns:
        Dictionary with queue statistics (queue size, processing count, etc.)
    """
    stats = await queue_worker.get_queue_stats(db)
    return stats


@app.get("/api/status/{job_id}")
async def get_status(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get processing status"""
    job_data = await JobService.get_job_dict(db, job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JSONResponse(job_data)


@app.get("/api/download/{job_id}")
async def download_document(
    job_id: str,
    format: str = "docx",
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Download generated document in specified format"""
    job = await JobService.get_job(db, job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed yet")
    
    # Validate format
    if format not in ["docx", "html"]:
        raise HTTPException(status_code=400, detail="Invalid format. Allowed: docx, html")
    
    # Get output file path
    output_file = OUTPUT_DIR / f"{job_id}.{format}"
    
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Log activity
    await ActivityService.log_activity(
        db=db,
        user_id=current_user.id,
        action="EXPORT_DOC",
        description=f"User exported document: {job_id}.{format}",
        metadata={"job_id": job_id, "format": format},
        ip_address=get_client_ip(request)
    )
    
    logger.info("Document downloaded", job_id=job_id, format=format)
    
    return FileResponse(
        path=str(output_file),
        filename=f"document_{job_id}.{format}",
        media_type="application/octet-stream"
    )


# Video Upload endpoints
@app.get("/api/uploads", response_model=VideoUploadListResponse)
async def get_user_uploads(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    include_deleted: bool = Query(False, description="Include soft-deleted uploads"),
    application_name: Optional[str] = Query(None, description="Filter by application name"),
    language_code: Optional[str] = Query(None, description="Filter by language code"),
    priority: Optional[str] = Query(None, description="Filter by priority (normal, high)"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get paginated list of user's video uploads with filtering"""
    # Parse tags if provided
    tags_list = None
    if tags:
        tags_list = [t.strip() for t in tags.split(',')]
    
    uploads, total = await VideoUploadService.get_user_uploads(
        db=db,
        user_id=current_user.id,
        page=page,
        page_size=page_size,
        status=status,
        include_deleted=include_deleted,
        application_name=application_name,
        language_code=language_code,
        priority=priority,
        tags=tags_list
    )
    
    return VideoUploadListResponse(
        uploads=[VideoUploadResponse.model_validate(upload) for upload in uploads],
        total=total,
        page=page,
        page_size=page_size
    )


@app.get("/api/uploads/{upload_id}", response_model=VideoUploadResponse)
async def get_upload(
    upload_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get specific video upload by ID"""
    upload = await VideoUploadService.get_upload(db, upload_id, current_user.id)
    
    if not upload:
        raise HTTPException(status_code=404, detail="Video upload not found")
    
    return VideoUploadResponse.model_validate(upload)


@app.get("/api/videos/panel", response_model=VideoPanelResponse)
async def get_videos_panel(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    application_name: Optional[str] = Query(None, description="Filter by application name"),
    language_code: Optional[str] = Query(None, description="Filter by language code"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    sort_by: str = Query("updated_at", description="Sort field: updated_at, created_at, name"),
    sort_order: str = Query("desc", description="Sort order: asc, desc"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all videos for the panel/list view
    
    Returns videos with frame analysis statistics, suitable for displaying
    in a table/list panel similar to document management interfaces.
    """
    # Generate cache key
    tags_list = [t.strip() for t in tags.split(',')] if tags else None
    cache_key = CacheService._generate_cache_key(
        prefix="video_panel",
        user_id=current_user.id,
        page=page,
        page_size=page_size,
        status=status,
        application_name=application_name,
        language_code=language_code,
        priority=priority,
        tags=tags_list,
        sort_by=sort_by,
        sort_order=sort_order
    )
    
    # Try to get from cache
    cached_response = CacheService.get(cache_key, "video_panel")
    if cached_response is not None:
        return cached_response
    
    # Parse tags if provided
    if tags:
        tags_list = [t.strip() for t in tags.split(',')]
    
    # Validate sort parameters
    if sort_by not in ["updated_at", "created_at", "name", "status"]:
        sort_by = "updated_at"
    if sort_order not in ["asc", "desc"]:
        sort_order = "desc"
    
    # Get videos with stats
    videos_data, total = await VideoUploadService.get_user_uploads_with_stats(
        db=db,
        user_id=current_user.id,
        page=page,
        page_size=page_size,
        status=status,
        include_deleted=False,  # Don't show deleted videos in panel
        application_name=application_name,
        language_code=language_code,
        priority=priority,
        tags=tags_list,
        sort_by=sort_by,
        sort_order=sort_order
    )
    
    # Convert to panel items
    videos = [
        VideoPanelItem(
            id=video['id'],
            video_file_number=video['video_file_number'],
            name=video['name'],
            status=video['status'],
            created_at=video['created_at'],
            updated_at=video['updated_at'],
            last_activity=video['last_activity'],
            video_length_seconds=video['video_length_seconds'],
            video_size_bytes=video['video_size_bytes'],
            application_name=video['application_name'],
            tags=video['tags'],
            language_code=video['language_code'],
            priority=video['priority'],
            total_frames=video['total_frames'],
            frames_with_gpt=video['frames_with_gpt']
        )
        for video in videos_data
    ]
    
    response = VideoPanelResponse(
        videos=videos,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total
    )
    
    # Cache the response
    CacheService.set(cache_key, response, "video_panel")
    
    return response


@app.patch("/api/uploads/{upload_id}", response_model=VideoUploadResponse)
async def update_upload_metadata(
    request: Request,
    upload_id: UUID,
    update_data: VideoUploadUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update video upload metadata"""
    upload = await VideoUploadService.get_upload(db, upload_id, current_user.id)
    
    if not upload:
        raise HTTPException(status_code=404, detail="Video upload not found")
    
    if upload.is_deleted:
        raise HTTPException(status_code=400, detail="Cannot update deleted video upload")
    
    # Prepare updates
    updates = {}
    if update_data.name is not None:
        updates["name"] = update_data.name
    if update_data.status is not None:
        updates["status"] = update_data.status
    if update_data.application_name is not None:
        updates["application_name"] = update_data.application_name
    if update_data.tags is not None:
        updates["tags"] = update_data.tags
    if update_data.language_code is not None:
        updates["language_code"] = update_data.language_code
    if update_data.priority is not None:
        updates["priority"] = update_data.priority
    
    updated_upload = await VideoUploadService.update_upload(db, upload_id, updates, current_user.id)
    
    if not updated_upload:
        raise HTTPException(status_code=404, detail="Video upload not found")
    
    # Invalidate cache for this video and user's video panel
    CacheService.invalidate_video_cache(video_id=upload_id, video_file_number=updated_upload.video_file_number)
    CacheService.invalidate_user_cache(current_user.id)
    
    # Log activity
    await ActivityService.log_activity(
        db=db,
        user_id=current_user.id,
        action="UPDATE_VIDEO_METADATA",
        description=f"User updated video upload metadata: {upload_id}",
        metadata={"upload_id": str(upload_id), "updates": updates},
        ip_address=get_client_ip(request)
    )
    
    return VideoUploadResponse.model_validate(updated_upload)


@app.delete("/api/uploads/{upload_id}")
async def delete_upload(
    request: Request,
    upload_id: UUID,
    permanent: bool = Query(False, description="Permanently delete (hard delete)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a video upload (soft delete by default)"""
    if permanent:
        success = await VideoUploadService.hard_delete_upload(db, upload_id, current_user.id)
        action = "HARD_DELETE_VIDEO"
        message = "Video upload permanently deleted"
    else:
        success = await VideoUploadService.soft_delete_upload(db, upload_id, current_user.id)
        action = "DELETE_VIDEO"
        message = "Video upload deleted successfully"
    
    if not success:
        raise HTTPException(status_code=404, detail="Video upload not found")
    
    # Get video info before deletion for cache invalidation
    upload = await VideoUploadService.get_upload(db, upload_id, current_user.id)
    video_file_number = upload.video_file_number if upload else None
    
    # Invalidate cache for this video and user's video panel
    CacheService.invalidate_video_cache(video_id=upload_id, video_file_number=video_file_number)
    CacheService.invalidate_user_cache(current_user.id)
    
    # Log activity
    await ActivityService.log_activity(
        db=db,
        user_id=current_user.id,
        action=action,
        description=f"User deleted video upload: {upload_id}",
        metadata={"upload_id": str(upload_id), "permanent": permanent},
        ip_address=get_client_ip(request)
    )
    
    return {"message": message}


@app.post("/api/uploads/bulk-delete")
async def bulk_delete_uploads(
    request: Request,
    delete_request: BulkDeleteRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Bulk delete multiple video uploads"""
    if not delete_request.upload_ids:
        raise HTTPException(status_code=400, detail="No upload IDs provided")
    
    # Convert string IDs to UUIDs
    try:
        upload_uuids = [UUID(uid) for uid in delete_request.upload_ids]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid upload ID format: {str(e)}")
    
    deleted_count, failed_count = await VideoUploadService.bulk_delete_uploads(
        db=db,
        upload_ids=upload_uuids,
        user_id=current_user.id,
        permanent=delete_request.permanent
    )
    
    # Invalidate cache for all deleted videos and user's video panel
    for upload_id in upload_uuids:
        CacheService.invalidate_video_cache(video_id=upload_id)
    CacheService.invalidate_user_cache(current_user.id)
    
    # Log activity
    action = "BULK_HARD_DELETE_VIDEO" if delete_request.permanent else "BULK_DELETE_VIDEO"
    await ActivityService.log_activity(
        db=db,
        user_id=current_user.id,
        action=action,
        description=f"User bulk deleted {deleted_count} video upload(s)",
        metadata={
            "upload_ids": delete_request.upload_ids,
            "deleted_count": deleted_count,
            "failed_count": failed_count,
            "permanent": delete_request.permanent
        },
        ip_address=get_client_ip(request)
    )
    
    message = f"Successfully deleted {deleted_count} upload(s)"
    if failed_count > 0:
        message += f", {failed_count} failed"
    
    return {
        "message": message,
        "deleted_count": deleted_count,
        "failed_count": failed_count
    }


@app.post("/api/uploads/{upload_id}/restore", response_model=VideoUploadResponse)
async def restore_upload(
    request: Request,
    upload_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Restore a soft-deleted video upload"""
    success = await VideoUploadService.restore_upload(db, upload_id, current_user.id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Video upload not found")
    
    upload = await VideoUploadService.get_upload(db, upload_id, current_user.id)
    
    # Log activity
    await ActivityService.log_activity(
        db=db,
        user_id=current_user.id,
        action="RESTORE_VIDEO",
        description=f"User restored video upload: {upload_id}",
        metadata={"upload_id": str(upload_id)},
        ip_address=get_client_ip(request)
    )
    
    return VideoUploadResponse.model_validate(upload)


@app.post("/api/uploads/{upload_id}/retry", response_model=VideoUploadResponse)
async def retry_upload(
    request: Request,
    upload_id: UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Retry processing for a failed video upload"""
    upload = await VideoUploadService.get_upload(db, upload_id, current_user.id)
    
    if not upload:
        raise HTTPException(status_code=404, detail="Video upload not found")
    
    if upload.status != "failed":
        raise HTTPException(status_code=400, detail="Can only retry failed uploads")
    
    # Verify video file exists
    video_path = Path(upload.video_url)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    # Generate new job ID
    import uuid
    new_job_id = str(uuid.uuid4())
    
    # Initialize job status
    initial_status = {
        "status": "processing",
        "progress": 0,
        "message": "Retrying video processing...",
        "current_step": "upload",
        "step_progress": {
            "upload": "completed",
            "extract_audio": "pending",
            "transcribe": "pending",
            "extract_frames": "pending",
            "analyze_frames": "pending",
            "complete": "pending"
        }
    }
    
    await JobService.create_job(db, new_job_id, initial_status)
    
    # Update upload status to processing
    await VideoUploadService.update_upload_status(db, upload_id, "processing", new_job_id)
    
    # Log activity
    await ActivityService.log_activity(
        db=db,
        user_id=current_user.id,
        action="RETRY_VIDEO_PROCESSING",
        description=f"User retried video processing: {upload_id}",
        metadata={"upload_id": str(upload_id), "job_id": new_job_id},
        ip_address=get_client_ip(request)
    )
    
    # Start background processing
    background_tasks.add_task(process_video_task, str(video_path), new_job_id, str(upload_id))
    
    # Refresh upload to get updated job_id
    updated_upload = await VideoUploadService.get_upload(db, upload_id, current_user.id)
    
    return VideoUploadResponse.model_validate(updated_upload)


# Frame Analysis endpoints
@app.get("/api/videos/{video_id}/frames", response_model=FrameAnalysisListResponse)
async def get_video_frames(
    video_id: UUID,
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Maximum number of frames to return"),
    offset: int = Query(0, ge=0, description="Number of frames to skip"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get frame analyses for a video
    
    Returns JSON list of frame analyses with descriptions and OCR text
    """
    # Verify video belongs to user
    upload = await VideoUploadService.get_upload(db, video_id, current_user.id)
    if not upload:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Get frames
    frames = await frame_analysis_service.get_video_frames(
        db=db,
        video_id=video_id,
        limit=limit,
        offset=offset
    )
    
    # Get total count
    total = await frame_analysis_service.get_frame_count(db, video_id)
    
    return FrameAnalysisListResponse(
        frames=[FrameAnalysisResponse.model_validate(frame) for frame in frames],
        total=total,
        video_id=video_id,
        limit=limit,
        offset=offset
    )


async def process_video_frames_task(
    video_id: UUID,
    video_path: str
):
    """Background task to process video frames"""
    from app.database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as db:
        try:
            logger.info("Starting frame analysis task", video_id=str(video_id))
            
            # Process frames
            frame_analyses = await frame_analysis_service.process_video_frames(
                db=db,
                video_id=video_id,
                video_path=video_path,
                frames_dir=FRAMES_DIR,
                frames_per_second=settings.FRAMES_PER_SECOND
            )
            
            logger.info("Frame analysis task completed",
                       video_id=str(video_id),
                       frames_analyzed=len(frame_analyses))
            
        except Exception as e:
            logger.error("Frame analysis task failed",
                        video_id=str(video_id),
                        error=str(e),
                        exc_info=True)


# Activity Log endpoints
@app.get("/api/activity-logs", response_model=ActivityLogListResponse)
async def get_activity_logs(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Number of items per page"),
    action: Optional[str] = Query(None, description="Filter by action type"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format: YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format: YYYY-MM-DD)"),
    search: Optional[str] = Query(None, description="Search in descriptions"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get paginated activity logs for the current user with filtering
    
    Supports filtering by:
    - action: Filter by specific action type
    - start_date: Filter activities from this date onwards
    - end_date: Filter activities up to this date
    - search: Search in activity descriptions
    """
    # Parse dates if provided
    start_datetime = None
    end_datetime = None
    
    if start_date:
        try:
            # Try parsing as YYYY-MM-DD format
            if len(start_date) == 10:
                start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            else:
                # Try ISO format
                start_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD or ISO format")
    
    if end_date:
        try:
            # Try parsing as YYYY-MM-DD format
            if len(end_date) == 10:
                end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
                # Add one day to include the entire end date
                end_datetime = end_datetime + timedelta(days=1)
            else:
                # Try ISO format
                end_datetime = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                end_datetime = end_datetime + timedelta(days=1)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD or ISO format")
    
    logs, total = await ActivityService.get_user_activities_with_filters(
        db=db,
        user_id=current_user.id,
        page=page,
        page_size=page_size,
        action=action,
        start_date=start_datetime,
        end_date=end_datetime,
        search=search
    )
    
    response_data = ActivityLogListResponse(
        logs=[ActivityLogResponse(
            id=log.id,
            user_id=str(log.user_id),
            action=log.action,
            description=log.description,
            metadata=log.activity_metadata,
            ip_address=str(log.ip_address) if log.ip_address else None,
            created_at=log.created_at
        ) for log in logs],
        total=total,
        page=page,
        page_size=page_size
    )
    
    # Add cache headers for better performance
    # Use mode='json' to serialize datetime objects to ISO format strings
    from fastapi.responses import JSONResponse
    return JSONResponse(
        content=response_data.model_dump(mode='json'),
        headers={
            "Cache-Control": "private, max-age=60",  # Cache for 1 minute
            "X-Total-Count": str(total),
            "X-Page": str(page),
            "X-Page-Size": str(page_size)
        }
    )


# Note: Specific routes must be defined before parameterized routes
# to avoid route conflicts (e.g., /stats and /actions before /{log_id})
@app.get("/api/activity-logs/stats", response_model=ActivityLogStatsResponse)
async def get_activity_stats(
    days: int = Query(30, ge=1, le=365, description="Number of days to include in statistics"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get activity statistics for the current user"""
    # Generate cache key
    cache_key = CacheService._generate_cache_key(
        prefix="activity_stats",
        user_id=current_user.id,
        days=days
    )
    
    # Try to get from cache
    cached_response = CacheService.get(cache_key, "activity_stats")
    if cached_response is not None:
        return cached_response
    
    stats = await ActivityService.get_activity_stats(db, current_user.id, days=days)
    
    response = ActivityLogStatsResponse(
        total_activities=stats["total_activities"],
        activities_by_action=stats["activities_by_action"],
        recent_activities=[
            ActivityLogResponse(
                id=log.id,
                user_id=str(log.user_id),
                action=log.action,
                description=log.description,
                metadata=log.activity_metadata,
                ip_address=str(log.ip_address) if log.ip_address else None,
                created_at=log.created_at
            ) for log in stats["recent_activities"]
        ]
    )
    
    # Cache the response
    CacheService.set(cache_key, response, "activity_stats")
    
    return response


@app.get("/api/activity-logs/actions", response_model=List[str])
async def get_available_actions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get list of available action types for the current user"""
    actions = await ActivityService.get_available_actions(db, current_user.id)
    return sorted(actions)


@app.get("/api/activity-logs/{log_id}", response_model=ActivityLogResponse)
async def get_activity_log(
    log_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific activity log by ID"""
    log = await ActivityService.get_activity_by_id(db, log_id, current_user.id)
    
    if not log:
        raise HTTPException(status_code=404, detail="Activity log not found")
    
    return ActivityLogResponse(
        id=log.id,
        user_id=str(log.user_id),
        action=log.action,
        description=log.description,
        metadata=log.activity_metadata,
        ip_address=str(log.ip_address) if log.ip_address else None,
        created_at=log.created_at
    )


# User Settings endpoints
@app.get("/api/settings/prompt")
async def get_user_prompt(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get the current user's custom ANALYSIS RULES section"""
    from app.services.gpt_service import GPTService
    gpt_service = GPTService()
    
    # Refresh user from database to get latest prompt
    await db.refresh(current_user)
    
    # Get default ANALYSIS RULES for reference
    default_analysis_rules = gpt_service.get_default_analysis_rules()
    
    return {
        "analysis_rules": current_user.frame_analysis_prompt or default_analysis_rules,
        "has_custom_prompt": current_user.frame_analysis_prompt is not None,
        "default_analysis_rules": default_analysis_rules
    }


@app.put("/api/settings/prompt")
async def update_user_prompt(
    request: Request,
    prompt_data: dict,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update the current user's custom ANALYSIS RULES section"""
    analysis_rules = prompt_data.get("analysis_rules", "").strip()
    
    # If empty string or matches default, set to None to use default prompt
    from app.services.gpt_service import GPTService
    gpt_service = GPTService()
    default_analysis_rules = gpt_service.get_default_analysis_rules()
    
    if not analysis_rules or analysis_rules == default_analysis_rules:
        analysis_rules = None
    
    # Update user's ANALYSIS RULES
    current_user.frame_analysis_prompt = analysis_rules
    await db.commit()
    await db.refresh(current_user)
    
    # Log activity
    from app.services.activity_service import ActivityService
    await ActivityService.log_activity(
        db=db,
        user_id=current_user.id,
        action="UPDATE_PROMPT",
        description="Updated frame analysis ANALYSIS RULES",
        ip_address=get_client_ip(request)
    )
    
    return {
        "message": "Analysis rules updated successfully",
        "analysis_rules": current_user.frame_analysis_prompt or default_analysis_rules,
        "has_custom_prompt": current_user.frame_analysis_prompt is not None
    }


@app.get("/api/settings/prompt/default")
async def get_default_prompt(
    current_user: User = Depends(get_current_user)
):
    """Get the default prompt template and ANALYSIS RULES from prompt.txt file"""
    try:
        from app.services.gpt_service import GPTService
        gpt_service = GPTService()
        
        default_analysis_rules = gpt_service.get_default_analysis_rules()
        full_prompt = gpt_service.prompt_template
        
        return {
            "full_prompt": full_prompt,
            "analysis_rules": default_analysis_rules,
            "source": "prompt.txt"
        }
    except Exception as e:
        logger.error("Failed to load default prompt", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to load default prompt")


@app.get("/api/settings/openai-key")
async def get_user_openai_key(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get the current user's OpenAI API key (masked for security)"""
    await db.refresh(current_user)
    
    # Mask the API key for security (show only last 4 characters)
    masked_key = None
    if current_user.openai_api_key:
        key = current_user.openai_api_key
        if len(key) > 4:
            masked_key = "*" * (len(key) - 4) + key[-4:]
        else:
            masked_key = "*" * len(key)
    
    return {
        "has_key": current_user.openai_api_key is not None,
        "masked_key": masked_key,
        "key_length": len(current_user.openai_api_key) if current_user.openai_api_key else 0
    }


@app.get("/api/settings/openai-key/check")
async def check_openai_key_availability(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Check if OpenAI API key is available (user's key or system default)"""
    await db.refresh(current_user)
    
    # Check if user has a custom API key (and it's valid after decryption)
    has_user_key = False
    if current_user.openai_api_key:
        try:
            from app.utils.encryption import EncryptionService
            decrypted_key = EncryptionService.decrypt(current_user.openai_api_key)
            has_user_key = decrypted_key is not None and decrypted_key.strip() != ""
        except Exception as e:
            logger.warning("Failed to decrypt user API key during check", user_id=str(current_user.id), error=str(e))
            has_user_key = False
    
    # Check if system has a default API key
    has_system_key = settings.OPENAI_API_KEY is not None and settings.OPENAI_API_KEY.strip() != ""
    
    # Check if either key is available
    has_any_key = has_user_key or has_system_key
    
    return {
        "has_key": has_any_key,
        "has_user_key": has_user_key,
        "has_system_key": has_system_key,
        "message": "OpenAI API key is available" if has_any_key else "No OpenAI API key found. Please add your API key in Settings or contact administrator."
    }


@app.put("/api/settings/openai-key")
async def update_user_openai_key(
    request: Request,
    key_data: dict,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update the current user's OpenAI API key (encrypted in database)"""
    from app.utils.encryption import EncryptionService
    
    api_key = key_data.get("api_key", "").strip()
    
    # Validate OpenAI API key format (starts with sk- and is at least 20 characters)
    if api_key:
        if not api_key.startswith("sk-"):
            raise HTTPException(
                status_code=400, 
                detail="Invalid OpenAI API key format. OpenAI API keys should start with 'sk-'"
            )
        if len(api_key) < 20:
            raise HTTPException(
                status_code=400,
                detail="Invalid OpenAI API key format. API key appears to be too short."
            )
        
        # Encrypt the API key before storing
        encrypted_key = EncryptionService.encrypt(api_key)
        if not encrypted_key:
            raise HTTPException(
                status_code=500,
                detail="Failed to encrypt API key. Please try again."
            )
        api_key = encrypted_key
    else:
        # If empty string, set to None to use system default
        api_key = None
    
    # Update user's API key (encrypted)
    current_user.openai_api_key = api_key
    await db.commit()
    await db.refresh(current_user)
    
    # Log activity
    from app.services.activity_service import ActivityService
    await ActivityService.log_activity(
        db=db,
        user_id=current_user.id,
        action="UPDATE_OPENAI_KEY",
        description="Updated OpenAI API key" if api_key else "Removed OpenAI API key",
        ip_address=get_client_ip(request)
    )
    
    # Return masked key
    masked_key = None
    if current_user.openai_api_key:
        key = current_user.openai_api_key
        if len(key) > 4:
            masked_key = "*" * (len(key) - 4) + key[-4:]
        else:
            masked_key = "*" * len(key)
    
    return {
        "message": "OpenAI API key updated successfully" if api_key else "OpenAI API key removed. System default will be used.",
        "has_key": current_user.openai_api_key is not None,
        "masked_key": masked_key
    }


# GPT Response endpoints
@app.get("/api/videos/file-number/{video_file_number}/gpt-responses", response_model=GPTResponseListResponse)
async def get_gpt_responses_by_file_number(
    video_file_number: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all GPT responses for a video by video file number
    
    Returns all frame analyses with GPT responses for the specified video file number.
    """
    from app.services.video_file_number_service import VideoFileNumberService
    
    # Get video upload by file number
    upload = await VideoFileNumberService.get_upload_by_file_number(
        db, video_file_number, str(current_user.id)
    )
    
    if not upload:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Get all GPT responses for this video
    frame_analyses = await frame_analysis_service.get_gpt_responses_by_video_file_number(
        db=db,
        video_file_number=video_file_number,
        user_id=current_user.id
    )
    
    return GPTResponseListResponse(
        video_file_number=video_file_number,
        video_id=upload.id,
        video_name=upload.name,
        total_responses=len(frame_analyses),
        responses=[
            GPTResponseItem(
                frame_id=fa.id,
                timestamp=fa.timestamp,
                frame_number=fa.frame_number,
                image_path=fa.image_path,
                description=fa.description,
                ocr_text=fa.ocr_text,
                gpt_response=fa.gpt_response,
                processing_time_ms=fa.processing_time_ms,
                created_at=fa.created_at
            ) for fa in frame_analyses
        ]
    )


@app.get("/api/videos/file-number/{video_file_number}/audio")
async def get_audio_file(
    video_file_number: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get audio file for a video by file number"""
    from app.services.video_file_number_service import VideoFileNumberService
    
    upload = await VideoFileNumberService.get_upload_by_file_number(
        db, video_file_number, str(current_user.id)
    )
    
    if not upload:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if not upload.audio_url:
        raise HTTPException(status_code=404, detail="Audio file not available")
    
    audio_path = Path(upload.audio_url)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found on disk")
    
    return FileResponse(
        path=str(audio_path),
        filename=f"audio_{video_file_number}.mp3",
        media_type="audio/mpeg"
    )


@app.get("/api/videos/file-number/{video_file_number}/document", response_model=DocumentResponse)
async def get_document_by_file_number(
    request: Request,
    video_file_number: str,
    include_images: bool = Query(True, description="Include base64 images in response (can be slow for many frames)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Get complete document/data for a video file number
    
    Returns all frame analyses, GPT responses, video metadata, and summary statistics.
    This is the main endpoint to fetch all data for a video file once processing is complete.
    
    Args:
        include_images: If False, excludes base64_image from response (faster, smaller payload)
    """
    # Generate cache key (include include_images in key to cache separately)
    cache_key = CacheService._generate_cache_key(
        prefix="document_data",
        user_id=current_user.id,
        video_file_number=video_file_number,
        include_images=include_images
    )
    
    # Try to get from cache
    cached_response = CacheService.get(cache_key, "document_data")
    if cached_response is not None:
        return cached_response
    
    # Get complete document data (with option to exclude images for faster response)
    document_data = await frame_analysis_service.get_complete_document_data(
        db=db,
        video_file_number=video_file_number,
        user_id=current_user.id,
        include_base64_images=include_images
    )
    
    if not document_data:
        raise HTTPException(status_code=404, detail="Video not found")
    
    upload = document_data["video_metadata"]
    frames = document_data["frames"]
    
    # Get transcript from job status if job_id exists (optimize: only fetch transcript field)
    transcript = None
    if upload.job_id:
        from app.database import JobStatus
        from sqlalchemy import select
        job_query = select(JobStatus.transcript).where(JobStatus.job_id == upload.job_id)
        job_result = await db.execute(job_query)
        transcript = job_result.scalar_one_or_none()
    
    # Log activity in background (non-blocking) to reduce latency
    background_tasks.add_task(
        _log_document_fetch_activity,
        user_id=current_user.id,
        video_file_number=video_file_number,
        video_id=str(upload.id),
        total_frames=document_data["total_frames"],
        frames_with_gpt=document_data["frames_with_gpt"],
        ip_address=get_client_ip(request)
    )
    
    response = DocumentResponse(
        video_file_number=video_file_number,
        video_metadata=VideoMetadata(
            video_id=upload.id,
            video_file_number=upload.video_file_number,
            name=upload.name,
            status=upload.status,
            video_length_seconds=upload.video_length_seconds,
            video_size_bytes=upload.video_size_bytes,
            resolution_width=upload.resolution_width,
            resolution_height=upload.resolution_height,
            fps=upload.fps,
            application_name=upload.application_name,
            tags=upload.tags,
            language_code=upload.language_code,
            priority=upload.priority,
            audio_url=upload.audio_url,
            created_at=upload.created_at,
            updated_at=upload.updated_at
        ),
        total_frames=document_data["total_frames"],
        frames_with_gpt=document_data["frames_with_gpt"],
        frames=[
            FrameData(
                frame_id=fa.id,
                timestamp=fa.timestamp,
                frame_number=fa.frame_number,
                image_path=fa.image_path,
                base64_image=fa.base64_image,
                description=fa.description,
                ocr_text=fa.ocr_text,
                gpt_response=fa.gpt_response,
                processing_time_ms=fa.processing_time_ms,
                created_at=fa.created_at
            ) for fa in frames
        ],
        summary=document_data["summary"],
        transcript=transcript,
        created_at=datetime.utcnow()
    )
    
    # Cache the response (only if status is completed to avoid caching incomplete data)
    if upload.status == "completed":
        CacheService.set(cache_key, response, "document_data")
    
    return response


async def _log_document_fetch_activity(
    user_id: UUID,
    video_file_number: str,
    video_id: str,
    total_frames: int,
    frames_with_gpt: int,
    ip_address: Optional[str]
):
    """Background task to log document fetch activity without blocking response"""
    try:
        from app.database import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            await ActivityService.log_activity(
                db=session,
                user_id=user_id,
                action="FETCH_DOCUMENT",
                description=f"User fetched document for video: {video_file_number}",
                metadata={
                    "video_file_number": video_file_number,
                    "video_id": video_id,
                    "total_frames": total_frames,
                    "frames_with_gpt": frames_with_gpt
                },
                ip_address=ip_address
            )
            await session.commit()
    except Exception as e:
        logger.warning("Failed to log document fetch activity", error=str(e))


async def log_upload_activity(
    user_id: UUID,
    video_name: str,
    upload_id: str,
    job_id: str,
    filename: Optional[str],
    file_size_bytes: int,
    ip_address: Optional[str]
):
    """Background task to log upload activity"""
    from app.database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as db:
        try:
            await ActivityService.log_activity(
                db=db,
                user_id=user_id,
                action="UPLOAD_VIDEO",
                description=f"User uploaded video: {video_name}",
                metadata={
                    "upload_id": upload_id,
                    "job_id": job_id,
                    "filename": filename,
                    "size_bytes": file_size_bytes
                },
                ip_address=ip_address
            )
        except Exception as e:
            logger.error("Failed to log upload activity", 
                       upload_id=upload_id, 
                       error=str(e))


async def extract_and_update_metadata(video_path: str, upload_id: UUID):
    """Background task to extract full video metadata and update the record"""
    from app.database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as db:
        try:
            # Extract full metadata
            metadata = VideoMetadataService.extract_metadata(video_path)
            
            # Update video upload record with full metadata
            from sqlalchemy import update
            from app.database import VideoUpload
            
            await db.execute(
                update(VideoUpload)
                .where(VideoUpload.id == upload_id)
                .values(
                    video_length_seconds=metadata.get("video_length_seconds"),
                    video_size_bytes=metadata.get("video_size_bytes"),
                    mime_type=metadata.get("mime_type"),
                    resolution_width=metadata.get("resolution_width"),
                    resolution_height=metadata.get("resolution_height"),
                    fps=metadata.get("fps")
                )
            )
            await db.commit()
            
            logger.info("Video metadata updated", 
                       upload_id=str(upload_id),
                       metadata=metadata)
        except Exception as e:
            logger.error("Failed to extract and update metadata", 
                       upload_id=str(upload_id), 
                       error=str(e),
                       exc_info=True)


async def process_video_task(file_path: str, job_id: str, upload_id: Optional[str] = None):
    """
    Production-ready background task to process video:
    1. Extract audio and transcribe using OpenAI Whisper
    2. Extract keyframes (1 every 2 seconds)
    3. Process frames in batches of 5 through ChatGPT 4o Mini
    4. Store everything in database
    
    Each video gets isolated temporary directories to prevent conflicts.
    """
    from app.database import AsyncSessionLocal
    from app.services.video_processing_service import VideoProcessingService
    import tempfile
    import uuid
    
    logger.info("Background video processing task started", 
               job_id=job_id, 
               upload_id=upload_id,
               file_path=file_path)
    
    # Verify file exists
    if not Path(file_path).exists():
        logger.error("Video file not found", job_id=job_id, file_path=file_path)
        async with AsyncSessionLocal() as db:
            await JobService.update_job(db, job_id, {
                "status": "failed",
                "message": f"Video file not found: {file_path}",
                "error": "File not found"
            })
        return
    
    # Create isolated temporary directories for this specific job
    # This ensures no conflicts when multiple videos are processed
    job_temp_dir = None
    job_frames_dir = None
    job_audio_dir = None
    
    try:
        # Create unique temp directory for this job
        job_temp_dir = Path(tempfile.gettempdir()) / f"video_processing_{job_id}_{uuid.uuid4().hex[:8]}"
        job_frames_dir = job_temp_dir / "frames"
        job_audio_dir = job_temp_dir / "audio"
        
        job_frames_dir.mkdir(parents=True, exist_ok=True)
        job_audio_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Created isolated temp directories for job",
                   job_id=job_id,
                   temp_dir=str(job_temp_dir),
                   frames_dir=str(job_frames_dir),
                   audio_dir=str(job_audio_dir))
    except Exception as dir_error:
        logger.error("Failed to create isolated temp directories",
                    job_id=job_id,
                    error=str(dir_error))
        # Fallback to global directories if temp creation fails
        job_temp_dir = None  # Mark as None so cleanup won't try to remove it
        job_frames_dir = FRAMES_DIR
        job_audio_dir = AUDIO_DIR
    
    async with AsyncSessionLocal() as db:
        try:
            # Check if OpenAI API key is available before starting processing
            from app.database import User, VideoUpload
            from sqlalchemy import select
            from app.utils.encryption import EncryptionService
            
            # Get user_id from video upload
            user_id = None
            if upload_id:
                from uuid import UUID
                try:
                    video_uuid = UUID(upload_id)
                    result = await db.execute(
                        select(VideoUpload.user_id).where(VideoUpload.id == video_uuid)
                    )
                    user_id = result.scalar_one_or_none()
                except Exception as e:
                    logger.warning("Failed to get user_id from upload", upload_id=upload_id, error=str(e))
            
            # Check if user has API key
            has_user_key = False
            if user_id:
                try:
                    result = await db.execute(
                        select(User.openai_api_key).where(User.id == user_id)
                    )
                    encrypted_key = result.scalar_one_or_none()
                    if encrypted_key:
                        decrypted_key = EncryptionService.decrypt(encrypted_key)
                        has_user_key = decrypted_key is not None and decrypted_key.strip() != ""
                except Exception as e:
                    logger.warning("Failed to check user API key", user_id=str(user_id), error=str(e))
            
            # Check system key
            has_system_key = settings.OPENAI_API_KEY is not None and settings.OPENAI_API_KEY.strip() != ""
            
            if not has_user_key and not has_system_key:
                error_message = "OpenAI API key is required to process videos. Please add your API key in Settings."
                logger.error("No OpenAI API key available", job_id=job_id, upload_id=upload_id, user_id=str(user_id) if user_id else None)
                await JobService.update_job(db, job_id, {
                    "status": "failed",
                    "message": error_message,
                    "error": error_message
                })
                # Update upload status to failed
                if upload_id:
                    try:
                        from app.services.video_upload_service import VideoUploadService
                        await VideoUploadService.update_upload_status(db, video_uuid, "failed", job_id)
                    except Exception as e:
                        logger.error("Failed to update upload status", error=str(e))
                return
            
            # Immediately update job status to show processing has started
            logger.info("Updating job status to show processing started", job_id=job_id)
            await JobService.update_job(db, job_id, {
                "progress": 5,
                "message": "Video uploaded successfully. Starting processing...",
                "current_step": "extract_audio",
                "step_progress": {
                    "upload": "completed",
                    "extract_audio": "processing",
                    "transcribe": "pending",
                    "extract_frames": "pending",
                    "analyze_frames": "pending",
                    "summary_generation": "pending",
                    "generate_pdf": "pending",
                    "complete": "pending"
                }
            })
            logger.info("Job status updated successfully", job_id=job_id)
            
            # Initialize processing service
            processing_service = VideoProcessingService()
            logger.info("Video processing service initialized", job_id=job_id)
            
            # Convert upload_id to UUID if provided
            video_uuid = None
            if upload_id:
                from uuid import UUID
                try:
                    video_uuid = UUID(upload_id)
                    logger.info("Converted upload_id to UUID", upload_id=upload_id, video_uuid=str(video_uuid))
                except ValueError as e:
                    logger.error("Invalid upload_id format", upload_id=upload_id, error=str(e))
                    raise ValueError(f"Invalid upload_id format: {upload_id}")
            
            if not video_uuid:
                raise ValueError("Video upload ID is required for processing")
            
            # Run complete processing pipeline with isolated directories
            # Use job-specific temp directories instead of shared global directories
            result = await processing_service.process_video_complete(
                video_path=file_path,
                video_id=video_uuid,
                job_id=job_id,
                frames_dir=job_frames_dir,  # Use isolated directory
                audio_dir=job_audio_dir,    # Use isolated directory
                db=db
            )
            
            # Update video upload status to completed
            try:
                # Ensure we have a fresh database session for this update to avoid connection busy errors
                await db.commit()  # Commit any pending changes first
                
                updated_upload = await VideoUploadService.update_upload_status(db, video_uuid, "completed", job_id)
                if updated_upload:
                    logger.info("Video upload status updated to completed", 
                               upload_id=str(upload_id),
                               video_id=str(video_uuid),
                               status=updated_upload.status)
                    # Invalidate cache for this video and user's video panel
                    if user_id:
                        CacheService.invalidate_video_cache(
                            video_id=video_uuid, 
                            video_file_number=updated_upload.video_file_number if hasattr(updated_upload, 'video_file_number') else None
                        )
                        CacheService.invalidate_user_cache(user_id)
                else:
                    logger.warning("Video upload not found when updating status", 
                                  upload_id=str(upload_id),
                                  video_id=str(video_uuid))
                    # Try to update directly if the service method didn't work
                    from sqlalchemy import update
                    from app.database import VideoUpload
                    # Flush before update for SQL Server
                    await db.flush()
                    result = await db.execute(
                        update(VideoUpload)
                        .where(VideoUpload.id == video_uuid)
                        .values(status="completed")
                    )
                    await db.flush()
                    await db.commit()
                    logger.info("Video upload status updated directly to completed", 
                               upload_id=str(upload_id),
                               video_id=str(video_uuid),
                               rows_affected=result.rowcount)
            except Exception as e:
                logger.error("Failed to update video upload status", 
                           upload_id=upload_id, 
                           video_id=str(video_uuid),
                           error=str(e),
                           exc_info=True)
                # Try to update directly as fallback
                try:
                    from sqlalchemy import update
                    from app.database import VideoUpload
                    # Ensure we flush before update for SQL Server
                    await db.flush()
                    result = await db.execute(
                        update(VideoUpload)
                        .where(VideoUpload.id == video_uuid)
                        .values(status="completed")
                    )
                    await db.commit()
                    logger.info("Video upload status updated via fallback method", 
                               upload_id=str(upload_id),
                               video_id=str(video_uuid),
                               rows_affected=result.rowcount)
                except Exception as fallback_error:
                    logger.error("Fallback status update also failed", 
                               upload_id=str(upload_id),
                               video_id=str(video_uuid),
                               error=str(fallback_error))
            
            logger.info("Video processing completed successfully", 
                       job_id=job_id, 
                       upload_id=upload_id,
                       transcript_length=len(result.get("transcript", "")),
                       frames_analyzed=result.get("frame_analyses_count", 0))
            
        except Exception as e:
            logger.error("Video processing failed", 
                        job_id=job_id, 
                        upload_id=upload_id,
                        error=str(e), 
                        exc_info=True)
            
            # Update job status
            try:
                await JobService.update_job(db, job_id, {
                    "status": "failed",
                    "message": f"Processing failed: {str(e)}",
                    "current_step": "failed",
                    "error": str(e)
                })
            except Exception as update_error:
                logger.error("Failed to update job status", 
                           job_id=job_id, 
                           error=str(update_error))
            
            # Update video upload status if upload_id provided
            if upload_id:
                try:
                    from uuid import UUID
                    await VideoUploadService.update_upload_status(
                        db, 
                        UUID(upload_id), 
                        "failed",
                        error=str(e)
                    )
                except Exception as upload_error:
                    logger.error("Failed to update video upload status", 
                               upload_id=upload_id, 
                               error=str(upload_error))
        
        finally:
            # Always cleanup isolated temp directories, even on error
            if job_temp_dir and job_temp_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(job_temp_dir)
                    logger.info("Cleaned up isolated temp directory",
                               job_id=job_id,
                               temp_dir=str(job_temp_dir))
                except Exception as cleanup_error:
                    logger.warning("Failed to cleanup temp directory",
                                 job_id=job_id,
                                 temp_dir=str(job_temp_dir),
                                 error=str(cleanup_error))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_config=None  # Use structlog
    )
