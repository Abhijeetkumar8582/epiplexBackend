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
from app.services.summary_service import SummaryService
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
    yield
    # Shutdown
    logger.info("Shutting down application")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

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


@app.get("/api/health")
async def api_health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "database": "operational",
            "openai": "configured" if settings.OPENAI_API_KEY else "not_configured"
        }
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
    return UserResponse.model_validate(current_user)


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
        
        # Determine redirect URL
        frontend_url = settings.CORS_ORIGINS[0] if settings.CORS_ORIGINS else "http://localhost:3000"
        
        # If state contains a redirect URI, decode and use it
        redirect_url = f"{frontend_url}/auth/callback"
        if state:
            try:
                import base64
                decoded_state = base64.urlsafe_b64decode(state.encode()).decode()
                if decoded_state.startswith("http"):
                    redirect_url = decoded_state
            except Exception:
                pass  # Use default redirect
        
        # Redirect to frontend with tokens in URL (or use a more secure method)
        # For better security, you might want to use a one-time token exchange
        redirect_url_with_tokens = f"{redirect_url}?token={access_token}&session={session.session_token}"
        
        return RedirectResponse(url=redirect_url_with_tokens)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Google OAuth callback error", error=str(e), exc_info=True)
        frontend_url = settings.CORS_ORIGINS[0] if settings.CORS_ORIGINS else "http://localhost:3000"
        return RedirectResponse(
            url=f"{frontend_url}/auth?error=oauth_failed&message=Authentication failed"
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
        
        return LoginResponse(
            access_token=access_token,
            session_token=session.session_token,
            user=UserResponse.model_validate(user),
            expires_at=session.expires_at
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Google OAuth token exchange error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to authenticate with Google")


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
        # Validate file
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
        
        # Initialize job status in database
        initial_status = {
            "status": "processing",
            "progress": 0,
            "message": "Video uploaded, starting processing...",
            "current_step": "upload",
            "step_progress": {
                "upload": "completed",
                "extract_audio": "pending",
                "transcribe": "pending",
                "extract_frames": "pending",
                "analyze_frames": "pending",
                "summary_generation": "pending",
                "generate_pdf": "pending",
                "complete": "pending"
            }
        }
        
        await JobService.create_job(db, job_id, initial_status)
        
        # Update upload status to processing
        await VideoUploadService.update_upload_status(db, video_upload.id, "processing", job_id)
        
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
        
        # Start background processing (complete pipeline: extract audio -> transcribe -> extract frames -> analyze with GPT -> store in DB)
        logger.info("Starting background video processing task", 
                   job_id=job_id, 
                   upload_id=str(video_upload.id),
                   file_path=str(file_path))
        background_tasks.add_task(process_video_task, str(file_path), job_id, str(video_upload.id))
        
        return VideoUploadResponse.model_validate(video_upload)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload error", error=str(e), exc_info=True)
        error_detail = str(e) if settings.DEBUG else "Failed to upload video"
        raise HTTPException(status_code=500, detail=error_detail)


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
    if format not in ["docx", "pdf", "html"]:
        raise HTTPException(status_code=400, detail="Invalid format. Allowed: docx, pdf, html")
    
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
    # Parse tags if provided
    tags_list = None
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
    
    return VideoPanelResponse(
        videos=videos,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total
    )


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
    from fastapi.responses import JSONResponse
    return JSONResponse(
        content=response_data.model_dump(),
        headers={
            "Cache-Control": "private, max-age=60",  # Cache for 1 minute
            "X-Total-Count": str(total),
            "X-Page": str(page),
            "X-Page-Size": str(page_size)
        }
    )


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


@app.get("/api/activity-logs/stats", response_model=ActivityLogStatsResponse)
async def get_activity_stats(
    days: int = Query(30, ge=1, le=365, description="Number of days to include in statistics"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get activity statistics for the current user"""
    stats = await ActivityService.get_activity_stats(db, current_user.id, days=days)
    
    return ActivityLogStatsResponse(
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


@app.get("/api/activity-logs/actions", response_model=List[str])
async def get_available_actions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get list of available action types for the current user"""
    actions = await ActivityService.get_available_actions(db, current_user.id)
    return sorted(actions)


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
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get complete document/data for a video file number
    
    Returns all frame analyses, GPT responses, video metadata, and summary statistics.
    This is the main endpoint to fetch all data for a video file once processing is complete.
    """
    # Get complete document data
    document_data = await frame_analysis_service.get_complete_document_data(
        db=db,
        video_file_number=video_file_number,
        user_id=current_user.id
    )
    
    if not document_data:
        raise HTTPException(status_code=404, detail="Video not found")
    
    upload = document_data["video_metadata"]
    frames = document_data["frames"]
    
    # Get transcript from job status if job_id exists
    transcript = None
    if upload.job_id:
        job = await JobService.get_job(db, upload.job_id)
        if job and job.transcript:
            transcript = job.transcript
    
    # Log activity
    await ActivityService.log_activity(
        db=db,
        user_id=current_user.id,
        action="FETCH_DOCUMENT",
        description=f"User fetched document for video: {video_file_number}",
        metadata={
            "video_file_number": video_file_number,
            "video_id": str(upload.id),
            "total_frames": document_data["total_frames"],
            "frames_with_gpt": document_data["frames_with_gpt"]
        },
        ip_address=get_client_ip(request)
    )
    
    return DocumentResponse(
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
            summary_pdf_url=upload.summary_pdf_url,
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
        summary_pdf_url=upload.summary_pdf_url,
        created_at=datetime.utcnow()
    )


@app.get("/api/videos/{video_id}/summaries")
async def get_video_summaries(
    video_id: UUID,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all summaries for a video
    
    Returns list of batch summaries generated from frame analyses.
    """
    try:
        # Verify video belongs to user
        from app.services.video_upload_service import VideoUploadService
        video_upload = await VideoUploadService.get_upload(db, video_id, current_user.id)
        
        if not video_upload:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Get summaries
        summary_service = SummaryService()
        summaries = await summary_service.get_video_summaries(db, video_id)
        
        # Log activity
        await ActivityService.log_activity(
            db=db,
            user_id=current_user.id,
            action="FETCH_SUMMARIES",
            description=f"User fetched summaries for video: {video_upload.video_file_number}",
            metadata={
                "video_id": str(video_id),
                "video_file_number": video_upload.video_file_number,
                "total_summaries": len(summaries)
            },
            ip_address=get_client_ip(request)
        )
        
        return {
            "video_id": str(video_id),
            "video_file_number": video_upload.video_file_number,
            "total_summaries": len(summaries),
            "summaries": summaries
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get video summaries",
                   video_id=str(video_id),
                   error=str(e),
                   exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get summaries: {str(e)}")


@app.post("/api/videos/{video_id}/summaries/generate")
async def generate_video_summaries(
    video_id: UUID,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Manually trigger summary generation for a video
    
    This will process all frame analyses in batches of 30 and generate summaries.
    """
    try:
        # Verify video belongs to user
        from app.services.video_upload_service import VideoUploadService
        video_upload = await VideoUploadService.get_upload(db, video_id, current_user.id)
        
        if not video_upload:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Generate summaries
        summary_service = SummaryService()
        summaries = await summary_service.generate_video_summaries(db, video_id)
        
        # Log activity
        await ActivityService.log_activity(
            db=db,
            user_id=current_user.id,
            action="GENERATE_SUMMARIES",
            description=f"User generated summaries for video: {video_upload.video_file_number}",
            metadata={
                "video_id": str(video_id),
                "video_file_number": video_upload.video_file_number,
                "total_summaries": len(summaries)
            },
            ip_address=get_client_ip(request)
        )
        
        return {
            "video_id": str(video_id),
            "video_file_number": video_upload.video_file_number,
            "total_summaries": len(summaries),
            "summaries": summaries,
            "message": f"Successfully generated {len(summaries)} summaries"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to generate video summaries",
                   video_id=str(video_id),
                   error=str(e),
                   exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate summaries: {str(e)}")


@app.get("/api/videos/{video_id}/summary-pdf")
async def download_summary_pdf(
    video_id: UUID,
    request: Request,
    token: Optional[str] = Query(None, description="Optional token for iframe access"),
    current_user: Optional[User] = Depends(lambda: None),  # Make optional for token-based access
    db: AsyncSession = Depends(get_db)
):
    """
    Download the generated summary PDF for a video
    Supports both Bearer token and query parameter token for iframe access
    """
    try:
        from app.services.video_upload_service import VideoUploadService
        from app.services.auth_service import AuthService
        
        # Handle authentication - either from current_user or token parameter
        user_id = None
        if current_user:
            user_id = current_user.id
        elif token:
            # Verify token and get user
            try:
                user_data = AuthService.verify_token(token)
                if user_data:
                    user_id = UUID(user_data.get("sub"))  # JWT sub claim contains user_id
            except Exception as token_error:
                logger.warning("Token verification failed", error=str(token_error))
        
        if not user_id:
            # Try to get user from Authorization header if token param failed
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                try:
                    token_from_header = auth_header.split(" ")[1]
                    user_data = AuthService.verify_token(token_from_header)
                    if user_data:
                        user_id = UUID(user_data.get("sub"))
                except Exception:
                    pass
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        # Verify video belongs to user
        video_upload = await VideoUploadService.get_upload(db, video_id, user_id)
        
        if not video_upload:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if not video_upload.summary_pdf_url:
            raise HTTPException(status_code=404, detail="Summary PDF not found for this video")
        
        # Get PDF path
        pdf_path = Path(video_upload.summary_pdf_url)
        
        # If it's a relative path, resolve it relative to OUTPUT_DIR
        if not pdf_path.is_absolute():
            pdf_path = settings.OUTPUT_DIR / pdf_path
        
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="Summary PDF file not found on server")
        
        # Log activity
        if user_id:
            await ActivityService.log_activity(
                db=db,
                user_id=user_id,
                action="DOWNLOAD_SUMMARY_PDF",
                description=f"User downloaded summary PDF for video: {video_upload.video_file_number}",
                metadata={
                    "video_id": str(video_id),
                    "video_file_number": video_upload.video_file_number
                },
                ip_address=get_client_ip(request)
            )
        
        return FileResponse(
            path=str(pdf_path),
            filename=f"video_summary_{video_upload.video_file_number}.pdf",
            media_type="application/pdf"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to download summary PDF",
                   video_id=str(video_id),
                   error=str(e),
                   exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to download PDF: {str(e)}")


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
    """
    from app.database import AsyncSessionLocal
    from app.services.video_processing_service import VideoProcessingService
    
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
    
    async with AsyncSessionLocal() as db:
        try:
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
            
            # Run complete processing pipeline
            result = await processing_service.process_video_complete(
                video_path=file_path,
                video_id=video_uuid,
                job_id=job_id,
                frames_dir=FRAMES_DIR,
                audio_dir=AUDIO_DIR,
                db=db
            )
            
            # Update video upload status to completed
            try:
                await VideoUploadService.update_upload_status(db, video_uuid, "completed", job_id)
            except Exception as e:
                logger.error("Failed to update video upload status", 
                           upload_id=upload_id, 
                           error=str(e))
            
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_config=None  # Use structlog
    )
