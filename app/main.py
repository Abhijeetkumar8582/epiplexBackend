from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, Request, Header, Query
from fastapi.middleware.cors import CORSMiddleware
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
from app.models import (
    UserSignup, UserLogin, SignupResponse, LoginResponse, UserResponse,
    VideoUploadCreate, VideoUploadResponse, VideoUploadListResponse, VideoUploadUpdate,
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
        
        # Save uploaded file
        file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info("File uploaded", job_id=job_id, filename=file.filename, size_mb=round(file_size_mb, 2))
        
        # Extract video metadata
        metadata = VideoMetadataService.extract_metadata(str(file_path))
        
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
        
        # Create video upload record
        video_upload = await VideoUploadService.create_upload(
            db=db,
            user_id=current_user.id,
            name=video_name,
            source_type="upload",
            video_url=str(file_path),  # Store local path, can be S3 URL in production
            original_input=file.filename or "unknown",
            status="uploaded",
            job_id=job_id,
            metadata=metadata,
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
                "upload": "processing",
                "analyze": "pending",
                "extract_frames": "pending",
                "process": "pending"
            }
        }
        
        await JobService.create_job(db, job_id, initial_status)
        
        # Update upload status to processing
        await VideoUploadService.update_upload_status(db, video_upload.id, "processing", job_id)
        
        # Log activity
        await ActivityService.log_activity(
            db=db,
            user_id=current_user.id,
            action="UPLOAD_VIDEO",
            description=f"User uploaded video: {video_name}",
            metadata={
                "upload_id": str(video_upload.id),
                "job_id": job_id,
                "filename": file.filename,
                "size_bytes": metadata.get("video_size_bytes")
            },
            ip_address=get_client_ip(request)
        )
        
        # Start background processing (complete pipeline: upload -> transcribe -> extract frames -> analyze with GPT -> store in DB)
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
    
    return ActivityLogListResponse(
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


async def process_video_task(file_path: str, job_id: str, upload_id: Optional[str] = None):
    """
    Production-ready background task to process video:
    1. Extract audio and transcribe using OpenAI Whisper
    2. Extract keyframes (1 per second)
    3. Process frames in batches of 5 through ChatGPT 4o Mini
    4. Store everything in database
    """
    from app.database import AsyncSessionLocal
    from app.services.video_processing_service import VideoProcessingService
    
    async with AsyncSessionLocal() as db:
        try:
            # Initialize processing service
            processing_service = VideoProcessingService()
            
            # Step 1: Mark upload as complete
            await JobService.update_job(db, job_id, {
                "progress": 5,
                "message": "Video uploaded successfully. Starting processing...",
                "current_step": "upload",
                "step_progress": {
                    "upload": "completed",
                    "transcribe": "pending",
                    "extract_frames": "pending",
                    "analyze_frames": "pending",
                    "complete": "pending"
                }
            })
            
            # Convert upload_id to UUID if provided
            video_uuid = None
            if upload_id:
                from uuid import UUID
                video_uuid = UUID(upload_id)
            
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
                    await VideoUploadService.update_upload_status(db, UUID(upload_id), "failed")
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
