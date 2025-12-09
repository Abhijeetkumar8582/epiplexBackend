from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, String, Integer, JSON, DateTime, Text, Boolean, ForeignKey, BigInteger, Float
from sqlalchemy.dialects.postgresql import UUID, INET, TIMESTAMP, JSONB
from sqlalchemy.sql import func
from datetime import datetime
import uuid
from app.config import settings

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name = Column(String(150), nullable=False)
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=True)  # Nullable for OAuth users
    google_id = Column(String(255), nullable=True, unique=True, index=True)  # Google OAuth ID
    provider = Column(String(50), nullable=False, default='email')  # 'email' or 'google'
    role = Column(String(50), nullable=False, default='user')
    is_active = Column(Boolean, nullable=False, default=True)
    last_login_at = Column(TIMESTAMP(timezone=True), nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    activity_logs = relationship("UserActivityLog", back_populates="user", cascade="all, delete-orphan")
    video_uploads = relationship("VideoUpload", back_populates="user", cascade="all, delete-orphan")


class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    session_token = Column(String(255), nullable=False, unique=True, index=True)
    ip_address = Column(INET, nullable=True)
    user_agent = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    expires_at = Column(TIMESTAMP(timezone=True), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="sessions")


class UserActivityLog(Base):
    __tablename__ = "user_activity_logs"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    action = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    metadata = Column(JSONB, nullable=True)
    ip_address = Column(INET, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="activity_logs")


class VideoUpload(Base):
    __tablename__ = "video_uploads"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Essential fields
    name = Column(String(255), nullable=False)  # Human-readable name
    source_type = Column(String(50), nullable=False, default='upload')  # 'upload' or 'url'
    video_url = Column(Text, nullable=False)  # Storage URL/path
    original_input = Column(Text, nullable=False)  # Original filename or URL
    status = Column(String(50), nullable=False, default='uploaded', index=True)  # uploaded, processing, completed, failed, cancelled
    
    # Video tech metadata
    video_length_seconds = Column(Float, nullable=True)
    video_size_bytes = Column(BigInteger, nullable=True)
    mime_type = Column(String(100), nullable=True)
    resolution_width = Column(Integer, nullable=True)
    resolution_height = Column(Integer, nullable=True)
    fps = Column(Float, nullable=True)
    
    # Business/Functional metadata
    application_name = Column(String(100), nullable=True, description="Application name (e.g., SAP, Salesforce)")
    tags = Column(JSONB, nullable=True, description="Tags as JSON array (e.g., [\"HR\", \"Payroll\"])")
    language_code = Column(String(10), nullable=True, description="Language code (e.g., en, hi)")
    priority = Column(String(20), nullable=True, default='normal', description="Priority: normal, high")
    
    # Soft delete support
    is_deleted = Column(Boolean, nullable=False, default=False, index=True)
    deleted_at = Column(TIMESTAMP(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Link to job_status (optional, for processing tracking)
    job_id = Column(String, nullable=True, index=True)
    
    # Video file number - unique identifier for the video file
    video_file_number = Column(String(50), nullable=True, unique=True, index=True, description="Unique video file number (e.g., VF-2024-001)")
    
    # Relationships
    user = relationship("User", back_populates="video_uploads")
    frame_analyses = relationship("FrameAnalysis", back_populates="video_upload", cascade="all, delete-orphan", order_by="FrameAnalysis.timestamp")


class FrameAnalysis(Base):
    __tablename__ = "frame_analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey('video_uploads.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Frame metadata
    timestamp = Column(Float, nullable=False, description="Timestamp in seconds")
    frame_number = Column(Integer, nullable=True, description="Frame number in video")
    image_path = Column(Text, nullable=False, description="Path to saved frame image")
    
    # Analysis results
    description = Column(Text, nullable=True, description="GPT-generated description/caption")
    ocr_text = Column(Text, nullable=True, description="Extracted OCR text from GPT analysis")
    gpt_response = Column(JSONB, nullable=True, description="Full GPT response JSON (for future use)")
    
    # Processing metadata
    processing_time_ms = Column(Integer, nullable=True, description="Time taken to process frame in milliseconds")
    
    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())
    
    # Relationships
    video_upload = relationship("VideoUpload", back_populates="frame_analyses")


class JobStatus(Base):
    __tablename__ = "job_status"
    
    job_id = Column(String, primary_key=True, index=True)
    status = Column(String, nullable=False, index=True)
    progress = Column(Integer, default=0)
    message = Column(Text)
    current_step = Column(String)
    step_progress = Column(JSON)
    output_files = Column(JSON)
    transcript = Column(Text)  # Store transcript text
    frame_analyses = Column(JSON)  # Store frame analyses
    error = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """Get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
