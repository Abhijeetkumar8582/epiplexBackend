"""Video upload and processing schemas"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID


class VideoUploadCreate(BaseModel):
    """Schema for creating a video upload"""
    name: str = Field(..., min_length=1, max_length=255, description="Human-readable name for the video")
    source_type: str = Field(
        default="upload",
        pattern="^(upload|url)$",
        description="Source type: 'upload' for file upload, 'url' for URL input"
    )
    application_name: Optional[str] = Field(None, max_length=100, description="Application name (e.g., SAP, Salesforce)")
    tags: Optional[List[str]] = Field(None, description="Tags as array (e.g., [\"HR\", \"Payroll\"])")
    language_code: Optional[str] = Field(None, max_length=10, description="Language code (e.g., en, hi)")
    priority: Optional[str] = Field(default="normal", pattern="^(normal|high)$", description="Priority: normal, high")


class VideoUploadResponse(BaseModel):
    """Schema for video upload response"""
    id: UUID
    user_id: UUID
    name: str
    source_type: str
    video_url: str = Field(..., description="Storage URL/path of the video")
    original_input: str = Field(..., description="Original filename or URL provided by user")
    status: str = Field(..., description="Upload status: uploaded, processing, completed, failed, cancelled")
    
    # Video metadata
    video_length_seconds: Optional[float] = Field(None, description="Video duration in seconds")
    video_size_bytes: Optional[int] = Field(None, description="Video file size in bytes")
    mime_type: Optional[str] = Field(None, description="Video MIME type (e.g., video/mp4)")
    resolution_width: Optional[int] = Field(None, description="Video width in pixels")
    resolution_height: Optional[int] = Field(None, description="Video height in pixels")
    fps: Optional[float] = Field(None, description="Frames per second")
    
    # Processing link
    job_id: Optional[str] = Field(None, description="Linked processing job ID")
    
    # Video file number
    video_file_number: Optional[str] = Field(None, description="Unique video file number (e.g., VF-2024-0001)")
    
    # Business/Functional metadata
    application_name: Optional[str] = Field(None, description="Application name (e.g., SAP, Salesforce)")
    tags: Optional[List[str]] = Field(None, description="Tags as array (e.g., [\"HR\", \"Payroll\"])")
    language_code: Optional[str] = Field(None, description="Language code (e.g., en, hi)")
    priority: Optional[str] = Field(None, description="Priority: normal, high")
    
    # Soft delete
    is_deleted: bool = Field(default=False, description="Soft delete flag")
    deleted_at: Optional[datetime] = Field(None, description="Timestamp when video was soft deleted")
    
    # Timestamps
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class VideoUploadListResponse(BaseModel):
    """Schema for paginated video upload list response"""
    uploads: List[VideoUploadResponse] = Field(..., description="List of video uploads")
    total: int = Field(..., description="Total number of uploads")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")


class VideoUploadUpdate(BaseModel):
    """Schema for updating video upload metadata"""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Update video name")
    status: Optional[str] = Field(
        None,
        pattern="^(uploaded|processing|completed|failed|cancelled)$",
        description="Update upload status"
    )
    application_name: Optional[str] = Field(None, max_length=100, description="Update application name")
    tags: Optional[List[str]] = Field(None, description="Update tags as array")
    language_code: Optional[str] = Field(None, max_length=10, description="Update language code")
    priority: Optional[str] = Field(None, pattern="^(normal|high)$", description="Update priority")


class BulkDeleteRequest(BaseModel):
    """Schema for bulk delete request"""
    upload_ids: List[str] = Field(..., min_items=1, description="List of upload IDs to delete")
    permanent: bool = Field(default=False, description="Permanently delete (hard delete)")