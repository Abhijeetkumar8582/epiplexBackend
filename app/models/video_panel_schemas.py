"""Schemas for video panel/list view"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID


class VideoPanelItem(BaseModel):
    """Schema for a single video item in the panel"""
    id: UUID = Field(..., description="Video upload ID")
    video_file_number: str = Field(..., description="Video file number (e.g., VF-2024-0001)")
    name: str = Field(..., description="Video name")
    status: str = Field(..., description="Status: uploaded, processing, completed, failed, cancelled")
    created_at: datetime = Field(..., description="When video was created")
    updated_at: datetime = Field(..., description="Last update timestamp")
    last_activity: datetime = Field(..., description="Last activity timestamp (same as updated_at)")
    
    # Additional metadata for panel display
    video_length_seconds: Optional[float] = Field(None, description="Video duration in seconds")
    video_size_bytes: Optional[int] = Field(None, description="Video file size in bytes")
    application_name: Optional[str] = Field(None, description="Application name")
    tags: Optional[List[str]] = Field(None, description="Tags")
    language_code: Optional[str] = Field(None, description="Language code")
    priority: Optional[str] = Field(None, description="Priority: normal, high")
    
    # Frame analysis stats
    total_frames: Optional[int] = Field(None, description="Total frames analyzed")
    frames_with_gpt: Optional[int] = Field(None, description="Frames with GPT responses")
    
    class Config:
        from_attributes = True


class VideoPanelResponse(BaseModel):
    """Schema for video panel/list response"""
    videos: List[VideoPanelItem] = Field(..., description="List of videos")
    total: int = Field(..., description="Total number of videos")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    has_more: bool = Field(..., description="Whether there are more pages")

