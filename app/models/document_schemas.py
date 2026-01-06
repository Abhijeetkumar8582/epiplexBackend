"""Schemas for document/data fetching"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID


class FrameData(BaseModel):
    """Schema for a single frame with GPT response"""
    frame_id: UUID = Field(..., description="Frame analysis ID")
    timestamp: float = Field(..., description="Frame timestamp in seconds")
    frame_number: Optional[int] = Field(None, description="Frame number in video")
    image_path: str = Field(..., description="Path to frame image")
    base64_image: Optional[str] = Field(None, description="Base64 encoded frame image")
    description: Optional[str] = Field(None, description="GPT-generated description")
    ocr_text: Optional[str] = Field(None, description="Extracted OCR text")
    gpt_response: Optional[Dict[str, Any]] = Field(None, description="Full GPT response JSON")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    created_at: datetime = Field(..., description="When frame was analyzed")

    class Config:
        from_attributes = True


class VideoMetadata(BaseModel):
    """Schema for video metadata"""
    video_id: UUID
    video_file_number: str
    name: str
    status: str
    video_length_seconds: Optional[float] = None
    video_size_bytes: Optional[int] = None
    resolution_width: Optional[int] = None
    resolution_height: Optional[int] = None
    fps: Optional[float] = None
    application_name: Optional[str] = None
    tags: Optional[List[str]] = None
    language_code: Optional[str] = None
    priority: Optional[str] = None
    audio_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentResponse(BaseModel):
    """Schema for complete document/data response"""
    video_file_number: str = Field(..., description="Video file number")
    video_metadata: VideoMetadata = Field(..., description="Video metadata")
    total_frames: int = Field(..., description="Total number of frames analyzed")
    frames_with_gpt: int = Field(..., description="Number of frames with GPT responses")
    frames: List[FrameData] = Field(..., description="List of all frame analyses")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")
    transcript: Optional[str] = Field(None, description="Transcribed text from video audio")
    created_at: datetime = Field(..., description="When document was generated")
    
    class Config:
        from_attributes = True

