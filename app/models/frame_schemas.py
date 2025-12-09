"""Frame analysis schemas"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID


class FrameAnalysisResponse(BaseModel):
    """Schema for frame analysis response"""
    id: UUID
    video_id: UUID
    timestamp: float = Field(..., description="Timestamp in seconds from video start")
    frame_number: Optional[int] = Field(None, description="Frame number in video")
    image_path: str = Field(..., description="Path to saved frame image")
    description: Optional[str] = Field(None, description="GPT-generated description/caption")
    ocr_text: Optional[str] = Field(None, description="Extracted OCR text")
    gpt_response: Optional[Dict[str, Any]] = Field(None, description="Full GPT response JSON")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    created_at: datetime

    class Config:
        from_attributes = True


class FrameAnalysisListResponse(BaseModel):
    """Schema for frame analysis list response"""
    frames: List[FrameAnalysisResponse] = Field(..., description="List of frame analyses")
    total: int = Field(..., description="Total number of frames")
    video_id: UUID = Field(..., description="Video ID")
    limit: Optional[int] = Field(None, description="Limit applied")
    offset: int = Field(0, description="Offset applied")

