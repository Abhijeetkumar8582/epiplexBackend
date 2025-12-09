"""Schemas for GPT response queries"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import UUID


class GPTResponseItem(BaseModel):
    """Schema for a single GPT response item"""
    frame_id: UUID = Field(..., description="Frame analysis ID")
    timestamp: float = Field(..., description="Frame timestamp in seconds")
    frame_number: Optional[int] = Field(None, description="Frame number in video")
    image_path: str = Field(..., description="Path to frame image")
    description: Optional[str] = Field(None, description="GPT-generated description")
    ocr_text: Optional[str] = Field(None, description="Extracted OCR text")
    gpt_response: Optional[Dict[str, Any]] = Field(None, description="Full GPT response JSON")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    created_at: datetime = Field(..., description="When frame was analyzed")

    class Config:
        from_attributes = True


class GPTResponseListResponse(BaseModel):
    """Schema for GPT response list"""
    video_file_number: str = Field(..., description="Video file number")
    video_id: UUID = Field(..., description="Video upload ID")
    video_name: str = Field(..., description="Video name")
    total_responses: int = Field(..., description="Total number of GPT responses")
    responses: List[GPTResponseItem] = Field(..., description="List of GPT responses")

