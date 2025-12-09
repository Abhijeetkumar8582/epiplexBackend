"""Job processing and status schemas"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime


class ProcessingStatus(BaseModel):
    """Schema for processing job status"""
    status: str = Field(..., description="Job status: processing, completed, failed")
    progress: int = Field(..., ge=0, le=100, description="Processing progress (0-100)")
    message: str = Field(..., description="Status message")
    output_files: Optional[Dict[str, str]] = Field(None, description="Generated output files")
    current_step: Optional[str] = Field(None, description="Current processing step")
    step_progress: Optional[Dict[str, str]] = Field(None, description="Progress of each step")
    transcript: Optional[str] = Field(None, description="Extracted transcript")
    frame_analyses: Optional[List[Dict]] = Field(None, description="Frame analysis results")
    error: Optional[str] = Field(None, description="Error message if failed")


class ProcessingResult(BaseModel):
    """Schema for completed processing result"""
    job_id: str = Field(..., description="Processing job ID")
    transcript: str = Field(..., description="Extracted transcript from video")
    frame_analyses: List[Dict] = Field(..., description="Analyzed frames with descriptions")
    output_files: Dict[str, str] = Field(..., description="Generated document files")


class JobStatusResponse(BaseModel):
    """Schema for job status API response"""
    job_id: str
    status: str
    progress: int
    message: str
    current_step: Optional[str] = None
    step_progress: Optional[Dict[str, str]] = None
    output_files: Optional[Dict[str, str]] = None
    transcript: Optional[str] = None
    frame_analyses: Optional[List[Dict]] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

