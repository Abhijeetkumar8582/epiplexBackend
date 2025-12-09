"""Activity log schemas"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class ActivityLogResponse(BaseModel):
    """Schema for activity log response"""
    id: int
    user_id: str = Field(..., description="User ID (UUID as string)")
    action: str = Field(..., description="Action type (e.g., LOGIN, UPLOAD_VIDEO)")
    description: Optional[str] = Field(None, description="Human-readable activity description")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional structured metadata")
    ip_address: Optional[str] = Field(None, description="IP address from which activity occurred")
    created_at: datetime = Field(..., description="Timestamp when activity occurred")

    class Config:
        from_attributes = True


class ActivityLogListResponse(BaseModel):
    """Schema for paginated activity log list response"""
    logs: List[ActivityLogResponse] = Field(..., description="List of activity logs")
    total: int = Field(..., description="Total number of logs")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")


class ActivityLogStatsResponse(BaseModel):
    """Schema for activity log statistics"""
    total_activities: int
    activities_by_action: Dict[str, int] = Field(..., description="Count of activities grouped by action type")
    recent_activities: List[ActivityLogResponse] = Field(..., description="Most recent activities")

