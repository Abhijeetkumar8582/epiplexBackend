"""Models package - exports all schemas"""

# Authentication schemas
from app.models.auth_schemas import (
    UserSignup,
    UserLogin,
    UserResponse,
    LoginResponse,
    SignupResponse
)

# Video upload schemas
from app.models.video_schemas import (
    VideoUploadCreate,
    VideoUploadResponse,
    VideoUploadListResponse,
    VideoUploadUpdate,
    BulkDeleteRequest
)

# Job processing schemas
from app.models.job_schemas import (
    ProcessingStatus,
    ProcessingResult,
    JobStatusResponse
)

# Frame analysis schemas
from app.models.frame_schemas import (
    FrameAnalysisResponse,
    FrameAnalysisListResponse
)

# Activity log schemas
from app.models.activity_schemas import (
    ActivityLogResponse,
    ActivityLogListResponse,
    ActivityLogStatsResponse
)

# Document schemas
from app.models.document_schemas import (
    DocumentResponse,
    VideoMetadata,
    FrameData
)

# Video panel schemas
from app.models.video_panel_schemas import (
    VideoPanelItem,
    VideoPanelResponse
)

__all__ = [
    # Auth schemas
    "UserSignup",
    "UserLogin",
    "UserResponse",
    "LoginResponse",
    "SignupResponse",
    # Video schemas
    "VideoUploadCreate",
    "VideoUploadResponse",
    "VideoUploadListResponse",
    "VideoUploadUpdate",
    "BulkDeleteRequest",
    # Job schemas
    "ProcessingStatus",
    "ProcessingResult",
    "JobStatusResponse",
    # Frame schemas
    "FrameAnalysisResponse",
    "FrameAnalysisListResponse",
    # Activity schemas
    "ActivityLogResponse",
    "ActivityLogListResponse",
    "ActivityLogStatsResponse",
    # Document schemas
    "DocumentResponse",
    "VideoMetadata",
    "FrameData",
    # Video panel schemas
    "VideoPanelItem",
    "VideoPanelResponse",
]
