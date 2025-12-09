from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Union, Optional
from pathlib import Path
import os


class Settings(BaseSettings):
    # API Settings
    API_TITLE: str = "Video Processing API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Production-ready video processing API"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # CORS Settings
    CORS_ORIGINS: Union[str, List[str]] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000"
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: Union[str, List[str]] = ["*"]
    CORS_ALLOW_HEADERS: Union[str, List[str]] = ["*"]
    
    # OpenAI Settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MAX_RETRIES: int = 3
    OPENAI_TIMEOUT: int = 300
    
    # File Settings
    UPLOAD_DIR: Union[str, Path] = Path("./uploads")
    OUTPUT_DIR: Union[str, Path] = Path("./outputs")
    FRAMES_DIR: Union[str, Path] = Path("./frames")  # Directory for extracted frames
    MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB
    ALLOWED_EXTENSIONS: Union[str, List[str]] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    
    # Frame Analysis Settings
    FRAMES_PER_SECOND: int = 1  # Extract 1 frame per second
    FRAME_ANALYSIS_WORKERS: int = 4  # Number of parallel workers for frame analysis
    
    # Database Settings
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/epiplex"
    
    # Auth Settings
    SECRET_KEY: str = "your-secret-key-change-in-production"  # Should be in env
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    SESSION_TOKEN_LENGTH: int = 32
    
    # Google OAuth2 Settings
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/auth/google/callback"
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 10
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Background Tasks
    MAX_WORKERS: int = 4
    
    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',') if origin.strip()]
        return v
    
    @field_validator('CORS_ALLOW_METHODS', mode='before')
    @classmethod
    def parse_cors_methods(cls, v):
        if isinstance(v, str):
            return [method.strip() for method in v.split(',') if method.strip()]
        return v
    
    @field_validator('CORS_ALLOW_HEADERS', mode='before')
    @classmethod
    def parse_cors_headers(cls, v):
        if isinstance(v, str):
            return [header.strip() for header in v.split(',') if header.strip()]
        return v
    
    @field_validator('UPLOAD_DIR', mode='before')
    @classmethod
    def parse_upload_dir(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v
    
    @field_validator('OUTPUT_DIR', mode='before')
    @classmethod
    def parse_output_dir(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v
    
    @field_validator('FRAMES_DIR', mode='before')
    @classmethod
    def parse_frames_dir(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v
    
    @field_validator('ALLOWED_EXTENSIONS', mode='before')
    @classmethod
    def parse_allowed_extensions(cls, v):
        if isinstance(v, str):
            return [ext.strip() for ext in v.split(',') if ext.strip()]
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        # Allow reading from environment variables
        env_file_encoding = 'utf-8'


settings = Settings()
