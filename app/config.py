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
    AUDIO_DIR: Union[str, Path] = Path("./audio")  # Directory for extracted audio files
    MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB
    ALLOWED_EXTENSIONS: Union[str, List[str]] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    
    # Frame Analysis Settings
    FRAMES_PER_SECOND: int = 1  # Extract 1 frame per second
    FRAME_ANALYSIS_WORKERS: int = 4  # Number of parallel workers for frame analysis
    
    # Database Settings
    # SQL Server format: mssql+aioodbc://user:password@host:port/database?driver=ODBC+Driver+17+for+SQL+Server
    # PostgreSQL format: postgresql+asyncpg://user:password@host:port/database
    # Instance: druidpartners.druidqa.druidplatform.com
    # Database: Druid_AbhijeetKumar
    # User: admin_abhijeetkumar
    DATABASE_URL: str = "mssql+aioodbc://admin_abhijeetkumar:BpGWuAPyCm2_j7VKRVUlEHn2vi94nB6Z@druidpartners.druidqa.druidplatform.com:1433/Druid_AbhijeetKumar?driver=ODBC+Driver+17+for+SQL+Server"
    
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
    
    @field_validator('OPENAI_API_KEY', mode='before')
    @classmethod
    def parse_openai_api_key(cls, v):
        """Clean and validate OpenAI API key"""
        if v is None:
            return None
        if isinstance(v, str):
            # Remove whitespace, newlines, and join if split across lines
            cleaned = ''.join(v.split()).strip()
            # Remove any quotes
            cleaned = cleaned.strip('"\'')
            # Check if it's still a placeholder value
            if cleaned and cleaned.lower() in ['none', 'null', '']:
                return None
            if cleaned and 'your_openai_api_key_here' in cleaned.lower():
                return None
            return cleaned if cleaned else None
        return v
    
    class Config:
        # Ensure .env file is loaded from the backend directory
        # Try multiple possible locations
        _backend_dir = Path(__file__).parent.parent
        _env_file = _backend_dir / ".env"
        env_file = str(_env_file) if _env_file.exists() else ".env"
        case_sensitive = True
        # Allow reading from environment variables
        env_file_encoding = 'utf-8'


# Try to load from environment variable as fallback if not in .env
_openai_key_from_env = os.getenv("OPENAI_API_KEY")
if _openai_key_from_env:
    # Clean the key from environment variable too
    _openai_key_from_env = ''.join(_openai_key_from_env.split()).strip().strip('"\'')
    if _openai_key_from_env and 'your_openai_api_key_here' not in _openai_key_from_env.lower():
        # Override with environment variable if it's valid
        os.environ["OPENAI_API_KEY"] = _openai_key_from_env

settings = Settings()

# Validate and log API key status on import
if settings.OPENAI_API_KEY:
    masked_key = f"{settings.OPENAI_API_KEY[:7]}...{settings.OPENAI_API_KEY[-4:]}" if len(settings.OPENAI_API_KEY) > 11 else "***"
    print(f"✓ OpenAI API key loaded: {masked_key}")
    # Verify it's not a placeholder
    if 'your_openai_api_key_here' in settings.OPENAI_API_KEY.lower() or 'your_ope' in settings.OPENAI_API_KEY.lower():
        print("⚠️  Warning: OPENAI_API_KEY appears to be a placeholder value!")
        print("   Please update your .env file with a valid API key")
        settings.OPENAI_API_KEY = None
else:
    print("⚠️  Warning: OPENAI_API_KEY not found in .env file or environment variables")
    print("   Please set OPENAI_API_KEY in your .env file or as an environment variable")
