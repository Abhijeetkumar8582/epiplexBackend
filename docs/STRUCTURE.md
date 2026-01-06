# Backend Project Structure

This document describes the organized structure of the Epiplex backend project.

## 📁 Directory Structure

```
backend/
├── app/                          # Main application package
│   ├── __init__.py
│   ├── main.py                   # FastAPI app entry point
│   ├── config.py                 # Configuration management
│   ├── database.py               # Database models and setup
│   ├── prompt.txt                # GPT prompt template
│   │
│   ├── middleware/              # Custom middleware
│   │   ├── __init__.py
│   │   ├── error_handler.py     # Global error handling
│   │   ├── rate_limit.py        # Rate limiting
│   │   └── security.py          # Security middleware
│   │
│   ├── models/                  # Pydantic schemas
│   │   ├── __init__.py
│   │   ├── activity_schemas.py
│   │   ├── auth_schemas.py
│   │   ├── document_schemas.py
│   │   ├── frame_schemas.py
│   │   ├── gpt_response_schemas.py
│   │   ├── job_schemas.py
│   │   ├── video_panel_schemas.py
│   │   └── video_schemas.py
│   │
│   ├── services/               # Business logic services
│   │   ├── __init__.py
│   │   ├── activity_service.py
│   │   ├── audio_extraction_service.py
│   │   ├── auth_service.py
│   │   ├── document_generator.py
│   │   ├── frame_analysis_service.py
│   │   ├── frame_extraction_service.py
│   │   ├── google_oauth_service.py
│   │   ├── gpt_mock_service.py
│   │   ├── gpt_service.py
│   │   ├── job_service.py
│   │   ├── video_file_number_service.py
│   │   ├── video_metadata_service.py
│   │   ├── video_processing_service.py
│   │   ├── video_processor.py
│   │   └── video_upload_service.py
│   │
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       ├── encryption.py        # Data encryption utilities
│       ├── logger.py            # Logging configuration
│       └── validators.py        # Input validation
│
├── docs/                        # Documentation
│   ├── API_ENDPOINTS.md
│   ├── AUTHENTICATION.md
│   ├── DATABASE_RELATIONSHIPS.md
│   ├── ENCRYPTION_SETUP.md
│   ├── ENV_SETUP.md
│   ├── FRAME_ANALYSIS.md
│   ├── GOOGLE_OAUTH_SETUP.md
│   ├── SQL_SERVER_SETUP.md
│   ├── SQL_SERVER_SETUP_STEPS.md
│   └── STRUCTURE.md             # This file
│
├── scripts/                     # Utility scripts
│   ├── __init__.py
│   ├── check_google_oauth.py
│   ├── generate_encryption_key.py
│   ├── run_migration.py
│   ├── run_migration_014.py
│   └── update_google_oauth_env.py
│
├── tests/                       # Test files
│   ├── __init__.py
│   ├── test_frame_analysis.py
│   └── test_login.py
│
├── migrations/                  # Database migrations
│   ├── README.md
│   └── [migration files].sql
│
├── audio/                       # Audio files (auto-deleted after processing)
│   └── .gitkeep
│
├── frames/                      # Frame images (auto-deleted after processing)
│   └── .gitkeep
│
├── uploads/                     # Uploaded video files
│   └── .gitkeep
│
├── outputs/                     # Generated documents (DOCX, HTML)
│   └── .gitkeep
│
├── .gitignore                   # Git ignore rules
├── Dockerfile                   # Docker configuration
├── README.md                    # Main README
├── requirements.txt             # Python dependencies
├── start.py                     # Application startup script
├── env.example                  # Environment variables template
└── env.production.example       # Production environment template
```

## 📝 File Organization

### Application Code (`app/`)
- **main.py**: FastAPI application with all API endpoints
- **config.py**: Centralized configuration management
- **database.py**: SQLAlchemy models and database setup
- **prompt.txt**: GPT prompt template for frame analysis

### Documentation (`docs/`)
All markdown documentation files are organized in the `docs/` folder for easy access.

### Scripts (`scripts/`)
Utility scripts for:
- Database migrations
- Encryption key generation
- Google OAuth setup
- Environment configuration

### Tests (`tests/`)
Test files for validating functionality.

### Data Directories
- **audio/**: Temporary audio files (auto-deleted after transcription)
- **frames/**: Temporary frame images (auto-deleted after base64 storage)
- **uploads/**: Uploaded video files
- **outputs/**: Generated documents (DOCX, HTML)

## 🔧 Running Scripts

All scripts should be run from the `backend/` directory:

```bash
# From backend directory
python scripts/generate_encryption_key.py
python scripts/run_migration.py
python scripts/check_google_oauth.py
```

## 📚 Documentation

All documentation is in the `docs/` folder. See `README.md` for quick start guide.

## 🧹 Cleanup

The system automatically cleans up temporary files:
- Frame files are deleted after base64 images are stored
- Audio files are deleted after transcription completes

