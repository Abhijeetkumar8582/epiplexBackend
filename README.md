# Epiplex Backend API

Production-ready FastAPI backend for video processing and analysis.

## 📁 Project Structure

```
backend/
├── app/                    # Main application code
│   ├── main.py             # FastAPI application entry point
│   ├── config.py           # Configuration settings
│   ├── database.py         # Database models and setup
│   ├── middleware/         # Custom middleware (error handling, rate limiting, security)
│   ├── models/             # Pydantic schemas for API requests/responses
│   ├── services/           # Business logic services
│   ├── utils/              # Utility functions (logger, encryption, validators)
│   └── prompt.txt          # GPT prompt template for frame analysis
│
├── docs/                   # Documentation files
│   ├── API_ENDPOINTS.md
│   ├── AUTHENTICATION.md
│   ├── DATABASE_RELATIONSHIPS.md
│   ├── ENCRYPTION_SETUP.md
│   ├── ENV_SETUP.md
│   ├── FRAME_ANALYSIS.md
│   ├── GOOGLE_OAUTH_SETUP.md
│   └── SQL_SERVER_SETUP.md
│
├── scripts/                # Utility scripts
│   ├── check_google_oauth.py
│   ├── generate_encryption_key.py
│   ├── run_migration.py
│   ├── run_migration_014.py
│   └── update_google_oauth_env.py
│
├── tests/                  # Test files
│   ├── test_frame_analysis.py
│   └── test_login.py
│
├── migrations/             # Database migration scripts
│
├── audio/                  # Extracted audio files (auto-deleted after processing)
├── frames/                 # Extracted frame images (auto-deleted after processing)
├── uploads/                # Uploaded video files
├── outputs/                # Generated documents (DOCX, HTML)
│
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── start.py                # Application startup script
├── env.example             # Environment variables template
└── env.production.example  # Production environment template
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL or SQL Server database
- FFmpeg (for audio extraction)
- OpenAI API key

### Installation

1. **Clone the repository and navigate to backend:**
   ```bash
   cd backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

5. **Run database migrations:**
   ```bash
   python scripts/run_migration.py
   ```

6. **Start the server:**
   ```bash
   python start.py
   # Or use uvicorn directly:
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## 📚 Documentation

All documentation is available in the `docs/` folder:

- **API_ENDPOINTS.md** - Complete API reference
- **AUTHENTICATION.md** - Authentication setup and usage
- **ENV_SETUP.md** - Environment configuration guide
- **DATABASE_RELATIONSHIPS.md** - Database schema documentation
- **FRAME_ANALYSIS.md** - Frame analysis workflow
- **GOOGLE_OAUTH_SETUP.md** - Google OAuth configuration
- **ENCRYPTION_SETUP.md** - Encryption key setup
- **SQL_SERVER_SETUP.md** - SQL Server database setup

## 🔧 Configuration

Key configuration files:

- **`.env`** - Development environment variables (create from `env.example`)
- **`env.production.example`** - Production environment template
- **`app/config.py`** - Application configuration settings
- **`app/prompt.txt`** - GPT prompt template for frame analysis

## 🏗️ Architecture

### Processing Pipeline

1. **Upload** - Video file uploaded and stored
2. **Extract Audio** - Audio extracted from video using FFmpeg
3. **Transcribe** - Audio transcribed using OpenAI Whisper
4. **Extract Frames** - Keyframes extracted (1 every 2 seconds)
5. **Analyze Frames** - Frames analyzed using GPT-4o-mini Vision API
6. **Complete** - All data stored in database, temporary files cleaned up

### Key Services

- **VideoProcessingService** - Main video processing pipeline
- **GPTService** - GPT-4o-mini Vision API integration
- **FrameExtractionService** - Frame extraction using OpenCV
- **AudioExtractionService** - Audio extraction using FFmpeg
- **VideoUploadService** - Video upload and management
- **JobService** - Job status tracking

## 🧪 Testing

Run tests from the `tests/` directory:

```bash
python tests/test_login.py
python tests/test_frame_analysis.py
```

## 🐳 Docker

Build and run with Docker:

```bash
docker build -t epiplex-backend .
docker run -p 8000:8000 --env-file .env epiplex-backend
```

## 📝 Scripts

Utility scripts in `scripts/`:

- **`generate_encryption_key.py`** - Generate encryption key for sensitive data
- **`run_migration.py`** - Run database migrations
- **`check_google_oauth.py`** - Verify Google OAuth configuration
- **`update_google_oauth_env.py`** - Update Google OAuth credentials in .env

## 🔒 Security

- Environment variables for sensitive data
- JWT token-based authentication
- Rate limiting on API endpoints
- Input validation and sanitization
- Encrypted storage for API keys

## 📊 Database

Supports:
- PostgreSQL (recommended)
- SQL Server

Database migrations are in `migrations/` directory.

## 🧹 Cleanup

The system automatically cleans up temporary files:
- **Frame files** - Deleted after base64 images are stored in database
- **Audio files** - Deleted after transcription is complete

## 📄 License

[Your License Here]

## 🤝 Contributing

[Contributing Guidelines]

