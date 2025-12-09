from fastapi import UploadFile, HTTPException
from pathlib import Path
from app.config import settings
from app.utils.logger import logger


def validate_file(file: UploadFile) -> None:
    """Validate uploaded file"""
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    # Check content type
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=400,
            detail="File must be a video file"
        )
    
    logger.info("File validation passed", filename=file.filename, content_type=file.content_type)


async def validate_file_size(file: UploadFile) -> None:
    """Validate file size"""
    # Read file in chunks to check size
    size = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        size += len(chunk)
        if size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.0f}MB"
            )
    
    # Reset file pointer
    await file.seek(0)
    
    logger.info("File size validation passed", filename=file.filename, size_mb=size / (1024*1024))
