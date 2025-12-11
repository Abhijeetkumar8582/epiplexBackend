"""Service for generating unique video file numbers"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from typing import Optional
from datetime import datetime

from app.database import VideoUpload
from app.utils.logger import logger
from app.config import settings


class VideoFileNumberService:
    """Service for generating and managing video file numbers"""
    
    @staticmethod
    async def generate_video_file_number(db: AsyncSession) -> str:
        """
        Generate a unique video file number in format: VF-YYYY-NNNN
        Example: VF-2024-0001, VF-2024-0002, etc.
        
        Returns:
            Unique video file number string
        """
        current_year = datetime.utcnow().year
        
        # Check if using SQL Server
        is_sql_server = "mssql" in settings.DATABASE_URL.lower()
        
        # Get the highest sequence number for current year
        # Using raw SQL for better performance with large datasets
        if is_sql_server:
            # SQL Server syntax: Use RIGHT to get last 4 characters (sequence number)
            # Format is VF-YYYY-NNNN, so RIGHT(video_file_number, 4) gets NNNN
            query = text("""
                SELECT COALESCE(MAX(
                    CAST(RIGHT(video_file_number, 4) AS INTEGER)
                ), 0) as max_seq
                FROM video_uploads
                WHERE video_file_number LIKE :pattern
            """)
        else:
            # PostgreSQL syntax: Use SUBSTRING with regex
            query = text("""
                SELECT COALESCE(MAX(
                    CAST(SUBSTRING(video_file_number FROM '\\d+$') AS INTEGER)
                ), 0) as max_seq
                FROM video_uploads
                WHERE video_file_number LIKE :pattern
            """)
        
        pattern = f"VF-{current_year}-%"
        result = await db.execute(query, {"pattern": pattern})
        row = result.fetchone()
        max_seq = row[0] if row else 0
        
        # Increment sequence
        next_seq = max_seq + 1
        
        # Format: VF-YYYY-NNNN (4-digit sequence)
        video_file_number = f"VF-{current_year}-{next_seq:04d}"
        
        logger.info("Generated video file number", 
                   video_file_number=video_file_number,
                   year=current_year,
                   sequence=next_seq)
        
        return video_file_number
    
    @staticmethod
    async def get_upload_by_file_number(
        db: AsyncSession,
        video_file_number: str,
        user_id: Optional[str] = None
    ) -> Optional[VideoUpload]:
        """
        Get video upload by video file number
        
        Args:
            db: Database session
            video_file_number: Video file number (e.g., VF-2024-0001)
            user_id: Optional user ID to filter by user
        
        Returns:
            VideoUpload if found, None otherwise
        """
        query = select(VideoUpload).where(
            VideoUpload.video_file_number == video_file_number
        )
        
        if user_id:
            from uuid import UUID
            query = query.where(VideoUpload.user_id == UUID(user_id))
        
        result = await db.execute(query)
        return result.scalar_one_or_none()

