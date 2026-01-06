"""High-performance frame analysis service with queues and threading"""
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from concurrent.futures import ThreadPoolExecutor
import time

from app.database import FrameAnalysis
from app.services.frame_extraction_service import FrameExtractionService
from app.services.gpt_service import GPTService
from app.utils.logger import logger


class FrameAnalysisService:
    """High-performance service for analyzing video frames"""
    
    def __init__(self, max_workers: int = 4):
        self.frame_extractor = FrameExtractionService()
        self.gpt_service = GPTService()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_video_frames(
        self,
        db: AsyncSession,
        video_id: UUID,
        video_path: str,
        frames_dir: Path,
        frames_per_second: float = 0.5
    ) -> List[FrameAnalysis]:
        """
        High-performance frame processing pipeline:
        1. Extract frames (async)
        2. Analyze frames in parallel (threading)
        3. Batch insert to database
        
        Args:
            db: Database session
            video_id: Video upload ID
            video_path: Path to video file
            frames_dir: Directory to save frames
            frames_per_second: Frames to extract per second
        
        Returns:
            List of FrameAnalysis objects
        """
        start_time = time.time()
        
        try:
            # Get user_id from video upload to use custom prompt
            from app.database import VideoUpload
            video_upload_result = await db.execute(
                select(VideoUpload.user_id).where(VideoUpload.id == video_id)
            )
            user_id = video_upload_result.scalar_one_or_none()
            
            # Step 1: Extract frames (async, non-blocking)
            logger.info("Starting frame extraction", video_id=str(video_id))
            frames = await self.frame_extractor.extract_frames_async(
                video_path=video_path,
                output_dir=frames_dir,
                video_id=str(video_id),
                frames_per_second=frames_per_second
            )
            
            if not frames:
                logger.warning("No frames extracted", video_id=str(video_id))
                return []
            
            logger.info("Frames extracted", 
                       video_id=str(video_id), 
                       frame_count=len(frames))
            
            # Step 2: Analyze frames in parallel (async) with batching
            from app.config import settings
            logger.info("Starting frame analysis with GPT", 
                       video_id=str(video_id),
                       frame_count=len(frames),
                       batch_size=getattr(settings, 'GPT_BATCH_SIZE', 10),
                       user_id=str(user_id) if user_id else None)
            
            # Use async batch analysis with GPT service (pass user_id for custom prompt)
            analyzed_frames = await self.gpt_service.batch_analyze_frames(
                frames=frames,
                max_workers=4,
                batch_size=getattr(settings, 'GPT_BATCH_SIZE', 10),
                user_id=str(user_id) if user_id else None,
                db=db
            )
            
            logger.info("Frame analysis completed",
                       video_id=str(video_id),
                       analyzed_count=len(analyzed_frames))
            
            # Step 3: Batch insert to database with full GPT response
            frame_analyses = []
            for frame_data in analyzed_frames:
                # Store full GPT response as JSONB
                gpt_response = {
                    "description": frame_data.get("description"),
                    "ocr_text": frame_data.get("ocr_text"),
                    "meta_tags": frame_data.get("meta_tags"),  # Include meta_tags
                    "processing_time_ms": frame_data.get("processing_time_ms"),
                    "timestamp": frame_data.get("timestamp"),
                    "frame_number": frame_data.get("frame_number"),
                    "image_path": frame_data.get("image_path")
                }
                
                frame_analysis = FrameAnalysis(
                    video_id=video_id,
                    timestamp=frame_data["timestamp"],
                    frame_number=frame_data.get("frame_number"),
                    image_path=frame_data["image_path"],
                    description=frame_data.get("description"),
                    ocr_text=frame_data.get("ocr_text"),
                    gpt_response=gpt_response,  # Store full GPT response
                    processing_time_ms=frame_data.get("processing_time_ms")
                )
                frame_analyses.append(frame_analysis)
                db.add(frame_analysis)
            
            # Batch commit
            await db.commit()
            
            # Refresh all objects
            for fa in frame_analyses:
                await db.refresh(fa)
            
            total_time = time.time() - start_time
            logger.info("Frame analysis pipeline completed",
                       video_id=str(video_id),
                       total_frames=len(frame_analyses),
                       total_time_seconds=round(total_time, 2))
            
            return frame_analyses
            
        except Exception as e:
            logger.error("Frame analysis pipeline failed",
                        video_id=str(video_id),
                        error=str(e),
                        exc_info=True)
            await db.rollback()
            raise
    
    async def get_video_frames(
        self,
        db: AsyncSession,
        video_id: UUID,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[FrameAnalysis]:
        """Get frame analyses for a video"""
        query = select(FrameAnalysis).where(
            FrameAnalysis.video_id == video_id
        ).order_by(FrameAnalysis.timestamp)
        
        if limit:
            query = query.limit(limit).offset(offset)
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    async def get_frame_count(
        self,
        db: AsyncSession,
        video_id: UUID
    ) -> int:
        """Get total frame count for a video"""
        from sqlalchemy import func
        query = select(func.count(FrameAnalysis.id)).where(
            FrameAnalysis.video_id == video_id
        )
        result = await db.execute(query)
        return result.scalar_one() or 0
    
    async def get_gpt_responses_by_video_file_number(
        self,
        db: AsyncSession,
        video_file_number: str,
        user_id: Optional[UUID] = None
    ) -> List[FrameAnalysis]:
        """
        Get all GPT responses for a video by video file number
        
        Args:
            db: Database session
            video_file_number: Video file number (e.g., VF-2024-0001)
            user_id: Optional user ID to filter by user
        
        Returns:
            List of FrameAnalysis objects with GPT responses
        """
        from app.database import VideoUpload
        from app.services.video_file_number_service import VideoFileNumberService
        
        # Get video upload by file number
        upload = await VideoFileNumberService.get_upload_by_file_number(
            db, video_file_number, str(user_id) if user_id else None
        )
        
        if not upload:
            return []
        
        # Get all frame analyses for this video
        query = select(FrameAnalysis).where(
            FrameAnalysis.video_id == upload.id
        ).order_by(FrameAnalysis.timestamp)
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    async def get_complete_document_data(
        self,
        db: AsyncSession,
        video_file_number: str,
        user_id: Optional[UUID] = None,
        include_base64_images: bool = True
    ) -> Dict[str, Any]:
        """
        Get complete document data for a video file number
        Includes all frame analyses, GPT responses, and video metadata
        
        Args:
            db: Database session
            video_file_number: Video file number (e.g., VF-2024-0001)
            user_id: Optional user ID to filter by user
        
        Returns:
            Dictionary with complete document data
        """
        from app.database import VideoUpload
        from app.services.video_file_number_service import VideoFileNumberService
        from sqlalchemy import func
        
        # Get video upload by file number
        upload = await VideoFileNumberService.get_upload_by_file_number(
            db, video_file_number, str(user_id) if user_id else None
        )
        
        if not upload:
            return None
        
        # Calculate statistics using SQL aggregations (much faster than Python loops)
        stats_query = select(
            func.count(FrameAnalysis.id).label('total_frames'),
            func.count(FrameAnalysis.gpt_response).label('frames_with_gpt'),
            func.count(FrameAnalysis.ocr_text).label('frames_with_ocr'),
            func.count(FrameAnalysis.description).label('frames_with_description'),
            func.avg(FrameAnalysis.processing_time_ms).label('avg_processing_time'),
            func.min(FrameAnalysis.timestamp).label('first_frame_timestamp'),
            func.max(FrameAnalysis.timestamp).label('last_frame_timestamp')
        ).where(FrameAnalysis.video_id == upload.id)
        
        stats_result = await db.execute(stats_query)
        stats = stats_result.first()
        
        total_frames = stats.total_frames or 0
        frames_with_gpt = stats.frames_with_gpt or 0
        frames_with_ocr = stats.frames_with_ocr or 0
        frames_with_description = stats.frames_with_description or 0
        avg_processing_time = float(stats.avg_processing_time or 0)
        first_frame_timestamp = stats.first_frame_timestamp
        last_frame_timestamp = stats.last_frame_timestamp
        
        # Get all frame analyses for this video
        # Optimize: Use defer() to exclude base64_image from query when not needed
        # This avoids loading large base64 data from database entirely
        if include_base64_images:
            frames_query = select(FrameAnalysis).where(
                FrameAnalysis.video_id == upload.id
            ).order_by(FrameAnalysis.timestamp)
        else:
            # Use defer to exclude base64_image column from loading
            # This significantly reduces database transfer time and memory usage
            from sqlalchemy.orm import defer
            frames_query = select(FrameAnalysis).options(
                defer(FrameAnalysis.base64_image)
            ).where(
                FrameAnalysis.video_id == upload.id
            ).order_by(FrameAnalysis.timestamp)
        
        frames_result = await db.execute(frames_query)
        frames = list(frames_result.scalars().all())
        
        # Ensure base64_image is None when not included (defer sets it to None)
        if not include_base64_images:
            for frame in frames:
                frame.base64_image = None
        
        return {
            "video_file_number": video_file_number,
            "video_metadata": upload,
            "total_frames": total_frames,
            "frames_with_gpt": frames_with_gpt,
            "frames": frames,
            "summary": {
                "total_frames": total_frames,
                "frames_with_gpt": frames_with_gpt,
                "frames_with_ocr": frames_with_ocr,
                "frames_with_description": frames_with_description,
                "avg_processing_time_ms": round(avg_processing_time, 2),
                "first_frame_timestamp": first_frame_timestamp,
                "last_frame_timestamp": last_frame_timestamp,
                "video_duration_seconds": upload.video_length_seconds,
                "processing_complete": frames_with_gpt == total_frames if total_frames > 0 else False
            }
        }

