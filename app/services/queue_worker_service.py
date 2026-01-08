"""Queue worker service for processing videos with 'uploaded' status"""
import asyncio
from pathlib import Path
from typing import Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, case
from sqlalchemy.orm import selectinload

from app.database import AsyncSessionLocal, VideoUpload
from app.services.video_upload_service import VideoUploadService
from app.services.job_service import JobService
from app.utils.logger import logger
from app.config import settings


class QueueWorkerService:
    """Background worker that processes videos in queue"""
    
    def __init__(self, poll_interval: int = 5):
        """
        Initialize queue worker
        
        Args:
            poll_interval: Seconds between queue polls (legacy parameter, kept for backward compatibility)
        """
        self.poll_interval = poll_interval  # Legacy, not used in new implementation
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self._processing_lock = asyncio.Lock()  # Lock to ensure only one video processes at a time
        
        # Production queue worker settings
        self.processing_delay = getattr(settings, 'QUEUE_PROCESSING_DELAY', 60)  # 1 minute between videos
        self.crawler_interval = getattr(settings, 'QUEUE_CRAWLER_INTERVAL', 1800)  # 30 minutes when queue empty
    
    async def get_next_video_in_queue(self, db: AsyncSession) -> Optional[VideoUpload]:
        """
        Get next video in queue with status 'uploaded' (strictly)
        Priority order: high priority first, then by created_at (oldest first)
        
        Returns:
            VideoUpload object or None if queue is empty
        """
        try:
            # Query ONLY for videos with status 'uploaded' (strictly)
            # No recovery of stuck processing videos - production behavior
            query = select(VideoUpload).where(
                and_(
                    VideoUpload.status == "uploaded",
                    VideoUpload.is_deleted == False
                )
            ).order_by(
                # High priority first, then by created_at (oldest first)
                # Use CASE to handle NULLs for SQL Server compatibility (NULLs last)
                # Non-NULL priorities get 0 (sorted first), NULLs get 1 (sorted last)
                case((VideoUpload.priority.is_(None), 1), else_=0).asc(),
                VideoUpload.priority.desc(),  # 'high' > 'normal' > NULL
                VideoUpload.created_at.asc()  # Oldest first
            ).limit(1)
            
            result = await db.execute(query)
            video = result.scalar_one_or_none()
            
            if video:
                logger.info("Found video in queue",
                           upload_id=str(video.id),
                           video_file_number=video.video_file_number,
                           priority=video.priority or "normal",
                           created_at=str(video.created_at),
                           status=video.status)
            
            return video
        except Exception as e:
            logger.error("Error getting next video from queue", error=str(e), exc_info=True)
            return None
    
    async def cleanup_video_file(self, video: VideoUpload) -> bool:
        """
        Delete video file from local storage after processing completes
        
        Args:
            video: VideoUpload object with completed processing
            
        Returns:
            True if file was deleted successfully, False otherwise
        """
        try:
            # Handle both relative and absolute paths
            video_url = video.video_url
            if not video_url:
                logger.debug("Video URL is empty, skipping deletion", upload_id=str(video.id))
                return False
            
            # Check if it's a URL (starts with http:// or https://)
            if video_url.startswith(('http://', 'https://')):
                logger.debug("Video URL is a remote URL, skipping deletion", upload_id=str(video.id))
                return False
            
            # Resolve path - handle both relative and absolute paths
            video_file_path = Path(video_url)
            if not video_file_path.is_absolute():
                # If relative, resolve against UPLOAD_DIR from settings
                upload_dir = Path(settings.UPLOAD_DIR)
                video_file_path = upload_dir / video_file_path
            
            # Only delete if it's a local file (not a URL)
            if video_file_path.exists() and video_file_path.is_file():
                try:
                    video_file_path.unlink()
                    logger.info("Video file deleted after processing",
                               upload_id=str(video.id),
                               file_path=str(video_file_path))
                    return True
                except Exception as e:
                    logger.error("Failed to delete video file",
                                upload_id=str(video.id),
                                file_path=str(video_file_path),
                                error=str(e))
                    return False
            else:
                # File doesn't exist or is not a local file (might be URL)
                logger.debug("Video file not found or is not a local file, skipping deletion",
                            upload_id=str(video.id),
                            file_path=str(video_file_path))
                return False
        except Exception as e:
            logger.error("Error in cleanup_video_file",
                        upload_id=str(video.id),
                        error=str(e),
                        exc_info=True)
            return False
    
    async def process_video_from_queue(self, video: VideoUpload) -> bool:
        """
        Process a single video from the queue
        
        Args:
            video: VideoUpload object to process
            
        Returns:
            True if processing completed successfully, False if failed
        """
        from app.main import process_video_task
        
        # Defensive check: Skip if video status is not 'uploaded'
        if video.status != "uploaded":
            logger.warning("Skipping video - status is not 'uploaded'",
                          upload_id=str(video.id),
                          current_status=video.status)
            return False
        
        # Defensive check: Skip if video is already completed (prevent reprocessing)
        if video.status == "completed":
            logger.warning("Skipping video - already completed",
                          upload_id=str(video.id))
            return False
        
        job_id = None
        try:
            # Generate job_id if not exists
            if not video.job_id:
                import uuid
                job_id = str(uuid.uuid4())
            else:
                job_id = video.job_id
            
            # Update status to 'processing'
            async with AsyncSessionLocal() as db:
                await VideoUploadService.update_upload_status(
                    db, video.id, "processing", job_id
                )
                
                # Initialize job status
                initial_status = {
                    "status": "processing",
                    "progress": 0,
                    "message": "Video picked from queue, starting processing...",
                    "current_step": "upload",
                    "error": None,  # Clear any previous error
                    "step_progress": {
                        "upload": "completed",
                        "extract_audio": "pending",
                        "transcribe": "pending",
                        "extract_frames": "pending",
                        "analyze_frames": "pending",
                        "summary_generation": "pending",
                        "generate_pdf": "pending",
                        "complete": "pending"
                    }
                }
                job = await JobService.create_job(db, job_id, initial_status)
                
                # If job already existed with failed status, reset it for retry
                if job.status == "failed":
                    logger.info("Resetting failed job for retry", job_id=job_id, upload_id=str(video.id))
                    await JobService.update_job(db, job_id, initial_status)
            
            # Start processing task
            logger.info("Starting video processing from queue",
                       upload_id=str(video.id),
                       video_file_number=video.video_file_number,
                       job_id=job_id,
                       file_path=video.video_url)
            
            # Acquire lock to ensure only one video processes at a time
            # This prevents resource conflicts when multiple videos are queued
            async with self._processing_lock:
                # Run processing task with timeout protection
                # Wrap in try-except to catch any errors and ensure we move to next video
                try:
                    await process_video_task(video.video_url, job_id, str(video.id))
                    
                    # Check final status to confirm success
                    async with AsyncSessionLocal() as db:
                        from app.database import VideoUpload
                        from sqlalchemy import select
                        result = await db.execute(
                            select(VideoUpload.status).where(VideoUpload.id == video.id)
                        )
                        final_status = result.scalar_one_or_none()
                        
                        if final_status == "completed":
                            logger.info("Video processing completed successfully from queue",
                                       upload_id=str(video.id),
                                       job_id=job_id)
                            
                            # Cleanup: Delete video file after successful processing
                            try:
                                # Refresh video object to get latest data
                                result = await db.execute(
                                    select(VideoUpload).where(VideoUpload.id == video.id)
                                )
                                completed_video = result.scalar_one_or_none()
                                if completed_video:
                                    await self.cleanup_video_file(completed_video)
                            except Exception as cleanup_error:
                                logger.error("Error during file cleanup after processing",
                                           upload_id=str(video.id),
                                           error=str(cleanup_error))
                                # Don't fail the whole operation if cleanup fails
                            
                            return True
                        elif final_status == "failed":
                            logger.warning("Video processing failed (marked as failed)",
                                          upload_id=str(video.id),
                                          job_id=job_id)
                            return False
                        else:
                            # Still processing or unknown status - log warning
                            logger.warning("Video processing status unclear after task completion",
                                          upload_id=str(video.id),
                                          job_id=job_id,
                                          status=final_status)
                            return False
                            
                except Exception as processing_error:
                    # Processing task itself raised an exception
                    logger.error("Processing task raised exception",
                               upload_id=str(video.id),
                               job_id=job_id,
                               error=str(processing_error),
                               exc_info=True)
                    
                    # Mark as failed
                    try:
                        async with AsyncSessionLocal() as db:
                            await VideoUploadService.update_upload_status(
                                db, video.id, "failed", job_id, error=str(processing_error)
                            )
                            # Also update job status
                            await JobService.update_job(db, job_id, {
                                "status": "failed",
                                "message": f"Processing failed: {str(processing_error)}",
                                "error": str(processing_error)
                            })
                    except Exception as update_error:
                        logger.error("Failed to update video status after processing error",
                                   upload_id=str(video.id),
                                   error=str(update_error))
                    
                    return False
            
        except Exception as e:
            # Outer exception - something went wrong before or during processing setup
            logger.error("Error processing video from queue",
                        upload_id=str(video.id),
                        job_id=job_id,
                        error=str(e),
                        exc_info=True)
            
            # Mark as failed
            try:
                async with AsyncSessionLocal() as db:
                    await VideoUploadService.update_upload_status(
                        db, video.id, "failed", job_id, error=str(e)
                    )
                    # Also update job status if job_id exists
                    if job_id:
                        try:
                            await JobService.update_job(db, job_id, {
                                "status": "failed",
                                "message": f"Processing failed: {str(e)}",
                                "error": str(e)
                            })
                        except Exception:
                            pass  # Job might not exist yet
            except Exception as update_error:
                logger.error("Failed to update video status to failed",
                            upload_id=str(video.id),
                            error=str(update_error))
            
            return False
    
    async def worker_loop(self):
        """
        Main worker loop that polls queue and processes videos
        Implements two-tier polling:
        - Active polling: When queue has items, process one video, wait 1 minute, check again
        - Crawler polling: When queue is empty, wait 30 minutes before next check
        """
        logger.info("Queue worker started (production mode)",
                   processing_delay=self.processing_delay,
                   crawler_interval=self.crawler_interval)
        
        while self.is_running:
            try:
                async with AsyncSessionLocal() as db:
                    # Get next video in queue
                    video = await self.get_next_video_in_queue(db)
                    
                    if video:
                        # Queue has items - active polling mode
                        logger.info("Processing video from queue",
                                   upload_id=str(video.id),
                                   video_file_number=video.video_file_number)
                        
                        try:
                            # Process the video (this will run synchronously)
                            # We process one at a time to avoid resource conflicts
                            success = await self.process_video_from_queue(video)
                            
                            if success:
                                logger.info("Video processing completed from queue",
                                           upload_id=str(video.id))
                            else:
                                logger.warning("Video processing failed from queue - moving to next",
                                             upload_id=str(video.id))
                        except Exception as process_error:
                            # Even if process_video_from_queue fails to catch something, we continue
                            logger.error("Unexpected error in process_video_from_queue - moving to next video",
                                       upload_id=str(video.id),
                                       error=str(process_error),
                                       exc_info=True)
                            
                            # Try to mark as failed as a last resort
                            try:
                                async with AsyncSessionLocal() as error_db:
                                    await VideoUploadService.update_upload_status(
                                        error_db, video.id, "failed", video.job_id, 
                                        error=f"Unexpected error: {str(process_error)}"
                                    )
                            except Exception:
                                pass  # Best effort - continue anyway
                        
                        # Wait 1 minute before processing next video (active polling delay)
                        logger.info(f"Waiting {self.processing_delay} seconds before checking next video")
                        await asyncio.sleep(self.processing_delay)
                    else:
                        # Queue is empty - crawler mode
                        logger.info(f"Queue is empty, waiting {self.crawler_interval} seconds (crawler interval) before next check")
                        await asyncio.sleep(self.crawler_interval)
                
            except asyncio.CancelledError:
                # Worker is being stopped, exit gracefully
                logger.info("Queue worker loop cancelled")
                break
            except Exception as e:
                logger.error("Error in queue worker loop", error=str(e), exc_info=True)
                # Wait before retrying (use crawler interval for errors to avoid tight loops)
                # Add exponential backoff for persistent errors (max 5 minutes)
                await asyncio.sleep(min(self.crawler_interval, 300))  # Max 5 minutes on error
    
    def start(self):
        """Start the queue worker"""
        if self.is_running:
            logger.warning("Queue worker is already running")
            return
        
        self.is_running = True
        self._task = asyncio.create_task(self.worker_loop())
        logger.info("Queue worker started")
    
    def stop(self):
        """Stop the queue worker gracefully"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self._task:
            self._task.cancel()
        logger.info("Queue worker stopped")
    
    async def get_queue_stats(self, db: AsyncSession) -> dict:
        """
        Get queue statistics
        
        Returns:
            Dictionary with queue statistics
        """
        try:
            # Count videos by status
            from sqlalchemy import func
            
            stats_query = select(
                VideoUpload.status,
                func.count(VideoUpload.id).label('count')
            ).where(
                VideoUpload.is_deleted == False
            ).group_by(VideoUpload.status)
            
            result = await db.execute(stats_query)
            stats = {row.status: row.count for row in result.all()}
            
            return {
                "queue_size": stats.get("uploaded", 0),
                "processing": stats.get("processing", 0),
                "completed": stats.get("completed", 0),
                "failed": stats.get("failed", 0),
                "total": sum(stats.values()),
                "worker_running": self.is_running
            }
        except Exception as e:
            logger.error("Error getting queue stats", error=str(e), exc_info=True)
            return {
                "queue_size": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "total": 0,
                "worker_running": self.is_running,
                "error": str(e)
            }


# Global queue worker instance
queue_worker = QueueWorkerService(poll_interval=getattr(settings, 'QUEUE_POLL_INTERVAL', 5))

