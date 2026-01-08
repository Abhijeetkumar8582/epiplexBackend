from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from app.database import JobStatus
from app.utils.logger import logger
from datetime import datetime
from typing import Optional, Dict, Any


class JobService:
    @staticmethod
    async def create_job(db: AsyncSession, job_id: str, initial_status: Dict[str, Any]) -> JobStatus:
        """Create a new job status record, or return existing one if it already exists"""
        # Check if job already exists
        existing_job = await JobService.get_job(db, job_id)
        if existing_job:
            logger.info("Job already exists, returning existing job", job_id=job_id, status=existing_job.status)
            return existing_job
        
        # Create new job
        job = JobStatus(
            job_id=job_id,
            **initial_status
        )
        db.add(job)
        
        try:
            await db.commit()
            await db.refresh(job)
            logger.info("Job created", job_id=job_id, status=initial_status.get("status"))
            return job
        except IntegrityError as e:
            await db.rollback()
            # Job was created by another process (race condition), fetch and return it
            logger.info("Job was created by another process (race condition), fetching existing job", 
                       job_id=job_id, error_code=str(e.orig.args[0]) if hasattr(e, 'orig') and e.orig else None)
            existing_job = await JobService.get_job(db, job_id)
            if existing_job:
                return existing_job
            else:
                # If we still can't find it, log error and re-raise
                logger.error("Duplicate key error but job not found after rollback", job_id=job_id, error=str(e))
                raise
        except Exception as e:
            await db.rollback()
            # Some other error, re-raise it
            logger.error("Failed to create job", job_id=job_id, error=str(e), exc_info=True)
            raise
    
    @staticmethod
    async def get_job(db: AsyncSession, job_id: str) -> Optional[JobStatus]:
        """Get job status by ID"""
        result = await db.execute(select(JobStatus).where(JobStatus.job_id == job_id))
        return result.scalar_one_or_none()
    
    @staticmethod
    async def update_job(
        db: AsyncSession,
        job_id: str,
        updates: Dict[str, Any]
    ) -> Optional[JobStatus]:
        """Update job status - merges step_progress instead of replacing it"""
        from app.database import _is_sql_server
        
        # For SQL Server, we need to handle connection busy issues
        # Get current job to merge step_progress
        current_job = None
        try:
            current_job = await JobService.get_job(db, job_id)
            # For SQL Server, ensure result is fully consumed by accessing the data
            if current_job and _is_sql_server:
                # Access all attributes to ensure result is consumed
                _ = current_job.job_id
                _ = current_job.step_progress
        except Exception as e:
            logger.warning(f"Failed to get current job for merging, continuing without merge: {str(e)}")
        
        if current_job and "step_progress" in updates:
            # Merge step_progress instead of replacing
            current_step_progress = current_job.step_progress or {}
            new_step_progress = updates.get("step_progress", {})
            # Merge: new values override old ones, but keep existing steps
            merged_step_progress = {**current_step_progress, **new_step_progress}
            updates["step_progress"] = merged_step_progress
        
        updates["updated_at"] = datetime.utcnow()
        
        # For SQL Server, ensure any pending operations are flushed first
        # to avoid "connection is busy" errors
        try:
            if _is_sql_server:
                # Flush any pending operations before starting new query
                await db.flush()
                # Small delay to ensure connection is ready
                import asyncio
                await asyncio.sleep(0.01)
            
            await db.execute(
                update(JobStatus)
                .where(JobStatus.job_id == job_id)
                .values(**updates)
            )
            
            if _is_sql_server:
                # For SQL Server, flush before commit to avoid connection busy errors
                await db.flush()
                # Small delay before commit
                import asyncio
                await asyncio.sleep(0.01)
            
            await db.commit()
        except Exception as e:
            await db.rollback()
            # If it's a connection busy error, retry once after a short delay
            if _is_sql_server and "busy with results" in str(e).lower():
                logger.warning("Connection busy error detected, retrying after delay", job_id=job_id)
                import asyncio
                await asyncio.sleep(0.1)
                try:
                    await db.flush()
                    await db.execute(
                        update(JobStatus)
                        .where(JobStatus.job_id == job_id)
                        .values(**updates)
                    )
                    await db.flush()
                    await db.commit()
                    logger.info("Job update succeeded on retry", job_id=job_id)
                except Exception as retry_error:
                    await db.rollback()
                    logger.error(f"Failed to update job status on retry: {str(retry_error)}", job_id=job_id, exc_info=True)
                    raise
            else:
                logger.error(f"Failed to update job status: {str(e)}", job_id=job_id, exc_info=True)
                raise
        
        # Don't fetch again for SQL Server to avoid connection busy errors
        # Just return None or log the update
        if _is_sql_server:
            logger.info("Job updated", job_id=job_id, updates=updates)
            return None
        else:
            job = await JobService.get_job(db, job_id)
            if job:
                logger.info("Job updated", job_id=job_id, updates=updates)
            return job
    
    @staticmethod
    async def get_job_dict(db: AsyncSession, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job as dictionary"""
        job = await JobService.get_job(db, job_id)
        if not job:
            return None
        
        return {
            "status": job.status,
            "progress": job.progress,
            "message": job.message,
            "current_step": job.current_step,
            "step_progress": job.step_progress or {},
            "output_files": job.output_files,
            "transcript": job.transcript,
            "frame_analyses": job.frame_analyses or [],
            "error": job.error
        }

