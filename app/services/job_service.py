from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from app.database import JobStatus
from app.utils.logger import logger
from datetime import datetime
from typing import Optional, Dict, Any


class JobService:
    @staticmethod
    async def create_job(db: AsyncSession, job_id: str, initial_status: Dict[str, Any]) -> JobStatus:
        """Create a new job status record"""
        job = JobStatus(
            job_id=job_id,
            **initial_status
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        logger.info("Job created", job_id=job_id, status=initial_status.get("status"))
        return job
    
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
        """Update job status"""
        updates["updated_at"] = datetime.utcnow()
        await db.execute(
            update(JobStatus)
            .where(JobStatus.job_id == job_id)
            .values(**updates)
        )
        await db.commit()
        
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

