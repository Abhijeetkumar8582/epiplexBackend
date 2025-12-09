from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime

from app.database import VideoUpload
from app.utils.logger import logger


class VideoUploadService:
    @staticmethod
    async def create_upload(
        db: AsyncSession,
        user_id: UUID,
        name: str,
        source_type: str,
        video_url: str,
        original_input: str,
        status: str = "uploaded",
        job_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        application_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        language_code: Optional[str] = None,
        priority: Optional[str] = "normal"
    ) -> VideoUpload:
        """Create a new video upload record with unique video file number"""
        # Generate unique video file number
        video_file_number = await VideoFileNumberService.generate_video_file_number(db)
        
        upload = VideoUpload(
            user_id=user_id,
            name=name,
            source_type=source_type,
            video_url=video_url,
            original_input=original_input,
            status=status,
            job_id=job_id,
            video_file_number=video_file_number,
            video_length_seconds=metadata.get("video_length_seconds") if metadata else None,
            video_size_bytes=metadata.get("video_size_bytes") if metadata else None,
            mime_type=metadata.get("mime_type") if metadata else None,
            resolution_width=metadata.get("resolution_width") if metadata else None,
            resolution_height=metadata.get("resolution_height") if metadata else None,
            fps=metadata.get("fps") if metadata else None,
            application_name=application_name,
            tags=tags,  # JSONB will handle list conversion
            language_code=language_code,
            priority=priority or "normal",
            is_deleted=False
        )
        
        db.add(upload)
        await db.commit()
        await db.refresh(upload)
        
        logger.info("Video upload created", 
                   upload_id=str(upload.id), 
                   video_file_number=video_file_number,
                   user_id=str(user_id), 
                   name=name)
        return upload
    
    @staticmethod
    async def get_upload(
        db: AsyncSession,
        upload_id: UUID,
        user_id: Optional[UUID] = None
    ) -> Optional[VideoUpload]:
        """Get video upload by ID, optionally filtered by user_id"""
        query = select(VideoUpload).where(VideoUpload.id == upload_id)
        
        if user_id:
            query = query.where(VideoUpload.user_id == user_id)
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_user_uploads(
        db: AsyncSession,
        user_id: UUID,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        include_deleted: bool = False,
        application_name: Optional[str] = None,
        language_code: Optional[str] = None,
        priority: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Tuple[List[VideoUpload], int]:
        """Get paginated list of user's video uploads"""
        # Build query
        query = select(VideoUpload).where(VideoUpload.user_id == user_id)
        
        # Filter out deleted items by default
        if not include_deleted:
            query = query.where(VideoUpload.is_deleted == False)
        
        if status:
            query = query.where(VideoUpload.status == status)
        
        if application_name:
            query = query.where(VideoUpload.application_name == application_name)
        
        if language_code:
            query = query.where(VideoUpload.language_code == language_code)
        
        if priority:
            query = query.where(VideoUpload.priority == priority)
        
        if tags:
            # Search for videos that contain any of the specified tags
            # Using JSONB containment operator
            from sqlalchemy import text
            for tag in tags:
                query = query.where(VideoUpload.tags.contains([tag]))
        
        # Get total count
        count_query = select(func.count()).select_from(VideoUpload).where(VideoUpload.user_id == user_id)
        if not include_deleted:
            count_query = count_query.where(VideoUpload.is_deleted == False)
        if status:
            count_query = count_query.where(VideoUpload.status == status)
        if application_name:
            count_query = count_query.where(VideoUpload.application_name == application_name)
        if language_code:
            count_query = count_query.where(VideoUpload.language_code == language_code)
        if priority:
            count_query = count_query.where(VideoUpload.priority == priority)
        if tags:
            for tag in tags:
                count_query = count_query.where(VideoUpload.tags.contains([tag]))
        
        total_result = await db.execute(count_query)
        total = total_result.scalar_one()
        
        # Apply pagination and ordering
        query = query.order_by(desc(VideoUpload.created_at))
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await db.execute(query)
        uploads = list(result.scalars().all())
        
        return uploads, total
    
    @staticmethod
    async def get_user_uploads_with_stats(
        db: AsyncSession,
        user_id: UUID,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        include_deleted: bool = False,
        application_name: Optional[str] = None,
        language_code: Optional[str] = None,
        priority: Optional[str] = None,
        tags: Optional[List[str]] = None,
        sort_by: str = "updated_at",
        sort_order: str = "desc"
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get user uploads with frame analysis statistics
        
        Returns:
            Tuple of (list of video dicts with stats, total count)
        """
        # Get uploads with sorting support
        # We need to call get_user_uploads but with sorting
        # Since get_user_uploads doesn't support sort_by/sort_order yet, we'll build the query here
        query = select(VideoUpload).where(VideoUpload.user_id == user_id)
        
        # Filter out deleted items by default
        if not include_deleted:
            query = query.where(VideoUpload.is_deleted == False)
        
        if status:
            query = query.where(VideoUpload.status == status)
        
        if application_name:
            query = query.where(VideoUpload.application_name == application_name)
        
        if language_code:
            query = query.where(VideoUpload.language_code == language_code)
        
        if priority:
            query = query.where(VideoUpload.priority == priority)
        
        if tags:
            for tag in tags:
                query = query.where(VideoUpload.tags.contains([tag]))
        
        # Get total count
        count_query = select(func.count()).select_from(VideoUpload).where(VideoUpload.user_id == user_id)
        if not include_deleted:
            count_query = count_query.where(VideoUpload.is_deleted == False)
        if status:
            count_query = count_query.where(VideoUpload.status == status)
        if application_name:
            count_query = count_query.where(VideoUpload.application_name == application_name)
        if language_code:
            count_query = count_query.where(VideoUpload.language_code == language_code)
        if priority:
            count_query = count_query.where(VideoUpload.priority == priority)
        if tags:
            for tag in tags:
                count_query = count_query.where(VideoUpload.tags.contains([tag]))
        
        total_result = await db.execute(count_query)
        total = total_result.scalar_one()
        
        # Apply sorting
        if sort_by == "updated_at":
            order_col = VideoUpload.updated_at
        elif sort_by == "created_at":
            order_col = VideoUpload.created_at
        elif sort_by == "name":
            order_col = VideoUpload.name
        elif sort_by == "status":
            order_col = VideoUpload.status
        else:
            order_col = VideoUpload.updated_at
        
        if sort_order == "desc":
            query = query.order_by(desc(order_col))
        else:
            query = query.order_by(order_col)
        
        # Apply pagination
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await db.execute(query)
        uploads = list(result.scalars().all())
        
        # Get frame stats for each upload
        video_ids = [upload.id for upload in uploads]
        
        if video_ids:
            # Query frame counts per video
            frame_stats_query = select(
                FrameAnalysis.video_id,
                func.count(FrameAnalysis.id).label('total_frames'),
                func.count(FrameAnalysis.id).filter(
                    FrameAnalysis.gpt_response.isnot(None)
                ).label('frames_with_gpt')
            ).where(
                FrameAnalysis.video_id.in_(video_ids)
            ).group_by(FrameAnalysis.video_id)
            
            frame_stats_result = await db.execute(frame_stats_query)
            frame_stats = {
                row.video_id: {
                    'total_frames': row.total_frames,
                    'frames_with_gpt': row.frames_with_gpt
                }
                for row in frame_stats_result.all()
            }
        else:
            frame_stats = {}
        
        # Combine upload data with stats
        videos_with_stats = []
        for upload in uploads:
            stats = frame_stats.get(upload.id, {'total_frames': 0, 'frames_with_gpt': 0})
            
            video_dict = {
                'id': upload.id,
                'video_file_number': upload.video_file_number,
                'name': upload.name,
                'status': upload.status,
                'created_at': upload.created_at,
                'updated_at': upload.updated_at,
                'last_activity': upload.updated_at,  # Use updated_at as last activity
                'video_length_seconds': upload.video_length_seconds,
                'video_size_bytes': upload.video_size_bytes,
                'application_name': upload.application_name,
                'tags': upload.tags,
                'language_code': upload.language_code,
                'priority': upload.priority,
                'total_frames': stats['total_frames'],
                'frames_with_gpt': stats['frames_with_gpt']
            }
            videos_with_stats.append(video_dict)
        
        return videos_with_stats, total
    
    @staticmethod
    async def update_upload(
        db: AsyncSession,
        upload_id: UUID,
        updates: Dict[str, Any],
        user_id: Optional[UUID] = None
    ) -> Optional[VideoUpload]:
        """Update video upload"""
        upload = await VideoUploadService.get_upload(db, upload_id, user_id)
        
        if not upload:
            return None
        
        # Update fields
        for key, value in updates.items():
            if hasattr(upload, key):
                setattr(upload, key, value)
        
        await db.commit()
        await db.refresh(upload)
        
        logger.info("Video upload updated", upload_id=str(upload_id), updates=updates)
        return upload
    
    @staticmethod
    async def update_upload_status(
        db: AsyncSession,
        upload_id: UUID,
        status: str,
        job_id: Optional[str] = None
    ) -> Optional[VideoUpload]:
        """Update upload status"""
        updates = {"status": status}
        if job_id:
            updates["job_id"] = job_id
        
        return await VideoUploadService.update_upload(db, upload_id, updates)
    
    @staticmethod
    async def soft_delete_upload(
        db: AsyncSession,
        upload_id: UUID,
        user_id: UUID
    ) -> bool:
        """Soft delete video upload (only by owner)"""
        upload = await VideoUploadService.get_upload(db, upload_id, user_id)
        
        if not upload:
            return False
        
        upload.is_deleted = True
        upload.deleted_at = datetime.utcnow()
        await db.commit()
        await db.refresh(upload)
        
        logger.info("Video upload soft deleted", upload_id=str(upload_id), user_id=str(user_id))
        return True
    
    @staticmethod
    async def restore_upload(
        db: AsyncSession,
        upload_id: UUID,
        user_id: UUID
    ) -> bool:
        """Restore soft-deleted video upload"""
        upload = await VideoUploadService.get_upload(db, upload_id, user_id)
        
        if not upload:
            return False
        
        upload.is_deleted = False
        upload.deleted_at = None
        await db.commit()
        await db.refresh(upload)
        
        logger.info("Video upload restored", upload_id=str(upload_id), user_id=str(user_id))
        return True
    
    @staticmethod
    async def hard_delete_upload(
        db: AsyncSession,
        upload_id: UUID,
        user_id: UUID
    ) -> bool:
        """Permanently delete video upload (only by owner)"""
        upload = await VideoUploadService.get_upload(db, upload_id, user_id)
        
        if not upload:
            return False
        
        await db.delete(upload)
        await db.commit()
        
        logger.info("Video upload permanently deleted", upload_id=str(upload_id), user_id=str(user_id))
        return True

