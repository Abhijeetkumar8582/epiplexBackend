from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func, and_
from typing import Optional, Dict, Any, List, Tuple
from uuid import UUID
from datetime import datetime, timedelta, timezone

from app.database import UserActivityLog, AsyncSessionLocal
from app.utils.logger import logger


class ActivityService:
    @staticmethod
    async def log_activity(
        db: Optional[AsyncSession],
        user_id: UUID,
        action: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ) -> Optional[UserActivityLog]:
        """
        Log a user activity
        
        Uses a separate session to avoid transaction conflicts.
        If the provided db session is used and fails, it won't affect the main transaction.
        """
        # Use a separate session to avoid transaction conflicts
        async with AsyncSessionLocal() as activity_db:
            try:
                # Always set current timestamp explicitly
                current_timestamp = datetime.now(timezone.utc)
                
                activity = UserActivityLog(
                    user_id=user_id,
                    action=action,
                    description=description,
                    activity_metadata=metadata,
                    ip_address=ip_address,
                    created_at=current_timestamp
                )
                
                activity_db.add(activity)
                await activity_db.commit()
                await activity_db.refresh(activity)
                
                logger.info("Activity logged", user_id=str(user_id), action=action)
                return activity
            except Exception as e:
                await activity_db.rollback()
                # Log the error but don't fail - activity logging is non-critical
                logger.warning("Failed to log activity", 
                             user_id=str(user_id), 
                             action=action, 
                             error=str(e))
                return None
    
    @staticmethod
    async def get_user_activities(
        db: AsyncSession,
        user_id: UUID,
        limit: int = 20,
        offset: int = 0
    ) -> list[UserActivityLog]:
        """Get user activity logs"""
        result = await db.execute(
            select(UserActivityLog)
            .where(UserActivityLog.user_id == user_id)
            .order_by(desc(UserActivityLog.created_at))
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())
    
    @staticmethod
    async def get_activities_by_action(
        db: AsyncSession,
        action: str,
        limit: int = 100
    ) -> list[UserActivityLog]:
        """Get activities by action type"""
        result = await db.execute(
            select(UserActivityLog)
            .where(UserActivityLog.action == action)
            .order_by(desc(UserActivityLog.created_at))
            .limit(limit)
        )
        return list(result.scalars().all())
    
    @staticmethod
    async def get_activity_by_id(
        db: AsyncSession,
        log_id: int,
        user_id: Optional[UUID] = None
    ) -> Optional[UserActivityLog]:
        """Get activity log by ID, optionally filtered by user_id"""
        query = select(UserActivityLog).where(UserActivityLog.id == log_id)
        
        if user_id:
            query = query.where(UserActivityLog.user_id == user_id)
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_user_activities_with_filters(
        db: AsyncSession,
        user_id: UUID,
        page: int = 1,
        page_size: int = 20,
        action: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        search: Optional[str] = None
    ) -> Tuple[List[UserActivityLog], int]:
        """
        Get paginated user activity logs with filtering
        
        Returns:
            Tuple of (logs list, total count)
        """
        # Build query
        query = select(UserActivityLog).where(UserActivityLog.user_id == user_id)
        count_query = select(func.count(UserActivityLog.id)).where(UserActivityLog.user_id == user_id)
        
        # Apply filters
        if action:
            query = query.where(UserActivityLog.action == action)
            count_query = count_query.where(UserActivityLog.action == action)
        
        if start_date:
            query = query.where(UserActivityLog.created_at >= start_date)
            count_query = count_query.where(UserActivityLog.created_at >= start_date)
        
        if end_date:
            query = query.where(UserActivityLog.created_at <= end_date)
            count_query = count_query.where(UserActivityLog.created_at <= end_date)
        
        if search:
            # Search in description
            search_pattern = f"%{search}%"
            query = query.where(UserActivityLog.description.ilike(search_pattern))
            count_query = count_query.where(UserActivityLog.description.ilike(search_pattern))
        
        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar_one() or 0
        
        # Apply pagination and ordering
        query = query.order_by(desc(UserActivityLog.created_at))
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await db.execute(query)
        logs = list(result.scalars().all())
        
        return logs, total
    
    @staticmethod
    async def get_activity_stats(
        db: AsyncSession,
        user_id: UUID,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get activity statistics for a user"""
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Total activities
        total_query = select(func.count(UserActivityLog.id)).where(
            and_(
                UserActivityLog.user_id == user_id,
                UserActivityLog.created_at >= start_date
            )
        )
        total_result = await db.execute(total_query)
        total = total_result.scalar_one() or 0
        
        # Activities by action
        action_query = select(
            UserActivityLog.action,
            func.count(UserActivityLog.id).label('count')
        ).where(
            and_(
                UserActivityLog.user_id == user_id,
                UserActivityLog.created_at >= start_date
            )
        ).group_by(UserActivityLog.action)
        
        action_result = await db.execute(action_query)
        activities_by_action = {row.action: row.count for row in action_result.all()}
        
        # Recent activities (last 10)
        recent_query = select(UserActivityLog).where(
            and_(
                UserActivityLog.user_id == user_id,
                UserActivityLog.created_at >= start_date
            )
        ).order_by(desc(UserActivityLog.created_at)).limit(10)
        
        recent_result = await db.execute(recent_query)
        recent_activities = list(recent_result.scalars().all())
        
        return {
            "total_activities": total,
            "activities_by_action": activities_by_action,
            "recent_activities": recent_activities
        }
    
    @staticmethod
    async def get_available_actions(
        db: AsyncSession,
        user_id: Optional[UUID] = None
    ) -> List[str]:
        """Get list of available action types"""
        query = select(UserActivityLog.action).distinct()
        
        if user_id:
            query = query.where(UserActivityLog.user_id == user_id)
        
        result = await db.execute(query)
        return [row[0] for row in result.all()]

