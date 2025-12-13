"""
Summary Service for generating batch summaries of frame analyses
Processes frames in batches of 30 and generates summaries using GPT-4o-mini
"""
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from openai import AsyncOpenAI

from app.config import settings
from app.database import FrameAnalysis, VideoUpload
from app.utils.logger import logger
from app.services.pdf_summary_service import PDFSummaryService


class SummaryService:
    """Service for generating batch summaries of frame analyses"""
    
    def __init__(self):
        """Initialize summary service with OpenAI client"""
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
        self.batch_size = 30  # Process 30 frame descriptions per batch
        self.model = "gpt-4o-mini"
        self.pdf_service = PDFSummaryService()
        
        # Load prompt from file
        self.prompt_template = self._load_prompt_template()
        
        if not self.client:
            logger.warning("OpenAI API key not configured. Summary service will not work.")
    
    def _load_prompt_template(self) -> str:
        """Load prompt template from prompt.txt file"""
        try:
            prompt_file = Path(__file__).parent.parent.parent / "prompt.txt"
            if prompt_file.exists():
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            else:
                # Default prompt if file doesn't exist
                logger.warning("prompt.txt not found, using default prompt")
                return """Analyze this video frame and provide:
1. A detailed description of what you see (UI elements, text, layout, etc.)
2. Extract any visible text (OCR) from the frame
3. Identify any important information or data displayed

Frame timestamp: {timestamp} seconds"""
        except Exception as e:
            logger.error("Failed to load prompt template", error=str(e))
            return """Analyze this video frame and provide:
1. A detailed description of what you see (UI elements, text, layout, etc.)
2. Extract any visible text (OCR) from the frame
3. Identify any important information or data displayed

Frame timestamp: {timestamp} seconds"""
    
    async def generate_batch_summary(
        self,
        frame_descriptions: List[Dict[str, Any]],
        batch_number: int,
        total_batches: int
    ) -> Dict[str, Any]:
        """
        Generate a summary for a batch of frame descriptions
        
        Args:
            frame_descriptions: List of frame description dictionaries
            batch_number: Current batch number (1-indexed)
            total_batches: Total number of batches
            
        Returns:
            Dictionary with summary_text and metadata
        """
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        import time
        start_time = time.time()
        
        try:
            # Prepare context from frame descriptions
            context_parts = []
            for i, frame in enumerate(frame_descriptions, 1):
                timestamp = frame.get('timestamp', 0.0)
                description = frame.get('description', 'No description available')
                ocr_text = frame.get('ocr_text', '')
                
                frame_context = f"Frame {i} (Timestamp: {timestamp:.2f}s):\n"
                frame_context += f"Description: {description}\n"
                if ocr_text:
                    frame_context += f"OCR Text: {ocr_text}\n"
                context_parts.append(frame_context)
            
            context = "\n\n".join(context_parts)
            
            # Create summary prompt
            summary_prompt = f"""You are analyzing a video that has been processed frame by frame. Below are the descriptions and OCR text from {len(frame_descriptions)} consecutive frames (batch {batch_number} of {total_batches}).

Your task is to create a comprehensive summary that:
1. Identifies the main themes, actions, or processes shown across these frames
2. Highlights any important information, data, or UI elements that appear
3. Notes any patterns, sequences, or workflows that are evident
4. Captures the overall context and purpose of this section of the video

Frame Analysis Data:
{context}

Please provide a detailed summary that captures the essence of what happens in these frames, maintaining context for the overall video analysis."""

            # Call GPT-4o-mini
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing video frames and creating comprehensive summaries that maintain context across multiple frames."
                    },
                    {
                        "role": "user",
                        "content": summary_prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            summary_text = response.choices[0].message.content
            processing_time = int((time.time() - start_time) * 1000)
            
            result = {
                "summary_text": summary_text,
                "batch_number": batch_number,
                "total_batches": total_batches,
                "frames_in_batch": len(frame_descriptions),
                "processing_time_ms": processing_time,
                "model": self.model
            }
            
            logger.info("Batch summary generated",
                       batch_number=batch_number,
                       total_batches=total_batches,
                       frames_in_batch=len(frame_descriptions),
                       processing_time_ms=processing_time)
            
            return result
            
        except Exception as e:
            logger.error("Batch summary generation failed",
                        batch_number=batch_number,
                        error=str(e),
                        exc_info=True)
            raise
    
    async def generate_video_summaries(
        self,
        db: AsyncSession,
        video_id: UUID,
        job_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate summaries for all frame analyses of a video in batches of 30
        
        Args:
            db: Database session
            video_id: Video upload ID
            job_id: Optional job ID for status updates
            
        Returns:
            List of summary dictionaries
        """
        try:
            # Get all frame analyses for this video, ordered by timestamp
            query = select(FrameAnalysis).where(
                FrameAnalysis.video_id == video_id
            ).order_by(FrameAnalysis.timestamp)
            
            result = await db.execute(query)
            frame_analyses = list(result.scalars().all())
            
            if not frame_analyses:
                logger.warning("No frame analyses found for video", video_id=str(video_id))
                return []
            
            total_frames = len(frame_analyses)
            total_batches = (total_frames + self.batch_size - 1) // self.batch_size
            
            logger.info("Starting video summary generation",
                       video_id=str(video_id),
                       total_frames=total_frames,
                       total_batches=total_batches,
                       batch_size=self.batch_size)
            
            # Get video upload info for PDF
            video_query = select(VideoUpload).where(VideoUpload.id == video_id)
            video_result = await db.execute(video_query)
            video_upload = video_result.scalar_one_or_none()
            
            if not video_upload:
                raise ValueError(f"Video upload not found: {video_id}")
            
            # Create PDF path
            pdf_filename = f"{video_id}_summary.pdf"
            pdf_path = settings.OUTPUT_DIR / pdf_filename
            
            # Initialize PDF with title page
            self.pdf_service.create_pdf(
                pdf_path=pdf_path,
                video_name=video_upload.name,
                video_file_number=video_upload.video_file_number
            )
            
            summaries = []
            
            # Process frames in batches of 30
            for batch_start in range(0, total_frames, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_frames)
                batch_frames = frame_analyses[batch_start:batch_end]
                batch_number = (batch_start // self.batch_size) + 1
                
                # Update job status if provided
                if job_id:
                    from app.services.job_service import JobService
                    progress = 90 + int((batch_number / total_batches) * 10)  # 90-100%
                    await JobService.update_job(db, job_id, {
                        "progress": progress,
                        "message": f"Generating summaries: batch {batch_number}/{total_batches}",
                        "current_step": "generate_summaries"
                    })
                
                # Prepare frame descriptions for this batch
                frame_descriptions = []
                for frame in batch_frames:
                    frame_descriptions.append({
                        "timestamp": frame.timestamp,
                        "frame_number": frame.frame_number,
                        "description": frame.description or "",
                        "ocr_text": frame.ocr_text or ""
                    })
                
                # Generate summary for this batch
                summary_result = await self.generate_batch_summary(
                    frame_descriptions=frame_descriptions,
                    batch_number=batch_number,
                    total_batches=total_batches
                )
                
                # Add batch frame range info
                summary_result["batch_start_frame"] = batch_start + 1
                summary_result["batch_end_frame"] = batch_end
                summary_result["video_id"] = str(video_id)
                
                # Save summary to database
                summary_record = await self._save_summary_to_db(
                    db=db,
                    video_id=video_id,
                    summary_data=summary_result
                )
                
                # Prepare frame data for PDF (with image paths and meta_tags)
                frame_data_for_pdf = []
                for frame in batch_frames:
                    # Extract meta_tags from gpt_response if available
                    meta_tags = []
                    if frame.gpt_response and isinstance(frame.gpt_response, dict):
                        meta_tags = frame.gpt_response.get("meta_tags", [])
                        if not isinstance(meta_tags, list):
                            meta_tags = []
                    
                    frame_data_for_pdf.append({
                        "timestamp": frame.timestamp,
                        "frame_number": frame.frame_number,
                        "description": frame.description or "",
                        "image_path": frame.image_path,
                        "meta_tags": meta_tags
                    })
                
                # Append batch to PDF
                try:
                    self.pdf_service.append_batch_to_pdf(
                        pdf_path=pdf_path,
                        batch_number=batch_number,
                        total_batches=total_batches,
                        summary_text=summary_result["summary_text"],
                        frames=frame_data_for_pdf
                    )
                    logger.info(f"Batch {batch_number} appended to PDF",
                               video_id=str(video_id),
                               batch_number=batch_number)
                except Exception as pdf_error:
                    logger.error("Failed to append batch to PDF, continuing",
                               video_id=str(video_id),
                               batch_number=batch_number,
                               error=str(pdf_error))
                    # Don't fail the whole process if PDF generation fails
                
                summaries.append(summary_result)
                
                logger.info(f"Batch {batch_number} summary saved",
                           video_id=str(video_id),
                           batch_number=batch_number)
            
            # Update video upload with PDF path
            try:
                from sqlalchemy import update
                await db.execute(
                    update(VideoUpload)
                    .where(VideoUpload.id == video_id)
                    .values(summary_pdf_url=str(pdf_path))
                )
                await db.commit()
                logger.info("PDF path saved to video upload",
                           video_id=str(video_id),
                           pdf_path=str(pdf_path))
            except Exception as db_error:
                logger.error("Failed to save PDF path to database",
                           video_id=str(video_id),
                           error=str(db_error))
                await db.rollback()
            
            logger.info("Video summary generation completed",
                       video_id=str(video_id),
                       total_batches=len(summaries),
                       pdf_path=str(pdf_path))
            
            return summaries
            
        except Exception as e:
            logger.error("Video summary generation failed",
                        video_id=str(video_id),
                        error=str(e),
                        exc_info=True)
            raise
    
    async def _save_summary_to_db(
        self,
        db: AsyncSession,
        video_id: UUID,
        summary_data: Dict[str, Any]
    ) -> Any:
        """
        Save summary to database
        
        Args:
            db: Database session
            video_id: Video upload ID
            summary_data: Summary data dictionary
            
        Returns:
            Summary record
        """
        try:
            # Import here to avoid circular dependencies
            from app.database import VideoSummary
            
            summary_record = VideoSummary(
                video_id=video_id,
                batch_number=summary_data["batch_number"],
                batch_start_frame=summary_data["batch_start_frame"],
                batch_end_frame=summary_data["batch_end_frame"],
                total_frames_in_batch=summary_data["frames_in_batch"],
                summary_text=summary_data["summary_text"],
                summary_metadata=str(summary_data),  # Store full data as JSON string
                processing_time_ms=summary_data.get("processing_time_ms"),
                model_used=summary_data.get("model", self.model)
            )
            
            db.add(summary_record)
            await db.commit()
            await db.refresh(summary_record)
            
            return summary_record
            
        except Exception as e:
            logger.error("Failed to save summary to database",
                        video_id=str(video_id),
                        error=str(e),
                        exc_info=True)
            await db.rollback()
            raise
    
    async def get_video_summaries(
        self,
        db: AsyncSession,
        video_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        Get all summaries for a video
        
        Args:
            db: Database session
            video_id: Video upload ID
            
        Returns:
            List of summary dictionaries
        """
        try:
            from app.database import VideoSummary
            
            query = select(VideoSummary).where(
                VideoSummary.video_id == video_id
            ).order_by(VideoSummary.batch_number)
            
            result = await db.execute(query)
            summaries = list(result.scalars().all())
            
            return [
                {
                    "id": str(s.id),
                    "batch_number": s.batch_number,
                    "batch_start_frame": s.batch_start_frame,
                    "batch_end_frame": s.batch_end_frame,
                    "total_frames_in_batch": s.total_frames_in_batch,
                    "summary_text": s.summary_text,
                    "processing_time_ms": s.processing_time_ms,
                    "model_used": s.model_used,
                    "created_at": s.created_at.isoformat() if s.created_at else None
                }
                for s in summaries
            ]
            
        except Exception as e:
            logger.error("Failed to get video summaries",
                        video_id=str(video_id),
                        error=str(e),
                        exc_info=True)
            return []

