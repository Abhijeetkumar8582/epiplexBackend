"""
Production-ready video processing service
Handles complete pipeline: upload -> audio extraction -> transcription -> keyframe extraction -> GPT analysis -> database storage
"""
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from openai import AsyncOpenAI

from app.config import settings
from app.database import FrameAnalysis
from app.services.frame_extraction_service import FrameExtractionService
from app.services.gpt_service import GPTService
from app.services.job_service import JobService
from app.services.audio_extraction_service import AudioExtractionService
from app.utils.logger import logger


class VideoProcessingService:
    """Production-ready video processing service"""
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
        self.frame_extractor = FrameExtractionService()
        self.gpt_service = GPTService()
        self.audio_extractor = AudioExtractionService()
        self.batch_size = 5  # Process 5 frames at a time
    
    async def extract_audio_and_transcribe(
        self,
        video_path: str,
        video_id: UUID,
        job_id: str,
        audio_dir: Path,
        db: AsyncSession
    ) -> Tuple[str, Optional[str]]:
        """
        Step 2: Extract audio from video, save it, and transcribe using OpenAI Whisper API
        
        Args:
            video_path: Path to video file
            video_id: Video upload ID
            job_id: Job ID for status updates
            audio_dir: Directory to save audio file
            db: Database session
            
        Returns:
            Tuple of (transcript text, audio_file_path)
        """
        try:
            # Update job status
            await JobService.update_job(db, job_id, {
                "progress": 15,
                "message": "Extracting audio and transcribing...",
                "current_step": "transcribe",
                "step_progress": {
                    "upload": "completed",
                    "transcribe": "processing",
                    "extract_frames": "pending",
                    "analyze_frames": "pending",
                    "complete": "pending"
                }
            })
            
            if not self.openai_client:
                raise ValueError("OpenAI API key not configured")
            
            logger.info("Starting audio extraction and transcription", 
                       job_id=job_id, 
                       video_path=video_path)
            
            # Step 1: Extract audio from video
            audio_path = await self.audio_extractor.extract_audio_async(
                video_path=video_path,
                output_dir=audio_dir,
                video_id=str(video_id),
                audio_format="mp3"
            )
            
            if not audio_path:
                raise ValueError("Failed to extract audio from video")
            
            # Step 2: Transcribe using OpenAI Whisper (use extracted audio file)
            with open(audio_path, 'rb') as audio_file:
                transcript_response = await self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            transcript = transcript_response if isinstance(transcript_response, str) else transcript_response.text
            
            # Update job status with transcript
            await JobService.update_job(db, job_id, {
                "progress": 30,
                "message": "Transcription completed successfully",
                "transcript": transcript,
                "step_progress": {
                    "upload": "completed",
                    "transcribe": "completed",
                    "extract_frames": "pending",
                    "analyze_frames": "pending",
                    "complete": "pending"
                }
            })
            
            logger.info("Transcription completed", 
                       job_id=job_id, 
                       transcript_length=len(transcript),
                       audio_path=audio_path)
            
            return transcript, audio_path
                        
        except Exception as e:
            logger.error("Transcription failed", 
                        job_id=job_id, 
                        error=str(e), 
                        exc_info=True)
            await JobService.update_job(db, job_id, {
                "status": "failed",
                "message": f"Transcription failed: {str(e)}",
                "error": str(e)
            })
            raise
    
    async def extract_keyframes(
        self,
        video_path: str,
        video_id: UUID,
        frames_dir: Path,
        job_id: str,
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """
        Step 3: Extract keyframes (1 per second) from video
        
        Args:
            video_path: Path to video file
            video_id: Video upload ID
            frames_dir: Directory to save frames
            job_id: Job ID for status updates
            db: Database session
            
        Returns:
            List of frame dictionaries with image_path, timestamp, etc.
        """
        try:
            # Update job status
            await JobService.update_job(db, job_id, {
                "progress": 35,
                "message": "Extracting keyframes from video (1 per second)...",
                "current_step": "extract_frames",
                "step_progress": {
                    "upload": "completed",
                    "transcribe": "completed",
                    "extract_frames": "processing",
                    "analyze_frames": "pending",
                    "complete": "pending"
                }
            })
            
            logger.info("Starting keyframe extraction", 
                       job_id=job_id, 
                       video_id=str(video_id))
            
            # Extract frames (1 per second)
            frames = await self.frame_extractor.extract_frames_async(
                video_path=video_path,
                output_dir=frames_dir,
                video_id=str(video_id),
                frames_per_second=1  # 1 frame per second
            )
            
            if not frames:
                logger.warning("No frames extracted", 
                             job_id=job_id, 
                             video_id=str(video_id))
                return []
            
            logger.info("Keyframes extracted", 
                       job_id=job_id, 
                       video_id=str(video_id), 
                       frame_count=len(frames))
            
            # Update job status
            await JobService.update_job(db, job_id, {
                "progress": 50,
                "message": f"Extracted {len(frames)} keyframes. Starting GPT analysis...",
                "step_progress": {
                    "upload": "completed",
                    "transcribe": "completed",
                    "extract_frames": "completed",
                    "analyze_frames": "processing",
                    "complete": "pending"
                }
            })
            
            return frames
            
        except Exception as e:
            logger.error("Keyframe extraction failed", 
                        job_id=job_id, 
                        video_id=str(video_id), 
                        error=str(e), 
                        exc_info=True)
            await JobService.update_job(db, job_id, {
                "status": "failed",
                "message": f"Keyframe extraction failed: {str(e)}",
                "error": str(e)
            })
            raise
    
    async def process_frames_with_gpt_batch(
        self,
        frames: List[Dict[str, Any]],
        video_id: UUID,
        job_id: str,
        db: AsyncSession
    ) -> List[FrameAnalysis]:
        """
        Step 4: Process frames in batches of 5 through ChatGPT 4o Mini and store in database
        
        Args:
            frames: List of frame dictionaries
            video_id: Video upload ID
            job_id: Job ID for status updates
            db: Database session
            
        Returns:
            List of FrameAnalysis objects stored in database
        """
        try:
            total_frames = len(frames)
            processed_frames = []
            
            logger.info("Starting batch frame processing with GPT", 
                       job_id=job_id, 
                       video_id=str(video_id), 
                       total_frames=total_frames,
                       batch_size=self.batch_size)
            
            # Process frames in batches of 5
            for batch_start in range(0, total_frames, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_frames)
                batch = frames[batch_start:batch_end]
                batch_num = (batch_start // self.batch_size) + 1
                total_batches = (total_frames + self.batch_size - 1) // self.batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches}", 
                           job_id=job_id, 
                           batch_start=batch_start, 
                           batch_end=batch_end)
                
                # Update progress
                progress = 50 + int((batch_start / total_frames) * 40)  # 50-90%
                await JobService.update_job(db, job_id, {
                    "progress": progress,
                    "message": f"Analyzing frames with GPT 4o Mini: batch {batch_num}/{total_batches} ({batch_start+1}-{batch_end}/{total_frames})",
                    "current_step": "analyze_frames"
                })
                
                # Analyze batch with GPT
                analyzed_batch = await self.gpt_service.batch_analyze_frames(
                    frames=batch,
                    max_workers=min(self.batch_size, 5)  # Process up to 5 concurrently
                )
                
                # Store batch in database
                batch_frame_analyses = []
                for frame_data in analyzed_batch:
                    # Create GPT response object
                    gpt_response = {
                        "description": frame_data.get("description"),
                        "ocr_text": frame_data.get("ocr_text"),
                        "processing_time_ms": frame_data.get("processing_time_ms"),
                        "timestamp": frame_data.get("timestamp"),
                        "frame_number": frame_data.get("frame_number"),
                        "image_path": frame_data.get("image_path"),
                        "model": "gpt-4o-mini"
                    }
                    
                    # Create FrameAnalysis object
                    frame_analysis = FrameAnalysis(
                        video_id=video_id,
                        timestamp=frame_data.get("timestamp", 0.0),
                        frame_number=frame_data.get("frame_number"),
                        image_path=frame_data.get("image_path"),
                        description=frame_data.get("description"),
                        ocr_text=frame_data.get("ocr_text"),
                        gpt_response=gpt_response,
                        processing_time_ms=frame_data.get("processing_time_ms")
                    )
                    
                    batch_frame_analyses.append(frame_analysis)
                    db.add(frame_analysis)
                
                # Commit batch to database
                await db.commit()
                
                # Refresh all objects
                for fa in batch_frame_analyses:
                    await db.refresh(fa)
                
                processed_frames.extend(batch_frame_analyses)
                
                logger.info(f"Batch {batch_num} completed and stored", 
                           job_id=job_id, 
                           frames_in_batch=len(batch_frame_analyses),
                           total_processed=len(processed_frames))
            
            logger.info("All frames processed and stored", 
                       job_id=job_id, 
                       video_id=str(video_id), 
                       total_frames=len(processed_frames))
            
            return processed_frames
            
        except Exception as e:
            logger.error("Frame processing failed", 
                        job_id=job_id, 
                        video_id=str(video_id), 
                        error=str(e), 
                        exc_info=True)
            await JobService.update_job(db, job_id, {
                "status": "failed",
                "message": f"Frame processing failed: {str(e)}",
                "error": str(e)
            })
            raise
    
    async def process_video_complete(
        self,
        video_path: str,
        video_id: UUID,
        job_id: str,
        frames_dir: Path,
        audio_dir: Path,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Complete video processing pipeline:
        1. Extract audio and transcribe
        2. Extract keyframes (1 per second)
        3. Process frames in batches of 5 through GPT 4o Mini
        4. Store everything in database
        
        Args:
            video_path: Path to video file
            video_id: Video upload ID
            job_id: Job ID for status updates
            frames_dir: Directory to save frames
            db: Database session
            
        Returns:
            Dictionary with transcript, frame_analyses, and processing summary
        """
        try:
            logger.info("Starting complete video processing pipeline", 
                       job_id=job_id, 
                       video_id=str(video_id))
            
            # Step 1: Extract audio and transcribe
            transcript, audio_path = await self.extract_audio_and_transcribe(
                video_path=video_path,
                video_id=video_id,
                job_id=job_id,
                audio_dir=audio_dir,
                db=db
            )
            
            # Save audio path to video upload
            from app.services.video_upload_service import VideoUploadService
            await VideoUploadService.update_upload_audio(db, video_id, audio_path)
            
            # Step 2: Extract keyframes
            frames = await self.extract_keyframes(
                video_path=video_path,
                video_id=video_id,
                frames_dir=frames_dir,
                job_id=job_id,
                db=db
            )
            
            if not frames:
                raise ValueError("No frames extracted from video")
            
            # Step 3: Process frames in batches and store in database
            frame_analyses = await self.process_frames_with_gpt_batch(
                frames=frames,
                video_id=video_id,
                job_id=job_id,
                db=db
            )
            
            # Step 4: Mark as completed
            await JobService.update_job(db, job_id, {
                "status": "completed",
                "progress": 100,
                "message": f"Processing completed successfully. Transcribed {len(transcript)} characters, analyzed {len(frame_analyses)} frames.",
                "current_step": "complete",
                "step_progress": {
                    "upload": "completed",
                    "transcribe": "completed",
                    "extract_frames": "completed",
                    "analyze_frames": "completed",
                    "complete": "completed"
                }
            })
            
            result = {
                "transcript": transcript,
                "frame_analyses_count": len(frame_analyses),
                "total_frames": len(frames),
                "status": "completed"
            }
            
            logger.info("Video processing pipeline completed successfully", 
                       job_id=job_id, 
                       video_id=str(video_id),
                       transcript_length=len(transcript),
                       frames_analyzed=len(frame_analyses))
            
            return result
            
        except Exception as e:
            logger.error("Video processing pipeline failed", 
                        job_id=job_id, 
                        video_id=str(video_id), 
                        error=str(e), 
                        exc_info=True)
            
            await JobService.update_job(db, job_id, {
                "status": "failed",
                "message": f"Processing failed: {str(e)}",
                "error": str(e)
            })
            
            raise

