"""High-performance frame extraction service using OpenCV"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
from PIL import Image

from app.utils.logger import logger


class FrameExtractionService:
    """Service for extracting frames from videos with high performance"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def extract_frames_opencv(
        self,
        video_path: str,
        output_dir: Path,
        video_id: str,
        frames_per_second: float = 0.5
    ) -> List[Dict]:
        """
        Extract frames from video using OpenCV
        Returns frames in memory (as numpy arrays) and saves to disk
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            video_id: Unique video identifier
            frames_per_second: Number of frames to extract per second (default: 0.5 = 1 frame every 2 seconds)
        
        Returns:
            List of frame dictionaries with timestamp, frame data, and path
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            raise Exception("Invalid video FPS")
        
        frame_interval = int(fps / frames_per_second) if frames_per_second > 0 else 1  # Frames to skip
        frame_count = 0
        timestamp = 0.0
        
        # Create output directory for this video
        video_frames_dir = output_dir / video_id
        video_frames_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting frame extraction", video_path=video_path, fps=fps, interval=frame_interval)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                # Calculate timestamp
                timestamp = frame_count / fps
                
                # Generate filename with timestamp
                frame_filename = f"frame_{int(timestamp):05d}.jpg"
                frame_path = video_frames_dir / frame_filename
                
                # Save frame to disk
                cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                # Store frame info (frame data in memory for processing)
                frames.append({
                    "timestamp": round(timestamp, 2),
                    "frame_number": frame_count,
                    "image_path": str(frame_path),
                    "frame_data": frame,  # Keep in memory for analysis
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                })
                
                logger.debug("Frame extracted", timestamp=timestamp, frame_number=frame_count)
            
            frame_count += 1
        
        cap.release()
        
        logger.info("Frame extraction completed", 
                   video_path=video_path, 
                   total_frames=len(frames),
                   video_duration=timestamp)
        
        return frames
    
    async def extract_frames_async(
        self,
        video_path: str,
        output_dir: Path,
        video_id: str,
        frames_per_second: float = 0.5
    ) -> List[Dict]:
        """Async wrapper for frame extraction"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.extract_frames_opencv,
            video_path,
            output_dir,
            video_id,
            frames_per_second
        )
    
    def cleanup_frames(self, output_dir: Path, video_id: str):
        """Clean up extracted frames for a video"""
        video_frames_dir = output_dir / video_id
        if video_frames_dir.exists():
            import shutil
            try:
                shutil.rmtree(video_frames_dir)
                logger.info("Frames cleaned up", video_id=video_id)
            except Exception as e:
                logger.error("Failed to cleanup frames", video_id=video_id, error=str(e))

