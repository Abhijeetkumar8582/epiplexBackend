import cv2
from pathlib import Path
from typing import Dict, Optional, Any
import os

from app.utils.logger import logger


class VideoMetadataService:
    """Service to extract video metadata using OpenCV"""
    
    @staticmethod
    def extract_metadata(video_path: str) -> Dict[str, Any]:
        """
        Extract video metadata from video file
        
        Returns:
            Dictionary with video metadata:
            - video_length_seconds: float
            - video_size_bytes: int
            - mime_type: str (inferred from extension)
            - resolution_width: int
            - resolution_height: int
            - fps: float
        """
        try:
            # Get file size
            file_size = os.path.getsize(video_path)
            
            # Open video with OpenCV
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.warning("Could not open video file", video_path=video_path)
                return {
                    "video_length_seconds": None,
                    "video_size_bytes": file_size,
                    "mime_type": VideoMetadataService._get_mime_type(video_path),
                    "resolution_width": None,
                    "resolution_height": None,
                    "fps": None
                }
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate duration
            duration = frame_count / fps if fps > 0 else None
            
            # Get mime type
            mime_type = VideoMetadataService._get_mime_type(video_path)
            
            cap.release()
            
            metadata = {
                "video_length_seconds": round(duration, 2) if duration else None,
                "video_size_bytes": file_size,
                "mime_type": mime_type,
                "resolution_width": width,
                "resolution_height": height,
                "fps": round(fps, 2) if fps else None
            }
            
            logger.info("Video metadata extracted", video_path=video_path, **metadata)
            return metadata
            
        except Exception as e:
            logger.error("Error extracting video metadata", video_path=video_path, error=str(e), exc_info=True)
            # Return partial metadata if possible
            try:
                file_size = os.path.getsize(video_path)
                return {
                    "video_length_seconds": None,
                    "video_size_bytes": file_size,
                    "mime_type": VideoMetadataService._get_mime_type(video_path),
                    "resolution_width": None,
                    "resolution_height": None,
                    "fps": None
                }
            except:
                return {
                    "video_length_seconds": None,
                    "video_size_bytes": None,
                    "mime_type": None,
                    "resolution_width": None,
                    "resolution_height": None,
                    "fps": None
                }
    
    @staticmethod
    def _get_mime_type(video_path: str) -> Optional[str]:
        """Get MIME type from file extension"""
        extension = Path(video_path).suffix.lower()
        mime_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
            '.flv': 'video/x-flv',
            '.wmv': 'video/x-ms-wmv',
            '.m4v': 'video/x-m4v'
        }
        return mime_types.get(extension, 'video/unknown')

