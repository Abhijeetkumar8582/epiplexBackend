"""Real GPT-4 Vision service for frame analysis"""
import base64
import time
from typing import Dict, Optional, List
import numpy as np
from pathlib import Path
import aiofiles
from openai import AsyncOpenAI

from app.config import settings
from app.utils.logger import logger


class GPTService:
    """Real GPT-4 Vision API service for analyzing video frames"""
    
    def __init__(self):
        """Initialize GPT service with OpenAI client"""
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY) if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY else None
        
        if not self.client:
            logger.warning("OpenAI API key not configured. GPT service will not work.")
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    async def analyze_frame(
        self,
        image_path: str,
        timestamp_seconds: float,
        frame_number: Optional[int] = None
    ) -> Dict:
        """
        Analyze a single frame using GPT-4 Vision API
        
        Args:
            image_path: Path to the frame image file
            timestamp_seconds: Timestamp of the frame in the video
            frame_number: Optional frame number
        
        Returns:
            Dictionary with 'description', 'ocr_text', and 'processing_time_ms'
        """
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        start_time = time.time()
        
        try:
            # Read and encode image
            async with aiofiles.open(image_path, 'rb') as f:
                image_data = await f.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Call GPT-4 Vision API
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # or "gpt-4-vision-preview" for better results
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this video frame and provide:
1. A detailed description of what you see (UI elements, text, layout, etc.)
2. Extract any visible text (OCR) from the frame
3. Identify any important information or data displayed

Frame timestamp: {:.2f} seconds""".format(timestamp_seconds)
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            # Extract description and OCR text from response
            # Simple parsing - you may want to improve this based on actual API response format
            description = content
            ocr_text = None
            
            # Try to extract OCR text if mentioned in response
            if "Text found:" in content or "OCR:" in content:
                # Simple extraction - adjust based on actual response format
                lines = content.split('\n')
                ocr_lines = []
                in_ocr_section = False
                for line in lines:
                    if "Text found:" in line or "OCR:" in line:
                        in_ocr_section = True
                        continue
                    if in_ocr_section and line.strip():
                        ocr_lines.append(line.strip())
                    elif in_ocr_section and not line.strip():
                        break
                ocr_text = '\n'.join(ocr_lines) if ocr_lines else None
            
            processing_time = int((time.time() - start_time) * 1000)
            
            result = {
                "description": description,
                "ocr_text": ocr_text,
                "processing_time_ms": processing_time,
                "model": "gpt-4o-mini",
                "timestamp": timestamp_seconds,
                "frame_number": frame_number
            }
            
            logger.info("Frame analyzed with GPT", 
                       image_path=image_path,
                       processing_time_ms=processing_time,
                       has_ocr=ocr_text is not None)
            
            return result
            
        except Exception as e:
            logger.error("GPT frame analysis failed",
                        image_path=image_path,
                        error=str(e),
                        exc_info=True)
            # Return error result
            return {
                "description": f"Error analyzing frame: {str(e)}",
                "ocr_text": None,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "error": str(e)
            }
    
    async def batch_analyze_frames(
        self,
        frames: List[Dict],
        max_workers: int = 5
    ) -> List[Dict]:
        """
        Analyze multiple frames in parallel (production-ready with error handling)
        
        Args:
            frames: List of frame dictionaries with 'image_path', 'timestamp', etc.
            max_workers: Maximum number of concurrent API calls (default: 5 for batch processing)
        
        Returns:
            List of analyzed frames with GPT responses
        """
        import asyncio
        
        if not self.client:
            logger.error("OpenAI API key not configured. Cannot analyze frames.")
            # Return frames with error messages
            for frame in frames:
                frame.update({
                    "description": "OpenAI API key not configured",
                    "ocr_text": None,
                    "processing_time_ms": 0,
                    "error": "OpenAI API key not configured"
                })
            return frames
        
        if not frames:
            return []
        
        # Create semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(max_workers)
        
        async def analyze_with_semaphore(frame_data):
            async with semaphore:
                try:
                    return await self.analyze_frame(
                        image_path=frame_data.get("image_path") or frame_data.get("frame_path"),
                        timestamp_seconds=frame_data.get("timestamp", 0.0),
                        frame_number=frame_data.get("frame_number")
                    )
                except Exception as e:
                    logger.error("Frame analysis exception",
                               frame_index=frame_data.get("frame_number"),
                               error=str(e))
                    return {
                        "description": f"Error analyzing frame: {str(e)}",
                        "ocr_text": None,
                        "processing_time_ms": 0,
                        "error": str(e)
                    }
        
        # Analyze all frames concurrently
        tasks = [analyze_with_semaphore(frame) for frame in frames]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results with frame data
        analyzed_frames = []
        for i, (frame, result) in enumerate(zip(frames, results)):
            if isinstance(result, Exception):
                logger.error("Frame analysis failed",
                           frame_index=i,
                           timestamp=frame.get("timestamp"),
                           error=str(result))
                frame.update({
                    "description": f"Error: {str(result)}",
                    "ocr_text": None,
                    "processing_time_ms": 0,
                    "error": str(result)
                })
            else:
                frame.update(result)
            analyzed_frames.append(frame)
        
        # Sort by timestamp to maintain order
        analyzed_frames.sort(key=lambda x: x.get("timestamp", 0))
        
        logger.info("Batch frame analysis completed",
                   total_frames=len(analyzed_frames),
                   successful=sum(1 for f in analyzed_frames if "error" not in f or not f.get("error")))
        
        return analyzed_frames

