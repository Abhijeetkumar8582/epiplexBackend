"""Service for extracting audio from videos"""
import asyncio
import subprocess
from pathlib import Path
from typing import Optional
from app.utils.logger import logger


class AudioExtractionService:
    """Service for extracting audio from video files"""
    
    @staticmethod
    async def extract_audio_async(
        video_path: str,
        output_dir: Path,
        video_id: str,
        audio_format: str = "mp3"
    ) -> Optional[str]:
        """
        Extract audio from video using ffmpeg
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save audio file
            video_id: Video ID for filename
            audio_format: Audio format (mp3, wav, etc.)
            
        Returns:
            Path to extracted audio file or None if failed
        """
        try:
            # Create video-specific audio directory
            audio_dir = output_dir / video_id
            audio_dir.mkdir(exist_ok=True, parents=True)
            
            # Generate audio filename
            audio_filename = f"audio.{audio_format}"
            audio_path = audio_dir / audio_filename
            
            # Use ffmpeg to extract audio
            # -i: input file
            # -vn: disable video
            # -acodec: audio codec (libmp3lame for mp3)
            # -ar: audio sample rate (44100 Hz)
            # -ac: audio channels (2 for stereo)
            # -y: overwrite output file
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vn",  # No video
                "-acodec", "libmp3lame" if audio_format == "mp3" else "pcm_s16le",
                "-ar", "44100",  # Sample rate
                "-ac", "2",  # Stereo
                "-y",  # Overwrite
                str(audio_path)
            ]
            
            logger.info("Extracting audio from video", 
                       video_path=video_path, 
                       audio_path=str(audio_path))
            
            # Run ffmpeg command asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error("Audio extraction failed", 
                           video_path=video_path,
                           error=stderr.decode())
                return None
            
            if audio_path.exists():
                logger.info("Audio extracted successfully", 
                           audio_path=str(audio_path),
                           size_mb=round(audio_path.stat().st_size / (1024 * 1024), 2))
                return str(audio_path)
            else:
                logger.error("Audio file not created", audio_path=str(audio_path))
                return None
                
        except Exception as e:
            logger.error("Audio extraction error", 
                       video_path=video_path,
                       error=str(e),
                       exc_info=True)
            return None

