"""Service for extracting audio from videos using ffmpeg-python"""
import asyncio
import subprocess
import os
from pathlib import Path
from typing import Optional, Tuple
import ffmpeg
from app.utils.logger import logger


class AudioExtractionService:
    """Service for extracting audio from video files using ffmpeg-python or subprocess"""
    
    @staticmethod
    def _refresh_path():
        """Refresh PATH environment variable from system"""
        try:
            import platform
            if platform.system() == 'Windows':
                # Get PATH from system and user environment variables
                machine_path = os.environ.get('PATH', '')
                # Also try to get from registry/system
                try:
                    import winreg
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
                        system_path = winreg.QueryValueEx(key, "PATH")[0]
                    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                        try:
                            user_path = winreg.QueryValueEx(key, "PATH")[0]
                        except FileNotFoundError:
                            user_path = ""
                    os.environ['PATH'] = f"{system_path};{user_path};{machine_path}"
                except Exception:
                    # Fallback: just use current PATH
                    pass
        except Exception as e:
            logger.warning("Could not refresh PATH", error=str(e))
    
    @staticmethod
    def _find_ffmpeg_path() -> Optional[str]:
        """Try to find FFmpeg executable in common locations"""
        import platform
        if platform.system() == 'Windows':
            # Common Windows locations
            common_paths = [
                r"C:\ffmpeg\bin\ffmpeg.exe",
                r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Packages"),
            ]
            
            # Search in WinGet packages directory
            winget_base = os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Packages")
            if os.path.exists(winget_base):
                for root, dirs, files in os.walk(winget_base):
                    if 'ffmpeg.exe' in files:
                        ffmpeg_path = os.path.join(root, 'ffmpeg.exe')
                        if os.path.exists(ffmpeg_path):
                            return ffmpeg_path
            
            # Check common paths
            for path in common_paths:
                if os.path.exists(path):
                    return path
        
        return None
    
    @staticmethod
    def _check_ffmpeg_available() -> Tuple[bool, str]:
        """Check if ffmpeg is available on the system
        
        Returns:
            Tuple of (is_available, error_message)
        """
        # Refresh PATH to get latest system PATH
        AudioExtractionService._refresh_path()
        
        # Try to find FFmpeg in PATH first
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=5,
                env=os.environ.copy()
            )
            if result.returncode == 0:
                return True, ""
        except FileNotFoundError:
            pass
        except subprocess.TimeoutExpired:
            return False, "FFmpeg version check timed out"
        except Exception as e:
            logger.warning("Error checking FFmpeg in PATH", error=str(e))
        
        # If not in PATH, try to find it manually in common locations
        ffmpeg_path = AudioExtractionService._find_ffmpeg_path()
        if ffmpeg_path and os.path.exists(ffmpeg_path):
            try:
                result = subprocess.run(
                    [ffmpeg_path, '-version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # Add to PATH for this process
                    ffmpeg_dir = os.path.dirname(ffmpeg_path)
                    current_path = os.environ.get('PATH', '')
                    if ffmpeg_dir not in current_path:
                        os.environ['PATH'] = f"{ffmpeg_dir};{current_path}"
                    logger.info("Found FFmpeg and added to PATH", ffmpeg_path=ffmpeg_path)
                    return True, ""
            except Exception as e:
                logger.warning("Error checking found FFmpeg", ffmpeg_path=ffmpeg_path, error=str(e))
        
        return False, "FFmpeg is not found in PATH. Please RESTART your backend server to refresh PATH environment variable. FFmpeg was installed but the server process needs to be restarted to see it."
    
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
            # Check if video file exists
            video_file = Path(video_path)
            if not video_file.exists():
                logger.error("Video file not found", video_path=video_path)
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Check if ffmpeg is available
            ffmpeg_available, ffmpeg_error = AudioExtractionService._check_ffmpeg_available()
            if not ffmpeg_available:
                error_msg = f"FFmpeg is not available: {ffmpeg_error}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Create video-specific audio directory
            audio_dir = output_dir / video_id
            audio_dir.mkdir(exist_ok=True, parents=True)
            
            # Generate audio filename
            audio_filename = f"audio.{audio_format}"
            audio_path = audio_dir / audio_filename
            
            logger.info("Extracting audio from video", 
                       video_path=video_path, 
                       audio_path=str(audio_path),
                       video_id=video_id)
            
            # Use ffmpeg-python for audio extraction
            def extract_with_ffmpeg_python():
                """Extract audio using ffmpeg-python library"""
                try:
                    (
                        ffmpeg
                        .input(str(video_path))
                        .output(
                            str(audio_path),
                            format=audio_format,
                            acodec='libmp3lame' if audio_format == 'mp3' else 'pcm_s16le',
                            ar=44100,  # Sample rate
                            ac=2  # Stereo
                        )
                        .overwrite_output()
                        .run(quiet=True, capture_stdout=True, capture_stderr=True)
                    )
                    return True
                except ffmpeg.Error as e:
                    logger.error("FFmpeg-python error", 
                               error=str(e),
                               stderr=e.stderr.decode() if e.stderr else None,
                               stdout=e.stdout.decode() if e.stdout else None)
                    return False
                except Exception as e:
                    logger.error("FFmpeg-python unexpected error", error=str(e), exc_info=True)
                    return False
            
            # Run in thread pool to make it async
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, extract_with_ffmpeg_python)
            
            if not success:
                logger.error("Audio extraction failed", video_path=video_path)
                raise RuntimeError("Failed to extract audio from video using ffmpeg")
            
            # Verify audio file was created
            if not audio_path.exists():
                logger.error("Audio file not created after extraction", audio_path=str(audio_path))
                raise FileNotFoundError(f"Audio file was not created: {audio_path}")
            
            # Check file size
            file_size = audio_path.stat().st_size
            if file_size == 0:
                logger.error("Audio file is empty", audio_path=str(audio_path))
                raise ValueError(f"Audio file is empty: {audio_path}")
            
            file_size_mb = file_size / (1024 * 1024)
            logger.info("Audio extracted successfully", 
                       audio_path=str(audio_path),
                       size_mb=round(file_size_mb, 2),
                       video_id=video_id)
            return str(audio_path)
                
        except FileNotFoundError as e:
            logger.error("File not found during audio extraction", 
                       video_path=video_path,
                       error=str(e),
                       exc_info=True)
            raise
        except RuntimeError as e:
            logger.error("Runtime error during audio extraction", 
                       video_path=video_path,
                       error=str(e),
                       exc_info=True)
            raise
        except Exception as e:
            logger.error("Unexpected error during audio extraction", 
                       video_path=video_path,
                       error=str(e),
                       exc_info=True)
            raise RuntimeError(f"Audio extraction failed: {str(e)}") from e

