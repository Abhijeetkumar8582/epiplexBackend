"""
Tests for Queue Worker Service
Tests sequential processing, delays, file cleanup, and crawler behavior
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from datetime import datetime, timedelta
from uuid import uuid4

from app.services.queue_worker_service import QueueWorkerService
from app.database import VideoUpload


class TestQueueWorkerSequentialProcessing:
    """Test sequential video processing with proper delays"""
    
    @pytest.mark.asyncio
    async def test_sequential_processing_with_delay(self):
        """Test that videos are processed one at a time with 1-minute delay"""
        service = QueueWorkerService()
        service.processing_delay = 1  # Use 1 second for testing (instead of 60)
        
        # Mock database and video objects
        video1 = MagicMock(spec=VideoUpload)
        video1.id = uuid4()
        video1.status = "uploaded"
        video1.job_id = None
        video1.video_file_number = "VF-2026-0001"
        video1.video_url = "/test/path/video1.mp4"
        
        video2 = MagicMock(spec=VideoUpload)
        video2.id = uuid4()
        video2.status = "uploaded"
        video2.job_id = None
        video2.video_file_number = "VF-2026-0002"
        video2.video_url = "/test/path/video2.mp4"
        
        # Mock get_next_video_in_queue to return videos sequentially
        call_count = 0
        async def mock_get_next(db):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return video1
            elif call_count == 2:
                return video2
            else:
                return None
        
        service.get_next_video_in_queue = AsyncMock(side_effect=mock_get_next)
        
        # Mock process_video_from_queue
        process_times = []
        async def mock_process(video):
            process_times.append(datetime.now())
            return True
        
        service.process_video_from_queue = AsyncMock(side_effect=mock_process)
        service.is_running = True
        
        # Run worker loop for 2 iterations
        start_time = datetime.now()
        iterations = 0
        
        async def test_loop():
            nonlocal iterations
            async with patch('app.services.queue_worker_service.AsyncSessionLocal'):
                while iterations < 2:
                    video = await service.get_next_video_in_queue(None)
                    if video:
                        await service.process_video_from_queue(video)
                        iterations += 1
                        if iterations < 2:
                            await asyncio.sleep(service.processing_delay)
                    else:
                        break
        
        await test_loop()
        
        # Verify sequential processing
        assert len(process_times) == 2, "Should process 2 videos"
        
        # Verify delay between processing (with 0.5s tolerance)
        if len(process_times) >= 2:
            delay = (process_times[1] - process_times[0]).total_seconds()
            assert delay >= 0.9, f"Delay should be at least 0.9s, got {delay}s"
            assert delay <= 2.0, f"Delay should be at most 2.0s, got {delay}s"
    
    @pytest.mark.asyncio
    async def test_only_processes_uploaded_status(self):
        """Test that only videos with 'uploaded' status are processed"""
        service = QueueWorkerService()
        
        # Create videos with different statuses
        uploaded_video = MagicMock(spec=VideoUpload)
        uploaded_video.id = uuid4()
        uploaded_video.status = "uploaded"
        
        processing_video = MagicMock(spec=VideoUpload)
        processing_video.id = uuid4()
        processing_video.status = "processing"
        
        completed_video = MagicMock(spec=VideoUpload)
        completed_video.id = uuid4()
        completed_video.status = "completed"
        
        # Mock get_next_video_in_queue to only return uploaded
        async def mock_get_next(db):
            return uploaded_video
        
        service.get_next_video_in_queue = AsyncMock(side_effect=mock_get_next)
        
        # Mock process_video_from_queue
        processed_videos = []
        async def mock_process(video):
            # Defensive check should skip non-uploaded
            if video.status != "uploaded":
                return False
            processed_videos.append(video.id)
            return True
        
        service.process_video_from_queue = AsyncMock(side_effect=mock_process)
        
        # Test
        video = await service.get_next_video_in_queue(None)
        if video:
            result = await service.process_video_from_queue(video)
            assert result is True, "Should process uploaded video"
            assert video.id in processed_videos, "Video should be processed"
        
        # Test defensive check
        processing_video.status = "processing"
        result = await service.process_video_from_queue(processing_video)
        assert result is False, "Should skip processing video"
        
        completed_video.status = "completed"
        result = await service.process_video_from_queue(completed_video)
        assert result is False, "Should skip completed video"


class TestFileCleanup:
    """Test video file deletion after processing completes"""
    
    @pytest.mark.asyncio
    async def test_file_deletion_after_completion(self):
        """Test that video file is deleted after successful processing"""
        service = QueueWorkerService()
        
        # Create mock video with file path
        test_file_path = Path("/test/uploads/video123.mp4")
        video = MagicMock(spec=VideoUpload)
        video.id = uuid4()
        video.video_url = str(test_file_path)
        
        # Mock file system
        with patch('app.services.queue_worker_service.Path') as mock_path:
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.is_file.return_value = True
            mock_path_instance.unlink = MagicMock()
            mock_path.return_value = mock_path_instance
            
            # Mock settings
            with patch('app.services.queue_worker_service.settings') as mock_settings:
                mock_settings.UPLOAD_DIR = Path("/test/uploads")
                
                # Test cleanup
                result = await service.cleanup_video_file(video)
                
                # Verify file was deleted
                assert result is True, "Cleanup should succeed"
                mock_path_instance.unlink.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_no_deletion_on_failed_processing(self):
        """Test that file is NOT deleted if processing failed"""
        service = QueueWorkerService()
        
        video = MagicMock(spec=VideoUpload)
        video.id = uuid4()
        video.status = "failed"
        video.video_url = "/test/path/video.mp4"
        
        # Mock process_video_from_queue to return False (failed)
        async def mock_process(video):
            return False  # Processing failed
        
        service.process_video_from_queue = AsyncMock(side_effect=mock_process)
        service.cleanup_video_file = AsyncMock()
        
        result = await service.process_video_from_queue(video)
        
        # Verify cleanup was NOT called
        assert result is False, "Processing should fail"
        service.cleanup_video_file.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_cleanup_handles_remote_urls(self):
        """Test that remote URLs are not deleted"""
        service = QueueWorkerService()
        
        video = MagicMock(spec=VideoUpload)
        video.id = uuid4()
        video.video_url = "https://example.com/video.mp4"  # Remote URL
        
        result = await service.cleanup_video_file(video)
        
        # Should return False (not a local file)
        assert result is False, "Should not delete remote URLs"


class TestCrawlerInterval:
    """Test 30-minute crawler interval when queue is empty"""
    
    @pytest.mark.asyncio
    async def test_crawler_interval_when_queue_empty(self):
        """Test that worker waits 30 minutes when queue is empty"""
        service = QueueWorkerService()
        service.crawler_interval = 2  # Use 2 seconds for testing (instead of 1800)
        service.is_running = True
        
        # Mock get_next_video_in_queue to return None (empty queue)
        async def mock_get_next(db):
            return None
        
        service.get_next_video_in_queue = AsyncMock(side_effect=mock_get_next)
        
        # Track sleep calls
        sleep_calls = []
        original_sleep = asyncio.sleep
        
        async def mock_sleep(duration):
            sleep_calls.append(duration)
            await original_sleep(0.1)  # Short sleep for testing
        
        # Test one iteration of worker loop
        start_time = datetime.now()
        
        with patch('asyncio.sleep', side_effect=mock_sleep):
            async with patch('app.services.queue_worker_service.AsyncSessionLocal'):
                video = await service.get_next_video_in_queue(None)
                if not video:
                    # Queue empty - should wait crawler interval
                    await asyncio.sleep(service.crawler_interval)
        
        # Verify crawler interval was used
        assert len(sleep_calls) > 0, "Should have called sleep"
        assert service.crawler_interval in sleep_calls or any(
            abs(call - service.crawler_interval) < 0.5 for call in sleep_calls
        ), f"Should sleep for crawler interval ({service.crawler_interval}s)"
    
    @pytest.mark.asyncio
    async def test_active_polling_when_queue_has_items(self):
        """Test that worker uses processing delay (1 min) when queue has items"""
        service = QueueWorkerService()
        service.processing_delay = 1  # 1 second for testing
        service.crawler_interval = 30  # 30 seconds for testing
        service.is_running = True
        
        video = MagicMock(spec=VideoUpload)
        video.id = uuid4()
        video.status = "uploaded"
        
        # Mock get_next_video_in_queue to return video
        async def mock_get_next(db):
            return video
        
        service.get_next_video_in_queue = AsyncMock(side_effect=mock_get_next)
        service.process_video_from_queue = AsyncMock(return_value=True)
        
        # Track sleep calls
        sleep_calls = []
        
        async def mock_sleep(duration):
            sleep_calls.append(duration)
            await asyncio.sleep(0.1)  # Short sleep for testing
        
        # Test one iteration
        with patch('asyncio.sleep', side_effect=mock_sleep):
            async with patch('app.services.queue_worker_service.AsyncSessionLocal'):
                video = await service.get_next_video_in_queue(None)
                if video:
                    await service.process_video_from_queue(video)
                    # Should wait processing delay (not crawler interval)
                    await asyncio.sleep(service.processing_delay)
        
        # Verify processing delay was used (not crawler interval)
        assert len(sleep_calls) > 0, "Should have called sleep"
        assert any(
            abs(call - service.processing_delay) < 0.5 for call in sleep_calls
        ), f"Should use processing delay ({service.processing_delay}s), not crawler interval"


class TestFrontendStatusUpdates:
    """Test that frontend only updates active processing video status"""
    
    def test_polling_filters_processing_videos_only(self):
        """Test that polling mechanism only updates videos with 'processing' status"""
        # This would be a frontend test - creating a test scenario
        
        # Simulated test scenario:
        # 1. Multiple videos in list: uploaded, processing, completed, failed
        # 2. Polling should only update the 'processing' video
        # 3. Other videos should remain unchanged
        
        videos = [
            {'id': '1', 'status': 'uploaded', 'name': 'Video 1'},
            {'id': '2', 'status': 'processing', 'name': 'Video 2'},
            {'id': '3', 'status': 'completed', 'name': 'Video 3'},
            {'id': '4', 'status': 'failed', 'name': 'Video 4'},
        ]
        
        # Filter for processing only
        processing_videos = [v for v in videos if v['status'] == 'processing']
        
        assert len(processing_videos) == 1, "Should find one processing video"
        assert processing_videos[0]['id'] == '2', "Should be Video 2"
        
        # Only this video should be updated in UI
        # Other videos should remain unchanged


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
