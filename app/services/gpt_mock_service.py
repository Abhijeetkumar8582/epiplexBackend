"""Mock GPT-4 Vision service for frame analysis"""
import random
import time
from typing import Dict, Optional
import numpy as np
from PIL import Image
import io

from app.utils.logger import logger


class GPTMockService:
    """Mock service that simulates GPT-4 Vision API calls"""
    
    # Mock captions pool for variety
    CAPTIONS_POOL = [
        "A user interface showing a dashboard with multiple widgets and data visualizations.",
        "A form with input fields and buttons, likely for data entry or configuration.",
        "A table displaying rows of data with columns for various attributes.",
        "A navigation menu with multiple options and submenus visible.",
        "A dialog box or modal window with confirmation buttons.",
        "A chart or graph displaying statistical data in visual format.",
        "A login screen with username and password input fields.",
        "A settings page with various configuration options and toggles.",
        "A list view showing multiple items with details and action buttons.",
        "A workflow diagram or process visualization with connected nodes.",
        "A search interface with filters and result display area.",
        "A notification panel showing alerts and messages.",
        "A user profile page with personal information and preferences.",
        "A calendar view with scheduled events and appointments.",
        "A file browser or document management interface.",
    ]
    
    # Mock OCR text pool
    OCR_TEXT_POOL = [
        "User ID: 12345\nStatus: Active\nDate: 2024-01-15",
        "Total Amount: $1,234.56\nPayment Method: Credit Card\nTransaction ID: TXN-789",
        "Employee Name: John Doe\nDepartment: HR\nPosition: Manager",
        "Order #: ORD-12345\nItems: 3\nTotal: $99.99",
        "Customer: Jane Smith\nEmail: jane@example.com\nPhone: +1-555-0123",
        "Product: Widget Pro\nSKU: WID-001\nPrice: $49.99",
        "Invoice #: INV-456\nDue Date: 2024-02-01\nAmount Due: $500.00",
        "Task: Complete Report\nAssignee: Team A\nDue: 2024-01-20",
        "Account Balance: $10,000.00\nLast Transaction: 2024-01-10\nStatus: Verified",
        "Report Generated: 2024-01-15\nTotal Records: 1,234\nFilter: Active Only",
    ]
    
    def __init__(self, simulate_delay: bool = True, delay_range_ms: tuple = (100, 500)):
        """
        Initialize mock GPT service
        
        Args:
            simulate_delay: Whether to simulate API call delay
            delay_range_ms: Range of delay in milliseconds (min, max)
        """
        self.simulate_delay = simulate_delay
        self.delay_range = delay_range_ms
    
    def analyze_frame(
        self,
        frame_data: np.ndarray,
        image_path: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Mock GPT-4 Vision analysis of a frame
        
        Args:
            frame_data: Frame as numpy array (from OpenCV)
            image_path: Optional path to saved frame image
        
        Returns:
            Dictionary with 'description' and 'ocr_text'
        """
        start_time = time.time()
        
        # Simulate API call delay
        if self.simulate_delay:
            delay_ms = random.randint(*self.delay_range)
            time.sleep(delay_ms / 1000.0)
        
        # Generate mock description (randomly select from pool)
        description = random.choice(self.CAPTIONS_POOL)
        
        # Add some variation based on frame characteristics
        if frame_data is not None:
            height, width = frame_data.shape[:2]
            if width > height:
                description += " The interface appears to be in landscape orientation."
            else:
                description += " The interface appears to be in portrait orientation."
        
        # Generate mock OCR text (randomly select from pool)
        ocr_text = random.choice(self.OCR_TEXT_POOL)
        
        # Add some randomness to OCR text
        if random.random() > 0.7:  # 30% chance of no OCR text
            ocr_text = None
        
        processing_time = int((time.time() - start_time) * 1000)
        
        result = {
            "description": description,
            "ocr_text": ocr_text,
            "processing_time_ms": processing_time
        }
        
        logger.debug("Frame analyzed (mock)", 
                    processing_time_ms=processing_time,
                    has_ocr=ocr_text is not None)
        
        return result
    
    async def analyze_frame_async(
        self,
        frame_data: np.ndarray,
        image_path: Optional[str] = None
    ) -> Dict[str, str]:
        """Async wrapper for frame analysis"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.analyze_frame,
            frame_data,
            image_path
        )
    
    def batch_analyze_frames(
        self,
        frames: List[Dict],
        max_workers: int = 4
    ) -> List[Dict]:
        """
        Analyze multiple frames in parallel
        
        Args:
            frames: List of frame dictionaries with 'frame_data' key
            max_workers: Number of parallel workers
        
        Returns:
            List of analysis results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all frame analysis tasks
            future_to_frame = {
                executor.submit(
                    self.analyze_frame,
                    frame.get("frame_data"),
                    frame.get("image_path")
                ): frame
                for frame in frames
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_frame):
                frame = future_to_frame[future]
                try:
                    analysis = future.result()
                    # Merge analysis results with frame data
                    frame.update(analysis)
                    results.append(frame)
                except Exception as e:
                    logger.error("Frame analysis failed", 
                               frame_timestamp=frame.get("timestamp"),
                               error=str(e))
                    # Add error result
                    frame.update({
                        "description": f"Error analyzing frame: {str(e)}",
                        "ocr_text": None,
                        "processing_time_ms": 0
                    })
                    results.append(frame)
        
        # Sort results by timestamp
        results.sort(key=lambda x: x.get("timestamp", 0))
        
        return results

