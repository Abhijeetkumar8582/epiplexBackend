"""Real GPT-4 Vision service for frame analysis"""
import base64
import time
import json
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
        
        # Load prompt from file
        self.prompt_template = self._load_prompt_template()
        
        if not self.client:
            logger.warning("OpenAI API key not configured. GPT service will not work.")
    
    def _load_prompt_template(self) -> str:
        """Load prompt template from prompt.txt file"""
        try:
            prompt_file = Path(__file__).parent.parent.parent / "prompt.txt"
            logger.info("Loading prompt template", prompt_file=str(prompt_file))
            if prompt_file.exists():
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt_content = f.read().strip()
                    logger.info("Prompt template loaded successfully", 
                              prompt_length=len(prompt_content),
                              prompt_preview=prompt_content[:200])
                    print(f"[GPT Service] Prompt loaded from: {prompt_file}")
                    print(f"[GPT Service] Prompt length: {len(prompt_content)} characters")
                    return prompt_content
            else:
                # Default prompt if file doesn't exist
                logger.warning("prompt.txt not found, using default prompt", prompt_file=str(prompt_file))
                print(f"[GPT Service] WARNING: prompt.txt not found at {prompt_file}")
                return """Analyze this video frame and provide:
1. A detailed description of what you see (UI elements, text, layout, etc.)
2. Extract any visible text (OCR) from the frame
3. Identify any important information or data displayed

Frame timestamp: {timestamp} seconds"""
        except Exception as e:
            logger.error("Failed to load prompt template", error=str(e))
            print(f"[GPT Service] ERROR loading prompt: {str(e)}")
            return """Analyze this video frame and provide:
1. A detailed description of what you see (UI elements, text, layout, etc.)
2. Extract any visible text (OCR) from the frame
3. Identify any important information or data displayed

Frame timestamp: {timestamp} seconds"""
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _strip_markdown_code_blocks(self, text: str) -> str:
        """Strip markdown code blocks (```json ... ```) from text"""
        import re
        # Remove markdown code blocks
        text = re.sub(r'```json\s*\n?', '', text)
        text = re.sub(r'```\s*\n?', '', text)
        # Also handle cases where there might be ``` at the start/end
        text = text.strip()
        if text.startswith('```'):
            text = text[3:].strip()
        if text.endswith('```'):
            text = text[:-3].strip()
        return text.strip()
    
    def _repair_json(self, json_str: str) -> str:
        """Attempt to repair common JSON issues"""
        import re
        # Remove leading/trailing whitespace and newlines
        json_str = json_str.strip()
        
        # If empty, return empty
        if not json_str:
            return "{}"
        
        # Remove any leading newlines, spaces, or other whitespace
        json_str = json_str.lstrip('\n\r\t ')
        
        # If it doesn't start with {, try to find the JSON object
        if not json_str.startswith('{'):
            # Find first {
            start = json_str.find('{')
            if start != -1:
                json_str = json_str[start:]
            else:
                # No { found - might be just JSON content without braces
                # Check if it looks like JSON key-value pairs
                if '"' in json_str or 'timestamp' in json_str.lower() or 'description' in json_str.lower():
                    # Wrap in braces
                    json_str = '{' + json_str
                    # Try to add closing brace if missing
                    if json_str.count('{') > json_str.count('}'):
                        json_str = json_str + '}'
        
        # If it doesn't end with }, try to find the end
        if not json_str.endswith('}'):
            # Find last }
            end = json_str.rfind('}')
            if end != -1:
                json_str = json_str[:end + 1]
            else:
                # No } found - try to add it if we have an opening brace
                if json_str.startswith('{') and json_str.count('{') > json_str.count('}'):
                    json_str = json_str + '}'
        
        # Remove any trailing newlines or whitespace
        json_str = json_str.rstrip('\n\r\t ')
        
        # Final check - ensure it starts with { and ends with }
        if not json_str.startswith('{'):
            json_str = '{' + json_str
        if not json_str.endswith('}'):
            json_str = json_str + '}'
        
        return json_str.strip()
    
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
            
            # Format prompt with timestamp
            prompt_text = self.prompt_template.format(timestamp=timestamp_seconds)
            print(f"[GPT Service] Formatted prompt for timestamp: {timestamp_seconds}")
            print(f"[GPT Service] Prompt text length: {len(prompt_text)} characters")
            print(f"[GPT Service] Prompt preview: {prompt_text[:300]}...")
            
            # Always request JSON since our prompt explicitly asks for JSON format
            # The prompt.txt file specifies "strict JSON response"
            request_json = True  # Always true since prompt.txt requires JSON
            print(f"[GPT Service] Requesting JSON format: {request_json}")
            
            # Call GPT-4 Vision API
            api_params = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt_text
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
                "max_tokens": 1000
            }
            
            # Only request JSON if prompt explicitly asks for it
            if request_json:
                api_params["response_format"] = {"type": "json_object"}
                print(f"[GPT Service] Added response_format: json_object to API params")
            
            print(f"[GPT Service] Making API call to GPT-4o-mini...")
            print(f"[GPT Service] Image path: {image_path}")
            print(f"[GPT Service] Image size: {len(base64_image)} base64 characters")
            
            response = await self.client.chat.completions.create(**api_params)
            
            print(f"[GPT Service] API call completed successfully")
            
            # Parse response
            content = response.choices[0].message.content
            
            # Print and log raw response for debugging
            print(f"[GPT Service] ========== GPT RESPONSE RECEIVED ==========")
            print(f"[GPT Service] Response length: {len(content) if content else 0} characters")
            print(f"[GPT Service] Full response content:")
            print(content)
            print(f"[GPT Service] ============================================")
            
            logger.warning("Raw GPT response received",
                        content_preview=content[:500] if content else "None",
                        content_length=len(content) if content else 0,
                        timestamp=timestamp_seconds)
            
            # Initialize defaults
            description = ""
            ocr_text = None
            meta_tags = []
            
            # Since we're using response_format: {"type": "json_object"}, 
            # OpenAI should return valid JSON. Parse it directly.
            try:
                # Content should be valid JSON, but strip whitespace just in case
                content_cleaned = content.strip() if content else ""
                
                if not content_cleaned:
                    raise ValueError("Empty response from GPT API")
                
                # Try direct JSON parse first (should work with response_format)
                print(f"[GPT Service] Attempting to parse JSON response...")
                print(f"[GPT Service] Content to parse (first 200 chars): {content_cleaned[:200]}")
                try:
                    json_response = json.loads(content_cleaned)
                    print(f"[GPT Service] ✓ Direct JSON parse succeeded!")
                except json.JSONDecodeError as parse_err:
                    # If direct parse fails, try stripping markdown and repairing
                    print(f"[GPT Service] ✗ Direct JSON parse failed: {str(parse_err)}")
                    print(f"[GPT Service] Attempting repair...")
                    logger.warning("Direct JSON parse failed, attempting repair",
                                content_preview=content_cleaned[:200],
                                error=str(parse_err))
                    content_cleaned = self._strip_markdown_code_blocks(content_cleaned).strip()
                    
                    # Extract JSON object if embedded
                    json_start = content_cleaned.find('{')
                    json_end = content_cleaned.rfind('}')
                    
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        json_str = content_cleaned[json_start:json_end + 1]
                        print(f"[GPT Service] Extracted JSON from position {json_start} to {json_end}")
                    else:
                        json_str = content_cleaned
                        print(f"[GPT Service] Using full content as JSON string")
                    
                    # Repair and parse
                    print(f"[GPT Service] Repairing JSON string...")
                    json_str = self._repair_json(json_str)
                    print(f"[GPT Service] Repaired JSON (first 200 chars): {json_str[:200]}")
                    json_response = json.loads(json_str)
                    print(f"[GPT Service] ✓ JSON parse succeeded after repair!")
                    
                # Extract fields from JSON response according to prompt format
                # Expected format: {"timestamp": number, "description": string, "meta_tags": [string, string, string]}
                if not isinstance(json_response, dict):
                    raise ValueError(f"JSON response is not a dictionary, got {type(json_response).__name__}")
                
                # Get description (required field)
                description = json_response.get("description", "")
                if not description or not isinstance(description, str):
                    logger.warning("Missing or invalid description in GPT response", 
                                json_keys=list(json_response.keys()),
                                description_type=type(json_response.get("description")).__name__ if "description" in json_response else "missing")
                    description = description if description else "No description provided"
                
                # Get meta_tags (required field, should be array of exactly 3)
                meta_tags = json_response.get("meta_tags", [])
                if not isinstance(meta_tags, list):
                    logger.warning("meta_tags is not a list", 
                                meta_tags_type=type(meta_tags).__name__,
                                meta_tags_value=meta_tags)
                    meta_tags = []
                elif len(meta_tags) != 3:
                    logger.warning("meta_tags should have exactly 3 items", 
                                meta_tags_count=len(meta_tags),
                                meta_tags=meta_tags)
                    # Keep what we have, but log the issue
                
                # Get timestamp from response (optional, prompt asks for it)
                if "timestamp" in json_response:
                    response_ts = json_response.get("timestamp")
                    if isinstance(response_ts, (int, float)):
                        timestamp_seconds = float(response_ts)
                    else:
                        logger.warning("Timestamp in response is not a number", 
                                    timestamp_type=type(response_ts).__name__)
                
                # OCR text is not in the prompt format, but check just in case
                ocr_text = json_response.get("ocr_text") or json_response.get("text")
                
                print(f"[GPT Service] ========== PARSED JSON FIELDS ==========")
                print(f"[GPT Service] Description: {description[:100]}..." if len(description) > 100 else f"[GPT Service] Description: {description}")
                print(f"[GPT Service] Meta tags: {meta_tags}")
                print(f"[GPT Service] Meta tags count: {len(meta_tags)}")
                print(f"[GPT Service] Timestamp: {timestamp_seconds}")
                print(f"[GPT Service] JSON keys: {list(json_response.keys())}")
                print(f"[GPT Service] ========================================")
                
                logger.info("Successfully parsed JSON response from GPT",
                           has_description=bool(description),
                           description_length=len(description) if description else 0,
                           meta_tags_count=len(meta_tags) if meta_tags else 0,
                           timestamp=timestamp_seconds,
                           json_keys=list(json_response.keys()))
                        
            except json.JSONDecodeError as e:
                # JSON parsing failed - log detailed error
                error_msg = str(e)
                error_pos = getattr(e, 'pos', 'unknown')
                logger.error("Failed to parse JSON response from GPT",
                            error=error_msg,
                            error_position=error_pos,
                            content_preview=content[:500] if content else "None",
                            content_length=len(content) if content else 0,
                            timestamp=timestamp_seconds)
                logger.error("Full raw response content", content=content)
                
                # Try one final repair attempt
                try:
                    repaired = self._repair_json(content)
                    logger.warning("Attempting final JSON repair", repaired_preview=repaired[:200])
                    json_response = json.loads(repaired)
                    
                    if isinstance(json_response, dict):
                        description = json_response.get("description", "Error: Could not parse GPT response")
                        meta_tags = json_response.get("meta_tags", [])
                        if not isinstance(meta_tags, list):
                            meta_tags = []
                        logger.info("JSON repair succeeded after initial failure")
                    else:
                        raise ValueError("Repaired JSON is not a dictionary")
                except Exception as repair_error:
                    logger.error("JSON repair attempt also failed", repair_error=str(repair_error))
                    # Re-raise with clear error message
                    raise ValueError(f"Failed to parse GPT JSON response: {error_msg}. Content preview: {content[:200]}")
                    
            except Exception as e:
                # Unexpected error during JSON parsing
                logger.error("Unexpected error parsing JSON response",
                            error=str(e),
                            error_type=type(e).__name__,
                            content_preview=content[:500] if content else "None",
                            timestamp=timestamp_seconds,
                            exc_info=True)
                raise
            
            processing_time = int((time.time() - start_time) * 1000)
            
            result = {
                "description": description,
                "ocr_text": ocr_text,
                "meta_tags": meta_tags,  # Add meta_tags to result
                "processing_time_ms": processing_time,
                "model": "gpt-4o-mini",
                "timestamp": timestamp_seconds,
                "frame_number": frame_number
            }
            
            # Store full response in gpt_response for metadata
            if meta_tags:
                result["gpt_response"] = {
                    "description": description,
                    "ocr_text": ocr_text,
                    "meta_tags": meta_tags,
                    "timestamp": timestamp_seconds,
                    "frame_number": frame_number
                }
            
            logger.info("Frame analyzed with GPT", 
                       image_path=image_path,
                       processing_time_ms=processing_time,
                       has_ocr=ocr_text is not None)
            
            return result
            
        except Exception as e:
            error_str = str(e)
            # Clean up error message - remove confusing JSON decode details
            if "JSON parsing failed" in error_str or "Expecting" in error_str:
                # Extract a cleaner error message
                if "Response preview:" in error_str:
                    # Use the response preview part
                    error_str = "Invalid JSON response from GPT API"
                else:
                    error_str = "Failed to parse GPT response as JSON"
            
            logger.error("GPT frame analysis failed",
                        image_path=image_path,
                        error=error_str,
                        error_type=type(e).__name__,
                        exc_info=True)
            # Return error result with cleaner message
            return {
                "description": f"Error analyzing frame: {error_str}",
                "ocr_text": None,
                "meta_tags": None,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "error": error_str
            }
    
    async def analyze_frame_batch(
        self,
        frames_batch: List[Dict]
    ) -> List[Dict]:
        """
        Analyze a batch of frames in a single GPT API call (up to 10 frames)
        
        Args:
            frames_batch: List of frame dictionaries (max 10 frames)
        
        Returns:
            List of analyzed frames with GPT responses
        """
        if not self.client:
            raise ValueError("OpenAI API key not configured")
        
        if not frames_batch:
            return []
        
        start_time = time.time()
        
        try:
            # Prepare images for batch processing
            image_contents = []
            for frame in frames_batch:
                image_path = frame.get("image_path") or frame.get("frame_path")
                if not image_path:
                    continue
                
                async with aiofiles.open(image_path, 'rb') as f:
                    image_data = await f.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
            
            if not image_contents:
                logger.warning("No valid images in batch")
                return frames_batch
            
            # Create prompt that asks for analysis of all frames
            timestamps = [f.get("timestamp", 0.0) for f in frames_batch]
            prompt_text = f"""Analyze these {len(frames_batch)} video frames and provide a JSON response for EACH frame.

For each frame, provide:
1. A detailed description of what you see (UI elements, text, layout, etc.)
2. Extract any visible text (OCR) from the frame
3. Three meta tags that describe the frame content

IMPORTANT: Return a JSON object with a "frames" key containing an array. Each element in the array corresponds to a frame in order:
{{
  "frames": [
    {{
      "timestamp": <timestamp_in_seconds>,
      "description": "<detailed_description>",
      "ocr_text": "<extracted_text_or_null>",
      "meta_tags": ["tag1", "tag2", "tag3"]
    }},
    ...
  ]
}}

Frame timestamps (in order): {', '.join([str(ts) for ts in timestamps])}"""
            
            # Call GPT-4 Vision API with multiple images
            api_params = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt_text
                            }
                        ] + image_contents
                    }
                ],
                "max_tokens": 4000,  # Increased for batch processing
                "response_format": {"type": "json_object"}
            }
            
            logger.info("Making batch GPT API call", 
                       frame_count=len(frames_batch),
                       image_count=len(image_contents))
            
            # Add timeout to prevent hanging
            import asyncio
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(**api_params),
                    timeout=300.0  # 5 minute timeout per batch
                )
            except asyncio.TimeoutError:
                logger.error("GPT API call timed out after 5 minutes",
                           batch_size=len(frames_batch))
                raise Exception("GPT API call timed out after 5 minutes. Batch may be too large or API is slow.")
            
            # Parse response
            content = response.choices[0].message.content
            processing_time = int((time.time() - start_time) * 1000)
            
            # Parse JSON response
            try:
                content_cleaned = content.strip() if content else ""
                json_response = json.loads(content_cleaned)
                
                logger.info("Parsed JSON response", 
                           response_type=type(json_response).__name__,
                           has_frames_key="frames" in json_response if isinstance(json_response, dict) else False,
                           keys=list(json_response.keys()) if isinstance(json_response, dict) else None)
                
                # Handle response format - expect object with "frames" array
                if isinstance(json_response, dict) and "frames" in json_response:
                    results = json_response["frames"]
                    if not isinstance(results, list):
                        logger.warning("Frames key exists but is not a list", frames_type=type(results).__name__)
                        results = [results] if results else []
                elif isinstance(json_response, list):
                    # Fallback: if we get an array directly, use it
                    logger.warning("Received array instead of object with frames key")
                    results = json_response
                elif isinstance(json_response, dict):
                    # Try to find frame data in other keys
                    logger.warning("No 'frames' key found, checking for alternative structure", keys=list(json_response.keys()))
                    # Check if it's a single frame response
                    if "timestamp" in json_response or "description" in json_response:
                        results = [json_response]
                    else:
                        # Try to extract any array from the response
                        for key, value in json_response.items():
                            if isinstance(value, list):
                                results = value
                                logger.info("Found array in key", key=key, array_length=len(value))
                                break
                        else:
                            raise ValueError(f"Could not find frames array in response. Keys: {list(json_response.keys())}")
                else:
                    raise ValueError(f"Unexpected response format: {type(json_response)}")
                
                logger.info("Extracted results", results_count=len(results), expected_count=len(frames_batch))
                
                # Map results back to frames
                analyzed_frames = []
                for i, frame in enumerate(frames_batch):
                    if i < len(results):
                        result = results[i]
                        frame.update({
                            "description": result.get("description", ""),
                            "ocr_text": result.get("ocr_text"),
                            "meta_tags": result.get("meta_tags", []),
                            "processing_time_ms": processing_time // len(frames_batch),  # Divide time per frame
                            "gpt_response": result
                        })
                    else:
                        # Missing result for this frame
                        frame.update({
                            "description": "No analysis result",
                            "ocr_text": None,
                            "meta_tags": [],
                            "processing_time_ms": 0,
                            "error": "Missing result in batch response"
                        })
                    analyzed_frames.append(frame)
                
                logger.info("Batch frame analysis completed successfully",
                           batch_size=len(frames_batch),
                           processing_time_ms=processing_time,
                           results_count=len(analyzed_frames))
                
                return analyzed_frames
                
            except json.JSONDecodeError as e:
                logger.error("Failed to parse batch JSON response",
                           error=str(e),
                           content_preview=content[:500])
                # Return frames with error
                for frame in frames_batch:
                    frame.update({
                        "description": f"Error parsing batch response: {str(e)}",
                        "ocr_text": None,
                        "meta_tags": [],
                        "processing_time_ms": processing_time // len(frames_batch),
                        "error": str(e)
                    })
                return frames_batch
                
        except Exception as e:
            logger.error("Batch frame analysis failed",
                        error=str(e),
                        batch_size=len(frames_batch),
                        exc_info=True)
            # Return frames with error
            for frame in frames_batch:
                frame.update({
                    "description": f"Error in batch analysis: {str(e)}",
                    "ocr_text": None,
                    "meta_tags": [],
                    "processing_time_ms": 0,
                    "error": str(e)
                })
            return frames_batch
    
    async def batch_analyze_frames(
        self,
        frames: List[Dict],
        max_workers: int = 5,
        batch_size: int = 10
    ) -> List[Dict]:
        """
        Analyze multiple frames in batches (production-ready with error handling)
        
        Args:
            frames: List of frame dictionaries with 'image_path', 'timestamp', etc.
            max_workers: Maximum number of concurrent batch API calls (default: 5)
            batch_size: Number of frames to send in each batch (default: 10)
        
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
                    "meta_tags": [],
                    "processing_time_ms": 0,
                    "error": "OpenAI API key not configured"
                })
            return frames
        
        if not frames:
            return []
        
        # Split frames into batches
        frame_batches = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            frame_batches.append(batch)
        
        logger.info("Processing frames in batches",
                   total_frames=len(frames),
                   batch_size=batch_size,
                   total_batches=len(frame_batches))
        
        # Create semaphore to limit concurrent batch API calls
        semaphore = asyncio.Semaphore(max_workers)
        
        async def analyze_batch_with_semaphore(batch):
            async with semaphore:
                try:
                    return await self.analyze_frame_batch(batch)
                except Exception as e:
                    logger.error("Batch analysis exception",
                               batch_size=len(batch),
                               error=str(e))
                    # Return frames with error
                    for frame in batch:
                        frame.update({
                            "description": f"Error analyzing batch: {str(e)}",
                            "ocr_text": None,
                            "meta_tags": [],
                            "processing_time_ms": 0,
                            "error": str(e)
                        })
                    return batch
        
        # Process all batches concurrently
        tasks = [analyze_batch_with_semaphore(batch) for batch in frame_batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        analyzed_frames = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error("Batch processing failed", error=str(batch_result))
                # Create error frames for this batch
                for frame in frames[len(analyzed_frames):len(analyzed_frames) + batch_size]:
                    frame.update({
                        "description": f"Error: {str(batch_result)}",
                        "ocr_text": None,
                        "meta_tags": [],
                        "processing_time_ms": 0,
                        "error": str(batch_result)
                    })
                    analyzed_frames.append(frame)
            else:
                analyzed_frames.extend(batch_result)
        
        # Sort by timestamp to maintain order
        analyzed_frames.sort(key=lambda x: x.get("timestamp", 0))
        
        successful = sum(1 for f in analyzed_frames if "error" not in f or not f.get("error"))
        logger.info("Batch frame analysis completed",
                   total_frames=len(analyzed_frames),
                   successful=successful,
                   batches_processed=len(frame_batches))
        
        return analyzed_frames

