"""
Test script for frame analysis API
Tests the GPT service directly with a sample frame
"""
import asyncio
import sys
from pathlib import Path

# Add backend directory to path (parent of tests folder)
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.gpt_service import GPTService
from app.config import settings


async def test_frame_analysis():
    """Test frame analysis with a sample frame"""
    print("=" * 60)
    print("FRAME ANALYSIS TEST")
    print("=" * 60)
    
    # Check if API key is configured
    if not hasattr(settings, 'OPENAI_API_KEY') or not settings.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not configured in settings")
        print("Please set OPENAI_API_KEY in your .env file")
        return
    
    print(f"✓ OpenAI API Key configured: {settings.OPENAI_API_KEY[:10]}...")
    
    # Initialize GPT service
    print("\n[1] Initializing GPT Service...")
    gpt_service = GPTService()
    
    if not gpt_service.client:
        print("ERROR: GPT Service client not initialized")
        return
    
    print("✓ GPT Service initialized")
    print(f"✓ Prompt template loaded: {len(gpt_service.prompt_template)} characters")
    print(f"✓ Prompt preview: {gpt_service.prompt_template[:200]}...")
    
    # Find a test frame (frames directory is in backend root, not tests folder)
    frames_dir = Path(__file__).parent.parent / "frames"
    test_frame = None
    
    print("\n[2] Looking for test frames...")
    for video_dir in frames_dir.iterdir():
        if video_dir.is_dir():
            frames = list(video_dir.glob("frame_*.jpg"))
            if frames:
                test_frame = frames[0]
                print(f"✓ Found test frame: {test_frame}")
                break
    
    if not test_frame:
        print("ERROR: No test frames found in frames/ directory")
        print("Please process a video first to generate frames")
        return
    
    # Test frame analysis
    print("\n[3] Testing frame analysis...")
    print(f"   Frame path: {test_frame}")
    print(f"   Frame exists: {test_frame.exists()}")
    
    try:
        # Analyze the frame
        result = await gpt_service.analyze_frame(
            image_path=str(test_frame),
            timestamp_seconds=0.0,
            frame_number=0
        )
        
        print("\n[4] Analysis Result:")
        print("=" * 60)
        print(f"Description: {result.get('description', 'N/A')}")
        meta_tags = result.get('meta_tags') or []
        print(f"Meta tags: {meta_tags}")
        print(f"Meta tags count: {len(meta_tags)}")
        print(f"Timestamp: {result.get('timestamp', 'N/A')}")
        print(f"Processing time: {result.get('processing_time_ms', 'N/A')} ms")
        print(f"Error: {result.get('error', 'None')}")
        print("=" * 60)
        
        # Check if analysis was successful
        if result.get('error'):
            print(f"\n✗ Analysis failed with error: {result.get('error')}")
            return False
        elif result.get('description'):
            print(f"\n✓ Analysis successful!")
            if result.get('meta_tags'):
                print(f"✓ Meta tags extracted: {len(result.get('meta_tags'))} tags")
            return True
        else:
            print(f"\n⚠ Analysis completed but no description found")
            return False
            
    except Exception as e:
        print(f"\n✗ Exception during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\nStarting frame analysis test...\n")
    success = asyncio.run(test_frame_analysis())
    
    if success:
        print("\n✓ Test completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Test failed!")
        sys.exit(1)

