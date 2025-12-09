#!/usr/bin/env python3
"""
Startup script for the video processing backend
"""
import os
import sys
from pathlib import Path

# Check for .env file
env_file = Path(".env")
if not env_file.exists():
    print("⚠️  Warning: .env file not found!")
    print("Please create a .env file with your OPENAI_API_KEY")
    print()
    print("Quick setup:")
    print("  cp env.example .env")
    print("  # Then edit .env and add your OPENAI_API_KEY")
    print()
    print("Or set environment variables:")
    print("  OPENAI_API_KEY=your_key_here")
    print("  UPLOAD_DIR=./uploads")
    print("  OUTPUT_DIR=./outputs")
    print()
    
    # Check if OPENAI_API_KEY is set in environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables either.")
        print("Please set it before starting the server.")
        sys.exit(1)
    else:
        print("✓ Using OPENAI_API_KEY from environment variables")

# Check for required directories
upload_dir = Path(os.getenv("UPLOAD_DIR", "./uploads"))
output_dir = Path(os.getenv("OUTPUT_DIR", "./outputs"))
upload_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)
print(f"✓ Upload directory: {upload_dir.absolute()}")
print(f"✓ Output directory: {output_dir.absolute()}")

# Start the server
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting FastAPI server...")
    print("📡 API will be available at http://localhost:8000")
    print("📚 API docs at http://localhost:8000/docs\n")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

