"""
Script to generate a Fernet encryption key for encrypting sensitive data
Run this script to generate a secure encryption key for your .env file
"""
import sys
from pathlib import Path

# Add backend directory to path (parent of scripts folder)
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.encryption import EncryptionService

if __name__ == "__main__":
    print("=" * 60)
    print("Generating Encryption Key for OpenAI API Keys")
    print("=" * 60)
    print()
    
    key = EncryptionService.generate_key()
    
    print("Generated encryption key:")
    print("-" * 60)
    print(key)
    print("-" * 60)
    print()
    print("Add this to your .env file:")
    print(f"ENCRYPTION_KEY={key}")
    print()
    print("=" * 60)
    print("IMPORTANT: Keep this key secure and never commit it to version control!")
    print("=" * 60)

