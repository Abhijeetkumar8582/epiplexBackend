"""
Encryption utilities for sensitive data like API keys
Uses Fernet (symmetric encryption) from the cryptography library
"""
from cryptography.fernet import Fernet
from typing import Optional
import base64
import os

from app.config import settings
from app.utils.logger import logger


class EncryptionService:
    """Service for encrypting and decrypting sensitive data"""
    
    _fernet: Optional[Fernet] = None
    
    @classmethod
    def _get_fernet(cls) -> Fernet:
        """Get or create Fernet instance with encryption key"""
        if cls._fernet is not None:
            return cls._fernet
        
        # Get encryption key from environment or settings
        encryption_key = getattr(settings, 'ENCRYPTION_KEY', None)
        
        if not encryption_key:
            # Try to get from environment variable
            encryption_key = os.getenv('ENCRYPTION_KEY')
        
        if not encryption_key:
            # Generate a new key (for development only - should be set in production)
            logger.warning("ENCRYPTION_KEY not set. Generating a new key. This should be set in production!")
            encryption_key = Fernet.generate_key().decode()
            logger.warning(f"Generated encryption key: {encryption_key}")
            logger.warning("IMPORTANT: Set this key in your .env file as ENCRYPTION_KEY for production use!")
        
        # Fernet keys are base64-encoded 32-byte keys (44 characters)
        # They should be used directly as bytes
        if isinstance(encryption_key, str):
            # Fernet keys are already base64-encoded strings (44 chars)
            # Convert to bytes
            key_bytes = encryption_key.encode('utf-8')
            
            # Validate it's a valid Fernet key format (44 characters)
            if len(key_bytes) != 44:
                logger.warning("Invalid encryption key length. Generating a new key.")
                key_bytes = Fernet.generate_key()
            else:
                # Test if it's a valid Fernet key by trying to create a Fernet instance
                try:
                    test_fernet = Fernet(key_bytes)
                    cls._fernet = test_fernet
                    return cls._fernet
                except Exception as e:
                    logger.warning(f"Invalid encryption key format: {e}. Generating a new key.")
                    key_bytes = Fernet.generate_key()
        else:
            # Already bytes
            key_bytes = encryption_key
        
        cls._fernet = Fernet(key_bytes)
        return cls._fernet
    
    @classmethod
    def encrypt(cls, plaintext: str) -> Optional[str]:
        """
        Encrypt a plaintext string
        
        Args:
            plaintext: The string to encrypt
            
        Returns:
            Encrypted string (base64-encoded) or None if encryption fails
        """
        if not plaintext:
            return None
        
        try:
            fernet = cls._get_fernet()
            encrypted_bytes = fernet.encrypt(plaintext.encode())
            # Return as base64 string for easy storage in database
            return encrypted_bytes.decode()
        except Exception as e:
            logger.error("Failed to encrypt data", error=str(e), exc_info=True)
            return None
    
    @classmethod
    def decrypt(cls, ciphertext: str) -> Optional[str]:
        """
        Decrypt an encrypted string
        
        Args:
            ciphertext: The encrypted string (base64-encoded)
            
        Returns:
            Decrypted plaintext string or None if decryption fails
        """
        if not ciphertext:
            return None
        
        try:
            fernet = cls._get_fernet()
            decrypted_bytes = fernet.decrypt(ciphertext.encode())
            return decrypted_bytes.decode()
        except Exception as e:
            logger.error("Failed to decrypt data", error=str(e), exc_info=True)
            return None
    
    @classmethod
    def generate_key(cls) -> str:
        """
        Generate a new encryption key (for initial setup)
        
        Returns:
            Base64-encoded encryption key string
        """
        key = Fernet.generate_key()
        return key.decode()

