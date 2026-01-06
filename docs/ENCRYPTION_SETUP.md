# Encryption Setup for OpenAI API Keys

## Overview

OpenAI API keys are now **encrypted** before being stored in the database and **decrypted** when retrieved for use. This provides an additional layer of security for sensitive user data.

## How It Works

1. **When saving**: User's OpenAI API key is encrypted using Fernet (symmetric encryption) before storing in the database
2. **When retrieving**: The encrypted key is decrypted in memory only when needed for API calls
3. **In database**: Only the encrypted version is stored (never plaintext)

## Setup Instructions

### Step 1: Generate Encryption Key

Run the key generation script:

```bash
cd backend
python scripts/generate_encryption_key.py
```

This will output a key like:
```
WUFJLHqvNLwfKB3qZ2enwPwix0uSRMyC_llirry95dM=
```

### Step 2: Add to .env File

Add the encryption key to your `.env` file:

```env
ENCRYPTION_KEY=WUFJLHqvNLwfKB3qZ2enwPwix0uSRMyC_llirry95dM=
```

**⚠️ IMPORTANT**: 
- Keep this key **SECURE** and **NEVER** commit it to version control
- If you lose this key, you won't be able to decrypt existing encrypted API keys
- Use a different key for each environment (development, staging, production)

### Step 3: Install Dependencies

Make sure the `cryptography` package is installed:

```bash
pip install -r requirements.txt
```

The `cryptography==41.0.7` package is already added to `requirements.txt`.

### Step 4: Restart Backend Server

After adding the encryption key to your `.env` file, restart your backend server:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## How Encryption Works

### Encryption Service

The `EncryptionService` class in `backend/app/utils/encryption.py` handles all encryption/decryption:

- **Encrypt**: `EncryptionService.encrypt(plaintext)` - Encrypts a string
- **Decrypt**: `EncryptionService.decrypt(ciphertext)` - Decrypts an encrypted string
- **Generate Key**: `EncryptionService.generate_key()` - Generates a new encryption key

### Implementation Details

1. **Encryption Algorithm**: Fernet (symmetric encryption from the `cryptography` library)
2. **Key Format**: Base64-encoded 32-byte key (44 characters)
3. **Storage**: Encrypted keys are stored as base64 strings in the database

## Code Changes

### Backend Changes

1. **`backend/app/utils/encryption.py`**: New encryption service
2. **`backend/app/main.py`**: 
   - `PUT /api/settings/openai-key` - Encrypts API key before saving
3. **`backend/app/services/gpt_service.py`**: 
   - `_get_openai_client()` - Decrypts API key when retrieving from database
4. **`backend/app/config.py`**: 
   - Added `ENCRYPTION_KEY` setting

### Database

- The `openai_api_key` column stores **encrypted** values (base64 strings)
- Existing plaintext keys will need to be re-encrypted (users can update their keys)

## Security Notes

1. **Key Management**: 
   - Store the encryption key securely (environment variables, secret management service)
   - Never commit encryption keys to version control
   - Use different keys for different environments

2. **Key Rotation**:
   - If you need to rotate the encryption key, you'll need to:
     - Decrypt all existing keys with the old key
     - Re-encrypt with the new key
     - Update the `ENCRYPTION_KEY` in your environment

3. **Backup**:
   - Keep a secure backup of your encryption key
   - If lost, encrypted data cannot be recovered

## Testing

After setup, test the encryption:

1. Go to Settings page
2. Add your OpenAI API key
3. Check the database - the key should be encrypted (not readable)
4. Use the key for video processing - it should work normally

## Troubleshooting

### Error: "Failed to encrypt API key"

- Check that `ENCRYPTION_KEY` is set in your `.env` file
- Verify the key is 44 characters long (base64-encoded Fernet key)
- Check backend logs for detailed error messages

### Error: "Failed to decrypt data"

- Verify the `ENCRYPTION_KEY` matches the one used to encrypt the data
- If you changed the key, existing encrypted keys won't decrypt
- Users will need to re-enter their API keys

### Key Not Found

If `ENCRYPTION_KEY` is not set:
- The system will generate a new key automatically (development only)
- A warning will be logged
- **This is not recommended for production** - always set the key explicitly

