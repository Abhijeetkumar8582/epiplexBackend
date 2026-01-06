from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, Dict, Any, Tuple
import httpx
from urllib.parse import urlencode
import secrets
from uuid import UUID

from app.config import settings
from app.database import User
from app.utils.logger import logger
from fastapi import HTTPException, status


class GoogleOAuthService:
    # Google OAuth2 endpoints
    GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
    GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
    
    @staticmethod
    def get_authorization_url(state: Optional[str] = None) -> Tuple[str, str]:
        """Generate Google OAuth2 authorization URL"""
        if not settings.GOOGLE_CLIENT_ID:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Google OAuth2 not configured"
            )
        
        # Generate state if not provided (for CSRF protection)
        if not state:
            state = secrets.token_urlsafe(32)
        
        params = {
            "client_id": settings.GOOGLE_CLIENT_ID,
            "redirect_uri": settings.GOOGLE_REDIRECT_URI,
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "offline",
            "prompt": "consent",
            "state": state
        }
        
        auth_url = f"{GoogleOAuthService.GOOGLE_AUTH_URL}?{urlencode(params)}"
        return auth_url, state
    
    @staticmethod
    async def exchange_code_for_tokens(code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token"""
        if not settings.GOOGLE_CLIENT_ID or not settings.GOOGLE_CLIENT_SECRET:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Google OAuth2 not configured"
            )
        
        data = {
            "code": code,
            "client_id": settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "redirect_uri": settings.GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code"
        }
        
        # Log the request details (without sensitive data)
        logger.info("Exchanging code for tokens", 
                   client_id=settings.GOOGLE_CLIENT_ID[:20] + "..." if settings.GOOGLE_CLIENT_ID else None,
                   redirect_uri=settings.GOOGLE_REDIRECT_URI,
                   code_length=len(code) if code else 0)
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    GoogleOAuthService.GOOGLE_TOKEN_URL,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                # Log the actual error response from Google
                error_detail = "Unknown error"
                try:
                    error_response = e.response.json()
                    error_detail = error_response.get("error_description", error_response.get("error", str(e)))
                    logger.error("Failed to exchange code for tokens", 
                               error=error_detail, 
                               status_code=e.response.status_code,
                               error_response=error_response,
                               redirect_uri=settings.GOOGLE_REDIRECT_URI)
                except:
                    error_detail = e.response.text or str(e)
                    logger.error("Failed to exchange code for tokens", 
                               error=error_detail, 
                               status_code=e.response.status_code,
                               redirect_uri=settings.GOOGLE_REDIRECT_URI)
                
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to exchange authorization code: {error_detail}"
                )
            except Exception as e:
                logger.error("Unexpected error exchanging code for tokens", error=str(e), exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Unexpected error: {str(e)}"
                )
    
    @staticmethod
    async def get_user_info(access_token: str) -> Dict[str, Any]:
        """Get user information from Google"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    GoogleOAuthService.GOOGLE_USERINFO_URL,
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error("Failed to get user info from Google", error=str(e), status_code=e.response.status_code)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to get user information from Google"
                )
    
    @staticmethod
    async def get_or_create_user_from_google(
        db: AsyncSession,
        google_user_info: Dict[str, Any]
    ) -> User:
        """Get existing user or create new user from Google OAuth data"""
        google_id = google_user_info.get("id")
        email = google_user_info.get("email")
        full_name = google_user_info.get("name", "")
        picture = google_user_info.get("picture")
        
        if not google_id or not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Google user information"
            )
        
        # Normalize email
        normalized_email = email.lower().strip()
        
        # Check if user exists by Google ID
        result = await db.execute(select(User).where(User.google_id == google_id))
        user = result.scalar_one_or_none()
        
        if user:
            # Update user info if needed
            if user.email != normalized_email:
                user.email = normalized_email
            if user.full_name != full_name:
                user.full_name = full_name
            await db.commit()
            await db.refresh(user)
            logger.info("User found by Google ID", user_id=str(user.id), email=normalized_email)
            return user
        
        # Check if user exists by email (might have signed up with email/password) - case-insensitive
        from sqlalchemy import func
        result = await db.execute(
            select(User).where(func.lower(User.email) == normalized_email)
        )
        user = result.scalar_one_or_none()
        
        if user:
            # Link Google account to existing user
            if not user.google_id:
                user.google_id = google_id
                user.provider = "google"  # Update provider
                await db.commit()
                await db.refresh(user)
                logger.info("Linked Google account to existing user", user_id=str(user.id), email=normalized_email)
            return user
        
        # Create new user
        user = User(
            full_name=full_name,
            email=normalized_email,  # Store normalized email
            google_id=google_id,
            provider="google",
            password_hash=None,  # OAuth users don't have passwords
            role="user",
            is_active=True
        )
        
        try:
            db.add(user)
            await db.commit()
            await db.refresh(user)
            
            logger.info("User created from Google OAuth", user_id=str(user.id), email=normalized_email, google_id=google_id)
            return user
        except Exception as e:
            await db.rollback()
            # Check if it's a unique constraint violation
            if "unique" in str(e).lower() or "duplicate" in str(e).lower():
                # Try to get the existing user
                result = await db.execute(
                    select(User).where(func.lower(User.email) == normalized_email)
                )
                existing_user = result.scalar_one_or_none()
                if existing_user:
                    # Link Google account
                    if not existing_user.google_id:
                        existing_user.google_id = google_id
                        existing_user.provider = "google"
                        await db.commit()
                        await db.refresh(existing_user)
                    return existing_user
            logger.error("Error creating user from Google OAuth", error=str(e), email=normalized_email)
            raise
    
    @staticmethod
    async def authenticate_with_google(
        db: AsyncSession,
        code: str
    ) -> User:
        """Complete Google OAuth flow: exchange code, get user info, create/get user"""
        try:
            # Exchange code for tokens
            tokens = await GoogleOAuthService.exchange_code_for_tokens(code)
            access_token = tokens.get("access_token")
            
            if not access_token:
                logger.error("No access token in response from Google", tokens=tokens)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No access token received from Google"
                )
            
            # Get user info from Google
            google_user_info = await GoogleOAuthService.get_user_info(access_token)
            
            # Get or create user
            user = await GoogleOAuthService.get_or_create_user_from_google(db, google_user_info)
            
            return user
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error in authenticate_with_google", error=str(e), exc_info=True)
            raise

