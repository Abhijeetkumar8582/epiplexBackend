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
                logger.error("Failed to exchange code for tokens", error=str(e), status_code=e.response.status_code)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to exchange authorization code"
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
        
        # Check if user exists by Google ID
        result = await db.execute(select(User).where(User.google_id == google_id))
        user = result.scalar_one_or_none()
        
        if user:
            # Update user info if needed
            if user.email != email:
                user.email = email
            if user.full_name != full_name:
                user.full_name = full_name
            await db.commit()
            await db.refresh(user)
            logger.info("User found by Google ID", user_id=str(user.id), email=email)
            return user
        
        # Check if user exists by email (might have signed up with email/password)
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        
        if user:
            # Link Google account to existing user
            if not user.google_id:
                user.google_id = google_id
                user.provider = "google"  # Update provider
                await db.commit()
                await db.refresh(user)
                logger.info("Linked Google account to existing user", user_id=str(user.id), email=email)
            return user
        
        # Create new user
        user = User(
            full_name=full_name,
            email=email,
            google_id=google_id,
            provider="google",
            password_hash=None,  # OAuth users don't have passwords
            role="user",
            is_active=True
        )
        
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        logger.info("User created from Google OAuth", user_id=str(user.id), email=email, google_id=google_id)
        return user
    
    @staticmethod
    async def authenticate_with_google(
        db: AsyncSession,
        code: str
    ) -> User:
        """Complete Google OAuth flow: exchange code, get user info, create/get user"""
        # Exchange code for tokens
        tokens = await GoogleOAuthService.exchange_code_for_tokens(code)
        access_token = tokens.get("access_token")
        
        if not access_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No access token received from Google"
            )
        
        # Get user info from Google
        google_user_info = await GoogleOAuthService.get_user_info(access_token)
        
        # Get or create user
        user = await GoogleOAuthService.get_or_create_user_from_google(db, google_user_info)
        
        return user

