from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import bcrypt
from jose import JWTError, jwt
from datetime import datetime, timedelta
import secrets
from typing import Optional
from uuid import UUID

from app.config import settings
from app.database import User, UserSession
from app.utils.logger import logger
from fastapi import HTTPException, status


class AuthService:
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        # Ensure password is a string
        if not isinstance(password, str):
            password = str(password)
        
        # Bcrypt has a 72-byte limit - truncate password to 72 bytes if needed
        password_bytes = password.encode('utf-8')
        if len(password_bytes) > 72:
            # Truncate to exactly 72 bytes
            password_bytes = password_bytes[:72]
            # Decode back, handling any incomplete UTF-8 sequences
            try:
                password = password_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # If decode fails, remove last byte and try again
                password_bytes = password_bytes[:-1]
                password = password_bytes.decode('utf-8', errors='ignore')
        
        # Hash using bcrypt directly (more reliable than passlib for this use case)
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against a hash"""
        try:
            # Truncate password if needed (bcrypt 72-byte limit)
            password_bytes = plain_password.encode('utf-8')
            if len(password_bytes) > 72:
                password_bytes = password_bytes[:72]
                plain_password = password_bytes.decode('utf-8', errors='ignore')
            
            return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception as e:
            logger.error("Password verification error", error=str(e))
            return False
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def generate_session_token() -> str:
        """Generate a random session token"""
        return secrets.token_urlsafe(settings.SESSION_TOKEN_LENGTH)
    
    @staticmethod
    async def create_user(
        db: AsyncSession,
        full_name: str,
        email: str,
        password: str
    ) -> User:
        """Create a new user"""
        # Normalize email
        normalized_email = email.lower().strip()
        
        # Double-check if user already exists (case-insensitive)
        # Check with both normalized and original email to catch any edge cases
        existing_user = await AuthService.get_user_by_email(db, normalized_email)
        if existing_user:
            logger.warning("User creation blocked - email already exists", 
                          email=normalized_email, 
                          existing_user_id=str(existing_user.id))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Also check with original email (in case database has different case)
        # This is a safety check for SQL Server case-sensitivity issues
        from sqlalchemy import func, or_
        result = await db.execute(
            select(User).where(
                or_(
                    func.lower(User.email) == normalized_email,
                    User.email == email.strip()  # Also check exact match
                )
            )
        )
        existing_user_alt = result.scalar_one_or_none()
        if existing_user_alt:
            logger.warning("User creation blocked - email exists (alternative check)", 
                          email=normalized_email, 
                          existing_user_id=str(existing_user_alt.id))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Hash password
        password_hash = AuthService.hash_password(password)
        
        # Create user
        # Generate ID explicitly to ensure compatibility with SQLite (strings) and PostgreSQL/SQL Server (UUID objects)
        from app.database import generate_uuid
        user_id = generate_uuid()
        
        user = User(
            id=user_id,
            full_name=full_name,
            email=normalized_email,  # Store normalized email
            password_hash=password_hash,
            provider='email',
            role='user'
        )
        
        try:
            db.add(user)
            await db.commit()
            await db.refresh(user)
            
            logger.info("User created successfully", user_id=str(user.id), email=normalized_email)
            return user
        except Exception as e:
            await db.rollback()
            error_str = str(e).lower()
            
            # Check if it's a unique constraint violation (email already exists)
            if any(keyword in error_str for keyword in [
                "unique", "duplicate", "already exists", 
                "violation", "constraint", "idx_users_email",
                "cannot insert duplicate", "duplicate key"
            ]):
                # Double-check the database one more time
                final_check = await AuthService.get_user_by_email(db, normalized_email)
                if final_check:
                    logger.warning("User creation failed - email exists in database", 
                                  email=normalized_email,
                                  existing_user_id=str(final_check.id))
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Email already registered"
                    )
                else:
                    # This shouldn't happen, but log it
                    logger.error("Unique constraint violation but user not found on retry", 
                               email=normalized_email, error=str(e))
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Email already registered"
                    )
            
            # Re-raise other exceptions
            logger.error("Error creating user", error=str(e), email=normalized_email, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create user: {str(e) if settings.DEBUG else 'Internal server error'}"
            )
    
    @staticmethod
    async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
        """Get user by email (case-insensitive)"""
        # Normalize email to lowercase for comparison
        # SQL Server comparisons are case-insensitive by default, but be explicit
        from sqlalchemy import func
        normalized_email = email.lower().strip()
        result = await db.execute(
            select(User).where(func.lower(User.email) == normalized_email)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_user_by_id(db: AsyncSession, user_id: UUID) -> Optional[User]:
        """Get user by ID"""
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
    
    @staticmethod
    async def authenticate_user(
        db: AsyncSession,
        email: str,
        password: str
    ) -> Optional[User]:
        """Authenticate a user with email and password"""
        # Normalize email for lookup
        normalized_email = email.lower().strip()
        logger.info("Attempting authentication", email=normalized_email)
        
        user = await AuthService.get_user_by_email(db, normalized_email)
        if not user:
            logger.warning("Authentication failed - user not found", email=normalized_email)
            return None
        
        logger.info("User found", user_id=str(user.id), email=user.email, has_password=bool(user.password_hash))
        
        if not user.is_active:
            logger.warning("Authentication failed - user inactive", user_id=str(user.id), email=user.email)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is inactive"
            )
        
        # Check if user has a password (OAuth users don't have passwords)
        if not user.password_hash:
            logger.warning("Authentication failed - OAuth user trying password login", user_id=str(user.id), email=user.email)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This account was created with Google. Please sign in with Google."
            )
        
        # Verify password
        password_valid = AuthService.verify_password(password, user.password_hash)
        if not password_valid:
            logger.warning("Authentication failed - incorrect password", user_id=str(user.id), email=user.email)
            return None
        
        logger.info("Authentication successful", user_id=str(user.id), email=user.email)
        return user
    
    @staticmethod
    async def create_session(
        db: AsyncSession,
        user_id: UUID,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        expires_in_days: int = 7
    ) -> UserSession:
        """Create a new user session"""
        session_token = AuthService.generate_session_token()
        expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        session = UserSession(
            user_id=user_id,
            session_token=session_token,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at
        )
        
        db.add(session)
        await db.commit()
        await db.refresh(session)
        
        logger.info("Session created", user_id=str(user_id), session_id=str(session.id))
        return session
    
    @staticmethod
    async def get_session_by_token(
        db: AsyncSession,
        session_token: str
    ) -> Optional[UserSession]:
        """Get session by token"""
        result = await db.execute(
            select(UserSession).where(UserSession.session_token == session_token)
        )
        session = result.scalar_one_or_none()
        
        if session and session.expires_at < datetime.utcnow():
            # Session expired
            return None
        
        return session
    
    @staticmethod
    async def update_last_login(db: AsyncSession, user_id: UUID):
        """Update user's last login timestamp"""
        user = await AuthService.get_user_by_id(db, user_id)
        if user:
            user.last_login_at = datetime.utcnow()
            await db.commit()
            await db.refresh(user)
    
    @staticmethod
    def verify_token(token: str) -> dict:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            return payload
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

