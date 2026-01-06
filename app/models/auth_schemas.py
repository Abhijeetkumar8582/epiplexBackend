"""Authentication and user-related schemas"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from uuid import UUID


class UserSignup(BaseModel):
    """Schema for user registration"""
    full_name: str = Field(..., min_length=1, max_length=150, description="User's full name")
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., min_length=4, max_length=20, description="User's password (4-20 characters, can contain special characters)")


class UserLogin(BaseModel):
    """Schema for user login"""
    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")


class UserResponse(BaseModel):
    """Schema for user information response"""
    id: UUID
    full_name: str
    email: str
    role: str
    is_active: bool
    last_login_at: Optional[datetime] = None
    frame_analysis_prompt: Optional[str] = None
    openai_api_key: Optional[str] = None  # Note: Should be masked in responses
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LoginResponse(BaseModel):
    """Schema for login response with tokens"""
    access_token: str = Field(..., description="JWT access token")
    session_token: str = Field(..., description="Session token for tracking")
    token_type: str = Field(default="bearer", description="Token type")
    user: UserResponse = Field(..., description="User information")
    expires_at: datetime = Field(..., description="Token expiration time")


class SignupResponse(BaseModel):
    """Schema for signup response"""
    message: str = Field(..., description="Success message")
    user: UserResponse = Field(..., description="Created user information")

