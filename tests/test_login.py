#!/usr/bin/env python3
"""Test script to diagnose login issues"""
import asyncio
import sys
from pathlib import Path

# Add the backend directory to the path (parent of tests folder)
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import get_db, User
from app.services.auth_service import AuthService
from sqlalchemy import select

async def test_database():
    """Test database connection and check for users"""
    print("=" * 60)
    print("Testing Database Connection")
    print("=" * 60)
    
    try:
        async for db in get_db():
            # Test basic query
            result = await db.execute(select(User))
            users = result.scalars().all()
            
            print(f"\n[OK] Database connection successful!")
            print(f"[OK] Found {len(users)} user(s) in database\n")
            
            if users:
                print("Users in database:")
                for user in users:
                    print(f"  - Email: {user.email}")
                    print(f"    ID: {user.id}")
                    print(f"    Active: {user.is_active}")
                    print(f"    Has Password: {bool(user.password_hash)}")
                    print(f"    Provider: {user.provider}")
                    print()
            else:
                print("[WARNING] No users found in database!")
                print("   You need to sign up first before you can login.")
                print("   Use: POST /api/auth/signup")
            
            break
    except Exception as e:
        print(f"\n[ERROR] Database connection failed!")
        print(f"   Error: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

async def test_login(email: str, password: str):
    """Test login with provided credentials"""
    print("=" * 60)
    print(f"Testing Login: {email}")
    print("=" * 60)
    
    try:
        async for db in get_db():
            # Try to authenticate
            user = await AuthService.authenticate_user(
                db=db,
                email=email,
                password=password
            )
            
            if user:
                print(f"\n[OK] Login successful!")
                print(f"  User ID: {user.id}")
                print(f"  Email: {user.email}")
                print(f"  Name: {user.full_name}")
                print(f"  Role: {user.role}")
            else:
                print(f"\n[ERROR] Login failed!")
                print(f"  Possible reasons:")
                print(f"  1. User does not exist")
                print(f"  2. Incorrect password")
                print(f"  3. User account is inactive")
                print(f"  4. User was created with Google OAuth (no password)")
            
            break
    except Exception as e:
        print(f"\n[ERROR] Login test failed with exception!")
        print(f"   Error: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test database connection first
    db_ok = asyncio.run(test_database())
    
    # If database is OK and user provided credentials, test login
    if db_ok and len(sys.argv) >= 3:
        email = sys.argv[1]
        password = sys.argv[2]
        asyncio.run(test_login(email, password))
    elif len(sys.argv) >= 3:
        print("\n[WARNING] Skipping login test due to database connection issues")
    else:
        print("\n" + "=" * 60)
        print("Usage:")
        print("  python test_login.py                    # Test database only")
        print("  python test_login.py <email> <password> # Test database + login")
        print("=" * 60)

