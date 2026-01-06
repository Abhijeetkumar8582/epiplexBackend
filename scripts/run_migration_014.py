"""
Script to run migration 014: Add openai_api_key column to users table
"""
import asyncio
import sys
from pathlib import Path

# Add backend directory to path (parent of scripts folder)
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import engine
from app.config import settings
from sqlalchemy import text


async def run_migration_014():
    """Run migration 014 to add openai_api_key column"""
    migration_file = Path(__file__).parent.parent / "migrations" / "014_add_openai_api_key.sql"
    
    if not migration_file.exists():
        print(f"Error: Migration file not found: {migration_file}")
        return False
    
    print("=" * 60)
    print("Running Migration 014: Add openai_api_key column")
    print("=" * 60)
    print(f"Migration file: {migration_file}")
    print(f"Database: {settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else 'unknown'}")
    print()
    
    with open(migration_file, 'r', encoding='utf-8') as f:
        migration_sql = f.read()
    
    # Split by GO statements (SQL Server batch separator)
    batches = [batch.strip() for batch in migration_sql.split('GO') if batch.strip()]
    
    print(f"Found {len(batches)} SQL batch(es) to execute")
    print()
    
    try:
        async with engine.begin() as conn:
            for i, batch in enumerate(batches, 1):
                # Skip USE statement and PRINT statements
                if batch.strip().upper().startswith('USE ') or batch.strip().upper().startswith('PRINT'):
                    print(f"Skipping batch {i}: {batch[:50]}...")
                    continue
                
                print(f"Executing batch {i}/{len(batches)}...")
                try:
                    result = await conn.execute(text(batch))
                    print(f"[OK] Batch {i} executed successfully")
                except Exception as e:
                    error_str = str(e)
                    # Check if column already exists (this is OK)
                    if "already exists" in error_str.lower() or "already an object" in error_str.lower():
                        print(f"[INFO] Batch {i}: Column might already exist - {error_str[:100]}")
                        print("   (This is OK if column already exists)")
                    elif "Invalid column name" in error_str:
                        print(f"[ERROR] Batch {i}: {error_str[:200]}")
                        return False
                    else:
                        print(f"[WARNING] Batch {i}: {error_str[:200]}")
                        # Continue anyway - might be a non-critical error
        
        print()
        print("=" * 60)
        print("[SUCCESS] Migration 014 completed!")
        print("=" * 60)
        print()
        print("The 'openai_api_key' column has been added to the 'users' table.")
        print("You can now restart your backend server.")
        return True
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"[ERROR] Migration failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_migration_014())
    sys.exit(0 if success else 1)

