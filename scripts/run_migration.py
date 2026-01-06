"""
Script to run SQL Server migration
Run this script to create the video_uploads table and other required tables
"""
import asyncio
import sys
from pathlib import Path

# Add backend directory to path (parent of scripts folder)
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import engine, Base
from app.config import settings
from sqlalchemy import text


async def run_migration():
    """Run SQL Server migration to create tables"""
    migration_file = Path(__file__).parent.parent / "migrations" / "009_create_tables_sql_server.sql"
    
    if not migration_file.exists():
        print(f"Error: Migration file not found: {migration_file}")
        return
    
    print(f"Reading migration file: {migration_file}")
    with open(migration_file, 'r', encoding='utf-8') as f:
        migration_sql = f.read()
    
    # Split by GO statements (SQL Server batch separator)
    batches = [batch.strip() for batch in migration_sql.split('GO') if batch.strip()]
    
    print(f"Found {len(batches)} SQL batches to execute")
    print(f"Connecting to database: {settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else 'unknown'}")
    
    try:
        async with engine.begin() as conn:
            for i, batch in enumerate(batches, 1):
                # Skip USE statement and PRINT statements as they may not work in async context
                if batch.strip().upper().startswith('USE ') or batch.strip().upper().startswith('PRINT'):
                    print(f"Skipping batch {i}: {batch[:50]}...")
                    continue
                
                print(f"Executing batch {i}/{len(batches)}...")
                try:
                    await conn.execute(text(batch))
                    print(f"✓ Batch {i} executed successfully")
                except Exception as e:
                    # Some errors are expected (like table already exists)
                    if "already exists" in str(e).lower() or "already an object" in str(e).lower():
                        print(f"⚠ Batch {i}: Table/object already exists (this is OK)")
                    else:
                        print(f"✗ Batch {i} error: {e}")
                        # Continue with other batches
        print("\n✓ Migration completed!")
    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        raise


async def create_tables_with_sqlalchemy():
    """Alternative: Create tables using SQLAlchemy"""
    print("Creating tables using SQLAlchemy...")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all, checkfirst=True)
        print("✓ Tables created successfully using SQLAlchemy")
    except Exception as e:
        print(f"✗ Failed to create tables: {e}")
        raise


async def main():
    """Main function"""
    print("=" * 60)
    print("SQL Server Migration Script")
    print("=" * 60)
    
    # Check if using SQL Server
    if "mssql" not in settings.DATABASE_URL.lower():
        print("⚠ Warning: This script is for SQL Server. Your database URL doesn't contain 'mssql'")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    print("\nChoose migration method:")
    print("1. Run SQL migration script (009_create_tables_sql_server.sql)")
    print("2. Create tables using SQLAlchemy (may have limitations)")
    print("3. Both")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1" or choice == "3":
        await run_migration()
    
    if choice == "2" or choice == "3":
        await create_tables_with_sqlalchemy()
    
    print("\n" + "=" * 60)
    print("Migration process completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

