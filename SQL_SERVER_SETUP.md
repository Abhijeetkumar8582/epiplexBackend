# SQL Server Setup Guide

## Database Configuration

The application has been configured to use SQL Server with the following credentials:

- **Instance**: druidpartners.druidqa.druidplatform.com
- **Database**: Druid_AbhijeetKumar
- **User**: admin_abhijeetkumar
- **Port**: 1433 (default SQL Server port)

## Connection String

The connection string format for SQL Server is:
```
mssql+aioodbc://admin_abhijeetkumar:BpGWuAPyCm2_j7VKRVUlEHn2vi94nB6Z@druidpartners.druidqa.druidplatform.com:1433/Druid_AbhijeetKumar?driver=ODBC+Driver+17+for+SQL+Server
```

## Prerequisites

1. **Install ODBC Driver 17 for SQL Server**
   - Windows: Download from [Microsoft](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)
   - Linux: `sudo apt-get install odbcinst1debian2 unixodbc unixodbc-dev` then install the driver

2. **Install Python Dependencies**
   ```bash
   pip install aioodbc pyodbc
   ```

## Database Type Compatibility Notes

⚠️ **Important**: The current database models use PostgreSQL-specific types:
- `UUID` - PostgreSQL UUID type
- `INET` - PostgreSQL INET type for IP addresses
- `TIMESTAMP(timezone=True)` - PostgreSQL timezone-aware timestamps
- `JSONB` - PostgreSQL JSONB type

For full SQL Server compatibility, these may need to be adjusted:
- `UUID` → `String(36)` or `GUID` type
- `INET` → `String(45)` (IPv6 max length)
- `TIMESTAMP(timezone=True)` → `DateTime` or `DATETIMEOFFSET`
- `JSONB` → `JSON` or `NVARCHAR(MAX)`

## Testing the Connection

1. Update your `.env` file with the SQL Server connection string
2. Test the connection:
   ```python
   from app.database import engine
   import asyncio
   
   async def test_connection():
       async with engine.connect() as conn:
           result = await conn.execute("SELECT 1")
           print("Connection successful!")
   
   asyncio.run(test_connection())
   ```

## Migration Notes

When migrating from PostgreSQL to SQL Server:

1. **Run migrations**: The SQL migration files in `migrations/` are PostgreSQL-specific. You may need to:
   - Convert SQL syntax to SQL Server (T-SQL)
   - Adjust data types
   - Update function calls (e.g., `gen_random_uuid()` → `NEWID()`)

2. **Update migration files**: Consider creating SQL Server-specific migrations or use Alembic for database-agnostic migrations.

## Troubleshooting

### Connection Issues
- Verify the SQL Server instance is accessible from your network
- Check firewall rules allow port 1433
- Ensure the ODBC driver is installed correctly

### Authentication Issues
- Verify username and password are correct
- Check if SQL Server authentication is enabled (not just Windows auth)

### Driver Issues
- Ensure ODBC Driver 17 for SQL Server is installed
- On Linux, verify the driver path in connection string if needed

