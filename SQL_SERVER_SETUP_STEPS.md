# SQL Server Setup - Steps to Fix Database Connection

## Issue
The backend was using SQLite instead of SQL Server, so data was being saved locally instead of in your SQL Server database.

## ✅ What Has Been Fixed

1. **Updated `.env` file** - Changed `DATABASE_URL` to use SQL Server
2. **Installed required packages** - `aioodbc` and `pyodbc` are now installed
3. **Updated database types** - Modified `database.py` to use SQL Server-compatible types

## ⚠️ Required Next Steps

### Step 1: Create Tables in SQL Server

**You MUST run the migration script in SQL Server Management Studio (SSMS) first!**

1. Open **SQL Server Management Studio**
2. Connect to your server:
   - Server: `druidpartners.druidqa.druidplatform.com`
   - Authentication: SQL Server Authentication
   - Login: `admin_abhijeetkumar`
   - Password: `BpGWuAPyCm2_j7VKRVUlEHn2vi94nB6Z`
3. Open the migration script:
   - File: `backend/migrations/009_create_tables_sql_server.sql`
4. Execute the script (Press F5 or click Execute)
5. Verify tables were created:
   ```sql
   USE Druid_AbhijeetKumar;
   SELECT TABLE_NAME 
   FROM INFORMATION_SCHEMA.TABLES 
   WHERE TABLE_TYPE = 'BASE TABLE'
   ORDER BY TABLE_NAME;
   ```
   You should see: `users`, `user_sessions`, `user_activity_logs`, `video_uploads`, `frame_analyses`, `job_status`

### Step 2: Restart the Backend Server

After running the migration script, restart your backend server:

```powershell
# Stop the current backend (if running)
# Then start it again:
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Step 3: Test the Connection

After restarting, try creating a new user account through the signup page. The user should now appear in SQL Server.

## Verify Data in SQL Server

After creating a user, check SQL Server:

```sql
USE Druid_AbhijeetKumar;
SELECT * FROM users;
```

## Current Configuration

- **Database URL**: `mssql+aioodbc://admin_abhijeetkumar:BpGWuAPyCm2_j7VKRVUlEHn2vi94nB6Z@druidpartners.druidqa.druidplatform.com:1433/Druid_AbhijeetKumar?driver=ODBC+Driver+17+for+SQL+Server`
- **Database**: `Druid_AbhijeetKumar`
- **Driver**: ODBC Driver 17 for SQL Server

## Troubleshooting

### If you get "ODBC Driver 17 for SQL Server" error:
- Download and install: https://aka.ms/downloadmsodbcsql

### If tables already exist:
- The migration script uses `IF NOT EXISTS`, so it's safe to run multiple times

### If you want to switch back to SQLite:
- Update `.env` file: `DATABASE_URL=sqlite+aiosqlite:///./video_processing.db`

