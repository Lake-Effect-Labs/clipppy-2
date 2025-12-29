@echo off
REM Start Celery Beat Scheduler for Clipppy
REM =========================================
REM Handles periodic tasks (YouTube queue processing, cleanup, etc.)

echo.
echo ========================================
echo   Starting Celery Beat Scheduler
echo ========================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo Starting Celery Beat scheduler...
echo This will trigger periodic tasks:
echo   - YouTube queue processing (every hour)
echo   - Temp file cleanup (daily)
echo.
echo Press Ctrl+C to stop scheduler
echo.

REM Start Celery Beat
celery -A celery_tasks beat --loglevel=info

pause

