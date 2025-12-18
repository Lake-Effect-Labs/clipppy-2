@echo off
REM Start Celery Worker for Clipppy
REM =================================
REM Handles video enhancement tasks from Redis queue

echo.
echo ========================================
echo   Starting Celery Worker for Clipppy
echo ========================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo Starting Celery worker...
echo Worker will process enhancement tasks from Redis queue
echo.
echo Press Ctrl+C to stop worker
echo.

REM Start Celery worker with:
REM - 1 concurrent task (video enhancement is CPU/GPU intensive)
REM - Info log level
REM - Auto-reload on code changes (for development)
celery -A celery_tasks worker --loglevel=info --concurrency=1 --pool=solo

pause

