@echo off
REM Start Flower Monitoring Dashboard
REM ===================================
REM Web UI for monitoring Celery tasks

echo.
echo ========================================
echo   Starting Flower Dashboard
echo ========================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo Starting Flower dashboard...
echo.
echo Dashboard will be available at: http://localhost:5555
echo.
echo Press Ctrl+C to stop dashboard
echo.

celery -A celery_tasks flower --port=5555

pause

