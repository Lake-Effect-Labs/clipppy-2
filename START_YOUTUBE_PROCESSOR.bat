@echo off
REM Start YouTube Queue Processor Loop
REM This will check and upload YouTube videos every hour automatically

echo.
echo ========================================
echo   YouTube Auto-Uploader
echo ========================================
echo.
echo This will check for YouTube uploads every hour
echo Press Ctrl+C to stop
echo.

:loop
echo [%TIME%] Checking YouTube queue...
python process_youtube_queue.py
echo.
echo Waiting 1 hour until next check...
timeout /t 3600 /nobreak >nul
goto loop

