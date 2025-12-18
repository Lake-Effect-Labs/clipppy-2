@echo off
REM Stop All Clipppy Services
REM ===========================

echo.
echo ========================================
echo   Stopping All Clipppy Services
echo ========================================
echo.

echo Stopping Redis...
taskkill /FI "WINDOWTITLE eq Clipppy - Redis*" /F >nul 2>&1
taskkill /IM redis-server.exe /F >nul 2>&1

echo Stopping Celery Worker...
taskkill /FI "WINDOWTITLE eq Clipppy - Celery Worker*" /F >nul 2>&1
taskkill /IM python.exe /FI "COMMANDLINE eq *celery*" /F >nul 2>&1

echo Stopping Flower Dashboard...
taskkill /FI "WINDOWTITLE eq Clipppy - Flower*" /F >nul 2>&1
taskkill /IM python.exe /FI "COMMANDLINE eq *flower*" /F >nul 2>&1

echo Stopping Controller...
taskkill /FI "WINDOWTITLE eq Clipppy - Controller*" /F >nul 2>&1
taskkill /IM python.exe /FI "COMMANDLINE eq *launch_always_on*" /F >nul 2>&1

echo Stopping Listeners...
taskkill /IM python.exe /FI "COMMANDLINE eq *twitch_clip_bot*" /F >nul 2>&1

echo.
echo All services stopped.
echo.
pause

