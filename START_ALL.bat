@echo off
REM Start All Clipppy Services
REM ============================
REM Launches Redis, Celery Worker, Flower Dashboard, and Controller
REM Press Ctrl+C to stop all services

echo.
echo ========================================
echo   Starting Clipppy Services
echo ========================================
echo.

echo Starting services...
echo Press Ctrl+C to stop all services
echo.

REM Start Redis in new window
echo Starting Redis...
start "Clipppy - Redis" cmd /k start_redis.bat

REM Wait for Redis to start
timeout /t 3 /nobreak >nul

REM Start Celery Worker in new window
echo Starting Celery Worker...
start "Clipppy - Celery Worker" cmd /k start_celery_worker.bat

REM Wait for Celery to start
timeout /t 3 /nobreak >nul

REM Start Flower Dashboard in new window
echo Starting Flower Dashboard...
start "Clipppy - Flower" cmd /k start_flower.bat

REM Wait for Flower to start
timeout /t 3 /nobreak >nul

REM Start Always-On Controller in new window
echo Starting Always-On Controller...
start "Clipppy - Controller" cmd /k python launch_always_on.py

echo.
echo ========================================
echo   All Services Started!
echo ========================================
echo.
echo   Redis:      localhost:6379
echo   Flower:     http://localhost:5555
echo   Controller: Running in separate window
echo.
echo Press Ctrl+C to stop all services
echo.
echo Or run STOP_ALL.bat in another terminal
echo.

REM Use PowerShell for proper Ctrl+C handling
powershell -NoExit -Command "function Stop-AllServices { Write-Host ''; Write-Host 'Stopping all services...' -ForegroundColor Red; Get-Process | Where-Object { $_.MainWindowTitle -like '*Clipppy*' } | Stop-Process -Force -ErrorAction SilentlyContinue; Get-Process redis-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue; taskkill /FI 'WINDOWTITLE eq Clipppy*' /F 2>$null | Out-Null; Write-Host 'All services stopped.' -ForegroundColor Green }; Register-EngineEvent PowerShell.Exiting -Action { Stop-AllServices } -ErrorAction SilentlyContinue | Out-Null; try { while ($true) { Start-Sleep -Seconds 1 } } catch { Stop-AllServices; exit }"

REM If PowerShell exits, run cleanup
call :CLEANUP
exit /b 0

:CLEANUP
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
timeout /t 2 /nobreak >nul
exit /b 0
