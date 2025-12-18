@echo off
REM Start Redis Server for Clipppy
REM ================================
REM Make sure Redis is installed:
REM   - Download from: https://github.com/microsoftarchive/redis/releases
REM   - Or install via Chocolatey: choco install redis-64

echo.
echo ========================================
echo   Starting Redis Server for Clipppy
echo ========================================
echo.

REM Check if Redis is installed (try OneDrive Desktop location first)
set REDIS_PATH=%USERPROFILE%\OneDrive\Desktop\Redis-x64-3.0.504\redis-server.exe

if not exist "%REDIS_PATH%" (
    REM Try standard location
    where redis-server >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Redis not found!
        echo.
        echo Please install Redis:
        echo   1. Download from: https://github.com/microsoftarchive/redis/releases
        echo   2. Extract to Desktop: Redis-x64-3.0.504
        echo   3. Or install via Chocolatey: choco install redis-64
        echo.
        pause
        exit /b 1
    )
    echo Starting Redis on localhost:6379...
    echo.
    echo Press Ctrl+C to stop Redis
    echo.
    redis-server
) else (
    echo Starting Redis from Desktop location...
    echo.
    echo Press Ctrl+C to stop Redis
    echo.
    "%REDIS_PATH%"
)

