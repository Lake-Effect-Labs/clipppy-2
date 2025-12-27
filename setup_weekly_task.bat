@echo off
REM ===================================================================
REM SETUP WEEKLY COMPILATION TASK
REM Creates Windows Task Scheduler task to run every Sunday at 11:59 PM
REM ===================================================================

echo.
echo ================================================================
echo WEEKLY COMPILATION - TASK SCHEDULER SETUP
echo ================================================================
echo.
echo This will create a Windows scheduled task that runs every Sunday
echo at 11:59 PM to automatically:
echo   1. Gather all clips from the past week
echo   2. Create a compilation video (landscape 16:9)
echo   3. Upload to YouTube automatically
echo.
echo The task will run even if you're not logged in.
echo.
pause

REM Delete existing task if it exists
schtasks /Delete /TN "Clipppy Weekly Compilation" /F >nul 2>&1

REM Create new task
echo Creating scheduled task...
schtasks /Create ^
  /TN "Clipppy Weekly Compilation" ^
  /TR "\"%CD%\run_weekly_compilation.bat\"" ^
  /SC WEEKLY ^
  /D SUN ^
  /ST 23:59 ^
  /RU "%USERNAME%" ^
  /RL HIGHEST ^
  /F

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================
    echo OK - Task created successfully!
    echo ================================================================
    echo.
    echo Task Details:
    echo   Name: Clipppy Weekly Compilation
    echo   Schedule: Every Sunday at 11:59 PM
    echo   Action: Creates and uploads weekly compilation
    echo.
    echo To view task: Task Scheduler ^> Clipppy Weekly Compilation
    echo To disable: schtasks /Change /TN "Clipppy Weekly Compilation" /DISABLE
    echo To delete: schtasks /Delete /TN "Clipppy Weekly Compilation" /F
    echo.
) else (
    echo.
    echo ================================================================
    echo ERROR - Failed to create task
    echo ================================================================
    echo.
    echo Make sure you ran this as Administrator.
    echo.
)

pause

