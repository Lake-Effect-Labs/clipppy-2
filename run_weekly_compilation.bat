@echo off
REM ===================================================================
REM WEEKLY COMPILATION - AUTOMATED TASK
REM Runs every Sunday to create and upload weekly compilation
REM Then cleans up old files to save disk space
REM ===================================================================

cd "C:\Users\samfi\OneDrive\Desktop\cursor projects\clipppy 2"

echo [%date% %time%] Starting weekly tasks... >> logs\weekly_compilation_task.log

REM Create compilation from all clips in the past 7 days (no max limit)
echo [%date% %time%] Creating compilation... >> logs\weekly_compilation_task.log
python twitch_clip_bot.py weekly-compilation --streamer theburntpeanut --days 7 --yes

REM Clean up old files to save disk space
echo [%date% %time%] Running cleanup... >> logs\weekly_compilation_task.log
python weekly_cleanup.py

REM Log completion
echo [%date% %time%] Weekly tasks completed >> logs\weekly_compilation_task.log
echo. >> logs\weekly_compilation_task.log

