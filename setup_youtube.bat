@echo off
REM YouTube Automation Quick Setup Script

echo =========================================
echo   CLIPPPY YOUTUBE AUTOMATION SETUP
echo =========================================
echo.

echo Step 1: Installing Dependencies...
pip install google-api-python-client google-auth-oauthlib anthropic
echo.

echo Step 2: Checking for YouTube credentials...
if exist "config\youtube_credentials.json" (
    echo   ✅ Credentials file found!
) else (
    echo   ❌ Credentials file NOT found!
    echo.
    echo   📝 You need to:
    echo   1. Go to: https://console.cloud.google.com/
    echo   2. Create a new project
    echo   3. Enable YouTube Data API v3
    echo   4. Create OAuth 2.0 credentials
    echo   5. Download as: config\youtube_credentials.json
    echo.
    pause
    exit /b 1
)
echo.

echo Step 3: Authenticating with YouTube...
python twitch_clip_bot.py youtube-auth
echo.

echo Step 4: Checking configuration...
python -c "import yaml; config = yaml.safe_load(open('config/config.yaml')); yt = config.get('youtube', {}); print('✅ YouTube enabled!' if yt.get('enabled') else '⚠️  YouTube not enabled in config')"
echo.

echo =========================================
echo   SETUP COMPLETE!
echo =========================================
echo.
echo Next steps:
echo   1. Check queue: python twitch_clip_bot.py youtube-queue-status
echo   2. Start system: python always_on_controller.py
echo   3. Let it run! Clips will auto-upload to YouTube
echo.
echo For full documentation, see:
echo   YOUTUBE_AUTOMATION_GUIDE.md
echo.
pause

