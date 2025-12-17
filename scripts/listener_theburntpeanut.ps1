# Set execution policy and error handling
Set-ExecutionPolicy Bypass -Scope Process -Force
$ProgressPreference = 'SilentlyContinue'

# Configure window
$Host.UI.RawUI.WindowTitle = 'Clipppy - THEBURNTPEANUT'

# Prevent sleep/hibernate for this session
Add-Type -TypeDefinition @"
    using System;
    using System.Runtime.InteropServices;
    public class DisplayState {
        [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        public static extern uint SetThreadExecutionState(uint esFlags);
        public const uint ES_CONTINUOUS = 0x80000000;
        public const uint ES_SYSTEM_REQUIRED = 0x00000001;
        public const uint ES_DISPLAY_REQUIRED = 0x00000002;
    }
"@
[DisplayState]::SetThreadExecutionState([DisplayState]::ES_CONTINUOUS -bor [DisplayState]::ES_SYSTEM_REQUIRED -bor [DisplayState]::ES_DISPLAY_REQUIRED)

# Set process priority
$process = Get-Process -Id $pid
$process.PriorityClass = 'High'

# Change to working directory
cd 'C:\Users\samfi\OneDrive\Desktop\cursor projects\clipppy 2'

# Activate Python environment if it exists
if (Test-Path "venv/Scripts/activate.ps1") {
    ./venv/Scripts/activate.ps1
} elseif (Test-Path ".venv/Scripts/activate.ps1") {
    ./.venv/Scripts/activate.ps1
}

# Write startup message
Write-Host 'Starting Clipppy listener for theburntpeanut...' -ForegroundColor Green
Write-Host "Working directory: $(Get-Location)" -ForegroundColor Cyan
Write-Host ""

# Set UTF-8 encoding for Python output
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Run bot with output visible in console AND logged to file
$ErrorActionPreference = 'Continue'
$logFile = "logs/listener_theburntpeanut.log"

Write-Host "Running Python bot..." -ForegroundColor Cyan
Write-Host ""

# Run Python and capture all output
try {
    python twitch_clip_bot.py start --streamer theburntpeanut --always-on-mode 2>&1 | ForEach-Object {
        # Write to console immediately
        Write-Host $_
        # Write to log file
        Add-Content -Path $logFile -Value $_ -Encoding UTF8
    }
    $exitCode = $LASTEXITCODE
} catch {
    Write-Host "PowerShell error: $_" -ForegroundColor Red
    $exitCode = 1
}

Write-Host ""
Write-Host "Python process exited with code: $exitCode" -ForegroundColor $(if ($exitCode -eq 0) { "Green" } else { "Red" })

# Show error message if Python failed
if ($exitCode -ne 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "ERROR: Python script failed!" -ForegroundColor Red
    Write-Host "Exit code: $exitCode" -ForegroundColor Red
    Write-Host ""
    Write-Host "Last 30 lines from log:" -ForegroundColor Yellow
    if (Test-Path $logFile) {
        Get-Content $logFile -Tail 30 | Write-Host
    } else {
        Write-Host "Log file not found: $logFile" -ForegroundColor Yellow
    }
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host 'Window will remain open for 60 seconds...' -ForegroundColor Yellow
    Start-Sleep -Seconds 60
} else {
    Write-Host "Bot exited normally (stream may be offline)" -ForegroundColor Green
    Write-Host "Window will close in 5 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}
