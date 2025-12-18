# Start All Clipppy Services
# ============================
# Launches Redis, Celery Worker, Flower Dashboard, and Controller
# Press Ctrl+C to stop all services

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting Clipppy Services" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Starting services..." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop all services" -ForegroundColor Yellow
Write-Host ""

# Store process IDs for cleanup
$processes = @()

# Function to cleanup all processes
function Stop-AllServices {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  Stopping All Clipppy Services" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    
    Write-Host "Stopping Redis..." -ForegroundColor Yellow
    Get-Process | Where-Object { $_.MainWindowTitle -like "*Clipppy - Redis*" } | Stop-Process -Force -ErrorAction SilentlyContinue
    Get-Process redis-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    
    Write-Host "Stopping Celery Worker..." -ForegroundColor Yellow
    Get-Process | Where-Object { $_.MainWindowTitle -like "*Clipppy - Celery Worker*" } | Stop-Process -Force -ErrorAction SilentlyContinue
    Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*celery*" } | Stop-Process -Force -ErrorAction SilentlyContinue
    
    Write-Host "Stopping Flower Dashboard..." -ForegroundColor Yellow
    Get-Process | Where-Object { $_.MainWindowTitle -like "*Clipppy - Flower*" } | Stop-Process -Force -ErrorAction SilentlyContinue
    Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*flower*" } | Stop-Process -Force -ErrorAction SilentlyContinue
    
    Write-Host "Stopping Controller..." -ForegroundColor Yellow
    Get-Process | Where-Object { $_.MainWindowTitle -like "*Clipppy - Controller*" } | Stop-Process -Force -ErrorAction SilentlyContinue
    Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*launch_always_on*" } | Stop-Process -Force -ErrorAction SilentlyContinue
    
    Write-Host ""
    Write-Host "All services stopped." -ForegroundColor Green
    Write-Host ""
}

# Register cleanup on Ctrl+C
$null = Register-EngineEvent PowerShell.Exiting -Action { Stop-AllServices } -ErrorAction SilentlyContinue

# Trap Ctrl+C
trap {
    Stop-AllServices
    break
}

# Start Redis
Write-Host "Starting Redis..." -ForegroundColor Cyan
$redisProcess = Start-Process -FilePath "cmd.exe" -ArgumentList "/k", "start_redis.bat" -WindowStyle Normal -PassThru
$processes += $redisProcess
Start-Sleep -Seconds 3

# Start Celery Worker
Write-Host "Starting Celery Worker..." -ForegroundColor Cyan
$celeryProcess = Start-Process -FilePath "cmd.exe" -ArgumentList "/k", "start_celery_worker.bat" -WindowStyle Normal -PassThru
$processes += $celeryProcess
Start-Sleep -Seconds 3

# Start Flower Dashboard
Write-Host "Starting Flower Dashboard..." -ForegroundColor Cyan
$flowerProcess = Start-Process -FilePath "cmd.exe" -ArgumentList "/k", "start_flower.bat" -WindowStyle Normal -PassThru
$processes += $flowerProcess
Start-Sleep -Seconds 3

# Start Controller
Write-Host "Starting Always-On Controller..." -ForegroundColor Cyan
$controllerProcess = Start-Process -FilePath "python.exe" -ArgumentList "launch_always_on.py" -WindowStyle Normal -PassThru
$processes += $controllerProcess

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  All Services Started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Redis:      localhost:6379" -ForegroundColor White
Write-Host "  Flower:     http://localhost:5555" -ForegroundColor White
Write-Host "  Controller: Running in separate window" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop all services" -ForegroundColor Yellow
Write-Host ""

# Wait for Ctrl+C
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} catch {
    # Ctrl+C was pressed
    Stop-AllServices
}

