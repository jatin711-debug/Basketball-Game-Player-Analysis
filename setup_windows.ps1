# Basketball Analysis System - Windows Setup Script
# Run this in PowerShell to set up the system

Write-Host "🏀 Basketball Analysis System - Windows Setup" -ForegroundColor Green
Write-Host "="*50 -ForegroundColor Yellow

# Check if Python is installed
Write-Host "🔍 Checking Python installation..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python not found"
    }
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8+ from python.org" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if pip is available
Write-Host "🔍 Checking pip..." -ForegroundColor Cyan
try {
    python -m pip --version | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ pip is available" -ForegroundColor Green
    } else {
        throw "pip not available"
    }
} catch {
    Write-Host "❌ pip not available" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Update pip
Write-Host "📦 Updating pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Run the Python setup script
Write-Host "🚀 Running Python setup script..." -ForegroundColor Cyan
python setup.py

# Check if setup was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Setup completed successfully!" -ForegroundColor Green
    
    Write-Host "`n🎯 Quick Start Options:" -ForegroundColor Yellow
    Write-Host "1. Interactive setup: python quick_start.py" -ForegroundColor White
    Write-Host "2. Test system: python test_system.py" -ForegroundColor White
    Write-Host "3. Manual analysis: python script.py --video your_video.mp4 --visualize" -ForegroundColor White
    
    Write-Host "`n📝 Important Notes for RTX 3050 Users:" -ForegroundColor Yellow
    Write-Host "• Close other GPU-intensive applications during analysis" -ForegroundColor White
    Write-Host "• Monitor GPU temperature (use MSI Afterburner or similar)" -ForegroundColor White
    Write-Host "• System is optimized for your GPU with 80% memory usage" -ForegroundColor White
    Write-Host "• For best performance, use videos under 1080p resolution" -ForegroundColor White
    
} else {
    Write-Host "❌ Setup failed. Please check the error messages above." -ForegroundColor Red
}

Write-Host "`nPress Enter to continue..."
Read-Host
