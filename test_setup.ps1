# Diagnostic script to test the setup
Write-Host "=== Testing Project Setup ===" -ForegroundColor Cyan
Write-Host ""

# Test 1: Check virtual environment
Write-Host "1. Checking virtual environment..." -ForegroundColor Yellow
if (Test-Path "C:\venv\ts_project\Scripts\Activate.ps1") {
    Write-Host "   ✅ Virtual environment exists" -ForegroundColor Green
} else {
    Write-Host "   ❌ Virtual environment NOT found!" -ForegroundColor Red
    exit 1
}

# Test 2: Activate and check Python
Write-Host "2. Activating virtual environment and checking Python..." -ForegroundColor Yellow
& "C:\venv\ts_project\Scripts\Activate.ps1"
$pythonVersion = python --version 2>&1
Write-Host "   Python: $pythonVersion" -ForegroundColor Green

# Test 3: Check TensorFlow
Write-Host "3. Checking TensorFlow..." -ForegroundColor Yellow
try {
    $tfVersion = python -c "import tensorflow as tf; print(tf.__version__)" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ TensorFlow installed: $tfVersion" -ForegroundColor Green
    } else {
        Write-Host "   ❌ TensorFlow import failed!" -ForegroundColor Red
    }
} catch {
    Write-Host "   ❌ Error checking TensorFlow: $_" -ForegroundColor Red
}

# Test 4: Check Streamlit
Write-Host "4. Checking Streamlit..." -ForegroundColor Yellow
try {
    $stVersion = python -c "import streamlit; print(streamlit.__version__)" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ Streamlit installed: $stVersion" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Streamlit import failed!" -ForegroundColor Red
    }
} catch {
    Write-Host "   ❌ Error checking Streamlit: $_" -ForegroundColor Red
}

# Test 5: Check project files
Write-Host "5. Checking project files..." -ForegroundColor Yellow
$projectPath = "C:\Users\Lenovo\Desktop\TIME SERIES PROJECT"
if (Test-Path "$projectPath\dashboard\app.py") {
    Write-Host "   ✅ Dashboard app.py found" -ForegroundColor Green
} else {
    Write-Host "   ❌ Dashboard app.py NOT found!" -ForegroundColor Red
}

if (Test-Path "$projectPath\911.csv") {
    Write-Host "   ✅ Data file 911.csv found" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Data file 911.csv NOT found (optional)" -ForegroundColor Yellow
}

# Test 6: Try importing dashboard
Write-Host "6. Testing dashboard imports..." -ForegroundColor Yellow
Set-Location $projectPath
try {
    $importTest = python -c "import sys; sys.path.insert(0, '.'); from data_preprocessing import process_dataset; print('OK')" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✅ Dashboard modules can be imported" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Import error: $importTest" -ForegroundColor Red
    }
} catch {
    Write-Host "   ❌ Error: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Setup Test Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the dashboard, use:" -ForegroundColor Yellow
Write-Host "  .\run_with_tensorflow.ps1" -ForegroundColor White
Write-Host "  OR" -ForegroundColor Yellow
Write-Host "  Double-click: run_with_tensorflow.bat" -ForegroundColor White

