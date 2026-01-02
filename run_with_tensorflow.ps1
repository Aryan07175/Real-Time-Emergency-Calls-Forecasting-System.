# PowerShell script to run the dashboard with TensorFlow support
# This uses the virtual environment where TensorFlow is installed

Write-Host "🚀 Starting Emergency Calls Forecasting Dashboard..." -ForegroundColor Green
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "C:\venv\ts_project\Scripts\Activate.ps1"

# Change to project directory
Set-Location "C:\Users\Lenovo\Desktop\TIME SERIES PROJECT"

# Run Streamlit dashboard
Write-Host "Starting Streamlit dashboard..." -ForegroundColor Yellow
Write-Host "Dashboard will open at: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""

streamlit run dashboard/app.py --server.port 8501

