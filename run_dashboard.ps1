# Emergency Calls Forecasting Dashboard Launcher
Set-Location $PSScriptRoot
Write-Host "Starting Emergency Calls Forecasting Dashboard..." -ForegroundColor Green
Write-Host ""
python -m streamlit run dashboard/app.py --server.port 8501

