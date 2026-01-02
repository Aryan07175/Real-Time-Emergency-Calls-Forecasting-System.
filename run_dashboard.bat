@echo off
cd /d "%~dp0"
echo Starting Emergency Calls Forecasting Dashboard...
echo.
python -m streamlit run dashboard/app.py --server.port 8501
pause

