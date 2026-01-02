@echo off
title Emergency Calls Forecasting Dashboard
color 0A

echo.
echo ========================================
echo   Emergency Calls Forecasting Dashboard
echo   (With TensorFlow Support)
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "C:\venv\ts_project\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at C:\venv\ts_project
    echo.
    echo Please create the virtual environment first:
    echo   python -m venv C:\venv\ts_project
    echo   C:\venv\ts_project\Scripts\activate.bat
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo [1/3] Activating virtual environment...
call C:\venv\ts_project\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo [2/3] Changing to project directory...
cd /d "%~dp0"
if errorlevel 1 (
    echo ERROR: Failed to change directory
    pause
    exit /b 1
)

echo [3/3] Starting Streamlit dashboard...
echo.
echo ========================================
echo   Dashboard will open in your browser
echo   URL: http://localhost:8501
echo ========================================
echo.
echo Press Ctrl+C to stop the server
echo.

REM Check if port is already in use and try alternative port
netstat -ano | findstr :8501 >nul
if errorlevel 0 (
    echo WARNING: Port 8501 is already in use!
    echo Trying to use port 8502 instead...
    echo.
    streamlit run dashboard/app.py --server.port 8502
) else (
    streamlit run dashboard/app.py --server.port 8501
)

if errorlevel 1 (
    echo.
    echo ERROR: Failed to start Streamlit
    echo.
    echo Troubleshooting steps:
    echo 1. Make sure you're in the project directory
    echo 2. Check if Python and Streamlit are installed correctly
    echo 3. Try running manually:
    echo    C:\venv\ts_project\Scripts\activate.bat
    echo    streamlit run dashboard/app.py
    echo.
    pause
    exit /b 1
)

pause

