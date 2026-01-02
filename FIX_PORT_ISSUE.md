# 🔧 Fix: Port Already in Use

## Problem
If you see an error that port 8501 is already in use, it means another Streamlit instance is running.

## Quick Fix

### Option 1: Kill the existing process (Recommended)
1. Open PowerShell or Command Prompt
2. Run:
   ```powershell
   netstat -ano | findstr :8501
   ```
3. Note the PID (Process ID) number
4. Kill the process:
   ```powershell
   taskkill /F /PID [PID_NUMBER]
   ```
5. Run the dashboard again

### Option 2: Use a different port
The updated batch files (`run_with_tensorflow.bat` and `START_DASHBOARD.bat`) now automatically use port 8502 if 8501 is busy.

### Option 3: Manual port selection
Run Streamlit with a custom port:
```powershell
C:\venv\ts_project\Scripts\Activate.ps1
cd "C:\Users\Lenovo\Desktop\TIME SERIES PROJECT"
streamlit run dashboard/app.py --server.port 8503
```

## Prevention
Always stop the Streamlit server properly by pressing `Ctrl+C` in the terminal where it's running, rather than just closing the browser.

