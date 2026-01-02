# 🛠️ VS Code Setup Guide

## Quick Setup Steps

### 1. Install Recommended Extensions

VS Code will prompt you to install recommended extensions when you open the project. Or manually install:

- **Python** (ms-python.python)
- **Pylance** (ms-python.vscode-pylance)
- **Jupyter** (ms-toolsai.jupyter)

### 2. Select Python Interpreter

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type "Python: Select Interpreter"
3. Choose your Python 3.8+ interpreter

### 3. Install Dependencies

**Option A: Using VS Code Terminal**
1. Open terminal: `` Ctrl+` ``
2. Run: `pip install -r requirements.txt`

**Option B: Using VS Code Tasks**
1. Press `Ctrl+Shift+P`
2. Type "Tasks: Run Task"
3. Select "Install Requirements"

### 4. Run the Dashboard

**Option A: Using Launch Configuration**
1. Go to Run and Debug (`F5` or `Ctrl+Shift+D`)
2. Select "Python: Streamlit Dashboard"
3. Click the green play button

**Option B: Using Terminal**
1. Open terminal: `` Ctrl+` ``
2. Run: `streamlit run dashboard/app.py`

**Option C: Using Tasks**
1. Press `Ctrl+Shift+P`
2. Type "Tasks: Run Task"
3. Select "Run Streamlit Dashboard"

## VS Code Features Available

### Launch Configurations (F5)
- **Python: Streamlit Dashboard** - Run the dashboard
- **Python: Train Models** - Train ARIMA and Prophet models
- **Python: Data Preprocessing** - Test data loading
- **Python: Current File** - Run any Python file

### Tasks (Ctrl+Shift+P → Tasks: Run Task)
- **Install Requirements** - Install all dependencies
- **Run Streamlit Dashboard** - Start the dashboard
- **Train Models** - Train forecasting models
- **Test Data Preprocessing** - Test data loading

### Keyboard Shortcuts
- `` Ctrl+` `` - Toggle terminal
- `F5` - Start debugging/run
- `Ctrl+Shift+P` - Command palette
- `Ctrl+B` - Toggle sidebar

## Working with Jupyter Notebooks

1. Open `train_models.ipynb`
2. VS Code will automatically detect it as a Jupyter notebook
3. Click "Run All" or run cells individually
4. View outputs inline

## Debugging

1. Set breakpoints by clicking left of line numbers
2. Press `F5` to start debugging
3. Use debug controls to step through code
4. View variables in the Debug panel

## Project Structure in VS Code

```
📁 TIME SERIES PROJECT
├── 📄 911.csv
├── 📄 data_preprocessing.py
├── 📄 model_training.py
├── 📄 real_time_simulation.py
├── 📄 train_models.py
├── 📓 train_models.ipynb
├── 📄 requirements.txt
├── 📄 QUICK_START.md
├── 📄 VS_CODE_SETUP.md
├── 📁 .vscode/
│   ├── settings.json      # VS Code settings
│   ├── launch.json        # Debug configurations
│   ├── tasks.json         # Task definitions
│   └── extensions.json    # Recommended extensions
├── 📁 dashboard/
│   └── app.py
├── 📁 models/
│   ├── arima_model.pkl
│   └── prophet_model.pkl
└── 📁 utils/
    └── helpers.py
```

## Tips for VS Code

1. **IntelliSense**: Get autocomplete and suggestions as you type
2. **Error Detection**: Red squiggles show errors before running
3. **Git Integration**: Built-in Git support for version control
4. **Terminal Integration**: Multiple terminals in one window
5. **Code Formatting**: Right-click → Format Document (if formatter installed)

## Troubleshooting

### Python Interpreter Not Found
1. Install Python from python.org
2. Restart VS Code
3. Select interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter"

### Extensions Not Working
1. Reload VS Code: `Ctrl+Shift+P` → "Developer: Reload Window"
2. Check extension is enabled in Extensions panel

### Import Errors
1. Make sure you're in the project root
2. Check Python interpreter is correct
3. Verify dependencies are installed: `pip list`

---

**Happy Coding! 🚀**

