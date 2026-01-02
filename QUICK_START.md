# 🚀 Quick Start Guide - Emergency Calls Forecasting System

## Prerequisites
- Python 3.8 or higher
- VS Code (recommended) or any Python IDE
- Git (optional)

## Step 1: Install Dependencies

Open a terminal in VS Code (`` Ctrl+` `` or `Terminal > New Terminal`) and run:

```bash
pip install -r requirements.txt
```

**Note:** If Prophet installation fails due to Windows long path issues, it's already installed and you can proceed.

## Step 2: Verify Installation

Test that all modules can be imported:

```bash
python -c "import pandas, numpy, streamlit, plotly, statsmodels; print('All packages installed successfully!')"
```

## Step 3: Load and Preprocess Data

The dataset (`911.csv`) should already be in the project root. To verify:

```bash
python data_preprocessing.py
```

This will load and preprocess the data, showing statistics.

## Step 4: Train Models (Optional - Models Already Trained)

If you want to retrain the models:

```bash
python train_models.py
```

Or use the Jupyter notebook for interactive training:

```bash
jupyter notebook train_models.ipynb
```

**Note:** Pre-trained models are already available in the `models/` folder.

## Step 5: Run the Dashboard

### Option 1: Using VS Code
1. Open `dashboard/app.py` in VS Code
2. Right-click and select "Run Python File in Terminal"
3. Or use the Run button (▶️) in VS Code

### Option 2: Using Terminal
```bash
streamlit run dashboard/app.py
```

### Option 3: Using Batch File (Windows)
Double-click `run_dashboard.bat`

### Option 4: Using PowerShell Script
```powershell
.\run_dashboard.ps1
```

The dashboard will open automatically at: **http://localhost:8501**

## Step 6: Using the Dashboard

1. **Load Data**: Click "Load Default Dataset (911.csv)" in the sidebar
2. **Load Models**: Click "Load Saved Models" to load ARIMA and Prophet models
3. **Explore**:
   - 📈 **Hourly Trends**: View call patterns over time
   - 🧭 **Seasonal Patterns**: Analyze hourly and daily patterns
   - 🔮 **Forecasts**: Get 24-hour ahead predictions
   - 📍 **Location Map**: Visualize call locations
   - 🔁 **Real-Time Simulation**: Run streaming simulations

## Project Structure

```
TIME SERIES PROJECT/
├── 911.csv                      # Dataset
├── data_preprocessing.py        # Data loading and preprocessing
├── model_training.py            # ARIMA and Prophet training
├── real_time_simulation.py      # Real-time simulation
├── train_models.py              # Training script
├── train_models.ipynb           # Jupyter notebook for training
├── requirements.txt             # Python dependencies
├── run_dashboard.bat            # Windows batch launcher
├── run_dashboard.ps1            # PowerShell launcher
├── QUICK_START.md               # This file
├── README.md                    # Full documentation
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── models/
│   ├── arima_model.pkl          # Trained ARIMA model
│   └── prophet_model.pkl        # Trained Prophet model
└── utils/
    └── helpers.py               # Utility functions
```

## VS Code Setup

### Recommended Extensions
1. **Python** (Microsoft) - Python language support
2. **Jupyter** (Microsoft) - Jupyter notebook support
3. **Python Docstring Generator** - Auto-generate docstrings
4. **Pylance** - Fast Python language server

### VS Code Settings
Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
```

## Troubleshooting

### Issue: Streamlit not found
```bash
pip install streamlit
```

### Issue: Module not found errors
Make sure you're in the project root directory:
```bash
cd "C:\Users\Lenovo\Desktop\TIME SERIES PROJECT"
```

### Issue: Port 8501 already in use
Use a different port:
```bash
streamlit run dashboard/app.py --server.port 8502
```

### Issue: Prophet installation fails
Prophet is already installed. If you need to reinstall:
```bash
pip install prophet --no-cache-dir
```

### Issue: Models not loading
Make sure the `models/` folder exists and contains:
- `arima_model.pkl`
- `prophet_model.pkl`

If missing, run:
```bash
python train_models.py
```

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Test data preprocessing
python data_preprocessing.py

# Train models
python train_models.py

# Run dashboard
streamlit run dashboard/app.py

# Run Jupyter notebook
jupyter notebook train_models.ipynb
```

## Next Steps

1. ✅ Install dependencies
2. ✅ Load data in dashboard
3. ✅ Load models in dashboard
4. ✅ Explore visualizations
5. ✅ Generate forecasts
6. ✅ Run real-time simulations

## Support

For issues or questions:
- Check the `README.md` for detailed documentation
- Review error messages in the terminal
- Ensure all dependencies are installed

---

**Happy Forecasting! 🚨📊**

