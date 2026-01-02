# 📍 Model Code Locations - SARIMA & LSTM

## ✅ Code Status

All model code is present in the project! Here's where everything is located:

### 📁 File Structure

```
TIME SERIES PROJECT/
├── model_training.py          # ⭐ ALL MODEL CODE IS HERE
│   ├── train_arima()          # Line ~81
│   ├── train_sarima()         # Line 333 ✅
│   ├── train_prophet()        # Line ~200
│   ├── train_lstm()           # Line 438 ✅
│   ├── forecast_sarima()       # Line 370 ✅
│   ├── forecast_lstm()        # Line 516 ✅
│   ├── save_sarima_model()    # Line 577 ✅
│   ├── load_sarima_model()    # Line 590 ✅
│   ├── save_lstm_model()      # Line 602 ✅
│   └── load_lstm_model()      # Line 628 ✅
│
├── train_models.py            # Training script (includes SARIMA & LSTM)
│   └── main()                 # Trains all 4 models
│
├── dashboard/app.py           # Dashboard (supports all 4 models)
│   └── Uses all models including SARIMA & LSTM
│
└── models/                    # Saved model files
    ├── arima_model.pkl        ✅ Exists
    ├── prophet_model.pkl      ✅ Exists
    ├── sarima_model.pkl       ❌ Missing (needs training)
    └── lstm_model.pkl         ❌ Missing (needs training)
```

## 🔍 Code Details

### SARIMA Model (Seasonal ARIMA)

**Location:** `model_training.py`

- **Training Function:** `train_sarima()` - Line 333
  - Parameters: `order=(2, 1, 2)`, `seasonal_order=(1, 1, 1, 24)`
  - Uses `SARIMAX` from statsmodels
  - Handles daily seasonality (24-hour cycle)

- **Forecast Function:** `forecast_sarima()` - Line 370
  - Generates forecasts with confidence intervals
  - Returns DataFrame with forecast, lower_bound, upper_bound

- **Save/Load Functions:**
  - `save_sarima_model()` - Line 577
  - `load_sarima_model()` - Line 590

### LSTM Model (Long Short-Term Memory)

**Location:** `model_training.py`

- **Training Function:** `train_lstm()` - Line 438
  - Parameters: `lookback=24`, `epochs=50`, `batch_size=32`, `units=50`
  - Uses TensorFlow/Keras
  - Requires TensorFlow to be installed
  - Two-layer LSTM with dropout

- **Forecast Function:** `forecast_lstm()` - Line 516
  - Generates step-by-step forecasts
  - Includes confidence intervals (approximated)

- **Save/Load Functions:**
  - `save_lstm_model()` - Line 602
  - `load_lstm_model()` - Line 628
  - Saves Keras model separately as `.h5` file

- **Helper Function:** `prepare_lstm_data()` - Line 413
  - Prepares time series data for LSTM training
  - Uses MinMaxScaler for normalization

## 🚀 How to Train Missing Models

### Option 1: Train All Models (Recommended)
```powershell
# Activate virtual environment
C:\venv\ts_project\Scripts\Activate.ps1

# Navigate to project
cd "C:\Users\Lenovo\Desktop\TIME SERIES PROJECT"

# Run training script (trains ARIMA, SARIMA, Prophet, and LSTM)
python train_models.py
```

### Option 2: Train Only SARIMA
```python
from data_preprocessing import process_dataset
from model_training import train_sarima, save_sarima_model

# Load data
hourly_df, _, _ = process_dataset('911.csv')
train_df = hourly_df.iloc[:int(len(hourly_df) * 0.8)]

# Train SARIMA
sarima_model = train_sarima(train_df, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24))

# Save model
if sarima_model:
    save_sarima_model(sarima_model, 'models/sarima_model.pkl')
```

### Option 3: Train Only LSTM
```python
from data_preprocessing import process_dataset
from model_training import train_lstm, save_lstm_model

# Load data
hourly_df, _, _ = process_dataset('911.csv')
train_df = hourly_df.iloc[:int(len(hourly_df) * 0.8)]

# Train LSTM (requires TensorFlow)
lstm_model = train_lstm(train_df, lookback=24, epochs=30, batch_size=32, units=50)

# Save model
if lstm_model:
    save_lstm_model(lstm_model, 'models/lstm_model.pkl')
```

## 📊 Model Usage in Dashboard

The Streamlit dashboard (`dashboard/app.py`) already supports all 4 models:

1. **ARIMA** - ✅ Working (model exists)
2. **SARIMA** - ⚠️ Code ready, model needs training
3. **Prophet** - ✅ Working (model exists)
4. **LSTM** - ⚠️ Code ready, model needs training

Once you train and save the SARIMA and LSTM models, they will automatically appear in the dashboard!

## ⏱️ Training Time Estimates

- **SARIMA**: ~2-5 minutes (depends on data size)
- **LSTM**: ~10-30 minutes (depends on epochs and data size)

## ✅ Summary

- ✅ **SARIMA code**: Complete in `model_training.py` (Line 333-410)
- ✅ **LSTM code**: Complete in `model_training.py` (Line 438-574)
- ✅ **Training script**: Includes both models (`train_models.py`)
- ✅ **Dashboard support**: Ready for both models
- ❌ **Saved models**: SARIMA and LSTM need to be trained

**Next Step:** Run `python train_models.py` to train and save all models!

