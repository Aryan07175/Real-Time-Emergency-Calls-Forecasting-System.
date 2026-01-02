# 🚨 Real-Time Emergency Calls Forecasting System

A comprehensive time series analysis system for predicting ambulance demand using ARIMA and Prophet models.

## 📋 Project Overview

This system forecasts emergency 911 calls 1-24 hours ahead to help optimize ambulance resource allocation. It includes:

- **Data Preprocessing**: Loads and processes emergency call data
- **Model Training**: ARIMA and Prophet forecasting models
- **Real-Time Simulation**: Simulates streaming data and rolling predictions
- **Interactive Dashboard**: Streamlit-based visualization and analysis

## 🗂️ Project Structure

```
project/
├── data/                    # Data directory (place your CSV here)
├── data_preprocessing.py    # Data loading and preprocessing
├── model_training.py        # ARIMA and Prophet model training
├── real_time_simulation.py  # Real-time simulation pipeline
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── utils/
│   └── helpers.py          # Utility functions
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 📊 Dataset Format

The system expects a CSV file with the following columns:

- `timeStamp`: Timestamp of the emergency call (datetime format)
- `title`: Call type (e.g., "EMS: BACK PAINS/INJURY", "Fire: GAS-ODOR/LEAK", "Traffic: VEHICLE ACCIDENT")
- `lat`, `lng`: Geographic coordinates
- `twp`: Township/location
- `addr`: Address

**Note**: The system will automatically extract call types and create priority levels.

## 🚀 Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your dataset**:
   - Place your CSV file in the project root directory
   - Name it `911.csv` (or update the file path in the code)

## 📖 Usage

### 1. Data Preprocessing

```python
from data_preprocessing import process_dataset

# Load and preprocess data
hourly_df, processed_df, location_df = process_dataset('911.csv')
```

### 2. Model Training

```python
from model_training import train_arima, train_prophet, forecast_arima, forecast_prophet

# Train ARIMA model
arima_model = train_arima(hourly_df, auto_tune=False, order=(2, 1, 2))

# Generate forecast
arima_forecast = forecast_arima(arima_model, steps=24)

# Train Prophet model
prophet_model = train_prophet(hourly_df)

# Generate forecast
prophet_forecast = forecast_prophet(prophet_model, periods=24)
```

### 3. Real-Time Simulation

```python
from real_time_simulation import simulate_real_time

# Run simulation
simulator, predictions = simulate_real_time(
    historical_df=hourly_df,
    num_hours=48,
    retrain_interval=24
)
```

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 🎯 Dashboard Features

The Streamlit dashboard includes:

1. **📊 Hourly Trends**: Interactive time series visualization
2. **🧭 Seasonal Patterns**: Hour-of-day and day-of-week patterns
3. **🔮 Forecasts**: 24-hour ahead predictions with confidence intervals
4. **📍 Location Map**: Geographic visualization of emergency calls
5. **🔁 Real-Time Simulation**: Simulate streaming data and rolling predictions

## 🔧 Configuration

### Model Parameters

- **ARIMA**: Default order (2, 1, 2). Set `auto_tune=True` for automatic parameter selection (slower).
- **Prophet**: Configured with yearly, weekly, and daily seasonality.

### Real-Time Simulation

- **Retrain Interval**: How often to retrain models (default: 24 hours)
- **Simulation Hours**: Number of hours to simulate (default: 48)

## 📈 Model Performance

The system provides evaluation metrics:

- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory Issues**: For large datasets, consider:
   - Using a subset of data for training
   - Reducing the number of simulation hours
   - Sampling location data for map visualization

3. **Prophet Installation**: If Prophet fails to install:
   ```bash
   pip install prophet
   ```
   On Windows, you may need to install Visual C++ Build Tools.

## 📝 Notes

- The system automatically handles missing values and resamples data to hourly intervals
- Models are trained on 80% of the data by default
- Real-time simulation generates synthetic data based on historical patterns
- The dashboard supports interactive exploration of forecasts and patterns

## 🤝 Contributing

Feel free to extend this system with:
- Additional forecasting models (LSTM, XGBoost, etc.)
- More sophisticated feature engineering
- Real-time data integration APIs
- Advanced visualization options

## 📄 License

This project is open source and available for educational and research purposes.

## 👤 Author

Built for Time Series Analysis Project

---

**Happy Forecasting! 🚨📊**

