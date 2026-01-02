"""
Model Training Module
Trains ARIMA, SARIMA, Prophet, and LSTM models for time series forecasting
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from prophet import Prophet
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Try importing TensorFlow/Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("[WARNING] TensorFlow not available. LSTM model will not be available.")


def check_stationarity(ts):
    """
    Check if time series is stationary using Augmented Dickey-Fuller test
    
    Args:
        ts (pd.Series): Time series data
        
    Returns:
        bool: True if stationary
    """
    result = adfuller(ts.dropna())
    return result[1] <= 0.05  # p-value <= 0.05 means stationary


def auto_arima_params(ts, max_p=5, max_d=2, max_q=5):
    """
    Automatically find best ARIMA parameters using AIC
    
    Args:
        ts (pd.Series): Time series data
        max_p (int): Maximum AR order
        max_d (int): Maximum differencing order
        max_q (int): Maximum MA order
        
    Returns:
        tuple: Best (p, d, q) parameters
    """
    best_aic = np.inf
    best_params = (1, 1, 1)
    
    print("Searching for best ARIMA parameters...")
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(ts, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                        
                except Exception as e:
                    continue
    
    print(f"[OK] Best ARIMA parameters: {best_params} (AIC: {best_aic:.2f})")
    return best_params


def train_arima(df, auto_tune=True, order=None):
    """
    Train ARIMA model on hourly call counts
    
    Args:
        df (pd.DataFrame): Time series dataframe with 'timeStamp' and 'call_count'
        auto_tune (bool): Whether to auto-tune parameters
        order (tuple): Manual (p, d, q) order if auto_tune is False
        
    Returns:
        fitted ARIMA model
    """
    if df is None or df.empty:
        print("[ERROR] Empty dataset")
        return None
    
    # Prepare time series
    ts = df.set_index('timeStamp')['call_count']
    
    # Auto-tune parameters if requested
    if auto_tune:
        order = auto_arima_params(ts)
    elif order is None:
        order = (1, 1, 1)  # Default
    
    print(f"Training ARIMA{order} model...")
    
    try:
        # Fit ARIMA model
        model = ARIMA(ts, order=order)
        fitted_model = model.fit()
        
        print(f"[OK] ARIMA model trained successfully")
        print(f"  AIC: {fitted_model.aic:.2f}")
        print(f"  BIC: {fitted_model.bic:.2f}")
        
        return fitted_model
        
    except Exception as e:
        print(f"[ERROR] Error training ARIMA model: {str(e)}")
        return None


def forecast_arima(model, steps=24):
    """
    Generate forecast using ARIMA model
    
    Args:
        model: Fitted ARIMA model
        steps (int): Number of steps ahead to forecast
        
    Returns:
        pd.DataFrame: Forecast with confidence intervals
    """
    if model is None:
        return None
    
    try:
        # Generate forecast
        forecast = model.forecast(steps=steps)
        conf_int = model.get_forecast(steps=steps).conf_int()
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'forecast': forecast.values,
            'lower_bound': conf_int.iloc[:, 0].values,
            'upper_bound': conf_int.iloc[:, 1].values
        })
        
        # Generate future timestamps
        last_timestamp = model.model.data.dates[-1]
        future_dates = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=steps,
            freq='H'
        )
        forecast_df.index = future_dates
        forecast_df.index.name = 'timeStamp'
        
        return forecast_df.reset_index()
        
    except Exception as e:
        print(f"[ERROR] Error generating ARIMA forecast: {str(e)}")
        return None


def train_prophet(df):
    """
    Train Prophet model on hourly call counts
    
    Args:
        df (pd.DataFrame): Time series dataframe with 'timeStamp' and 'call_count'
        
    Returns:
        fitted Prophet model
    """
    if df is None or df.empty:
        print("[ERROR] Empty dataset")
        return None
    
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    prophet_df = df[['timeStamp', 'call_count']].copy()
    prophet_df.columns = ['ds', 'y']
    
    print("Training Prophet model...")
    
    try:
        # Initialize Prophet with seasonality
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative',
            interval_width=0.95
        )
        
        # Fit model
        model.fit(prophet_df)
        
        print("[OK] Prophet model trained successfully")
        
        return model
        
    except Exception as e:
        print(f"[ERROR] Error training Prophet model: {str(e)}")
        return None


def forecast_prophet(model, periods=24):
    """
    Generate forecast using Prophet model
    
    Args:
        model: Fitted Prophet model
        periods (int): Number of periods ahead to forecast (hours)
        
    Returns:
        pd.DataFrame: Forecast with confidence intervals
    """
    if model is None:
        return None
    
    try:
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq='H')
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Extract only future predictions
        forecast_df = forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_df.columns = ['timeStamp', 'forecast', 'lower_bound', 'upper_bound']
        
        return forecast_df
        
    except Exception as e:
        print(f"[ERROR] Error generating Prophet forecast: {str(e)}")
        return None


def save_arima_model(model, filepath='models/arima_model.pkl'):
    """
    Save trained ARIMA model to disk
    
    Args:
        model: Fitted ARIMA model
        filepath (str): Path to save the model
        
    Returns:
        bool: True if successful
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"[OK] ARIMA model saved to {filepath}")
        return True
    except Exception as e:
        print(f"[ERROR] Error saving ARIMA model: {str(e)}")
        return False


def load_arima_model(filepath='models/arima_model.pkl'):
    """
    Load ARIMA model from disk
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        Fitted ARIMA model or None
    """
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"[OK] ARIMA model loaded from {filepath}")
        return model
    except Exception as e:
        print(f"[ERROR] Error loading ARIMA model: {str(e)}")
        return None


def save_prophet_model(model, filepath='models/prophet_model.pkl'):
    """
    Save trained Prophet model to disk
    
    Args:
        model: Fitted Prophet model
        filepath (str): Path to save the model
        
    Returns:
        bool: True if successful
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"[OK] Prophet model saved to {filepath}")
        return True
    except Exception as e:
        print(f"[ERROR] Error saving Prophet model: {str(e)}")
        return False


def load_prophet_model(filepath='models/prophet_model.pkl'):
    """
    Load Prophet model from disk
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        Fitted Prophet model or None
    """
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"[OK] Prophet model loaded from {filepath}")
        return model
    except Exception as e:
        print(f"[ERROR] Error loading Prophet model: {str(e)}")
        return None


def train_sarima(df, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24)):
    """
    Train SARIMA (Seasonal ARIMA) model on hourly call counts
    
    Args:
        df (pd.DataFrame): Time series dataframe with 'timeStamp' and 'call_count'
        order (tuple): (p, d, q) parameters
        seasonal_order (tuple): (P, D, Q, s) seasonal parameters (s=24 for daily seasonality)
        
    Returns:
        fitted SARIMA model
    """
    if df is None or df.empty:
        print("[ERROR] Empty dataset")
        return None
    
    # Prepare time series
    ts = df.set_index('timeStamp')['call_count']
    
    print(f"Training SARIMA{order}x{seasonal_order} model...")
    
    try:
        # Fit SARIMA model
        model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        
        print(f"[OK] SARIMA model trained successfully")
        print(f"  AIC: {fitted_model.aic:.2f}")
        print(f"  BIC: {fitted_model.bic:.2f}")
        
        return fitted_model
        
    except Exception as e:
        print(f"[ERROR] Error training SARIMA model: {str(e)}")
        return None


def forecast_sarima(model, steps=24):
    """
    Generate forecast using SARIMA model
    
    Args:
        model: Fitted SARIMA model
        steps (int): Number of steps ahead to forecast
        
    Returns:
        pd.DataFrame: Forecast with confidence intervals
    """
    if model is None:
        return None
    
    try:
        # Generate forecast
        forecast = model.forecast(steps=steps)
        conf_int = model.get_forecast(steps=steps).conf_int()
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'forecast': forecast.values,
            'lower_bound': conf_int.iloc[:, 0].values,
            'upper_bound': conf_int.iloc[:, 1].values
        })
        
        # Generate future timestamps
        last_timestamp = model.model.data.dates[-1]
        future_dates = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=steps,
            freq='H'
        )
        forecast_df.index = future_dates
        forecast_df.index.name = 'timeStamp'
        
        return forecast_df.reset_index()
        
    except Exception as e:
        print(f"[ERROR] Error generating SARIMA forecast: {str(e)}")
        return None


def prepare_lstm_data(data, lookback=24):
    """
    Prepare data for LSTM model
    
    Args:
        data (pd.Series): Time series data
        lookback (int): Number of previous time steps to use
        
    Returns:
        tuple: (X, y) arrays for training
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler


def train_lstm(df, lookback=24, epochs=50, batch_size=32, units=50):
    """
    Train LSTM model on hourly call counts
    
    Args:
        df (pd.DataFrame): Time series dataframe with 'timeStamp' and 'call_count'
        lookback (int): Number of previous time steps to use
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        units (int): Number of LSTM units
        
    Returns:
        dict: Dictionary containing model, scaler, and last_data
    """
    if not TENSORFLOW_AVAILABLE:
        print("[ERROR] TensorFlow not available. Cannot train LSTM model.")
        return None
    
    if df is None or df.empty:
        print("[ERROR] Empty dataset")
        return None
    
    # Prepare time series
    ts = df.set_index('timeStamp')['call_count']
    
    print(f"Training LSTM model (lookback={lookback}, units={units})...")
    
    try:
        # Prepare data
        X, y, scaler = prepare_lstm_data(ts, lookback=lookback)
        
        # Split into train and validation
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(units=units, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(units=units, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=0
        )
        
        print(f"[OK] LSTM model trained successfully")
        print(f"  Final training loss: {history.history['loss'][-1]:.6f}")
        print(f"  Final validation loss: {history.history['val_loss'][-1]:.6f}")
        
        # Store last lookback values for forecasting
        last_data = ts.values[-lookback:].reshape(1, lookback, 1)
        last_data_scaled = scaler.transform(last_data.reshape(-1, 1)).reshape(1, lookback, 1)
        last_timestamp = ts.index[-1]
        
        return {
            'model': model,
            'scaler': scaler,
            'lookback': lookback,
            'last_data': last_data_scaled,
            'last_timestamp': last_timestamp
        }
        
    except Exception as e:
        print(f"[ERROR] Error training LSTM model: {str(e)}")
        return None


def forecast_lstm(model_dict, steps=24):
    """
    Generate forecast using LSTM model
    
    Args:
        model_dict: Dictionary containing model, scaler, and last_data
        steps (int): Number of steps ahead to forecast
        
    Returns:
        pd.DataFrame: Forecast with confidence intervals
    """
    if model_dict is None or not TENSORFLOW_AVAILABLE:
        return None
    
    try:
        model = model_dict['model']
        scaler = model_dict['scaler']
        lookback = model_dict['lookback']
        last_data = model_dict['last_data'].copy()
        
        forecasts = []
        
        # Generate forecasts step by step
        for _ in range(steps):
            # Predict next value
            next_pred = model.predict(last_data, verbose=0)
            forecasts.append(next_pred[0, 0])
            
            # Update last_data for next prediction
            last_data = np.append(last_data[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)
        
        # Inverse transform
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts = scaler.inverse_transform(forecasts).flatten()
        
        # Create forecast dataframe (LSTM doesn't provide confidence intervals easily)
        # Use a simple approximation based on historical variance
        std_dev = np.std(forecasts) * 0.1  # 10% of std as approximation
        forecast_df = pd.DataFrame({
            'forecast': forecasts,
            'lower_bound': forecasts - 1.96 * std_dev,
            'upper_bound': forecasts + 1.96 * std_dev
        })
        
        # Generate future timestamps
        last_timestamp = model_dict.get('last_timestamp', pd.Timestamp.now())
        future_dates = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=steps,
            freq='H'
        )
        forecast_df.index = future_dates
        forecast_df.index.name = 'timeStamp'
        
        return forecast_df.reset_index()
        
    except Exception as e:
        print(f"[ERROR] Error generating LSTM forecast: {str(e)}")
        return None


def save_sarima_model(model, filepath='models/sarima_model.pkl'):
    """Save trained SARIMA model to disk"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"[OK] SARIMA model saved to {filepath}")
        return True
    except Exception as e:
        print(f"[ERROR] Error saving SARIMA model: {str(e)}")
        return False


def load_sarima_model(filepath='models/sarima_model.pkl'):
    """Load SARIMA model from disk"""
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"[OK] SARIMA model loaded from {filepath}")
        return model
    except Exception as e:
        print(f"[ERROR] Error loading SARIMA model: {str(e)}")
        return None


def save_lstm_model(model_dict, filepath='models/lstm_model.pkl'):
    """Save trained LSTM model to disk"""
    if not TENSORFLOW_AVAILABLE:
        return False
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Save model separately, then save dict
        model_path = filepath.replace('.pkl', '_keras.h5')
        model_dict['model'].save(model_path)
        # Save scaler and metadata
        save_dict = {
            'scaler': model_dict['scaler'],
            'lookback': model_dict['lookback'],
            'last_data': model_dict['last_data'],
            'last_timestamp': model_dict.get('last_timestamp', pd.Timestamp.now()),
            'model_path': model_path
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"[OK] LSTM model saved to {filepath}")
        return True
    except Exception as e:
        print(f"[ERROR] Error saving LSTM model: {str(e)}")
        return False


def load_lstm_model(filepath='models/lstm_model.pkl'):
    """Load LSTM model from disk"""
    if not TENSORFLOW_AVAILABLE:
        return None
    try:
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        # Load Keras model
        model = keras.models.load_model(save_dict['model_path'])
        save_dict['model'] = model
        print(f"[OK] LSTM model loaded from {filepath}")
        return save_dict
    except Exception as e:
        print(f"[ERROR] Error loading LSTM model: {str(e)}")
        return None


def evaluate_model(actual, forecast):
    """
    Evaluate model performance using common metrics
    
    Args:
        actual (pd.Series): Actual values
        forecast (pd.Series): Forecasted values
        
    Returns:
        dict: Evaluation metrics
    """
    if actual is None or forecast is None:
        return None
    
    # Align indices
    common_idx = actual.index.intersection(forecast.index)
    if len(common_idx) == 0:
        return None
    
    actual_aligned = actual.loc[common_idx]
    forecast_aligned = forecast.loc[common_idx]
    
    # Calculate metrics
    mae = np.mean(np.abs(actual_aligned - forecast_aligned))
    mse = np.mean((actual_aligned - forecast_aligned) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual_aligned - forecast_aligned) / actual_aligned)) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }


if __name__ == "__main__":
    # Test model training
    from data_preprocessing import process_dataset
    
    print("=" * 60)
    print("MODEL TRAINING TEST")
    print("=" * 60)
    
    # Load and preprocess data
    hourly_df, _, _ = process_dataset('911.csv')
    
    if hourly_df is not None:
        # Use last 80% for training
        split_idx = int(len(hourly_df) * 0.8)
        train_df = hourly_df.iloc[:split_idx]
        test_df = hourly_df.iloc[split_idx:]
        
        # Train ARIMA
        print("\n" + "=" * 60)
        arima_model = train_arima(train_df, auto_tune=False, order=(2, 1, 2))
        if arima_model:
            arima_forecast = forecast_arima(arima_model, steps=24)
            print(f"\nARIMA Forecast (next 24 hours):")
            print(arima_forecast.head())
        
        # Train Prophet
        print("\n" + "=" * 60)
        prophet_model = train_prophet(train_df)
        if prophet_model:
            prophet_forecast = forecast_prophet(prophet_model, periods=24)
            print(f"\nProphet Forecast (next 24 hours):")
            print(prophet_forecast.head())

