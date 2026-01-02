"""
Training Script - Train and save models systematically
Run this script to train ARIMA, SARIMA, Prophet, and LSTM models
"""

import pandas as pd
import numpy as np
import os
from data_preprocessing import process_dataset
from model_training import (
    train_arima, forecast_arima, save_arima_model,
    train_sarima, forecast_sarima, save_sarima_model,
    auto_arima_params, TENSORFLOW_AVAILABLE
)

# Try importing Prophet - it may not be available on all systems
try:
    from model_training import train_prophet, forecast_prophet, save_prophet_model
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("[WARNING] Prophet not available.")

# Try importing LSTM functions
try:
    from model_training import train_lstm, forecast_lstm, save_lstm_model
    LSTM_AVAILABLE = TENSORFLOW_AVAILABLE
except ImportError:
    LSTM_AVAILABLE = False
    print("[WARNING] LSTM not available.")

def main():
    print("=" * 70)
    print("EMERGENCY CALLS FORECASTING - MODEL TRAINING")
    print("=" * 70)
    print("Training: ARIMA, SARIMA, Prophet, LSTM")
    print("=" * 70)
    
    # Step 1: Load and preprocess data
    print("\n[Step 1/9] Loading and preprocessing data...")
    hourly_df, processed_df, location_df = process_dataset('911.csv')
    
    if hourly_df is None:
        print("[ERROR] Failed to load data. Exiting.")
        return
    
    print(f"[OK] Data loaded: {len(hourly_df)} hourly records")
    
    # Step 2: Train-Test Split
    print("\n[Step 2/9] Splitting data into train and test sets...")
    split_idx = int(len(hourly_df) * 0.8)
    train_df = hourly_df.iloc[:split_idx].copy()
    test_df = hourly_df.iloc[split_idx:].copy()
    
    print(f"[OK] Training set: {len(train_df)} records ({len(train_df)/len(hourly_df)*100:.1f}%)")
    print(f"[OK] Test set: {len(test_df)} records ({len(test_df)/len(hourly_df)*100:.1f}%)")
    
    # Step 3: Train ARIMA Model
    print("\n[Step 3/9] Training ARIMA model...")
    arima_order = (2, 1, 2)  # Can be changed or auto-tuned
    print(f"  Using ARIMA order: {arima_order}")
    
    arima_model = train_arima(train_df, auto_tune=False, order=arima_order)
    
    if arima_model is None:
        print("[ERROR] ARIMA training failed. Continuing with Prophet...")
    else:
        print("[OK] ARIMA model trained successfully")
        print(f"  AIC: {arima_model.aic:.2f}, BIC: {arima_model.bic:.2f}")
    
    # Step 4: Train SARIMA Model
    print("\n[Step 4/9] Training SARIMA model...")
    sarima_order = (2, 1, 2)
    sarima_seasonal_order = (1, 1, 1, 24)  # Daily seasonality
    print(f"  Using SARIMA order: {sarima_order}x{sarima_seasonal_order}")
    
    sarima_model = train_sarima(train_df, order=sarima_order, seasonal_order=sarima_seasonal_order)
    
    if sarima_model is None:
        print("[ERROR] SARIMA training failed.")
    else:
        print("[OK] SARIMA model trained successfully")
        print(f"  AIC: {sarima_model.aic:.2f}, BIC: {sarima_model.bic:.2f}")
    
    # Step 5: Train Prophet Model
    prophet_model = None
    if PROPHET_AVAILABLE:
        print("\n[Step 5/9] Training Prophet model...")
        prophet_model = train_prophet(train_df)
        
        if prophet_model is None:
            print("[ERROR] Prophet training failed.")
        else:
            print("[OK] Prophet model trained successfully")
    else:
        print("\n[Step 5/9] Skipping Prophet model (not available)")
    
    # Step 6: Train LSTM Model
    lstm_model = None
    if LSTM_AVAILABLE:
        print("\n[Step 6/9] Training LSTM model...")
        print("  This may take several minutes...")
        lstm_model = train_lstm(train_df, lookback=24, epochs=30, batch_size=32, units=50)
        
        if lstm_model is None:
            print("[ERROR] LSTM training failed.")
        else:
            print("[OK] LSTM model trained successfully")
    else:
        print("\n[Step 6/9] Skipping LSTM model (TensorFlow not available)")
    
    # Step 7: Evaluate models on test set
    print("\n[Step 7/9] Evaluating models on test set...")
    
    if arima_model is not None:
        test_steps = len(test_df)
        arima_forecast = forecast_arima(arima_model, steps=test_steps)
        
        if arima_forecast is not None:
            test_ts = test_df.set_index('timeStamp')['call_count']
            forecast_ts = arima_forecast.set_index('timeStamp')['forecast']
            common_idx = test_ts.index.intersection(forecast_ts.index)
            
            if len(common_idx) > 0:
                test_aligned = test_ts.loc[common_idx]
                forecast_aligned = forecast_ts.loc[common_idx]
                
                mae = np.mean(np.abs(test_aligned - forecast_aligned))
                rmse = np.sqrt(np.mean((test_aligned - forecast_aligned) ** 2))
                mape = np.mean(np.abs((test_aligned - forecast_aligned) / test_aligned)) * 100
                
                print(f"  ARIMA Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    
    if PROPHET_AVAILABLE and prophet_model is not None:
        test_periods = len(test_df)
        prophet_forecast = forecast_prophet(prophet_model, periods=test_periods)
        
        if prophet_forecast is not None:
            test_ts = test_df.set_index('timeStamp')['call_count']
            forecast_ts = prophet_forecast.set_index('timeStamp')['forecast']
            common_idx = test_ts.index.intersection(forecast_ts.index)
            
            if len(common_idx) > 0:
                test_aligned = test_ts.loc[common_idx]
                forecast_aligned = forecast_ts.loc[common_idx]
                
                mae = np.mean(np.abs(test_aligned - forecast_aligned))
                rmse = np.sqrt(np.mean((test_aligned - forecast_aligned) ** 2))
                mape = np.mean(np.abs((test_aligned - forecast_aligned) / test_aligned)) * 100
                
                print(f"  Prophet Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    
    if sarima_model is not None:
        test_steps = len(test_df)
        sarima_forecast = forecast_sarima(sarima_model, steps=test_steps)
        
        if sarima_forecast is not None:
            test_ts = test_df.set_index('timeStamp')['call_count']
            forecast_ts = sarima_forecast.set_index('timeStamp')['forecast']
            common_idx = test_ts.index.intersection(forecast_ts.index)
            
            if len(common_idx) > 0:
                test_aligned = test_ts.loc[common_idx]
                forecast_aligned = forecast_ts.loc[common_idx]
                
                mae = np.mean(np.abs(test_aligned - forecast_aligned))
                rmse = np.sqrt(np.mean((test_aligned - forecast_aligned) ** 2))
                mape = np.mean(np.abs((test_aligned - forecast_aligned) / test_aligned)) * 100
                
                print(f"  SARIMA Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    
    if LSTM_AVAILABLE and lstm_model is not None:
        test_steps = len(test_df)
        lstm_forecast = forecast_lstm(lstm_model, steps=test_steps)
        
        if lstm_forecast is not None:
            test_ts = test_df.set_index('timeStamp')['call_count']
            forecast_ts = lstm_forecast.set_index('timeStamp')['forecast']
            common_idx = test_ts.index.intersection(forecast_ts.index)
            
            if len(common_idx) > 0:
                test_aligned = test_ts.loc[common_idx]
                forecast_aligned = forecast_ts.loc[common_idx]
                
                mae = np.mean(np.abs(test_aligned - forecast_aligned))
                rmse = np.sqrt(np.mean((test_aligned - forecast_aligned) ** 2))
                mape = np.mean(np.abs((test_aligned - forecast_aligned) / test_aligned)) * 100
                
                print(f"  LSTM Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
    
    # Step 8: Save models
    print("\n[Step 8/9] Saving trained models...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    if arima_model is not None:
        arima_saved = save_arima_model(arima_model, filepath='models/arima_model.pkl')
        if not arima_saved:
            print("[ERROR] Failed to save ARIMA model")
    else:
        print("[WARNING] No ARIMA model to save")
    
    if sarima_model is not None:
        sarima_saved = save_sarima_model(sarima_model, filepath='models/sarima_model.pkl')
        if not sarima_saved:
            print("[ERROR] Failed to save SARIMA model")
    
    if PROPHET_AVAILABLE and prophet_model is not None:
        prophet_saved = save_prophet_model(prophet_model, filepath='models/prophet_model.pkl')
        if not prophet_saved:
            print("[ERROR] Failed to save Prophet model")
    else:
        print("[WARNING] No Prophet model to save")
    
    if LSTM_AVAILABLE and lstm_model is not None:
        lstm_saved = save_lstm_model(lstm_model, filepath='models/lstm_model.pkl')
        if not lstm_saved:
            print("[ERROR] Failed to save LSTM model")
    else:
        print("[WARNING] No LSTM model to save")
    
    # Step 9: Generate 24-hour forecast
    print("\n[Step 9/9] Generating 24-hour ahead forecasts...")
    
    if arima_model is not None:
        future_arima = forecast_arima(arima_model, steps=24)
        if future_arima is not None:
            print("[OK] ARIMA 24-hour forecast generated")
            print(f"  Forecast range: {future_arima['forecast'].min():.1f} - {future_arima['forecast'].max():.1f} calls/hour")
    
    if sarima_model is not None:
        future_sarima = forecast_sarima(sarima_model, steps=24)
        if future_sarima is not None:
            print("[OK] SARIMA 24-hour forecast generated")
            print(f"  Forecast range: {future_sarima['forecast'].min():.1f} - {future_sarima['forecast'].max():.1f} calls/hour")
    
    if PROPHET_AVAILABLE and prophet_model is not None:
        future_prophet = forecast_prophet(prophet_model, periods=24)
        if future_prophet is not None:
            print("[OK] Prophet 24-hour forecast generated")
            print(f"  Forecast range: {future_prophet['forecast'].min():.1f} - {future_prophet['forecast'].max():.1f} calls/hour")
    
    if LSTM_AVAILABLE and lstm_model is not None:
        future_lstm = forecast_lstm(lstm_model, steps=24)
        if future_lstm is not None:
            print("[OK] LSTM 24-hour forecast generated")
            print(f"  Forecast range: {future_lstm['forecast'].min():.1f} - {future_lstm['forecast'].max():.1f} calls/hour")
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\n[OK] Models saved to:")
    if arima_model is not None:
        print("  - models/arima_model.pkl")
    if sarima_model is not None:
        print("  - models/sarima_model.pkl")
    if prophet_model is not None:
        print("  - models/prophet_model.pkl")
    if lstm_model is not None:
        print("  - models/lstm_model.pkl")
    print("\nYou can now use these models in the Streamlit dashboard or for predictions.")
    print("=" * 70)

if __name__ == "__main__":
    main()

