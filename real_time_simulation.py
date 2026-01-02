"""
Real-Time Simulation Module
Simulates real-time streaming of emergency calls and rolling predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from model_training import (
    train_arima, forecast_arima,
    train_sarima, forecast_sarima,
    train_prophet, forecast_prophet,
    train_lstm, forecast_lstm, TENSORFLOW_AVAILABLE
)


class RealTimeSimulator:
    """
    Simulates real-time emergency call streaming
    """
    
    def __init__(self, historical_df, retrain_interval=24):
        """
        Initialize the simulator
        
        Args:
            historical_df (pd.DataFrame): Historical data for initialization
            retrain_interval (int): Number of hours before retraining model
        """
        self.historical_df = historical_df.copy()
        self.current_df = historical_df.copy()
        self.retrain_interval = retrain_interval
        self.step_count = 0
        self.arima_model = None
        self.sarima_model = None
        self.prophet_model = None
        self.lstm_model = None
        self.last_retrain_step = 0
        
    def generate_synthetic_call(self, base_timestamp):
        """
        Generate a synthetic emergency call based on historical patterns
        
        Args:
            base_timestamp (pd.Timestamp): Base timestamp for the call
            
        Returns:
            dict: Synthetic call data
        """
        # Extract hour and day of week patterns
        hour = base_timestamp.hour
        day_of_week = base_timestamp.dayofweek
        
        # Simulate call count based on time patterns
        # Higher during day hours (8-20) and weekdays
        base_rate = 5
        hour_multiplier = 1.5 if 8 <= hour <= 20 else 0.7
        day_multiplier = 1.2 if day_of_week < 5 else 0.9
        
        # Generate Poisson-distributed call count
        call_count = np.random.poisson(base_rate * hour_multiplier * day_multiplier)
        
        return {
            'timeStamp': base_timestamp,
            'call_count': call_count
        }
    
    def stream_next_hour(self):
        """
        Simulate streaming the next hour of data
        
        Returns:
            pd.DataFrame: New hourly data
        """
        # Get last timestamp
        last_timestamp = self.current_df['timeStamp'].max()
        next_timestamp = last_timestamp + pd.Timedelta(hours=1)
        
        # Generate synthetic call
        new_call = self.generate_synthetic_call(next_timestamp)
        
        # Create new row
        new_row = pd.DataFrame([new_call])
        
        # Add derived features
        new_row['hour'] = new_row['timeStamp'].dt.hour
        new_row['day_of_week'] = new_row['timeStamp'].dt.dayofweek
        new_row['month'] = new_row['timeStamp'].dt.month
        
        # Append to current dataframe
        self.current_df = pd.concat([self.current_df, new_row], ignore_index=True)
        self.step_count += 1
        
        return new_row
    
    def should_retrain(self):
        """
        Check if model should be retrained
        
        Returns:
            bool: True if retraining is needed
        """
        return (self.step_count - self.last_retrain_step) >= self.retrain_interval
    
    def retrain_models(self, include_lstm=False):
        """
        Retrain all models on current data
        
        Args:
            include_lstm (bool): Whether to retrain LSTM (slower)
        """
        print(f"\n[RETRAIN] Retraining models at step {self.step_count}...")
        
        # Use last 80% of data for training
        train_size = int(len(self.current_df) * 0.8)
        train_df = self.current_df.iloc[-train_size:]
        
        # Retrain ARIMA
        print("  Training ARIMA...")
        self.arima_model = train_arima(train_df, auto_tune=False, order=(2, 1, 2))
        
        # Retrain SARIMA
        print("  Training SARIMA...")
        self.sarima_model = train_sarima(train_df, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24))
        
        # Retrain Prophet
        print("  Training Prophet...")
        try:
            self.prophet_model = train_prophet(train_df)
        except Exception as e:
            print(f"  [WARNING] Prophet training failed: {str(e)}")
        
        # Retrain LSTM (optional, slower)
        if include_lstm and TENSORFLOW_AVAILABLE:
            print("  Training LSTM (this may take a while)...")
            try:
                self.lstm_model = train_lstm(train_df, lookback=24, epochs=20, batch_size=32, units=50)
            except Exception as e:
                print(f"  [WARNING] LSTM training failed: {str(e)}")
        elif include_lstm and not TENSORFLOW_AVAILABLE:
            print("  [SKIP] LSTM training skipped (TensorFlow not available)")
        
        self.last_retrain_step = self.step_count
        print("[OK] Models retrained successfully")
    
    def get_predictions(self, steps=24):
        """
        Get predictions from all models
        
        Args:
            steps (int): Number of steps ahead to predict
            
        Returns:
            dict: Predictions from all models
        """
        predictions = {
            'arima': None,
            'sarima': None,
            'prophet': None,
            'lstm': None
        }
        
        if self.arima_model is not None:
            try:
                predictions['arima'] = forecast_arima(self.arima_model, steps=steps)
            except Exception as e:
                print(f"  [WARNING] ARIMA forecast failed: {str(e)}")
        
        if self.sarima_model is not None:
            try:
                predictions['sarima'] = forecast_sarima(self.sarima_model, steps=steps)
            except Exception as e:
                print(f"  [WARNING] SARIMA forecast failed: {str(e)}")
        
        if self.prophet_model is not None:
            try:
                predictions['prophet'] = forecast_prophet(self.prophet_model, periods=steps)
            except Exception as e:
                print(f"  [WARNING] Prophet forecast failed: {str(e)}")
        
        if self.lstm_model is not None and TENSORFLOW_AVAILABLE:
            try:
                predictions['lstm'] = forecast_lstm(self.lstm_model, steps=steps)
            except Exception as e:
                print(f"  [WARNING] LSTM forecast failed: {str(e)}")
        
        return predictions
    
    def run_simulation(self, num_hours=48, retrain=True, include_lstm=False):
        """
        Run the real-time simulation
        
        Args:
            num_hours (int): Number of hours to simulate
            retrain (bool): Whether to retrain models during simulation
            include_lstm (bool): Whether to include LSTM in retraining (slower)
            
        Returns:
            list: List of predictions at each step
        """
        print("=" * 60)
        print("REAL-TIME SIMULATION")
        print("=" * 60)
        print(f"Simulating {num_hours} hours...")
        print(f"Retrain interval: {self.retrain_interval} hours")
        print(f"Include LSTM: {include_lstm}")
        
        # Initial training
        if retrain:
            self.retrain_models(include_lstm=include_lstm)
        
        all_predictions = []
        
        for hour in range(num_hours):
            # Stream next hour
            new_data = self.stream_next_hour()
            
            # Check if retraining is needed
            if retrain and self.should_retrain():
                self.retrain_models(include_lstm=include_lstm)
            
            # Get predictions
            predictions = self.get_predictions(steps=24)
            
            all_predictions.append({
                'step': self.step_count,
                'timestamp': new_data['timeStamp'].iloc[0],
                'actual_calls': new_data['call_count'].iloc[0],
                'predictions': predictions
            })
            
            if (hour + 1) % 10 == 0:
                print(f"  Processed {hour + 1}/{num_hours} hours...")
        
        print("=" * 60)
        print("[OK] Simulation complete!")
        print("=" * 60)
        
        return all_predictions


def simulate_real_time(historical_df, num_hours=48, retrain_interval=24):
    """
    Convenience function to run real-time simulation
    
    Args:
        historical_df (pd.DataFrame): Historical hourly data
        num_hours (int): Number of hours to simulate
        retrain_interval (int): Hours between retraining
        
    Returns:
        RealTimeSimulator: Simulator object with results
    """
    simulator = RealTimeSimulator(historical_df, retrain_interval=retrain_interval)
    predictions = simulator.run_simulation(num_hours=num_hours, retrain=True)
    return simulator, predictions


if __name__ == "__main__":
    # Test real-time simulation
    from data_preprocessing import process_dataset
    
    print("=" * 60)
    print("REAL-TIME SIMULATION TEST")
    print("=" * 60)
    
    # Load and preprocess data
    hourly_df, _, _ = process_dataset('911.csv')
    
    if hourly_df is not None:
        # Use subset for faster testing
        test_df = hourly_df.tail(1000)
        
        # Run simulation
        simulator, predictions = simulate_real_time(
            test_df,
            num_hours=24,
            retrain_interval=12
        )
        
        print(f"\nSimulation completed with {len(predictions)} steps")
        print(f"Final dataset size: {len(simulator.current_df)} records")

