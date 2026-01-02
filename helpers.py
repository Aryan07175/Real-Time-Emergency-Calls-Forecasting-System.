"""
Helper Utilities
Common utility functions for the project
"""

import pandas as pd
import numpy as np
from datetime import datetime


def format_timestamp(ts):
    """
    Format timestamp for display
    
    Args:
        ts: Timestamp (pd.Timestamp or str)
        
    Returns:
        str: Formatted timestamp
    """
    if isinstance(ts, str):
        ts = pd.to_datetime(ts)
    return ts.strftime('%Y-%m-%d %H:%M:%S')


def calculate_statistics(df, column='call_count'):
    """
    Calculate basic statistics for a column
    
    Args:
        df (pd.DataFrame): Dataframe
        column (str): Column name
        
    Returns:
        dict: Statistics
    """
    if df is None or column not in df.columns:
        return None
    
    return {
        'mean': df[column].mean(),
        'median': df[column].median(),
        'std': df[column].std(),
        'min': df[column].min(),
        'max': df[column].max(),
        'total': df[column].sum()
    }


def get_hourly_pattern(df):
    """
    Get average call count by hour of day
    
    Args:
        df (pd.DataFrame): Dataframe with 'hour' and 'call_count' columns
        
    Returns:
        pd.DataFrame: Average calls by hour
    """
    if df is None or 'hour' not in df.columns:
        return None
    
    hourly_pattern = df.groupby('hour')['call_count'].mean().reset_index()
    hourly_pattern.columns = ['hour', 'avg_calls']
    
    return hourly_pattern


def get_daily_pattern(df):
    """
    Get average call count by day of week
    
    Args:
        df (pd.DataFrame): Dataframe with 'day_of_week' and 'call_count' columns
        
    Returns:
        pd.DataFrame: Average calls by day
    """
    if df is None or 'day_of_week' not in df.columns:
        return None
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    daily_pattern = df.groupby('day_of_week')['call_count'].mean().reset_index()
    daily_pattern['day_name'] = daily_pattern['day_of_week'].map(lambda x: day_names[x])
    
    return daily_pattern


def prepare_forecast_plot_data(historical_df, forecast_df, model_name='Model'):
    """
    Prepare data for plotting historical and forecast data together
    
    Args:
        historical_df (pd.DataFrame): Historical data
        forecast_df (pd.DataFrame): Forecast data
        model_name (str): Name of the model
        
    Returns:
        pd.DataFrame: Combined data for plotting
    """
    if historical_df is None or forecast_df is None:
        return None
    
    # Prepare historical data
    hist_plot = historical_df[['timeStamp', 'call_count']].copy()
    hist_plot['type'] = 'Historical'
    
    # Prepare forecast data
    forecast_plot = forecast_df[['timeStamp', 'forecast']].copy()
    forecast_plot.columns = ['timeStamp', 'call_count']
    forecast_plot['type'] = f'{model_name} Forecast'
    
    # Combine
    combined = pd.concat([hist_plot, forecast_plot], ignore_index=True)
    
    return combined

