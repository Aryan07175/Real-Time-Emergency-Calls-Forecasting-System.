"""
Data Preprocessing Module
Loads and preprocesses 911 emergency call data for time series analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_dataset(file_path='911.csv'):
    """
    Load the 911 emergency calls dataset from CSV
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Raw dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"[OK] Loaded {len(df)} records from {file_path}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] File {file_path} not found")
        return None


def preprocess_data(df):
    """
    Preprocess the raw dataset for time series analysis
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Preprocessed time series dataframe with hourly call counts
    """
    if df is None or df.empty:
        print("[ERROR] Empty dataset")
        return None
    
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Convert timestamp to datetime
    df_processed['timeStamp'] = pd.to_datetime(df_processed['timeStamp'], errors='coerce')
    
    # Extract call type from title (EMS, Fire, Traffic)
    df_processed['call_type'] = df_processed['title'].str.split(':').str[0]
    
    # Create priority level based on call type
    priority_map = {
        'EMS': 1,  # Highest priority
        'Fire': 2,
        'Traffic': 3
    }
    df_processed['priority_level'] = df_processed['call_type'].map(priority_map).fillna(3)
    
    # Create location identifier (using township or address)
    df_processed['location'] = df_processed['twp'].fillna(df_processed['addr'])
    
    # Remove rows with invalid timestamps
    df_processed = df_processed.dropna(subset=['timeStamp'])
    
    # Sort by timestamp
    df_processed = df_processed.sort_values('timeStamp').reset_index(drop=True)
    
    print(f"[OK] Preprocessed {len(df_processed)} valid records")
    print(f"  Date range: {df_processed['timeStamp'].min()} to {df_processed['timeStamp'].max()}")
    
    return df_processed


def resample_to_hourly(df):
    """
    Resample the dataset to hourly call counts
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
        
    Returns:
        pd.DataFrame: Time series with hourly call counts
    """
    if df is None or df.empty:
        return None
    
    # Set timestamp as index
    df_indexed = df.set_index('timeStamp')
    
    # Resample to hourly and count calls
    hourly_counts = df_indexed.resample('H').size().reset_index(name='call_count')
    
    # Add additional features
    hourly_counts['hour'] = hourly_counts['timeStamp'].dt.hour
    hourly_counts['day_of_week'] = hourly_counts['timeStamp'].dt.dayofweek
    hourly_counts['month'] = hourly_counts['timeStamp'].dt.month
    
    # Fill missing hours with 0 (if any gaps)
    date_range = pd.date_range(
        start=hourly_counts['timeStamp'].min(),
        end=hourly_counts['timeStamp'].max(),
        freq='H'
    )
    hourly_counts = hourly_counts.set_index('timeStamp').reindex(date_range, fill_value=0).reset_index()
    hourly_counts.rename(columns={'index': 'timeStamp'}, inplace=True)
    
    # Recalculate derived features
    hourly_counts['hour'] = hourly_counts['timeStamp'].dt.hour
    hourly_counts['day_of_week'] = hourly_counts['timeStamp'].dt.dayofweek
    hourly_counts['month'] = hourly_counts['timeStamp'].dt.month
    
    print(f"[OK] Resampled to {len(hourly_counts)} hourly intervals")
    print(f"  Average calls per hour: {hourly_counts['call_count'].mean():.2f}")
    
    return hourly_counts


def get_call_type_counts(df):
    """
    Get hourly counts by call type
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
        
    Returns:
        pd.DataFrame: Hourly counts by call type
    """
    if df is None or df.empty:
        return None
    
    df_indexed = df.set_index('timeStamp')
    
    # Resample by call type
    call_type_counts = df_indexed.groupby('call_type').resample('H').size().unstack(fill_value=0)
    call_type_counts = call_type_counts.T.reset_index()
    call_type_counts.rename(columns={'timeStamp': 'timestamp'}, inplace=True)
    
    return call_type_counts


def get_location_data(df):
    """
    Extract location data for mapping
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
        
    Returns:
        pd.DataFrame: Location data with coordinates
    """
    if df is None or df.empty:
        return None
    
    location_df = df[['timeStamp', 'lat', 'lng', 'call_type', 'location', 'title']].copy()
    location_df = location_df.dropna(subset=['lat', 'lng'])
    
    return location_df


def process_dataset(file_path='911.csv'):
    """
    Complete preprocessing pipeline
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        tuple: (hourly_counts_df, preprocessed_df, location_df)
    """
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Load dataset
    df_raw = load_dataset(file_path)
    if df_raw is None:
        return None, None, None
    
    # Preprocess
    df_processed = preprocess_data(df_raw)
    if df_processed is None:
        return None, None, None
    
    # Resample to hourly
    hourly_counts = resample_to_hourly(df_processed)
    
    # Get location data
    location_df = get_location_data(df_processed)
    
    print("=" * 60)
    print("[OK] Preprocessing complete!")
    print("=" * 60)
    
    return hourly_counts, df_processed, location_df


if __name__ == "__main__":
    # Test the preprocessing
    hourly_df, processed_df, location_df = process_dataset('911.csv')
    
    if hourly_df is not None:
        print("\nFirst few rows of hourly data:")
        print(hourly_df.head())
        print("\nDataset info:")
        print(hourly_df.info())

