"""
Streamlit Dashboard for Emergency Calls Forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import process_dataset
from model_training import (
    train_arima, forecast_arima, load_arima_model,
    train_sarima, forecast_sarima, load_sarima_model,
    train_prophet, forecast_prophet, load_prophet_model,
    train_lstm, forecast_lstm, load_lstm_model, TENSORFLOW_AVAILABLE
)
from real_time_simulation import RealTimeSimulator
from utils.helpers import get_hourly_pattern, get_daily_pattern, prepare_forecast_plot_data

# Page configuration
st.set_page_config(
    page_title="Emergency Calls Forecasting",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF4444;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF4444;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'hourly_df' not in st.session_state:
    st.session_state.hourly_df = None
if 'location_df' not in st.session_state:
    st.session_state.location_df = None
if 'arima_model' not in st.session_state:
    st.session_state.arima_model = None
if 'sarima_model' not in st.session_state:
    st.session_state.sarima_model = None
if 'prophet_model' not in st.session_state:
    st.session_state.prophet_model = None
if 'lstm_model' not in st.session_state:
    st.session_state.lstm_model = None
if 'simulator' not in st.session_state:
    st.session_state.simulator = None

# Sidebar
st.sidebar.title("🚨 Emergency Calls Forecasting")
st.sidebar.markdown("---")

# Data loading section
st.sidebar.subheader("📁 Data Management")
data_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'], key='data_upload')

if st.sidebar.button("Load Default Dataset (911.csv)") or st.session_state.data_loaded:
    if not st.session_state.data_loaded:
        with st.spinner("Loading and preprocessing data..."):
            try:
                hourly_df, processed_df, location_df = process_dataset('911.csv')
                if hourly_df is not None:
                    st.session_state.hourly_df = hourly_df
                    st.session_state.location_df = location_df
                    st.session_state.data_loaded = True
                    st.sidebar.success("[OK] Data loaded successfully!")
                else:
                    st.sidebar.error("[ERROR] Failed to load data")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")

# Model loading section
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Model Management")

if st.session_state.data_loaded:
    if st.sidebar.button("Load Saved Models"):
        with st.spinner("Loading saved models..."):
            try:
                # Try loading ARIMA
                if os.path.exists('models/arima_model.pkl'):
                    st.session_state.arima_model = load_arima_model('models/arima_model.pkl')
                    if st.session_state.arima_model:
                        st.sidebar.success("[OK] ARIMA model loaded!")
                else:
                    st.sidebar.warning("ARIMA model file not found")
                
                # Try loading SARIMA
                if os.path.exists('models/sarima_model.pkl'):
                    st.session_state.sarima_model = load_sarima_model('models/sarima_model.pkl')
                    if st.session_state.sarima_model:
                        st.sidebar.success("[OK] SARIMA model loaded!")
                else:
                    st.sidebar.warning("SARIMA model file not found")
                
                # Try loading Prophet
                if os.path.exists('models/prophet_model.pkl'):
                    st.session_state.prophet_model = load_prophet_model('models/prophet_model.pkl')
                    if st.session_state.prophet_model:
                        st.sidebar.success("[OK] Prophet model loaded!")
                else:
                    st.sidebar.warning("Prophet model file not found")
                
                # Try loading LSTM
                if TENSORFLOW_AVAILABLE and os.path.exists('models/lstm_model.pkl'):
                    st.session_state.lstm_model = load_lstm_model('models/lstm_model.pkl')
                    if st.session_state.lstm_model:
                        st.sidebar.success("[OK] LSTM model loaded!")
                elif not TENSORFLOW_AVAILABLE:
                    st.sidebar.warning("LSTM not available (TensorFlow not installed)")
                else:
                    st.sidebar.warning("LSTM model file not found")
            except Exception as e:
                st.sidebar.error(f"Error loading models: {str(e)}")
    
    if st.sidebar.button("Train ARIMA Model"):
        with st.spinner("Training ARIMA model..."):
            try:
                train_size = int(len(st.session_state.hourly_df) * 0.8)
                train_df = st.session_state.hourly_df.iloc[:train_size]
                st.session_state.arima_model = train_arima(train_df, auto_tune=False, order=(2, 1, 2))
                if st.session_state.arima_model:
                    st.sidebar.success("[OK] ARIMA model trained!")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    
    if st.sidebar.button("Train SARIMA Model"):
        with st.spinner("Training SARIMA model..."):
            try:
                train_size = int(len(st.session_state.hourly_df) * 0.8)
                train_df = st.session_state.hourly_df.iloc[:train_size]
                st.session_state.sarima_model = train_sarima(train_df, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24))
                if st.session_state.sarima_model:
                    st.sidebar.success("[OK] SARIMA model trained!")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    
    if st.sidebar.button("Train Prophet Model"):
        with st.spinner("Training Prophet model..."):
            try:
                train_size = int(len(st.session_state.hourly_df) * 0.8)
                train_df = st.session_state.hourly_df.iloc[:train_size]
                st.session_state.prophet_model = train_prophet(train_df)
                if st.session_state.prophet_model:
                    st.sidebar.success("[OK] Prophet model trained!")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    
    if TENSORFLOW_AVAILABLE and st.sidebar.button("Train LSTM Model"):
        with st.spinner("Training LSTM model (this may take several minutes)..."):
            try:
                train_size = int(len(st.session_state.hourly_df) * 0.8)
                train_df = st.session_state.hourly_df.iloc[:train_size]
                st.session_state.lstm_model = train_lstm(train_df, lookback=24, epochs=30, batch_size=32, units=50)
                if st.session_state.lstm_model:
                    st.sidebar.success("[OK] LSTM model trained!")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    elif not TENSORFLOW_AVAILABLE:
        st.sidebar.info("LSTM training requires TensorFlow (not installed)")

# Main content
st.markdown('<h1 class="main-header">🚨 Real-Time Emergency Calls Forecasting System</h1>', unsafe_allow_html=True)

if not st.session_state.data_loaded:
    st.info("👈 Please load the dataset from the sidebar to begin")
else:
    hourly_df = st.session_state.hourly_df
    location_df = st.session_state.location_df
    
    # Key Metrics
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Hours", len(hourly_df))
    with col2:
        st.metric("Total Calls", int(hourly_df['call_count'].sum()))
    with col3:
        st.metric("Avg Calls/Hour", f"{hourly_df['call_count'].mean():.2f}")
    with col4:
        st.metric("Date Range", f"{hourly_df['timeStamp'].min().date()} to {hourly_df['timeStamp'].max().date()}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Hourly Trends",
        "🧭 Seasonal Patterns",
        "🔮 Forecasts",
        "📍 Location Map",
        "🔁 Real-Time Simulation"
    ])
    
    # Tab 1: Hourly Trends
    with tab1:
        st.subheader("Hourly Call Count Trend")
        
        # Time range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=hourly_df['timeStamp'].min().date())
        with col2:
            end_date = st.date_input("End Date", value=hourly_df['timeStamp'].max().date())
        
        # Filter data
        filtered_df = hourly_df[
            (hourly_df['timeStamp'].dt.date >= start_date) &
            (hourly_df['timeStamp'].dt.date <= end_date)
        ]
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_df['timeStamp'],
            y=filtered_df['call_count'],
            mode='lines',
            name='Call Count',
            line=dict(color='#FF4444', width=2)
        ))
        fig.update_layout(
            title="Emergency Calls Over Time",
            xaxis_title="Timestamp",
            yaxis_title="Number of Calls",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.subheader("Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean", f"{filtered_df['call_count'].mean():.2f}")
        with col2:
            st.metric("Median", f"{filtered_df['call_count'].median():.2f}")
        with col3:
            st.metric("Std Dev", f"{filtered_df['call_count'].std():.2f}")
    
    # Tab 2: Seasonal Patterns
    with tab2:
        st.subheader("Seasonal Decomposition")
        
        # Hourly pattern
        hourly_pattern = get_hourly_pattern(hourly_df)
        if hourly_pattern is not None:
            fig1 = px.bar(
                hourly_pattern,
                x='hour',
                y='avg_calls',
                title="Average Calls by Hour of Day",
                labels={'hour': 'Hour of Day', 'avg_calls': 'Average Calls'},
                color='avg_calls',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        # Daily pattern
        daily_pattern = get_daily_pattern(hourly_df)
        if daily_pattern is not None:
            fig2 = px.bar(
                daily_pattern,
                x='day_name',
                y='call_count',
                title="Average Calls by Day of Week",
                labels={'day_name': 'Day of Week', 'call_count': 'Average Calls'},
                color='call_count',
                color_continuous_scale='Oranges'
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # Tab 3: Forecasts
    with tab3:
        st.subheader("24-Hour Ahead Forecasts")
        
        model_options = ["ARIMA", "SARIMA", "Prophet", "LSTM", "All Models"]
        model_choice = st.radio("Select Model", model_options, horizontal=True)
        forecast_steps = st.slider("Forecast Steps (hours)", 1, 48, 24)
        
        # Helper function to plot forecast
        def plot_forecast(forecast_df, model_name, color, hist_df=None):
            if forecast_df is None:
                return None
            fig = go.Figure()
            if hist_df is None:
                hist_df = hourly_df.tail(168)
            fig.add_trace(go.Scatter(
                x=hist_df['timeStamp'],
                y=hist_df['call_count'],
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['timeStamp'],
                y=forecast_df['forecast'],
                mode='lines',
                name=f'{model_name} Forecast',
                line=dict(color=color, width=2, dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['timeStamp'],
                y=forecast_df['upper_bound'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            # Convert hex to rgba for fill
            if color.startswith('#'):
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                fill_color = f'rgba({r},{g},{b},0.2)'
            else:
                fill_color = 'rgba(128,128,128,0.2)'
            fig.add_trace(go.Scatter(
                x=forecast_df['timeStamp'],
                y=forecast_df['lower_bound'],
                mode='lines',
                name='Confidence Interval',
                fill='tonexty',
                fillcolor=fill_color,
                line=dict(width=0)
            ))
            fig.update_layout(
                title=f"{model_name} Forecast with Confidence Intervals",
                xaxis_title="Timestamp",
                yaxis_title="Number of Calls",
                hovermode='x unified',
                height=500
            )
            return fig
        
        # ARIMA Forecast
        if model_choice in ["ARIMA", "All Models"]:
            if st.session_state.arima_model is None:
                st.warning("⚠ ARIMA model not loaded. Please load or train it from the sidebar.")
            else:
                st.subheader("ARIMA Forecast")
                arima_forecast = forecast_arima(st.session_state.arima_model, steps=forecast_steps)
                if arima_forecast is not None:
                    fig = plot_forecast(arima_forecast, "ARIMA", "#FF0000")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(arima_forecast, use_container_width=True)
        
        # SARIMA Forecast
        if model_choice in ["SARIMA", "All Models"]:
            if st.session_state.sarima_model is None:
                st.warning("⚠ SARIMA model not loaded. Please load or train it from the sidebar.")
            else:
                st.subheader("SARIMA Forecast")
                sarima_forecast = forecast_sarima(st.session_state.sarima_model, steps=forecast_steps)
                if sarima_forecast is not None:
                    fig = plot_forecast(sarima_forecast, "SARIMA", "#FF6600")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(sarima_forecast, use_container_width=True)
        
        # Prophet Forecast
        if model_choice in ["Prophet", "All Models"]:
            if st.session_state.prophet_model is None:
                st.warning("⚠ Prophet model not loaded. Please load or train it from the sidebar.")
            else:
                st.subheader("Prophet Forecast")
                prophet_forecast = forecast_prophet(st.session_state.prophet_model, periods=forecast_steps)
                if prophet_forecast is not None:
                    fig = plot_forecast(prophet_forecast, "Prophet", "#00FF00")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(prophet_forecast, use_container_width=True)
        
        # LSTM Forecast
        if model_choice in ["LSTM", "All Models"]:
            if not TENSORFLOW_AVAILABLE:
                st.warning("⚠ LSTM requires TensorFlow (not installed)")
            elif st.session_state.lstm_model is None:
                st.warning("⚠ LSTM model not loaded. Please load or train it from the sidebar.")
            else:
                st.subheader("LSTM Forecast")
                lstm_forecast = forecast_lstm(st.session_state.lstm_model, steps=forecast_steps)
                if lstm_forecast is not None:
                    fig = plot_forecast(lstm_forecast, "LSTM", "#8000FF")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(lstm_forecast, use_container_width=True)
        
        # All Models Comparison
        if model_choice == "All Models":
            st.subheader("Model Comparison")
            hist_df = hourly_df.tail(168)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_df['timeStamp'],
                y=hist_df['call_count'],
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            ))
            
            if st.session_state.arima_model:
                arima_fc = forecast_arima(st.session_state.arima_model, steps=forecast_steps)
                if arima_fc is not None:
                    fig.add_trace(go.Scatter(
                        x=arima_fc['timeStamp'],
                        y=arima_fc['forecast'],
                        mode='lines',
                        name='ARIMA',
                        line=dict(color='red', width=2, dash='dash')
                    ))
            
            if st.session_state.sarima_model:
                sarima_fc = forecast_sarima(st.session_state.sarima_model, steps=forecast_steps)
                if sarima_fc is not None:
                    fig.add_trace(go.Scatter(
                        x=sarima_fc['timeStamp'],
                        y=sarima_fc['forecast'],
                        mode='lines',
                        name='SARIMA',
                        line=dict(color='orange', width=2, dash='dash')
                    ))
            
            if st.session_state.prophet_model:
                prophet_fc = forecast_prophet(st.session_state.prophet_model, periods=forecast_steps)
                if prophet_fc is not None:
                    fig.add_trace(go.Scatter(
                        x=prophet_fc['timeStamp'],
                        y=prophet_fc['forecast'],
                        mode='lines',
                        name='Prophet',
                        line=dict(color='green', width=2, dash='dash')
                    ))
            
            if TENSORFLOW_AVAILABLE and st.session_state.lstm_model:
                lstm_fc = forecast_lstm(st.session_state.lstm_model, steps=forecast_steps)
                if lstm_fc is not None:
                    fig.add_trace(go.Scatter(
                        x=lstm_fc['timeStamp'],
                        y=lstm_fc['forecast'],
                        mode='lines',
                        name='LSTM',
                        line=dict(color='purple', width=2, dash='dash')
                    ))
            
            fig.update_layout(
                title="All Models Forecast Comparison",
                xaxis_title="Timestamp",
                yaxis_title="Number of Calls",
                hovermode='x unified',
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Location Map
    with tab4:
        st.subheader("Emergency Call Locations")
        
        if location_df is not None and len(location_df) > 0:
            # Sample data for performance (map can be slow with too many points)
            sample_size = st.slider("Number of points to display", 100, min(10000, len(location_df)), 1000)
            map_df = location_df.sample(n=min(sample_size, len(location_df)))
            
            # Color by call type
            color_map = {'EMS': 'red', 'Fire': 'orange', 'Traffic': 'blue'}
            map_df['color'] = map_df['call_type'].map(color_map)
            
            # Create map
            fig = px.scatter_mapbox(
                map_df,
                lat='lat',
                lon='lng',
                color='call_type',
                hover_data=['title', 'location', 'timeStamp'],
                zoom=10,
                height=600,
                color_discrete_map=color_map
            )
            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No location data available")
    
    # Tab 5: Real-Time Simulation
    with tab5:
        st.subheader("Real-Time Simulation")
        
        col1, col2 = st.columns(2)
        with col1:
            sim_hours = st.number_input("Simulation Hours", min_value=1, max_value=168, value=24)
        with col2:
            retrain_interval = st.number_input("Retrain Interval (hours)", min_value=1, max_value=48, value=12)
        
        if st.button("Start Simulation"):
            with st.spinner("Running simulation..."):
                try:
                    # Use subset for faster simulation
                    sim_df = hourly_df.tail(500)
                    simulator = RealTimeSimulator(sim_df, retrain_interval=retrain_interval)
                    predictions = simulator.run_simulation(num_hours=sim_hours, retrain=True)
                    
                    st.session_state.simulator = simulator
                    
                    st.success(f"[OK] Simulation completed! Processed {len(predictions)} hours")
                    
                    # Display results
                    if len(predictions) > 0:
                        # Plot actual vs predictions
                        actuals = [p['actual_calls'] for p in predictions]
                        timestamps = [p['timestamp'] for p in predictions]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=timestamps,
                            y=actuals,
                            mode='lines+markers',
                            name='Actual Calls',
                            line=dict(color='blue', width=2)
                        ))
                        
                        fig.update_layout(
                            title="Real-Time Simulation Results",
                            xaxis_title="Timestamp",
                            yaxis_title="Number of Calls",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Summary statistics
                        st.subheader("Simulation Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Calls", sum(actuals))
                        with col2:
                            st.metric("Avg Calls/Hour", f"{np.mean(actuals):.2f}")
                        with col3:
                            st.metric("Max Calls/Hour", max(actuals))
                
                except Exception as e:
                    st.error(f"Error during simulation: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Emergency Calls Forecasting System | Built with Streamlit</p>", unsafe_allow_html=True)

