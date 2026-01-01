# 🚑 Real-Time Emergency Calls Forecasting System

A time-series based emergency call forecasting system designed to predict ambulance demand using historical 911 call data. This project leverages statistical and deep learning models to improve Emergency Medical Services (EMS) planning and response efficiency.

---

## 📌 Project Overview

Emergency Medical Services (EMS) face increasing pressure due to rising emergency calls and limited resources. Accurate forecasting of emergency call volume can help in better ambulance allocation and faster response times.

This project focuses on predicting hourly emergency call demand using multiple time-series forecasting models and comparing their performance.

---

## 🎯 Objectives

- Forecast hourly emergency emergency call volumes
- Compare traditional and deep learning time-series models
- Simulate real-time emergency demand prediction
- Assist EMS authorities in efficient ambulance deployment

---

## 📊 Dataset

- **Source:** 911 Emergency Call Dataset  
- **Date Range:** 2015 – 2020  
- **Total Calls:** 663,522  
- **Total Hours:** 40,634  
- **Average Calls per Hour:** 16.33  

### Data Preprocessing
- Timestamp conversion
- Hourly resampling
- Missing value handling
- Outlier removal
- Feature engineering
- Normalization (for LSTM)

---

## 🧠 Methodology

### Models Implemented
- **ARIMA** – Baseline forecasting model  
- **SARIMA** – Seasonal time-series forecasting  
- **Facebook Prophet** – Trend and seasonality-aware model  
- **LSTM** – Deep learning model for long-term dependencies  

### Workflow
