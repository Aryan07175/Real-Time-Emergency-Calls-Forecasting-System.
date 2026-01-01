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

Data Collection
↓
Preprocessing
↓
Train-Test Split
↓
Model Training
↓
Forecasting
↓
Real-Time Simulation & Visualization


---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries & Tools:**
  - pandas, numpy
  - statsmodels
  - Prophet
  - TensorFlow / Keras
  - Streamlit

---

## 📈 Results & Discussion

- **SARIMA** performed best for short-term forecasting
- **LSTM** captured long-term trends effectively
- **Prophet** handled seasonality and trend changes well
- **ARIMA** struggled with strong seasonal patterns

The system demonstrated improved forecasting accuracy, making it suitable for EMS planning.

---

## ✅ Conclusion

This project proves that time-series forecasting can significantly enhance emergency response systems. Accurate predictions enable better ambulance allocation, reduced response times, and improved public safety.

---

## 🔮 Future Scope

- Integration of weather and traffic data
- Cloud deployment for real-time usage
- Transformer-based forecasting models
- Reinforcement learning for dynamic ambulance allocation
- Anomaly detection for emergency surges



## 📚 References

- Jones, L., & Brown, T. (2019)  
- Klem, R., & Ibrahim, S. (2020)  
- Zhang, P., Liu, Y., & Chen, H. (2021)  
- US EMS Analytics Report (2022)  
- Chen, J., & Wang, H. (2023)  

-
