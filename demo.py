# app.py
import streamlit as st
import pandas as pd
import numpy as np

# Page settings
st.set_page_config(page_title="Solar Forecasting - Manual Input", layout="centered")
st.title("☀️ Manual Input Solar Prediction (Simulated)")
st.markdown("Enter sensor data and model settings to simulate a solar prediction.")

# Sidebar model and horizon selection
st.sidebar.header("⚙️ Prediction Settings")
model_choice = st.sidebar.selectbox("Choose Model", [
    "LSTM", "CNN", "GRU", "Bi-LSTM", "CNN-LSTM", "TFT-GRU", "Informer", "Autoformer", "TC-Former"
])
horizon = st.sidebar.selectbox("Prediction Horizon", [
    "30 minutes", "1 hour", "2 hours", "24 hours"
])

# Input form
st.markdown("### 📥 Sensor Inputs")
with st.form("solar_form"):
    ghi_pyr = st.number_input("ghi_pyr", min_value=0.0, max_value=1500.0, step=10.0)
    ghi_rsi = st.number_input("ghi_rsi", min_value=0.0, max_value=1500.0, step=10.0)
    dni = st.number_input("dni", min_value=0.0, max_value=1500.0, step=10.0)
    dhi = st.number_input("dhi", min_value=0.0, max_value=1500.0, step=10.0)
    air_temp = st.number_input("air_temperature", min_value=-30.0, max_value=60.0, step=0.5)
    humidity = st.number_input("relative_humidity", min_value=0.0, max_value=100.0, step=1.0)
    wind_speed = st.number_input("wind_speed", min_value=0.0, max_value=50.0, step=0.5)
    gust = st.number_input("wind_speed_of_gust", min_value=0.0, max_value=100.0, step=0.5)
    wind_dir_std = st.number_input("wind_from_direction_st_dev", min_value=0.0, max_value=360.0, step=1.0)
    wind_dir = st.number_input("wind_from_direction", min_value=0.0, max_value=360.0, step=1.0)
    pressure = st.number_input("barometric_pressure", min_value=800.0, max_value=1100.0, step=0.5)
    cleaning = st.number_input("sensor_cleaning", min_value=0.0, max_value=1.0, step=0.1)

    submit = st.form_submit_button("⚡ Predict")

# Simulated prediction logic
def simulate_prediction(features, model_name, horizon_label):
    weights = {
        'ghi_pyr': 0.5, 'ghi_rsi': 0.2, 'dni': 0.1, 'dhi': 0.1,
        'air_temperature': 0.01, 'relative_humidity': -0.01,
        'wind_speed': -0.01, 'wind_speed_of_gust': -0.01,
        'wind_from_direction_st_dev': 0.0, 'wind_from_direction': 0.0,
        'barometric_pressure': 0.01, 'sensor_cleaning': 0.0
    }
    base = sum(features[k] * weights.get(k, 0) for k in features)
    noise = np.random.normal(0, 5)  # simulate noise
    horizon_multiplier = {
        "30 minutes": 0.95,
        "1 hour": 1.0,
        "2 hours": 1.05,
        "24 hours": 1.1
    }.get(horizon_label, 1.0)

    return base * horizon_multiplier + noise

# On submit
if submit:
    features = {
        'ghi_pyr': ghi_pyr,
        'ghi_rsi': ghi_rsi,
        'dni': dni,
        'dhi': dhi,
        'air_temperature': air_temp,
        'relative_humidity': humidity,
        'wind_speed': wind_speed,
        'wind_speed_of_gust': gust,
        'wind_from_direction_st_dev': wind_dir_std,
        'wind_from_direction': wind_dir,
        'barometric_pressure': pressure,
        'sensor_cleaning': cleaning
    }

    prediction = simulate_prediction(features, model_choice, horizon)

    st.success(f"✅ Simulated Predicted GHI ({model_choice}, {horizon}): **{prediction:.2f} W/m²**")

    # Show inputs + result
    result_df = pd.DataFrame([features])
    result_df["Model"] = model_choice
    result_df["Horizon"] = horizon
    result_df["Predicted_GHI"] = prediction

    st.markdown("### 📋 Input Summary")
    st.dataframe(result_df.T.rename(columns={0: "Value"}))

    # Download
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Result as CSV", csv, "solar_prediction_result.csv", "text/csv")
