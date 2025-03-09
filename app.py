import warnings
import numpy as np
import base64
import streamlit as st
import pandas as pd
import joblib
import time
import os
import matplotlib.pyplot as plt
import plotly.express as px
import shap  # Ensure SHAP is installed (pip install shap)
from sklearn.preprocessing import StandardScaler
import random

# ---------------------- Suppress Warnings ----------------------
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress all FutureWarnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress common UserWarnings

# ---------------------- Utility Functions ----------------------
def get_base64_of_bin_file(bin_file):
    """Encode binary file as Base64 string."""
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

def get_img_with_base64(file_path):
    """Return Base64 string for embedding image."""
    return f"data:image/jpeg;base64,{get_base64_of_bin_file(file_path)}"

# ---------------------- Try Importing `st_autorefresh` ----------------------
try:
    from streamlit_autorefresh import st_autorefresh
except ModuleNotFoundError:
    st_autorefresh = lambda **kwargs: 0  # Dummy function to prevent crashes
    st.warning("⚠️ `streamlit-autorefresh` not found. Auto-refresh simulation disabled.")

# ---------------------- Page Configuration ----------------------
st.set_page_config(page_title="ICU Sepsis Monitoring", layout="wide")

# ---------------------- Caching for Model & Scaler ----------------------
@st.cache_resource
def load_model_and_scaler():
    gb_model = joblib.load("Sepsis_gb_model.pkl")
    scaler = joblib.load("sepsis_scaler.pkl")
    return gb_model, scaler

gb_model, scaler = load_model_and_scaler()

# ---------------------- Data Persistence Setup ----------------------
DATA_FILE = "patient_data_log.csv"
if "patient_data_log" not in st.session_state:
    if os.path.exists(DATA_FILE):
        st.session_state.patient_data_log = pd.read_csv(DATA_FILE)
    else:
        st.session_state.patient_data_log = pd.DataFrame(columns=[
            "Timestamp", "Patient_ID", "Patient_Name", "Plasma_glucose", "Blood_Work_R1",
            "Blood_Pressure", "Blood_Work_R3", "BMI", "Blood_Work_R4", "Patient_age", "Sepsis_Risk"
        ])

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

# ---------------------- Data Transformation Fix ----------------------
# Ensure feature names and order match exactly before transforming
expected_columns = list(scaler.feature_names_in_)

if not st.session_state.patient_data_log.empty:
    X_train = st.session_state.patient_data_log.copy()  # Create a copy to avoid modifying original data
    
    # Handle missing or extra columns gracefully
    missing_features = set(expected_columns) - set(X_train.columns)
    extra_features = set(X_train.columns) - set(expected_columns)

    if missing_features:
        st.warning(f"⚠️ Missing columns in data: {missing_features}. Filling with default values.")
        for col in missing_features:
            X_train[col] = 0  # Fill missing columns with neutral values

    if extra_features:
        st.warning(f"⚠️ Extra columns detected: {extra_features}. Dropping them.")
        X_train = X_train[expected_columns]  # Drop extra columns to match expected ones

    # Strip spaces, ensure proper ordering, and convert to NumPy
    X_train.columns = X_train.columns.str.strip()  # Remove unwanted spaces
    X_train = X_train.loc[:, expected_columns]  # Ensure order consistency
    X_train = X_train.to_numpy()  # Convert to NumPy array

    # Transform with the fitted scaler
    X_train_scaled = scaler.transform(X_train)

    # Convert back to DataFrame (if needed for further operations)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=expected_columns)

    # ---------------------- SHAP Analysis ----------------------
    explainer = shap.Explainer(gb_model)
    shap_values = explainer(X_train_scaled)

    # SHAP Summary Plot
    st.write("### SHAP Summary Plot")
    fig = plt.figure(figsize=(10, 6))
    try:
        shap.summary_plot(shap_values, X_train_scaled_df, show=False, color_bar=False)
        for ax in fig.axes:
            if hasattr(ax, 'images') and len(ax.images) > 0:
                ax.images = []
    except ValueError as e:
        st.error("Error generating SHAP summary plot: " + str(e))
    
    st.pyplot(fig)
    plt.close(fig)

# ---------------------- Simulation for Automatic Data Submission ----------------------
simulate = st.sidebar.checkbox("Simulate Automatic Data Submission", value=False)
if simulate:
    refresh_count = st_autorefresh(interval=5000, limit=100, key="data_simulation")
    current_time = time.strftime("%H:%M:%S")
    simulated_data = {
        "Timestamp": current_time,
        "Patient_ID": f"Sim-{random.randint(100,999)}",
        "Patient_Name": f"Simulated Patient {random.randint(1,50)}",
        "Plasma_glucose": random.randint(80, 400),
        "Blood_Work_R1": random.randint(50, 400),
        "Blood_Work_R3": random.randint(10, 250),
        "Blood_Pressure": random.randint(40, 300),
        "BMI": round(random.uniform(18, 50), 1),
        "Blood_Work_R4": round(random.uniform(0, 7), 1),
        "Patient_age": random.randint(20, 100),
        "Sepsis_Risk": round(random.uniform(0, 1), 2)
    }
    new_entry = pd.DataFrame([simulated_data])
    st.session_state.patient_data_log = pd.concat(
        [st.session_state.patient_data_log, new_entry],
        ignore_index=True
    )
    save_data(st.session_state.patient_data_log)
    st.sidebar.write(f"Simulated data added at {current_time}. Refresh count: {refresh_count}")

# ---------------------- Monitoring Dashboard ----------------------
st.header("Monitoring Dashboard")
if st.session_state.patient_data_log.empty:
    st.info("No patient data available yet.")
else:
    show_risk_line = st.checkbox("Show Sepsis Risk Trend", value=True, key="toggle_trend")
    if show_risk_line:
        st.subheader("Sepsis Risk Trend Over Time")
        df = st.session_state.patient_data_log.copy()
        fig_trend = px.line(
            df, 
            x="Timestamp", 
            y="Sepsis_Risk", 
            color="Patient_ID",
            markers=True, 
            title="Sepsis Risk Progression Over Time"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    st.subheader("Patient Data Log")
    try:
        st.dataframe(st.session_state.patient_data_log)
    except Exception as e:
        st.dataframe(st.session_state.patient_data_log.astype(str))
