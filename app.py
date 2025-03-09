import warnings  # Ensure warnings module is imported first

# Suppress specific warnings
warnings.filterwarnings("ignore", message="In the future `np.bool` will be defined as the corresponding NumPy scalar.")
warnings.filterwarnings("ignore", message="The `use_column_width` parameter has been deprecated.*")
warnings.filterwarnings("ignore", message="Serialization of dataframe to Arrow table was unsuccessful.*")

import numpy as np
import base64
import streamlit as st
import pandas as pd
import joblib
import time
import os
import matplotlib.pyplot as plt
import plotly.express as px
import shap  # Ensure SHAP is installed (`pip install shap`)
from sklearn.preprocessing import StandardScaler
import random

# Fix potential issue with NumPy boolean aliasing
if not hasattr(np, 'bool'):
    np.bool = bool

# Try to import st_autorefresh; if not available, define a dummy function.
try:
    from streamlit_autorefresh import st_autorefresh
except ModuleNotFoundError:
    st_autorefresh = lambda **kwargs: 0  # Dummy function returning 0 refresh count
    st.warning("streamlit-autorefresh module not found. Auto-refresh simulation will be disabled.")

# ---------------------- Utility Functions ----------------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_img_with_base64(file_path):
    img_base64 = get_base64_of_bin_file(file_path)
    return f"data:image/jpeg;base64,{img_base64}"

# ---------------------- Page Configuration ----------------------
st.set_page_config(page_title="ICU Sepsis Monitoring", layout="wide")

# ---------------------- Theme Toggle ----------------------
theme_choice = st.sidebar.radio("Select Theme", ["Light", "Dark"])
if theme_choice == "Dark":
    sidebar_bg = "#2A2A3D"
    app_bg = "#1E1E2F"
    text_color = "#FFFFFF"
else:
    sidebar_bg = "#FFFFFF"
    app_bg = "#F7F7F7"
    text_color = "#333333"

st.markdown(f"""
    <style>
    .stApp {{
        background-color: {app_bg};
        color: {text_color};
    }}
    h1, h2, h3, h4, h5, h6, p, label {{
        color: {text_color};
    }}
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg};
    }}
    </style>
    """, unsafe_allow_html=True)

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

# ---------------------- Application Navigation ----------------------
tabs = st.tabs(["Home", "Monitoring Dashboard", "Model Insights"])

# ---------------------- Tab 0: Home ----------------------
with tabs[0]:
    st.header("ICU Sepsis Monitoring System")

    # Display background image if available
    img_path = "sepsis.jpg"
    if os.path.exists(img_path):
        img_base64 = get_img_with_base64(img_path)
    else:
        img_base64 = "https://via.placeholder.com/1000x400?text=Image+Not+Found"

    st.markdown(f"""
    <div style="
        background: url('{img_base64}') no-repeat center center fixed;
        background-size: cover;
        padding: 100px 50px;
        text-align: center;
        color: white;
        border-radius: 10px;">
        <h1>ICU Sepsis Monitoring System</h1>
        <h3>Real-time Monitoring & Insights</h3>
        <p>Welcome to our advanced monitoring system that leverages a Gradient Boosting model to assess sepsis risk in ICU patients.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("""
        **Navigation:**
        - **Monitoring Dashboard:** View trends and logs of patient data.
        - **Model Insights:** Understand model predictions through SHAP explanations.
        
        Use the sidebar to simulate automatic data submissions and switch between Light and Dark themes.
    """)
    st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------- Tab 1: Monitoring Dashboard ----------------------
with tabs[1]:  # FIXED: Correct tab index
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
        except Exception:
            st.dataframe(st.session_state.patient_data_log.astype(str))

# ---------------------- Tab 2: Model Insights (with SHAP) ----------------------
with tabs[2]:  # FIXED: Correct tab index
    st.header("Model Insights")
    st.write("Generating SHAP feature importance for: **Gradient Boosting Model**")
    
    if st.session_state.patient_data_log.empty:
        st.info("No patient data available for SHAP analysis. Using a dummy sample.")
        X_train = pd.DataFrame({
            "Plasma_glucose": np.linspace(100, 150, 10),
            "Blood_Work_R1": np.linspace(120, 160, 10),
            "Blood_Work_R3": np.linspace(30, 50, 10),
            "Blood_Pressure": np.linspace(80, 120, 10),
            "BMI": np.linspace(25, 30, 10),
            "Blood_Work_R4": np.linspace(0.5, 1.0, 10),
            "Patient_age": np.linspace(40, 60, 10)
        })
    else:
        expected_columns = list(scaler.feature_names_in_)  # FIXED: Ensure correct feature order
        X_train = st.session_state.patient_data_log[expected_columns]
    
    # Ensure there is data before scaling
    if not X_train.empty:
        X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=expected_columns)
        explainer = shap.Explainer(gb_model)
        shap_values = explainer(X_train_scaled)
        
        st.write("### SHAP Summary Plot")
        fig = plt.figure(figsize=(10, 6))
        try:
            shap.summary_plot(shap_values, X_train, show=False)
        except ValueError as e:
            st.error(f"Error generating SHAP summary plot: {e}")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("No valid data available for SHAP interpretation.")

    with st.expander("About SHAP Feature Importance"):
        st.write("""
        SHAP (SHapley Additive exPlanations) assigns each feature an importance value for a particular prediction.
        - Features at the top of the plot have the highest impact on the model output.
        - This visualization helps in understanding how each vital sign contributes to the sepsis risk prediction.
        """)
