import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
np.bool = bool  # Ensure compatibility with older NumPy versions

import base64
import streamlit as st
import pandas as pd
import joblib
import time
import os
import matplotlib.pyplot as plt
import plotly.express as px
import shap  # Ensure shap is installed (pip install shap)
from sklearn.preprocessing import StandardScaler
import random

# Try to import st_autorefresh; if not available, define a dummy function.
try:
    from streamlit_autorefresh import st_autorefresh
except ModuleNotFoundError:
    st_autorefresh = lambda **kwargs: 0  # Dummy function
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
st.set_page_config(
    page_title="ICU Sepsis Monitoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Sidebar Navigation ----------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Patient Entry", "Monitoring Dashboard", "Model Insights"]
)

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

# ---------------------- Home Page ----------------------
if menu == "Home":
    st.title("ICU Sepsis Monitoring System")
    st.write("Welcome to the ICU Sepsis Monitoring System. Use the sidebar to navigate through different functionalities.")

# ---------------------- Patient Entry ----------------------
elif menu == "Patient Entry":
    st.header("Patient Data Entry")
    with st.form(key="patient_entry_form", clear_on_submit=True):
        patient_id = st.text_input("Enter Patient ID", help="Unique identifier for the patient")
        patient_name = st.text_input("Enter Patient Name", help="Full name of the patient")
        plasma_glucose = st.slider("Plasma Glucose", 0, 600, 100)
        blood_work_r1 = st.slider("Blood Work R1", 50, 500, 120)
        blood_pressure = st.slider("Blood Pressure", 40, 300, 80)
        blood_work_r3 = st.slider("Blood Work R3", 0, 300, 30)
        bmi = st.slider("BMI", 10.0, 70.0, 25.0)
        blood_work_r4 = st.slider("Blood Work R4", 0.0, 7.0, 0.5)
        age = st.slider("Age", 18, 110, 40)
        submit_button = st.form_submit_button("Submit Data")
    
    if submit_button and patient_id and patient_name:
        current_time = time.strftime("%H:%M:%S")
        new_entry = pd.DataFrame([{
            "Timestamp": current_time,
            "Patient_ID": patient_id,
            "Patient_Name": patient_name,
            "Plasma_glucose": plasma_glucose,
            "Blood_Work_R1": blood_work_r1,
            "Blood_Pressure": blood_pressure,
            "Blood_Work_R3": blood_work_r3,
            "BMI": bmi,
            "Blood_Work_R4": blood_work_r4,
            "Patient_age": age,
            "Sepsis_Risk": round(random.uniform(0, 1), 2)
        }])
        st.session_state.patient_data_log = pd.concat([st.session_state.patient_data_log, new_entry], ignore_index=True)
        save_data(st.session_state.patient_data_log)
        st.success("Patient data submitted successfully!")

# ---------------------- Monitoring Dashboard ----------------------
elif menu == "Monitoring Dashboard":
    st.header("Monitoring Dashboard")
    df = st.session_state.patient_data_log
    patient_options = ["All Patients"] + df["Patient_ID"].unique().tolist()
    selected_patient = st.selectbox("Select Patient", patient_options)
    
    if selected_patient != "All Patients":
        df = df[df["Patient_ID"] == selected_patient]
    
    st.subheader("Sepsis Risk Trend")
    fig = px.line(df, x="Timestamp", y="Sepsis_Risk", color="Patient_ID", markers=True, title="Sepsis Risk Trend")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Patient Data Logs")
    st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)

# ---------------------- Model Insights ----------------------
elif menu == "Model Insights":
    st.header("Model Insights")
    df = st.session_state.patient_data_log[["Plasma_glucose", "Blood_Work_R1", "Blood_Pressure", "Blood_Work_R3", "BMI", "Blood_Work_R4", "Patient_age"]]
    X_scaled = scaler.transform(df)
    explainer = shap.Explainer(gb_model)
    shap_values = explainer(X_scaled)
    
    st.subheader("SHAP Summary Plot")
    fig = plt.figure()
    shap.summary_plot(shap_values, df, show=False)
    st.pyplot(fig)
    plt.close(fig)
