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
import shap
from sklearn.preprocessing import StandardScaler
import random

# Suppress warnings
warnings.filterwarnings("ignore", message="In the future `np.bool` will be defined as the corresponding NumPy scalar.")
warnings.filterwarnings("ignore", message="The `use_column_width` parameter has been deprecated.*")
warnings.filterwarnings("ignore", message="Serialization of dataframe to Arrow table was unsuccessful.*")

# Try importing st_autorefresh; if unavailable, define a dummy function
try:
    from streamlit_autorefresh import st_autorefresh
except ModuleNotFoundError:
    st_autorefresh = lambda **kwargs: 0  
    st.warning("streamlit-autorefresh module not found. Auto-refresh simulation will be disabled.")

# ---------------------- Page Configuration ----------------------
st.set_page_config(page_title="ICU Sepsis Monitoring", layout="wide")

# ---------------------- Caching for Model & Scaler ----------------------
@st.cache_resource
def load_model_and_scaler():
    gb_model = joblib.load("Sepsis_gb_model.pkl")
    scaler = joblib.load("sepsis_scaler.pkl")
    return gb_model, scaler

gb_model, scaler = load_model_and_scaler()

# Extract expected feature names from the trained scaler
expected_columns = list(scaler.feature_names_in_)

# ---------------------- Data Persistence Setup ----------------------
DATA_FILE = "patient_data_log.csv"
if "patient_data_log" not in st.session_state:
    if os.path.exists(DATA_FILE):
        st.session_state.patient_data_log = pd.read_csv(DATA_FILE)
    else:
        st.session_state.patient_data_log = pd.DataFrame(columns=[
            "Timestamp", "Patient_ID", "Patient_Name"] + expected_columns + ["Sepsis_Risk"]
        )

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

# ---------------------- Application Navigation ----------------------
tabs = st.tabs(["Home", "Patient Entry", "Monitoring Dashboard", "Model Insights"])

# ---------------------- Tab 1: Patient Entry ----------------------
with tabs[1]:
    st.header("Patient Data Entry")
    with st.form(key="patient_entry_form", clear_on_submit=True):
        patient_mode = st.selectbox("Select Mode", ["New Patient", "Monitor Existing Patient"], key="entry_mode")

        if patient_mode == "New Patient":
            patient_id_input = st.text_input("Enter Patient ID")
            patient_name_input = st.text_input("Enter Patient Name")
        else:
            existing_patients = st.session_state.patient_data_log[["Patient_ID", "Patient_Name"]].drop_duplicates()
            options = [f"{row['Patient_ID']} - {row['Patient_Name']}" for _, row in existing_patients.iterrows()]
            selected_patient_option = st.selectbox("Select Patient", options)
            selected_patient_id = selected_patient_option.split(" - ")[0]

        st.subheader("Enter Vital Signs")
        data_dict = {col: st.slider(col, 0, 300, 100) for col in expected_columns}
        submit_button = st.form_submit_button("Submit Data")

    if submit_button:
        data_dict["Timestamp"] = time.strftime("%H:%M:%S")
        if patient_mode == "New Patient":
            data_dict["Patient_ID"] = patient_id_input.strip()
            data_dict["Patient_Name"] = patient_name_input.strip()
        else:
            data_dict["Patient_ID"] = selected_patient_id
            data_dict["Patient_Name"] = existing_patients.loc[
                existing_patients["Patient_ID"] == selected_patient_id, "Patient_Name"
            ].iloc[-1]

        input_df = pd.DataFrame([data_dict], columns=expected_columns)
        scaled_data = pd.DataFrame(scaler.transform(input_df), columns=expected_columns)
        data_dict["Sepsis_Risk"] = gb_model.predict_proba(scaled_data)[0][1]

        new_entry = pd.DataFrame([data_dict])
        st.session_state.patient_data_log = pd.concat([st.session_state.patient_data_log, new_entry], ignore_index=True)
        save_data(st.session_state.patient_data_log)

# ---------------------- Tab 2: Monitoring Dashboard ----------------------
with tabs[2]:
    st.header("Monitoring Dashboard")

    if st.session_state.patient_data_log.empty:
        st.info("No patient data available yet.")
    else:
        # Select a specific patient to monitor
        patient_options = st.session_state.patient_data_log["Patient_ID"].unique().tolist()
        selected_patient_id = st.selectbox("Select Patient to Monitor", ["All Patients"] + patient_options)

        # Filter data based on selection
        if selected_patient_id != "All Patients":
            df = st.session_state.patient_data_log[st.session_state.patient_data_log["Patient_ID"] == selected_patient_id]
        else:
            df = st.session_state.patient_data_log

        show_risk_line = st.checkbox("Show Sepsis Risk Trend", value=True)
        if show_risk_line:
            st.subheader("Sepsis Risk Trend Over Time")
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
        st.dataframe(df)

# ---------------------- Tab 3: Model Insights ----------------------
with tabs[3]:
    st.header("Model Insights")
    
    if not st.session_state.patient_data_log.empty:
        X_train = st.session_state.patient_data_log[expected_columns]
        X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=expected_columns)

        explainer = shap.Explainer(gb_model)
        shap_values = explainer(X_train_scaled)

        st.write("### SHAP Summary Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_train, show=False)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No data available for SHAP analysis.")
