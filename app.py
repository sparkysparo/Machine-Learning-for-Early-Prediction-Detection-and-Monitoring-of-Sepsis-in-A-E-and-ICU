import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool

import streamlit as st
import pandas as pd
import joblib
import time
import os
import matplotlib.pyplot as plt
import plotly.express as px
import shap  # Ensure shap is installed (pip install shap)
from sklearn.preprocessing import StandardScaler

# ---------------------- Page Configuration ----------------------
st.set_page_config(page_title="ICU Sepsis Monitoring", layout="wide")

# ---------------------- Theme Toggle CSS ----------------------
# You can customize these CSS settings to suit your preferences.
theme_choice = st.sidebar.radio("Select Theme", ["Light", "Dark"])
if theme_choice == "Dark":
    st.markdown("""
        <style>
        /* Apply background and text color for the entire app */
        body, .stApp {
            background-color: #2C2C2C;
            color: #F0F0F0;
        }
        /* Header, paragraphs, and other text elements */
        h1, h2, h3, h4, h5, h6, p, label, .stMetric {
            color: #F0F0F0;
        }
        /* Sidebar styling */
        .css-1d391kg, .css-1d391kg .block-container {
            background-color: #1F1F1F;
            color: #F0F0F0;
        }
        /* Container styling for blocks */
        .block-container {
            background-color: #2C2C2C;
        }
        /* Button styling */
        .stButton>button {
            background-color: #444444;
            color: #F0F0F0;
        }
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #FFFFFF;
            color: #333333;
        }
        h1, h2, h3, h4, h5, h6, p, label, .stMetric {
            color: #333333;
        }
        .css-1d391kg, .css-1d391kg .block-container {
            background-color: #f0f2f6;
            color: #333333;
        }
        .block-container {
            background-color: #FFFFFF;
        }
        .stButton>button {
            background-color: #e0e0e0;
            color: #333333;
        }
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

# ---------------------- Application Tabs ----------------------
tab1, tab2, tab3 = st.tabs(["Patient Entry", "Monitoring Dashboard", "Model Insights"])

# ---------------------- Tab 1: Patient Entry ----------------------
with tab1:
    st.header("Patient Data Entry")
    
    # Patient mode: New entry or updating an existing patient
    patient_mode = st.selectbox("Select Mode", ["New Patient", "Monitor Existing Patient"])
    
    if patient_mode == "New Patient":
        patient_id_input = st.text_input("Enter Patient ID", help="Unique identifier for the patient")
        patient_name_input = st.text_input("Enter Patient Name", help="Full name of the patient")
    else:
        if not st.session_state.patient_data_log.empty:
            existing_patients = st.session_state.patient_data_log[["Patient_ID", "Patient_Name"]].drop_duplicates()
            options = [f"{row['Patient_ID']} - {row['Patient_Name']}" for index, row in existing_patients.iterrows()]
            selected_patient_option = st.selectbox("Select Patient", options)
            selected_patient_id = selected_patient_option.split(" - ")[0]
        else:
            st.info("No existing patient data available. Switch to 'New Patient' mode.")
    
    st.subheader("Enter Vital Signs")
    PRG = st.slider("PRG (Plasma Glucose)", 0, 600, 100, help="Glucose level in plasma (mg/dL)")
    PL = st.slider("PL (Blood Work R1)", 50, 500, 120, help="Result from blood work test R1")
    PR = st.slider("PR (Blood Pressure)", 40, 300, 80, help="Blood pressure reading (mm Hg)")
    SK = st.slider("SK (Blood Work R3)", 0, 300, 30, help="Result from blood work test R3")
    M11 = st.slider("M11 (BMI)", 10.0, 70.0, 25.0, help="Body Mass Index (kg/m²)")
    BD2 = st.slider("BD2 (Blood Work R4)", 0.0, 7.0, 0.5, help="Result from blood work test R4")
    Age = st.slider("Age", 18, 110, 40, help="Patient age in years (max capped at 110)")
    
    if st.button("Submit Data"):
        current_time = time.strftime("%H:%M:%S")
        data_dict = {
            "Timestamp": current_time,
            "Plasma_glucose": PRG,
            "Blood_Work_R1": PL,
            "Blood_Pressure": PR,
            "Blood_Work_R3": SK,
            "BMI": M11,
            "Blood_Work_R4": BD2,
            "Patient_age": Age
        }
        if patient_mode == "New Patient":
            if patient_id_input.strip() == "" or patient_name_input.strip() == "":
                st.error("Please enter both Patient ID and Patient Name for a new patient.")
            else:
                data_dict["Patient_ID"] = patient_id_input.strip()
                data_dict["Patient_Name"] = patient_name_input.strip()
        else:
            if st.session_state.patient_data_log.empty:
                st.error("No existing patient data available to monitor.")
            else:
                data_dict["Patient_ID"] = selected_patient_id
                existing_name = st.session_state.patient_data_log.loc[
                    st.session_state.patient_data_log["Patient_ID"] == selected_patient_id, "Patient_Name"
                ].iloc[-1]
                data_dict["Patient_Name"] = existing_name
        
        input_df = pd.DataFrame([[
            data_dict["Plasma_glucose"],
            data_dict["Blood_Work_R1"],
            data_dict["Blood_Pressure"],
            data_dict["Blood_Work_R3"],
            data_dict["BMI"],
            data_dict["Blood_Work_R4"],
            data_dict["Patient_age"]
        ]], columns=["Plasma_glucose", "Blood_Work_R1", "Blood_Pressure", 
                     "Blood_Work_R3", "BMI", "Blood_Work_R4", "Patient_age"])
        
        scaled_data = scaler.transform(input_df)
        sepsis_risk = gb_model.predict_proba(scaled_data)[0][1]
        data_dict["Sepsis_Risk"] = sepsis_risk
        
        new_entry = pd.DataFrame([data_dict])
        st.session_state.patient_data_log = pd.concat(
            [st.session_state.patient_data_log, new_entry], ignore_index=True
        )
        save_data(st.session_state.patient_data_log)
        
        st.subheader("Sepsis Risk Prediction")
        st.metric(label="Sepsis Risk Score", value=f"{sepsis_risk:.2f}")
        if sepsis_risk < 0.3:
            risk_level = "🟢 LOW RISK"
        elif sepsis_risk < 0.7:
            risk_level = "🟡 MEDIUM RISK"
        else:
            risk_level = "🔴 HIGH RISK - ALERT ICU 🚨"
            st.error("High sepsis risk detected! Immediate intervention required.")
        st.write(f"Risk Category: {risk_level}")
        
        vitals = {
            "Vital": ["Plasma_glucose", "Blood_Work_R1", "Blood_Pressure", "Blood_Work_R3", "BMI", "Blood_Work_R4", "Patient_age"],
            "Value": [PRG, PL, PR, SK, M11, BD2, Age]
        }
        fig_vitals = px.bar(pd.DataFrame(vitals), x="Vital", y="Value", title="Current Patient Vitals")
        st.plotly_chart(fig_vitals, use_container_width=True)

# ---------------------- Tab 2: Monitoring Dashboard ----------------------
with tab2:
    st.header("Monitoring Dashboard")
    if st.session_state.patient_data_log.empty:
        st.info("No patient data available yet.")
    else:
        show_risk_line = st.checkbox("Show Sepsis Risk Trend", value=True)
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
        st.dataframe(st.session_state.patient_data_log)

# ---------------------- Tab 3: Model Insights (with SHAP) ----------------------
with tab3:
    st.header("Model Insights")
    st.write("Generating SHAP feature importance for: **Gradient Boosting Model**")
    
    # Use all seven features to match scaler's fit
    if st.session_state.patient_data_log.empty:
        st.info("No patient data available for SHAP analysis. Using a dummy sample.")
        X_train = pd.DataFrame({
            "Plasma_glucose": np.linspace(100, 150, 10),
            "Blood_Work_R1": np.linspace(120, 160, 10),
            "Blood_Work_Pressure": np.linspace(80, 120, 10),
            "Blood_Work_R3": np.linspace(30, 50, 10),
            "BMI": np.linspace(25, 30, 10),
            "Blood_Work_R4": np.linspace(0.5, 1.0, 10),
            "Patient_age": np.linspace(40, 60, 10)
        })
    else:
        X_train = st.session_state.patient_data_log[[
            "Plasma_glucose", "Blood_Work_R1", "Blood_Work_Pressure", 
            "Blood_Work_R3", "BMI", "Blood_Work_R4", "Patient_age"
        ]]
    X_train_scaled = scaler.transform(X_train)
    
    explainer = shap.Explainer(gb_model)
    shap_values = explainer(X_train_scaled)
    
    st.write("### SHAP Summary Plot")
    fig = plt.figure(figsize=(10, 6))
    try:
        # Disable the colorbar to avoid extra graph elements
        shap.summary_plot(shap_values, X_train, show=False, color_bar=False)
        for ax in fig.axes:
            if hasattr(ax, 'images') and len(ax.images) > 0:
                ax.images = []
    except ValueError as e:
        st.error("Error generating SHAP summary plot: " + str(e))
    st.pyplot(fig)
    plt.close(fig)
    
    with st.expander("About SHAP Feature Importance"):
        st.write("""
        SHAP (SHapley Additive exPlanations) assigns each feature an importance value for a particular prediction.
        - Features at the top of the plot have the highest impact on the model output.
        - This visualization helps in understanding how each vital sign contributes to the sepsis risk prediction.
        """)
