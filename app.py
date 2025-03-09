import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", message="In the future `np.bool` will be defined as the corresponding NumPy scalar.")
warnings.filterwarnings("ignore", message="The `use_column_width` parameter has been deprecated.*")
warnings.filterwarnings("ignore", message="Serialization of dataframe to Arrow table was unsuccessful.*")

import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool

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
    st_autorefresh = lambda **kwargs: 0  # dummy function returning 0 refresh count
    st.warning("streamlit-autorefresh module not found. Auto-refresh simulation will be disabled.")

# ---------------------- Custom CSS for Button & Theme ----------------------
st.set_page_config(page_title="ICU Sepsis Monitoring", layout="wide")

# Custom CSS for the submit button
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50 !important;  /* A bright green for visibility */
        color: white !important;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049 !important;
    }
    </style>
    """, unsafe_allow_html=True)

theme_choice = st.sidebar.radio("Select Theme", ["Light", "Dark"])
if theme_choice == "Dark":
    sidebar_bg = "#2A2A3D"
    app_bg = "#1E1E2F"
    text_color = "#FFFFFF"
    btn_bg = "#3C3C55"
    btn_hover = "#57578A"
    metric_bg = "#2A2A3D"
else:
    sidebar_bg = "#FFFFFF"
    app_bg = "#F7F7F7"
    text_color = "#333333"
    btn_bg = "#E0E0E0"
    btn_hover = "#CCCCCC"
    metric_bg = "#F0F0F0"

st.markdown(f"""
    <style>
    .stApp {{
        background-color: {app_bg};
        color: {text_color};
        font-family: 'Segoe UI', sans-serif;
    }}
    h1, h2, h3, h4, h5, h6, p, label {{
        color: {text_color};
    }}
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg};
    }}
    [data-testid="stSidebar"] * {{
        color: {text_color};
    }}
    .block-container {{
        background-color: {app_bg};
    }}
    .stButton>button {{
        background-color: {btn_bg} !important;
        color: {text_color} !important;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }}
    .stButton>button:hover {{
        background-color: {btn_hover} !important;
    }}
    .stMetric {{
        background-color: {metric_bg};
        border: 1px solid #CCCCCC;
        border-radius: 8px;
    }}
    .main .element-container {{
        background-color: {app_bg};
    }}
    </style>
    """, unsafe_allow_html=True)

# ---------------------- Utility Functions ----------------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_img_with_base64(file_path):
    img_base64 = get_base64_of_bin_file(file_path)
    return f"data:image/jpeg;base64,{img_base64}"

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

# ---------------------- Sidebar Navigation ----------------------
# Instead of using tabs, we use a sidebar radio button to navigate
page = st.sidebar.radio("Navigation", ["Home", "Patient Entry", "Monitoring Dashboard", "Model Insights"])

# ====================== Home Page ======================
if page == "Home":
    img_path = "sepsis.jpg"
    if os.path.exists(img_path):
        img_base64 = get_img_with_base64(img_path)
    else:
        img_base64 = "https://via.placeholder.com/1000x400?text=Image+Not+Found"
    
    st.markdown(f"""
    <style>
    .home-page {{
         background: url('{img_base64}') no-repeat center center fixed;
         background-size: cover;
         padding: 100px 50px;
         min-height: 600px;
         border-radius: 10px;
         position: relative;
         z-index: 1;
    }}
    .home-page::before {{
         content: "";
         position: absolute;
         top: 0;
         left: 0;
         right: 0;
         bottom: 0;
         background: rgba(0, 0, 0, 0.5);
         border-radius: 10px;
         z-index: -1;
    }}
    .home-page-text {{
         position: relative;
         z-index: 2;
         text-align: center;
         color: #FFFFFF;
    }}
    </style>
    <div class="home-page">
         <div class="home-page-text">
             <h1 style="font-size: 3.5em; margin-bottom: 0; color: white;">ICU Sepsis Monitoring System</h1>
             <h3 style="font-weight: normal; margin-top: 0; color: white;">Real-time Monitoring & Insights</h3>
             <p style="font-size: 1.2em; margin-top: 20px; color: white;">
                Welcome to our advanced monitoring system that leverages a Gradient Boosting model to assess sepsis risk in ICU patients.
                Navigate through the sidebar to input data, view patient trends, and explore model insights.
             </p>
         </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("""
        **Navigation:**
        - **Patient Entry:** Add new patient data or update existing records.
        - **Monitoring Dashboard:** View trends and logs of patient data with filtering options.
        - **Model Insights:** Understand model predictions through SHAP explanations.
        
        Use the sidebar to simulate automatic data submissions and switch between Light and Dark themes.
    """)
    st.markdown("<hr>", unsafe_allow_html=True)

# ====================== Patient Entry Page ======================
elif page == "Patient Entry":
    st.header("Patient Data Entry")
    with st.form(key="patient_entry_form", clear_on_submit=True):
        patient_mode = st.selectbox("Select Mode", ["New Patient", "Monitor Existing Patient"], key="entry_mode")
        
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
        
        submit_button = st.form_submit_button("Submit Data")
    
    if submit_button:
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

# ====================== Monitoring Dashboard Page ======================
elif page == "Monitoring Dashboard":
    st.header("Monitoring Dashboard")
    if st.session_state.patient_data_log.empty:
        st.info("No patient data available yet.")
    else:
        # Add a selectbox to filter by patient
        patient_ids = sorted(st.session_state.patient_data_log["Patient_ID"].unique())
        selected_patient = st.selectbox("Filter by Patient (select 'All' to view every record)", ["All"] + patient_ids)
        
        if selected_patient != "All":
            df = st.session_state.patient_data_log[st.session_state.patient_data_log["Patient_ID"] == selected_patient]
        else:
            df = st.session_state.patient_data_log.copy()
        
        # Sepsis risk trend line chart
        show_risk_line = st.checkbox("Show Sepsis Risk Trend", value=True, key="toggle_trend")
        if show_risk_line:
            st.subheader("Sepsis Risk Trend Over Time" + (f" for {selected_patient}" if selected_patient != "All" else ""))
            if selected_patient == "All":
                fig_trend = px.line(df, x="Timestamp", y="Sepsis_Risk", color="Patient_ID", markers=True,
                                     title="Sepsis Risk Progression Over Time")
            else:
                fig_trend = px.line(df, x="Timestamp", y="Sepsis_Risk", markers=True,
                                     title=f"Sepsis Risk Progression for Patient {selected_patient}")
            st.plotly_chart(fig_trend, use_container_width=True)
        
        # If a specific patient is selected, show a bar chart for their latest vitals
        if selected_patient != "All":
            st.subheader(f"Latest Vital Signs for Patient {selected_patient}")
            latest_record = df.iloc[-1]
            vitals = {
                "Vital": ["Plasma_glucose", "Blood_Work_R1", "Blood_Pressure", "Blood_Work_R3", "BMI", "Blood_Work_R4", "Patient_age"],
                "Value": [latest_record["Plasma_glucose"], latest_record["Blood_Work_R1"],
                          latest_record["Blood_Pressure"], latest_record["Blood_Work_R3"],
                          latest_record["BMI"], latest_record["Blood_Work_R4"], latest_record["Patient_age"]]
            }
            fig_vitals = px.bar(pd.DataFrame(vitals), x="Vital", y="Value", title="Current Patient Vitals")
            st.plotly_chart(fig_vitals, use_container_width=True)
        
        st.subheader("Patient Data Log")
        st.dataframe(df)

# ====================== Model Insights Page ======================
elif page == "Model Insights":
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
        X_train = st.session_state.patient_data_log[[
            "Plasma_glucose", "Blood_Work_R1", "Blood_Pressure", 
            "Blood_Work_R3", "BMI", "Blood_Work_R4", "Patient_age"
        ]]
    X_train_scaled = scaler.transform(X_train)
    
    explainer = shap.Explainer(gb_model)
    shap_values = explainer(X_train_scaled)
    
    st.write("### SHAP Summary Plot")
    fig = plt.figure(figsize=(10, 6))
    try:
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
