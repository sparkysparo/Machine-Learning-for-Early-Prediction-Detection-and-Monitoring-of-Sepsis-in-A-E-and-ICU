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

# Suppress specific warnings
warnings.filterwarnings("ignore", message="In the future `np.bool` will be defined as the corresponding NumPy scalar.")
warnings.filterwarnings("ignore", message="The `use_column_width` parameter has been deprecated.*")
warnings.filterwarnings("ignore", message="Serialization of dataframe to Arrow table was unsuccessful.*")

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

# ---------------------- Caching for Model & Scaler ----------------------
@st.cache_resource
def load_model_and_scaler():
    gb_model = joblib.load("Sepsis_gb_model.pkl")
    scaler = joblib.load("sepsis_scaler.pkl")
    return gb_model, scaler

gb_model, scaler = load_model_and_scaler()

# Extract expected feature names from scaler
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
        "Blood_Pressure": random.randint(40, 300),
        "Blood_Work_R3": random.randint(10, 250),
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

# ---------------------- Application Navigation ----------------------
tabs = st.tabs(["Home", "Patient Entry", "Monitoring Dashboard", "Model Insights"])
# ---------------------- Tab 0: Home ----------------------
with tabs[0]:
    # Get the background image as a base64 string.
    img_path = "sepsis.jpg"
    if os.path.exists(img_path):
        img_base64 = get_img_with_base64(img_path)
    else:
        img_base64 = "https://via.placeholder.com/1000x400?text=Image+Not+Found"
    
    # Use CSS to set the entire background of the Home tab with a dark overlay.
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
    /* Overlay to darken the background image for text readability */
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
                Navigate through the tabs to input data, view patient trends, and explore model insights.
             </p>
         </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("""
        **Navigation:**
        - **Patient Entry:** Add new patient data or update existing records.
        - **Monitoring Dashboard:** View trends and logs of patient data.
        - **Model Insights:** Understand model predictions through SHAP explanations.
        
        Use the sidebar to simulate automatic data submissions and switch between Light and Dark themes.
    """)
    st.markdown("<hr>", unsafe_allow_html=True)
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
