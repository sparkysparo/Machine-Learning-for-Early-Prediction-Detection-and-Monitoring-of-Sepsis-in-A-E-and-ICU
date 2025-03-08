import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Load Model & Scaler (Cached for Performance)
# -------------------------------
@st.cache_resource
def load_model_and_scaler():
    gb_model_path = "Sepsis_gb_model.pkl"
    scaler_path = "sepsis_scaler.pkl"
    try:
        gb_model = joblib.load(gb_model_path)
        scaler = joblib.load(scaler_path)
        return gb_model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()

gb_model, scaler = load_model_and_scaler()

# -------------------------------
# Prediction Function
# -------------------------------
def predict_sepsis_risk(df):
    """
    Scales the input data and returns the sepsis risk probability.
    """
    scaled_df = scaler.transform(df)
    prediction = gb_model.predict_proba(scaled_df)
    return prediction[0][1]  # Probability for sepsis (class 1)

# -------------------------------
# Initialize Session State for Patient Data Log
# -------------------------------
if "patient_data_log" not in st.session_state:
    st.session_state.patient_data_log = pd.DataFrame(columns=[
        "Timestamp", "Patient_ID", "Patient_Name", "Plasma_glucose", "Blood_Work_R1",
        "Blood_Pressure", "Blood_Work_R3", "BMI", "Blood_Work_R4", "Patient_age", "Sepsis_Risk"
    ])

# -------------------------------
# App Title & Introduction
# -------------------------------
st.title("ICU Sepsis Monitoring System")
st.markdown("""
This dashboard is designed for clinical use to monitor and predict sepsis risk in ICU patients.
Choose a mode from the sidebar to either add a new patient or monitor an existing patient.
Enter the vital signs and click **Submit Patient Data** to record the entry.
""")
st.markdown("---")

# -------------------------------
# Sidebar: Mode Selection & Patient Details
# -------------------------------
st.sidebar.header("Patient Mode")
mode = st.sidebar.selectbox("Select Mode", ["Add New Patient", "Monitor Patient"])

if mode == "Add New Patient":
    patient_id = st.sidebar.text_input("Patient ID", placeholder="e.g., 12345", help="Enter the unique patient identifier.")
    patient_name = st.sidebar.text_input("Patient Name", placeholder="e.g., John Doe", help="Enter the patient's full name.")
elif mode == "Monitor Patient":
    existing_patients = st.session_state.patient_data_log["Patient_ID"].unique().tolist()
    if len(existing_patients) == 0:
        st.sidebar.warning("No existing patients found. Please add a new patient first.")
        patient_id = ""
        patient_name = ""
    else:
        selected_patient = st.sidebar.selectbox("Select Patient", existing_patients)
        # Retrieve the latest record for the selected patient
        patient_rec = st.session_state.patient_data_log[st.session_state.patient_data_log["Patient_ID"] == selected_patient].iloc[-1]
        patient_id = patient_rec["Patient_ID"]
        patient_name = patient_rec["Patient_Name"]
        st.sidebar.write(f"Monitoring Patient: **{patient_name}** (ID: {patient_id})")

st.sidebar.markdown("---")
st.sidebar.header("Vital Signs Input")
# Adjusted slider ranges and steps (clinical values)
PRG = st.sidebar.slider("Plasma Glucose (PRG)", 50, 300, 120, step=5, help="Plasma glucose level in mg/dL.")
PL = st.sidebar.slider("Blood Work R1 (PL)", 50, 250, 120, step=5, help="Blood work result 1.")
PR = st.sidebar.slider("Blood Pressure (PR)", 80, 200, 120, step=5, help="Blood pressure in mmHg.")
SK = st.sidebar.slider("Blood Work R3 (SK)", 0, 100, 30, step=1, help="Blood work result 3.")
M11 = st.sidebar.slider("BMI (M11)", 10.0, 50.0, 25.0, step=0.5, help="Body Mass Index (BMI).")
BD2 = st.sidebar.slider("Blood Work R4 (BD2)", 0.0, 5.0, 1.0, step=0.1, help="Blood work result 4.")
Age = st.sidebar.slider("Patient Age", 18, 100, 40, step=1, help="Patient's age in years.")

submit_btn = st.sidebar.button("✅ Submit Patient Data")

# -------------------------------
# Process Input on Submission
# -------------------------------
if submit_btn:
    # Validate required inputs
    if patient_id.strip() == "" or patient_name.strip() == "":
        st.error("❌ Please enter both Patient ID and Patient Name.")
        st.stop()

    current_time = time.strftime("%H:%M:%S")
    new_entry = pd.DataFrame(
        [[current_time, patient_id.strip(), patient_name.strip(), PRG, PL, PR, SK, M11, BD2, Age]],
        columns=["Timestamp", "Patient_ID", "Patient_Name", "Plasma_glucose", "Blood_Work_R1",
                 "Blood_Pressure", "Blood_Work_R3", "BMI", "Blood_Work_R4", "Patient_age"]
    )
    
    # Make prediction using only numeric columns
    numeric_input = new_entry.drop(columns=["Timestamp", "Patient_ID", "Patient_Name"])
    sepsis_risk = predict_sepsis_risk(numeric_input)
    new_entry["Sepsis_Risk"] = sepsis_risk

    # Append new record (for both modes, this is a new measurement)
    st.session_state.patient_data_log = pd.concat(
        [st.session_state.patient_data_log, new_entry],
        ignore_index=True
    )
    
    st.success("✅ Patient data submitted successfully!")
    st.subheader("📈 Sepsis Risk Prediction")
    if sepsis_risk >= 0.7:
        st.error(f"High Sepsis Risk: {sepsis_risk:.2f}")
    elif sepsis_risk >= 0.3:
        st.warning(f"Moderate Sepsis Risk: {sepsis_risk:.2f}")
    else:
        st.success(f"Low Sepsis Risk: {sepsis_risk:.2f}")

# -------------------------------
# Main Dashboard: Data Log & Visualizations
# -------------------------------
st.markdown("---")
st.subheader("📋 Patient Data Log")
if mode == "Monitor Patient" and patient_id:
    filtered_log = st.session_state.patient_data_log[st.session_state.patient_data_log["Patient_ID"] == patient_id]
    st.dataframe(filtered_log)
else:
    st.dataframe(st.session_state.patient_data_log)

# Interactive Bar Chart: Latest Patient Vitals (for monitored patient)
if mode == "Monitor Patient" and patient_id:
    filtered_log = st.session_state.patient_data_log[st.session_state.patient_data_log["Patient_ID"] == patient_id]
    if not filtered_log.empty:
        last_record = filtered_log.iloc[-1]
        vitals = ["Plasma_glucose", "Blood_Work_R1", "Blood_Pressure", "Blood_Work_R3", "BMI", "Blood_Work_R4", "Patient_age"]
        values = [last_record[v] for v in vitals]
        df_vitals = pd.DataFrame({"Vital": vitals, "Value": values})
        try:
            fig_bar = px.bar(df_vitals, x="Vital", y="Value", text="Value", title="📊 Latest Patient Vitals")
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar)
        except Exception as e:
            st.error(f"Error rendering bar chart: {e}")
    else:
        st.warning("No data found for the selected patient.")

# Interactive Line Graph: Average Sepsis Risk Over Time (All Patients)
if not st.session_state.patient_data_log.empty:
    df_all = st.session_state.patient_data_log.copy()
    # Convert Timestamp (HH:MM:SS) to datetime for plotting purposes
    df_all["Timestamp_dt"] = pd.to_datetime(df_all["Timestamp"], format="%H:%M:%S", errors="coerce")
    df_all = df_all.sort_values(by="Timestamp_dt")
    # Group by Timestamp and compute average sepsis risk
    df_grouped = df_all.groupby("Timestamp_dt", as_index=False)["Sepsis_Risk"].mean()
    
    try:
        fig_line = px.line(
            df_grouped, x="Timestamp_dt", y="Sepsis_Risk",
            title="📈 Average Sepsis Risk Over Time",
            markers=True, line_shape="spline", template="plotly_white"
        )
        fig_line.update_layout(
            xaxis_title="Time",
            yaxis_title="Average Sepsis Risk",
            hovermode="x unified",
            font=dict(size=12)
        )
        st.plotly_chart(fig_line)
    except Exception as e:
        st.error(f"Error rendering line graph: {e}")

st.markdown("---")
st.subheader("💡 Clinical Insights")
st.write("""
- **Patient Identification:** Verify that the patient’s ID and name are entered correctly.
- **Risk Monitoring:** High sepsis risk patients require immediate clinical intervention.
- **Data Logging:** Use the patient data log to analyze trends over time.
""")
