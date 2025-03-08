import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Load Model & Scaler
# -------------------------------
# Ensure these files are in the same folder as app.py (or adjust paths accordingly)
gb_model_path = "Sepsis_gb_model.pkl"
scaler_path = "sepsis_scaler.pkl"
try:
    gb_model = joblib.load(gb_model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# -------------------------------
# App Title & Description
# -------------------------------
st.title("🏥 ICU Sepsis Monitoring System")
st.markdown(
    """
    This dashboard provides real-time sepsis risk prediction and monitoring. 
    Enter patient vitals in the sidebar and review dynamic alerts and interactive visualizations.
    """
)

# -------------------------------
# Sidebar: Patient Data Input
# -------------------------------
st.sidebar.header("🩺 Patient Data Input")
st.sidebar.write("Enter patient vitals for real-time sepsis risk prediction.")

# Optional: Auto-refresh the dashboard every 60 seconds
#st.experimental_autorefresh(interval=60000, limit=100, key="data_refresh")

# -------------------------------
# Session State: Initialize Data Log
# -------------------------------
if "patient_data_log" not in st.session_state:
    st.session_state.patient_data_log = pd.DataFrame(columns=[
        "Timestamp", "Patient_ID", "Plasma_glucose", "Blood_Work_R1",
        "Blood_Pressure", "Blood_Work_R3", "BMI", "Blood_Work_R4",
        "Patient_age", "Sepsis_Risk"
    ])

# -------------------------------
# Patient Selection
# -------------------------------
patients = st.session_state.patient_data_log["Patient_ID"].unique().tolist()
selected_patient = st.sidebar.selectbox(
    "Select Patient (New or Existing)",
    ["New Patient"] + list(patients)
)

# -------------------------------
# Patient Input Sliders
# -------------------------------
PRG = st.sidebar.slider("PRG (Plasma Glucose)", 0, 200, 100)
PL = st.sidebar.slider("PL (Blood Work R1)", 50, 180, 120)
PR = st.sidebar.slider("PR (Blood Pressure)", 40, 200, 80)
SK = st.sidebar.slider("SK (Blood Work R3)", 0, 100, 30)
M11 = st.sidebar.slider("M11 (BMI)", 10.0, 50.0, 25.0)
BD2 = st.sidebar.slider("BD2 (Blood Work R4)", 0.0, 3.0, 0.5)
Age = st.sidebar.slider("Age", 18, 100, 40)

# -------------------------------
# Create Patient DataFrame
# -------------------------------
current_time = time.strftime("%H:%M:%S")
patient_data = pd.DataFrame(
    [[current_time, PRG, PL, PR, SK, M11, BD2, Age]],
    columns=["Timestamp", "PRG", "PL", "PR", "SK", "M11", "BD2", "Age"]
)
patient_data.rename(columns={
    "PRG": "Plasma_glucose",
    "PL": "Blood_Work_R1",
    "PR": "Blood_Pressure",
    "SK": "Blood_Work_R3",
    "M11": "BMI",
    "BD2": "Blood_Work_R4",
    "Age": "Patient_age"
}, inplace=True)

# -------------------------------
# Normalize Input & Predict Sepsis Risk
# -------------------------------
scaled_data = scaler.transform(patient_data.drop(columns=["Timestamp"]))
sepsis_risk = gb_model.predict_proba(scaled_data)[0][1]

# -------------------------------
# Determine Risk Category & Display Clinical Alert
# -------------------------------
if sepsis_risk >= 0.7:
    risk_level = "🔴 HIGH RISK - ALERT ICU 🚨"
    st.error(f"High Sepsis Risk: {sepsis_risk:.2f}")
elif sepsis_risk >= 0.3:
    risk_level = "🟡 MEDIUM RISK"
    st.warning(f"Moderate Sepsis Risk: {sepsis_risk:.2f}")
else:
    risk_level = "🟢 LOW RISK"
    st.success(f"Low Sepsis Risk: {sepsis_risk:.2f}")

# -------------------------------
# Assign or Update Patient ID
# -------------------------------
if selected_patient == "New Patient":
    patient_id = len(patients) + 1
else:
    patient_id = selected_patient

# -------------------------------
# Update Session State: Patient Data Log
# -------------------------------
patient_entry = pd.DataFrame([[
    current_time, patient_id, PRG, PL, PR, SK, M11, BD2, Age, sepsis_risk
]], columns=[
    "Timestamp", "Patient_ID", "Plasma_glucose", "Blood_Work_R1",
    "Blood_Pressure", "Blood_Work_R3", "BMI", "Blood_Work_R4",
    "Patient_age", "Sepsis_Risk"
])
if patient_id in patients:
    st.session_state.patient_data_log.loc[
        st.session_state.patient_data_log["Patient_ID"] == patient_id
    ] = patient_entry.values
else:
    st.session_state.patient_data_log = pd.concat(
        [st.session_state.patient_data_log, patient_entry],
        ignore_index=True
    )

# -------------------------------
# Interactive Bar Chart: Patient Vitals (Using Plotly)
# -------------------------------
vitals = patient_data.columns[1:]  # Exclude "Timestamp"
values = patient_data.iloc[0, 1:]
df_vitals = pd.DataFrame({"Vital": vitals, "Value": values})
fig_bar = px.bar(df_vitals, x="Vital", y="Value", text="Value",
                 title="Patient Vitals Overview")
fig_bar.update_traces(textposition="outside")
st.plotly_chart(fig_bar)

# -------------------------------
# Interactive Line Chart: Sepsis Risk Progression
# -------------------------------
if not st.session_state.patient_data_log.empty:
    # Convert Timestamp (HH:MM:SS) to a datetime for plotting purposes.
    df_line = st.session_state.patient_data_log.copy()
    df_line["Timestamp_dt"] = pd.to_datetime(df_line["Timestamp"], format="%H:%M:%S")
    fig_line = px.line(
        df_line, x="Timestamp_dt", y="Sepsis_Risk", color="Patient_ID",
        title="Sepsis Risk Progression Over Time", markers=True
    )
    # Add threshold lines for clinical alerts
    fig_line.add_hline(y=0.3, line_dash="dash", annotation_text="Low Risk Threshold", annotation_position="bottom left")
    fig_line.add_hline(y=0.7, line_dash="dash", annotation_text="High Risk Threshold", annotation_position="top left")
    st.plotly_chart(fig_line)

# -------------------------------
# Patient Data Log & Filtering Options
# -------------------------------
st.sidebar.subheader("Filter Data Log")
patient_filter = st.sidebar.multiselect(
    "Select Patient ID(s)",
    options=st.session_state.patient_data_log["Patient_ID"].unique()
)
if patient_filter:
    filtered_log = st.session_state.patient_data_log[
        st.session_state.patient_data_log["Patient_ID"].isin(patient_filter)
    ]
else:
    filtered_log = st.session_state.patient_data_log

st.subheader("📋 Patient Data Log")
st.dataframe(filtered_log)

# -------------------------------
# Refresh Button for Manual Update
# -------------------------------
if st.button("Refresh Data"):
    st.experimental_rerun()

# -------------------------------
# Clinical Insights Section
# -------------------------------
st.subheader("💡 Clinical Insights")
st.write(
    """
    - **High Risk Alert:** Patients with high blood pressure and high plasma glucose levels have an increased sepsis risk.
    - **Trend Monitoring:** Regular review of the vitals and sepsis risk progression is essential for early intervention.
    - **Data Logging:** The patient data log records all entries. Use the filtering option to review specific patients.
    """
)
