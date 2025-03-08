
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load model & scaler
gb_model_path = "Sepsis_gb_model.pkl"
scaler_path = "sepsis_scaler.pkl"
gb_model = joblib.load(gb_model_path)
scaler = joblib.load(scaler_path)

st.title(" ICU Sepsis Monitoring System (Gradient Boosting Model)")
st.sidebar.header("🩺 Patient Data Input")
st.sidebar.write("Enter patient vitals for real-time sepsis risk prediction.")

if "patient_data_log" not in st.session_state:
    st.session_state.patient_data_log = pd.DataFrame(columns=[
        "Timestamp", "Patient_ID", "Plasma_glucose", "Blood_Work_R1",
        "Blood_Pressure", "Blood_Work_R3", "BMI", "Blood_Work_R4",
        "Patient_age", "Sepsis_Risk"
    ])

patients = st.session_state.patient_data_log["Patient_ID"].unique().tolist()
selected_patient = st.sidebar.selectbox("Select Patient (New or Existing)", ["New Patient"] + list(patients))

PRG = st.sidebar.slider("PRG (Plasma Glucose)", 0, 200, 100)
PL = st.sidebar.slider("PL (Blood Work R1)", 50, 180, 120)
PR = st.sidebar.slider("PR (Blood Pressure)", 40, 200, 80)
SK = st.sidebar.slider("SK (Blood Work R3)", 0, 100, 30)
M11 = st.sidebar.slider("M11 (BMI)", 10.0, 50.0, 25.0)
BD2 = st.sidebar.slider("BD2 (Blood Work R4)", 0.0, 3.0, 0.5)
Age = st.sidebar.slider("Age", 18, 100, 40)

patient_data = pd.DataFrame(
    [[time.strftime("%H:%M:%S"), PRG, PL, PR, SK, M11, BD2, Age]],
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

scaled_data = scaler.transform(patient_data.drop(columns=["Timestamp"]))
sepsis_risk = gb_model.predict_proba(scaled_data)[0][1]

if sepsis_risk < 0.3:
    risk_level = "🟢 LOW RISK"
elif sepsis_risk < 0.7:
    risk_level = "🟡 MEDIUM RISK"
else:
    risk_level = "🔴 HIGH RISK - ALERT ICU 🚨"

if selected_patient == "New Patient":
    patient_id = len(patients) + 1
else:
    patient_id = selected_patient

patient_entry = pd.DataFrame([[
    time.strftime("%H:%M:%S"), patient_id, PRG, PL, PR, SK, M11, BD2, Age, sepsis_risk
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
        [st.session_state.patient_data_log, patient_entry], ignore_index=True
    )

st.subheader(" Sepsis Risk Prediction")
st.metric(label="Sepsis Risk Score", value=f"{sepsis_risk:.2f}")
st.write(f"Risk Category: {risk_level}")

st.subheader(" Patient Health Trends")
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(patient_data.columns[1:], patient_data.iloc[0, 1:])
ax.set_ylabel("Vital Measurements")
ax.set_title("Patient Vitals Overview")
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader(" Sepsis Risk Progression Over Time")
fig2, ax2 = plt.subplots(figsize=(10, 5))
for p_id in st.session_state.patient_data_log["Patient_ID"].unique():
    subset = st.session_state.patient_data_log[
        st.session_state.patient_data_log["Patient_ID"] == p_id
    ]
    ax2.plot(subset["Timestamp"], subset["Sepsis_Risk"], marker='o', linestyle='-', label=f"Patient {p_id}")
ax2.set_ylabel("Sepsis Risk Score")
ax2.set_xlabel("Time (Entries)")
ax2.set_title("Sepsis Risk Trend")
ax2.axhline(y=0.3, linestyle='--', label="Low Risk Threshold")
ax2.axhline(y=0.7, linestyle='--', label="High Risk Threshold")
ax2.legend()
st.pyplot(fig2)

st.subheader(" ICU Insights")
st.write("""
- Patients with high blood pressure & high glucose levels have increased sepsis risk.
- Monitoring BMI & blood work results helps detect early warning signs.
- This dashboard dynamically updates to monitor ICU patients in real-time.
""")

st.subheader(" Patient Data Log")
st.dataframe(st.session_state.patient_data_log)
