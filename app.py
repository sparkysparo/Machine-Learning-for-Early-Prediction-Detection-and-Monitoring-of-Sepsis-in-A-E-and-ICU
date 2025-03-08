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
gb_model_path = "Sepsis_gb_model.pkl"
scaler_path = "sepsis_scaler.pkl"

try:
    gb_model = joblib.load(gb_model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# -------------------------------
# Prediction Function (Adapted from FastAPI Code)
# -------------------------------
def predict_sepsis_risk(df):
    """
    Takes a DataFrame with numeric vital sign data,
    scales the data, and returns a dictionary with the prediction result.
    """
    scaled_df = scaler.transform(df)
    prediction = gb_model.predict_proba(scaled_df)
    # Use the predicted probability for sepsis (class 1)
    risk_prob = prediction[0][1]
    
    # Determine label based on a threshold
    if risk_prob >= 0.7:
        label = "Patient has sepsis"
    elif risk_prob >= 0.3:
        label = "Patient might have sepsis"
    else:
        label = "Patient does not have sepsis"
    
    return {
        "prediction": label,
        "probability": f"{risk_prob:.2f}"
    }

# -------------------------------
# App Title & Introduction
# -------------------------------
st.title("ICU Sepsis Monitoring System")
st.markdown("""
This dashboard is designed for clinical use to monitor and predict sepsis risk in ICU patients.
Enter the patient’s details and vital signs, then click **Submit Patient Data** to record the entry.
""")
st.markdown("---")

# -------------------------------
# Sidebar: Patient Information & Vital Signs Input
# -------------------------------
st.sidebar.header("Patient Information")
patient_id = st.sidebar.text_input("Patient ID", placeholder="e.g., 12345")
patient_name = st.sidebar.text_input("Patient Name", placeholder="e.g., John Doe")

st.sidebar.markdown("---")
st.sidebar.header("Vital Signs Input")
# Adjusted slider ranges and steps:
PRG = st.sidebar.slider("Plasma Glucose (PRG)", 50, 300, 120, step=5)
PL = st.sidebar.slider("Blood Work R1 (PL)", 50, 250, 120, step=5)
PR = st.sidebar.slider("Blood Pressure (PR)", 80, 200, 120, step=5)
SK = st.sidebar.slider("Blood Work R3 (SK)", 0, 100, 30, step=1)
M11 = st.sidebar.slider("BMI (M11)", 10.0, 50.0, 25.0, step=0.5)
BD2 = st.sidebar.slider("Blood Work R4 (BD2)", 0.0, 5.0, 1.0, step=0.1)
Age = st.sidebar.slider("Patient Age", 18, 100, 40, step=1)

submit_btn = st.sidebar.button("Submit Patient Data")

# -------------------------------
# Initialize Session State for Patient Data Log
# -------------------------------
if "patient_data_log" not in st.session_state:
    st.session_state.patient_data_log = pd.DataFrame(columns=[
        "Timestamp", "Patient_ID", "Patient_Name", "Plasma_glucose", "Blood_Work_R1",
        "Blood_Pressure", "Blood_Work_R3", "BMI", "Blood_Work_R4", "Patient_age", "Sepsis_Risk"
    ])

# -------------------------------
# Process Input When Submit Button is Clicked
# -------------------------------
if submit_btn:
    # Validate input: require Patient ID and Patient Name.
    if patient_id.strip() == "":
        st.error("Please enter the Patient ID.")
        st.stop()
    if patient_name.strip() == "":
        st.error("Please enter the Patient Name.")
        st.stop()

    current_time = time.strftime("%H:%M:%S")
    # Create a DataFrame for the current patient input
    patient_data = pd.DataFrame(
        [[current_time, patient_id.strip(), patient_name.strip(), PRG, PL, PR, SK, M11, BD2, Age]],
        columns=["Timestamp", "Patient_ID", "Patient_Name", "Plasma_glucose", "Blood_Work_R1",
                 "Blood_Pressure", "Blood_Work_R3", "BMI", "Blood_Work_R4", "Patient_age"]
    )
    
    # Use the predict_sepsis_risk function on the numeric data
    scaled_input = patient_data.drop(columns=["Timestamp", "Patient_ID", "Patient_Name"])
    prediction_output = predict_sepsis_risk(scaled_input)
    sepsis_risk = float(prediction_output["probability"])

    # Append prediction to the DataFrame
    patient_data["Sepsis_Risk"] = sepsis_risk

    # Update the patient data log: if Patient ID exists, update; otherwise, append new entry.
    if patient_id.strip() in st.session_state.patient_data_log["Patient_ID"].astype(str).values:
        st.session_state.patient_data_log.loc[
            st.session_state.patient_data_log["Patient_ID"] == patient_id.strip()
        ] = patient_data.values
    else:
        st.session_state.patient_data_log = pd.concat(
            [st.session_state.patient_data_log, patient_data],
            ignore_index=True
        )

    # Display sepsis risk result with clinical alert
    st.subheader("Sepsis Risk Prediction")
    st.write(f"Prediction: **{prediction_output['prediction']}**")
    st.write(f"Probability: **{prediction_output['probability']}**")

# -------------------------------
# Main Dashboard: Data Log & Visualizations
# -------------------------------
st.markdown("---")
st.subheader("Patient Data Log")
st.dataframe(st.session_state.patient_data_log)

# Interactive Bar Chart: Display latest patient’s vitals
if not st.session_state.patient_data_log.empty:
    last_record = st.session_state.patient_data_log.iloc[-1]
    vitals = ["Plasma_glucose", "Blood_Work_R1", "Blood_Pressure", "Blood_Work_R3", "BMI", "Blood_Work_R4", "Patient_age"]
    values = [last_record[v] for v in vitals]
    df_vitals = pd.DataFrame({"Vital": vitals, "Value": values})
    fig_bar = px.bar(df_vitals, x="Vital", y="Value", text="Value", title="Latest Patient Vitals")
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar)

# Interactive Line Chart: Sepsis Risk Progression Over Time (curved)
if not st.session_state.patient_data_log.empty:
    df_line = st.session_state.patient_data_log.copy()
    # Convert Timestamp (HH:MM:SS) to a datetime object for plotting purposes.
    df_line["Timestamp_dt"] = pd.to_datetime(df_line["Timestamp"], format="%H:%M:%S", errors='coerce')
    fig_line = px.line(
        df_line, x="Timestamp_dt", y="Sepsis_Risk", color="Patient_ID", markers=True,
        title="Sepsis Risk Progression Over Time", line_shape="spline"
    )
    fig_line.add_hline(y=0.3, line_dash="dash", annotation_text="Low Risk Threshold", annotation_position="bottom left")
    fig_line.add_hline(y=0.7, line_dash="dash", annotation_text="High Risk Threshold", annotation_position="top left")
    st.plotly_chart(fig_line)

st.markdown("---")
st.subheader("Clinical Insights")
st.write("""
- **Patient Identification:** Ensure that the patient’s ID and name are entered correctly.
- **Risk Monitoring:** High sepsis risk patients should be prioritized for immediate clinical intervention.
- **Data Logging:** The patient data log captures all entries for trend analysis over time.
""")
