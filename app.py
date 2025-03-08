import streamlit as st
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt

# Load model & scaler
gb_model_path = "Sepsis_gb_model.pkl"
scaler_path = "sepsis_scaler.pkl"
gb_model = joblib.load(gb_model_path)
scaler = joblib.load(scaler_path)

st.title("ICU Sepsis Monitoring System (Gradient Boosting Model)")
st.sidebar.header("🩺 Patient Data Input")
st.sidebar.write("Enter patient vitals for real-time sepsis risk prediction.")

# Initialize patient log with additional column for Patient_Name
if "patient_data_log" not in st.session_state:
    st.session_state.patient_data_log = pd.DataFrame(columns=[
        "Timestamp", "Patient_ID", "Patient_Name", "Plasma_glucose", "Blood_Work_R1",
        "Blood_Pressure", "Blood_Work_R3", "BMI", "Blood_Work_R4",
        "Patient_age", "Sepsis_Risk"
    ])

# Patient mode selection: New or Monitor Existing
patient_mode = st.sidebar.selectbox("Select Mode", ["New Patient", "Monitor Existing Patient"])

if patient_mode == "New Patient":
    patient_id_input = st.sidebar.text_input("Enter Patient ID")
    patient_name_input = st.sidebar.text_input("Enter Patient Name")
else:
    if not st.session_state.patient_data_log.empty:
        # Create a combined list for clarity
        existing_patients = st.session_state.patient_data_log[["Patient_ID", "Patient_Name"]].drop_duplicates()
        options = [f"{row['Patient_ID']} - {row['Patient_Name']}" for index, row in existing_patients.iterrows()]
        selected_patient_option = st.sidebar.selectbox("Select Patient", options)
        # Extract patient ID from the selection
        selected_patient_id = selected_patient_option.split(" - ")[0]
    else:
        st.sidebar.write("No existing patient data available.")

# Vitals input via sliders (Increased Maximum Values)
PRG = st.sidebar.slider("PRG (Plasma Glucose)", 0, 500, 100)       # increased max from 300 to 500
PL = st.sidebar.slider("PL (Blood Work R1)", 50, 350, 120)           # increased max from 250 to 350
PR = st.sidebar.slider("PR (Blood Pressure)", 40, 300, 80)           # increased max from 250 to 300
SK = st.sidebar.slider("SK (Blood Work R3)", 0, 200, 30)             # increased max from 150 to 200
M11 = st.sidebar.slider("M11 (BMI)", 10.0, 70.0, 25.0)               # increased max from 60.0 to 70.0
BD2 = st.sidebar.slider("BD2 (Blood Work R4)", 0.0, 7.0, 0.5)         # increased max from 5.0 to 7.0
Age = st.sidebar.slider("Age", 18, 110, 40)                          # Age max remains at 110

# Submit button: Process input when clicked
if st.sidebar.button("Submit Data"):
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
        # Monitor an existing patient
        if st.session_state.patient_data_log.empty:
            st.error("No existing patient data available to monitor.")
        else:
            data_dict["Patient_ID"] = selected_patient_id
            # Get the latest name for the selected patient
            existing_name = st.session_state.patient_data_log.loc[
                st.session_state.patient_data_log["Patient_ID"] == selected_patient_id, "Patient_Name"
            ].iloc[-1]
            data_dict["Patient_Name"] = existing_name

    # Prepare the input for prediction (scaler expects only vital columns)
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
    
    # Append new entry to the patient log
    new_entry = pd.DataFrame([data_dict])
    st.session_state.patient_data_log = pd.concat(
        [st.session_state.patient_data_log, new_entry], ignore_index=True
    )
    
    # Display prediction results
    st.subheader("Sepsis Risk Prediction")
    st.metric(label="Sepsis Risk Score", value=f"{sepsis_risk:.2f}")
    if sepsis_risk < 0.3:
        risk_level = "🟢 LOW RISK"
    elif sepsis_risk < 0.7:
        risk_level = "🟡 MEDIUM RISK"
    else:
        risk_level = "🔴 HIGH RISK - ALERT ICU 🚨"
    st.write(f"Risk Category: {risk_level}")
    
    # Patient vitals overview (bar chart)
    st.subheader("Patient Vitals Overview")
    fig, ax = plt.subplots(figsize=(10, 5))
    vitals = ["Plasma_glucose", "Blood_Work_R1", "Blood_Pressure", 
              "Blood_Work_R3", "BMI", "Blood_Work_R4", "Patient_age"]
    ax.bar(vitals, input_df.iloc[0])
    ax.set_ylabel("Vital Measurements")
    ax.set_title("Current Patient Vitals")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Sepsis risk progression over time
    st.subheader("Sepsis Risk Progression Over Time")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for pid in st.session_state.patient_data_log["Patient_ID"].unique():
        subset = st.session_state.patient_data_log[st.session_state.patient_data_log["Patient_ID"] == pid]
        ax2.plot(subset["Timestamp"], subset["Sepsis_Risk"], marker='o', linestyle='-', label=f"Patient {pid}")
    ax2.set_ylabel("Sepsis Risk Score")
    ax2.set_xlabel("Timestamp")
    ax2.set_title("Sepsis Risk Trend")
    ax2.axhline(y=0.3, linestyle='--', label="Low Risk Threshold")
    ax2.axhline(y=0.7, linestyle='--', label="High Risk Threshold")
    ax2.legend()
    st.pyplot(fig2)

st.subheader("ICU Insights")
st.write("""
- Patients with high blood pressure & high glucose levels have increased sepsis risk.
- Monitoring BMI & blood work results helps detect early warning signs.
- This dashboard dynamically updates to monitor ICU patients in real-time.
""")

st.subheader("Patient Data Log")
st.dataframe(st.session_state.patient_data_log)

