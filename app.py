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

# ---------------------- Caching for Model & Scaler ----------------------
@st.cache_resource
def load_model_and_scaler():
    gb_model = joblib.load("Sepsis_gb_model.pkl")
    scaler = joblib.load("sepsis_scaler.pkl")
    return gb_model, scaler

gb_model, scaler = load_model_and_scaler()

# ---------------------- Ensure Feature Order Before Transformation ----------------------
expected_features = [
    "Plasma_glucose", "Blood_Work_R1", "Blood_Pressure",
    "Blood_Work_R3", "BMI", "Blood_Work_R4", "Patient_age"
]

# ---------------------- Data Processing ----------------------
if "patient_data_log" not in st.session_state:
    st.session_state.patient_data_log = pd.DataFrame(columns=expected_features + ["Sepsis_Risk"])

def validate_feature_order(df):
    if list(df.columns) != expected_features:
        print("Feature mismatch detected!")
        print("Expected:", expected_features)
        print("Actual:", list(df.columns))
        df = df[expected_features]
    return df

# ---------------------- Scaling Data ----------------------
def transform_data(X):
    X = validate_feature_order(X)
    return scaler.transform(X)

# ---------------------- Model Predictions ----------------------
def predict_sepsis_risk(X):
    X_scaled = transform_data(X)
    return gb_model.predict_proba(X_scaled)[:, 1]

# ---------------------- Tab 3: Model Insights ----------------------
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
    X_train = st.session_state.patient_data_log[expected_features]
X_train = validate_feature_order(X_train)
X_train_scaled = transform_data(X_train)

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
