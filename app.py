import streamlit as st  # Ensure streamlit is imported first

# Set page configuration (THIS MUST BE THE FIRST STREAMLIT COMMAND)
st.set_page_config(page_title="ICU Sepsis Monitoring", layout="wide")

# Now import everything else
import warnings
import numpy as np
import base64
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
warnings.filterwarnings("ignore")

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

# Apply Theme
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {app_bg};
        color: {text_color};
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
    </style>
    """, unsafe_allow_html=True)

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

# ---------------------- Data Persistence ----------------------
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

# ---------------------- Tab 0: Home ----------------------
with tabs[0]:
    st.header(" ICU Sepsis Monitoring System")

    st.markdown(f"""
    <div style="background-color: {app_bg}; padding: 20px; border-radius: 10px;">
        <h1 style="color: {text_color};">ICU Sepsis Monitoring</h1>
        <h3 style="color: {text_color};">Real-time Patient Monitoring & Insights</h3>
        <p style="font-size: 1.2em; color: {text_color};">
            This system leverages a Gradient Boosting model to predict sepsis risk in ICU patients.  
            Navigate through the tabs to input data, monitor patient trends, and explore model insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------- Tab 3: Model Insights ----------------------
with tabs[3]:
    st.header("Model Insights")

    if not st.session_state.patient_data_log.empty:
        X_train = st.session_state.patient_data_log[expected_columns]
        X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=expected_columns)

        explainer = shap.Explainer(gb_model)
        shap_values = explainer(X_train_scaled)

        # SHAP Summary Plot
        st.write("### SHAP Summary Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_train, show=False)
        st.pyplot(fig)
        plt.close(fig)

        # Extract Feature Importance
        feature_importance = np.abs(shap_values.values).mean(axis=0)
        importance_df = pd.DataFrame({
            "Feature": expected_columns,
            "Importance": feature_importance
        }).sort_values(by="Importance", ascending=False)

        # Get top 3 most important features
        top_features = importance_df.head(3)["Feature"].tolist()

        with st.expander("About SHAP Feature Importance"):
            st.write("""
            **SHAP (SHapley Additive exPlanations) assigns each feature an importance value for a particular prediction.**  
            - Features at the top of the plot have the highest impact on the model output.  
            - This visualization helps in understanding how each vital sign contributes to sepsis risk prediction.  
            """)

            if top_features:
                st.write("###  **Key Insights from the Model:**")
                st.write(f" **{top_features[0]}** has the highest influence on predicting sepsis risk.")
                if len(top_features) > 1:
                    st.write(f" **{top_features[1]}** is the second most important feature.")
                if len(top_features) > 2:
                    st.write(f" **{top_features[2]}** also plays a significant role.")

    else:
        st.info("No data available for SHAP analysis.")
