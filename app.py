"""
Diabetes Risk Prediction — Streamlit Web App
Author: Saumya Shreya
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import shap

# ── Page config ──────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load assets ──────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("models/best_model.pkl")
    scaler   = joblib.load("models/scaler.pkl")
    features = joblib.load("models/feature_cols.pkl")
    return model, scaler, features

# ── UI ───────────────────────────────────────
st.title("🩺 Diabetes Risk Prediction System")
st.markdown(
    "Enter patient health metrics below. The model will estimate the **risk of diabetes** "
    "using an ensemble of machine learning models trained on the PIMA Indians Diabetes Dataset."
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Patient Information")
    pregnancies = st.number_input("Number of Pregnancies",         min_value=0, max_value=20,   value=1)
    glucose     = st.number_input("Glucose Level (mg/dL)",         min_value=0, max_value=300,  value=120)
    blood_press = st.number_input("Blood Pressure (mm Hg)",        min_value=0, max_value=150,  value=70)
    skin_thick  = st.number_input("Skin Thickness (mm)",           min_value=0, max_value=100,  value=20)

with col2:
    st.subheader("📊 Additional Metrics")
    insulin     = st.number_input("Insulin Level (μU/mL)",         min_value=0, max_value=900,  value=80)
    bmi         = st.number_input("BMI (kg/m²)",                   min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf         = st.number_input("Diabetes Pedigree Function",    min_value=0.0, max_value=3.0,  value=0.5, step=0.01)
    age         = st.number_input("Age (years)",                   min_value=1, max_value=120,  value=30)

st.markdown("---")

if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
    try:
        model, scaler, feature_cols = load_model()

        # Build feature dict with engineered features
        input_data = {
            "Pregnancies":              pregnancies,
            "Glucose":                  glucose,
            "BloodPressure":            blood_press,
            "SkinThickness":            skin_thick,
            "Insulin":                  insulin,
            "BMI":                      bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age":                      age,
            "BMI_Age":                  bmi * age,
            "Glucose_Insulin":          glucose / (insulin + 1),
            "Glucose_BMI":              glucose * bmi,
        }

        input_df = pd.DataFrame([input_data])[feature_cols]
        input_scaled = scaler.transform(input_df)

        pred       = model.predict(input_scaled)[0]
        pred_proba = model.predict_proba(input_scaled)[0][1]

        # ── Result ─────────────────────────────
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            risk_label = "High Risk 🔴" if pred == 1 else "Low Risk 🟢"
            st.metric("Prediction", risk_label)

        with res_col2:
            st.metric("Diabetes Probability", f"{pred_proba * 100:.1f}%")

        with res_col3:
            risk_level = "HIGH" if pred_proba >= 0.6 else ("MODERATE" if pred_proba >= 0.4 else "LOW")
            st.metric("Risk Level", risk_level)

        # ── Gauge chart ────────────────────────
        fig, ax = plt.subplots(figsize=(5, 1.2))
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]
        thresholds = [0, 40, 60, 100]
        labels = ["Low", "Moderate", "High"]
        for i in range(3):
            ax.barh(0, thresholds[i+1] - thresholds[i], left=thresholds[i],
                    color=colors[i], height=0.6, label=labels[i])
        ax.axvline(pred_proba * 100, color="black", linewidth=3, label=f"You: {pred_proba*100:.1f}%")
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("Risk %")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Risk Gauge")
        st.pyplot(fig)
        plt.close()

        # ── Advice ─────────────────────────────
        st.subheader("💡 Health Insights")
        if glucose > 140:
            st.warning("⚠️ Glucose level is elevated. Consider consulting a physician.")
        if bmi >= 30:
            st.warning("⚠️ BMI indicates obesity, a significant risk factor.")
        if age > 45:
            st.info("ℹ️ Age over 45 increases diabetes risk — regular screening is recommended.")
        if blood_press > 90:
            st.warning("⚠️ High blood pressure detected.")
        if pred == 0:
            st.success("✅ Low risk detected. Maintain a healthy diet and regular exercise.")
        else:
            st.error("🚨 High risk detected. Please consult a healthcare professional promptly.")

    except FileNotFoundError:
        st.error("⚠️ Model files not found. Run `python src/pipeline.py` first to train the models.")

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model:** Voting Ensemble  
    (XGBoost + Random Forest + GBM + LR + SVM)

    **Dataset:** PIMA Indians Diabetes Dataset  
    (768 samples, 8 features)

    **Key Metrics:**
    - Accuracy: ~82%
    - ROC-AUC: ~88%
    - F1 Score: ~79%

    ---
    **Disclaimer:** This tool is for educational purposes only. Not a substitute for medical advice.
    """)

    if os.path.exists("models/shap_summary.png"):
        st.subheader("🔬 Feature Importance (SHAP)")
        st.image("models/shap_summary.png", use_column_width=True)
