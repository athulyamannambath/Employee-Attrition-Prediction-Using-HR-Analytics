import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# -------------------------
# Load artifacts
# -------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

st.title("👥 Employee Attrition Prediction App")
st.write("Predict whether an employee will leave using ML + SHAP explanations")

# -------------------------
# INPUT SECTION (must match dataset logic)
# -------------------------
age = st.slider("Age", 18, 60, 30)
income = st.number_input("Monthly Income", 1000, 20000, 5000)
distance = st.slider("Distance From Home", 1, 30, 5)
job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
overtime = st.selectbox("OverTime", ["Yes", "No"])
job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)

# encode categorical
overtime_val = 1 if overtime == "Yes" else 0

# -------------------------
# BUILD INPUT ROW
# -------------------------
input_dict = {
    "Age": age,
    "MonthlyIncome": income,
    "DistanceFromHome": distance,
    "JobLevel": job_level,
    "OverTime": overtime_val,
    "JobSatisfaction": job_satisfaction
}

df = pd.DataFrame([input_dict])

# add missing columns (VERY IMPORTANT)
for col in features:
    if col not in df.columns:
        df[col] = 0

# reorder columns exactly like training
df = df[features]

# scale input
df_scaled = scaler.transform(df)

# -------------------------
# PREDICTION
# -------------------------
if st.button("Predict Attrition"):

    prob = model.predict_proba(df_scaled)[0][1]
    pred = model.predict(df_scaled)[0]

    if pred == 1:
        st.error(f"🔴 Employee likely to LEAVE (Risk: {prob:.2f})")
    else:
        st.success(f"🟢 Employee likely to STAY (Risk: {prob:.2f})")

    # -------------------------
    # SHAP EXPLANATION
    # -------------------------
    st.subheader("🔬 SHAP Explainability")

    explainer = shap.Explainer(model)
    shap_values = explainer(df_scaled)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    st.info("Red = increases attrition risk, Green = decreases risk")