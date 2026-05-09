import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ---------------- LOAD ----------------
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("👥 Employee Attrition Prediction System")

st.sidebar.header("Employee Details")

# ---------------- INPUTS (ONLY NUMERIC / SAFE FEATURES) ----------------
age = st.sidebar.slider("Age", 18, 60, 30)
distance = st.sidebar.slider("Distance From Home", 1, 30, 5)
monthly_income = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000)
job_level = st.sidebar.slider("Job Level", 1, 5, 2)
total_years = st.sidebar.slider("Total Working Years", 0, 40, 10)
years_company = st.sidebar.slider("Years at Company", 0, 40, 5)
job_satisfaction = st.sidebar.slider("Job Satisfaction", 1, 4, 3)
work_life_balance = st.sidebar.slider("Work Life Balance", 1, 4, 3)
env_satisfaction = st.sidebar.slider("Environment Satisfaction", 1, 4, 3)
job_involvement = st.sidebar.slider("Job Involvement", 1, 4, 3)

overtime = st.sidebar.selectbox("OverTime (0 = No, 1 = Yes)", [0, 1])

# ---------------- BUILD INPUT ----------------
input_dict = {col: 0 for col in features}

input_dict.update({
    "Age": age,
    "DistanceFromHome": distance,
    "MonthlyIncome": monthly_income,
    "JobLevel": job_level,
    "TotalWorkingYears": total_years,
    "YearsAtCompany": years_company,
    "JobSatisfaction": job_satisfaction,
    "WorkLifeBalance": work_life_balance,
    "EnvironmentSatisfaction": env_satisfaction,
    "JobInvolvement": job_involvement,
    "OverTime": overtime
})

input_df = pd.DataFrame([input_dict])

# ensure correct column order
input_df = input_df[features]

# ---------------- PREDICTION ----------------
if st.button("Predict Attrition Risk"):

    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.error(f"⚠️ Employee likely to LEAVE (Risk: {prob:.2f})")
    else:
        st.success(f"✅ Employee likely to STAY (Risk: {prob:.2f})")

    # ---------------- SHAP ----------------
    st.subheader("🔬 Why this prediction?")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_df, show=False)
    st.pyplot(fig)
