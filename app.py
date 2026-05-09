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

# ---------------- 4 INPUTS ONLY ----------------
job_satisfaction = st.sidebar.slider("Job Satisfaction", 1, 4, 3,
    help="1 = Low, 2 = Medium, 3 = High, 4 = Very High")

overtime = st.sidebar.selectbox("OverTime", ["No", "Yes"])

years_company = st.sidebar.slider("Years at Company", 0, 40, 5)

department = st.sidebar.selectbox("Department",
    ["Sales", "Research & Development", "Human Resources"])

# ---------------- BUILD INPUT (all 15 features, defaults for the rest) ----------------
input_dict = {
    "OverTime":               1 if overtime == "Yes" else 0,
    "YearsWithCurrManager":   4,    # average default
    "MonthlyIncome":          5000, # average default
    "MaritalStatus":          1,    # 0=Divorced,1=Married,2=Single (median)
    "DistanceFromHome":       7,    # average default
    "JobRole":                0,    # encoded default
    "YearsInCurrentRole":     3,    # average default
    "JobLevel":               2,    # average default
    "TotalWorkingYears":      10,   # average default
    "EnvironmentSatisfaction":3,    # average default
    "YearsAtCompany":         years_company,
    "Age":                    36,   # average default
    "StockOptionLevel":       1,    # average default
    "JobInvolvement":         3,    # average default
    "JobSatisfaction":        job_satisfaction,
}

input_df = pd.DataFrame([input_dict])[features]  # correct column order

# ---------------- PREDICT ----------------
if st.button("Predict Attrition Risk"):

    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.markdown("---")
    if pred == 1:
        st.error(f"⚠️ Employee likely to **LEAVE**  —  Risk Score: `{prob:.2f}`")
    else:
        st.success(f"✅ Employee likely to **STAY**  —  Risk Score: `{prob:.2f}`")

    # ---------------- SHAP ----------------
    st.subheader("🔬 Why this prediction?")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, input_df, show=False)
    st.pyplot(fig)
    