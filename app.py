import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------- LOAD ----------------
model   = joblib.load("model.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

# ---------------- HEADER ----------------
st.title("👥 Employee Attrition Prediction System")
st.markdown("Fill in the employee details on the left and click **Predict** to see the results.")
st.markdown("---")

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("🧾 Employee Details")

job_satisfaction = st.sidebar.slider(
    "Job Satisfaction", 1, 4, 3,
    help="1 = Low  |  2 = Medium  |  3 = High  |  4 = Very High")

overtime = st.sidebar.selectbox("OverTime", ["No", "Yes"])

years_company = st.sidebar.slider("Years at Company", 0, 40, 5)

department = st.sidebar.selectbox(
    "Department", ["Sales", "Research & Development", "Human Resources"],
    help="Used for display only — model uses role-level features.")

# ---------------- BUILD FULL FEATURE VECTOR ----------------
input_dict = {
    "OverTime":               1 if overtime == "Yes" else 0,
    "YearsWithCurrManager":   4,
    "MonthlyIncome":          5000,
    "MaritalStatus":          1,
    "DistanceFromHome":       7,
    "JobRole":                0,
    "YearsInCurrentRole":     3,
    "JobLevel":               2,
    "TotalWorkingYears":      10,
    "EnvironmentSatisfaction":3,
    "YearsAtCompany":         years_company,
    "Age":                    36,
    "StockOptionLevel":       1,
    "JobInvolvement":         3,
    "JobSatisfaction":        job_satisfaction,
}

input_df = pd.DataFrame([input_dict])[features]

# ---------------- PREDICT BUTTON ----------------
if st.sidebar.button("🔍 Predict Attrition Risk", use_container_width=True):

    prob_leave = model.predict_proba(input_df)[0][1]
    prob_stay  = model.predict_proba(input_df)[0][0]
    pred       = model.predict(input_df)[0]

    # ── Row 1: Verdict + Confidence Gauge ──────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Prediction Result")
        if pred == 1:
            st.error("### ⚠️ Likely to LEAVE")
        else:
            st.success("### ✅ Likely to STAY")

        st.markdown(f"""
| | |
|---|---|
| **Department** | {department} |
| **Job Satisfaction** | {job_satisfaction} / 4 |
| **OverTime** | {overtime} |
| **Years at Company** | {years_company} |
        """)

    with col2:
        st.subheader("🎯 Confidence Score")

        fig_gauge, ax = plt.subplots(figsize=(4, 2.5),
                                     subplot_kw=dict(aspect="equal"))
        fig_gauge.patch.set_alpha(0)

        for t1, t2, color in [
            (np.pi, np.pi*2/3, "#2ecc71"),
            (np.pi*2/3, np.pi/3, "#f39c12"),
            (np.pi/3, 0, "#e74c3c")
        ]:
            xs = np.cos(np.linspace(t1, t2, 100))
            ys = np.sin(np.linspace(t1, t2, 100))
            ax.fill_between(xs, ys, xs*0.6, color=color, alpha=0.85)
            ax.fill_betweenx(ys*0.6, xs*0.6, xs, color=color, alpha=0.0)

        angle = np.pi * (1 - prob_leave)
        ax.annotate("", xy=(0.55 * np.cos(angle), 0.55 * np.sin(angle)),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2.5))
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.2, 1.1)
        ax.axis("off")
        ax.text(0, -0.15, f"{prob_leave*100:.1f}%", ha="center",
                fontsize=18, fontweight="bold",
                color="#e74c3c" if prob_leave > 0.5 else "#2ecc71")
        ax.text(-1.0, -0.15, "Low", fontsize=8, color="#2ecc71")
        ax.text(0.75, -0.15, "High", fontsize=8, color="#e74c3c")
        ax.set_title("Attrition Risk", fontsize=11, pad=4)
        st.pyplot(fig_gauge, use_container_width=True)

    st.markdown("---")

    # ── Row 2: Class Probabilities + SHAP ──────────────────────────────
    col3, col4 = st.columns([1, 1])

    with col3:
        st.subheader("📊 Class Probabilities")

        fig_prob, ax2 = plt.subplots(figsize=(5, 3))
        fig_prob.patch.set_alpha(0)
        bars = ax2.barh(
            ["Will Stay", "Will Leave"],
            [prob_stay, prob_leave],
            color=["#2ecc71", "#e74c3c"],
            edgecolor="none", height=0.5
        )
        for bar, val in zip(bars, [prob_stay, prob_leave]):
            ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val*100:.1f}%", va="center", fontsize=13,
                     fontweight="bold")
        ax2.set_xlim(0, 1.15)
        ax2.set_xlabel("Probability", fontsize=10)
        ax2.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax2.spines[["top", "right", "left"]].set_visible(False)
        ax2.tick_params(axis="y", labelsize=12)
        ax2.set_facecolor("none")
        st.pyplot(fig_prob, use_container_width=True)

    with col4:
        st.subheader("🔬 SHAP — Why This Prediction?")
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        fig_shap, ax3 = plt.subplots(figsize=(5, 4))
        shap.summary_plot(shap_values, input_df, show=False, plot_size=None)
        st.pyplot(fig_shap, use_container_width=True)

    st.markdown("---")

    # ── Row 3: Global Feature Importance ───────────────────────────────
    st.subheader("📈 Global Feature Importance (Model-Level)")

    importance = pd.Series(model.feature_importances_, index=features) \
                   .sort_values(ascending=True)

    user_features = ["OverTime", "JobSatisfaction", "YearsAtCompany"]
    colors = ["#e74c3c" if f in user_features else "#5b9bd5"
              for f in importance.index]

    fig_fi, ax4 = plt.subplots(figsize=(8, 5))
    fig_fi.patch.set_alpha(0)
    bars = ax4.barh(importance.index, importance.values,
                    color=colors, edgecolor="none", height=0.6)
    for bar, val in zip(bars, importance.values):
        ax4.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9)

    ax4.set_xlabel("Importance Score", fontsize=11)
    ax4.set_title("Feature Importance — XGBoost Model", fontsize=13, pad=10)
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.set_facecolor("none")
    red_patch  = mpatches.Patch(color="#e74c3c", label="Your selected features")
    blue_patch = mpatches.Patch(color="#5b9bd5", label="Background features")
    ax4.legend(handles=[red_patch, blue_patch], fontsize=9, loc="lower right")
    st.pyplot(fig_fi, use_container_width=True)

    st.markdown("---")

    # ── Risk Summary Metrics ────────────────────────────────────────────
    st.subheader("💡 Risk Interpretation")
    risk_level = ("🔴 High"   if prob_leave > 0.65 else
                  "🟡 Medium" if prob_leave > 0.35 else
                  "🟢 Low")

    c1, c2, c3 = st.columns(3)
    c1.metric("Attrition Risk Level",    risk_level)
    c2.metric("Probability of Leaving",  f"{prob_leave*100:.1f}%")
    c3.metric("Probability of Staying",  f"{prob_stay*100:.1f}%")

    if prob_leave > 0.5:
        st.warning("""
**Suggested Retention Actions:**
- 🕐 Review and reduce overtime workload
- 💬 Conduct a 1-on-1 engagement conversation
- 📈 Discuss career growth and promotion path
- 💰 Benchmark and review compensation
        """)
    else:
        st.info("""
**Employee appears stable. To maintain retention:**
- 🌟 Recognise and reward achievements regularly
- 🔄 Continue providing growth opportunities
- 📊 Monitor satisfaction scores periodically
        """)

else:
    st.info("👈 Set employee details in the sidebar and click **Predict Attrition Risk** to begin.")