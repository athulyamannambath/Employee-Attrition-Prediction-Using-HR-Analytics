import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------- GLOBAL CHART STYLE (white text for dark theme) ----------------
plt.rcParams.update({
    "text.color":        "white",
    "axes.labelcolor":   "white",
    "xtick.color":       "white",
    "ytick.color":       "white",
    "axes.edgecolor":    "white",
    "figure.facecolor":  "none",
    "axes.facecolor":    "none",
    "axes.titlecolor":   "white",
    "legend.facecolor":  "#1e1e2e",
    "legend.edgecolor":  "white",
    "legend.labelcolor": "white",
})

# ---------------- LOAD ----------------
model    = joblib.load("model.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

st.title("👥 Employee Attrition Prediction System")
st.markdown("Fill in the employee details on the left and click **Predict** to see the results.")
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🧾 Employee Details")

job_satisfaction = st.sidebar.slider(
    "Job Satisfaction", 1, 4, 3,
    help="1 = Low  |  2 = Medium  |  3 = High  |  4 = Very High")

overtime = st.sidebar.selectbox("OverTime", ["No", "Yes"])

years_company = st.sidebar.slider("Years at Company", 0, 40, 5)

department = st.sidebar.selectbox(
    "Department", ["Sales", "Research & Development", "Human Resources"])

# ---------------- FEATURE VECTOR ----------------
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

# ---------------- PREDICT ----------------
if st.sidebar.button("🔍 Predict Attrition Risk", use_container_width=True):

    prob_leave = model.predict_proba(input_df)[0][1]
    prob_stay  = model.predict_proba(input_df)[0][0]
    pred       = model.predict(input_df)[0]

    # ── Row 1: Verdict + Gauge ──────────────────────────────────────────
    col1, col2 = st.columns(2)

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
        st.subheader("🎯 Confidence Gauge")

        fig_g, ax_g = plt.subplots(figsize=(4, 2.8),
                                   subplot_kw=dict(aspect="equal"))
        for t1, t2, c in [
            (np.pi, np.pi*2/3, "#2ecc71"),
            (np.pi*2/3, np.pi/3, "#f39c12"),
            (np.pi/3, 0,        "#e74c3c"),
        ]:
            xs = np.cos(np.linspace(t1, t2, 100))
            ys = np.sin(np.linspace(t1, t2, 100))
            xs2 = 0.6 * xs;  ys2 = 0.6 * ys
            ax_g.fill(
                np.concatenate([xs, xs2[::-1]]),
                np.concatenate([ys, ys2[::-1]]),
                color=c, alpha=0.9
            )
        angle = np.pi * (1 - prob_leave)
        ax_g.annotate("", xy=(0.52*np.cos(angle), 0.52*np.sin(angle)),
                      xytext=(0, 0),
                      arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5))
        ax_g.set_xlim(-1.15, 1.15);  ax_g.set_ylim(-0.25, 1.15)
        ax_g.axis("off")
        ax_g.text(0, -0.18, f"{prob_leave*100:.1f}%", ha="center",
                  fontsize=20, fontweight="bold", color="white")
        ax_g.text(-1.05, -0.18, "Low",  fontsize=9, color="#2ecc71")
        ax_g.text(0.78,  -0.18, "High", fontsize=9, color="#e74c3c")
        ax_g.set_title("Attrition Risk", fontsize=12, pad=6, color="white")
        st.pyplot(fig_g, use_container_width=True)

    st.markdown("---")

    # ── Row 2: Class Probabilities + Feature Importance ────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📊 Class Probabilities")

        fig_p, ax_p = plt.subplots(figsize=(5, 3))
        labels = ["Will Stay", "Will Leave"]
        values = [prob_stay, prob_leave]
        colors = ["#2ecc71", "#e74c3c"]
        bars = ax_p.barh(labels, values, color=colors, edgecolor="none", height=0.45)

        for bar, val in zip(bars, values):
            ax_p.text(
                min(val + 0.02, 1.0),
                bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%",
                va="center", fontsize=13, fontweight="bold", color="white"
            )

        ax_p.set_xlim(0, 1.2)
        ax_p.set_xlabel("Probability", fontsize=10, color="white")
        ax_p.axvline(0.5, color="white", linestyle="--", linewidth=1, alpha=0.4)
        ax_p.spines[["top", "right", "left"]].set_visible(False)
        ax_p.tick_params(axis="y", labelsize=13, colors="white")
        ax_p.tick_params(axis="x", colors="white")
        st.pyplot(fig_p, use_container_width=True)

    with col4:
        st.subheader("📈 Feature Importance")

        importance = pd.Series(model.feature_importances_, index=features) \
                       .sort_values(ascending=True)
        user_feats = ["OverTime", "JobSatisfaction", "YearsAtCompany"]
        bar_colors = ["#e74c3c" if f in user_feats else "#5b9bd5"
                      for f in importance.index]

        fig_fi, ax_fi = plt.subplots(figsize=(5, 5))
        bars_fi = ax_fi.barh(importance.index, importance.values,
                             color=bar_colors, edgecolor="none", height=0.6)

        for bar, val in zip(bars_fi, importance.values):
            ax_fi.text(val + 0.002,
                       bar.get_y() + bar.get_height() / 2,
                       f"{val:.3f}", va="center", fontsize=8, color="white")

        ax_fi.set_xlabel("Importance Score", fontsize=10, color="white")
        ax_fi.set_title("XGBoost Feature Importance", fontsize=11,
                        pad=8, color="white")
        ax_fi.spines[["top", "right"]].set_visible(False)
        ax_fi.tick_params(colors="white", labelsize=9)

        red_p  = mpatches.Patch(color="#e74c3c", label="Your input features")
        blue_p = mpatches.Patch(color="#5b9bd5", label="Background features")
        ax_fi.legend(handles=[red_p, blue_p], fontsize=8, loc="lower right")
        st.pyplot(fig_fi, use_container_width=True)

    st.markdown("---")

    # ── Risk Metrics + Suggestions ──────────────────────────────────────
    st.subheader("💡 Risk Interpretation")
    risk_level = ("🔴 High"   if prob_leave > 0.65 else
                  "🟡 Medium" if prob_leave > 0.35 else
                  "🟢 Low")

    c1, c2, c3 = st.columns(3)
    c1.metric("Attrition Risk Level",   risk_level)
    c2.metric("Probability of Leaving", f"{prob_leave*100:.1f}%")
    c3.metric("Probability of Staying", f"{prob_stay*100:.1f}%")

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