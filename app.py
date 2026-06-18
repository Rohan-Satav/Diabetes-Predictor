"""
app.py — Streamlit UI for Diabetes Prediction.
Random Forest model only. Shows Model Performance and Predict pages.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import model as ml

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header{font-size:2.4rem;font-weight:700;color:#1a5276;text-align:center;padding:1rem 0 .3rem}
    .sub-header{font-size:1rem;color:#5d6d7e;text-align:center;margin-bottom:1.5rem}
    .metric-card{background:#eaf4fb;border-radius:10px;padding:1rem 1.5rem;text-align:center}
    .metric-value{font-size:2rem;font-weight:700;color:#1a5276}
    .metric-label{font-size:.85rem;color:#5d6d7e}
    .predict-pos{background:#fadbd8;border-left:5px solid #e74c3c;padding:1rem 1.5rem;
                 border-radius:8px;font-size:1.1rem;font-weight:600;color:#922b21}
    .predict-neg{background:#d5f5e3;border-left:5px solid #27ae60;padding:1rem 1.5rem;
                 border-radius:8px;font-size:1.1rem;font-weight:600;color:#1e8449}
</style>
""", unsafe_allow_html=True)

DATA_PATH = "diabetes.csv"

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/diabetes.png", width=80)
st.sidebar.title("🩺 Diabetes Prediction")
st.sidebar.markdown("**Model:** Random Forest (150 trees)")

page = st.sidebar.radio(
    "Navigate",
    ["🤖 Model Performance", "🔮 Predict"]
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Dataset**: Pima Indians Diabetes\n\n"
    "**Records**: 768 patients\n\n"
    "**Target**: Diabetic (1) / Non-Diabetic (0)"
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

import os
if not os.path.exists(DATA_PATH):
    st.error(
        f"❌ `{DATA_PATH}` not found.  \n"
        "Place `diabetes.csv` in the same folder as `app.py` and restart."
    )
    st.stop()

df = load_data()

# ── Train (cached) ────────────────────────────────────────────────────────────
@st.cache_resource
def get_trained():
    result = ml.train(df, "Random Forest")
    ml.save(result["model"], result["scaler"], "Random Forest")
    return result["model"], result["scaler"], result["metrics"]

model, scaler, metrics = get_trained()

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🤖 Model Performance":
    st.markdown('<div class="main-header">🤖 Model Performance</div>', unsafe_allow_html=True)
    st.markdown("### Model: **Random Forest** (150 estimators)")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    for col, label, key in zip(
        [c1, c2, c3, c4],
        ["Accuracy", "F1 Score", "Precision", "Recall"],
        ["accuracy", "f1", "precision", "recall"]
    ):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{metrics[key]:.2%}</div>'
                     f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(metrics["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Diabetes", "Diabetes"],
                    yticklabels=["No Diabetes", "Diabetes"],
                    linewidths=1, linecolor="white")
        ax.set_xlabel("Predicted", fontweight="bold")
        ax.set_ylabel("Actual", fontweight="bold")
        ax.set_title("Confusion Matrix", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col_r:
        st.subheader(f"ROC Curve  (AUC = {metrics['auc']:.3f})")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(metrics["fpr"], metrics["tpr"], color="#1a5276", lw=2.5,
                label=f"AUC = {metrics['auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
        ax.fill_between(metrics["fpr"], metrics["tpr"], alpha=0.08, color="#1a5276")
        ax.set_xlabel("False Positive Rate", fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontweight="bold")
        ax.set_title("ROC Curve", fontweight="bold")
        ax.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    if "feature_importances" in metrics:
        st.subheader("🌲 Feature Importances")
        imp = metrics["feature_importances"].sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(imp.index, imp.values,
                       color=plt.cm.Blues(np.linspace(0.4, 0.9, len(imp))))
        ax.set_xlabel("Importance Score", fontweight="bold")
        ax.set_title("Feature Importances (Random Forest)", fontweight="bold")
        for bar, val in zip(bars, imp.values):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.subheader("📋 Classification Report")
    report_df = pd.DataFrame(metrics["report"]).T
    st.dataframe(
        report_df.style.background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"]),
        use_container_width=True
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown('<div class="main-header">🔮 Predict Diabetes Risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter patient details below</div>', unsafe_allow_html=True)
    st.markdown("---")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            pregnancies    = st.number_input("Pregnancies", 0, 20, 1)
            glucose        = st.number_input("Glucose (mg/dL)", 0, 300, 120)
            blood_pressure = st.number_input("Blood Pressure (mmHg)", 0, 150, 70)
        with c2:
            skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
            insulin        = st.number_input("Insulin (μU/mL)", 0, 900, 80)
            bmi            = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
        with c3:
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.47, step=0.01)
            age = st.number_input("Age (years)", 1, 120, 30)

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        result = ml.predict(
            model, scaler,
            [pregnancies, glucose, blood_pressure, skin_thickness,
             insulin, bmi, dpf, age]
        )
        pred  = result["prediction"]
        proba = result["probability"]

        st.markdown("---")
        col_res, col_prob = st.columns(2)

        with col_res:
            st.subheader("Prediction Result")
            if pred == 1:
                st.markdown(
                    '<div class="predict-pos">⚠️ High Risk: Patient is likely Diabetic</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="predict-neg">✅ Low Risk: Patient is likely Non-Diabetic</div>',
                    unsafe_allow_html=True
                )
            st.markdown(f"""
| Metric | Value |
|--------|-------|
| Non-Diabetic probability | **{proba[0]:.2%}** |
| Diabetic probability | **{proba[1]:.2%}** |
| Model used | **Random Forest** |
""")

        with col_prob:
            st.subheader("Probability Breakdown")
            fig, ax = plt.subplots(figsize=(4, 3))
            bars = ax.bar(["Non-Diabetic", "Diabetic"], proba,
                          color=["#2ecc71", "#e74c3c"], edgecolor="white", linewidth=2, width=0.5)
            for bar, val in zip(bars, proba):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.1%}", ha="center", fontweight="bold", fontsize=12)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Probability", fontweight="bold")
            ax.set_title("Risk Probability", fontweight="bold")
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        st.info(
            "⚠️ **Disclaimer**: This tool is for educational purposes only. "
            "Always consult a qualified medical professional for diagnosis and treatment."
        )
