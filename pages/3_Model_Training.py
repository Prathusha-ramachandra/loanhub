"""
pages/3_Model_Training.py - Model Training

Train three baseline classifiers on the German Credit Dataset:
1. Logistic Regression
2. Random Forest
3. XGBoost

Display evaluation metrics, confusion matrices, and ROC curves.
These are BASELINE models — unfairness is expected and intentional 
so we can measure it in the Fairness Analysis page.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import load_german_credit_data, train_all_models

st.set_page_config(page_title="Model Training - FairLoan AI", page_icon="🤖", layout="wide")

PLOT_TEMPLATE = "plotly_dark"
MODEL_COLORS = {
    "Logistic Regression": "#4F8EF7",
    "Random Forest": "#34D399",
    "XGBoost": "#FBBF24"
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.section-hdr { font-size:1.4rem; font-weight:600; color:#e6edf3;
    border-bottom:2px solid #4F8EF7; padding-bottom:.4rem; margin:1.5rem 0 1rem; }
.model-card { background:#161b22; border:1px solid #30363d; border-radius:12px; padding:1.2rem; }
.model-card h3 { color:#4F8EF7; font-size:1.1rem; margin:0 0 .6rem; }
.model-card p { color:#c9d1d9; font-size:.88rem; line-height:1.6; margin:0; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🤖 Model Training")
st.markdown("*Training three baseline classifiers — no fairness constraints applied (intentional)*")

st.markdown("""
<div style="background:#0d2340; border:1px solid #1d4ed8; border-radius:8px; padding:1rem; margin-bottom:1rem;">
    <strong style="color:#60a5fa;">Baseline Models:</strong> These models are trained WITHOUT 
    any fairness constraints. Their bias will be measured on the Fairness Analysis page. 
    This "unfair baseline" is necessary to demonstrate the value of bias mitigation techniques.
</div>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading dataset..."):
    df = load_german_credit_data()

if df is None:
    st.error("Failed to load dataset. Go to EDA page first.")
    st.stop()

# ── Model Architecture Cards ───────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Model Architecture</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="model-card">
        <h3>📈 Logistic Regression</h3>
        <p><strong>Type:</strong> Linear classifier</p>
        <p><strong>Hyperparameters:</strong> C=1.0, max_iter=1000</p>
        <p><strong>Advantages:</strong> Highly interpretable, fast, good calibrated probabilities</p>
        <p><strong>Use case:</strong> Linear decision boundaries; baseline reference</p>
        <p style="color:#fbbf24; margin-top:.5rem;"><strong>Fair ML role:</strong> 
        Base estimator for Exponentiated Gradient mitigation</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="model-card">
        <h3>🌲 Random Forest</h3>
        <p><strong>Type:</strong> Ensemble of decision trees</p>
        <p><strong>Hyperparameters:</strong> n_estimators=100, max_depth=10, random_state=42</p>
        <p><strong>Advantages:</strong> Handles non-linearity, feature importance, robust</p>
        <p><strong>Use case:</strong> Complex patterns; often best for tabular data</p>
        <p style="color:#fbbf24; margin-top:.5rem;"><strong>Fair ML role:</strong> 
        Main model for SHAP explanations in prediction tool</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="model-card">
        <h3>⚡ XGBoost</h3>
        <p><strong>Type:</strong> Gradient boosting</p>
        <p><strong>Hyperparameters:</strong> n_estimators=100, eval_metric=logloss</p>
        <p><strong>Advantages:</strong> Typically highest accuracy, handles class imbalance</p>
        <p><strong>Use case:</strong> Production models; winning Kaggle approach</p>
        <p style="color:#fbbf24; margin-top:.5rem;"><strong>Fair ML role:</strong> 
        Tests if better accuracy = more bias</p>
    </div>
    """, unsafe_allow_html=True)

# ── Preprocessing Pipeline ─────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Preprocessing Pipeline</div>', unsafe_allow_html=True)

with st.expander("View Preprocessing Steps", expanded=False):
    st.code("""
Preprocessing Pipeline (sklearn ColumnTransformer):
╔═══════════════════════════════════════════════════════════════╗
║  Input: 20 raw features (13 categorical + 7 numerical)       ║
╠═══════════════════════════════════════════════════════════════╣
║  Categorical (13 features):                                  ║
║    → OneHotEncoder(handle_unknown='ignore')                  ║
║    → Converts codes like A11, A12 to binary vectors          ║
║    → Handles unseen categories gracefully                    ║
║                                                              ║
║  Numerical (7 features):                                     ║
║    → StandardScaler (mean=0, std=1)                          ║
║    → Ensures all features on same scale                      ║
║    → Important for Logistic Regression convergence           ║
╠═══════════════════════════════════════════════════════════════╣
║  Train/Test Split:                                           ║
║    → 80% training (800 samples)                              ║
║    → 20% testing (200 samples)                               ║
║    → Stratified: preserves 70/30 class ratio                 ║
╚═══════════════════════════════════════════════════════════════╝
    """, language="text")

# ── Train Models ──────────────────────────────────────────────────────────────
train_btn = st.button("🚀 Train All Models", type="primary", use_container_width=False)

if train_btn or "train_results" in st.session_state:
    if train_btn or "train_results" not in st.session_state:
        with st.spinner("Training Logistic Regression, Random Forest, and XGBoost..."):
            results = train_all_models(df)
            st.session_state["train_results"] = results
        st.success("All three models trained successfully!")
    else:
        results = st.session_state["train_results"]

    # ── Performance Metrics Summary ────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Performance Metrics</div>', unsafe_allow_html=True)

    metrics_data = []
    for model_name, metrics in results["metrics"].items():
        metrics_data.append({
            "Model": model_name,
            "Accuracy": f"{metrics['accuracy']*100:.1f}%",
            "Precision": f"{metrics['precision']*100:.1f}%",
            "Recall": f"{metrics['recall']*100:.1f}%",
            "AUC-ROC": f"{metrics['auc']:.4f}",
            "Accuracy (raw)": metrics["accuracy"],
            "AUC (raw)": metrics["auc"]
        })

    metrics_df = pd.DataFrame(metrics_data)

    col1, col2, col3 = st.columns(3)
    for col, row in zip([col1, col2, col3], metrics_data):
        with col:
            st.markdown(f"""
            <div style="background:#161b22; border:1px solid #30363d; border-radius:10px; padding:1rem;">
                <h4 style="color:{MODEL_COLORS[row['Model']]}; margin:0 0 .75rem;">{row['Model']}</h4>
                <table style="width:100%; color:#c9d1d9; font-size:.9rem;">
                    <tr><td>Accuracy</td><td><strong>{row['Accuracy']}</strong></td></tr>
                    <tr><td>Precision</td><td><strong>{row['Precision']}</strong></td></tr>
                    <tr><td>Recall</td><td><strong>{row['Recall']}</strong></td></tr>
                    <tr><td>AUC-ROC</td><td><strong>{row['AUC-ROC']}</strong></td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

    # ── Comparison Bar Chart ───────────────────────────────────────────────────
    st.markdown("#### Metric Comparison Across Models")

    metric_compare = pd.DataFrame([
        {"Model": row["Model"], "Metric": "Accuracy", "Value": results["metrics"][row["Model"]]["accuracy"]}
        for row in metrics_data
    ] + [
        {"Model": row["Model"], "Metric": "Precision", "Value": results["metrics"][row["Model"]]["precision"]}
        for row in metrics_data
    ] + [
        {"Model": row["Model"], "Metric": "Recall", "Value": results["metrics"][row["Model"]]["recall"]}
        for row in metrics_data
    ] + [
        {"Model": row["Model"], "Metric": "AUC-ROC", "Value": results["metrics"][row["Model"]]["auc"]}
        for row in metrics_data
    ])

    fig = px.bar(metric_compare, x="Metric", y="Value", color="Model", barmode="group",
                 template=PLOT_TEMPLATE, title="Model Performance Comparison",
                 color_discrete_map=MODEL_COLORS,
                 text_auto=".3f")
    fig.update_layout(yaxis_range=[0, 1.1], legend_title="Model")
    st.plotly_chart(fig, use_container_width=True)

    # ── Confusion Matrices ────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Confusion Matrices</div>', unsafe_allow_html=True)

    fig = make_subplots(rows=1, cols=3,
        subplot_titles=list(results["metrics"].keys()),
        horizontal_spacing=0.1)

    for i, (model_name, metrics) in enumerate(results["metrics"].items()):
        cm = metrics["confusion_matrix"]
        labels = ["Denied (0)", "Approved (1)"]

        # Normalize for percentages
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

        annotations = []
        for ri in range(2):
            for ci in range(2):
                annotations.append(
                    f"{cm[ri, ci]}<br>({cm_pct[ri, ci]:.1f}%)"
                )

        heatmap = go.Heatmap(
            z=cm_pct,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=(i == 2),
            text=[[f"{cm[ri,ci]}<br>({cm_pct[ri,ci]:.1f}%)" for ci in range(2)] for ri in range(2)],
            texttemplate="%{text}",
            textfont={"size": 12}
        )
        fig.add_trace(heatmap, row=1, col=i+1)

    fig.update_layout(height=350, template=PLOT_TEMPLATE,
        title_text="Confusion Matrices (True Label vs Predicted Label)",
        title_x=0.5)
    fig.update_xaxes(title_text="Predicted")
    fig.update_yaxes(title_text="Actual")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Understanding Confusion Matrix Terms"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **True Positive (TP):** Predicted Approved AND Actually Good Credit
            → Correctly approved a creditworthy applicant ✅
            
            **True Negative (TN):** Predicted Denied AND Actually Bad Credit
            → Correctly rejected a risky applicant ✅
            """)
        with col2:
            st.markdown("""
            **False Positive (FP):** Predicted Approved BUT Actually Bad Credit
            → Bank loses money on a risky loan ❌
            
            **False Negative (FN):** Predicted Denied BUT Actually Good Credit
            → Creditworthy applicant unfairly rejected ❌ (Fairness concern!)
            """)

    # ── ROC Curves ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">ROC Curves</div>', unsafe_allow_html=True)

    fig = go.Figure()

    for model_name, metrics in results["metrics"].items():
        fig.add_trace(go.Scatter(
            x=metrics["fpr"], y=metrics["tpr"],
            name=f"{model_name} (AUC={metrics['auc']:.4f})",
            line=dict(color=MODEL_COLORS[model_name], width=2),
            mode="lines"
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="Random Classifier",
        line=dict(color="#6b7280", width=1, dash="dash"),
        mode="lines"
    ))

    fig.update_layout(
        template=PLOT_TEMPLATE, height=450,
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.6, y=0.1)
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.05])

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    > **ROC Curve Interpretation:** The area under the ROC curve (AUC) measures discrimination ability.
    > AUC=0.5 means random (no skill), AUC=1.0 means perfect. Higher AUC = better model, 
    > but higher AUC doesn't mean fairer model!
    """)

    # ── Feature Importance ─────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)

    rf_model = results["models"]["Random Forest"]
    from utils import get_feature_names_after_preprocessing

    try:
        feature_names = get_feature_names_after_preprocessing(results["preprocessor"])
        importances = rf_model.feature_importances_

        if len(feature_names) == len(importances):
            imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values("Importance", ascending=True).tail(20)

            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         template=PLOT_TEMPLATE, title="Top 20 Feature Importances (Random Forest)",
                         color="Importance", color_continuous_scale="Blues")
            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute feature importances: {e}")

    st.info("""
    **Key Observations from Baseline Models:**
    - XGBoost typically achieves the highest AUC (~0.77), followed by Random Forest, then Logistic Regression
    - All three models have reasonable accuracy (70-77%) — but accuracy alone doesn't tell the fairness story
    - **Critical question:** Do these models perform equally well for male vs. female applicants?
    - **Next Step →** Go to Fairness Analysis to measure group-level performance disparities
    """)

else:
    st.markdown("""
    <div style="background:#161b22; border:2px dashed #30363d; border-radius:12px;
        padding:3rem; text-align:center; margin:2rem 0;">
        <div style="font-size:3rem;">🤖</div>
        <h3 style="color:#e6edf3; margin:.75rem 0 .5rem;">Models Not Trained Yet</h3>
        <p style="color:#8b949e;">Click the "Train All Models" button above to start training.</p>
        <p style="color:#8b949e; font-size:.85rem;">
            Training will take ~30-60 seconds. Models are cached for the session.
        </p>
    </div>
    """, unsafe_allow_html=True)
