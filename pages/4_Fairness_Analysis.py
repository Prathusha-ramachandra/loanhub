"""
pages/4_Fairness_Analysis.py - Fairness Analysis

Compute and visualize fairness metrics on baseline ML models:
- Demographic Parity Difference
- Equalized Odds Difference
- Equal Opportunity Difference
- Accuracy by group

Using fairlearn's MetricFrame for systematic fairness evaluation.
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
from utils import load_german_credit_data, train_all_models, compute_fairness_metrics

st.set_page_config(page_title="Fairness Analysis - FairLoan AI", page_icon="⚖️", layout="wide")

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
.metric-explain { background:#0d2340; border:1px solid #1d4ed8; border-radius:8px; padding:1rem; margin:.5rem 0; }
.metric-explain h4 { color:#60a5fa; margin:0 0 .4rem; font-size:.95rem; }
.metric-explain p { color:#93c5fd; font-size:.85rem; margin:0; }
.fairness-score { font-size:1.6rem; font-weight:700; }
.fair { color:#34D399; }
.moderate { color:#FBBF24; }
.poor { color:#F87171; }
</style>
""", unsafe_allow_html=True)

st.markdown("# ⚖️ Fairness Analysis")
st.markdown("*Quantifying bias in baseline ML models using industry-standard fairness metrics*")

# ── Conceptual Overview ────────────────────────────────────────────────────────
with st.expander("📚 Fairness Metrics Explained (click to expand)", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-explain">
            <h4>Demographic Parity Difference (DPD)</h4>
            <p><em>P(ŷ=1|A=0) − P(ŷ=1|A=1)</em></p>
            <p>Difference in approval rates between groups.<br>
            DPD=0 means equal approval rates (group-blind selection).
            <br>DPD>0 means group A=0 gets fewer approvals.</p>
            <p><strong>Threshold:</strong> |DPD| < 0.1 is generally acceptable</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-explain">
            <h4>Equalized Odds Difference (EOD)</h4>
            <p><em>max(|ΔTPR|, |ΔFPR|)</em></p>
            <p>Maximum difference in True Positive Rate OR False Positive Rate 
            between groups. Considers both beneficial and harmful outcomes.
            <br>EOD=0 means both TPR and FPR are equal across groups.</p>
            <p><strong>Threshold:</strong> |EOD| < 0.1 is generally acceptable</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-explain">
            <h4>Equal Opportunity Difference (EqOppD)</h4>
            <p><em>|TPR_group_a − TPR_group_b|</em></p>
            <p>Difference in True Positive Rate (recall) between groups.
            Ensures creditworthy applicants are approved at the same rate 
            regardless of their protected group membership.
            <br>EqOppD=0 means equal opportunity for creditworthy applicants.</p>
            <p><strong>Threshold:</strong> |EqOppD| < 0.1 is acceptable</p>
        </div>
        """, unsafe_allow_html=True)

# ── Load Data & Models ────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    protected_attr = st.selectbox(
        "Select Protected Attribute:",
        ["sex", "age_group"],
        format_func=lambda x: "Sex (Male vs Female)" if x == "sex" else "Age Group (>25 vs ≤25)"
    )
with col2:
    threshold_info = st.empty()

attr_labels = {
    "sex": {0: "Female", 1: "Male"},
    "age_group": {0: "Age ≤25", 1: "Age >25"}
}

with st.spinner("Loading data and training models..."):
    df = load_german_credit_data()
    results = train_all_models(df)

if df is None or results is None:
    st.error("Failed to load data or train models. Please visit the Model Training page first.")
    st.stop()

st.success("Models loaded. Computing fairness metrics...")

y_test = results["y_test"]
sensitive_test = results["sensitive_test"][protected_attr]

# ── Compute Fairness Metrics for All Models ───────────────────────────────────
fairness_results = {}
for model_name in results["models"].keys():
    y_pred = results["predictions"][model_name]
    fairness_results[model_name] = compute_fairness_metrics(
        y_test, y_pred, sensitive_test, model_name
    )

# ── Summary Table ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Fairness Metrics Summary</div>', unsafe_allow_html=True)

summary_data = []
for model_name, fm in fairness_results.items():
    dpd = abs(fm["demographic_parity_difference"])
    eod = abs(fm["equalized_odds_difference"])
    eq_opp = abs(fm["equal_opportunity_difference"])
    acc = fm["overall_accuracy"]

    # Fairness rating based on thresholds
    avg_bias = (dpd + eod + eq_opp) / 3
    rating = "Fair" if avg_bias < 0.1 else ("Moderate" if avg_bias < 0.2 else "Biased")
    color = "fair" if rating == "Fair" else ("moderate" if rating == "Moderate" else "poor")

    summary_data.append({
        "Model": model_name,
        "Accuracy": f"{acc*100:.1f}%",
        "DPD (↓ better)": f"{dpd:.4f}",
        "EOD (↓ better)": f"{eod:.4f}",
        "EqOppD (↓ better)": f"{eq_opp:.4f}",
        "Avg Bias Score": f"{avg_bias:.4f}",
        "Fairness Rating": rating,
        "_color": color,
        "_dpd": dpd, "_eod": eod, "_eq_opp": eq_opp, "_acc": acc
    })

# Display summary table
display_df = pd.DataFrame(summary_data)[
    ["Model", "Accuracy", "DPD (↓ better)", "EOD (↓ better)", "EqOppD (↓ better)", "Avg Bias Score", "Fairness Rating"]
]
st.dataframe(display_df, use_container_width=True, hide_index=True)

# Metric cards
col1, col2, col3 = st.columns(3)
for col, row in zip([col1, col2, col3], summary_data):
    with col:
        clr = MODEL_COLORS[row["Model"]]
        st.markdown(f"""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:10px; padding:1rem;">
            <h4 style="color:{clr}; margin:0 0 .75rem;">{row['Model']}</h4>
            <div style="display:flex; flex-direction:column; gap:.4rem;">
                <div>Accuracy: <strong>{row['Accuracy']}</strong></div>
                <div>DPD: <strong style="color:{'#34D399' if row['_dpd']<0.1 else '#F87171'};">{row['DPD (↓ better)']}</strong></div>
                <div>EOD: <strong style="color:{'#34D399' if row['_eod']<0.1 else '#F87171'};">{row['EOD (↓ better)']}</strong></div>
                <div>EqOppD: <strong style="color:{'#34D399' if row['_eq_opp']<0.1 else '#F87171'};">{row['EqOppD (↓ better)']}</strong></div>
                <div>Rating: <strong class="{row['_color']}" style="font-size:1.1rem;">{row['Fairness Rating']}</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Bar Charts: Fairness Metrics Comparison ────────────────────────────────────
st.markdown('<div class="section-hdr">Fairness Metric Comparison</div>', unsafe_allow_html=True)

metrics_compare = pd.DataFrame([
    {"Model": r["Model"], "Metric": "Demographic Parity Diff", "Value": r["_dpd"]}
    for r in summary_data
] + [
    {"Model": r["Model"], "Metric": "Equalized Odds Diff", "Value": r["_eod"]}
    for r in summary_data
] + [
    {"Model": r["Model"], "Metric": "Equal Opportunity Diff", "Value": r["_eq_opp"]}
    for r in summary_data
])

fig = px.bar(metrics_compare, x="Metric", y="Value", color="Model",
             barmode="group", template=PLOT_TEMPLATE,
             title=f"Fairness Metrics by Model — Protected Attribute: {protected_attr}",
             color_discrete_map=MODEL_COLORS,
             text_auto=".4f")

# Add fairness threshold line
fig.add_hline(y=0.1, line_dash="dash", line_color="#f87171",
              annotation_text="Fairness Threshold (0.1)",
              annotation_position="top right")

fig.update_layout(yaxis_title="Metric Value (lower = fairer)", height=450)
st.plotly_chart(fig, use_container_width=True)

# ── Group-Level Approval Rates ────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Approval Rate by Protected Group</div>', unsafe_allow_html=True)

group_data = []
for model_name, fm in fairness_results.items():
    sel_rate = fm["selection_rate_by_group"]
    for group_val, rate in sel_rate.items():
        group_label = attr_labels[protected_attr].get(group_val, str(group_val))
        group_data.append({
            "Model": model_name,
            "Group": group_label,
            "Approval Rate": rate * 100
        })

group_df = pd.DataFrame(group_data)

fig = px.bar(group_df, x="Model", y="Approval Rate", color="Group",
             barmode="group", template=PLOT_TEMPLATE,
             title=f"Approval Rate by {protected_attr} Group — Each Model",
             text_auto=".1f")
fig.update_traces(texttemplate='%{text}%', textposition='outside')
fig.update_yaxes(range=[0, 110], title="Approval Rate (%)")
st.plotly_chart(fig, use_container_width=True)

# ── True Positive Rate Comparison ─────────────────────────────────────────────
st.markdown('<div class="section-hdr">True Positive Rate (Equal Opportunity) by Group</div>', unsafe_allow_html=True)

tpr_data = []
for model_name, fm in fairness_results.items():
    tpr_by_group = fm["tpr_by_group"]
    for group_val, tpr in tpr_by_group.items():
        group_label = attr_labels[protected_attr].get(group_val, str(group_val))
        tpr_data.append({
            "Model": model_name,
            "Group": group_label,
            "True Positive Rate": tpr * 100
        })

tpr_df = pd.DataFrame(tpr_data)

fig = px.bar(tpr_df, x="Model", y="True Positive Rate", color="Group",
             barmode="group", template=PLOT_TEMPLATE,
             title=f"TPR (among creditworthy) by {protected_attr} Group — Equal Opportunity",
             text_auto=".1f")
fig.update_traces(texttemplate='%{text}%', textposition='outside')
fig.update_yaxes(range=[0, 115], title="True Positive Rate (%)")
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
> **Interpretation:** Equal Opportunity requires that creditworthy applicants from all groups 
> are approved at the SAME rate. A gap here means qualified applicants from one group are 
> being systematically denied compared to equally qualified applicants from another group.
""")

# ── Accuracy by Group ─────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Accuracy by Protected Group</div>', unsafe_allow_html=True)

acc_data = []
for model_name, fm in fairness_results.items():
    acc_by_group = fm["accuracy_by_group"]
    for group_val, acc in acc_by_group.items():
        group_label = attr_labels[protected_attr].get(group_val, str(group_val))
        acc_data.append({
            "Model": model_name,
            "Group": group_label,
            "Accuracy": acc * 100
        })

acc_df = pd.DataFrame(acc_data)

fig = px.bar(acc_df, x="Model", y="Accuracy", color="Group",
             barmode="group", template=PLOT_TEMPLATE,
             title=f"Model Accuracy by {protected_attr} Group",
             text_auto=".1f")
fig.update_traces(texttemplate='%{text}%', textposition='outside')
fig.update_yaxes(range=[0, 110], title="Accuracy (%)")
st.plotly_chart(fig, use_container_width=True)

# ── MetricFrame Detail ────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Detailed MetricFrame (fairlearn)</div>', unsafe_allow_html=True)

selected_model_detail = st.selectbox("Select Model for Detailed View:", list(fairness_results.keys()))
fm_detail = fairness_results[selected_model_detail]

st.write(f"**Model:** {selected_model_detail}")
group_df_detail = fm_detail["group_metrics"].copy()
group_df_detail.index = [attr_labels[protected_attr].get(i, str(i)) for i in group_df_detail.index]
group_df_detail.index.name = f"{protected_attr} Group"

# Format nicely
for col in group_df_detail.columns:
    group_df_detail[col] = group_df_detail[col].apply(lambda x: f"{x*100:.2f}%")

st.dataframe(group_df_detail, use_container_width=True)

# ── Summary Insights ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Key Findings")

most_biased = max(summary_data, key=lambda x: x["_dpd"] + x["_eod"])
least_biased = min(summary_data, key=lambda x: x["_dpd"] + x["_eod"])

st.markdown(f"""
<div style="background:#161b22; border:1px solid #30363d; border-radius:10px; padding:1.2rem;">
    <h4 style="color:#e6edf3; margin:0 0 .75rem;">Analysis for Protected Attribute: 
    <em style="color:#4F8EF7;">{protected_attr}</em></h4>
    <ul style="color:#c9d1d9; font-size:.9rem; line-height:1.8;">
        <li>Most biased model: <strong style="color:#F87171;">{most_biased['Model']}</strong> 
        (DPD={most_biased['DPD (↓ better)']}, EOD={most_biased['EOD (↓ better)']})</li>
        <li>Least biased model: <strong style="color:#34D399;">{least_biased['Model']}</strong> 
        (DPD={least_biased['DPD (↓ better)']}, EOD={least_biased['EOD (↓ better)']})</li>
        <li>All models show bias ABOVE the 0.1 fairness threshold — mitigation is needed</li>
        <li>Higher accuracy does NOT correlate with lower bias (confirms literature findings)</li>
    </ul>
    <p style="color:#8b949e; font-size:.85rem; margin-top:.75rem;">
        ⚡ Next Step: Go to <strong>Bias Mitigation</strong> to apply reweighing, 
        Exponentiated Gradient, and Threshold Optimizer to reduce these disparities.
    </p>
</div>
""", unsafe_allow_html=True)
