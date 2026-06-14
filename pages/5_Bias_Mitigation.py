"""
pages/5_Bias_Mitigation.py - Bias Mitigation Techniques

Compare three bias mitigation approaches:
1. Pre-processing: Reweighing (Kamiran & Calders, 2012)
2. In-processing: Exponentiated Gradient Reduction (Agarwal et al., 2018)
3. Post-processing: Threshold Optimizer (Hardt et al., 2016)

Show side-by-side comparison of fairness improvement vs accuracy tradeoff.
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
from utils import load_german_credit_data, train_all_models, run_bias_mitigation

st.set_page_config(page_title="Bias Mitigation - FairLoan AI", page_icon="🔧", layout="wide")

PLOT_TEMPLATE = "plotly_dark"
TECHNIQUE_COLORS = {
    "baseline": "#F87171",
    "reweighing": "#FBBF24",
    "exp_gradient": "#34D399",
    "threshold_opt": "#4F8EF7"
}
TECHNIQUE_NAMES = {
    "baseline": "Baseline (No Mitigation)",
    "reweighing": "Reweighing (Pre-process)",
    "exp_gradient": "Exponentiated Gradient (In-process)",
    "threshold_opt": "Threshold Optimizer (Post-process)"
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.section-hdr { font-size:1.4rem; font-weight:600; color:#e6edf3;
    border-bottom:2px solid #4F8EF7; padding-bottom:.4rem; margin:1.5rem 0 1rem; }
.technique-card { background:#161b22; border:1px solid #30363d; border-radius:12px; padding:1.2rem; height:100%; }
.technique-card h3 { font-size:1rem; margin:0 0 .6rem; }
.technique-card p, .technique-card li { color:#c9d1d9; font-size:.85rem; line-height:1.6; }
.improvement { color:#34D399; font-weight:600; }
.degraded { color:#F87171; font-weight:600; }
.neutral { color:#FBBF24; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🔧 Bias Mitigation Techniques")
st.markdown("*Applying three categories of bias mitigation and measuring improvement*")

# ── Technique Explanations ────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Mitigation Technique Overview</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="technique-card">
        <h3 style="color:#FBBF24;">1. Pre-Processing: Reweighing</h3>
        <p><strong>When:</strong> Before model training</p>
        <p><strong>Mechanism:</strong> Assigns sample weights to training examples to compensate 
        for historical bias. Disadvantaged groups with positive outcomes are upweighted.</p>
        <p><strong>Formula:</strong><br>
        w(A=a, Y=y) = P(A=a)·P(Y=y) / P(A=a, Y=y)</p>
        <ul>
            <li>Model-agnostic (works with any classifier)</li>
            <li>Simple and interpretable</li>
            <li>Minimal impact on model architecture</li>
        </ul>
        <p style="color:#FBBF24;"><strong>Reference:</strong> Kamiran & Calders, KAIS 2012</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="technique-card">
        <h3 style="color:#34D399;">2. In-Processing: Exponentiated Gradient</h3>
        <p><strong>When:</strong> During model training</p>
        <p><strong>Mechanism:</strong> Reduces fairness-constrained classification to 
        a sequence of cost-sensitive learning problems. Uses Lagrangian relaxation to 
        enforce demographic parity or equalized odds constraints.</p>
        <p><strong>Constraint:</strong> Demographic Parity<br>
        |P(ŷ=1|A=0) − P(ŷ=1|A=1)| ≤ ε</p>
        <ul>
            <li>Direct fairness constraint enforcement</li>
            <li>Flexible: supports multiple fairness criteria</li>
            <li>May have higher computational cost</li>
        </ul>
        <p style="color:#34D399;"><strong>Reference:</strong> Agarwal et al., ICML 2018</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="technique-card">
        <h3 style="color:#4F8EF7;">3. Post-Processing: Threshold Optimizer</h3>
        <p><strong>When:</strong> After model training</p>
        <p><strong>Mechanism:</strong> Finds group-specific decision thresholds that satisfy 
        a fairness constraint. Instead of threshold=0.5 for all, uses different cutoffs 
        per protected group.</p>
        <p><strong>Constraint:</strong> Equalized Odds<br>
        TPR and FPR equalized across groups</p>
        <ul>
            <li>Works on any pre-trained model</li>
            <li>Post-hoc: no retraining required</li>
            <li>Strong fairness guarantees (provably optimal)</li>
        </ul>
        <p style="color:#4F8EF7;"><strong>Reference:</strong> Hardt et al., NeurIPS 2016</p>
    </div>
    """, unsafe_allow_html=True)

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    protected_attr = st.selectbox(
        "Protected Attribute:",
        ["sex", "age_group"],
        format_func=lambda x: "Sex (Male vs Female)" if x == "sex" else "Age Group (>25 vs ≤25)"
    )

# ── Load & Run Mitigation ─────────────────────────────────────────────────────
with st.spinner("Training models and applying bias mitigation techniques..."):
    df = load_german_credit_data()
    train_results = train_all_models(df)
    mitigation_results = run_bias_mitigation(train_results, df, protected_attr)

st.success("All three mitigation techniques applied. Comparing results...")

# ── Comparison Table ──────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Side-by-Side Comparison</div>', unsafe_allow_html=True)

comparison_data = []
for key, display_name in TECHNIQUE_NAMES.items():
    if key in mitigation_results and "error" not in mitigation_results[key]:
        fm = mitigation_results[key]
        dpd = abs(fm["demographic_parity_difference"])
        eod = abs(fm["equalized_odds_difference"])
        eq_opp = abs(fm["equal_opportunity_difference"])
        acc = fm["overall_accuracy"]
        avg_bias = (dpd + eod + eq_opp) / 3

        comparison_data.append({
            "Technique": display_name,
            "Stage": key.replace("_", " ").title() if key != "baseline" else "Baseline",
            "Accuracy": acc,
            "DPD": dpd,
            "EOD": eod,
            "EqOppD": eq_opp,
            "Avg Bias": avg_bias,
            "_key": key
        })

if not comparison_data:
    st.error("Mitigation failed. Check library installation.")
    st.stop()

comp_df = pd.DataFrame(comparison_data)

# Display formatted table
display_table = comp_df.copy()
display_table["Accuracy"] = display_table["Accuracy"].apply(lambda x: f"{x*100:.2f}%")
display_table["DPD"] = display_table["DPD"].apply(lambda x: f"{x:.4f}")
display_table["EOD"] = display_table["EOD"].apply(lambda x: f"{x:.4f}")
display_table["EqOppD"] = display_table["EqOppD"].apply(lambda x: f"{x:.4f}")
display_table["Avg Bias"] = display_table["Avg Bias"].apply(lambda x: f"{x:.4f}")
display_table = display_table.drop(columns=["_key", "Stage"])

st.dataframe(display_table, use_container_width=True, hide_index=True)

# ── Visual Comparison Charts ───────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Fairness vs Accuracy Trade-off</div>', unsafe_allow_html=True)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Fairness Metrics (Lower = Fairer)", "Accuracy Comparison"),
    horizontal_spacing=0.12
)

technique_labels = [TECHNIQUE_NAMES[r["_key"]] for r in comparison_data]
colors = [TECHNIQUE_COLORS[r["_key"]] for r in comparison_data]

# Fairness metrics grouped bar
metrics_labels = ["DPD", "EOD", "EqOppD"]
for i, (metric, values) in enumerate([
    ("DPD", [r["DPD"] for r in comparison_data]),
    ("EOD", [r["EOD"] for r in comparison_data]),
    ("EqOppD", [r["EqOppD"] for r in comparison_data])
]):
    fig.add_trace(
        go.Bar(name=metric, x=technique_labels, y=values,
               marker_color=["#4F8EF7", "#34D399", "#FBBF24"][i],
               showlegend=True),
        row=1, col=1
    )

# Accuracy bar
for i, row in enumerate(comparison_data):
    fig.add_trace(
        go.Bar(
            name=TECHNIQUE_NAMES[row["_key"]],
            x=[TECHNIQUE_NAMES[row["_key"]]],
            y=[row["Accuracy"] * 100],
            marker_color=colors[i],
            text=f"{row['Accuracy']*100:.1f}%",
            textposition="outside",
            showlegend=False
        ),
        row=1, col=2
    )

fig.add_hline(y=0.1, row=1, col=1, line_dash="dash", line_color="#F87171",
              annotation_text="Threshold")

fig.update_layout(template=PLOT_TEMPLATE, height=450, barmode="group",
    title_text=f"Bias Mitigation Results — {protected_attr}",
    title_x=0.5)
fig.update_yaxes(title_text="Metric Value", row=1, col=1)
fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)

st.plotly_chart(fig, use_container_width=True)

# ── Radar Chart ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Fairness-Accuracy Trade-off Radar</div>', unsafe_allow_html=True)

fig_radar = go.Figure()

categories = ["Accuracy", "Low DPD", "Low EOD", "Low EqOppD"]
categories_closed = categories + [categories[0]]

for row in comparison_data:
    # Invert bias metrics: 1 - metric = "goodness"
    values = [
        row["Accuracy"],
        max(0, 1 - row["DPD"] * 5),  # Scale: 0.2 DPD → score=0
        max(0, 1 - row["EOD"] * 5),
        max(0, 1 - row["EqOppD"] * 5)
    ]
    values_closed = values + [values[0]]

    fig_radar.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        name=TECHNIQUE_NAMES[row["_key"]],
        line_color=TECHNIQUE_COLORS[row["_key"]],
        opacity=0.5
    ))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    template=PLOT_TEMPLATE, height=450,
    title="Fairness-Accuracy Radar (1.0 = Perfect)",
    legend=dict(orientation="h", y=-0.15)
)
st.plotly_chart(fig_radar, use_container_width=True)

# ── Improvement Analysis ───────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Improvement Over Baseline</div>', unsafe_allow_html=True)

if len(comparison_data) > 1:
    baseline_row = next((r for r in comparison_data if r["_key"] == "baseline"), None)

    if baseline_row:
        improvement_data = []
        for row in comparison_data:
            if row["_key"] != "baseline":
                dpd_improvement = (baseline_row["DPD"] - row["DPD"]) / max(baseline_row["DPD"], 1e-6) * 100
                eod_improvement = (baseline_row["EOD"] - row["EOD"]) / max(baseline_row["EOD"], 1e-6) * 100
                eq_improvement = (baseline_row["EqOppD"] - row["EqOppD"]) / max(baseline_row["EqOppD"], 1e-6) * 100
                acc_change = (row["Accuracy"] - baseline_row["Accuracy"]) * 100

                improvement_data.append({
                    "Technique": TECHNIQUE_NAMES[row["_key"]],
                    "DPD Reduction (%)": dpd_improvement,
                    "EOD Reduction (%)": eod_improvement,
                    "EqOppD Reduction (%)": eq_improvement,
                    "Accuracy Change (pp)": acc_change
                })

        if improvement_data:
            imp_df = pd.DataFrame(improvement_data)

            # Format for display
            display_imp = imp_df.copy()
            for col in ["DPD Reduction (%)", "EOD Reduction (%)", "EqOppD Reduction (%)", "Accuracy Change (pp)"]:
                display_imp[col] = display_imp[col].apply(
                    lambda x: f"+{x:.1f}%" if x >= 0 else f"{x:.1f}%"
                )

            st.dataframe(display_imp, use_container_width=True, hide_index=True)

            st.markdown("""
            > **Positive values** in bias reduction columns = fairness **improved** (bias decreased)
            > **Negative values** in accuracy change = accuracy **decreased** (acceptable tradeoff)
            > The goal is maximum bias reduction with minimum accuracy loss
            """)

            # Visual
            imp_melt = imp_df.melt(
                id_vars="Technique",
                value_vars=["DPD Reduction (%)", "EOD Reduction (%)", "EqOppD Reduction (%)"],
                var_name="Metric", value_name="Improvement (%)"
            )

            fig_imp = px.bar(imp_melt, x="Technique", y="Improvement (%)", color="Metric",
                             barmode="group", template=PLOT_TEMPLATE,
                             title="Bias Reduction vs Baseline (%)")
            fig_imp.add_hline(y=0, line_color="#8b949e", line_width=1)
            st.plotly_chart(fig_imp, use_container_width=True)

# ── Recommendations ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Recommendations")

st.markdown("""
<div style="background:#161b22; border:1px solid #30363d; border-radius:10px; padding:1.2rem;">
    <h4 style="color:#e6edf3; margin:0 0 .75rem;">Which Technique Should You Use?</h4>
    <table style="width:100%; color:#c9d1d9; font-size:.88rem;">
        <tr style="color:#4F8EF7;">
            <th>Scenario</th>
            <th>Recommended Technique</th>
            <th>Reason</th>
        </tr>
        <tr>
            <td>Can retrain model with new data</td>
            <td>🟡 Reweighing</td>
            <td>Simple, model-agnostic, minimal overhead</td>
        </tr>
        <tr>
            <td>Need provable fairness guarantees</td>
            <td>🟢 Exponentiated Gradient</td>
            <td>Mathematically rigorous constraint enforcement</td>
        </tr>
        <tr>
            <td>Cannot retrain (legacy model)</td>
            <td>🔵 Threshold Optimizer</td>
            <td>Works post-hoc on any existing model</td>
        </tr>
        <tr>
            <td>Equal opportunity is the priority</td>
            <td>🔵 Threshold Optimizer</td>
            <td>Optimizes equalized odds by design</td>
        </tr>
        <tr>
            <td>Demographic parity required (regulation)</td>
            <td>🟢 Exponentiated Gradient</td>
            <td>Direct demographic parity constraint</td>
        </tr>
    </table>
    <p style="color:#8b949e; font-size:.82rem; margin-top:.75rem;">
        Note: No single technique dominates all scenarios. The best choice depends on 
        your fairness objective, legal requirements, and operational constraints.
        The EU AI Act recommends a combination of pre- and post-processing techniques 
        with continuous monitoring.
    </p>
</div>
""", unsafe_allow_html=True)
