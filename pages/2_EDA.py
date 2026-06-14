"""
pages/2_EDA.py - Exploratory Data Analysis

Automatically downloads the UCI German Credit Dataset and provides:
- Dataset statistics and info
- Missing value analysis
- Feature distributions
- Correlation heatmap
- Bias preview by protected attributes (Sex, Age)
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
from utils import load_german_credit_data, CATEGORICAL_FEATURES, NUMERICAL_FEATURES

st.set_page_config(page_title="EDA - FairLoan AI", page_icon="📊", layout="wide")

PLOT_TEMPLATE = "plotly_dark"
COLORS = {"primary": "#4F8EF7", "secondary": "#34D399", "danger": "#F87171",
          "warning": "#FBBF24", "male": "#60a5fa", "female": "#f472b6"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.section-hdr { font-size:1.4rem; font-weight:600; color:#e6edf3;
    border-bottom:2px solid #4F8EF7; padding-bottom:.4rem; margin:1.5rem 0 1rem; }
.info-card { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📊 Exploratory Data Analysis")
st.markdown("*Exploring the UCI German Credit Dataset for patterns and bias*")

# ── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading German Credit Dataset from UCI repository..."):
    df = load_german_credit_data()

if df is None:
    st.error("Failed to load dataset. Please check your internet connection.")
    st.stop()

st.success(f"Dataset loaded: {len(df):,} rows × {df.shape[1]} columns")

# ── Dataset Overview ───────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Dataset Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Samples", f"{len(df):,}")
col2.metric("Features", "20")
col3.metric("Good Credit", f"{(df['target']==1).sum():,}", f"{(df['target']==1).mean()*100:.1f}%")
col4.metric("Bad Credit", f"{(df['target']==0).sum():,}", f"{(df['target']==0).mean()*100:.1f}%")
col5.metric("Missing Values", "0", "Clean dataset")

with st.expander("📋 View Column Descriptions", expanded=False):
    col_desc = pd.DataFrame({
        "Column": ["status", "duration", "credit_history", "purpose", "credit_amount",
                   "savings", "employment", "installment_rate", "personal_status",
                   "other_debtors", "residence_since", "property", "age",
                   "other_installment", "housing", "existing_credits", "job",
                   "liable_people", "telephone", "foreign_worker", "class",
                   "sex", "age_group", "target"],
        "Type": ["Categorical"]*10 + ["Numerical"] + ["Categorical"] + ["Numerical"] +
                ["Categorical"]*3 + ["Numerical"] + ["Categorical"] + ["Numerical"] +
                ["Categorical"] + ["Numerical"] + ["Derived"]*3,
        "Description": [
            "Status of existing checking account",
            "Duration in months",
            "Credit history",
            "Purpose of credit",
            "Credit amount (DM)",
            "Savings account/bonds",
            "Present employment duration",
            "Installment rate % of income",
            "Personal status and sex (encodes both)",
            "Other debtors or guarantors",
            "Present residence duration (years)",
            "Property type",
            "Age in years",
            "Other installment plans",
            "Housing type",
            "Number of existing credits at bank",
            "Job type",
            "Number of liable dependents",
            "Telephone ownership",
            "Foreign worker status",
            "1=Good credit, 2=Bad credit (original)",
            "Derived: 0=Female, 1=Male",
            "Derived: 0=Age≤25, 1=Age>25",
            "1=Approved, 0=Denied (for modeling)"
        ]
    })
    st.dataframe(col_desc, use_container_width=True, hide_index=True)

with st.expander("🔍 Raw Data Preview", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    st.caption("Showing first 20 rows of the dataset")

# ── Missing Values ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Missing Values Analysis</div>', unsafe_allow_html=True)
missing = df.isnull().sum()
if missing.sum() == 0:
    st.markdown("""
    <div style="background:#0d2b1e; border:1px solid #16a34a; border-radius:8px; padding:1rem;">
        ✅ <strong style="color:#86efac;">No missing values found!</strong> 
        The German Credit Dataset is clean and complete (1,000 samples, all features present).
        This is a high-quality benchmark dataset used extensively in ML fairness research.
    </div>
    """, unsafe_allow_html=True)
else:
    fig = px.bar(x=missing[missing > 0].index, y=missing[missing > 0].values,
                 template=PLOT_TEMPLATE, title="Missing Values per Column",
                 color_discrete_sequence=[COLORS["danger"]])
    st.plotly_chart(fig, use_container_width=True)

# ── Target Distribution ────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Target Variable Distribution</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    target_counts = df["target"].value_counts().reset_index()
    target_counts["label"] = target_counts["target"].map({1: "Good Credit (Approved)", 0: "Bad Credit (Denied)"})
    fig = px.pie(target_counts, values="count", names="label",
                 template=PLOT_TEMPLATE, title="Loan Approval Distribution",
                 color_discrete_sequence=[COLORS["secondary"], COLORS["danger"]])
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Class Imbalance Note")
    st.markdown("""
    The dataset has a **70/30 split** (700 good : 300 bad).
    
    This is a common issue in credit datasets — most applicants are creditworthy.
    
    **Implications for ML:**
    - Models may be biased toward predicting "good credit" (majority class)
    - Accuracy can be misleading (70% with naive classifier)
    - Use Precision, Recall, F1, and AUC instead of just accuracy
    
    **Fairness implication:**
    - If minority groups are overrepresented in the "bad credit" class 
      (due to historical discrimination), models trained on this data 
      will perpetuate that discrimination.
    """)

# ── Numerical Feature Distributions ───────────────────────────────────────────
st.markdown('<div class="section-hdr">Numerical Feature Distributions</div>', unsafe_allow_html=True)

fig = make_subplots(rows=2, cols=4,
    subplot_titles=NUMERICAL_FEATURES,
    vertical_spacing=0.15, horizontal_spacing=0.08)

for i, feat in enumerate(NUMERICAL_FEATURES):
    row, col = i // 4 + 1, i % 4 + 1
    good = df[df["target"] == 1][feat]
    bad = df[df["target"] == 0][feat]
    fig.add_trace(go.Histogram(x=good, name="Good Credit", opacity=0.7,
        marker_color=COLORS["secondary"], showlegend=(i == 0),
        legendgroup="good", nbinsx=20), row=row, col=col)
    fig.add_trace(go.Histogram(x=bad, name="Bad Credit", opacity=0.7,
        marker_color=COLORS["danger"], showlegend=(i == 0),
        legendgroup="bad", nbinsx=20), row=row, col=col)

fig.update_layout(height=500, template=PLOT_TEMPLATE, barmode="overlay",
    title_text="Distribution of Numerical Features by Credit Outcome",
    title_x=0.5, legend=dict(orientation="h", y=1.05))
st.plotly_chart(fig, use_container_width=True)

# ── Categorical Feature Analysis ───────────────────────────────────────────────
st.markdown('<div class="section-hdr">Categorical Feature Analysis</div>', unsafe_allow_html=True)

selected_cat = st.selectbox("Select Categorical Feature:", CATEGORICAL_FEATURES, index=0)

cat_data = df.groupby([selected_cat, "target"]).size().reset_index(name="count")
cat_data["outcome"] = cat_data["target"].map({1: "Good Credit", 0: "Bad Credit"})
cat_data["pct"] = cat_data.groupby(selected_cat)["count"].transform(lambda x: 100 * x / x.sum())

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(cat_data, x=selected_cat, y="count", color="outcome",
                 template=PLOT_TEMPLATE, barmode="group",
                 title=f"Credit Outcome by {selected_cat}",
                 color_discrete_map={"Good Credit": COLORS["secondary"], "Bad Credit": COLORS["danger"]})
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig2 = px.bar(cat_data, x=selected_cat, y="pct", color="outcome",
                  template=PLOT_TEMPLATE, barmode="stack",
                  title=f"Approval Rate by {selected_cat} (%)",
                  color_discrete_map={"Good Credit": COLORS["secondary"], "Bad Credit": COLORS["danger"]})
    fig2.update_yaxes(title="Percentage (%)")
    st.plotly_chart(fig2, use_container_width=True)

# ── Correlation Heatmap ────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Correlation Heatmap (Numerical Features)</div>', unsafe_allow_html=True)

num_df = df[NUMERICAL_FEATURES + ["target"]].copy()
corr = num_df.corr()

fig = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.columns.tolist(),
    colorscale="RdBu",
    zmid=0,
    text=np.round(corr.values, 2),
    texttemplate="%{text}",
    textfont={"size": 11},
    colorbar=dict(title="Correlation")
))
fig.update_layout(template=PLOT_TEMPLATE, height=450,
    title="Correlation Matrix — Numerical Features + Target",
    title_x=0.5)
st.plotly_chart(fig, use_container_width=True)

with st.expander("📌 Correlation Insights"):
    st.markdown("""
    - **credit_amount ↔ duration**: Strong positive correlation (0.62) — larger loans → longer terms
    - **age ↔ existing_credits**: Older applicants tend to have more prior credits
    - **target ↔ duration**: Longer loan duration correlates with higher default risk
    - **target ↔ credit_amount**: Higher credit amounts associated with slightly higher risk
    """)

# ── Bias Preview by Protected Attributes ──────────────────────────────────────
st.markdown('<div class="section-hdr">⚠️ Bias Preview by Protected Attributes</div>', unsafe_allow_html=True)

st.markdown("""
<div style="background:#2d1a0e; border:1px solid #d97706; border-radius:8px; padding:1rem; margin-bottom:1rem;">
    ⚠️ <strong style="color:#fbbf24;">Bias Alert</strong> — The following analysis reveals 
    <em>statistical disparities</em> in loan approval rates across protected groups (Sex and Age). 
    These disparities in the raw data may lead ML models to perpetuate discriminatory patterns.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Approval Rate by Sex")
    sex_stats = df.groupby("sex_label")["target"].agg(["mean", "count"]).reset_index()
    sex_stats.columns = ["Sex", "Approval Rate", "Count"]
    sex_stats["Approval Rate %"] = (sex_stats["Approval Rate"] * 100).round(1)
    sex_stats["Denial Rate %"] = (100 - sex_stats["Approval Rate %"]).round(1)

    fig = px.bar(sex_stats, x="Sex", y="Approval Rate %",
                 template=PLOT_TEMPLATE,
                 color="Sex", text="Approval Rate %",
                 color_discrete_map={"Male": COLORS["male"], "Female": COLORS["female"]},
                 title="Loan Approval Rate by Sex")
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    diff_sex = abs(sex_stats[sex_stats["Sex"] == "Male"]["Approval Rate %"].values[0] -
                   sex_stats[sex_stats["Sex"] == "Female"]["Approval Rate %"].values[0])
    st.markdown(f"""
    <div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:.8rem;">
        <strong>Approval Rate Gap by Sex: {diff_sex:.1f}%</strong>
        <br><small style="color:#8b949e;">This gap exists in raw data before any ML model is applied.</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("#### Approval Rate by Age Group")
    age_stats = df.groupby("age_group_label")["target"].agg(["mean", "count"]).reset_index()
    age_stats.columns = ["Age Group", "Approval Rate", "Count"]
    age_stats["Approval Rate %"] = (age_stats["Approval Rate"] * 100).round(1)

    fig = px.bar(age_stats, x="Age Group", y="Approval Rate %",
                 template=PLOT_TEMPLATE, color="Age Group",
                 text="Approval Rate %",
                 color_discrete_sequence=[COLORS["warning"], COLORS["primary"]],
                 title="Loan Approval Rate by Age Group")
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    diff_age = abs(age_stats["Approval Rate %"].max() - age_stats["Approval Rate %"].min())
    st.markdown(f"""
    <div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:.8rem;">
        <strong>Approval Rate Gap by Age: {diff_age:.1f}%</strong>
        <br><small style="color:#8b949e;">Younger applicants (≤25) face lower approval rates.</small>
    </div>
    """, unsafe_allow_html=True)

# ── Cross-tabulation ───────────────────────────────────────────────────────────
st.markdown("#### Detailed Statistics by Protected Group")

tab1, tab2 = st.tabs(["By Sex", "By Age Group"])

with tab1:
    detailed_sex = df.groupby(["sex_label", "target"]).size().unstack(fill_value=0)
    detailed_sex.columns = ["Denied (Bad Credit)", "Approved (Good Credit)"]
    detailed_sex["Total"] = detailed_sex.sum(axis=1)
    detailed_sex["Approval Rate %"] = (
        detailed_sex["Approved (Good Credit)"] / detailed_sex["Total"] * 100
    ).round(1)
    st.dataframe(detailed_sex, use_container_width=True)

with tab2:
    detailed_age = df.groupby(["age_group_label", "target"]).size().unstack(fill_value=0)
    detailed_age.columns = ["Denied (Bad Credit)", "Approved (Good Credit)"]
    detailed_age["Total"] = detailed_age.sum(axis=1)
    detailed_age["Approval Rate %"] = (
        detailed_age["Approved (Good Credit)"] / detailed_age["Total"] * 100
    ).round(1)
    st.dataframe(detailed_age, use_container_width=True)

# ── Age Distribution ───────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Age Distribution Analysis</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(df, x="age", color="sex_label",
                       template=PLOT_TEMPLATE, nbins=30,
                       title="Age Distribution by Sex",
                       color_discrete_map={"Male": COLORS["male"], "Female": COLORS["female"]},
                       barmode="overlay", opacity=0.75)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(df, x="sex_label", y="credit_amount", color="sex_label",
                 template=PLOT_TEMPLATE,
                 title="Credit Amount Distribution by Sex",
                 color_discrete_map={"Male": COLORS["male"], "Female": COLORS["female"]})
    st.plotly_chart(fig, use_container_width=True)

st.info("""
**Key EDA Takeaways:**
1. The dataset has a 70/30 good/bad credit split — moderate class imbalance
2. Male applicants receive loans at a higher rate than female applicants in raw data
3. Younger applicants (≤25) have a lower approval rate than older applicants
4. Credit amount and duration are positively correlated
5. These patterns suggest models trained on this data may be biased against women and younger applicants

**Next Step →** Train ML models and quantify exactly how much bias each model exhibits
""")
