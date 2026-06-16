"""
app.py - FairLoan AI: Fairness & Bias Mitigation in Loan Approval using ML

Main entry point for the Streamlit multi-page application.

Navigation:
  1. Home           - Project overview, objectives, regulatory context
  2. EDA            - Exploratory Data Analysis of German Credit Dataset
  3. Model Training - Train Logistic Regression, Random Forest, XGBoost
  4. Fairness       - Compute fairness metrics on baseline models
  5. Mitigation     - Apply pre/in/post-processing bias mitigation
  6. Loan Tool      - Interactive loan prediction with SHAP explanations
  7. Report         - Overall comparison dashboard + exports

Run:
    streamlit run app.py
"""

import streamlit as st  # type: ignore[import]

# ── Page Configuration ────────────────────────────────────────────────────────
# Must be the FIRST Streamlit call in the script
st.set_page_config(
    page_title="FairLoan AI - Bias Mitigation in Loan Approval",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/fairlearn/fairlearn",
        "Report a bug": None,
        "About": "FairLoan AI - A Final Year Project on Fairness in ML for Loan Approval"
    }
)

# ── Custom CSS for Professional UI ────────────────────────────────────────────
# Inject custom CSS to override Streamlit defaults for a polished look
CUSTOM_CSS = """
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Typography */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Sidebar Branding */
[data-testid="stSidebarContent"] {
    background: linear-gradient(160deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #30363d;
}

/* Main Content Area */
.main .block-container {
    padding: 1.5rem 2rem 2rem 2rem;
    max-width: 1200px;
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #1a237e 0%, #283593 40%, #1565c0 100%);
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid #1e3a8a;
    box-shadow: 0 8px 32px rgba(79, 142, 247, 0.15);
}

.hero-banner h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.5rem 0;
    line-height: 1.3;
}

.hero-banner p {
    color: #93c5fd;
    font-size: 1.05rem;
    margin: 0;
}

/* Metric Cards */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: all 0.2s ease;
}

.metric-card:hover {
    border-color: #4F8EF7;
    box-shadow: 0 4px 16px rgba(79, 142, 247, 0.1);
}

.metric-card .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #4F8EF7;
}

.metric-card .metric-label {
    font-size: 0.85rem;
    color: #8b949e;
    margin-top: 0.25rem;
}

/* Info Boxes */
.info-box {
    background: #0d2340;
    border: 1px solid #1d4ed8;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}

.warning-box {
    background: #2d1a0e;
    border: 1px solid #d97706;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}

.success-box {
    background: #0d2b1e;
    border: 1px solid #16a34a;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}

.danger-box {
    background: #2d0d0d;
    border: 1px solid #dc2626;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}

/* Section Headers */
.section-header {
    font-size: 1.4rem;
    font-weight: 600;
    color: #e6edf3;
    border-bottom: 2px solid #4F8EF7;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

/* Fairness Score Badge */
.fairness-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}

.fairness-good { background: #14532d; color: #86efac; }
.fairness-medium { background: #713f12; color: #fde047; }
.fairness-poor { background: #7f1d1d; color: #fca5a5; }

/* Sidebar Navigation */
.sidebar-logo {
    font-size: 1.6rem;
    font-weight: 800;
    color: #4F8EF7;
    text-align: center;
    padding: 0.5rem 0;
    letter-spacing: -0.5px;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 0.95rem;
    font-weight: 500;
}

/* Tables */
.dataframe {
    font-size: 0.85rem;
}

/* Custom Toggle */
.theme-toggle {
    cursor: pointer;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    border: 1px solid #30363d;
    background: #21262d;
    color: #e6edf3;
    font-size: 0.85rem;
}

/* Sidebar divider */
.sidebar-divider {
    border-top: 1px solid #30363d;
    margin: 0.75rem 0;
}

/* Footer */
.footer {
    text-align: center;
    color: #8b949e;
    font-size: 0.8rem;
    padding: 1rem 0;
    border-top: 1px solid #21262d;
    margin-top: 2rem;
}

/* Stmetric styling */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem;
}

/* Chart containers */
.js-plotly-plot {
    border-radius: 10px;
    overflow: hidden;
}

/* Progress bar customization */
.stProgress > div > div > div > div {
    background-color: #4F8EF7;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8);
    box-shadow: 0 4px 12px rgba(29, 78, 216, 0.3);
}

/* Download button */
.stDownloadButton > button {
    background: #21262d;
    color: #58a6ff;
    border: 1px solid #30363d;
    border-radius: 8px;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    border-radius: 8px;
}

/* Multiselect */
div[data-baseweb="multi-select"] > div {
    border-radius: 8px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Session State Initialization ──────────────────────────────────────────────
# Persist data across page navigations using session state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "models_trained" not in st.session_state:
    st.session_state.models_trained = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">⚖️ FairLoan AI</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="text-align:center; color:#8b949e; font-size:0.8rem; '
        'margin-bottom:1rem;">Fairness & Bias Mitigation in ML</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Navigation")
    st.markdown("""
    Use the **pages** in the left sidebar to navigate:

    - 🏠 **Home** — Overview & context
    - 📊 **EDA** — Data exploration
    - 🤖 **Model Training** — Train classifiers
    - ⚖️ **Fairness Analysis** — Measure bias
    - 🔧 **Bias Mitigation** — Fix the bias
    - 🔮 **Loan Prediction** — Try it yourself
    - 📋 **Results & Report** — Export & compare
    """)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Project Info")
    st.markdown("""
    **Dataset**: UCI German Credit  
    **Models**: LR, RF, XGBoost  
    **Fairness**: fairlearn library  
    **Explainability**: SHAP  
    """)


# ── Main Page Content ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>⚖️ FairLoan AI</h1>
    <p>Fairness &amp; Bias Mitigation in Loan Approval using Machine Learning</p>
    <p style="margin-top:0.5rem; font-size:0.9rem; color:#bfdbfe;">
        Navigate using the sidebar pages to explore the complete fairness-aware ML pipeline.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Dataset", "1,000 Loans", "UCI German Credit")
with col2:
    st.metric("ML Models", "3 Baseline", "LR · RF · XGBoost")
with col3:
    st.metric("Fairness Metrics", "4 Computed", "DPD · EOD · EqOpp · Acc")
with col4:
    st.metric("Mitigation Techniques", "3 Applied", "Pre · In · Post Processing")

st.markdown("---")

# Quick guide cards
st.markdown("### Getting Started")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Step 1: Explore the Data**
    
    Go to **EDA** to:
    - Load the German Credit Dataset
    - See distributions of features
    - Identify bias in raw data
    - View protected attribute statistics
    """)

with col2:
    st.markdown("""
    **Step 2: Train & Evaluate**
    
    Go to **Model Training** to:
    - Train Logistic Regression, RF, XGBoost
    - View accuracy, precision, recall
    - See confusion matrices & ROC curves
    - Compare model performance
    """)

with col3:
    st.markdown("""
    **Step 3: Measure & Fix Bias**
    
    Go to **Fairness Analysis** then **Bias Mitigation** to:
    - Compute fairness metrics per group
    - Apply reweighing, ExponGrad, Threshold Opt
    - Compare before/after fairness scores
    - Export a full fairness report
    """)
