"""
pages/1_Home.py - Home Page

Project overview, abstract, objectives, regulatory context,
and team/college information.

Key topics covered:
- Why fairness in lending matters
- EU AI Act (2024) requirements for high-risk AI systems
- RBI (Reserve Bank of India) guidelines on algorithmic fairness
- Project objectives and scope
"""

import streamlit as st

st.set_page_config(
    page_title="Home - FairLoan AI",
    page_icon="🏠",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero { background: linear-gradient(135deg, #1a237e 0%, #283593 40%, #1565c0 100%);
    border-radius: 16px; padding: 2.5rem; margin-bottom: 1.5rem;
    border: 1px solid #1e3a8a; box-shadow: 0 8px 32px rgba(79,142,247,.15); }
.hero h1 { font-size: 2.4rem; font-weight: 800; color: #fff; margin: 0 0 .5rem; }
.hero p { color: #93c5fd; font-size: 1.05rem; }

.card { background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 1.4rem; margin-bottom: 1rem; }
.card h3 { color: #4F8EF7; margin: 0 0 .75rem; font-size: 1.1rem; }
.card p, .card li { color: #c9d1d9; font-size: .9rem; line-height: 1.6; }

.tag { display: inline-block; background: #1d4ed8; color: #bfdbfe;
    border-radius: 20px; padding: .2rem .7rem; font-size: .78rem; margin: .15rem; }
.tag-green { background: #14532d; color: #86efac; }
.tag-orange { background: #713f12; color: #fde047; }
.tag-purple { background: #4c1d95; color: #d8b4fe; }

.reg-box { background: #0d2340; border-left: 4px solid #3b82f6;
    border-radius: 8px; padding: 1rem 1.2rem; margin: .75rem 0; }
.reg-box h4 { color: #60a5fa; margin: 0 0 .4rem; font-size: 1rem; }
.reg-box p { color: #93c5fd; font-size: .88rem; margin: 0; }

.team-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 1rem; text-align: center; }
.team-card .avatar { font-size: 2.5rem; }
.team-card h4 { color: #e6edf3; margin: .4rem 0 .15rem; font-size: .95rem; }
.team-card p { color: #8b949e; font-size: .8rem; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>⚖️ FairLoan AI</h1>
    <h2 style="color:#bfdbfe; font-size:1.3rem; font-weight:500; margin:.3rem 0 .75rem;">
        Fairness &amp; Bias Mitigation in Loan Approval using Machine Learning
    </h2>
    <p>
        A comprehensive final-year project exploring how algorithmic bias affects 
        credit decisions — and how fairness-aware ML can fix it.
    </p>
    <div style="margin-top:1rem;">
        <span class="tag">Python</span>
        <span class="tag">scikit-learn</span>
        <span class="tag">XGBoost</span>
        <span class="tag">fairlearn</span>
        <span class="tag">SHAP</span>
        <span class="tag">Streamlit</span>
        <span class="tag tag-green">German Credit Dataset</span>
        <span class="tag tag-orange">EU AI Act</span>
        <span class="tag tag-purple">RBI Guidelines</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Abstract ──────────────────────────────────────────────────────────────────
st.markdown("## Abstract")
st.markdown("""
<div class="card">
    <p>
    Machine learning models are increasingly used in financial institutions to automate 
    loan approval decisions. While these systems improve efficiency, they can 
    <strong style="color:#fca5a5;">perpetuate and amplify historical biases</strong> 
    present in training data — leading to discriminatory outcomes for protected groups 
    such as women, minorities, and younger applicants.
    </p>
    <p>
    This project investigates fairness and bias in loan approval using the 
    <strong style="color:#60a5fa;">UCI German Credit Dataset</strong> (1,000 real loan applications). 
    We train three baseline classifiers (Logistic Regression, Random Forest, XGBoost), 
    measure bias using industry-standard fairness metrics (Demographic Parity, Equalized Odds, 
    Equal Opportunity), and apply three categories of bias mitigation techniques.
    </p>
    <p>
    Our results demonstrate that fairness-aware ML can significantly reduce bias with 
    minimal accuracy trade-off, providing actionable insights for financial institutions 
    seeking to comply with emerging AI regulations.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Objectives ────────────────────────────────────────────────────────────────
st.markdown("## Objectives")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <h3>Primary Objectives</h3>
        <ul>
            <li>Analyse bias patterns in loan approval ML models using protected attributes (Sex, Age)</li>
            <li>Train and evaluate three baseline ML classifiers on German Credit Dataset</li>
            <li>Quantify bias using Demographic Parity, Equalized Odds, and Equal Opportunity metrics</li>
            <li>Apply and compare three bias mitigation techniques across pre/in/post-processing stages</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>Secondary Objectives</h3>
        <ul>
            <li>Provide SHAP-based explanations for individual loan decisions</li>
            <li>Build an interactive loan prediction tool with fairness warnings</li>
            <li>Demonstrate the fairness-accuracy tradeoff in financial ML</li>
            <li>Generate exportable fairness reports for documentation and compliance</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ── Why Fairness Matters ───────────────────────────────────────────────────────
st.markdown("## Why Fairness in Lending Matters")

tab1, tab2, tab3, tab4 = st.tabs([
    "⚠️ The Problem",
    "🇪🇺 EU AI Act",
    "🇮🇳 RBI Guidelines",
    "📚 Research Evidence"
])

with tab1:
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        ### Algorithmic Discrimination in Credit
        
        When ML models learn from historical lending data, they inherit past discrimination patterns:
        
        - **Training Data Bias**: Historical data reflects past human decisions that were biased
        - **Feature Proxies**: Zip code, job type, or education can serve as proxies for race/gender
        - **Feedback Loops**: Biased models → biased decisions → biased future training data
        - **Opacity Problem**: Black-box models make discrimination hard to detect or challenge
        
        **Real-world impact:**
        - Women systematically receive smaller credit limits for identical profiles [Apple Card, 2019]
        - ZIP-code based features can encode racial discrimination (redlining)  
        - Automated decisions affect millions with little recourse for applicants
        """)
    with col2:
        st.markdown("""
        ### Protected Attributes
        
        The German Credit Dataset allows analysis of:
        
        **Sex (derived from personal status)**
        - Male: divorced, single, married/widowed
        - Female: divorced, single
        
        **Age Group**
        - Age > 25 (majority group)
        - Age ≤ 25 (younger applicants)
        
        These are called **sensitive** or **protected** attributes because using 
        them directly or through proxies to make decisions is ethically problematic 
        and in many jurisdictions, illegal.
        """)

with tab2:
    st.markdown("""
    <div class="reg-box">
        <h4>EU Artificial Intelligence Act (2024)</h4>
        <p>Enacted: August 1, 2024 — First comprehensive AI regulation globally</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### High-Risk AI Classification
        
        The EU AI Act classifies **credit scoring and loan approval systems as HIGH-RISK AI** 
        (Annex III, Point 5b). This means:
        
        **Mandatory Requirements:**
        - ✅ Risk management system throughout AI lifecycle
        - ✅ Data governance & bias testing before deployment
        - ✅ Technical documentation & transparency
        - ✅ Human oversight mechanisms
        - ✅ Accuracy, robustness, and cybersecurity standards
        - ✅ Registration in EU database
        """)
    with col2:
        st.markdown("""
        ### Fairness Requirements
        
        **Article 10 (Data & Governance):**
        "Training, validation and testing data sets shall be relevant, representative, 
        free of errors and complete... They shall have appropriate statistical properties, 
        including, where applicable, as regards the persons or groups of persons in relation 
        to whom the high-risk AI system is intended to be used."
        
        **Article 9 (Risk Management):**
        Requires identification and mitigation of risks related to discrimination, 
        including testing across demographic groups.
        
        **Article 13 (Transparency):**
        Applicants must receive meaningful explanations for automated decisions.
        
        **Penalties:** Up to €30M or 6% of global annual turnover
        """)

with tab3:
    st.markdown("""
    <div class="reg-box">
        <h4>Reserve Bank of India (RBI) Guidelines on Algorithmic Credit</h4>
        <p>RBI has issued multiple circulars addressing AI/ML in financial services</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Key RBI Directives
        
        **Fair Lending Practices Circular (2023):**
        - Lenders must ensure AI-driven decisions are fair and non-discriminatory
        - Prohibited from using protected characteristics (gender, religion, caste) 
          directly or as proxies
        - Must provide reasons for credit rejection to applicants
        
        **Digital Lending Guidelines (2022):**
        - Algorithmic systems must be auditable
        - Credit decisions must be explainable
        - Board-approved policies for AI use in credit
        """)
    with col2:
        st.markdown("""
        ### Consumer Protection Focus
        
        **Grievance Redressal:**
        - Customers must be able to appeal automated credit decisions
        - Designated officer for credit-related complaints
        
        **Data Privacy (DPDP Act 2023):**
        - Consent requirements for using personal data in credit models
        - Right to know about automated decision-making
        
        **Regulatory Expectations:**
        - Model validation including bias testing
        - Regular monitoring for fairness drift
        - Explainability documentation
        """)

with tab4:
    st.markdown("""
    ### Seminal Research in Algorithmic Fairness
    
    | Paper | Authors | Key Contribution |
    |-------|---------|-----------------|
    | Equality of Opportunity in Supervised Learning | Hardt et al. (2016) | Defined Equal Opportunity & Equalized Odds fairness criteria |
    | Fairness through awareness | Dwork et al. (2012) | Individual fairness: similar individuals → similar treatment |
    | A Reductions Approach to Fair Classification | Agarwal et al. (2018) | Exponentiated Gradient algorithm for fairness constraints |
    | Data Preprocessing for Classification without Discrimination | Kamiran & Calders (2012) | Reweighing pre-processing technique |
    | COMPAS Racial Bias Study | Angwin et al. (2016) | Demonstrated racial bias in criminal justice AI (ProPublica) |
    | Dissecting Racial Bias in Credit Scoring | Blattner et al. (2021) | Quantified bias in FICO credit scores across racial groups |
    
    > **Key Insight**: There is no single definition of "fairness" — different metrics 
    > capture different aspects, and some are mathematically incompatible 
    > (Chouldechova, 2017). The choice of fairness criterion must reflect 
    > the application's ethical priorities.
    """)

# ── Fairness Metrics Overview ─────────────────────────────────────────────────
st.markdown("## Fairness Metrics Used in This Project")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="card">
        <h3>Demographic Parity Difference</h3>
        <p><strong style="color:#fbbf24;">DPD = P(ŷ=1|A=a) − P(ŷ=1|A=b)</strong></p>
        <p>Measures the difference in approval rates between protected groups. 
        A model is fair if both groups receive loans at the same rate, 
        regardless of true creditworthiness.</p>
        <p><em>Also called: Statistical Parity</em></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>Equalized Odds Difference</h3>
        <p><strong style="color:#fbbf24;">EOD = max(ΔTPR, ΔFPR)</strong></p>
        <p>Requires equal True Positive Rates AND False Positive Rates across 
        groups. Considers both beneficial outcomes (approved when creditworthy) 
        and harmful outcomes (denied when creditworthy).</p>
        <p><em>Strictest standard: satisfies both TPR & FPR equality</em></p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <h3>Equal Opportunity Difference</h3>
        <p><strong style="color:#fbbf24;">EqOppD = |TPR_a − TPR_b|</strong></p>
        <p>Focuses only on True Positive Rate equality — ensuring creditworthy 
        applicants are approved at the same rate regardless of group membership. 
        Relaxation of Equalized Odds.</p>
        <p><em>Focuses on beneficial outcomes only</em></p>
    </div>
    """, unsafe_allow_html=True)

# ── Project Architecture ───────────────────────────────────────────────────────
st.markdown("## Project Architecture")
st.code("""
FairLoan AI Project Structure
═══════════════════════════════════════════════════════════
app.py                    Main Streamlit entry point
├── pages/
│   ├── 1_Home.py         Project overview & regulatory context
│   ├── 2_EDA.py          Exploratory Data Analysis
│   ├── 3_Model_Training.py    Baseline ML model training
│   ├── 4_Fairness_Analysis.py  Fairness metrics computation
│   ├── 5_Bias_Mitigation.py    Mitigation techniques comparison
│   ├── 6_Loan_Prediction.py    Interactive prediction tool
│   └── 7_Results_Report.py     Dashboard & report export
├── utils.py              Shared utilities:
│   ├── Data loading & preprocessing
│   ├── Model training pipeline
│   ├── Fairness metric computation (fairlearn)
│   ├── Bias mitigation techniques
│   └── SHAP explainability
├── data/
│   └── german_credit.csv       UCI German Credit Dataset (auto-downloaded)
├── models/
│   ├── logistic_regression.pkl  Saved trained models
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── preprocessor.pkl
├── .streamlit/
│   └── config.toml             Theme & server config
└── requirements.txt            Python dependencies
═══════════════════════════════════════════════════════════

Data Flow:
  German Credit Data → EDA → Preprocessing Pipeline
    → Model Training → Fairness Analysis → Bias Mitigation
      → Loan Prediction Tool → Results Export
""", language="text")

# ── Team Information ──────────────────────────────────────────────────────────