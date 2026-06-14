"""
pages/6_Loan_Prediction.py - Interactive Loan Prediction Tool

Form for all 20+ features from the German Credit Dataset.
Predicts loan approval probability and shows:
- Prediction decision with confidence
- Feature importance explanation (SHAP if available, fallback to RF importance)
- Fairness warning based on applicant's protected group
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import (
    load_german_credit_data, train_all_models, predict_single_applicant,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES
)

st.set_page_config(page_title="Loan Prediction - FairLoan AI", page_icon="🔮", layout="wide")

PLOT_TEMPLATE = "plotly_dark"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.section-hdr { font-size:1.4rem; font-weight:600; color:#e6edf3;
    border-bottom:2px solid #4F8EF7; padding-bottom:.4rem; margin:1.5rem 0 1rem; }
.decision-approved { background:linear-gradient(135deg, #052e16, #0f5132);
    border:2px solid #16a34a; border-radius:14px; padding:2rem; text-align:center; }
.decision-denied { background:linear-gradient(135deg, #2d0d0d, #4c1314);
    border:2px solid #dc2626; border-radius:14px; padding:2rem; text-align:center; }
.decision-title { font-size:2rem; font-weight:800; margin:.5rem 0; }
.approved-text { color:#4ade80; }
.denied-text { color:#f87171; }
.warning-bias { background:#2d1a0e; border:1px solid #d97706;
    border-radius:10px; padding:1rem; margin-top:1rem; }
.input-group { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:1rem; margin-bottom:.75rem; }
.input-group h4 { color:#4F8EF7; font-size:.95rem; margin:0 0 .75rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🔮 Loan Prediction Tool")
st.markdown("*Enter applicant details to get an AI-powered loan decision with explanation*")

# ── Load Data & Train Models ───────────────────────────────────────────────────
with st.spinner("Loading trained models..."):
    df = load_german_credit_data()
    train_results = train_all_models(df)

# ── Model Selection ────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    Fill in the applicant's details below. All fields correspond directly to the 
    20 features in the German Credit Dataset. The model will predict loan approval 
    probability and explain which factors were most influential.
    """)
with col2:
    model_choice = st.selectbox(
        "Model:",
        ["Random Forest", "Logistic Regression", "XGBoost"],
        index=0
    )

st.markdown("---")

# ── Input Form ─────────────────────────────────────────────────────────────────
st.markdown("### Applicant Information")

with st.form("loan_prediction_form"):

    # ── Section 1: Account & Credit History ──────────────────────────────────
    st.markdown('<div class="section-hdr">Account & Credit Information</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        status = st.selectbox("Checking Account Status", options=[
            "A11", "A12", "A13", "A14"
        ], format_func=lambda x: {
            "A11": "A11: < 0 DM (overdrawn)",
            "A12": "A12: 0–200 DM",
            "A13": "A13: ≥ 200 DM",
            "A14": "A14: No checking account"
        }[x])

        credit_history = st.selectbox("Credit History", options=[
            "A30", "A31", "A32", "A33", "A34"
        ], format_func=lambda x: {
            "A30": "A30: No credits taken",
            "A31": "A31: All paid back duly",
            "A32": "A32: Existing credits paid back duly",
            "A33": "A33: Delay in paying off",
            "A34": "A34: Critical account"
        }[x])

        savings = st.selectbox("Savings Account/Bonds", options=[
            "A61", "A62", "A63", "A64", "A65"
        ], format_func=lambda x: {
            "A61": "A61: < 100 DM",
            "A62": "A62: 100–500 DM",
            "A63": "A63: 500–1000 DM",
            "A64": "A64: ≥ 1000 DM",
            "A65": "A65: Unknown/No savings"
        }[x])

    with col2:
        duration = st.slider("Loan Duration (months)", 4, 72, 24, 1)
        credit_amount = st.number_input("Credit Amount (DM)", 250, 20000, 2000, 100)
        installment_rate = st.slider("Installment Rate (% of income)", 1, 4, 3, 1)

    with col3:
        purpose = st.selectbox("Purpose of Credit", options=[
            "A40", "A41", "A42", "A43", "A44", "A45", "A46", "A48", "A49", "A410"
        ], format_func=lambda x: {
            "A40": "New car",
            "A41": "Used car",
            "A42": "Furniture/equipment",
            "A43": "Radio/television",
            "A44": "Domestic appliances",
            "A45": "Repairs",
            "A46": "Education",
            "A48": "Retraining",
            "A49": "Business",
            "A410": "Others"
        }[x])

        employment = st.selectbox("Employment Duration", options=[
            "A71", "A72", "A73", "A74", "A75"
        ], format_func=lambda x: {
            "A71": "A71: Unemployed",
            "A72": "A72: < 1 year",
            "A73": "A73: 1–4 years",
            "A74": "A74: 4–7 years",
            "A75": "A75: ≥ 7 years"
        }[x])

    # ── Section 2: Personal Information ───────────────────────────────────────
    st.markdown('<div class="section-hdr">Personal Information</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        personal_status = st.selectbox("Personal Status & Sex", options=[
            "A91", "A92", "A93", "A94", "A95"
        ], format_func=lambda x: {
            "A91": "A91: Male — divorced/separated",
            "A92": "A92: Female — divorced/separated/married",
            "A93": "A93: Male — single",
            "A94": "A94: Male — married/widowed",
            "A95": "A95: Female — single"
        }[x])

        age = st.slider("Age (years)", 18, 75, 35, 1)

        foreign_worker = st.selectbox("Foreign Worker", options=["A201", "A202"],
            format_func=lambda x: {"A201": "Yes", "A202": "No"}[x])

    with col2:
        other_debtors = st.selectbox("Other Debtors/Guarantors", options=["A101", "A102", "A103"],
            format_func=lambda x: {"A101": "None", "A102": "Co-applicant", "A103": "Guarantor"}[x])

        residence_since = st.slider("Present Residence Since (years)", 1, 4, 2, 1)

        property = st.selectbox("Property", options=["A121", "A122", "A123", "A124"],
            format_func=lambda x: {
                "A121": "Real estate",
                "A122": "Building society savings / life insurance",
                "A123": "Car (not A121)",
                "A124": "Unknown / no property"
            }[x])

    with col3:
        telephone = st.selectbox("Telephone", options=["A191", "A192"],
            format_func=lambda x: {"A191": "None", "A192": "Yes, registered under customer name"}[x])

        housing = st.selectbox("Housing", options=["A151", "A152", "A153"],
            format_func=lambda x: {"A151": "Free", "A152": "Own", "A153": "For free"}[x])

        job = st.selectbox("Job", options=["A171", "A172", "A173", "A174"],
            format_func=lambda x: {
                "A171": "Unemployed / unskilled — non-resident",
                "A172": "Unskilled — resident",
                "A173": "Skilled employee / official",
                "A174": "Management / highly qualified"
            }[x])

    # ── Section 3: Financial Details ───────────────────────────────────────────
    st.markdown('<div class="section-hdr">Financial Details</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        other_installment = st.selectbox("Other Installment Plans", options=["A141", "A142", "A143"],
            format_func=lambda x: {"A141": "Bank", "A142": "Stores", "A143": "None"}[x])

    with col2:
        existing_credits = st.slider("Number of Existing Credits at This Bank", 1, 4, 1, 1)
        liable_people = st.selectbox("Number of Dependents", options=[1, 2],
            format_func=lambda x: f"{x} person(s)")

    submit_btn = st.form_submit_button("🔮 Predict Loan Decision", type="primary", use_container_width=True)

# ── Make Prediction ────────────────────────────────────────────────────────────
if submit_btn:
    feature_dict = {
        "status": status,
        "duration": duration,
        "credit_history": credit_history,
        "purpose": purpose,
        "credit_amount": credit_amount,
        "savings": savings,
        "employment": employment,
        "installment_rate": installment_rate,
        "personal_status": personal_status,
        "other_debtors": other_debtors,
        "residence_since": residence_since,
        "property": property,
        "age": age,
        "other_installment": other_installment,
        "housing": housing,
        "existing_credits": existing_credits,
        "job": job,
        "liable_people": liable_people,
        "telephone": telephone,
        "foreign_worker": foreign_worker,
    }

    with st.spinner("Computing loan decision..."):
        result = predict_single_applicant(feature_dict, model_choice)

    if "error" in result:
        st.error(f"Prediction error: {result['error']}")
    else:
        st.markdown("---")
        st.markdown("## Prediction Result")

        col1, col2 = st.columns([2, 3])

        with col1:
            if result["prediction"] == 1:
                probability_pct = result["probability"] * 100
                st.markdown(f"""
                <div class="decision-approved">
                    <div style="font-size:3rem;">✅</div>
                    <div class="decision-title approved-text">APPROVED</div>
                    <div style="color:#86efac; font-size:1.2rem; margin:.5rem 0;">
                        Approval Confidence: {probability_pct:.1f}%
                    </div>
                    <div style="color:#4ade80; font-size:.9rem; margin-top:.5rem;">
                        Model: {model_choice}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                probability_pct = (1 - result["probability"]) * 100
                st.markdown(f"""
                <div class="decision-denied">
                    <div style="font-size:3rem;">❌</div>
                    <div class="decision-title denied-text">DENIED</div>
                    <div style="color:#fca5a5; font-size:1.2rem; margin:.5rem 0;">
                        Denial Confidence: {probability_pct:.1f}%
                    </div>
                    <div style="color:#f87171; font-size:.9rem; margin-top:.5rem;">
                        Model: {model_choice}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["probability"] * 100,
                title={"text": "Approval Probability (%)", "font": {"color": "#e6edf3"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                    "bar": {"color": "#4F8EF7"},
                    "bgcolor": "#161b22",
                    "bordercolor": "#30363d",
                    "steps": [
                        {"range": [0, 40], "color": "#2d0d0d"},
                        {"range": [40, 60], "color": "#2d1a0e"},
                        {"range": [60, 100], "color": "#052e16"}
                    ],
                    "threshold": {
                        "line": {"color": "#ffffff", "width": 2},
                        "thickness": 0.75,
                        "value": 50
                    }
                },
                number={"font": {"color": "#e6edf3"}, "suffix": "%"}
            ))
            fig_gauge.update_layout(
                template=PLOT_TEMPLATE, height=250,
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            # ── Feature Importance Explanation ────────────────────────────────
            st.markdown("#### Feature Importance Explanation")
            st.markdown("""
            *Which features most influenced this decision?*
            (Using Random Forest feature importances as SHAP approximation)
            """)

            try:
                import joblib
                preprocessor = joblib.load("models/preprocessor.pkl")
                rf_model = joblib.load("models/random_forest.pkl")

                input_df = pd.DataFrame([feature_dict])[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
                X_processed = preprocessor.transform(input_df)

                from utils import get_feature_names_after_preprocessing
                feature_names = get_feature_names_after_preprocessing(preprocessor)

                importances = rf_model.feature_importances_

                if len(feature_names) == len(importances):
                    importance_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": importances
                    }).sort_values("Importance", ascending=True).tail(15)

                    fig_imp = px.bar(
                        importance_df, x="Importance", y="Feature", orientation="h",
                        template=PLOT_TEMPLATE,
                        title="Top Features for Decision (RF Importance)",
                        color="Importance",
                        color_continuous_scale="Blues"
                    )
                    fig_imp.update_layout(height=380, showlegend=False)
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.warning("Feature names don't match model dimensions.")

            except Exception as e:
                st.warning(f"Could not load model for explanation: {str(e)}")
                st.info("Train models on the Model Training page first.")

        # ── Fairness Warning ────────────────────────────────────────────────────
        st.markdown("#### Fairness Assessment")

        # Derive protected attributes from input
        sex_derived = 0 if personal_status in ["A92", "A95"] else 1  # 0=female, 1=male
        age_group = 1 if age > 25 else 0

        sex_label = "Male" if sex_derived == 1 else "Female"
        age_label = "Age >25 (majority)" if age_group == 1 else "Age ≤25 (minority)"

        # Compute group-level bias statistics from trained models
        y_test = train_results["y_test"]
        sens_test_sex = train_results["sensitive_test"]["sex"]
        sens_test_age = train_results["sensitive_test"]["age_group"]
        rf_pred = train_results["predictions"][model_choice]

        # Compute group approval rates
        sex_approval = {}
        for s_val in [0, 1]:
            mask = (sens_test_sex == s_val)
            if mask.sum() > 0:
                sex_approval[s_val] = rf_pred[mask].mean()

        age_approval = {}
        for a_val in [0, 1]:
            mask = (sens_test_age == a_val)
            if mask.sum() > 0:
                age_approval[a_val] = rf_pred[mask].mean()

        col1, col2 = st.columns(2)

        with col1:
            sex_group_rate = sex_approval.get(sex_derived, 0.5)
            other_sex_rate = sex_approval.get(1 - sex_derived, 0.5)
            sex_gap = abs(sex_group_rate - other_sex_rate)

            st.markdown(f"""
            <div class="{'warning-bias' if sex_gap > 0.05 else 'input-group'}">
                <h4>Sex-based Fairness ({sex_label})</h4>
                <p>Your group approval rate: <strong>{sex_group_rate*100:.1f}%</strong></p>
                <p>Other sex approval rate: <strong>{other_sex_rate*100:.1f}%</strong></p>
                <p>Disparity gap: <strong style="color:{'#fbbf24' if sex_gap > 0.05 else '#34d399'};">{sex_gap*100:.1f}%</strong></p>
                {'<p style="color:#fbbf24;">⚠️ This model shows notable sex-based disparity. '
                'If denied, your gender may have contributed to the decision.</p>'
                if sex_gap > 0.05 else
                '<p style="color:#34d399;">✅ Minimal sex-based disparity detected for this model.</p>'}
            </div>
            """, unsafe_allow_html=True)

        with col2:
            age_group_rate = age_approval.get(age_group, 0.5)
            other_age_rate = age_approval.get(1 - age_group, 0.5)
            age_gap = abs(age_group_rate - other_age_rate)

            st.markdown(f"""
            <div class="{'warning-bias' if age_gap > 0.05 else 'input-group'}">
                <h4>Age-based Fairness ({age_label})</h4>
                <p>Your age group approval rate: <strong>{age_group_rate*100:.1f}%</strong></p>
                <p>Other age group approval rate: <strong>{other_age_rate*100:.1f}%</strong></p>
                <p>Disparity gap: <strong style="color:{'#fbbf24' if age_gap > 0.05 else '#34d399'};">{age_gap*100:.1f}%</strong></p>
                {'<p style="color:#fbbf24;">⚠️ This model shows age-based disparity. '
                'Younger applicants are statistically disadvantaged.</p>'
                if age_gap > 0.05 else
                '<p style="color:#34d399;">✅ Minimal age-based disparity detected for this model.</p>'}
            </div>
            """, unsafe_allow_html=True)

        # ── SHAP Explanation (if available) ────────────────────────────────────
        st.markdown("#### SHAP Explanation (if available)")

        try:
            import shap
            import joblib
            preprocessor = joblib.load("models/preprocessor.pkl")
            rf_model = joblib.load("models/random_forest.pkl")

            input_df = pd.DataFrame([feature_dict])[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
            X_proc = preprocessor.transform(input_df)

            # Use TreeExplainer for tree-based models (fast)
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_proc)

            from utils import get_feature_names_after_preprocessing
            feature_names = get_feature_names_after_preprocessing(preprocessor)

            if isinstance(shap_values, list):
                shap_for_positive = shap_values[1][0]
            else:
                shap_for_positive = shap_values[0]

            if len(feature_names) == len(shap_for_positive):
                shap_df = pd.DataFrame({
                    "Feature": feature_names,
                    "SHAP Value": shap_for_positive
                }).sort_values("SHAP Value", key=abs, ascending=True).tail(15)

                shap_df["Direction"] = shap_df["SHAP Value"].apply(
                    lambda x: "Increases Approval ↑" if x > 0 else "Decreases Approval ↓"
                )

                fig_shap = px.bar(
                    shap_df, x="SHAP Value", y="Feature", orientation="h",
                    color="Direction",
                    color_discrete_map={
                        "Increases Approval ↑": "#34D399",
                        "Decreases Approval ↓": "#F87171"
                    },
                    template=PLOT_TEMPLATE,
                    title="SHAP Values — Feature Contribution to Approval Probability"
                )
                fig_shap.update_layout(height=420)
                st.plotly_chart(fig_shap, use_container_width=True)

                st.markdown("""
                > **SHAP Interpretation:** Green bars push the prediction towards approval.
                > Red bars push towards denial. The magnitude shows how much each feature contributed.
                > This is the applicant's right to explanation under EU AI Act Article 13.
                """)

        except ImportError:
            st.info("SHAP library not installed. Showing feature importances instead (see above).")
        except Exception as e:
            st.info(f"SHAP explanation unavailable: {str(e)}")

        # ── Regulatory Notice ───────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:8px; padding:1rem;">
            <h4 style="color:#e6edf3; margin:0 0 .5rem;">Regulatory Notice</h4>
            <p style="color:#8b949e; font-size:.85rem; margin:0;">
                Under the <strong style="color:#c9d1d9;">EU AI Act (2024)</strong> and 
                <strong style="color:#c9d1d9;">RBI Digital Lending Guidelines (2022)</strong>, 
                applicants have the right to:
                (1) Know that an automated system was used in the decision,
                (2) Receive a meaningful explanation of the factors influencing the decision,
                (3) Request human review of the decision.
                This tool provides (1) and (2). For (3), contact the lending institution.
            </p>
        </div>
        """, unsafe_allow_html=True)
