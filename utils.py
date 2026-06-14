"""
utils.py - Shared utility functions for FairLoan AI

This module handles:
- Dataset downloading and loading (UCI German Credit Dataset)
- Feature engineering (deriving Sex and Age group as protected attributes)
- Preprocessing pipelines (one-hot encoding + scaling)
- Model training (Logistic Regression, Random Forest, XGBoost)
- Fairness metric computation using fairlearn
- Bias mitigation techniques (reweighing, Exponentiated Gradient, Threshold Optimizer)
"""

import os
import io
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
from typing import Tuple, Dict, Any

# ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve
)

# XGBoost
from xgboost import XGBClassifier

# Fairness library
# fairlearn provides tools to measure and mitigate algorithmic bias
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    true_positive_rate
)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer

# ─── Dataset Configuration ──────────────────────────────────────────────────

# UCI German Credit Dataset URL (space-separated format)
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# The German Credit Dataset has 20 attributes + 1 class label
# These are the official column names for all 21 columns
COLUMN_NAMES = [
    "status",           # 1. Status of existing checking account
    "duration",         # 2. Duration in months (numerical)
    "credit_history",   # 3. Credit history
    "purpose",          # 4. Purpose of credit
    "credit_amount",    # 5. Credit amount (numerical)
    "savings",          # 6. Savings account/bonds
    "employment",       # 7. Present employment since
    "installment_rate", # 8. Installment rate as % of disposable income (numerical)
    "personal_status",  # 9. Personal status and sex (encodes BOTH sex and marital status)
    "other_debtors",    # 10. Other debtors/guarantors
    "residence_since",  # 11. Present residence since (numerical)
    "property",         # 12. Property
    "age",              # 13. Age in years (numerical)
    "other_installment",# 14. Other installment plans
    "housing",          # 15. Housing
    "existing_credits", # 16. Number of existing credits at this bank (numerical)
    "job",              # 17. Job
    "liable_people",    # 18. Number of people being liable to provide maintenance (numerical)
    "telephone",        # 19. Telephone
    "foreign_worker",   # 20. Foreign worker
    "class",            # 21. Credit class (1=good, 2=bad)
]

# Categorical features for one-hot encoding
CATEGORICAL_FEATURES = [
    "status", "credit_history", "purpose", "savings", "employment",
    "personal_status", "other_debtors", "property", "other_installment",
    "housing", "job", "telephone", "foreign_worker"
]

# Numerical features for scaling
NUMERICAL_FEATURES = [
    "duration", "credit_amount", "installment_rate", "residence_since",
    "age", "existing_credits", "liable_people"
]

# ─── Data Loading ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_german_credit_data() -> pd.DataFrame:
    """
    Download and load the UCI German Credit Dataset.

    The dataset contains 1000 loan applications with 20 features.
    Class 1 = Good credit (approved), Class 2 = Bad credit (denied).

    Protected attributes to be derived:
    - Sex: from 'personal_status' column (A91/A93/A94 = male, A92/A95 = female)
    - Age group: from 'age' column (>25 = older group, <=25 = younger group)

    Returns:
        pd.DataFrame with all columns + derived 'sex' and 'age_group' columns
    """
    data_path = "data/german_credit.csv"
    os.makedirs("data", exist_ok=True)

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        try:
            response = requests.get(DATA_URL, timeout=30)
            response.raise_for_status()
            df = pd.read_csv(
                io.StringIO(response.text),
                sep=r'\s+',
                header=None,
                names=COLUMN_NAMES
            )
            df.to_csv(data_path, index=False)
        except Exception as e:
            st.error(f"Failed to download dataset: {e}")
            return None

    # ── Derive Protected Attributes ──────────────────────────────────────────
    # personal_status codes:
    #   A91 = male : divorced/separated
    #   A92 = female : divorced/separated/married  ← female
    #   A93 = male : single                         ← male
    #   A94 = male : married/widowed
    #   A95 = female : single                       ← female
    df["sex"] = df["personal_status"].apply(
        lambda x: 0 if x in ["A92", "A95"] else 1  # 0=female, 1=male
    )
    df["sex_label"] = df["sex"].map({0: "Female", 1: "Male"})

    # Age group: >25 is the majority group (1), <=25 is the minority (0)
    df["age_group"] = (df["age"] > 25).astype(int)  # 1 if >25, 0 if <=25
    df["age_group_label"] = df["age_group"].map({0: "Age ≤25", 1: "Age >25"})

    # Convert class: 1 (good) → 1, 2 (bad) → 0
    # This makes 1 = approved, 0 = denied (standard binary classification)
    df["target"] = (df["class"] == 1).astype(int)

    return df


def get_feature_target_split(df: pd.DataFrame, drop_sensitive: bool = False):
    """
    Split dataframe into features (X) and target (y).

    Args:
        df: Full dataset
        drop_sensitive: If True, drop sex and personal_status columns
                        (used for some fairness evaluations)

    Returns:
        X: Feature DataFrame
        y: Target Series (1=approved, 0=denied)
    """
    drop_cols = ["class", "target", "sex", "sex_label", "age_group_label"]
    if drop_sensitive:
        drop_cols += ["personal_status", "age_group"]

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df["target"].copy()
    return X, y


# ─── Preprocessing Pipeline ──────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer that:
    - One-hot encodes categorical features (handles unknown categories)
    - Standard-scales numerical features

    This is a standard preprocessing approach for mixed-type tabular data.

    Returns:
        ColumnTransformer object (not yet fitted)
    """
    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    numerical_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ("num", numerical_transformer, NUMERICAL_FEATURES)
    ])

    return preprocessor


def get_feature_names_after_preprocessing(preprocessor: ColumnTransformer) -> list:
    """Get human-readable feature names after one-hot encoding."""
    try:
        cat_encoder = preprocessor.named_transformers_["cat"]["onehot"]
        cat_features = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        num_features = NUMERICAL_FEATURES
        return cat_features + num_features
    except Exception:
        return []


# ─── Model Training ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def train_all_models(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train three baseline ML models on the German Credit Dataset:
    1. Logistic Regression - Linear, interpretable baseline
    2. Random Forest - Ensemble tree model, handles non-linearity
    3. XGBoost - Gradient boosting, typically best raw accuracy

    Note: These are BASELINE models with NO fairness constraints.
    Fairness analysis on these reveals bias before mitigation.

    Returns:
        Dict containing trained models, preprocessor, train/test splits,
        and evaluation metrics for each model.
    """
    os.makedirs("models", exist_ok=True)

    X, y = get_feature_target_split(df)

    # 80/20 train-test split with stratification (maintains class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Also save the sensitive attributes aligned with test set
    sensitive_test = df.loc[X_test.index][["sex", "sex_label", "age_group", "age_group_label"]]

    preprocessor = build_preprocessor()

    models_config = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, C=1.0
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=42, eval_metric="logloss",
            verbosity=0
        )
    }

    results = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "sensitive_test": sensitive_test,
        "preprocessor": preprocessor,
        "models": {},
        "metrics": {},
        "predictions": {},
        "probabilities": {}
    }

    # Fit preprocessor once, transform for all models
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    results["X_train_processed"] = X_train_processed
    results["X_test_processed"] = X_test_processed

    for model_name, model in models_config.items():
        # Train the model
        model.fit(X_train_processed, y_train)

        # Predictions
        y_pred = model.predict(X_test_processed)
        y_prob = model.predict_proba(X_test_processed)[:, 1]

        # Compute standard classification metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        results["models"][model_name] = model
        results["metrics"][model_name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "confusion_matrix": cm,
            "auc": auc,
            "fpr": fpr,
            "tpr": tpr,
        }
        results["predictions"][model_name] = y_pred
        results["probabilities"][model_name] = y_prob

        # Save model + preprocessor to disk for later use in Loan Prediction
        joblib.dump(model, f"models/{model_name.replace(' ', '_').lower()}.pkl")

    joblib.dump(preprocessor, "models/preprocessor.pkl")

    return results


# ─── Fairness Metrics ────────────────────────────────────────────────────────

def compute_fairness_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    sensitive_feature: pd.Series,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Compute fairness metrics using fairlearn's MetricFrame.

    Fairness concepts explained:
    ─────────────────────────────
    1. Demographic Parity Difference (DPD):
       Measures difference in selection rates between groups.
       DPD = P(ŷ=1|A=0) - P(ŷ=1|A=1)
       Perfect fairness: DPD = 0
       Positive DPD means privileged group gets more approvals.

    2. Equalized Odds Difference (EOD):
       Measures max difference in TRUE POSITIVE RATE and FALSE POSITIVE RATE.
       Considers both beneficial (loans given to creditworthy) and
       harmful outcomes (loans denied to creditworthy) simultaneously.
       Perfect fairness: EOD = 0

    3. Equal Opportunity Difference (EqOppD):
       Measures difference in TRUE POSITIVE RATE (recall) between groups.
       Only considers beneficial outcomes: P(ŷ=1|y=1, A=0) vs P(ŷ=1|y=1, A=1)
       Perfect fairness: EqOppD = 0

    Reference: Hardt et al., "Equality of Opportunity in Supervised Learning", NeurIPS 2016

    Args:
        y_true: True labels (1=approved, 0=denied)
        y_pred: Predicted labels
        sensitive_feature: Protected attribute (e.g., sex or age_group)
        model_name: Name of the model (for display)

    Returns:
        Dict of fairness metrics and group-level metrics
    """
    # MetricFrame computes metrics for each group of the sensitive feature
    mf = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": lambda y, ŷ: np.mean(ŷ),
            "true_positive_rate": true_positive_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )

    # Group-level metrics (how each protected group fares)
    group_metrics = mf.by_group

    # Overall fairness metrics
    dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_feature)
    eod = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_feature)

    # Equal opportunity = difference in true positive rates
    tpr_by_group = mf.by_group["true_positive_rate"]
    eq_opp_diff = float(tpr_by_group.max() - tpr_by_group.min())

    return {
        "model_name": model_name,
        "group_metrics": group_metrics,
        "overall_accuracy": accuracy_score(y_true, y_pred),
        "demographic_parity_difference": dpd,
        "equalized_odds_difference": eod,
        "equal_opportunity_difference": eq_opp_diff,
        "selection_rate_by_group": mf.by_group["selection_rate"],
        "tpr_by_group": mf.by_group["true_positive_rate"],
        "accuracy_by_group": mf.by_group["accuracy"],
    }


# ─── Bias Mitigation ─────────────────────────────────────────────────────────

def apply_reweighing(
    X_train: np.ndarray,
    y_train: pd.Series,
    sensitive_train: pd.Series
) -> np.ndarray:
    """
    Pre-processing bias mitigation: Reweighing.

    Reweighing assigns sample weights to training examples to compensate for
    historical bias. Samples from disadvantaged groups that were positively
    labeled are given higher weights, making the classifier "pay more attention"
    to these underrepresented cases.

    This is a PRE-PROCESSING technique - it modifies the training data
    (via sample weights) before model training.

    Reference: Kamiran & Calders, "Data Preprocessing Techniques for
    Classification without Discrimination", KAIS 2012

    Args:
        X_train: Training features (processed)
        y_train: Training labels
        sensitive_train: Protected attribute for training samples

    Returns:
        sample_weights: Array of weights for each training sample
    """
    # Compute sample weights using a reweighing scheme
    # Groups: (sensitive_feature, y_true) pairs
    sensitive_arr = sensitive_train.values
    y_arr = y_train.values
    n = len(y_arr)

    # Compute expected vs. observed frequencies for each group
    groups = list(set(zip(sensitive_arr, y_arr)))
    weights = np.ones(n)

    for sens_val, y_val in groups:
        # Expected probability: P(A=a) * P(Y=y) (if independent)
        p_a = np.mean(sensitive_arr == sens_val)
        p_y = np.mean(y_arr == y_val)
        p_expected = p_a * p_y

        # Observed probability: P(A=a, Y=y)
        mask = (sensitive_arr == sens_val) & (y_arr == y_val)
        p_observed = np.mean(mask)

        if p_observed > 0:
            # Weight = expected / observed
            # If a group is under-represented, weight > 1 (upweight)
            # If a group is over-represented, weight < 1 (downweight)
            w = p_expected / p_observed
            weights[mask] = w

    return weights


@st.cache_data(show_spinner=False)
def run_bias_mitigation(_train_results: Dict, df: pd.DataFrame,
                        protected_attr: str = "sex") -> Dict[str, Any]:
    """
    Apply and evaluate three bias mitigation techniques:

    1. Pre-processing: Reweighing
       - Modifies training sample weights to compensate for historical bias
       - Advantage: Model-agnostic, easy to implement
       - Limitation: May reduce accuracy for majority group

    2. In-processing: Exponentiated Gradient Reduction (fairlearn)
       - Trains classifier with fairness constraints during learning
       - Uses Lagrangian relaxation to minimize loss subject to fairness bounds
       - Reference: Agarwal et al., "A Reductions Approach to Fair Classification", ICML 2018

    3. Post-processing: Threshold Optimizer (fairlearn)
       - Finds group-specific decision thresholds that satisfy fairness constraints
       - Applies different cutoffs for each protected group
       - Reference: Hardt et al., "Equality of Opportunity in Supervised Learning", 2016

    Args:
        _train_results: Results from train_all_models()
        df: Full dataset
        protected_attr: Protected attribute name ("sex" or "age_group")

    Returns:
        Dict with mitigation results for each technique
    """
    X_train = _train_results["X_train"]
    X_test = _train_results["X_test"]
    y_train = _train_results["y_train"]
    y_test = _train_results["y_test"]
    preprocessor = _train_results["preprocessor"]

    X_train_proc = _train_results["X_train_processed"]
    X_test_proc = _train_results["X_test_processed"]

    # Get sensitive attributes aligned with train/test
    sensitive_train = df.loc[X_train.index][protected_attr]
    sensitive_test = df.loc[X_test.index][protected_attr]

    # Use Logistic Regression as the base model for mitigation
    base_model_name = "Logistic Regression"
    baseline_model = _train_results["models"][base_model_name]
    baseline_pred = _train_results["predictions"][base_model_name]
    baseline_metrics = compute_fairness_metrics(
        y_test, baseline_pred, sensitive_test, "Baseline (No Mitigation)"
    )

    mitigation_results = {"baseline": baseline_metrics}

    # ── 1. Pre-processing: Reweighing ────────────────────────────────────────
    try:
        sample_weights = apply_reweighing(X_train_proc, y_train, sensitive_train)
        rw_model = LogisticRegression(max_iter=1000, random_state=42)
        rw_model.fit(X_train_proc, y_train, sample_weight=sample_weights)
        rw_pred = rw_model.predict(X_test_proc)

        rw_metrics = compute_fairness_metrics(
            y_test, rw_pred, sensitive_test, "Reweighing (Pre-processing)"
        )
        mitigation_results["reweighing"] = rw_metrics
    except Exception as e:
        mitigation_results["reweighing"] = {"error": str(e)}

    # ── 2. In-processing: Exponentiated Gradient ─────────────────────────────
    # This uses fairness constraints directly during model training
    try:
        base_estimator = LogisticRegression(max_iter=1000, random_state=42)
        # DemographicParity constraint ensures similar selection rates across groups
        eg_model = ExponentiatedGradient(
            estimator=base_estimator,
            constraints=DemographicParity(),
            eps=0.05,  # fairness tolerance (5%)
            max_iter=50
        )
        eg_model.fit(X_train_proc, y_train, sensitive_features=sensitive_train)
        eg_pred = eg_model.predict(X_test_proc)

        eg_metrics = compute_fairness_metrics(
            y_test, eg_pred, sensitive_test, "Exponentiated Gradient (In-processing)"
        )
        mitigation_results["exp_gradient"] = eg_metrics
    except Exception as e:
        mitigation_results["exp_gradient"] = {"error": str(e)}

    # ── 3. Post-processing: Threshold Optimizer ───────────────────────────────
    # Finds group-specific thresholds after training to achieve fairness
    try:
        base_lr = LogisticRegression(max_iter=1000, random_state=42)
        base_lr.fit(X_train_proc, y_train)

        # ThresholdOptimizer finds optimal thresholds per group
        # to satisfy the chosen fairness constraint
        to_model = ThresholdOptimizer(
            estimator=base_lr,
            constraints="equalized_odds",  # Equalized Odds fairness criterion
            objective="accuracy_score",
            predict_method="predict_proba",
        )
        to_model.fit(X_train_proc, y_train, sensitive_features=sensitive_train)
        to_pred = to_model.predict(
            X_test_proc, sensitive_features=sensitive_test, random_state=42
        )

        to_metrics = compute_fairness_metrics(
            y_test, to_pred, sensitive_test, "Threshold Optimizer (Post-processing)"
        )
        mitigation_results["threshold_opt"] = to_metrics
    except Exception as e:
        mitigation_results["threshold_opt"] = {"error": str(e)}

    return mitigation_results


# ─── SHAP Explainability ─────────────────────────────────────────────────────

def get_shap_values(model, X_processed: np.ndarray, feature_names: list, n_samples: int = 100):
    """
    Compute SHAP (SHapley Additive exPlanations) values.

    SHAP explains individual predictions by assigning each feature a
    contribution score. Based on cooperative game theory (Shapley values).

    SHAP helps answer: "Why did the model predict loan approval/denial
    for THIS specific applicant?"

    Reference: Lundberg & Lee, "A Unified Approach to Interpreting Model
    Predictions", NeurIPS 2017

    Args:
        model: Trained sklearn/XGBoost model
        X_processed: Processed feature matrix
        feature_names: Names of features after preprocessing
        n_samples: Number of background samples for SHAP

    Returns:
        shap_values: SHAP values array
        explainer: SHAP explainer object
    """
    try:
        import shap

        # Use a background dataset for SHAP approximation
        background = X_processed[:min(n_samples, len(X_processed))]

        if hasattr(model, "predict_proba"):
            explainer = shap.Explainer(
                model.predict_proba,
                background,
                feature_names=feature_names,
                output_names=["denied", "approved"]
            )
        else:
            explainer = shap.Explainer(
                model.predict,
                background,
                feature_names=feature_names
            )

        # Compute SHAP for a subset (full computation is slow)
        shap_values = explainer(X_processed[:min(50, len(X_processed))])
        return shap_values, explainer
    except Exception as e:
        return None, None


# ─── Loan Prediction Helper ──────────────────────────────────────────────────

def predict_single_applicant(
    feature_dict: dict,
    model_name: str = "Random Forest"
) -> Dict[str, Any]:
    """
    Make a loan decision for a single applicant.

    Args:
        feature_dict: Dictionary of feature values (all 20 features)
        model_name: Which trained model to use

    Returns:
        Dict with prediction, probability, and fairness warning
    """
    try:
        preprocessor = joblib.load("models/preprocessor.pkl")
        model_file = f"models/{model_name.replace(' ', '_').lower()}.pkl"
        model = joblib.load(model_file)

        # Create DataFrame with correct column order
        input_df = pd.DataFrame([feature_dict])[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
        X_processed = preprocessor.transform(input_df)

        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0][1]

        # Determine if the decision might be biased based on applicant's group
        # This is a simplified fairness warning based on group-level statistics
        sex = feature_dict.get("sex_derived", 1)  # from personal_status
        age = feature_dict.get("age", 30)
        age_group = 1 if age > 25 else 0

        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "decision": "APPROVED" if prediction == 1 else "DENIED",
            "sex": sex,
            "age_group": age_group,
        }
    except Exception as e:
        return {"error": str(e)}
