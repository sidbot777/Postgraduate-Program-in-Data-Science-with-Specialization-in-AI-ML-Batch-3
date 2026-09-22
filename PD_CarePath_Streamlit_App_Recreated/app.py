import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PD Care Pathway Recommendation System",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# CONSTANTS
# ============================================================

MODEL_FILE = Path("decision_tree_final.pkl")
ASSET_FILE = Path("deployment_assets.joblib")

OFF_ITEMS = [
    "speech_off",
    "facial_expression_off",
    "rigidity_neck_off",
    "rigidity_rue_off",
    "rigidity_lue_off",
    "rigidity_rle_off",
    "rigidity_lle_off",
    "finger_tapping_r_off",
    "finger_tapping_l_off",
    "hand_movements_r_off",
    "hand_movements_l_off",
    "pronation_supination_r_off",
    "pronation_supination_l_off",
    "toe_tapping_r_off",
    "toe_tapping_l_off",
    "leg_agility_r_off",
    "leg_agility_l_off",
    "arising_from_chair_off",
    "gait_off",
    "freezing_of_gait_off",
    "postural_stability_off",
    "posture_off",
    "body_bradykinesia_off",
    "postural_tremor_r_off",
    "postural_tremor_l_off",
    "kinetic_tremor_r_off",
    "kinetic_tremor_l_off",
    "rest_tremor_rue_off",
    "rest_tremor_lue_off",
    "rest_tremor_rle_off",
    "rest_tremor_lle_off",
    "rest_tremor_lip_jaw_off",
    "constancy_of_rest_tremor_off",
]

ON_ITEMS = [x.replace("_off", "_on") for x in OFF_ITEMS]

OTHER_FEATURES = [
    "age",
    "disease_duration_years",
    "hoehn_yahr_stage_off",
    "hoehn_yahr_stage_on",
    "dyskinesia",
    "psychiatric_score",
    "psychiatric_present",
    "cognition_moca",
]

PATHWAY_LABELS = {
    "Pharmacological_Management": "Pharmacological Management",
    "Rehab_Referral": "Rehabilitation Referral",
    "Surgical_Referral": "Surgical Referral",
}

# ============================================================
# LOAD MODEL / DEPLOYMENT ASSETS
# ============================================================

@st.cache_resource
def load_deployment():
    model = None
    assets = {}

    if MODEL_FILE.exists():
        model = joblib.load(MODEL_FILE)

    if ASSET_FILE.exists():
        assets = joblib.load(ASSET_FILE)
        if model is None and "model" in assets:
            model = assets["model"]

    return model, assets


model, assets = load_deployment()

# ============================================================
# HEADER
# ============================================================

st.title("🧠 Parkinson's Disease Care Pathway Recommendation System")
st.caption(
    "Machine Learning-Based Care Pathway Classification Using MDS-UPDRS Part III Scores"
)

st.info(
    "Academic prototype using synthetic patient data. "
    "This system is not clinically validated and must not be used "
    "for real patient-care decisions."
)

if model is None:
    st.error(
        "Trained model not found. Please place "
        "`decision_tree_final.pkl` or `deployment_assets.joblib` "
        "in the same folder as `app.py`."
    )
    st.stop()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_derived_features(data):
    """Calculate the same derived features used in the project."""

    off_total = sum(float(data[col]) for col in OFF_ITEMS)
    on_total = sum(float(data[col]) for col in ON_ITEMS)

    delta = off_total - on_total

    if off_total > 0:
        response_pct = (delta / off_total) * 100
    else:
        response_pct = 0.0

    data["updrs3_off_total"] = off_total
    data["updrs3_on_total"] = on_total
    data["on_off_delta"] = delta
    data["levodopa_response_pct"] = response_pct

    return data


def predict_patient(data):
    """Create DataFrame in the model's expected feature order."""

    data = calculate_derived_features(data)

    feature_columns = assets.get("feature_columns")

    if feature_columns is None:
        feature_columns = list(data.keys())

    X = pd.DataFrame([data])

    # Add any missing expected columns as zero.
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_columns]

    prediction = model.predict(X)[0]

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]

    return prediction, probabilities, X


def pathway_name(value):
    return PATHWAY_LABELS.get(value, str(value))


def score_input(label, default=0, help_text=None, key=None):
    return st.number_input(
        label,
        min_value=0,
        max_value=4,
        value=default,
        step=1,
        help=help_text,
        key=key
    )


# ============================================================
# TABS
# ============================================================

tab_prediction, tab_samples, tab_failure, tab_info = st.tabs(
    [
        "Patient Prediction",
        "3 Sample Predictions",
        "Known Failure Case",
        "Project Information",
    ]
)

# ============================================================
# TAB 1 — PATIENT PREDICTION
# ============================================================

with tab_prediction:

    st.subheader("Patient Assessment")

    st.write(
        "Enter the patient's demographic, clinical and MDS-UPDRS Part III "
        "scores. Derived motor features are calculated automatically."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=60,
            step=1
        )

    with col2:
        disease_duration = st.number_input(
            "Disease Duration (years)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5
        )

    with col3:
        moca = st.number_input(
            "MoCA Score",
            min_value=0.0,
            max_value=30.0,
            value=25.0,
            step=1.0
        )

    st.divider()

    # --------------------------------------------------------
    # MDS-UPDRS OFF
    # --------------------------------------------------------

    st.subheader("MDS-UPDRS Part III — OFF State")

    off_values = {}

    columns = st.columns(3)

    for i, item in enumerate(OFF_ITEMS):
        clean_name = item.replace("_off", "").replace("_", " ").title()

        with columns[i % 3]:
            off_values[item] = score_input(clean_name, 0, key=item)

    st.divider()

    # --------------------------------------------------------
    # MDS-UPDRS ON
    # --------------------------------------------------------

    st.subheader("MDS-UPDRS Part III — ON State")

    on_values = {}

    columns = st.columns(3)

    for i, item in enumerate(ON_ITEMS):
        clean_name = item.replace("_on", "").replace("_", " ").title()

        with columns[i % 3]:
            on_values[item] = score_input(clean_name, 0, key=item)

    st.divider()

    # --------------------------------------------------------
    # OTHER CLINICAL FEATURES
    # --------------------------------------------------------

    st.subheader("Additional Clinical Features")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        hy_off = st.number_input(
            "Hoehn & Yahr — OFF",
            min_value=0.0,
            max_value=5.0,
            value=2.0,
            step=0.5
        )

    with col2:
        hy_on = st.number_input(
            "Hoehn & Yahr — ON",
            min_value=0.0,
            max_value=5.0,
            value=2.0,
            step=0.5
        )

    with col3:
        dyskinesia = st.number_input(
            "Dyskinesia",
            min_value=0,
            max_value=4,
            value=0,
            step=1
        )

    with col4:
        psychiatric_score = st.number_input(
            "Psychiatric Score",
            min_value=0.0,
            max_value=4.0,
            value=0.0,
            step=1.0
        )

    psychiatric_present = st.selectbox(
        "Psychiatric Symptoms Present?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    # --------------------------------------------------------
    # BUILD INPUT
    # --------------------------------------------------------

    patient_data = {
        "age": age,
        "disease_duration_years": disease_duration,
        "hoehn_yahr_stage_off": hy_off,
        "hoehn_yahr_stage_on": hy_on,
        "dyskinesia": dyskinesia,
        "psychiatric_score": psychiatric_score,
        "psychiatric_present": psychiatric_present,
        "cognition_moca": moca,
    }

    patient_data.update(off_values)
    patient_data.update(on_values)

    # --------------------------------------------------------
    # CALCULATE DERIVED FEATURES FOR DISPLAY
    # --------------------------------------------------------

    preview_data = calculate_derived_features(patient_data.copy())

    st.subheader("Calculated Motor Features")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "UPDRS III OFF Total",
        f"{preview_data['updrs3_off_total']:.0f}"
    )

    c2.metric(
        "UPDRS III ON Total",
        f"{preview_data['updrs3_on_total']:.0f}"
    )

    c3.metric(
        "ON–OFF Delta",
        f"{preview_data['on_off_delta']:.0f}"
    )

    c4.metric(
        "Levodopa Response",
        f"{preview_data['levodopa_response_pct']:.1f}%"
    )

    st.divider()

    if st.button(
        "🔍 Predict Care Pathway",
        type="primary",
        use_container_width=True
    ):

        prediction, probabilities, X_input = predict_patient(patient_data)

        st.success(
            f"Predicted Care Pathway: **{pathway_name(prediction)}**"
        )

        if probabilities is not None:

            st.subheader("Model Probability Output")

            classes = getattr(
                model,
                "classes_",
                assets.get("classes", [])
            )

            probability_df = pd.DataFrame({
                "Care Pathway": [
                    pathway_name(c) for c in classes
                ],
                "Probability": probabilities
            })

            probability_df["Probability"] = (
                probability_df["Probability"] * 100
            ).round(2)

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                probability_df.set_index("Care Pathway")
            )

# ============================================================
# TAB 2 — THREE SAMPLE PREDICTIONS
# ============================================================

with tab_samples:

    st.subheader("Three Sample Predictions")

    st.write(
        "These examples are taken from the project test data and are "
        "intended to demonstrate how the trained model behaves."
    )

    sample_cases = assets.get("sample_cases", [])

    if not sample_cases:
        st.warning(
            "Sample prediction cases were not found in deployment_assets.joblib."
        )
    else:

        for i, case in enumerate(sample_cases, start=1):

            st.markdown(f"### Sample {i}")

            actual = case.get("actual", "Not available")
            predicted = case.get("predicted", "Not available")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Actual Care Pathway",
                pathway_name(actual)
            )

            c2.metric(
                "Model Prediction",
                pathway_name(predicted)
            )

            if actual == predicted:
                c3.success("Correct prediction")
            else:
                c3.error("Misclassification")

            probabilities = case.get("probabilities")

            classes = case.get(
                "classes",
                getattr(model, "classes_", [])
            )

            if probabilities and len(classes) == len(probabilities):

                probability_df = pd.DataFrame({
                    "Care Pathway": [
                        pathway_name(c) for c in classes
                    ],
                    "Probability (%)": [
                        round(p * 100, 2)
                        for p in probabilities
                    ]
                })

                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True
                )

            st.divider()

# ============================================================
# TAB 3 — KNOWN FAILURE CASE
# ============================================================

with tab_failure:

    st.subheader("Known Model Failure Case")

    st.write(
        "This is an actual misclassification from the held-out test set. "
        "It is included to demonstrate model limitations and error analysis."
    )

    failure = assets.get("known_failure")

    if not failure:
        st.warning(
            "The known failure case was not found in deployment_assets.joblib."
        )
    else:

        actual = failure.get("actual", "Not available")
        predicted = failure.get("predicted", "Not available")

        c1, c2 = st.columns(2)

        c1.metric(
            "Actual Care Pathway",
            pathway_name(actual)
        )

        c2.metric(
            "Predicted Care Pathway",
            pathway_name(predicted)
        )

        st.error(
            "Known failure: the model prediction does not match "
            "the actual held-out test-set label."
        )

        features = failure.get("features", {})

        if features:

            st.subheader("Input Features for Failure Case")

            feature_df = pd.DataFrame(
                list(features.items()),
                columns=["Feature", "Value"]
            )

            st.dataframe(
                feature_df,
                use_container_width=True,
                hide_index=True
            )

# ============================================================
# TAB 4 — PROJECT INFORMATION
# ============================================================

with tab_info:

    st.subheader("Project Overview")

    st.markdown(
        """
### Objective

To develop an interpretable machine learning system that classifies
Parkinson's disease patients into three predefined care pathways using
clinical variables and MDS-UPDRS Part III scores.

### Care Pathways

1. Pharmacological Management
2. Rehabilitation Referral
3. Surgical Referral

### Dataset

- 1,000 synthetic patient records
- 80 variables
- No missing values
- No duplicate records
- Synthetic data generated for academic modelling

### Models

**Baseline:** Logistic Regression

**Final model:** Decision Tree

Final Decision Tree configuration:

- Criterion: entropy
- Maximum depth: 5
- Minimum samples per leaf: 2
- Random state: 42

### Test-set Performance

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression | 73.50% | 67.11% |
| Decision Tree | 93.00% | 87.62% |

The final Decision Tree was selected after controlled hyperparameter
experimentation using the training/validation data and was evaluated
once on the untouched held-out test set.

### Important Limitation

This is an academic prototype using synthetic data. The predictions
are not clinically validated and should not be interpreted as medical
advice or as an actual clinical referral recommendation.
"""
    )

    st.divider()

    st.caption(
        "Academic AI/ML Capstone Project — Parkinson's Disease Care Pathway Classification"
    )
