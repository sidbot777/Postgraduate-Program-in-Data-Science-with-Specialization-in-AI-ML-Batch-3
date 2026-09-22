# Parkinson's Disease Care Pathway Recommendation System

Academic Streamlit prototype for classifying Parkinson's disease care pathways using synthetic data.

## Care pathways

- Pharmacological Management
- Rehabilitation Referral
- Surgical Referral

## Models

Baseline: Logistic Regression

Final model: Decision Tree

Final Decision Tree:
- criterion = entropy
- max_depth = 5
- min_samples_leaf = 2
- random_state = 42

## Performance

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression | 73.50% | 67.11% |
| Decision Tree | 93.00% | 87.62% |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place `decision_tree_final.pkl` and/or `deployment_assets.joblib` in the same folder as `app.py`.

This is an academic prototype using synthetic data and is not clinically validated.
