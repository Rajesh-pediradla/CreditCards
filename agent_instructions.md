# Agent Instructions

## Project: Credit Card Fraud Detection — ML Web App

### Overview
This project is a complete end-to-end machine learning application for detecting
fraudulent credit-card transactions. It uses the `Credit Card Fraud 2026` dataset.

### Problem Type
Binary Classification — predict `is_fraud` (0 = Legitimate, 1 = Fraud).

### Key Files
| File | Purpose |
|---|---|
| `train_model.py` | Loads data, preprocesses, compares models, saves best pipeline |
| `app.py` | Streamlit UI — accepts user inputs, loads model, returns prediction |
| `models/model.pkl` | Serialized sklearn Pipeline + metadata (joblib) |
| `data/credit_card_fraud_2026.csv` | Raw dataset |

### Workflow
1. Run `python train_model.py` to train and persist the model.
2. Run `streamlit run app.py` to launch the web interface.

### Preprocessing Steps (in `train_model.py`)
- Drop `transaction_id` (identifier column).
- Map boolean string columns (`True`/`False`) to integer 0/1.
- Apply `StandardScaler` to numeric features.
- Apply `OneHotEncoder(handle_unknown='ignore')` to categorical features.
- Use `ColumnTransformer` + `Pipeline` for reproducibility.

### Model Selection
Four classifiers are compared with 5-fold stratified cross-validation scored by
ROC-AUC. The best-performing model is retrained on the full training split.

### Deployment Notes
- Uses **relative paths** only — no absolute paths.
- No API keys or LLM calls — purely offline sklearn inference.
- Compatible with Streamlit Community Cloud: push the repo including `models/model.pkl`.

### Extending the Project
- To add a new model: insert it into the `candidates` dict in `train_model.py` and re-run.
- To swap datasets: replace `data/credit_card_fraud_2026.csv` and update column names.
