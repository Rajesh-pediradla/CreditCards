"""
train_model.py
Credit Card Fraud Detection — 2026 Dataset
Trains multiple classifiers, selects the best by ROC-AUC, and saves the full
sklearn pipeline to models/model.pkl.
"""

import os
import pathlib
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "credit_card_fraud_2026.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load dataset ─────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# ── Basic inspection ──────────────────────────────────────────────────────────
TARGET = "is_fraud"
DROP_COLS = ["transaction_id"]          # identifier — not a feature

print(f"\nTarget column : '{TARGET}'")
print(f"Class distribution:\n{df[TARGET].value_counts()}")

# ── Drop ID column ────────────────────────────────────────────────────────────
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# ── Encode boolean-string columns (True/False stored as strings) ──────────────
bool_str_cols = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
for col in bool_str_cols:
    if col in df.columns:
        df[col] = df[col].map({"True": 1, "False": 0, True: 1, False: 0}).astype(int)

# ── Split features / target ───────────────────────────────────────────────────
X = df.drop(columns=[TARGET])
y = df[TARGET]

# ── Identify categorical vs numeric columns ───────────────────────────────────
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

print(f"\nNumeric features  ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ── Preprocessing pipeline ────────────────────────────────────────────────────
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(
    steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# ── Candidate models ──────────────────────────────────────────────────────────
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, n_jobs=1, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
    ),
}

# ── Cross-validated comparison ────────────────────────────────────────────────
print("\n--- Model Comparison (5-fold Stratified CV - ROC-AUC) ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, clf in candidates.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
    results[name] = scores.mean()
    print(f"  {name:<25}  AUC = {scores.mean():.4f}  (+/-{scores.std():.4f})")

best_name = max(results, key=results.get)
print(f"\nBest model: {best_name}  (AUC = {results[best_name]:.4f})")

# ── Retrain best model on full training set ───────────────────────────────────
best_clf = candidates[best_name]
best_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", best_clf)]
)
best_pipeline.fit(X_train, y_train)

# ── Evaluate on held-out test set ────────────────────────────────────────────
y_pred = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

print("\n--- Test-Set Evaluation ---")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── Save pipeline metadata alongside the model ────────────────────────────────
metadata = {
    "model_name": best_name,
    "target": TARGET,
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "bool_str_cols": bool_str_cols,
    "feature_order": numeric_cols + categorical_cols,   # consistent input ordering
    "pipeline": best_pipeline,
    "roc_auc_test": round(roc_auc_score(y_test, y_proba), 4),
    "cv_results": {k: float(round(v, 4)) for k, v in results.items()},
}

joblib.dump(metadata, MODEL_PATH)
print(f"\nModel saved -> {MODEL_PATH}")
