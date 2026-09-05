"""
app.py
Credit Card Fraud Detection — Streamlit Web Application
Trains the model on first run (cached) and serves live predictions.
This avoids pickle version-mismatch errors across Python/sklearn versions.
"""

import pathlib
import warnings

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "credit_card_fraud_2026.csv"

TARGET = "is_fraud"
DROP_COLS = ["transaction_id"]
BOOL_STR_COLS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]


# ── Train model (cached — only runs once per session) ─────────────────────────
@st.cache_resource(show_spinner="Training model on first run … (this takes ~30 seconds)")
def train_model():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    for col in BOOL_STR_COLS:
        if col in df.columns:
            df[col] = df[col].map({"True": 1, "False": 0, True: 1, False: 0}).astype(int)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for name, clf in candidates.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        cv_results[name] = float(round(scores.mean(), 4))

    best_name = max(cv_results, key=cv_results.get)
    best_clf = candidates[best_name]
    best_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", best_clf)]
    )
    best_pipeline.fit(X_train, y_train)

    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    roc_auc = round(roc_auc_score(y_test, y_proba), 4)

    return {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "bool_str_cols": BOOL_STR_COLS,
        "roc_auc_test": roc_auc,
        "cv_results": cv_results,
    }


# ── Load / train ──────────────────────────────────────────────────────────────
if not DATA_PATH.exists():
    st.error("❌ Dataset not found. Please ensure `data/credit_card_fraud_2026.csv` is present.")
    st.stop()

metadata = train_model()

pipeline     = metadata["pipeline"]
model_name   = metadata["model_name"]
numeric_cols = metadata["numeric_cols"]
categorical_cols = metadata["categorical_cols"]
bool_str_cols    = metadata["bool_str_cols"]
roc_auc          = metadata["roc_auc_test"]
cv_results       = metadata["cv_results"]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛡️ Credit Card Fraud Detection")
st.markdown(
    f"""
    **Model in use:** `{model_name}` &nbsp;|&nbsp; **Test ROC-AUC:** `{roc_auc}`

    Fill in the transaction details below and click **Predict** to check whether the
    transaction is *fraudulent* or *legitimate*.
    """
)
st.divider()

# ── Sidebar — model info ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Performance")
    st.caption("5-Fold Cross-Validated ROC-AUC")
    for name, score in sorted(cv_results.items(), key=lambda x: -x[1]):
        marker = "✅" if name == model_name else "  "
        st.write(f"{marker} **{name}**: `{score}`")
    st.divider()
    st.caption("Dataset: Credit Card Fraud 2026")

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Transaction Details")

MERCHANT_CATEGORIES = [
    "Restaurants", "Online Retail", "Groceries", "Streaming", "Travel",
    "Gift Cards", "Electronics", "Fuel", "Gaming", "Utilities",
    "Crypto Exchange", "Healthcare", "Education", "Clothing", "Jewelry",
    "ATM Withdrawal", "Insurance", "Real Estate", "Charity",
]
CARD_TYPES   = ["Visa", "Mastercard", "Amex", "Discover", "RuPay"]
AUTH_METHODS = ["OTP", "3D Secure", "PIN", "Biometric", "No Authentication", "Password"]
CHANNELS     = ["Online", "POS", "Contactless", "In-App", "ATM"]
DEVICE_TYPES = [
    "Android Phone", "iPhone", "Mac", "Windows PC", "Tablet",
    "POS Terminal", "ATM Machine", "Smart Watch",
]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Transaction**")
    amount_usd          = st.number_input("Amount (USD)", min_value=0.01, max_value=100000.0, value=50.0, step=0.01)
    merchant_category   = st.selectbox("Merchant Category", MERCHANT_CATEGORIES)
    channel             = st.selectbox("Channel", CHANNELS)
    time_of_day_hour    = st.slider("Hour of Day", 0, 23, 12)
    day_of_week         = st.selectbox("Day of Week", list(range(7)), format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
    hours_since_last_txn = st.number_input("Hours Since Last Transaction", min_value=0.0, max_value=720.0, value=5.0, step=0.1)
    txn_count_last_24h  = st.number_input("Transactions in Last 24 Hours", min_value=0, max_value=100, value=3)

with col2:
    st.markdown("**Card & Authentication**")
    card_type              = st.selectbox("Card Type", CARD_TYPES)
    auth_method            = st.selectbox("Auth Method", AUTH_METHODS)
    device_type            = st.selectbox("Device Type", DEVICE_TYPES)
    card_age_months        = st.number_input("Card Age (months)", min_value=0, max_value=300, value=36)
    cvv_retry_count        = st.number_input("CVV Retry Count", min_value=0, max_value=10, value=0)
    is_foreign_transaction = st.checkbox("Foreign Transaction")
    is_new_merchant        = st.checkbox("New / Unknown Merchant")

with col3:
    st.markdown("**Customer & Risk**")
    customer_age                = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
    account_balance_usd         = st.number_input("Account Balance (USD)", min_value=0.0, max_value=200000.0, value=2000.0, step=10.0)
    distance_from_home_km       = st.number_input("Distance from Home (km)", min_value=0.0, max_value=10000.0, value=15.0, step=0.1)
    velocity_score              = st.number_input("Velocity Score", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
    merchant_risk_score         = st.number_input("Merchant Risk Score", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
    prior_disputes              = st.number_input("Prior Disputes", min_value=0, max_value=20, value=0)
    used_vpn                    = st.checkbox("VPN Used")
    ip_country_mismatch         = st.checkbox("IP Country Mismatch")
    billing_shipping_mismatch   = st.checkbox("Billing / Shipping Mismatch")
    is_ai_generated_scam_attempt = st.checkbox("AI-Generated Scam Attempt Flag")

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
predict_btn = st.button("🔍 Predict Fraud", type="primary", use_container_width=True)

if predict_btn:
    input_dict = {
        "amount_usd":                   [amount_usd],
        "hours_since_last_txn":         [hours_since_last_txn],
        "txn_count_last_24h":           [txn_count_last_24h],
        "distance_from_home_km":        [distance_from_home_km],
        "card_age_months":              [card_age_months],
        "customer_age":                 [customer_age],
        "account_balance_usd":          [account_balance_usd],
        "cvv_retry_count":              [cvv_retry_count],
        "velocity_score":               [velocity_score],
        "time_of_day_hour":             [time_of_day_hour],
        "day_of_week":                  [day_of_week],
        "merchant_risk_score":          [merchant_risk_score],
        "prior_disputes":               [prior_disputes],
        "is_foreign_transaction":       [int(is_foreign_transaction)],
        "is_new_merchant":              [int(is_new_merchant)],
        "used_vpn":                     [int(used_vpn)],
        "ip_country_mismatch":          [int(ip_country_mismatch)],
        "billing_shipping_mismatch":    [int(billing_shipping_mismatch)],
        "is_ai_generated_scam_attempt": [int(is_ai_generated_scam_attempt)],
        "merchant_category":            [merchant_category],
        "card_type":                    [card_type],
        "auth_method":                  [auth_method],
        "channel":                      [channel],
        "device_type":                  [device_type],
    }

    input_df = pd.DataFrame(input_dict)

    prediction  = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0]
    fraud_prob  = probability[1]
    legit_prob  = probability[0]

    st.subheader("Prediction Result")

    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        if prediction == 1:
            st.error("🚨 **FRAUDULENT Transaction**", icon="🚨")
        else:
            st.success("✅ **LEGITIMATE Transaction**", icon="✅")

    with res_col2:
        st.metric("Fraud Probability",     f"{fraud_prob:.2%}")
        st.metric("Legitimate Probability", f"{legit_prob:.2%}")
        st.progress(float(fraud_prob), text=f"Fraud risk: {fraud_prob:.2%}")

    with st.expander("📋 Input Summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)
