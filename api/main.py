from fastapi import FastAPI
import pandas as pd
import joblib
import os
import datetime

# -------------------------
# CREATE FASTAPI APP (FIRST!)
# -------------------------
app = FastAPI(title="Customer Churn Prediction API")

# -------------------------
# PATH SETUP
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "churn_xgboost_model.joblib")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "encoded_churn.csv")
MONITOR_DIR = os.path.join(BASE_DIR, "..", "monitoring")

os.makedirs(MONITOR_DIR, exist_ok=True)

# -------------------------
# LOAD MODEL & FEATURES
# -------------------------
model = joblib.load(MODEL_PATH)

feature_template = pd.read_csv(DATA_PATH)
feature_columns = feature_template.drop("Churn", axis=1).columns

# -------------------------
# HOME ENDPOINT
# -------------------------
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

# -------------------------
# PREDICT ENDPOINT
# -------------------------
@app.post("/predict")
def predict_churn(customer: dict):

    # Convert input JSON to DataFrame
    df = pd.DataFrame([customer])

    # Add missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    df = df[feature_columns]

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # -------------------------
    # MODEL PREDICTION
    # -------------------------
    churn_probability = model.predict_proba(df)[0][1]
    churn_prediction = model.predict(df)[0]

    # -------------------------
    # MONITORING
    # -------------------------
    timestamp = datetime.datetime.now()

    # Input stats
    input_log = df.copy()
    input_log["timestamp"] = timestamp

    input_log.to_csv(
        os.path.join(MONITOR_DIR, "input_stats.csv"),
        mode="a",
        header=not os.path.exists(os.path.join(MONITOR_DIR, "input_stats.csv")),
        index=False
    )

    # Prediction stats
    prediction_log = pd.DataFrame([{
        "timestamp": timestamp,
        "churn_probability": churn_probability,
        "churn_prediction": churn_prediction
    }])

    prediction_log.to_csv(
        os.path.join(MONITOR_DIR, "prediction_stats.csv"),
        mode="a",
        header=not os.path.exists(os.path.join(MONITOR_DIR, "prediction_stats.csv")),
        index=False
    )

    # -------------------------
    # RESPONSE
    # -------------------------
    return {
        "churn_probability": round(float(churn_probability), 3),
        "churn_prediction": "YES" if churn_prediction == 1 else "NO"
    }
