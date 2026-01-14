import pandas as pd
import joblib
import os
from xgboost import XGBClassifier

# -------------------------
# PATH SETUP
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "encoded_churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "churn_xgboost_model.joblib")

# -------------------------
# LOAD TRAINING DATA
# -------------------------
data = pd.read_csv(TRAIN_DATA_PATH)

# -------------------------
# PREPARE FEATURES & TARGET
# -------------------------
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Ensure clean labels
y = y.dropna()
X = X.loc[y.index]

# Numeric safety
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# -------------------------
# RETRAIN MODEL
# -------------------------
print("🚨 Drift detected → Retraining model with clean labeled data...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

model.fit(X, y)

# -------------------------
# SAVE NEW MODEL
# -------------------------
joblib.dump(model, MODEL_PATH)

print("✅ Model retrained and saved successfully")
