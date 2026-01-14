import pandas as pd
import os

# -------------------------
# PATH SETUP (SAFE METHOD)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OLD_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "encoded_churn.csv")
NEW_DATA_PATH = os.path.join(BASE_DIR, "input_stats.csv")

# -------------------------
# LOAD OLD & NEW DATA
# -------------------------
old_data = pd.read_csv(OLD_DATA_PATH)
new_data = pd.read_csv(NEW_DATA_PATH)

# Remove target & timestamp
old_data = old_data.drop(columns=["Churn"], errors="ignore")
new_data = new_data.drop(columns=["timestamp"], errors="ignore")

# -------------------------
# CALCULATE MEANS
# -------------------------
old_means = old_data.mean()
new_means = new_data.mean()

# -------------------------
# DRIFT CALCULATION
# -------------------------
drift_report = []

THRESHOLD = 0.20  # 20%

for feature in old_means.index:
    if feature in new_means.index:
        old_val = old_means[feature]
        new_val = new_means[feature]

        if old_val == 0:
            continue

        change = abs(new_val - old_val) / abs(old_val)

        drift_report.append({
            "feature": feature,
            "old_mean": round(old_val, 3),
            "new_mean": round(new_val, 3),
            "change_%": round(change * 100, 2),
            "drift_detected": change > THRESHOLD
        })

# -------------------------
# RESULT
# -------------------------
drift_df = pd.DataFrame(drift_report)

print("\nDRIFT DETECTION REPORT\n")
print(drift_df.sort_values("change_%", ascending=False))
