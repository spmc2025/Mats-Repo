# ablation tests:
"""
Program 3: Fraud Detection Inference (Production Style)
-------------------------------------------------------
Loads trained supervised + unsupervised models and applies them to
new unseen transactions (new_transactions.csv). Produces fraud predictions
without retraining.

Inputs:
- new_transactions.csv (NO fraud labels)
- fraud_model.pkl (supervised model)
- kmeans_model.pkl (unsupervised clustering model)
- isoforest_model.pkl (unsupervised anomaly model)

Outputs:
- scored_new_transactions.csv (with fraud_prediction column)
"""

import pandas as pd
import joblib
import sys

# ------------------------------
# STEP 1: Load Data & Models
# ------------------------------
try:
    new_data = pd.read_csv("new_transactions.csv")
except FileNotFoundError:
    print("❌ ERROR: new_transactions.csv not found.")
    sys.exit(1)

try:
    clf = joblib.load("fraud_model.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    iso = joblib.load("isoforest_model.pkl")
except FileNotFoundError as e:
    print(f"❌ ERROR: Missing model file: {e.filename}")
    sys.exit(1)

# ------------------------------
# STEP 2: Validate Input Schema
# ------------------------------
required_cols = ["transaction_id", "amount", "merchant_id", "device_id", "location", "time"]
missing = [c for c in required_cols if c not in new_data.columns]
if missing:
    print(f"❌ ERROR: Missing required columns in new_transactions.csv: {missing}")
    sys.exit(1)

# ------------------------------
# STEP 3: Apply Unsupervised Models
# ------------------------------
features = new_data[["amount"]]

# Apply pre-trained KMeans
new_data["cluster_id"] = kmeans.predict(features)

# Apply pre-trained Isolation Forest
new_data["anomaly_score"] = iso.predict(features)  # -1 = anomaly, 1 = normal

# ------------------------------
# STEP 4: Supervised Prediction
# ------------------------------
X_new = new_data[["amount", "cluster_id", "anomaly_score"]]
new_data["fraud_prediction"] = clf.predict(X_new)

# ------------------------------
# STEP 5: Save Predictions
# ------------------------------
new_data.to_csv("scored_new_transactions.csv", index=False)

print("✅ Inference complete. Output saved to scored_new_transactions.csv")
