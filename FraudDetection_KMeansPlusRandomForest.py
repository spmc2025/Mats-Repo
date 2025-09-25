"""
ablation tests:
F1, ROC-AUC, PR-AUC, Recall (very critical for fraud).

Program 2: Fraud Detection Training Pipeline
--------------------------------------------
Reads FraudBankingData.csv, generates unsupervised features (clustering + anomaly score),
then trains a supervised model to classify fraud. Validates model and saves everything.

Outputs:
- FraudBankingData_OUT.csv (unsupervised features)
- FraudBankingData_Final_OUT.csv (final predictions)
- fraud_model.pkl (supervised model)
- kmeans_model.pkl (unsupervised clustering model)
- isoforest_model.pkl (unsupervised anomaly model)
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# ------------------------------
# STEP 1: Load initial CSV
# ------------------------------
transactions = pd.read_csv("E:\\AI_Training\\CPT_Data\\Inbound_Data\\FraudBankingData.csv")

# Convert Yes/No to numeric for model training
transactions["is_fraud_numeric"] = transactions["is_fraud"].map({"Yes": 1, "No": 0})

# Keep aside transaction_id
transaction_ids = transactions["transaction_id"]

# ------------------------------
# STEP 2: Prepare Features
# ------------------------------
feature_cols = [col for col in transactions.columns if col not in ["transaction_id", "is_fraud", "is_fraud_numeric"]]

X_raw = transactions[feature_cols]

# Identify categorical vs numeric
categorical_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X_raw.select_dtypes(exclude=["object"]).columns.tolist()
# print("categorical_cols: ", categorical_cols)
# print("\nnumeric_cols: ", numeric_cols)

# Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# Below is doing preprocessing of your raw features(X_raw) & converting them into a numeric matrix ready for ML
# X_processed is now a 2D numeric array or sparse matrix:
    # Rows = number of transactions (len(X_raw))
    # Columns = numeric columns + new columns created by one-hot encoding categorical variables
        # If there are 3 unique values for 1 categorical column, 
        # then 3 column will be added to sparse matrix with original column value set to 1. Example below.
        # If there are 3 unique values (N, S, E, W) for 1 categorical column for 1000 transactions, then
        # For each row there will have 4 column with original value set to 1 and rest all set to 0
            # the same is repeated for all unique columns and each row.
            # Here, rows will be same as original, but columns will be much more
X_processed = preprocessor.fit_transform(X_raw)
# print("\nX_processed: ", X_processed.shape)

# ------------------------------
# STEP 3: Unsupervised Learning
# ------------------------------

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_ids = kmeans.fit_predict(X_processed)
# print("cluster_ids: ", cluster_ids)

iso = IsolationForest(contamination=0.05, random_state=42)
anomaly_scores = iso.fit_predict(X_processed)
# print("\nanomaly_scores: ", anomaly_scores)

transactions["cluster_id"] = cluster_ids
transactions["anomaly_score"] = anomaly_scores
transactions["cluster_label"] = transactions["cluster_id"].apply(lambda x: f"Cluster_{x}")
transactions["predicted_anomaly"] = transactions["anomaly_score"].map({1: "Normal", -1: "Anomaly"})
# below can be used instead of just above line and don't need "transactions["anomaly_score"] = anomaly_scores"
# transactions["predicted_anomaly"] = pd.Series(anomaly_scores).map({1: "Normal", -1: "Anomaly"})

# STEP 3a: Additional Algorithmic Features
# Ensure X_processed is dense for LOF/PCA/KMeans distances
# Checking if X_processed has the method 'toarray' (i.e., it is a sparse matrix)
if hasattr(X_processed, "toarray"):
    # Convert the sparse matrix into a dense NumPy array
    X_dense = X_processed.toarray()
else:
    # If it is already a dense array, just use it directly
    X_dense = X_processed

# 1️⃣ KMeans distances
# kmeans.transform(X_dense) computes the Euclidean distance of each transaction to each cluster centroid
# If there is 3 clusters and 5 features then there will have 3 cluster centroid for 5 features(column)
# See last for example
dists = kmeans.transform(X_dense)
# print("dists :", dists)

# Adding one new column to each transaction (row) to transactions
# Below says each transaction now has one extra number: “How far am I from my nearest cluster"
transactions["dist_to_closest_centroid"] = np.min(dists, axis=1)

# # Adding "cluster number of new columns" to each transaction to transactions
for i in range(dists.shape[1]):
    transactions[f"dist_to_centroid_{i}"] = dists[:, i]

# printing centroids
# for i, center in enumerate(kmeans.cluster_centers_):
#     print(f"Cluster {i}: {center}")


# 2️⃣ Local Outlier Factor (LOF)
# Below line creates a Local Outlier Factor (LOF) model, which is an unsupervised anomaly detection algorithm
# LOF looks at the local density of data points
# For each transaction, it compares how dense its neighborhood is versus how dense its neighbors’ neighborhoods are
# If a transaction is in a region much less dense than its neighbors, it’s considered an outlier/anomaly
    # n_neighbors=20
        # Each transaction’s density is compared with its 20 nearest neighbors
        # More neighbors = smoother detection (less sensitive to noise)
        # Fewer neighbors = more sensitive (but risk of false anomalies)
    # contamination=0.05
        # You tell LOF what fraction of your dataset you expect to be anomalies.
        # Here, 0.05 = 5% → It assumes ~5% of transactions are anomalies.
    # novelty=False
        # False= LOF is used only for fitting & predicting on the same dataset (it can’t generalize to new unseen data)
        # True= LOF is trained to later detect anomalies in new transactions (like real-time fraud detection)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=False)

# fit_predict():
    # fit ==> Trains the LOF model on your dataset (X_dense)
        # It finds the 20 nearest neighbors for each transaction
        # Computes each transaction’s local density and compares it with neighbors
    # predict means Assigns each transaction a label based on whether it looks “normal” or “anomaly”
# lof_labels:
    # A NumPy array of size = number of transactions.
    # Each entry is either: 1 =>> Normal transaction OR -1 =>> Anomalous transaction
# So in one sentence, it both fits and predicts on the data
lof_labels = lof.fit_predict(X_dense)

# If lof_labels == 1 then assign "Normal" Else (-1) then assign "Anomaly"
# Adds another column called "lof_label"
transactions["lof_label"] = np.where(lof_labels == 1, "Normal", "Anomaly")

# lof.negative_outlier_factor_ gives the LOF score for each transaction
# It’s a continuous value (negative), showing how “outlier-like” a point is
    # Values near -1.0 is very normal
    # Much smaller values (e.g., -3.0, -7.5) shows stronger anomalies
# Adds another column "lof_score" to store the degree of anomaly
transactions["lof_score"] = lof.negative_outlier_factor_

# 3️⃣ PCA reconstruction error
# PCA() = Principal Component Analysis
    # It reduces the number of features (columns) while keeping most of the variance (information)
    # It’s often used for dimensionality reduction, noise removal, or feature extraction
# n_components=min(10, X_dense.shape[1])

# X_dense.shape[1] = number of columns (features) in your dataset
    # min(10, X_dense.shape[1]) = pick the smaller value between 10 and the number of features
    # If you have fewer than 10 features → use all features
    # If you have more than 10 features → limit PCA to 10 components
# random_state=42
    # Fixes randomness for reproducibility
# This ensures PCA never tries to keep more components than you actually have
pca = PCA(n_components=min(10, X_dense.shape[1]), random_state=42)
print("X_dense.shape[1]: ", X_dense.shape[1])

# fit
    # Learns the PCA transformation from your dataset X_dense.
    # Finds the principal components (directions of maximum variance in the data).
    # Each component is a linear combination of your original features.
# transform
    # Projects your data onto those principal components.
    # The result is a new dataset with fewer dimensions (at most 10 in this case).
# Output (X_pca) is a NumPy array of shape:(n_samples, n_components)
X_pca = pca.fit_transform(X_dense)
print("X_pca shape: ", X_pca.shape)

# Below is the reverse step of PCA
    # X_pca = your data after PCA compression (fewer dimensions, e.g. 10 instead of 50).
    # inverse_transform = expands X_pca back into the original feature space.
    # Actually, it tries to reconstruct the original transactions from the reduced representation.
X_recon = pca.inverse_transform(X_pca)

# PCA is turned into an anomaly score
# X_dense → your original feature matrix (all transactions with all features).
# X_recon → the reconstructed version of the same data after PCA compression + inverse transform.
# X_dense - X_recon
    # Computes the difference (error) between the original and reconstructed values for 
    # every feature in every transaction
# (X_dense - X_recon) ** 2
    # Squares the differences: This ensures all errors are positive, and emphasizes larger errors
# np.sum(..., axis=1)
    # Adds up squared errors per transaction (row-wise)
# Result: one number per transaction = reconstruction error
# Adds another column "pca_recon_error" to transactions
transactions["pca_recon_error"] = np.sum((X_dense - X_recon) ** 2, axis=1)

# Update numeric_cols to include new features
# new_features = ["dist_to_closest_centroid"] + [f"dist_to_centroid_{i}" for i in range(dists.shape[1])] + ["lof_score", "pca_recon_error"]
new_features = (
    ["dist_to_closest_centroid"]  # First part: one column that stores the distance to the nearest cluster
    + [f"dist_to_centroid_{i}" for i in range(dists.shape[1])] # Second part: one column per cluster (e.g. dist_to_centroid_0, dist_to_centroid_1, …)
    + ["lof_score", "pca_recon_error"] # Third part: two anomaly scores (LOF score and PCA reconstruction error)
)
print("\nnew_features: ", new_features)
print("\nnumeric_cols: ", numeric_cols)

# numeric_cols NOT USED
# numeric_cols += new_features

# ------------------------------
# STEP 3b: Combine Original Features + New Unsupervised Features
# ------------------------------

# Convert original processed features (X_processed) to dense if needed
# if hasattr(X_processed, "toarray"):
#     X_processed_dense = X_processed.toarray()
# else:
#     X_processed_dense = X_processed

# Stack unsupervised numeric features alongside the original processed features
X_supervised = np.hstack([X_dense, transactions[new_features].values])
# print("\nX_supervised: \n", X_supervised)



unsupervised_features = transactions[["transaction_id", "cluster_label", "predicted_anomaly"]]
unsupervised_features.to_csv("E:\\AI_Training\\CPT_Data\\Outbound_Data\\FraudBankingData_OUT1.csv", index=False)

# ------------------------------
# STEP 4: Supervised Learning
# ------------------------------
y = transactions["is_fraud_numeric"]

X_train, X_test, y_train, y_test = train_test_split(
    X_supervised, y, test_size=0.3, random_state=42, stratify=y
)

# clf = RandomForestClassifier(
#     n_estimators=200, random_state=42, class_weight="balanced"
# )

# Initialize RandomForestClassifier (Production-ready)
# We are explicitly setting important parameters for clarity
# This is the "baseline production model" initialization
# NOTE: You can have N number of but ONLY ONE will be used in GridSearchCV,others will NOT be used anymore
clf = RandomForestClassifier(
    n_estimators=50,          # Number of trees in the forest(200); more trees → more stable but slower
    criterion="gini",          # Function to measure quality of a split ("gini" or "entropy")
    max_depth=None,            # Maximum depth of each tree; None → grow until leaves are pure
    min_samples_split=2,       # Minimum samples required to split a node
    min_samples_leaf=1,        # Minimum samples required at a leaf node
    max_features="sqrt",       # Number of features to consider at each split; "sqrt" = sqrt(total_features)
    bootstrap=True,            # Use bootstrap samples to build each tree (standard RF)
    oob_score=False,           # Out-of-bag score estimate (optional, useful for quick validation)
    class_weight="balanced",   # Adjust weights inversely proportional to class frequencies
    random_state=42,           # Fix randomness for reproducibility
    n_jobs=-1,                 # Use all CPU cores to speed up training
    verbose=0                  # No logging; set >0 for debug info per tree
)

# Initialize a RandomForestClassifier (will be tuned)
clf1 = RandomForestClassifier(random_state=42, class_weight="balanced")

# ---------------------------------------------------------
# STEP 4a: Hyperparameter Tuning with GridSearchCV (Optional)
# ---------------------------------------------------------
# Here we search for the best combination of RandomForest parameters
# to maximize validation performance before training the final model.
# Define parameter grid for RandomForest
    # NOTE: Whatever the values given in RandomForestClassifier will be overriden in param_grid
    # NOTE: If any param is not given in "param_grid", the original value from RF will be used in GridSearchCV
# Anything related to which parameters to try goes in param_grid
param_grid = {
    "n_estimators": [1, 2, 3],          # number of trees
    "max_depth": [None, 10, 20, 30],          # maximum depth of each tree
    "min_samples_split": [2, 5, 10],          # minimum samples required to split a node
    "min_samples_leaf": [1, 2, 4],            # minimum samples required at a leaf node
    "random_state": [2,5],
    "max_features": ["sqrt", "log2", None],   # number of features to consider at each split
    "bootstrap": [True, False],               # whether bootstrap samples are used
}

# Initialize GridSearchCV
# Parameters that were in param_grid = take the best value found during search
# Parameters not in param_grid will remain same as in the original RF initialization used in GridSearchCV
# Anything related to how the search is executed (folds, scoring metric, parallelism, verbosity, refitting) is passed to GridSearchCV.
grid_search = GridSearchCV(
    estimator=clf1, # You may use clf or clf1. But only one
    param_grid=param_grid,
    cv=3,                     # 3-fold cross-validation
    scoring="f1",             # use F1-score as evaluation metric
    n_jobs=-1,                # use all CPU cores
    verbose=2                 # print progress
)

# Fit GridSearchCV on training data
grid_search.fit(X_train, y_train)

# Best combination of hyperparameters
print("\n✅ Best hyperparameters found by GridSearchCV:")
print(grid_search.best_params_)

# Replace original classifier with best found
    # Anything you set in the original RandomForestClassifier that is not in your param_grid will still 
    # be used by the model during the search and in the best_estimator_, but it won’t appear in best_params_, 
    # because best_params_ only lists the parameters that were actually searched/tuned.
# NOTE: "grid_search.best_estimator_" will assign the best RandomForestClassifier with all parameters fitted in
clf = grid_search.best_estimator_
clf.fit(X_train, y_train)

# ---------------------------------------------------------
# Model Evaluation (on unseen test data only)
# ---------------------------------------------------------
# Here we use only the test split (X_test, y_test).
# This tells us how well the model generalizes to new data.
y_pred = clf.predict(X_test)

# ------------------------------
# STEP 5: Validation Metrics
# ------------------------------
print("\n📊 Supervised Model Evaluation Metrics @@@@ UNCOMMENT BELOW Metrices")
# print("--------------------------------------")
# print("Accuracy :", accuracy_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred))
# print("Recall   :", recall_score(y_test, y_pred))
# print("F1-score :", f1_score(y_test, y_pred))

print("\nDetailed Report: @@@@ UNCOMMENT BELOW classification_report")
# print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:@@@@ UNCOMMENT BELOW confusion_matrix")
# print(cm)

# ------------------------------
# STEP 6a: Final Retraining on 100% Data (Production Model)
# ------------------------------
# At this point, we have already evaluated the model on the test set,
# so we know it generalizes well. Now, for production, we retrain on ALL data
# (X_supervised, y) to maximize learning before saving the final model.
clf_final = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
clf_final.fit(X_supervised, y)

# Replace the old clf with this final version for saving and predictions
clf = clf_final

# ------------------------------
# STEP 6: Final Predictions CSV
# You already validated your model and know it generalizes well, training on all available data 
# lets the model learn patterns from every example. This can give slightly better predictions in production.
    # NOTE: it’s a conventional step, not a strict requirement
# ------------------------------
transactions["fraud_prediction"] = clf.predict(X_supervised)
transactions["predicted_fraud"] = transactions["fraud_prediction"].map({1: "Yes", 0: "No"})
transactions = transactions.rename(columns={"is_fraud": "original_is_fraud"})

final_output = transactions[
    [
        "transaction_id",
        "cluster_label",
        "predicted_anomaly",
        "original_is_fraud",
        "predicted_fraud",
    ]
]
final_output.to_csv("E:\\AI_Training\\CPT_Data\\Outbound_Data\\FraudBankingData_Final_OUT1.csv", index=False)

# ------------------------------
# STEP 7: Save Models
# ------------------------------
joblib.dump(clf, "fraud_model.pkl")
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(iso, "isoforest_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(lof, "lof_model.pkl")
joblib.dump(pca, "pca_model.pkl")

print("\n✅ Training pipeline complete. Outputs saved: @@@@ UNCOMMENT BELOW")
# print("- FraudBankingData_OUT.csv")
# print("- FraudBankingData_Final_OUT.csv")
# print("- fraud_model.pkl")
# print("- kmeans_model.pkl")
# print("- isoforest_model.pkl")
# print("- preprocessor.pkl")

# -------------------------------------------------------------
# STEP 8: Feature Importance Visualization (RandomForest)
# -------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Generate actual feature names for X_supervised
# -----------------------------
dense_feature_names = []
for name, transformer, cols in preprocessor.transformers_:
    if name == "cat":
        ohe = transformer
        try:
            cat_names = ohe.get_feature_names_out(cols)
        except:
            cat_names = ohe.get_feature_names(cols)
        dense_feature_names.extend(cat_names)
    elif name == "num":
        dense_feature_names.extend(cols)

all_feature_names = dense_feature_names + new_features

importances = clf.feature_importances_

# -----------------------------
# Sort descending and keep top 20 for clarity
# -----------------------------
indices = np.argsort(importances)[::-1]
top_n = 20
top_indices = indices[:top_n]
top_feature_names = [all_feature_names[i] for i in top_indices]
top_importances = importances[top_indices]

# -----------------------------
# Plot horizontal bar chart
# -----------------------------
plt.figure(figsize=(10, 8))
bars = plt.barh(top_feature_names, top_importances, color="skyblue")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title(f"Top {top_n} RandomForest Feature Importances")
plt.gca().invert_yaxis()  # most important on top

# Add importance values at end of bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
             f"{width:.3f}", va='center', fontsize=9)

# Save figure as PNG
plt.savefig("E:\\AI_Training\\CPT_Data\\Plots\\feature_importances_top20kk.png", bbox_inches="tight")

plt.show()



########################## BELOW ARE ALL COMMENTS #################################################
# ---------------------------------------------------------
# NOTE: Evaluating the usefulness of enriched features
# ---------------------------------------------------------
# Q: We added extra features from KMeans (cluster distances),
#    IsolationForest (anomaly score), LOF (local outlier score),
#    and PCA (reconstruction error). How do we know these help?
#
# A:
# 1. Their standalone "accuracy" cannot be measured because they are unsupervised.
# 2. We measure their usefulness *indirectly*:
#    - Check RandomForest feature_importances_ to see which features matter most.
#    - Perform ablation tests: retrain the model with/without certain features
#      and compare metrics (ROC-AUC, Precision, Recall).
#    - Tune unsupervised model parameters (e.g., n_clusters for KMeans,
#      n_neighbors for LOF, n_components for PCA) using GridSearchCV or experiments.
# 3. If a feature adds no value (low importance, no metric improvement),
#    it can be safely removed from new_features.



# Euclidean distance calculation example
"""
Got it ✅ Let’s do the full toy example again with actual numbers this time.

We’ll use:

3 transactions (rows)
5 features (columns)
2 clusters (C0, C1)

Step 1: Transactions (the data matrix)
Tran	F1	F2	F3	F4	F5
Txn1	2	4	6	8	10
Txn2	3	5	7	9	11
Txn3	12	14	16	18	20

So:

Txn1 = [2, 4, 6, 8, 10]
Txn2 = [3, 5, 7, 9, 11]
Txn3 = [12, 14, 16, 18, 20]

Step 2: Cluster assignment (say KMeans assigns)
Txn1 → Cluster 0
Txn2 → Cluster 0
Txn3 → Cluster 1
Step 3: Centroids

Compute average for each cluster:
Cluster 0 centroid (C0) = average of Txn1 & Txn2
C0 = [(2+3)/2, (4+5)/2, (6+7)/2, (8+9)/2, (10+11)/2]
   = [2.5, 4.5, 6.5, 8.5, 10.5]


Cluster 1 centroid (C1) = Txn3 alone
C1 = [12, 14, 16, 18, 20]

Step 4: Distances
We use Euclidean distance:

distance
🔹 Txn1 → [2,4,6,8,10]
To C0 [2.5,4.5,6.5,8.5,10.5]
= sqrt((2-2.5)^2 + (4-4.5)^2 + (6-6.5)^2 + (8-8.5)^2 + (10-10.5)^2)
= sqrt((-0.5)^2 + (-0.5)^2 + (-0.5)^2 + (-0.5)^2 + (-0.5)^2)
= sqrt(0.25*5)
= sqrt(1.25)
≈ 1.118

To C1 [12,14,16,18,20]
= sqrt((2-12)^2 + (4-14)^2 + (6-16)^2 + (8-18)^2 + (10-20)^2)
= sqrt((-10)^2 + (-10)^2 + (-10)^2 + (-10)^2 + (-10)^2)
= sqrt(100*5)
= sqrt(500)
≈ 22.361

🔹 Txn2 → [3,5,7,9,11]
To C0 [2.5,4.5,6.5,8.5,10.5]
= sqrt((3-2.5)^2 + (5-4.5)^2 + (7-6.5)^2 + (9-8.5)^2 + (11-10.5)^2)
= sqrt((0.5)^2*5)
= sqrt(0.25*5)
= sqrt(1.25)
≈ 1.118

To C1 [12,14,16,18,20]
= sqrt((3-12)^2 + (5-14)^2 + (7-16)^2 + (9-18)^2 + (11-20)^2)
= sqrt((-9)^2 + (-9)^2 + (-9)^2 + (-9)^2 + (-9)^2)
= sqrt(81*5)
= sqrt(405)
≈ 20.124

🔹 Txn3 → [12,14,16,18,20]
To C0 [2.5,4.5,6.5,8.5,10.5]
= sqrt((12-2.5)^2 + (14-4.5)^2 + (16-6.5)^2 + (18-8.5)^2 + (20-10.5)^2)
= sqrt(9.5^2 + 9.5^2 + 9.5^2 + 9.5^2 + 9.5^2)
= sqrt(90.25*5)
= sqrt(451.25)
≈ 21.249

To C1 [12,14,16,18,20]
= sqrt((12-12)^2 + (14-14)^2 + (16-16)^2 + (18-18)^2 + (20-20)^2)
= sqrt(0)
= 0

Step 5: Distances table (dists)
Transaction	Dist to C0	Dist to C1
Txn1	1.118	22.361
Txn2	1.118	20.124
Txn3	21.249	0.000

✅ This is exactly what kmeans.transform(X_dense) gives:

One row per transaction

One column per cluster (distance to centroid)
"""
"""
📊 Metrics Priority in Fraud Detection
1️⃣ Recall (a.k.a. True Positive Rate, Sensitivity)

Why #1:
Missing a fraud (False Negative) is usually the worst outcome (loss of money, chargebacks, regulatory risks).

Interpretation: Of all actual frauds, how many did we catch?

Goal: Maximize Recall, even if it means catching more false positives.

2️⃣ Precision

Why #2:
Too many false alarms (False Positives) overwhelm investigators and annoy customers (legit transactions get blocked).

Interpretation: Of all transactions flagged as fraud, how many are truly fraud?

Balance: We want high enough Precision so alerts are actionable.

3️⃣ F1-score

Why #3:
Combines Recall & Precision into one number.

High F1 = balance between catching fraud & keeping alerts useful.

Especially important if dataset is imbalanced (fraud is rare).

4️⃣ PR-AUC (Precision–Recall AUC)

Why #4:
Much better than ROC-AUC in imbalanced fraud data.

It shows how Precision vs Recall trade off at different thresholds.

A high PR-AUC = model maintains good balance across thresholds.

5️⃣ ROC-AUC

Why #5:
Still useful, but can look “good” even in imbalanced fraud datasets because TNs dominate.

Example: 99% legitimate, 1% fraud → a weak model can still show high ROC-AUC.

Use as a supporting metric, not the primary one.

6️⃣ Confusion Matrix

Why #6:
Gives absolute counts of TP, FP, TN, FN.

Useful for business teams to understand scale of misses/false alarms.

Helps translate into real-world cost impact.

7️⃣ Accuracy

Lowest priority in fraud detection

Can be misleading when fraud is rare.

Example: If 99% are legitimate, predicting “No fraud” always → 99% accuracy but 0% Recall.

Should never be the main metric in fraud.

✅ Practical Priority Order

👉 Recall > Precision > F1 > PR-AUC > ROC-AUC > Confusion Matrix > Accuracy
"""