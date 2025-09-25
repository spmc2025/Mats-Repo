# StudentPerformance_LightGBM_Production_UserReady.py
"""
Production-Ready Multi-Class Classification Pipeline for Student Performance
-------------------------------------------------------------------------------
- Inputs: Student performance CSV
- Outputs:
    1. Technical outputs: Enriched CSV, LightGBM model, plots, SHAP plots
    2. User-ready outputs: Human-readable CSV with predicted grades, total_score, percentage
       and plots showing grade distributions
- Enrichment: MiniBatchKMeans, PCA, UMAP, IsolationForest
- Stratified train/test split for realistic evaluation
- Timestamped outputs for versioning
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import shap
from datetime import datetime

# -------------------------
# Paths
# -------------------------
input_csv = r"E:\AI_Training\CPT_Data\Inbound_Data\student_performance_data.csv"
output_csv_folder = r"E:\AI_Training\CPT_Data\Outbound_Data\\"
plots_folder = r"E:\AI_Training\CPT_Data\Plots\\"
models_folder = r"E:\AI_Training\CPT_Data\Models_Created\\"

os.makedirs(output_csv_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)
os.makedirs(models_folder, exist_ok=True)

# -------------------------
# Timestamp for outputs
# -------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# -------------------------
# Load Data
# -------------------------
df = pd.read_csv(input_csv)
print("Data loaded:", df.shape)
# print(df.head())

# Compute total_score and percentage
# Automatically detects all columns(subject1, subject2,...). If yo add subject7, then no need to change code
# It will ensure consistency for total and percentage
subject_cols = []
# Loop through all column names
for c in df.columns:
    # Check if column name starts with 'subject'
    if c.startswith('subject'):
        # Add it to the list
        subject_cols.append(c)
# print("subject_cols:  ", subject_cols)
df['total_score'] = df[subject_cols].sum(axis=1)
df['percentage'] = df['total_score'] / (len(subject_cols)*100) * 100
# print("df['total_score']:  ", df['total_score'])
# print("df['percentage']:  ", df['percentage'])

# Identify Features & Target
feature_cols = []
# Loop through all columns in the DataFrame
for c in df.columns:
    # If the column is not in the list of excluded columns, add it
    if c not in ["student_id", "student_name", "grade_class", "total_score", "percentage"]:
        feature_cols.append(c)
# print("Feature columns from CSV:", feature_cols)
X = df[feature_cols].copy()
y = df["grade_class"]
# print("X: ",X)

# Extract & Encode categorical features if any, which usually means categorical/text columns
cat_cols = X.select_dtypes(include='object').columns.tolist()
# print("cat_cols: ", cat_cols)
for col in cat_cols:
    le = LabelEncoder()
    # Encoding categorical columns into numeric values so they can be used by LightGBM
    X[col] = le.fit_transform(X[col])

# Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Unsupervised Feature Enrichment
# 1. MiniBatchKMeans cluster labels
kmeans = MiniBatchKMeans(n_clusters=3, random_state=42)
# fit_predict does two things:
    # Fit: Finds 3 cluster centroids based on the training data
    # Predict: Assigns each row in X_train to a cluster label (0, 1, or 2)
X_train['kmeans_label'] = kmeans.fit_predict(X_train)
# Assign clusters to test data
# For the test set, we do not want to refit the clusters
# Instead, we use the centroids learned from training data to assign each test row to the nearest cluster
# This ensures that the test features are consistent with training features
# This is real prediction on test data as kmeans already learned using kmeans.fit_predict(X_train)
X_test['kmeans_label'] = kmeans.predict(X_test)

# 2. PCA (first 2 components)
pca = PCA(n_components=2, random_state=42)

# fit_transform: it does 2 things: learns the components and applies the transformation in one step
pca_train = pca.fit_transform(X_train)
# Adds the first and second principal components as new features to X_train
X_train['pca_1'] = pca_train[:,0]
X_train['pca_2'] = pca_train[:,1]

# transform only applies the existing PCA transformation learned from training to the new data (X_test)
pca_test = pca.transform(X_test)
# Adds the first and second principal components as new features to X_test
X_test['pca_1'] = pca_test[:,0]
X_test['pca_2'] = pca_test[:,1]
# print("X_test['pca_1']: ", X_test['pca_1'])
# print("X_test['pca_2']: ", X_test['pca_2'])

# 3. UMAP (2D): 
# UMAP (Uniform Manifold Approximation and Projection) is a nonlinear dimensionality reduction technique
# Creates a UMAP model to reduce your data to 2 dimensions.
# n_components=2 means we only keep 2 new features.
# random_state=42 ensures reproducibility of the UMAP results.
umap_model = umap.UMAP(n_components=2, random_state=42)

# Fit: UMAP learns a nonlinear manifold from the training data
    # It tries to preserve the local structure (how points are near each other) of the high-dimensional data in 2D
# Transform: Projects the training data onto the 2D manifold
# umap_train is now a (num_train_samples, 2) array representing your training data in a new 2-dimensional space
umap_train = umap_model.fit_transform(X_train)
X_train['umap_1'] = umap_train[:,0]
X_train['umap_2'] = umap_train[:,1]
umap_test = umap_model.transform(X_test)
X_test['umap_1'] = umap_test[:,0]
X_test['umap_2'] = umap_test[:,1]

# 4. Isolation Forest anomaly score
iso = IsolationForest(contamination=0.05, random_state=42)
X_train['iso_score'] = -iso.fit_predict(X_train)
X_test['iso_score'] = -iso.predict(X_test)

# Encode Target
# Creates an instance of LabelEncoder from sklearn
# LabelEncoder is used to convert categorical labels (strings like "Low", "Medium", "High") into numbers (0, 1, 2)
# ML models like LightGBM require numeric targets for classification
le_target = LabelEncoder()
y_train_enc = le_target.fit_transform(y_train)
y_test_enc = le_target.transform(y_test)
# print("y_train_enc: ", y_train_enc)
# print("y_test_enc: ", y_test_enc)

# LightGBM Training
lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=len(le_target.classes_),
    n_estimators=500,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
lgb_model.fit(X_train, y_train_enc)

# Predictions on Test Set
# Gives the predicted class labels (the most likely class for each sample). 0,1,2
y_pred = lgb_model.predict(X_test)

# Gives the predicted probabilities for each class for each sample (student row)
# it says what is the probability of student1 getting probability of low,mediun,high and it is 70%, 20%, 10%
    # How confident machine thinks to get the probability of low, medium & high==>probability distribution
# It is used for calculating ROC-AUC or PR-AUC (probabilistic evaluation metrics)
y_proba = lgb_model.predict_proba(X_test)
# print("y_pred: ",y_pred)
# print("y_proba: ",y_proba)

# Map predictions back to human readable labels
y_pred_labels = le_target.inverse_transform(y_pred)
# print("y_pred_labels: ", y_pred_labels)

# User-Ready CSV
# Selecting the rows from your original df that correspond to the test set, and making a copy of them
# X_test.index
    # X_test is your test feature set (after the train/test split)
    # .index gives the row indices in the original DataFrame df that were selected for testing.
    # Example: If X_test has rows [5, 10, 23, 42], then .index = [5, 10, 23, 42]
# df.loc[X_test.index]
    # .loc selects rows from df using their indices
    # So this gives you all columns from the original df but only for the test samples
# .copy(): Creates a separate copy of the selected rows
    # Prevents warnings or accidental changes to the original df
    # After this, df_test is an independent DataFrame that you can modify safely
df_test = df.loc[X_test.index].copy()
df_test['predicted_grade_class'] = y_pred_labels
faculty_csv_path = os.path.join(output_csv_folder, f"student_performance_faculty_{timestamp}.csv")
df_test.to_csv(faculty_csv_path, index=False)
print(f"✅ Faculty-ready CSV saved: {faculty_csv_path}")

# Technical Enriched CSV
# Makes a separate copy of the human-ready test DataFrame (df_test)
# df_enriched will start with the original student info + predicted grades, same as df_test
df_enriched = df_test.copy()

# Loops through all columns in X_test, which includes:
# Original features (subject1, subject2, …)
# Enriched features added in the pipeline:
    # kmeans_label
    # pca_1, pca_2
    # umap_1, umap_2
    # iso_score
# Copies these columns into df_enriched, so now df_enriched contains:
    # Original student info (student_id, student_name, etc.)
    # Original features used in the model (subject1, subject2, …)
    # Engineered features (PCA, UMAP, KMeans, Isolation Forest)
    # Predicted labels (predicted_grade_class)
for col in X_test.columns:
    df_enriched[col] = X_test[col]
df_enriched['grade_class_encoded'] = y_test_enc
technical_csv_path = os.path.join(output_csv_folder, f"student_performance_enriched_test_{timestamp}.csv")
df_enriched.to_csv(technical_csv_path, index=False)
print(f"✅ Technical enriched CSV saved: {technical_csv_path}")

# Evaluation
print("\nClassification Report Below:")
print(classification_report(y_test_enc, y_pred, target_names=le_target.classes_))

# Confusion Matrix plot
# Confusion matrix is a table used to evaluate the performance of a classification model. 
# It compares the true labels (y_test_enc) with the predicted labels (y_pred)
cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le_target.classes_, yticklabels=le_target.classes_, cmap="Blues")
plt.xlabel("Prediction")
plt.ylabel("Actuals")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
cm_path = os.path.join(plots_folder, f"confusion_matrix_test_{timestamp}.png")
plt.savefig(cm_path)
# plt.show()
plt.close()

# ROC-AUC & PR-AUC (macro)
# Below is turning your encoded multi-class target into a binary indicator matrix, 
# which is often needed for multi-class ROC-AUC or PR-AUC calculations
    # y_test_enc: This is encoded test labels from the LabelEncoder (0, 2, 1, 0,...)
    # le_target.classes_ contains the original class names in order: le_target.classes_ = ["Low","Medium","High"]
    # len(le_target.classes_) = 3 (number of classes)
    # np.arange(len(le_target.classes_)): Creates an array of NOTE: "class indices" for label_binarize
        # np.arange(3) = array([0, 1, 2])
        # This tells label_binarize what classes exist in your data
    # label_binarize(y_test_enc, classes=...)
        # Converts your encoded labels into a binary NOTE:"(one-hot)" format
        # one-hot:: But here it uses indices of original value to 1. NOTE: 1 marks the correct class, 0 otherwise
y_test_bin = label_binarize(y_test_enc, classes=np.arange(len(le_target.classes_)))
# print("y_test_bin: ", y_test_bin)

# Below calculates the multi-class ROC-AUC score for your model’s predictions
# NOTE: ""ROC-AUC stands for Receiver Operating Characteristic – Area Under the Curve""
# NOTE: ROC-AUC: checks ranking ability (how well positives are ranked above negatives)NOTE
    # Here the comparison is column wise which means classwise. 
    # So if any of student gets more than lowest of low/medium/high in the dataset, it will not be very good. 
    # This is what this ROC-AUC computation look into
    # EXAMPLE: take 4 students true labeels 
        # student 1 = [1,0,0] means low
        # student 2 = [0,1,0] means medium
        # student 3 = [0,0,1] means high
        # student 4 = [0,1,0] means medium
    # probabilities
        # student 1 = [.7, .2, .1] means 70% low, 20% medium and 10% high probabilities
        # student 2 = [.3, .4, .3] means 30% low, 40% medium and 30% high probabilities
        # student 3 = [.4, .1, .5] means 40% low, 10% medium and 50% high probabilities
        # student 4 = [.2, .1, .7] means 20% low, 10% medium and 70% high probabilities
    # Now look at column wise (NOT STUDENT WISE)
        # for low 1st student and probability is .7 and for all other students probability for low <.7 & ROC is 1
        # for medium (2 students - 2nd student & 4th student)
            # 2nd student prob for medium is .4 and all other students probability for medium <.4 & ROC 1
            # But 4th student prob for medium is .1 and other students probability for medium >= .1 & reduce ROC score
        # for high: only 3rd student and probability is .5 but student has .7 probability and reduce ROC score
# This is how Total ROC is alculated. The value will be between 0 & 1
    # 1.0 → perfect model: for every class, all true students get higher probability than all non-true ones.
    # 0.5 → random guessing (no better than chance).
    # < 0.5 → systematically wrong (the model tends to rank wrong class higher than the correct one).
    # Between 0.5 and 1 → partially good: better than chance but not perfect.
roc_auc = roc_auc_score(y_test_bin, y_proba, average='macro')

# Below line computes the macro-averaged ""Precision-Recall AUC (PR-AUC)""(Area Under the Curve) for a multi-class classifier
# NOTE:PR-AUC: checks precision–recall tradeoff (especially useful with imbalanced classes)NOTE
# average_precision_score: calculates the area under the Precision–Recall curve (PR-AUC).
# NOTE:NOTE:NOTE: Precision = of all predicted positives, how many are correct. Precision = TP / (TP + FP)​
# NOTE:NOTE:NOTE: Recall = of all actual positives, how many did we catch. Recall = TP / Total Positives
# y_test_bin: the one-hot encoded true labels.
# y_proba: predicted probabilities for each class.
# average="macro": compute PR-AUC separately for each class, then take the unweighted mean across classes
# NOTE: pr_auc: lies between 0.0 (worst) and 1.0 (perfect)NOTE 
# EXAMPLE: Suppose: Student 1 = Low; Student 2 = Medium; Student 3 = High; Student 4 = Medium
    # y_test_bin =
        #  [1,0,0],   # Student 1
        #  [0,1,0],   # Student 2
        #  [0,0,1],   # Student 3
        #  [0,1,0]    # Student 4
    # y_proba =
        #  [0.7, 0.2, 0.1],   # Student 1 = confident Low
        #  [0.3, 0.4, 0.3],   # Student 2 = highest Medium
        #  [0.4, 0.1, 0.5],   # Student 3 = highest High
        #  [0.2, 0.1, 0.7]    # Student 4 = wrongly predicts High instead of Medium
# Compute PR-AUC classwise
# We treat each class column separately (One-vs-Rest == OvR)
# for class Precision–recall curve compitation use some formula: GOOGLE IT
    # Class Low (column 1):
        # True labels = [1,0,0,0]
        # Probabilities = [0.7,0.3,0.4,0.2]
        # Compute precision-recall curve, then area under → PR-AUC ~ 0.92
    # Class Medium (column 2):
        # True labels = [0,1,0,1]
        # Probabilities = [0.2,0.4,0.1,0.1]
        # Precision–recall isn’t perfect (student 4 predicted low).
        # PR-AUC ~ 0.56
    # Class High (column 3):
        # True labels = [0,0,1,0]
        # Probabilities = [0.1,0.3,0.5,0.7]
        # Student 3 is correct but Student 4 has higher prob (false positive).
        # PR-AUC ~ 0.75 
    # Macro-average = (.92 + .56 + .75) / 3 = .74
pr_auc = average_precision_score(y_test_bin, y_proba, average='macro')
print(f"Multi-class ROC-AUC (macro): {roc_auc:.4f}")
print(f"Multi-class PR-AUC (macro): {pr_auc:.4f}")

# User-Friendly Plots
# 1. Grade Distribution
# Below block of code creates and saves a bar chart of how many students fall into each predicted grade
grade_dist_path = os.path.join(plots_folder, f"grade_distribution_{timestamp}.png")
plt.figure(figsize=(6,4))
sns.countplot(x='predicted_grade_class', data=df_test, order=['Low','Medium','High'])
plt.title("Predicted Grade Distribution")
plt.tight_layout()
plt.savefig(grade_dist_path)
# plt.show()
plt.close()

# 2. Total Score Distribution by Grade
# Below code block makes a boxplot showing how students total scores are distributed within each predicted grade
    # Box = the middle 50% of student scores
    # Horizontal line inside box = median score
    # Whiskers = range of most scores
    # Dots = outliers (students with unusually high/low total scores)
# So the plot visually compares:
    # “Do High-grade students really have higher total scores than Medium and Low?”
    # “How spread out are the scores within each grade?”
score_dist_path = os.path.join(plots_folder, f"total_score_by_grade_{timestamp}.png")
plt.figure(figsize=(8,5))
sns.boxplot(x='predicted_grade_class', y='total_score', data=df_test, order=['Low','Medium','High'])
plt.title("Total Score Distribution by Predicted Grade")
plt.tight_layout()
plt.savefig(score_dist_path)
# plt.show()
plt.close()

# 3. Subject-wise Performance
# Below block generates and saves one boxplot per subject to show how students’ scores in that 
# subject vary across the predicted grades (Low, Medium, High)
# This lets faculty visually compare:
    # Do High-grade students consistently score higher in subject1, subject2, etc.?
    # Are some subjects more “spread out” (big variation) than others?
for subj in subject_cols:
    subj_path = os.path.join(plots_folder, f"{subj}_by_grade_{timestamp}.png")
    plt.figure(figsize=(8,5))
    sns.boxplot(x='predicted_grade_class', y=subj, data=df_test, order=['Low','Medium','High'])
    plt.title(f"{subj} Performance by Predicted Grade")
    plt.tight_layout()
    plt.savefig(subj_path)
    # plt.show()
    plt.close()

# SHAP (SHapley Additive exPlanations) Plots
# TreeExplainer is designed for tree-based models like LightGBM.
# It calculates the contribution of each feature to the predictions for every student
# Essentially, it tells “how much each feature pushed the prediction higher or lower”
explainer = shap.TreeExplainer(lgb_model)

# For each test sample and each feature, SHAP gives a value:
    # Positive means feature pushes model toward a higher prediction for that class
    # Negative means feature pushes model toward a lower prediction for that class
# For multi-class LightGBM, shap_values is a list of arrays, one per class.
shap_values = explainer.shap_values(X_test)

shap_bar_path = os.path.join(plots_folder, f"shap_summary_bar_{timestamp}.png")
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(shap_bar_path)
plt.show()
plt.close()

# This dot plot is more detailed than the bar plot: it shows per-student impact and feature value distribution, 
# helping you see not just which features matter, but how they affect predictions for different students
shap_dot_path = os.path.join(plots_folder, f"shap_summary_dot_{timestamp}.png")
shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
plt.tight_layout()
plt.savefig(shap_dot_path)
plt.show()
plt.close()
print("✅ SHAP plots saved")

# -------------------------
# Save Models
# -------------------------
joblib.dump(lgb_model, os.path.join(models_folder, f"lgb_model_student_performance_{timestamp}.pkl"))
joblib.dump(le_target, os.path.join(models_folder, f"label_encoder_target_{timestamp}.pkl"))
joblib.dump(kmeans, os.path.join(models_folder, f"kmeans_model_{timestamp}.pkl"))
joblib.dump(pca, os.path.join(models_folder, f"pca_model_{timestamp}.pkl"))
joblib.dump(umap_model, os.path.join(models_folder, f"umap_model_{timestamp}.pkl"))
joblib.dump(iso, os.path.join(models_folder, f"isolation_forest_model_{timestamp}.pkl"))
print("✅ All models saved")
