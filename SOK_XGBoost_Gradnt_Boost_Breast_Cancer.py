"""
BUSINESS FUNCTIONALITY
======================
This program is to predict whether a patient has breast cancer or not, 
using clinical data (like mean radius, texture, area, etc.) from a dataset
This is a binary classification using machine learning (XGBoost)
    Business Use Case:
        Hospitals or diagnostic centers can:
        Use this program as a decision-support tool for radiologists/doctors
        Reduce manual errors in early cancer detection
        Generate automated risk reports for patients
        Prioritize critical cases (cancer positive) for quicker follow-up

    EXAMPLE USAGE SCENARIO: Scenario: A diagnostic lab collects data from 500 patients
        After collecting 30 features per patient:
        Staff runs this program
        Outputs are generated: predictions, charts, and reports
        Doctors receive filtered patient lists (cancer vs non-cancer)
        Critical patients are called in first
        Lab saves the trained model for use next month

TECHNICAL FUNCTIONALITY — HIGH LEVEL
====================================
    ML Model: XGBoost (eXtreme Gradient Boosting)
        This is a high-performance, tree-based machine learning model that is fast, accurate, and interpretable
"""
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
# Load dataset (Once loaded and saved; then reuse it locally)
data = load_breast_cancer()

# Create DataFrame with features
df = pd.DataFrame(data.data, columns=data.feature_names)

# Map target: 0 -> 'cancer', 1 -> 'no cancer'
df['target'] = pd.Series(data.target).map({0: 'cancer', 1: 'no cancer'})

# Add Patient_ID and Patient_Name
df['Patient_ID'] = ['P' + str(i+1).zfill(3) for i in range(len(df))]
df['Patient_Name'] = ['Patient_' + str(i+1) for i in range(len(df))]

# Rearrange columns
cols = ['Patient_ID', 'Patient_Name'] + [col for col in df.columns if col not in ['Patient_ID', 'Patient_Name']]
df = df[cols]

# Save full dataset to CSV
df.to_csv("E:\\AI_Training\\CPT_Data\\Inbound_Data\\Breast_Cancer_Prediction_Data.csv", index=False)
"""

# Load data from CSV
df = pd.read_csv("E:\\AI_Training\\CPT_Data\\Inbound_Data\\Breast_Cancer_Prediction_Data.csv")
# print("ORIGINAL DF:\n", df)
# Prepare features(X) and numeric target(y) for training
    # dropping columns not included for training
X = df.drop(['target', 'Patient_ID', 'Patient_Name'], axis=1) 

# 0 = cancer, 1 = no cancer
y = df['target'].map({'cancer': 0, 'no cancer': 1})  
# print("X After Dropping 'target', 'Patient_ID', 'Patient_Name': \n", X)
# print("y: ", y)
# print("DF\n", df)

# Train/test split 80/20(this can be any propo)
# Here X is features and y is corresponding targets
    # X_train: 80% of X >> used to train the model
    # X_test: 20% of X >> used to test the model
    # y_train: 80% of y >> target values for training
    # y_test: 20% of y >> target values for testing

    # X: Input features (all columns except the target & personal info)
    # y: Target labels (what to predict, e.g., 0 for cancer, 1 for no cancer)
    # train_test_split(...): Function that splits X and y into training and testing subsets
    # test_size=0.2: 20% of the data will go into the test set, and 80% will go into the training set
    # random_state=42: Ensures the split is the same every time you run the code (for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# PRINT TRAIN SET AND TEST SET
# print("X_train: Features to identify cancer below\n", X_train)
# print("y_train: Result cancer or not\n", y_train)

# XGBoost DMatrices
    # DMatrix - It's a highly efficient data structure optimized for training with XGBoost
    # DMatrix - It supports missing value handling, sparsity-aware algorithms, and faster computation
        # sparsity-aware:
            # Recognizes missing values (like np.nan) automatically
            # Skips over missing values efficiently during tree split decisions
            # Optimizes memory and speed by not storing all the zero/NaN values
        # xgb.DMatrix(): Stores data automatically in compressed sparse row or compressed sparse column(CSC) format
        # During Trng; XGBoost learns the best direction to send missing values in decision trees (left or right),
            # instead of dropping or imputing them.
# Converts the training features (X_train) and labels (y_train) into an optimized internal data structure (DMatrix) used by XGBoost.
    # X_train: Feature matrix (can be NumPy array, pandas DataFrame, or SciPy sparse matrix)
    # label=y_train: Target labels for supervised training. It attaches the labels to the data
    # dtrain — an instance of xgb.DMatrix; an Object
dtrain = xgb.DMatrix(X_train, label=y_train)

# Below is same as previous line: dtrain = xgb.DMatrix(X_train, label=y_train)
# NOTE: Below is being test data 'label=y_test' not neccessary UNLESS accuracy used.
dtest = xgb.DMatrix(X_test, label=y_test)
# print("dtrain: ", dtrain)
# print("dtest: ", dtest)

# XGBoost parameters: START
# hyperparameters Setting (GENERAL)
    # Hyperparameters: Configuration setted before training ML model.This control how the model learns from data
    # They are not learned from the data but are manually set by the developer or chosen 
    # via techniques like grid search or random search.
# 'objective': 'binary:logistic'
    # Specifies the learning task: binary classification with logistic regression output. 
    # The model will predict probabilities between 0 and 1.
# 'max_depth': 4
    # Maximum depth of a tree. Controls model complexity and helps prevent overfitting. 
    # Shallower trees generalize better.
# 'eta': 0.1
    # Learning rate (also called shrinkage). Controls the step size during optimization. 
    # Lower values slow down learning but increase performance stability.
# 'eval_metric': 'logloss'
    # Evaluation metric: logarithmic loss, suitable for probabilistic binary classification. Lower is better.
# 'seed': 42
    # Random seed for reproducibility.
params = {
    'objective': 'binary:logistic',
    'max_depth': 4,
    'eta': 0.1,
    'eval_metric': 'logloss',
    'seed': 42
}

# Train model
    # num_boost_round=100: 
        # Number of boosting iterations (trees to build). This will:
        # Build 100 decision trees sequentially
        # Each tree tries to fix the errors of the previous one (gradient boosting)
        # Use logistic regression as the output (binary classification)        
    # Output model: Trained XGBoost booster object
        # model can be used for prediction, saving, or plotting feature importance
model = xgb.train(params, dtrain, num_boost_round=50)

# Predict probabilities for the test data using the trained XGBoost model.
    # dtest: The test data in DMatrix format.
    # y_pred_prob: A list/array of probabilities (FLOAT VALUES between 0 and 1) that indicate the likelihood of the positive class (label 1, i.e., "no cancer").
y_pred_prob = model.predict(dtest)
# print("\ny pred prob ORIGINAL PREDICTION BY MODEL:\n",y_pred_prob)

# Convert float values between 0 & 1 to Zero's and One's
y_pred = [1 if prob > 0.7 else 0 for prob in y_pred_prob]
# print("\ny pred prob CONVERTED TO zeros and ones:\n",y_pred)

# Convert to readable labels instead of 1 , 0
y_pred_label = ['cancer' if pred == 0 else 'no cancer' for pred in y_pred]
# print("\ny pred prob ORIGINAL AFTER CONVERSION (cancer or no cancer) OF 0s & 1s:\n",y_pred_label)

# Prepare DataFrame with predictions
df_predictions = df.loc[y_test.index].copy()
df_predictions['predicted'] = y_pred_label

# Remove target column for output files
df_predictions_no_target = df_predictions.drop(columns=['target'])

# Save all predictions
df_predictions_no_target.to_csv("E:\\AI_Training\\CPT_Data\\Outbound_Data\\All_Patient_Predictions_OUT.csv", index=False)

# Save only cancer cases
cancer_patients_df = df_predictions_no_target[df_predictions_no_target['predicted'] == 'cancer']
cancer_patients_df.to_csv("E:\\AI_Training\\CPT_Data\\Outbound_Data\\Cancer_Patient_Predictions_OUT.csv", index=False)

# Save only non-cancer cases
non_cancer_patients_df = df_predictions_no_target[df_predictions_no_target['predicted'] == 'no cancer']
non_cancer_patients_df.to_csv("E:\\AI_Training\\CPT_Data\\Outbound_Data\\Non_Cancer_Patient_Predictions_OUT.csv", index=False)

# Print counts
# print(f"Total patients predicted: {len(df_predictions_no_target)}")
# print(f"Cancer patients predicted: {len(cancer_patients_df)}")
# print(f"Non-cancer patients predicted: {len(non_cancer_patients_df)}")

# Evaluation
# It calculates the classification accuracy of your XGBoost model.
# accuracy_score() compares the true labels (y_test) with the predicted labels (y_pred).
# NOTE Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)
    # For Example:
    # y_test = [0, 1, 0, 1, 0]
    # y_pred = [0, 1, 1, 1, 0]
        # So; ACCURACY >> 4/5 = .8 which is 80%
    # You DO NOT HAVE ANY ROLE HERE other than passing y_test & y_pred properly
# Provides how many patients were correctly classified as cancer / non-cancer, out of all patients tested
# NOTE: WHO DECIDES accuracy Mr.???
    # Data Scientist / ML Engineer: Based on baseline data, experimentation, or benchmarks
    # Product Owner / Project Manager: Based on business needs — e.g., accuracy above 85%
    # Medical Expert / Healthcare Regulator: Based on risk tolerance — e.g., "We can’t miss more than 1 in 100 cancer cases"
    # Compliance Team / Audit: May demand strict thresholds — 95%+ recall for positive cases
"""
In business, we often celebrate 90% accuracy. But in healthcare, that could mean 10 out of 100 patients 
go undetected — which is unacceptable. That’s why accuracy is just a starting point — 
the real story comes from precision, recall, and confusion matrix.”
"""
# You can also pass "normalize=True, sample_weight=None" to accuracy = accuracy_score(y_test, y_pred)
    # normalize=True (default): Returns a fraction (e.g., 0.85)
    # normalize=False: Returns the count of correct predictions
    # sample_weight: Optional array of weights to give more importance to some samples. See below
        # weights = [1, 2, 1, 1, 3]
        # accuracy_score(y_test, y_pred, sample_weight=weights)    
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy############:", accuracy)

# Classification Report is important and it printed report details below
    # Precision: Out of all predicted positives, how many were actually correct
    # Recall: Out of all actual positives, how many did the model successfully find
    # F1-score: Harmonic mean of precision and recall — balances both
    # Support: Number of actual samples in each class in the test set
    # Macro avg: Simple average of scores across classes. Treats all classes equally.
    # Weighted avg: Weighted average by number of samples in each class (support).    
# EXAMPLE CLASSIFICATION REPORT 
    # Precision (0.94): Out of all patients predicted as cancer, 94% were truly cancer
    # Recall (0.91): Out of all actual cancer patients, 91% were correctly identified
    # F1-score (0.93): Combined score (good balance between precision & recall)
    # Support (45): There were 45 actual cancer cases in the test set
# BELOW IS VERY IMPORTANT TO UNDERSTAND
    # TP: True Positive == You predicted YES, and the actual answer was also YES. CORRECT PREDICTION
    # TN: True Negative	== You predicted NO, and the actual answer was also NO. CORRECT PREDICTION
    # FP: False Positive == You predicted YES, but the actual answer was NO. WRONG PREDICTION
    # FN: False Negative == You predicted NO, but the actual answer was YES. WRONG PREDICTION
# NOTE: If your model predicts "no cancer" for everyone, you might still get high accuracy — 
# But recall for cancer = 0, which is dangerous

"""
We have a test dataset of 20 patients.
The ground truth (actual condition) and the model’s predictions resulted in the below:
    True Positives (TP) = 4
        Model predicted cancer, and patient actually had cancer
    True Negatives (TN) = 10
        Model predicted no cancer, and patient actually had no cancer
    False Positives (FP) = 3
        Model predicted cancer, but patient actually had no cancer
    False Negatives (FN) = 3
        Model predicted no cancer, but patient actually had cancer
So total patients = TP + TN + FP + FN = 4 + 10 + 3 + 3 = 20 patients

NOTE:Next: we compute metrics for each class (0 = no cancer, 1 = cancer)
    NOTE: 1. Precision: Precision Formula = TP / (TP + FP)
    This means: Of all the people predicted as positive (cancer), how many were actually positive?
        For Class "1" (Cancer):
            TP₁ = 4 (correctly predicted cancer)
            FP₁ = 3 (predicted cancer but no cancer)
            Formula: Precision₁ = TP₁ / (TP₁ + FP₁) = 4 / (4 + 3) = 4 / 7 = 0.5714
            Meaning: When the model predicted “cancer”, it was correct 57.14% of the time.
        For Class 0 (No Cancer):
        Now, when computing for Class 0, we treat “no cancer” as the positive class.
            TP₀ = TN = 10 (correctly predicted no cancer)
            FP₀ = FN = 3 (predicted no cancer, but actually had cancer)
            Formula: Precision₀ = TP₀ / (TP₀ + FP₀) = 10 / (10 + 3) = 10 / 13 = 0.7692
            Meaning: When the model predicted “no cancer”, it was correct 76.92% of the time.
    
    2. Recall: Recall Formula = TP / (TP + FN)
    This means: Of all the people who actually had the disease, how many did the model correctly detect?
        For Class 1 (Cancer):
            TP₁ = 4 (predicted cancer, and had cancer)
            FN₁ = 3 (predicted no cancer, but had cancer)
            Formula: Recall₁ = TP₁ / (TP₁ + FN₁) = 4 / (4 + 3) = 4 / 7 = 0.5714
            Meaning: Out of all the actual cancer cases, the model caught 57.14% of them.
        For Class 0 (No Cancer):
        We treat “no cancer” as positive:
            TP₀ = 10 (predicted no cancer, actually no cancer)
            FN₀ = 3 (predicted cancer, but actually had no cancer)
            Formula: Recall₀ = TP₀ / (TP₀ + FN₀) = 10 / (10 + 3) = 10 / 13 = 0.7692
            Meaning: Out of all the actual no-cancer patients, the model correctly said no cancer for 76.92% of them.

    3. F1-Score F1-Score Formula: F1 = 2 × (Precision × Recall) / (Precision + Recall)
    This combines both precision and recall into one score.
        For Class 1 (Cancer):
            We already got: Precision₁ = 0.5714 & Recall₁ = 0.5714
            Formula: F1₁ = 2 × (0.5714 × 0.5714) / (0.5714 + 0.5714) = 2 × 0.3265 / 1.1428 = 0.6529 / 1.1428 = 0.5714
            Meaning: Balanced score for Class 1 is 57.14%
        For Class 0 (No Cancer):
            Precision₀ = 0.7692 & Recall₀ = 0.7692
            Formula: F1₀ = 2 × (0.7692 × 0.7692) / (0.7692 + 0.7692) = 2 × 0.5917 / 1.5384 = 1.1834 / 1.5384 = 0.7692
            Meaning: Balanced score for Class 0 is 76.92%
    
    4. Support
    Support = Number of actual occurrences of the class in test set
        From the original counts:
        Actual Class 1 (cancer) = TP + FN = 4 + 3 = 7
        Actual Class 0 (no cancer) = TN + FP = 10 + 3 = 13
        Meaning:
            There are 7 actual cancer patients.
            There are 13 actual no-cancer patients.

    5. Macro Average
    Formula: Macro Average = (Metric for Class 0 + Metric for Class 1) / 2
    It gives equal importance to both classes, no matter their size.
        Macro Precision: = (0.5714 + 0.7692) / 2 = 1.3406 / 2 = 0.6703
        Macro Recall: = (0.5714 + 0.7692) / 2 = 0.6703
        Macro F1: = (0.5714 + 0.7692) / 2 = 0.6703
        Meaning: The model performs at about 67.03% average across both classes equally.

    6. Weighted Average
    Formula: Weighted Avg = (Score₀ × Support₀ + Score₁ × Support₁) / Total Support
    Here: Support₀ = 13, Support₁ = 7, Total = 20. CHECK SUPPORT ABOVE
        Weighted Precision: = (0.7692 × 13 + 0.5714 × 7) / 20 = (10.0 + 4.0) / 20 = 14 / 20 = 0.7000
        Weighted Recall: = (0.7692 × 13 + 0.5714 × 7) / 20 = 0.7000
        Weighted F1: = (0.7692 × 13 + 0.5714 × 7) / 20 = 0.7000
        Meaning: Because there are more no-cancer cases, the overall weighted score is 70.00% 
        — gives more importance to Class 0 (majority class).

The Final Classification Report Will be Shown as Below.
Class             Precision     Recall      F1-Score    Support
0 (NoCancer)      0.7692        0.7692      0.7692      13      
1 (Cancer)        0.5714        0.5714      0.5714       7       
Macro avg         0.6703        0.6703      0.6703      20      
Weighted avg      0.7000        0.7000      0.7000      20      
"""
cls_report = classification_report(y_test, y_pred, target_names=["cancer", "no cancer"])
print("Classification Report:\n", cls_report)

# Save model for reuse
model.save_model("E:\\AI_Training\\CPT_Data\\Models_Created\\Breast_Cancer_Prediction_Model.json")

# Feature importance
# This plot shows which features the model used most frequently to make decisions — 
# helping you understand the relative importance of input features in the trained model
    # model: Your trained XGBoost model (usually from xgb.XGBClassifier() or xgb.train())
    # max_num_features=10: Show only the top 10 most important features 
    # height=0.5: Controls the height of each bar in the plot
    # importance_type='weight': Shows how many times a feature was used in all trees. Other values
        # 'weight': Count the number of times a feature appears in all decision trees (regardless of split quality)
        # 'gain': Average gain of splits using the feature → better at showing impact
        # 'cover': Average number of samples affected by splits using the feature
xgb.plot_importance(model, max_num_features=10, height=0.5, importance_type='weight')
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.savefig("E:\\AI_Training\\CPT_Data\\Plots\\Breast_Cancer_Feature_Importance_OUT.png")
plt.show()

# Bar chart of predictions
predicted_counts = pd.Series(y_pred_label).value_counts()
plt.figure(figsize=(6, 4))
bars = sns.barplot(x=predicted_counts.index, y=predicted_counts.values, palette="Set2")
for bar in bars.patches:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, int(height), ha='center', va='bottom')
plt.title("Predicted Cancer vs Non-Cancer Cases")
plt.xlabel("Prediction")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("E:\\AI_Training\\CPT_Data\\Plots\\Breast_Cancer_Barchart.png")
plt.show()


# Confusion matrix
# Plot shows below itesm
    # TP: True Positive == You predicted YES, and the actual answer was also YES. Prediction is correct
    # TN: True Negative	== You predicted NO, and the actual answer was also NO. Prediction is correct
    # FP: False Positive == You predicted YES, but the actual answer was NO. This is a wrong prediction
    # FN: False Negative == You predicted NO, but the actual answer was YES. This is also a wrong prediction
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["cancer", "no cancer"],
            yticklabels=["cancer", "no cancer"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("E:\\AI_Training\\CPT_Data\\Plots\\Breast_Cancer_CM_OUT.png")
plt.show()

"""
=====================================================================================
Now; we explain everything in plain, human language using a relatable scenario.

👨‍⚕️ Imagine the Scenario
Doctor gave:
    100 patient records.
    Out of these, 80 were used by the computer to learn.
    Now the system is tested on the remaining 20 patients to see how smart it has become.
Suppose the system made the following predictions:
    It said 5 people have cancer.
    Of those 5, 3 really had cancer, and 2 did not.
    There were 4 real cancer cases among the 20 patients.
    Of those 4, the system caught 3, but missed 1 (wrongly said "no cancer").

✅ 1. Precision (For Cancer Class)
    "Of the people the system said have cancer, how many actually had cancer?"
    Example:
        System said: 5 have cancer.
        Out of those, 3 actually had cancer.
    🔢 Precision = Correctly said cancer / All said cancer
    = 3 / 5
    = 0.60 or 60%

    📣 Doctor Explanation:
    "Doctor, when the system tells you a person has cancer, you can trust it 60% of the time. The rest might be false alarms."

✅ 2. Recall (For Cancer Class)
    "Of the people who truly had cancer, how many did the system find?"

    Example:
    Total real cancer cases: 4

    System correctly identified: 3

    🔢 Recall = Correctly found cancer / All real cancer cases
    = 3 / 4
    = 0.75 or 75%

    📣 Doctor Explanation:
    "Doctor, the system was able to find 75% of all real cancer cases. It missed 1 person who actually had cancer."

✅ 3. F1-Score (For Cancer Class)
    "A combined score — it balances how careful the system is when saying cancer (precision) and how well it catches all real cancer (recall)."

    🔢 F1 = 2 × (Precision × Recall) / (Precision + Recall)
    = 2 × (0.60 × 0.75) / (0.60 + 0.75)
    = 2 × 0.45 / 1.35 = 0.666... ≈ 0.67 or 67%

    📣 Doctor Explanation:
    "Doctor, if you want a balanced view of how precise and complete the cancer predictions are, it's about 67% reliable."    

✅ 4. Support (For Cancer Class)
    "How many actual cancer cases were in the test group?"

    📣 Answer: 4 people
    📣 Doctor Explanation:
    "Doctor, out of the 20 patients tested, 4 truly had cancer."

✅ 5. Precision (For No Cancer)
    "Of all the people the system said don’t have cancer, how many actually didn’t?"

    Example:
    System said 15 people have no cancer.

    Of those, 14 were really cancer-free.

    1 had cancer but was missed.

    🔢 Precision = 14 / 15 = 0.933 ≈ 93.3%
    📣 Doctor Explanation:
    "Doctor, when the system tells you someone is cancer-free, you can trust it 93% of the time."

✅ 6. Recall (For No Cancer)
    "Of all the people who actually didn’t have cancer, how many did the system correctly identify as no cancer?"

    Real no-cancer people: 16

    System caught 14 of them correctly.

    🔢 Recall = 14 / 16 = 0.875 or 87.5%
    📣 Doctor Explanation:
    "Doctor, the system correctly identified 87.5% of healthy patients."

✅ 7. F1-Score (For No Cancer)
    🔢 F1 = 2 × (Precision × Recall) / (Precision + Recall)
    = 2 × (0.933 × 0.875) / (0.933 + 0.875)
    ≈ 2 × 0.816 / 1.808 ≈ 0.902 or 90.2%

    📣 Doctor Explanation:
    "Doctor, for healthy patients, the system gives a balanced and reliable performance of about 90%."

✅ 8. Support (For No Cancer)
    "How many people in the test group actually had no cancer?"

    📣 Answer: 16 people

✅ 9. Macro Average
    "Average score for cancer and no cancer — treated equally."

    🔢 Macro Precision = (60% + 93.3%) / 2 = 76.65%
    🔢 Macro Recall = (75% + 87.5%) / 2 = 81.25%
    🔢 Macro F1 = (67% + 90.2%) / 2 = 78.6%
    📣 Doctor Explanation:
    "Doctor, if we treat both classes equally — regardless of how many — the system performs at around 78–81%."

✅ 10. Weighted Average
    "Same as above, but gives more weight to common cases (here, no cancer)."
   
    Cancer: 4 cases

    No Cancer: 16 cases
    Weighted F1 = (4×67%)+(16×90.2%)/20 = (268+1443.2)/20 = 1711.2/20 = ~85.56
    Doctor Explanation:
    "Doctor, since most people are healthy, the overall weighted score (more real-world realistic) is about 85.5%."
"""
