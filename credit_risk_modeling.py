#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete Credit Risk Modeling and Credit Scorecard Script

This script demonstrates how to build a credit risk model using logistic regression,
evaluate it, and convert the predicted probabilities into credit scores using a scorecard.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# ---------------------------
# 1. Data Loading and Preprocessing
# ---------------------------
# Load the dataset. Update the file path if needed.
data_path = 'loan_data_2007_2014.csv'
loan_data = pd.read_csv(data_path)

# Display initial dataset information
print("Initial Data Info:")
print(loan_data.info())
print("\nFirst few rows:")
print(loan_data.head())

# Drop columns with too many missing values or that are irrelevant for modeling.
# For instance, drop columns with more than 80% missing values.
na_threshold = loan_data.shape[0] * 0.8
loan_data.dropna(thresh=loan_data.shape[0]*0.2, axis=1, inplace=True)

# Drop redundant and forward-looking columns
cols_to_drop = ['id', 'member_id', 'sub_grade', 'emp_title', 'url', 'desc', 
                'title', 'zip_code', 'next_pymnt_d', 'recoveries', 
                'collection_recovery_fee', 'total_rec_prncp', 'total_rec_late_fee']
loan_data.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Create a binary target variable based on loan_status:
# For example, mark loans as 'bad' (0) if their status is one of the following,
# otherwise mark as 'good' (1).
bad_status = ['Charged Off', 'Default', 'Late (31-120 days)', 
              'Does not meet the credit policy. Status:Charged Off']
loan_data['good_bad'] = np.where(loan_data['loan_status'].isin(bad_status), 0, 1)
loan_data.drop(columns=['loan_status'], inplace=True)

print("\nTarget variable distribution:")
print(loan_data['good_bad'].value_counts(normalize=True))

# ---------------------------
# 2. Exploratory Data Analysis (EDA)
# ---------------------------
# Visualize the distribution of the target variable.
plt.figure(figsize=(8, 6))
sns.countplot(x='good_bad', data=loan_data)
plt.title('Distribution of Good vs. Bad Loans')
plt.xlabel("Loan Outcome (1: Good, 0: Bad)")
plt.ylabel("Count")
plt.show()

# ---------------------------
# 3. Data Splitting and Model Training
# ---------------------------
# Separate features and target variable
X = loan_data.drop('good_bad', axis=1)
y = loan_data['good_bad']

# For simplicity, let's fill any remaining missing values with the median (for numeric features)
X = X.apply(lambda col: col.fillna(col.median()) if col.dtype in ['float64', 'int64'] else col.fillna('Unknown'))

# Convert categorical features to dummy variables (if any)
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and testing sets (stratified split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict probabilities and classes on the test set
y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Clip probabilities to avoid exactly 0 or 1
epsilon = 1e-6
y_pred_prob_clipped = np.clip(y_pred_prob, epsilon, 1 - epsilon)

# Recompute odds and credit scores using the clipped probabilities
odds = y_pred_prob_clipped / (1 - y_pred_prob_clipped)
scores = offset - (factor * np.log(odds))

# Check the summary again
score_summary = pd.DataFrame({'Predicted_Probability': y_pred_prob_clipped, 'Credit_Score': scores})
print(score_summary.describe())

# Evaluate model performance using ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"\nROC-AUC Score: {roc_auc:.2f}")

# Display the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------
# 4. ROC Curve Plotting
# ---------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

# ---------------------------
# 5. Credit Scorecard Development
# ---------------------------
# The scorecard converts predicted probabilities into a score.
# Equation: Score = Offset - (Factor * ln(Odds))
# Where: Odds = P / (1 - P)
#
# Let's define:
#   PDO: Points to Double the Odds (commonly 20)
#   baseline_score: the score corresponding to a baseline odds, e.g., 600
#   baseline_odds: the odds corresponding to the baseline score (e.g., 1:50)

# a. Calculate the Factor (scaling factor)
PDO = 20  # Points to Double the Odds
factor = PDO / np.log(2)
print(f"\nFactor: {factor:.2f}")

# b. Calculate the Offset
baseline_score = 600
baseline_odds = 1 / 50  # For example, odds of 1:50
offset = baseline_score + (factor * np.log(baseline_odds))
print(f"Offset: {offset:.2f}")

# c. Transform predicted probabilities to credit scores
# Calculate odds for each predicted probability
odds = y_pred_prob / (1 - y_pred_prob)
# Calculate score for each applicant
scores = offset - (factor * np.log(odds))

# Create a DataFrame to summarize the results
score_summary = pd.DataFrame({'Predicted_Probability': y_pred_prob, 'Credit_Score': scores})
print("\nCredit Score Summary:")
print(score_summary.describe())

# Visualize the distribution of the credit scores
plt.figure(figsize=(8, 6))
sns.histplot(score_summary['Credit_Score'], kde=True)
plt.title("Distribution of Credit Scores")
plt.xlabel("Credit Score")
plt.ylabel("Frequency")
plt.show()

# ---------------------------
# End of Script
# ---------------------------
print("\nCredit risk modeling and scorecard development complete.")
