# model_building.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import argparse
import os

# Define file paths
processed_train_data_path = 'data/processed/train_data.csv'
processed_test_data_path = 'data/processed/test_data.csv'
model_dir = 'models'

# Create directories if they don't exist
os.makedirs(model_dir, exist_ok=True)

# Load processed training and testing data
print("Loading processed data...")
train_data = pd.read_csv(processed_train_data_path)
test_data = pd.read_csv(processed_test_data_path)

# Prepare data
X_train = train_data.drop(columns=['churn'])
y_train = train_data['churn']

X_test = test_data.drop(columns=['churn'])
y_test = test_data['churn']

# Function to train and evaluate a model
def train_and_evaluate(model, model_name):
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    model_path = os.path.join(model_dir, f"{model_name}_model.pkl")
    joblib.dump(model, model_path)
    print(f"{model_name} model saved to {model_path}")

# Train and evaluate logistic regression model
logistic_regression = LogisticRegression(random_state=42)
train_and_evaluate(logistic_regression, 'Logistic_Regression')

# Train and evaluate decision tree model
decision_tree = DecisionTreeClassifier(random_state=42)
train_and_evaluate(decision_tree, 'Decision_Tree')

# Train and evaluate random forest model
random_forest = RandomForestClassifier(random_state=42)
train_and_evaluate(random_forest, 'Random_Forest')

print("Model training and evaluation completed!")
