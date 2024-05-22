# feature_importance.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define file paths
processed_train_data_path = 'data/processed/train_data.csv'
model_path = 'models/Random_Forest_model.pkl'
feature_importance_plot_path = 'figures/feature_importance.png'

# Create directories if they don't exist
os.makedirs(os.path.dirname(feature_importance_plot_path), exist_ok=True)

# Load processed training data
print("Loading processed data...")
train_data = pd.read_csv(processed_train_data_path)

# Prepare data
X_train = train_data.drop(columns=['churn'])
y_train = train_data['churn']

# Load trained random forest model
print("Loading trained model...")
model = joblib.load(model_path)

# Calculate feature importance
print("Calculating feature importance...")
feature_importances = model.feature_importances_
feature_names = X_train.columns

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display feature importance
print("Feature Importance:")
print(importance_df)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(feature_importance_plot_path)
plt.show()

print("Feature importance analysis completed!")
