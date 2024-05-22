# exploratory_data_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define file paths
train_data_path = 'data/processed/train_data.csv'

# Load processed training data
print("Loading processed data...")
train_data = pd.read_csv(train_data_path)

# Display the first few rows of the dataset
print("Train Data:")
print(train_data.head())

# Basic statistics
print("Basic Statistics:")
print(train_data.describe())

# Churn distribution
print("Churn Distribution:")
print(train_data['churn'].value_counts(normalize=True))

# Plot churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='churn', data=train_data)
plt.title('Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.savefig('figures/churn_distribution.png')
plt.show()

# Correlation matrix
print("Correlation Matrix:")
correlation_matrix = train_data.corr()
print(correlation_matrix)

# Plot correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('figures/correlation_matrix.png')
plt.show()

# Distribution of numerical features
numerical_features = ['monthly_charges', 'tenure', 'total_charges']
for feature in numerical_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(train_data[feature], kde=True)
    plt.title(f'{feature.capitalize()} Distribution')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Frequency')
    plt.savefig(f'figures/{feature}_distribution.png')
    plt.show()

# Box plots for numerical features
for feature in numerical_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='churn', y=feature, data=train_data)
    plt.title(f'{feature.capitalize()} vs Churn')
    plt.xlabel('Churn')
    plt.ylabel(feature.capitalize())
    plt.savefig(f'figures/{feature}_boxplot.png')
    plt.show()

# Categorical feature distribution
categorical_features = ['gender', 'senior_citizen', 'partner', 'dependents', 'phone_service', 'multiple_lines', 
                        'internet_service', 'online_security', 'online_backup', 'device_protection', 
                        'tech_support', 'streaming_tv', 'streaming_movies', 'contract', 'paperless_billing', 
                        'payment_method']

for feature in categorical_features:
    plt.figure(figsize=(12, 6))
    sns.countplot(x=feature, hue='churn', data=train_data)
    plt.title(f'{feature.capitalize()} vs Churn')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Count')
    plt.legend(title='Churn', loc='upper right')
    plt.savefig(f'figures/{feature}_churn.png')
    plt.show()

print("Exploratory Data Analysis completed!")
