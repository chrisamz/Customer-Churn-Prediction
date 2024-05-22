# Customer Churn Prediction

## Project Overview

This project aims to develop a model to predict which customers are likely to churn (stop using a service) and identify factors contributing to churn. By accurately predicting customer churn, businesses can take proactive measures to retain customers and improve overall satisfaction. The project demonstrates skills in classification algorithms, logistic regression, decision trees, random forests, and feature importance analysis.

## Components

### 1. Data Collection and Preprocessing
Collect and preprocess data related to customer demographics, usage patterns, and service interactions. Ensure the data is clean, consistent, and ready for analysis.

- **Data Sources:** Customer demographics, service usage data, customer interaction records, and churn labels.
- **Techniques Used:** Data cleaning, normalization, handling missing values, feature engineering.

### 2. Exploratory Data Analysis (EDA)
Perform EDA to understand the data distribution, identify patterns, and gain insights into the factors contributing to customer churn.

- **Techniques Used:** Data visualization, summary statistics, correlation analysis.

### 3. Feature Engineering
Create new features based on existing data to capture essential aspects of customer behavior and interactions that may influence churn.

- **Techniques Used:** Feature extraction, transformation, and selection.

### 4. Model Building
Develop and evaluate different classification models to predict customer churn. Compare their performance to select the best model.

- **Techniques Used:** Logistic regression, decision trees, random forests.

### 5. Feature Importance Analysis
Analyze the importance of different features in predicting customer churn to understand the key drivers of churn.

- **Techniques Used:** Feature importance scores, SHAP values, permutation importance.

## Project Structure

customer_churn_prediction/
├── data/
│ ├── raw/
│ ├── processed/
├── notebooks/
│ ├── data_preprocessing.ipynb
│ ├── exploratory_data_analysis.ipynb
│ ├── feature_engineering.ipynb
│ ├── model_building.ipynb
│ ├── feature_importance.ipynb
├── models/
│ ├── logistic_regression_model.pkl
│ ├── decision_tree_model.pkl
│ ├── random_forest_model.pkl
├── src/
│ ├── data_preprocessing.py
│ ├── exploratory_data_analysis.py
│ ├── feature_engineering.py
│ ├── model_building.py
│ ├── feature_importance.py
├── README.md
├── requirements.txt
├── setup.py

## Getting Started

### Prerequisites
- Python 3.8 or above
- Required libraries listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/customer_churn_prediction.git
   cd customer_churn_prediction
   
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    
### Data Preparation

1. Place raw data files in the data/raw/ directory.
2. Run the data preprocessing script to prepare the data:
    ```bash
    python src/data_preprocessing.py
    
### Running the Notebooks

Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    
2. Open and run the notebooks in the notebooks/ directory to preprocess data, perform EDA, engineer features, build models, and analyze feature importance:
 - data_preprocessing.ipynb
 - exploratory_data_analysis.ipynb
 - feature_engineering.ipynb
 - model_building.ipynb
 - feature_importance.ipynb
   
### Training Models

1. Train the logistic regression model:
    ```bash
    python src/model_building.py --model logistic_regression
    
2. Train the decision tree model:
    ```bash
    python src/model_building.py --model decision_tree
    
3. Train the random forest model:
    ```bash
    python src/model_building.py --model random_forest
    
### Results and Evaluation

 - Logistic Regression: Evaluate the model using accuracy, precision, recall, F1-score, and ROC-AUC.
 - Decision Tree: Evaluate the model using accuracy, precision, recall, F1-score, and ROC-AUC.
 - Random Forest: Evaluate the model using accuracy, precision, recall, F1-score, and ROC-AUC.
   
### Feature Importance Analysis

Identify the key features contributing to customer churn using feature importance scores, SHAP values, and permutation importance.

### Contributing

We welcome contributions from the community. Please follow these steps:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Commit your changes (git commit -am 'Add new feature').
4. Push to the branch (git push origin feature-branch).
5. Create a new Pull Request.
   
### License

This project is licensed under the MIT License. See the LICENSE file for details.

### Acknowledgments

 - Thanks to all contributors and supporters of this project.
 - Special thanks to the data scientists and engineers who provided insights and data.
