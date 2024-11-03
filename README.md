# Fraud Detection System Using Machine Learning

## Introduction
This project builds a fraud detection system using machine learning techniques, based on the Credit Card Fraud Detection Dataset.

## Dataset
- **Source**: [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)
- **Description**: This dataset contains anonymized transaction features and labels (0 for non-fraud, 1 for fraud).
- **Download Instructions**: Download the dataset from Kaggle, save it in the `data/` folder of the project, and rename it to `creditcard.csv` if necessary.

## Environment Setup
- **Python** and **Jupyter Notebook**
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, `seaborn`

## Data Preprocessing
1. Handled class imbalance using SMOTE and under-sampling.
2. Scaled numerical features for model compatibility.

## Model Training
- Models used: Logistic Regression, Random Forest
- Class imbalance managed with `class_weight='balanced'`.

## Evaluation
### Confusion Matrix
Shows the performance of the model on non-fraud and fraud cases.

### Classification Report
- Precision: 6% for fraud cases, meaning many legitimate transactions are flagged as fraud.
- Recall: 93% for fraud cases, capturing most actual fraud cases.

### ROC Curve
- AUC score of 0.95, indicating strong separation between classes.

## Improvements
Future improvements include adjusting the classification threshold, feature engineering, and experimenting with advanced models like XGBoost.

## Conclusion
The model is highly effective in detecting fraud, with excellent recall and a high AUC score, though it could benefit from precision improvements.
