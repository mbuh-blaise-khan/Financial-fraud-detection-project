# Accounting Fraud Detection Using Machine Learning

## Project Overview

This project develops a machine-learning-based system for detecting accounting fraud from financial statement and accounting data.

The task is formulated as a supervised binary classification problem:

- `0` = Non-fraudulent observation
- `1` = Fraudulent/misstatement observation

## Dataset

The project uses the publicly available `data_FraudDetection_JAR2020.csv` dataset associated with:

Bao, Y., Ke, B., Li, B., Yu, J., & Zhang, J. (2020).
"Detecting Accounting Fraud in Publicly Traded U.S. Firms Using a Machine Learning Approach."
Journal of Accounting Research, 58(1), 199–235.

The dataset contains approximately 146,000 firm-year observations from publicly traded U.S. firms covering approximately 1990–2014.

## Current Progress

- Dataset loaded and inspected
- Target variable identified as `misstate`
- Initial data-quality assessment performed
- Initial time-based train/test strategy established
- Scikit-learn preprocessing pipeline implemented
- Logistic Regression baseline established
- Initial evaluation metrics and confusion matrix implemented

## Planned Work

1. Exploratory Data Analysis
2. Data-quality and missing-value analysis
3. Leakage assessment
4. Feature preparation
5. Feature-set comparison
6. Time-aware validation
7. Class-imbalance analysis
8. Multiple machine-learning models
9. Hyperparameter tuning
10. Model evaluation
11. Feature importance and interpretability
12. SHAP analysis where practical
13. Error analysis
14. Final model selection
15. Research conclusions and limitations

## Important Research Principle

The model is intended to identify predictive patterns associated with accounting fraud. Model findings should not be interpreted as evidence of causal relationships.

## Status

🚧 Research and development in progress.