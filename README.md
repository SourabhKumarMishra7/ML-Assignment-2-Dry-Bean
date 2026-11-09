
# Dry Bean Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to classify dry bean varieties
using five machine learning classification algorithms.

## 2. Dataset Description

Dataset: UCI Dry Bean Dataset

Total instances: 13,611

Total columns: 17

Input features: 16

Target column: Class

Number of classes: 7

## 3. Machine Learning Models

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

## 4. Evaluation Metrics

The models are evaluated using:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## 5. Dataset Preparation

- Dataset split: 80% training and 20% testing.
- Random state: 42.
- Stratified sampling was used.
- StandardScaler was applied to models that require feature scaling.

## 6. Model Comparison

The performance comparison is available in:

results/model_comparison.csv

## 7. Streamlit Application

The Streamlit application allows users to upload a labelled test CSV,
select a trained model, and view evaluation metrics and a confusion matrix.

## 8. Project Files

- app.py
- requirements.txt
- test_data.csv
- model/
- results/
- README.md

## 9. Conclusion

The best-performing model will be identified by comparing
the evaluation metrics of all five classifiers.
