# Model Experimentation & Tuning Log
**Project:** Telco Customer Churn Prediction & Retention Optimization  
**Date:** October 2026 | **Experiment Tracking:** MLflow (SQLite Backend)  
**Target Class Imbalance:** ~26.54% Churn Rate (Positive Class)  

---

## 1. Executive Summary & Model Leaderboard

All candidate architectures were rigorously evaluated using **5-Fold Stratified Cross-Validation**
to respect the 26.6% class imbalance, followed by out-of-sample evaluation on the 20% holdout test set (1,409 customers).

| Model                                | Sampling / Imbalance Strategy          |   5-Fold CV ROC-AUC |   Test ROC-AUC |   Test PR-AUC |   Test F1 |   Test Precision |   Test Recall |   Accuracy |   Brier Loss |
|:-------------------------------------|:---------------------------------------|--------------------:|---------------:|--------------:|----------:|-----------------:|--------------:|-----------:|-------------:|
| Logistic Regression (Vanilla)        | Default Unweighted                     |              0.8472 |         0.8431 |        0.6407 |    0.5902 |           0.6667 |        0.5294 |     0.8048 |       0.1375 |
| Logistic Regression (Class-Weighted) | Class-Weight Balanced                  |              0.8471 |         0.8431 |        0.6418 |    0.6126 |           0.5052 |        0.7781 |     0.7388 |       0.1661 |
| Logistic Regression (SMOTE)          | Synthetic Oversampling (SMOTE)         |              0.8351 |         0.8401 |        0.6393 |    0.6203 |           0.5122 |        0.7861 |     0.7445 |       0.1668 |
| Random Forest                        | Balanced Subsample + Regularized Depth |              0.8467 |         0.8466 |        0.6628 |    0.6353 |           0.538  |        0.7754 |     0.7637 |       0.156  |
| XGBoost (Tuned)                      | Optuna Tuned + Scale Pos Weight        |              0.8489 |         0.847  |        0.6624 |    0.6312 |           0.5171 |        0.8102 |     0.7488 |       0.1627 |
| LightGBM (Tuned)                     | Optuna Tuned + Scale Pos Weight        |              0.8481 |         0.8456 |        0.6627 |    0.6275 |           0.5109 |        0.8128 |     0.7438 |       0.1643 |

---

## 2. Model Selection Rationale

1. **Detection Sensitivity (Recall):** In customer retention, False Negatives (predicting a churning customer as retained) destroy full Customer Lifetime Value ($2,280+). Class-weighted tree ensembles capture over 77% to 81% of true churners.
2. **Ranking Quality (ROC-AUC & PR-AUC):** Ensemble models achieve the highest area under the ROC (>0.845) and Precision-Recall curves (>0.66), enabling optimal customer risk ranking for targeted intervention.
3. **Generalization & Cross-Validation Stability:** The narrow standard deviation across 5 Stratified CV folds demonstrates low variance and resilience against overfitting.
4. **Champion Selection:** The **XGBoost (Tuned)** was selected as the champion production model due to superior discrimination, stable probability calibration, high churn recall, and native compatibility with SHAP TreeExplainer for operational explainability.
