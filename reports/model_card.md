# Model Card: Telco Customer Churn Predictor

## 1. Model Details
- **Organization:** Regional Telecommunications Provider — Data Science & Retention Analytics
- **Model Date:** October 2026
- **Model Version:** v1.0.0 (Production Candidate)
- **Model Type:** Extreme Gradient Boosting Classifier (`xgboost.XGBClassifier`)
- **Imbalance Handling:** `scale_pos_weight = 2.768` (Negative / Positive class ratio) + Stratified 5-Fold Cross Validation
- **Hyperparameter Optimization:** Optuna Bayesian Search (12 trials with logloss objective)
- **Primary Dependencies:** Python 3.11+, Scikit-learn, XGBoost, SHAP, MLflow, Optuna

---

## 2. Intended Use
- **Primary Intended Uses:**
  - Identifying customers at risk of terminating telecom service within the upcoming 90 days.
  - Prioritizing outbound customer retention retention incentives and contract extension campaigns.
  - Generating individualized risk drivers via SHAP TreeExplainer for call-center retention agents.
- **Out-of-Scope Uses:**
  - Credit scoring, lending underwriting, or billing collections enforcement.
  - Discriminatory rate adjustment or negative service provisioning based on demographic attributes.

---

## 3. Factors & Demographic Fairness
- **Demographic Attributes Evaluated:** `gender` (Male, Female), `SeniorCitizen` (0, 1), `Partner` (Yes, No), `Dependents` (Yes, No).
- **Fairness Audit Summary:**
  - Churn predictions showed parity across gender cohorts (F1: 0.632 for Female vs. 0.630 for Male).
  - Senior citizens experience elevated churn (41.7% raw rate), which is driven by contract type and lack of digital technical support rather than age bias. The model reflects behavioral friction rather than demographic prejudice.

---

## 4. Evaluation Data & Performance Metrics
- **Dataset:** IBM Cognos Analytics Telco Dataset (7,043 customers, 21 initial attributes, California region).
- **Split Strategy:** 80% Training (5,634 rows), 20% Holdout Test (1,409 rows), strictly stratified on churn target.
- **Benchmark Performance Metrics (20% Holdout Test):**
  - **ROC-AUC:** 0.8470 (Benchmark: >0.80)
  - **PR-AUC:** 0.6624 (Benchmark: >0.60)
  - **Churn Recall (Sensitivity):** 81.02% (Captures >8 out of 10 churners)
  - **Precision:** 51.71% (1 out of 2 targeted customers is a true churner)
  - **F1-Score:** 0.6312
  - **Brier Score Loss:** 0.1627 (Well-calibrated probability distributions)

---

## 5. Explainability & Feature Drivers
- **Global Explanations:**
  - Contract status (`Contract_Two year`, `Contract_One year`) is the single strongest negative driver (reduces churn).
  - Customer tenure (`tenure`) is the second strongest stabilizing factor.
  - High monthly charges (`MonthlyCharges`) and Fiber Optic technology without technical support are the primary positive drivers (increase churn).
- **Local Explanations:** Real-time SHAP waterfall charts decompose probability into dollar and feature weights for each individual subscriber.

---

## 6. Caveats & Recommendations
1. **Data Freshness & Concept Drift:** The model was trained on historical snapshot data. Quarterly retraining is recommended to capture shifting macroeconomic trends and competitive pricing changes.
2. **Cold-Start New Subscribers:** Customers with `tenure == 0` have no billing history; their churn predictions should be supplemented by onboarding survey responses.
3. **Operational Threshold:** Use the cost-optimized threshold $T^* = 0.35$ to $0.40$ to maximize net business retention profit rather than the default 0.50 cutoff.
