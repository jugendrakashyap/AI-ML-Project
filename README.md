# Telco Customer Churn Prediction & Retention Optimization

**Track:** Track A — Binary Classification (Telecom / SaaS Retention)  
**Dataset:** IBM Cognos Analytics Telco Dataset (7,043 customers $\times$ 21 attributes)  
**Status:** Completed Weeks 1, 2, and 3 Tasks  

---

## Executive Overview

> 📖 **Comprehensive Master Project Report:** For an in-depth walkthrough of all mathematical formulations, business framing, quirk fixes, model benchmarks, SHAP explanations, segment error audits, and financial threshold curves, see [`docs/project_comprehensive_report.md`](docs/project_comprehensive_report.md).

This repository develops an enterprise-grade machine learning system designed to predict customer churn, explain churn drivers, and optimize proactive retention campaigns for a regional telecommunications provider.

### Completed Program Milestones:
1. **Week 1 (Business Framing & Data Acquisition):** BRD connecting churn to Customer Lifetime Value ($\text{CLV} = \text{MonthlyCharges} \times \text{tenure}$), raw data ingestion, and repository setup.
2. **Week 1 (EDA & Data Quality Audit):** Resolved `TotalCharges` blank-string quirk, collapsed pseudo-categories, validated core drivers (Contract, Tenure, Electronic Check, Fiber Optic), and formulated statistical hypotheses.
3. **Week 2 (Feature Engineering & Baseline Benchmark):** Built leakage-free preprocessing pipeline in [`src/features.py`](src/features.py), engineered domain features (`tenure_bucket`, `avg_monthly_spend`, `services_count`, `charge_ratio`, `clv`), evaluated class-weighting and SMOTE against 26.6% class imbalance, and established the Logistic Regression benchmark.
4. **Week 3 Task 1 (Model Experimentation & Tuning):** Trained Random Forest, XGBoost, and LightGBM using Stratified 5-Fold Cross Validation; executed Bayesian hyperparameter optimization with Optuna; tracked all runs in MLflow; selected tuned **XGBoost Classifier** as champion (`models/best_model.joblib`).
5. **Week 3 Task 2 (Explainability & Error Analysis):** Generated global/local SHAP explainability (beeswarm, bar, and waterfall plots); conducted deep segment error audit on high-value long-tenure churners; solved cost-sensitive threshold optimization ($T^* = 0.30$) maximizing net financial retention profit.

---

## Repository Structure

```text
AI-ML-Project/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   <- Raw dataset (7,043 rows)
│
├── docs/
│   └── business_requirements_document.md     <- Executive BRD tying churn to CLV economics
│
├── notebooks/
│   ├── 01_eda.ipynb                          <- Week 1: Exploratory Data Analysis & audit
│   ├── 02_features_and_baseline.ipynb        <- Week 2: Feature pipeline & baseline benchmark
│   ├── 03_model_experimentation.ipynb        <- Week 3: Tree ensembles, Optuna tuning & MLflow
│   └── 04_explainability_and_error_analysis.ipynb <- Week 3: SHAP, segment errors & cost tuning
│
├── models/
│   ├── baseline_logreg.joblib                <- Week 2: Class-weighted Logistic Regression
│   ├── best_model.joblib                     <- Week 3: Champion Tuned XGBoost Classifier
│   ├── model_metadata.joblib                 <- Model parameters and evaluation metadata
│   └── preprocessor.joblib                   <- Fitted Scikit-Learn ColumnTransformer pipeline
│
├── reports/
│   ├── data_quality_report.md                <- Week 1: Quirk remediation & hypothesis audit
│   ├── experiment_log.md                     <- Week 3: Full model leaderboard & MLflow logs
│   ├── shap_report.md                        <- Week 3: SHAP global & local interpretability
│   ├── model_card.md                         <- Week 3: Industry-standard Model Card (Mitchell et al.)
│   ├── error_analysis_writeup.md             <- Week 3: Segment error audit & threshold tuning
│   └── figures/                              <- High-resolution plots
│       ├── shap_summary_bar.png
│       ├── shap_beeswarm.png
│       ├── shap_waterfall_churner.png
│       ├── shap_waterfall_retained.png
│       ├── segment_confusion_by_tenure.png
│       ├── clv_at_risk_by_error_type.png
│       └── cost_sensitive_threshold_curve.png
│
├── src/
│   ├── __init__.py
│   ├── features.py                           <- Feature engineering & transformation pipeline
│   ├── train.py                              <- Model training, CV, Optuna tuning & MLflow
│   ├── explain.py                            <- SHAP explanations, error audit & cost curves
│   └── generate_notebooks.py                 <- Automated notebook builder & executor
│
├── mlflow.db                                 <- SQLite backend database for MLflow experiments
├── README.md                                 <- Project documentation
└── requirements.txt                          <- Pinned environment dependencies
```

---

## Detailed Deliverables by Week

### Week 2: Feature Engineering & Baseline Model Benchmark
- **Source Code:** [`src/features.py`](src/features.py)
  - `clean_raw_data()`: Converts `TotalCharges` whitespace to `$0.00` and collapses `"No internet service"` / `"No phone service"` into `"No"`.
  - `create_engineered_features()`:
    - `tenure_bucket`: 12-month lifecycle brackets (`0-12`, `13-24`, `25-36`, `37-48`, `49-60`, `61-72`).
    - `avg_monthly_spend`: True historical burn rate ($\text{TotalCharges} / \text{tenure}$).
    - `services_count`: Count of active value-add services ($0$ to $8$).
    - `charge_ratio`: Ratio of current `MonthlyCharges` to `avg_monthly_spend` (detects price hikes and promo expirations).
    - `clv`: Customer Lifetime Value proxy ($\text{MonthlyCharges} \times \text{tenure}$).
  - `build_preprocessor()`: Scikit-learn `ColumnTransformer` with `StandardScaler` on numericals and `OneHotEncoder(drop='first')` on categoricals.
  - Leakage-free train/test splitting strictly stratified on `Churn`.
- **Baseline Benchmark Results (20% Holdout Test, 1,409 Customers):**
  - **Vanilla Logistic Regression:** ROC-AUC: **0.8431**, F1: **0.5902**, Recall: **52.94%**
  - **Class-Weighted Logistic Regression:** ROC-AUC: **0.8431**, F1: **0.6126**, Recall: **77.81%** (Selected Baseline)
  - **SMOTE Logistic Regression:** ROC-AUC: **0.8401**, F1: **0.6203**, Recall: **78.61%**
- **Notebook:** [`notebooks/02_features_and_baseline.ipynb`](notebooks/02_features_and_baseline.ipynb)

---

### Week 3 Task 1: Model Experimentation & Tuning
- **Source Code:** [`src/train.py`](src/train.py)
- **Experiment Tracking:** Integrated SQLite MLflow tracking (`mlflow.db`, experiment: `telco_churn_experimentation`).
- **Validation Discipline:** 5-Fold Stratified Cross-Validation on training data to respect 26.6% class imbalance.
- **Model Leaderboard Comparison:**

| Model | Strategy | 5-Fold CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Test F1 | Test Precision | Test Recall | Test Accuracy | Brier Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Vanilla)** | Default Unweighted | 0.8472 | 0.8431 | 0.6407 | 0.5902 | 0.6667 | 0.5294 | 0.8048 | 0.1375 |
| **Logistic Regression (Balanced)** | Class-Weighted | 0.8471 | 0.8431 | 0.6418 | 0.6126 | 0.5052 | 0.7781 | 0.7388 | 0.1661 |
| **Logistic Regression (SMOTE)** | SMOTE Oversampling | 0.8351 | 0.8401 | 0.6393 | 0.6203 | 0.5122 | 0.7861 | 0.7445 | 0.1668 |
| **Random Forest (Balanced)** | Balanced Subsample | 0.8467 | 0.8466 | 0.6628 | 0.6353 | 0.5380 | 0.7754 | 0.7637 | 0.1560 |
| **XGBoost (Tuned Champion)** | Optuna + Scale Pos Weight | **0.8489** | **0.8470** | **0.6624** | **0.6312** | 0.5171 | **0.8102** | 0.7488 | 0.1627 |
| **LightGBM (Tuned)** | Optuna + Scale Pos Weight | 0.8481 | 0.8456 | 0.6627 | 0.6275 | 0.5109 | 0.8128 | 0.7438 | 0.1643 |

- **Champion Model:** Tuned **XGBoost Classifier** saved to [`models/best_model.joblib`](models/best_model.joblib).
- **Formal Deliverable:** [`reports/experiment_log.md`](reports/experiment_log.md)
- **Notebook:** [`notebooks/03_model_experimentation.ipynb`](notebooks/03_model_experimentation.ipynb)

---

### Week 3 Task 2: Explainability & Error Analysis
- **Source Code:** [`src/explain.py`](src/explain.py)
- **SHAP Global & Local Interpretability:**
  - `Contract_Two year` and `tenure` are the strongest negative drivers preventing churn.
  - `MonthlyCharges` and `InternetService_Fiber optic` are top positive drivers elevating churn risk.
  - Full analysis in [`reports/shap_report.md`](reports/shap_report.md).
- **High-Value Long-Tenure Segment Audit (`tenure > 24m`, `MonthlyCharges >= $70`):**
  - **Recall on segment:** **65.88%** (captures 56 out of 85 churners).
  - **Protected CLV:** **$203,724.70** protected; residual missed CLV limited to $153,045.80.
  - Most misses concentrate in the **0–12 month tenure** bucket due to minimal usage history.
- **Cost-Sensitive Decision Threshold Tuning:**
  - Evaluated retention economics ($35 outreach cost, 45% save rate, 12-month preserved CLV).
  - Standard classification cutoff ($0.50$) achieves **$100,951.93** in net retention savings.
  - Optimal threshold **$T^* = 0.30$** expands targeted base to 804 customers, increasing churn recall to **92.5%** and delivering **$110,156.16** in Net Retention Savings (an incremental profit uplift of **+$9,204.23** over default threshold).
- **Formal Deliverables:**
  - [`reports/shap_report.md`](reports/shap_report.md)
  - [`reports/error_analysis_writeup.md`](reports/error_analysis_writeup.md)
  - [`reports/model_card.md`](reports/model_card.md)
- **Notebook:** [`notebooks/04_explainability_and_error_analysis.ipynb`](notebooks/04_explainability_and_error_analysis.ipynb)

---

## How to Run & Reproduce

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run Feature Pipeline & Train Models
```powershell
# Run model training, tuning, and MLflow logging
python src/train.py

# Run SHAP interpretability, error analysis, and threshold optimization
python src/explain.py
```

### 3. Launch MLflow UI
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open [http://localhost:5000](http://localhost:5000) to inspect all logged runs, parameters, metrics, and model signatures.

### 4. Interactive Jupyter Notebooks
```powershell
jupyter notebook
```
- Open [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) (Week 1 EDA)
- Open [`notebooks/02_features_and_baseline.ipynb`](notebooks/02_features_and_baseline.ipynb) (Week 2 Pipeline & Baseline)
- Open [`notebooks/03_model_experimentation.ipynb`](notebooks/03_model_experimentation.ipynb) (Week 3 Model Tuning)
- Open [`notebooks/04_explainability_and_error_analysis.ipynb`](notebooks/04_explainability_and_error_analysis.ipynb) (Week 3 Explainability & Cost Tuning)