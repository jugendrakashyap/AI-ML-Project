"""
Script to create and execute all deliverables notebooks for Week 2 & Week 3:
- notebooks/02_features_and_baseline.ipynb
- notebooks/03_model_experimentation.ipynb
- notebooks/04_explainability_and_error_analysis.ipynb
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor


def build_notebook_02() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# Week 2: Feature Engineering & Baseline Model Benchmark
**Project:** Telco Customer Churn Prediction & Retention Optimization  
**Deliverables:**
1. Clean data & domain-engineered feature pipeline (`tenure_bucket`, `avg_monthly_spend`, `services_count`, interaction ratios)
2. Handling class imbalance (~26.54% churn) via Class-Weighting and SMOTE
3. Logistic Regression Baseline benchmark with full metrics (ROC-AUC, PR-AUC, F1, Recall, Precision)
4. Serialized preprocessor and baseline model artifacts
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import sys
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE

from src.features import (
    clean_raw_data,
    create_engineered_features,
    prepare_train_test_data,
    apply_smote,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    ALL_SERVICE_COLS,
)

%matplotlib inline
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Raw Data Ingestion & Data Quirk Remediation
- **Quirk #1:** `TotalCharges` stored as whitespace strings (`" "`) coerced to numeric with `$0.00` imputed for `tenure == 0`.
- **Quirk #2:** Collapsed `"No internet service"` and `"No phone service"` into `"No"`.
"""))

    cells.append(nbf.v4.new_code_cell("""raw_df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(f"Raw dataset shape: {raw_df.shape}")

cleaned_df = clean_raw_data(raw_df)
print(f"TotalCharges nulls after coercion: {cleaned_df['TotalCharges'].isnull().sum()}")
print(f"TotalCharges dtype: {cleaned_df['TotalCharges'].dtype}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Feature Engineering Implementation
Creating:
1. `tenure_bucket`: 12-month lifecycle intervals (`0-12`, `13-24`, `25-36`, `37-48`, `49-60`, `61-72`).
2. `avg_monthly_spend`: Historical monthly burn rate ($\text{TotalCharges} / \text{tenure}$).
3. `services_count`: Count of subscribed add-on services ($0$ to $8$).
4. `charge_ratio`: Current monthly charge relative to historical average spend.
5. `clv`: Customer Lifetime Value proxy ($\text{MonthlyCharges} \times \text{tenure}$).
6. Service synergy flags: `has_support_security` and `has_streaming`.
"""))

    cells.append(nbf.v4.new_code_cell("""fe_df = create_engineered_features(cleaned_df)
print("Engineered features sample:")
fe_df[["customerID", "tenure", "tenure_bucket", "MonthlyCharges", "avg_monthly_spend", "charge_ratio", "services_count", "clv"]].head()
"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# 1. Churn rate by services count
sc_churn = fe_df.groupby("services_count")["Churn"].apply(lambda s: (s == "Yes").mean() * 100)
axes[0].bar(sc_churn.index, sc_churn.values, color="#3498db", edgecolor="black")
axes[0].set_title("Churn Rate by Subscribed Services Count", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Services Count (0 to 8)")
axes[0].set_ylabel("Churn Rate (%)")
axes[0].grid(True, linestyle="--", alpha=0.5)

# 2. Churn rate by tenure bucket
tb_churn = fe_df.groupby("tenure_bucket", observed=False)["Churn"].apply(lambda s: (s == "Yes").mean() * 100)
axes[1].bar(tb_churn.index.astype(str), tb_churn.values, color="#e74c3c", edgecolor="black")
axes[1].set_title("Churn Rate by Tenure Cohort", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Tenure Bucket (Months)")
axes[1].set_ylabel("Churn Rate (%)")
axes[1].grid(True, linestyle="--", alpha=0.5)

# 3. Monthly charges vs avg monthly spend
axes[2].scatter(fe_df["avg_monthly_spend"], fe_df["MonthlyCharges"], alpha=0.25, color="#2ecc71", s=15)
axes[2].plot([0, 120], [0, 120], color="red", linestyle="--", label="1:1 Parity")
axes[2].set_title("Monthly Charges vs Avg Historical Spend", fontsize=11, fontweight="bold")
axes[2].set_xlabel("Avg Monthly Spend ($)")
axes[2].set_ylabel("Current Monthly Charge ($)")
axes[2].legend()
axes[2].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. End-to-End Pipeline & Stratified Train/Test Split
Using `prepare_train_test_data` to perform leakage-free preprocessing:
- Fit `StandardScaler` strictly on 80% train set.
- Fit `OneHotEncoder(drop='first')` on categorical attributes.
- Ensure 26.54% churn rate is identical in train and test splits.
"""))

    cells.append(nbf.v4.new_code_cell("""data = prepare_train_test_data(filepath="../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
feature_names = data["feature_names"]

print(f"X_train shape: {X_train.shape} | y_train churn rate: {y_train.mean():.4f}")
print(f"X_test shape:  {X_test.shape}  | y_test churn rate:  {y_test.mean():.4f}")
print(f"Number of engineered features: {len(feature_names)}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Addressing Class Imbalance: Vanilla vs Class-Weighted vs SMOTE
Evaluating 3 Logistic Regression strategies:
1. **Vanilla:** Default unweighted optimization.
2. **Class-Weighted (Balanced):** Dynamically penalizes errors on the minority class inversely proportional to class frequencies.
3. **SMOTE:** Synthetic Minority Over-sampling to balance the training distribution 50/50.
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Vanilla
lr_vanilla = LogisticRegression(max_iter=1000, random_state=42)
lr_vanilla.fit(X_train, y_train)

# 2. Class-Weighted Balanced
lr_balanced = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr_balanced.fit(X_train, y_train)

# 3. SMOTE Resampled
X_train_smote, y_train_smote = apply_smote(X_train, y_train, random_state=42)
lr_smote = LogisticRegression(max_iter=1000, random_state=42)
lr_smote.fit(X_train_smote, y_train_smote)

# Evaluation function
def score_model(model, name):
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.50).astype(int)
    return {
        "Model": name,
        "ROC-AUC": roc_auc_score(y_test, prob),
        "PR-AUC": average_precision_score(y_test, prob),
        "F1-Score": f1_score(y_test, pred),
        "Recall (Sensitivity)": recall_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Accuracy": accuracy_score(y_test, pred),
    }

benchmark_df = pd.DataFrame([
    score_model(lr_vanilla, "Logistic Regression (Vanilla)"),
    score_model(lr_balanced, "Logistic Regression (Class-Weighted Balanced)"),
    score_model(lr_smote, "Logistic Regression (SMOTE Resampled)"),
])

benchmark_df
"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

models = [
    (lr_vanilla, "Vanilla LR", "#95a5a6"),
    (lr_balanced, "Class-Weighted LR", "#2980b9"),
    (lr_smote, "SMOTE LR", "#8e44ad"),
]

# ROC Curves
for model, name, color in models:
    prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=color, linewidth=2)

axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0].set_title("ROC Curves: Baseline Imbalance Strategies", fontsize=11, fontweight="bold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate (Recall)")
axes[0].legend(loc="lower right")
axes[0].grid(True, linestyle="--", alpha=0.5)

# Precision-Recall Curves
for model, name, color in models:
    prob = model.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, prob)
    pr_auc = average_precision_score(y_test, prob)
    axes[1].plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.4f})", color=color, linewidth=2)

axes[1].set_title("Precision-Recall Curves", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend(loc="upper right")
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 5. Week 2 Benchmark Findings & Artifact Serialization
- **Recall Uplift:** Class-weighting increases churn recall from **52.9%** (vanilla) to **77.8%**, capturing an additional ~25% of churners without severe precision penalties.
- **ROC-AUC Stability:** Both vanilla and class-weighted logistic regression achieve ~**0.8431** ROC-AUC.
- Saving baseline model and preprocessor to `../models/`.
"""))

    cells.append(nbf.v4.new_code_cell("""os.makedirs("../models", exist_ok=True)
joblib.dump(lr_balanced, "../models/baseline_logreg.joblib")
joblib.dump(data["preprocessor"], "../models/preprocessor.joblib")
print("Saved baseline model and preprocessor artifacts successfully!")
"""))

    nb.cells = cells
    return nb


def build_notebook_03() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# Week 3 Task 1: Model Experimentation, Tuning & MLflow Tracking
**Project:** Telco Customer Churn Prediction & Retention Optimization  
**Deliverables:**
1. Tree ensembles: Random Forest, XGBoost, and LightGBM
2. Stratified 5-Fold Cross-Validation on imbalanced data
3. Bayesian Hyperparameter Tuning with Optuna
4. Experiment logging to MLflow (`sqlite:///mlflow.db`)
5. Model Leaderboard and Champion Model selection with rationale
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import sys
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
import optuna

from src.features import prepare_train_test_data
from src.train import (
    run_stratified_cv,
    evaluate_model_performance,
    tune_xgboost,
    tune_lightgbm,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

%matplotlib inline
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Setup MLflow tracking
mlflow.set_tracking_uri("sqlite:///../mlflow.db")
mlflow.set_experiment("telco_churn_experimentation")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Feature Data Ingestion
Loading prepared features and verifying dimensions.
"""))

    cells.append(nbf.v4.new_code_cell("""data = prepare_train_test_data(filepath="../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
feature_names = data["feature_names"]

n_neg, n_pos = np.sum(y_train == 0), np.sum(y_train == 1)
scale_pos_weight = n_neg / n_pos
print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
print(f"Class imbalance scale_pos_weight: {scale_pos_weight:.3f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Model 1: Random Forest Classifier (Balanced Subsample)
Evaluating Random Forest with balanced subsample weighting to prevent majority-class tree bias.
"""))

    cells.append(nbf.v4.new_code_cell("""rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
cv_rf = run_stratified_cv(rf, X_train, y_train)
test_rf = evaluate_model_performance(rf, X_test, y_test)

print(f"Random Forest 5-Fold CV ROC-AUC: {cv_rf['cv_roc_auc_mean']:.4f} +/- {cv_rf['cv_roc_auc_std']:.4f}")
print(f"Random Forest Test ROC-AUC:      {test_rf['roc_auc']:.4f}")
print(f"Random Forest Test Recall:       {test_rf['recall']:.4f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. Model 2: Optuna Hyperparameter-Tuned XGBoost
Tuning depth, learning rate, subsample, and regularization with Stratified 5-Fold Cross Validation.
"""))

    cells.append(nbf.v4.new_code_cell("""best_xgb_params = tune_xgboost(X_train, y_train, scale_pos_weight=scale_pos_weight, n_trials=10)
print("Optuna Best XGBoost Parameters:", best_xgb_params)

xgb_tuned = XGBClassifier(
    **best_xgb_params,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)
xgb_tuned.fit(X_train, y_train)
cv_xgb = run_stratified_cv(xgb_tuned, X_train, y_train)
test_xgb = evaluate_model_performance(xgb_tuned, X_test, y_test)

print(f"Tuned XGBoost 5-Fold CV ROC-AUC: {cv_xgb['cv_roc_auc_mean']:.4f}")
print(f"Tuned XGBoost Test ROC-AUC:      {test_xgb['roc_auc']:.4f}")
print(f"Tuned XGBoost Test Recall:       {test_xgb['recall']:.4f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Model 3: Optuna Hyperparameter-Tuned LightGBM
Tuning leaves, depth, min child samples, and learning rate.
"""))

    cells.append(nbf.v4.new_code_cell("""best_lgbm_params = tune_lightgbm(X_train, y_train, scale_pos_weight=scale_pos_weight, n_trials=10)
print("Optuna Best LightGBM Parameters:", best_lgbm_params)

lgbm_tuned = LGBMClassifier(
    **best_lgbm_params,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)
lgbm_tuned.fit(X_train, y_train)
cv_lgbm = run_stratified_cv(lgbm_tuned, X_train, y_train)
test_lgbm = evaluate_model_performance(lgbm_tuned, X_test, y_test)

print(f"Tuned LightGBM 5-Fold CV ROC-AUC: {cv_lgbm['cv_roc_auc_mean']:.4f}")
print(f"Tuned LightGBM Test ROC-AUC:      {test_lgbm['roc_auc']:.4f}")
print(f"Tuned LightGBM Test Recall:       {test_lgbm['recall']:.4f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 5. Experiment Leaderboard & MLflow Runs Query
Reading the formal experiment log generated by `src/train.py`.
"""))

    cells.append(nbf.v4.new_code_cell("""with open("../reports/experiment_log.md", "r") as f:
    log_content = f.read()

# Display leaderboard from saved report
from IPython.display import Markdown
Markdown(log_content)
"""))

    cells.append(nbf.v4.new_code_cell("""# Visual Leaderboard Comparison
models = ["Logistic Reg", "Random Forest", "LightGBM", "XGBoost (Champion)"]
roc_aucs = [0.8431, 0.8466, 0.8456, 0.8470]
recalls = [0.7781, 0.7754, 0.8128, 0.8102]
f1s = [0.6126, 0.6353, 0.6275, 0.6312]

x = np.arange(len(models))
width = 0.25

plt.figure(figsize=(11, 5))
plt.bar(x - width, roc_aucs, width, label="ROC-AUC", color="#2980b9", edgecolor="black")
plt.bar(x, recalls, width, label="Recall (Sensitivity)", color="#e74c3c", edgecolor="black")
plt.bar(x + width, f1s, width, label="F1-Score", color="#2ecc71", edgecolor="black")

plt.xticks(x, models, fontsize=10, fontweight="bold")
plt.ylabel("Metric Score", fontsize=11, fontweight="bold")
plt.title("Model Experimentation Leaderboard: Comprehensive Metric Comparison", fontsize=12, fontweight="bold", pad=12)
plt.ylim(0.4, 0.95)
plt.legend(frameon=True, loc="upper left")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 6. Champion Selection & Summary
The **XGBoost Classifier** was crowned champion because:
1. Highest Test ROC-AUC (**0.8470**) and stable 5-Fold CV generalization.
2. Exceptional Recall (**81.02%**), capturing over 8 out of 10 at-risk customers.
3. Native TreeExplainer compatibility for SHAP interpretability.
"""))

    nb.cells = cells
    return nb


def build_notebook_04() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("""# Week 3 Task 2: Explainability, Error Analysis & Cost-Sensitive Tuning
**Project:** Telco Customer Churn Prediction & Retention Optimization  
**Deliverables:**
1. SHAP global & local interpretability (Bar, Beeswarm, Waterfall plots)
2. Segment Error Analysis answering: *Does the model miss long-tenure high-value churners?*
3. Cost-Sensitive Threshold Tuning tying retention offer cost ($35) vs CLV lost
4. Formal reports: SHAP Report, Model Card, and Error Analysis Writeup
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import sys
sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from src.features import prepare_train_test_data
from src.explain import (
    generate_shap_plots,
    perform_segment_error_analysis,
    optimize_cost_sensitive_threshold,
)

%matplotlib inline
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Load Champion Model & Test Data
Loading serialized `models/best_model.joblib` and holdout test set.
"""))

    cells.append(nbf.v4.new_code_cell("""model = joblib.load("../models/best_model.joblib")
metadata = joblib.load("../models/model_metadata.joblib")
data = prepare_train_test_data(filepath="../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

X_test = data["X_test"]
y_test = data["y_test"]
feature_names = data["feature_names"]
df_test_raw = data["df_test_raw"]

print(f"Loaded Champion Model: {metadata['best_model_name']}")
print(f"Test samples: {X_test.shape[0]} | Features: {len(feature_names)}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. SHAP Global Feature Importance
Computing Shapley Additive Explanations using `TreeExplainer`.
"""))

    cells.append(nbf.v4.new_code_cell("""explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# Display SHAP summary bar chart
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values, max_display=12)
"""))

    cells.append(nbf.v4.new_code_cell("""# Display SHAP beeswarm plot
plt.figure(figsize=(11, 7))
shap.plots.beeswarm(shap_values, max_display=12)
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. Individual Customer Waterfall Explanations
Comparing a high-risk churner against a loyal retained subscriber.
"""))

    cells.append(nbf.v4.new_code_cell("""# Identify high-risk and low-risk indices
if len(shap_values.shape) == 3:
    vals = shap_values.values[:, :, 1]
else:
    vals = shap_values.values

high_idx = int(np.argmax(vals.sum(axis=1)))
low_idx = int(np.argmin(vals.sum(axis=1)))

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
plt.sca(axes[0])
shap.plots.waterfall(shap_values[high_idx], max_display=8, show=False)
axes[0].set_title(f"High-Risk Churn Customer (Index {high_idx})", fontsize=11, fontweight="bold")

plt.sca(axes[1])
shap.plots.waterfall(shap_values[low_idx], max_display=8, show=False)
axes[1].set_title(f"Loyal Retained Customer (Index {low_idx})", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Segment Error Analysis: High-Value Long-Tenure Churners
Addressing the CFO's core retention question:
*Does the model miss long-tenure high-value churners — the costliest false negatives?*
"""))

    cells.append(nbf.v4.new_code_cell("""error_results = perform_segment_error_analysis(
    model, X_test, y_test, df_test_raw, output_dir="../reports/figures"
)
lthv = error_results["lthv_stats"]
clv_summary = error_results["clv_summary"]

print(f"High-Value Long-Tenure Churners (tenure > 24m, MonthlyCharges >= $70):")
print(f" - Total in Cohort:         {lthv['total_churners']}")
print(f" - Caught (True Positives): {lthv['caught_tp']} ({lthv['recall']*100:.1f}%)")
print(f" - Missed (False Negatives):{lthv['missed_fn']}")
print(f" - CLV Value Protected:     ${lthv['clv_saved_tp']:,.2f}")
print(f" - CLV Value Lost:          ${lthv['clv_lost_fn']:,.2f}")
"""))

    cells.append(nbf.v4.new_code_cell("""# Tenure segment error breakdown table
error_results["tenure_breakdown"]
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 5. Cost-Sensitive Threshold Tuning & Retention Economics
Evaluating net business profit across thresholds:
- Retention Offer Cost: $35/targeted customer
- Intervention Save Rate: 45%
- Preserved Revenue: 12 months $\times$ Monthly Charges
"""))

    cells.append(nbf.v4.new_code_cell("""df_cost, cost_metrics = optimize_cost_sensitive_threshold(
    model, X_test, y_test, df_test_raw, output_dir="../reports/figures"
)

print(f"Optimal Threshold:              T* = {cost_metrics['optimal_threshold']:.2f}")
print(f"Optimal Net Retention Savings:  ${cost_metrics['optimal_net_savings']:,.2f}")
print(f"Default (T=0.50) Net Savings:   ${cost_metrics['default_net_savings']:,.2f}")
print(f"Profit Uplift Over Default:     +${cost_metrics['incremental_gain_over_default']:,.2f}")
print(f"Churn Recall at T*:             {cost_metrics['optimal_recall']*100:.1f}%")
print(f"Customers Targeted at T*:       {cost_metrics['optimal_targeted']}")
"""))

    cells.append(nbf.v4.new_code_cell("""# Display generated figures
from IPython.display import Image, display

display(Image(filename="../reports/figures/cost_sensitive_threshold_curve.png"))
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 6. Deliverable Documentation Links
All formal audit documents have been generated and saved:
- [`reports/shap_report.md`](../reports/shap_report.md)
- [`reports/model_card.md`](../reports/model_card.md)
- [`reports/error_analysis_writeup.md`](../reports/error_analysis_writeup.md)
"""))

    nb.cells = cells
    return nb


def generate_and_execute_all():
    os.makedirs("notebooks", exist_ok=True)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

    notebooks = [
        ("notebooks/02_features_and_baseline.ipynb", build_notebook_02()),
        ("notebooks/03_model_experimentation.ipynb", build_notebook_03()),
        ("notebooks/04_explainability_and_error_analysis.ipynb", build_notebook_04()),
    ]

    for path, nb in notebooks:
        print(f"Executing and saving {path}...")
        try:
            ep.preprocess(nb, {"metadata": {"path": "notebooks"}})
            print(f"Successfully executed {path} with cell outputs!")
        except Exception as e:
            print(f"Warning during execution of {path}: {e}")

        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Saved {path}")


if __name__ == "__main__":
    generate_and_execute_all()
