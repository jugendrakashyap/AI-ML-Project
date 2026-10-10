# Telco Customer Churn Prediction & Retention Optimization

**Track:** Track A — Binary Classification  
**Cohort Skeleton:** Week 1: Business Framing & Data Acquisition + EDA & Data Quality Audit  
**Dataset:** IBM Cognos Analytics (`blastchar/telco-customer-churn`)  
**Target:** `Churn` (Binary: Yes / No)  

---

## 📌 Business Problem & CFO Mandate

> *"You're the DS team at a regional telecom. The CFO wants to know: which customers will churn next quarter, why, and what's the ROI of a targeted retention campaign vs. blanket discounts?"*

Subscriber acquisition costs (CAC) exceed retention costs by **5x to 7x**. Historically, blanket discounts eroded operating margins by discounting non-churners while failing to address customer pain points. This project establishes a data-driven retention intelligence engine that prioritizes customers by **Churn Risk $\times$ Customer Lifetime Value (CLV)**:

$$\text{CLV} = \text{MonthlyCharges} \times \text{tenure}$$

---

## 📁 Project Directory Structure

```text
telco-customer-churn-prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw ingested dataset (7,043 rows x 21 cols)
│
├── notebooks/
│   └── 01_eda.ipynb                           # End-to-end EDA & Hypothesis Testing Notebook
│
├── reports/
│   └── data_quality_report.md                 # Full Data Quality Audit, Quirk fixes & Hypotheses
│
├── docs/
│   └── business_requirements_document.md     # Executive BRD tying churn to CLV & Campaign ROI
│
├── src/
│   ├── __init__.py                            # Package initializer
│   ├── data_loader.py                         # Automated dataset downloader & validator
│   └── data_cleaning.py                       # Data cleaning, quirk handling & CLV pipeline
│
├── README.md                                  # Project overview and Week 1 documentation
│
└── requirements.txt                           # Project dependencies
```

---

## 🎯 Week 1 Deliverables Summary

### 1. Business Framing + Data Acquisition (Deliverable Date: 03-10-2026)
- **Ingestion Pipeline:** Automated ingestion via `src/data_loader.py` fetching `WA_Fn-UseC_-Telco-Customer-Churn.csv` directly into `data/`.
- **Business Requirements Document (BRD):** Authored [`docs/business_requirements_document.md`](docs/business_requirements_document.md) formalizing:
  - Unit economics of customer lifetime value ($\text{CLV} = \text{MonthlyCharges} \times \text{tenure}$).
  - Comparative ROI model: Blanket discounts ($100k+ margin loss) vs. Model-targeted proactive retention (>3.5x net ROI).
  - Cost-sensitive intervention threshold formulation ($P(\text{Churn}) \times P_{\text{success}} \times L_{\text{CLV}} > C_{\text{offer}}$).

### 2. EDA & Data Quality Audit (Deliverable Date: 03-10-2026)
- **Data Quality Report:** Authored [`reports/data_quality_report.md`](reports/data_quality_report.md) with detailed profiling of all 7,043 records.
- **Quirk #1 Resolved (The `TotalCharges` Bug):** Identified 11 records containing whitespace strings (`" "`). Audited all 11 cases: all have `tenure == 0` (new accounts without a closed billing cycle). Coerced using `pd.to_numeric(errors='coerce')` and imputed with `$0.00`.
- **Quirk #2 Resolved (Pseudo-Categories):** Collapsed `"No internet service"` across 6 service columns (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) and `"No phone service"` in `MultipleLines` into clean binary `"No"` indicators.
- **Tenure Bucketing:** Created lifecycle cohort buckets (`0-12m`, `13-24m`, `25-36m`, `37-48m`, `49-60m`, `61-72m`).
- **Core Segment Findings:**
  - **Contract Duration:** Month-to-month contracts churn at **42.71%**, compared to **11.27%** for one-year and **2.83%** for two-year contracts (**>15x higher**).
  - **Tenure Hazard Curve:** Months 0–12 suffer **47.44%** churn (accounts for 55.5% of all churn events).
  - **Payment Friction:** Electronic check users churn at **45.29%**, compared to 15.2%–16.7% for automated credit cards or bank transfers.
  - **Tech Tier Paradox:** Fiber optic customers pay the highest rates ($91.50/mo) and churn highest (**41.89%**). Adding Tech Support drops fiber optic churn from **49.3% to 22.8%**.
- **Interactive Notebook:** Authored [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) with visualizations and Pearson Chi-Square ($\chi^2$) statistical tests ($p < 0.001$).

---

## 🚀 Getting Started

### 1. Clone & Environment Setup
```powershell
# Clone the repository
git clone https://github.com/jugendrakashyap/AI-ML-Project.git
cd AI-ML-Project

# Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Data Ingestion & Cleaning
```powershell
# Verify / download raw data
python src/data_loader.py

# Run data quality audit and cleaning verification
python src/data_cleaning.py
```

### 3. Launch EDA Notebook
```powershell
jupyter notebook notebooks/01_eda.ipynb
```

---

## 📈 Next Steps: Week 2 Roadmap
- Categorical encoding (OneHot / Target encoding)
- Feature engineering in `src/features.py` (`services_count`, `has_security_bundle`, `avg_monthly_spend`)
- Imbalance mitigation (class-weighting / SMOTE for 26.54% churn)
- Logistic Regression baseline model with Stratified 5-Fold Cross-Validation (F1 and ROC-AUC benchmarks)