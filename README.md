# Telco Customer Churn Prediction

**Track:** Track A — Binary Classification  
**Scope:** Week 1 Tasks  
1. **Business framing + data acquisition:** Download; write BRD tying churn to CLV ($\text{CLV} = \text{MonthlyCharges} \times \text{tenure}$); set up repo.  
2. **EDA & data quality audit:** Fix `TotalCharges` blank-string bug; analyze churn rate by Contract type (month-to-month driver), tenure buckets, PaymentMethod (electronic check); collapse pseudo-categories; hypothesis list.

---

## Folder Structure

```text
telco-customer-churn-prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── reports/
│   └── data_quality_report.md
│
├── docs/
│   └── business_requirements_document.md
│
├── src/
│
├── README.md
│
└── requirements.txt
```

---

## Week 1 Tasks & Deliverables

### Step 1: Business Framing + Data Acquisition
- **Deliverables:** BRD, GitHub repository, raw data ingested.
- **BRD Document:** [`docs/business_requirements_document.md`](docs/business_requirements_document.md)
  - Business problem: Regional telecom CFO retention inquiry.
  - Tying churn to Customer Lifetime Value: $\text{CLV} = \text{MonthlyCharges} \times \text{tenure}$.
  - Unit economics of model-targeted retention vs. blanket discount campaign.
  - Raw dataset ingested into [`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`](data/WA_Fn-UseC_-Telco-Customer-Churn.csv) (7,043 rows $\times$ 21 columns).

### Step 2: EDA & Data Quality Audit
- **Deliverables:** EDA notebook, Data Quality Report, hypothesis list.
- **Data Quality Report:** [`reports/data_quality_report.md`](reports/data_quality_report.md)
  - **Quirk #1 (`TotalCharges` bug):** 11 rows with whitespace string `" "` coerced using `pd.to_numeric(errors='coerce')` and imputed with `0.0` (all 11 have `tenure == 0`).
  - **Quirk #2 (Pseudo-categories):** Collapsed `"No internet service"` across 6 service attributes and `"No phone service"` in `MultipleLines` into `"No"`.
  - **Contract Analysis:** Month-to-month contracts exhibit **42.71%** churn vs. **11.27%** for one-year and **2.83%** for two-year contracts (>15x higher).
  - **Tenure Buckets:** Tenure 0–12 months exhibits **47.44%** churn (accounts for 55.5% of all churners).
  - **Payment Method:** Electronic check payment churn is **45.29%** vs. 15.2%–16.7% for automatic payment options.
  - **Hypothesis List:** Statistically validated hypotheses ($p < 0.001$, Pearson $\chi^2$).
- **EDA Notebook:** [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)
  - Self-contained, pre-executed Jupyter notebook with data auditing, distributions, visualizations, and statistical hypothesis tests.

---

## How to Run

1. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Launch the EDA Notebook:**
   ```powershell
   jupyter notebook notebooks/01_eda.ipynb
   ```