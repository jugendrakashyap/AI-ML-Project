# Business Requirements Document (BRD)
## Regional Telecom: Customer Churn Prediction & Targeted Retention Engine

**Document Version:** 1.0.0  
**Project Track:** Track A — Binary Classification  
**Target Variable:** `Churn` (Binary: Yes / No)  
**Stakeholder Audience:** Chief Financial Officer (CFO), VP of Customer Experience, Retention Marketing Operations  
**Date:** October 03, 2026 (Week 1 Delivery)  

---

## 1. Executive Summary & Business Framing

### 1.1 Context
In competitive regional telecommunications markets, subscriber acquisition costs (CAC) typically range from 5x to 7x the cost of retaining an existing subscriber. The company currently experiences an overall customer churn rate of **~26.5%**, eroding monthly recurring revenue (MRR) and depressing long-term enterprise valuation.

Historically, retention marketing has relied on **blanket discount campaigns** (e.g., blanket 15–20% rate reductions offered indiscriminately across expiring contracts). This introduces significant financial penalties:
1. **Margin Cannibalization:** Unnecessarily subsidizing "safe" customers who never intended to churn.
2. **Ineffective Intervention:** Offering generic price cuts to subscribers who leave due to onboarding friction, lack of technical support, or billing method inconvenience rather than pure price sensitivity.

### 1.2 Core Business Questions from the CFO
1. **Which customers will churn next quarter?**  
   Predict subscriber churn likelihood with calibrated probability scores.
2. **Why are they churning?**  
   Isolate root-cause churn drivers (contract duration, billing friction, lack of tech support/security bundles) to inform tailored interventions.
3. **What is the ROI of a targeted retention campaign vs. blanket discounts?**  
   Quantify campaign expenditure versus preserved recurring subscriber revenue using customer lifetime value.

---

## 2. Customer Lifetime Value (CLV) & Unit Economics

### 2.1 Customer Lifetime Value Definition
In this project, subscriber lifetime value is formalized as:

$$\text{CLV} = \text{MonthlyCharges} \times \text{tenure}$$

Where:
- $\text{MonthlyCharges}$: Monthly recurring billed amount in USD ($).
- $\text{tenure}$: Number of continuous months the subscriber has been active.

For forward-looking retention valuation over a post-intervention horizon of $T_{\text{horizon}} = 12$ months:
$$\text{Expected Value Saved} = P_{\text{retention}} \times (\text{MonthlyCharges} \times T_{\text{horizon}})$$

### 2.2 Financial Comparison: Blanket vs. Targeted Campaigns
Analysis on an active customer base of $N = 7,043$ accounts with baseline churn of $26.54\%$ ($\approx 1,869$ churners):

| Dimension | Strategy A: Blanket Discounts | Strategy B: Model-Targeted Retention |
| :--- | :--- | :--- |
| **Target Population** | 100% of expiring or active accounts ($7,043$) | Top risk deciles ($15\% - 20\%$ highest predicted risk $\times$ CLV) |
| **Intervention Type** | Generic monthly billing discount (e.g., $15/mo) | Driver-specific intervention (e.g., security bundle upgrade, automated billing perk, annual contract incentive) |
| **Cost per Subscriber** | High cumulative margin loss across non-churners | ~$30 – $50 targeted offer / onboarding support |
| **Total Campaign Spend** | **$105,645+** (widespread margin sacrifice) | **$35,000 – $45,000** (focused budget) |
| **Retention Precision** | Low (~15–20% uptake; high waste) | High (~35–45% proactive retention rate) |
| **Net Financial ROI** | Often negative or marginal (< 1.2x) | **High (> 3.5x ROI on retained CLV)** |

### 2.3 Cost-Sensitive Decision Matrix
Let:
- $C_{\text{offer}}$: Proactive retention intervention budget per customer (e.g., $40).
- $P(\text{Churn})$: Calibrated model churn probability.
- $P_{\text{success}}$: Probability the customer accepts the retention offer (~40%).
- $L_{\text{CLV}}$: Projected future revenue loss if the customer churns.

The decision rule to trigger outreach for a specific customer is:
$$P(\text{Churn}) \times P_{\text{success}} \times L_{\text{CLV}} > C_{\text{offer}}$$

---

## 3. Dataset Specifications & Known Quirks

- **Source:** IBM Cognos Analytics Sample Dataset (`blastchar/telco-customer-churn`)
- **Dimensions:** 7,043 customer records $\times$ 21 raw columns
- **Target Feature:** `Churn` (`Yes` / `No`) — moderately imbalanced (~26.54% positive class)
- **Feature Categories:**
  - **Identifiers:** `customerID`
  - **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - **Account & Financials:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
  - **Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Known Quirks Addressed in Week 1:**
  1. `TotalCharges` blank-string quirk: 11 rows contain `" "` (whitespace). Handled via `pd.to_numeric(errors='coerce')` and filled with `0.0` (all 11 instances have `tenure == 0`).
  2. Pseudo-categories in service columns: Features such as `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, and `StreamingMovies` include `"No internet service"`, and `MultipleLines` includes `"No phone service"`. These are collapsed into `"No"` to eliminate artificial multicollinearity.

---

## 4. Scope & Week 1 Deliverables

1. **Business Framing & Data Ingestion:**
   - Formalization of the business problem, tying churn to CLV ($\text{CLV} = \text{MonthlyCharges} \times \text{tenure}$).
   - Ingestion and validation of raw dataset into `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
2. **Exploratory Data Analysis & Data Quality Audit:**
   - Remediation of the `TotalCharges` blank-string quirk.
   - Consolidation of redundant pseudo-categories (`"No internet service"` / `"No phone service"` $\rightarrow$ `"No"`).
   - Cohort churn analysis across Contract type, Tenure buckets, and Payment channels.
   - Formal hypothesis formulation and statistical testing.

---

## 5. Acceptance Criteria & Success Benchmarks

1. **Data Integrity:** Zero unhandled null or invalid string values in pipeline inputs; raw data ingested reproducibly.
2. **Quality Audit:** Documented churn rates segmented by contract type, tenure buckets, and payment methods.
3. **Hypothesis Coverage:** At least 4 empirically validated hypotheses detailing operational retention opportunities.
4. **Reproducibility:** Self-contained Jupyter Notebook (`notebooks/01_eda.ipynb`) and modular cleaning scripts executing error-free in the Python virtual environment.
