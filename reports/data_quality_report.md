# Data Quality Report & Exploratory Data Analysis (EDA)
**Project:** Telco Customer Churn Prediction & Retention Optimization  
**Dataset:** `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (IBM Cognos Analytics)  
**Deliverable:** Week 1 — Data Quality Audit & Hypothesis Validation  
**Date:** October 03, 2026  

---

## 1. Dataset Dimensions & Executive Overview

| Metric | Value | Business / Analytical Meaning |
| :--- | :--- | :--- |
| **Total Rows (Customers)** | **7,043** | Distinct customer accounts recorded in California |
| **Total Features** | **21** | 1 ID, 4 Demographics, 6 Financial/Contract, 9 Services, 1 Target |
| **Total Churned Customers** | **1,869** | Subscribers who left within the last period |
| **Total Retained Customers** | **5,174** | Subscribers currently active |
| **Overall Churn Rate** | **26.54%** | Moderately imbalanced binary target class |
| **Average Customer Tenure** | **32.37 months** | Median = 29 months, Range = 0 to 72 months |
| **Average Monthly Charges** | **$64.76** | Median = $70.35, Range = $18.25 to $118.75 |
| **Average Customer Lifetime Value** | **$2,279.73** | Median = $1,394.55 (CLV = MonthlyCharges $\times$ tenure) |

---

## 2. Identified Data Quirks & Remediation Actions

### 2.1 The `TotalCharges` Blank-String Quirk
- **Quirk Observed:** In the raw CSV, `TotalCharges` was loaded as an `object` (string) column rather than `float64`. Standard `.isnull()` or `.isna()` checks showed **0 missing values**, but parsing as numbers failed.
- **Root Cause Analysis:** Exactly **11 records** contained a single whitespace string (`" "`). Inspecting these 11 customer records revealed that every single one had `tenure == 0` (brand new subscribers whose first billing cycle had not yet settled).
- **Remediation Implemented:**
  ```python
  df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
  ```
- **Validation:** Converted column to `float64`. All 11 records accurately populated with `$0.00`, preserving valid customer history without dropping records.

### 2.2 Collapsing "No internet service" & "No phone service" Pseudo-Categories
- **Quirk Observed:** 6 internet add-on service attributes (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) had 3 distinct categories: `Yes`, `No`, and `No internet service`. Similarly, `MultipleLines` had `No phone service`.
- **Root Cause Analysis:** Subscribers without internet automatically received `"No internet service"`. In feature encoding, treating this as an independent category introduces redundant degrees of freedom and collinearity with `InternetService == 'No'`.
- **Remediation Implemented:**
  Collapsed `"No internet service"` and `"No phone service"` into `"No"`.
  ```python
  internet_cols = [
      "OnlineSecurity", "OnlineBackup", "DeviceProtection",
      "TechSupport", "StreamingTV", "StreamingMovies"
  ]
  for col in internet_cols:
      df[col] = df[col].replace({"No internet service": "No"})
  df["MultipleLines"] = df["MultipleLines"].replace({"No phone service": "No"})
  ```
- **Impact:** Each feature is now a clean binary indicator (`Yes` vs `No`), reducing dimensionality and preventing model instability.

---

## 3. Segment Churn Rate Analyses

### 3.1 Churn Rate by Contract Type (The Dominant Driver)
Contract commitment duration represents the single strongest predictor of customer churn in the entire dataset:

| Contract Type | Customer Count | % of Active Base | Churn Rate (%) | Churn Ratio vs Two-Year |
| :--- | :--- | :--- | :--- | :--- |
| **Month-to-month** | 3,875 | 55.02% | **42.71%** | **15.1x higher** |
| **One year** | 1,473 | 20.91% | **11.27%** | **4.0x higher** |
| **Two year** | 1,695 | 24.07% | **2.83%** | Baseline (1.0x) |

> **Key Finding:** Month-to-month subscribers represent 55% of the base but contribute over 88% of all churn events. Converting month-to-month customers to annual contracts yields the highest mathematical reduction in churn.

### 3.2 Churn Rate by Tenure Buckets
Segmenting tenure into 12-month increments illustrates the customer lifecycle risk curve:

| Tenure Bucket | Customer Count | Churn Rate (%) | Cumulative Churn Share |
| :--- | :--- | :--- | :--- |
| **0–12 months** | 2,186 | **47.44%** | 55.48% of all churners |
| **13–24 months** | 1,024 | **28.71%** | 15.73% of all churners |
| **25–36 months** | 832 | **22.12%** | 9.84% of all churners |
| **37–48 months** | 762 | **18.77%** | 7.65% of all churners |
| **49–60 months** | 832 | **14.54%** | 6.47% of all churners |
| **61–72 months** | 1,407 | **6.61%** | 4.83% of all churners |

> **Key Finding:** The first 12 months is the "critical danger zone" where nearly 1 out of 2 new subscribers churns. Retention efforts must front-load onboarding check-ins during months 1 to 6.

### 3.3 Churn Rate by Payment Method (Billing Friction)
Payment channel correlates heavily with customer stickiness:

| Payment Method | Customer Count | Churn Rate (%) | Automated vs Manual |
| :--- | :--- | :--- | :--- |
| **Electronic check** | 2,365 | **45.29%** | Manual / High Friction |
| **Mailed check** | 1,612 | **19.11%** | Manual |
| **Bank transfer (automatic)** | 1,544 | **16.71%** | Automated |
| **Credit card (automatic)** | 1,522 | **15.24%** | Automated |

> **Key Finding:** Customers using Electronic Checks churn at almost 3x the rate of those on automated Credit Card or Bank Transfer payments. Transitioning subscribers to auto-pay provides immediate churn mitigation.

### 3.4 Churn Rate by Internet Service Technology
Technology tier exhibits counter-intuitive pricing friction:

| Internet Service | Customer Count | Avg Monthly Charge | Churn Rate (%) |
| :--- | :--- | :--- | :--- |
| **Fiber optic** | 3,096 | $91.50 | **41.89%** |
| **DSL** | 2,421 | $58.10 | **18.96%** |
| **No Internet** | 1,526 | $21.08 | **7.40%** |

> **Key Finding:** Fiber optic accounts have the highest monthly revenue ($91.50/mo) but the highest churn rate (41.89%). Losing a Fiber Optic customer destroys significant CLV, making them priority candidates for proactive retention.

---

## 4. Formal Hypothesis List & Strategic Implications

| # | Hypothesis Statement | Empirical Test Result | Strategic Retention Implication |
| :--- | :--- | :--- | :--- |
| **H1** | *Subscribers on Month-to-Month contracts have statistically significantly higher churn than contractual subscribers.* | **Validated** ($p < 0.001$, $\chi^2 = 1184.6$). Churn: 42.7% vs 2.8%. | Offer targeted term-extension credits (e.g. 10% off for 12-month renewal) to month-to-month accounts. |
| **H2** | *Electronic check payment method introduces friction that exacerbates churn rates compared to automated billing.* | **Validated** ($p < 0.001$, $\chi^2 = 648.4$). Churn: 45.3% vs 15.2%. | Implement a $5 bill credit incentive for switching from Electronic Check to Auto-Pay credit card/bank transfer. |
| **H3** | *Fiber optic subscribers churn due to high monthly bills and lack of technical support/security bundles.* | **Validated**. Fiber optic churn is 41.9%. When `TechSupport == 'No'`, churn reaches 49.3%; when `TechSupport == 'Yes'`, churn drops to 22.8%. | Bundle complimentary 6-month Tech Support & Online Security with premium Fiber tiers. |
| **H4** | *Customer churn probability declines logarithmically with tenure duration.* | **Validated**. Churn drops from 47.4% (months 0–12) to 6.6% (months 61–72). | Design a structured "First-Year Nurture Program" featuring proactive milestone outreach at day 30, 90, and 180. |
| **H5** | *Subscribers with multiple bundled value-add services exhibit substantially higher retention.* | **Validated**. Customers with 3+ security/backup add-ons churn at only 12.4% vs 39.8% for 0 add-ons. | Cross-sell cybersecurity and cloud backup bundles to increase account stickiness. |

---

## 5. Summary Conclusion & Readiness for Week 2
- Data quirks have been thoroughly documented, resolved, and verified.
- The raw dataset is clean, reproducible, and ready for Feature Engineering in Week 2 (`src/features.py`).
- All deliverables for Week 1 (BRD, Raw Ingestion, EDA Notebook, Data Quality Report, Hypothesis List) are fully met.
