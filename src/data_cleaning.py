"""
Data Cleaning and Quality Audit Utilities for Telco Customer Churn.
Week 1 Focus:
- Fix TotalCharges blank-string bug (errors='coerce')
- Collapse pseudo-categories ('No internet service' / 'No phone service' -> 'No')
- Generate tenure buckets
- Calculate Customer Lifetime Value (CLV = MonthlyCharges * tenure)
"""
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_raw_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV dataset from data folder."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run data_loader.py first.")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded raw dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def audit_data_quality(df: pd.DataFrame) -> dict:
    """Perform data quality checks on the raw dataset."""
    blank_total_charges = (df["TotalCharges"].astype(str).str.strip() == "").sum()
    
    pseudo_cat_counts = {}
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in service_cols:
        if col in df.columns:
            pseudo_cat_counts[col] = int((df[col] == "No internet service").sum())
            
    if "MultipleLines" in df.columns:
        pseudo_cat_counts["MultipleLines"] = int((df["MultipleLines"] == "No phone service").sum())

    total_rows = len(df)
    churn_rate = float((df["Churn"] == "Yes").mean() * 100) if "Churn" in df.columns else 0.0

    audit_summary = {
        "total_records": total_rows,
        "total_columns": df.shape[1],
        "blank_total_charges_count": int(blank_total_charges),
        "pseudo_categorical_counts": pseudo_cat_counts,
        "churn_rate_percent": round(churn_rate, 2),
        "missing_values_per_col": df.isnull().sum().to_dict()
    }
    return audit_summary


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Telco Customer Churn dataset according to Week 1 specifications:
    1. Fix TotalCharges blank-string quirk with pd.to_numeric(errors='coerce')
    2. Impute TotalCharges for 0-tenure accounts with 0.0
    3. Collapse 'No internet service' and 'No phone service' into 'No'
    4. Create tenure buckets (0-12m, 13-24m, 25-36m, 37-48m, 49-60m, 61-72m)
    5. Calculate CLV (Customer Lifetime Value = MonthlyCharges * tenure)
    6. Convert target Churn to numeric Churn_Binary (1 = Yes, 0 = No)
    """
    df_clean = df.copy()

    # 1. TotalCharges blank string fix
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
    # All 11 blank TotalCharges have tenure == 0, so TotalCharges is legitimately 0.0
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

    # 2. Collapse pseudo-categories
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in service_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace({"No internet service": "No"})

    if "MultipleLines" in df_clean.columns:
        df_clean["MultipleLines"] = df_clean["MultipleLines"].replace({"No phone service": "No"})

    # 3. Create tenure buckets
    bins = [-1, 12, 24, 36, 48, 60, 72]
    labels = ["0-12m", "13-24m", "25-36m", "37-48m", "49-60m", "61-72m"]
    df_clean["tenure_bucket"] = pd.cut(df_clean["tenure"], bins=bins, labels=labels)

    # 4. Calculate Customer Lifetime Value (CLV = MonthlyCharges * tenure)
    df_clean["CLV"] = df_clean["MonthlyCharges"] * df_clean["tenure"]

    # 5. Target binary column
    if "Churn" in df_clean.columns:
        df_clean["Churn_Binary"] = df_clean["Churn"].map({"Yes": 1, "No": 0})

    logger.info("Data cleaning completed successfully.")
    return df_clean


if __name__ == "__main__":
    raw_df = load_raw_data()
    audit = audit_data_quality(raw_df)
    print("Audit Results:", audit)
    cleaned_df = clean_telco_data(raw_df)
    print("Cleaned shape:", cleaned_df.shape)
