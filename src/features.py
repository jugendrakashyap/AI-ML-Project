"""
Feature Engineering & Data Preprocessing Pipeline for Telco Customer Churn.

This module provides end-to-end preprocessing, feature engineering, encoding,
scaling, train/test splitting, and class-imbalance mitigation routines (SMOTE, class weighting).
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE


# Categorical service columns subject to pseudo-category collapsing
INTERNET_SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# All 8 bundle add-on services evaluated for customer stickiness
ALL_SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Binary 2-value demographic and account features
BINARY_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
]

# Continuous numeric attributes
NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
    "services_count",
    "charge_ratio",
    "clv",
]

# Multi-class nominal / ordinal features to encode
CATEGORICAL_COLS = [
    "InternetService",
    "Contract",
    "PaymentMethod",
    "tenure_bucket",
]


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Telco churn dataset addressing known data quirks:
    1. Fix TotalCharges blank strings (' ') and coerce to float64, imputing $0.00 for tenure == 0.
    2. Collapse pseudo-categories 'No internet service' and 'No phone service' into 'No'.
    """
    data = df.copy()

    # 1. Quirk: TotalCharges blank-string conversion
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0.0)

    # 2. Quirk: Pseudo-categories collapsing
    for col in INTERNET_SERVICE_COLS:
        if col in data.columns:
            data[col] = data[col].replace({"No internet service": "No"})

    if "MultipleLines" in data.columns:
        data["MultipleLines"] = data["MultipleLines"].replace({"No phone service": "No"})

    return data


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive domain-informed features required by Week 2 specifications:
    - tenure_bucket: Segmented lifecycle cohorts (0-12, 13-24, 25-36, 37-48, 49-60, 61-72).
    - avg_monthly_spend: Historical monthly spend = TotalCharges / tenure (fallback to MonthlyCharges for tenure=0).
    - services_count: Total count of active bundled value-add services (0 to 8).
    - charge_ratio: Current MonthlyCharges vs avg_monthly_spend (detects price hikes / promo expirations).
    - clv: Customer Lifetime Value proxy = MonthlyCharges * tenure.
    - has_support_security: Binary flag for high-retention support services (TechSupport or OnlineSecurity).
    - has_streaming: Binary flag for media streaming services (StreamingTV or StreamingMovies).
    """
    data = df.copy()

    # 1. tenure_bucket
    data["tenure_bucket"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 36, 48, 60, 72],
        labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"],
    )

    # 2. avg_monthly_spend
    data["avg_monthly_spend"] = np.where(
        data["tenure"] > 0,
        data["TotalCharges"] / data["tenure"],
        data["MonthlyCharges"],
    )

    # 3. services_count
    active_services = [col for col in ALL_SERVICE_COLS if col in data.columns]
    data["services_count"] = (data[active_services] == "Yes").sum(axis=1)

    # 4. charge_ratio (detect price escalations vs historical average)
    data["charge_ratio"] = data["MonthlyCharges"] / (data["avg_monthly_spend"] + 1e-4)

    # 5. Customer Lifetime Value (CLV proxy)
    data["clv"] = data["MonthlyCharges"] * data["tenure"]

    # 6. High-impact service interaction flags
    if "OnlineSecurity" in data.columns and "TechSupport" in data.columns:
        data["has_support_security"] = (
            (data["OnlineSecurity"] == "Yes") | (data["TechSupport"] == "Yes")
        ).astype(int)

    if "StreamingTV" in data.columns and "StreamingMovies" in data.columns:
        data["has_streaming"] = (
            (data["StreamingTV"] == "Yes") | (data["StreamingMovies"] == "Yes")
        ).astype(int)

    return data


def encode_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 2-class string features into clean 0/1 integers.
    """
    data = df.copy()
    for col in BINARY_COLS:
        if col in data.columns:
            if col == "gender":
                data[col] = (data[col] == "Male").astype(int)
            else:
                data[col] = (data[col] == "Yes").astype(int)

    if "SeniorCitizen" in data.columns:
        data["SeniorCitizen"] = data["SeniorCitizen"].astype(int)

    return data


def build_preprocessor() -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer for scaling numeric features and
    one-hot encoding categorical features with drop='first' to prevent collinearity.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_COLS,
            ),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="passthrough",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, passthrough_cols: List[str]) -> List[str]:
    """
    Extract transformed feature names from fitted ColumnTransformer.
    """
    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLS).tolist()
    return NUMERIC_COLS + cat_names + passthrough_cols


def prepare_train_test_data(
    filepath: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Complete end-to-end data ingestion, cleaning, feature engineering,
    stratified train/test splitting, and leakage-free transformer fitting.

    Returns:
        dict containing:
            - X_train, X_test (numpy arrays ready for modeling)
            - y_train, y_test (1D target arrays: 1=Churn, 0=Retained)
            - feature_names (List[str] matching columns in X_train/X_test)
            - preprocessor (fitted ColumnTransformer)
            - df_train_raw, df_test_raw (unscaled DataFrames containing customerID, tenure, CLV for error analysis)
    """
    raw_df = pd.read_csv(filepath)
    cleaned_df = clean_raw_data(raw_df)
    fe_df = create_engineered_features(cleaned_df)

    # Target variable
    y = (fe_df["Churn"] == "Yes").astype(int).values

    # Keep customer tracking columns for downstream business analysis
    customer_ids = fe_df["customerID"].values

    # Pre-process binary columns
    encoded_df = encode_binary_features(fe_df)

    # Feature matrix X
    drop_cols = ["customerID", "Churn"]
    feature_df = encoded_df.drop(columns=drop_cols)

    # Stratified Split (80% Train, 20% Test)
    train_idx, test_idx = train_test_split(
        np.arange(len(fe_df)),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    df_train_raw = fe_df.iloc[train_idx].copy().reset_index(drop=True)
    df_test_raw = fe_df.iloc[test_idx].copy().reset_index(drop=True)

    X_train_df = feature_df.iloc[train_idx].copy().reset_index(drop=True)
    X_test_df = feature_df.iloc[test_idx].copy().reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Passthrough columns: binary columns + interaction indicators + SeniorCitizen
    passthrough_cols = [
        col for col in X_train_df.columns
        if col not in NUMERIC_COLS and col not in CATEGORICAL_COLS
    ]

    # Fit preprocessor strictly on training data
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train_df)
    X_test_transformed = preprocessor.transform(X_test_df)

    feature_names = get_feature_names(preprocessor, passthrough_cols)

    return {
        "X_train": X_train_transformed,
        "X_test": X_test_transformed,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "df_train_raw": df_train_raw,
        "df_test_raw": df_test_raw,
        "customer_ids_train": customer_ids[train_idx],
        "customer_ids_test": customer_ids[test_idx],
        "passthrough_cols": passthrough_cols,
    }


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Synthetic Minority Over-sampling Technique (SMOTE) strictly on training data
    to balance the minority churn class (~26.6% -> 50.0%).
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def get_class_weight_dict(y_train: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights for cost-sensitive learning algorithms.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {cls: weight for cls, weight in zip(classes, weights)}


if __name__ == "__main__":
    print("Testing Feature Pipeline...")
    data = prepare_train_test_data()
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"X_test shape: {data['X_test'].shape}")
    print(f"Total features: {len(data['feature_names'])}")
    print(f"Train churn rate: {data['y_train'].mean():.4f}")
    print(f"Test churn rate: {data['y_test'].mean():.4f}")
    print(f"Feature Names: {data['feature_names']}")
