"""
Feature engineering:
- Numeric expansions (indicator + log transforms)
- Rare category grouping "informed from data" ratio
- One-hot encoding for all categorical columns
"""

import numpy as np
import pandas as pd
from config import numerical_columns, RARE_CATEGORY_RATIO

def create_numeric_features(df):
    """
    For wage_per_hour, capital_gains, capital_losses, dividends_from_stocks:
      - indicator + log transform
    Also approximate annual wage, manager_flag, etc.
    """
    # wage
    df["wage_per_hour_indicator"] = (df["wage_per_hour"] > 0).astype(int)
    df["wage_per_hour_log"] = np.where(df["wage_per_hour"] > 0, np.log1p(df["wage_per_hour"]), 0)
    df["approx_annual_wage"] = np.where(df["wage_per_hour"] > 0,
        df["wage_per_hour"] * df["weeks_worked_in_year"] * 40, 0)

    # gains
    df["capital_gains_indicator"] = (df["capital_gains"] > 0).astype(int)
    df["capital_gains_log"] = np.where(df["capital_gains"] > 0, np.log1p(df["capital_gains"]), 0)

    # losses
    df["capital_losses_indicator"] = (df["capital_losses"] > 0).astype(int)
    df["capital_losses_log"] = np.where(df["capital_losses"] > 0, np.log1p(df["capital_losses"]), 0)

    # dividends
    df["dividends_indicator"] = (df["dividends_from_stocks"] > 0).astype(int)
    df["dividends_log"] = np.where(df["dividends_from_stocks"] > 0, np.log1p(df["dividends_from_stocks"]), 0)

    # manager
    df["manager_flag"] = (df["num_persons_worked_for_employer"] >= 1).astype(int)
    return df

def group_rare_categories(df, col, ratio):
    """
    ratio * len(df) => threshold for grouping into 'Other'.
    If freq < threshold => 'Other'
    """
    freq = df[col].value_counts()
    threshold = len(df) * ratio
    rare_cats = freq[freq < threshold].index
    df[col] = df[col].replace(rare_cats, "Other")
    return df


def process_categorical_features(df, categorical_columns, reference_columns=None):
    """
    1) Group rare categories for each col in categorical_columns
       using ratio=RARE_CATEGORY_RATIO
    2) One-hot encode all columns in categorical_columns
    3) If reference_columns is provided, ensure all columns in reference_columns exist in result
    """
    for col in categorical_columns:
        if col in df.columns:
            df = group_rare_categories(df, col, ratio=RARE_CATEGORY_RATIO)
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    # Get dummies
    df = pd.get_dummies(df, columns=[c for c in categorical_columns if c in df.columns], drop_first=False)
    
    # Add missing columns if reference_columns is provided
    if reference_columns is not None:
        for col in reference_columns:
            if col not in df.columns:
                df[col] = 0
    
    return df

def transform_dataset(df, categorical_columns, reference_columns=None):
    """
    Applies numeric expansions + categorical processing
    """
    # Numeric expansions
    df = create_numeric_features(df)
    
    # Ensure missing numeric columns exist
    for col in numerical_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Categorical expansions
    df = process_categorical_features(df, categorical_columns, reference_columns)
    
    return df

