"""
Simple rule-based (heuristic) baseline for the Census Income dataset.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def is_high_education(edu_str):
    """
    Checks if the education string includes Bachelors, Masters, Doctorate, or Prof school.
    """
    high_ed_keywords = ["bachelors", "masters", "doctorate", "prof school"]
    if isinstance(edu_str, str):
        low_str = edu_str.lower()
        return any(kw in low_str for kw in high_ed_keywords)
    return False

def heuristic_rule(row):
    """
    1) If capital_gains > 0 or dividends_from_stocks > 0 => high income
    2) Else if weeks_worked_in_year >= 50 and high_education => high income
    3) Else => low income
    """
    if row["capital_gains"] > 0 or row["dividends_from_stocks"] > 0:
        return 1
    elif (row["weeks_worked_in_year"] >= 50) and is_high_education(row["education"]):
        return 1
    else:
        return 0

def evaluate_heuristic(df, target_col="high_income"):
    """
    Applies the heuristic_rule to each row, compares with target_col
    and returns standard metrics.
    """
    df_copy = df.copy()

    df_copy["heuristic_pred"] = df_copy.apply(heuristic_rule, axis=1)
    y_true = df_copy[target_col]
    y_pred = df_copy["heuristic_pred"]

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_pred)
    return acc, prec, rec, f1, auc
