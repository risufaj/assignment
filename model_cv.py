"""
Cross-validation and final test evaluation code + MLflow logging.
"""

import numpy as np
import mlflow
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

def compute_metrics(y_true, y_pred, y_proba):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_proba)
    return acc, prec, rec, f1, auc

def cross_validate_and_log_model(model_name, pipeline, X, y, imbalance_str="none", cv_splits=5):
    """
    1) K-fold cross-validation on pipeline,
    2) Log average metrics to MLflow,
    3) Fit pipeline on entire training set => return pipeline
    """
    

    with mlflow.start_run(run_name=f"CV_{model_name}_{imbalance_str}"):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("imbalance_handling", imbalance_str)
        mlflow.log_param("n_splits", cv_splits)

        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc"
        }
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)

        acc_mean  = np.mean(results["test_accuracy"])
        prec_mean = np.mean(results["test_precision"])
        rec_mean  = np.mean(results["test_recall"])
        f1_mean   = np.mean(results["test_f1"])
        auc_mean  = np.mean(results["test_roc_auc"])

        mlflow.log_metric("accuracy_mean", acc_mean)
        mlflow.log_metric("precision_mean", prec_mean)
        mlflow.log_metric("recall_mean", rec_mean)
        mlflow.log_metric("f1_mean", f1_mean)
        mlflow.log_metric("roc_auc_mean", auc_mean)

        print(f"\nCross-Validation {cv_splits}-fold: {model_name} | {imbalance_str}")
        print(f"  Mean Accuracy:  {acc_mean:.4f}")
        print(f"  Mean Precision: {prec_mean:.4f}")
        print(f"  Mean Recall:    {rec_mean:.4f}")
        print(f"  Mean F1:        {f1_mean:.4f}")
        print(f"  Mean ROC AUC:   {auc_mean:.4f}")

        # Train final on full data
        pipeline.fit(X, y)
        mlflow.sklearn.log_model(pipeline, artifact_path=f"{model_name}_cv_pipeline")
    return pipeline

def final_test_evaluation(model_pipeline, X_test, y_test):
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    acc, prec, rec, f1, auc = compute_metrics(y_test, y_pred, y_proba)
    print("\n*** TEST SET EVALUATION ***")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {auc:.4f}")
