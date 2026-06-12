import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)
from config import TARGET, FRAUD_VAL, F_BETA


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "pr_auc":    round(float(average_precision_score(y_true, y_prob)), 6),
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "f2":        round(float(fbeta_score(y_true, y_pred, beta=F_BETA, zero_division=0)), 6),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold": threshold,
    }


def evaluate_clf(clf, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_prob = clf.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test.values, y_prob)
