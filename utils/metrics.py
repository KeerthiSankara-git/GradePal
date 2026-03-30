import numpy as np
from sklearn.metrics import (accuracy_score, f1_score,
                             cohen_kappa_score, confusion_matrix,
                             classification_report)

LABEL_NAMES = ["incorrect", "partially correct", "correct"]

def evaluate(y_true: list, y_pred: list, split_name: str = "") -> dict:
    """
    Run all grading metrics and print a summary.
    Returns a dict of all scores for logging to results.
    """
    acc   = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    f1_w  = f1_score(y_true, y_pred, average="weighted")
    f1_m  = f1_score(y_true, y_pred, average="macro")

    print(f"\n── Metrics: {split_name} ──")
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  Quadratic WK     : {kappa:.4f}")
    print(f"  Weighted F1      : {f1_w:.4f}")
    print(f"  Macro F1         : {f1_m:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=LABEL_NAMES)}")

    return {
        "split": split_name,
        "accuracy": round(acc, 4),
        "quadratic_wk": round(kappa, 4),
        "weighted_f1": round(f1_w, 4),
        "macro_f1": round(f1_m, 4)
    }


def over_grading_rate(y_true: list, y_pred: list,
                      passing_threshold: int = 1) -> float:
    """
    For adversarial evaluation.
    Returns fraction of adversarial answers (all should be 0)
    that received a passing score (>= threshold).
    """
    over_graded = sum(1 for yt, yp in zip(y_true, y_pred)
                      if yt < passing_threshold and yp >= passing_threshold)
    return round(over_graded / len(y_true), 4)