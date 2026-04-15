import numpy as np
import json
import os
from sklearn.metrics import (accuracy_score, f1_score,
                             cohen_kappa_score, confusion_matrix,
                             classification_report)

LABEL_NAMES = ["incorrect", "partially correct", "correct"]

def evaluate(y_true: list, y_pred: list, split_name: str = "") -> dict:
    """
    Run all grading metrics, print a summary, and automatically save 
    the results to system2_metrics.json in the results directory.
    """
    # 1. Calculate the core metrics
    acc   = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    f1_w  = f1_score(y_true, y_pred, average="weighted")
    f1_m  = f1_score(y_true, y_pred, average="macro")

    # 2. Print summary to terminal for immediate feedback
    print(f"\n── Metrics: {split_name} ──")
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  Quadratic WK     : {kappa:.4f}")
    print(f"  Weighted F1      : {f1_w:.4f}")
    print(f"  Macro F1         : {f1_m:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=LABEL_NAMES)}")

    # 3. Structure the data for JSON
    metrics_data = {
        "split": split_name,
        "accuracy": round(acc, 4),
        "quadratic_wk": round(kappa, 4),
        "weighted_f1": round(f1_w, 4),
        "macro_f1": round(f1_m, 4)
    }

    # 4. Handle JSON File Persistence (System 2 Naming)
    # Locates the /results folder relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.abspath(os.path.join(current_dir, '..', 'results', 'system2_metrics.json'))

    # Ensure results directory exists
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    all_metrics = []
    # If the file already exists, load existing data to append to it
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            try:
                all_metrics = json.load(f)
                # Avoid duplicates: remove previous entry for the same split if it exists
                all_metrics = [m for m in all_metrics if m.get('split') != split_name]
            except json.JSONDecodeError:
                all_metrics = []

    all_metrics.append(metrics_data)

    # Write the updated list back to the file
    with open(results_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    print(f"\n[✔] Results appended to: {results_path}")

    return metrics_data


def over_grading_rate(y_true: list, y_pred: list,
                      passing_threshold: int = 1) -> float:
    """
    For adversarial evaluation.
    Returns fraction of adversarial answers (all should be 0)
    that received a passing score (>= threshold).
    """
    if not y_true:
        return 0.0
    over_graded = sum(1 for yt, yp in zip(y_true, y_pred)
                      if yt < passing_threshold and yp >= passing_threshold)
    return round(over_graded / len(y_true), 4)