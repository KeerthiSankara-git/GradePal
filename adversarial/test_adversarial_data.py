"""
evaluation/adversarial_eval.py

Evaluates all three grading systems on the adversarial subset.
"""

import json
import time
import pandas as pd
from pathlib import Path
from collections import Counter

from config import BATCH_DELAY
from utils.metrics import over_grading_rate
from graders.baseline import grade_single as baseline_grade
from graders.static_rubric import grade as static_grade
from graders.gradeopt.grader_agent import grade_with_feedback



def compute_per_type_metrics(df, preds):
    from collections import Counter

    results = {}
    df_out = df.copy()
    df_out["predicted_label"] = preds

    for attack in df_out["attack_type"].unique():
        subset = df_out[df_out["attack_type"] == attack]
        type_preds = subset["predicted_label"].tolist()

        counts = Counter(type_preds)
        total = len(type_preds)

        results[attack] = {
            "over_grading_rate": over_grading_rate([0]*total, type_preds),
            "pct_0": counts.get(0, 0) / total,
            "pct_1": counts.get(1, 0) / total,
            "pct_2": counts.get(2, 0) / total,
            "count": total
        }

    return results

def save_per_type_comparison(all_metrics):
    comparison = {}

    for system in all_metrics:
        name = system["system"]
        per_type = system.get("per_attack_type", {})

        for attack, stats in per_type.items():
            comparison.setdefault(attack, {})
            comparison[attack][name] = stats["over_grading_rate"]

    with open("results/per_attack_type_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\nSaved per-attack comparison -> results/per_attack_type_comparison.json")
    return comparison


def evaluate_system(df: pd.DataFrame, system_name: str,
                    grade_fn, save_path: str) -> dict:
    """Run a grader on adversarial subset and compute over-grading rate."""
    preds = []
    start_index = 0

    # resume logic
    if Path(save_path).exists():
        existing = pd.read_csv(save_path)
        if "predicted_label" in existing.columns:
            preds = existing["predicted_label"].tolist()
            start_index = len(preds)
            print(f"  Resuming from row {start_index}/{len(df)}")

    for i, (_, row) in enumerate(df.iterrows()):
        if i < start_index:
            continue

        pred = grade_fn(row)
        preds.append(pred)

        # save progress
        df_so_far = df.iloc[:len(preds)].copy()
        df_so_far["predicted_label"] = preds
        df_so_far.to_csv(save_path, index=False)

        if i % 10 == 0:
            print(f"    progress: {i}/{len(df)}", end="\r")

        if i < len(df) - 1:
            time.sleep(BATCH_DELAY)

    print(f"    progress: {len(df)}/{len(df)} ✓")

    # all adversarial answers should be incorrect (0)
    y_true = [0] * len(preds)
    ogr = over_grading_rate(y_true, preds, passing_threshold=1)
    avg_score = sum(preds) / len(preds)

    label_counts = Counter(preds)
    total = len(preds)

    pct_0 = label_counts.get(0, 0) / total
    pct_1 = label_counts.get(1, 0) / total
    pct_2 = label_counts.get(2, 0) / total

    print(f"\n── Adversarial: {system_name} ──")
    print(f"  Over-grading rate: {ogr:.4f} ({ogr*100:.1f}% fooled)")
    print(f"  Average predicted score: {avg_score:.4f}")

    print(f"\n  Label distribution:")
    print(f"    0 (correct reject): {pct_0*100:.1f}%")
    print(f"    1 (partial credit): {pct_1*100:.1f}%")
    print(f"    2 (fully fooled):  {pct_2*100:.1f}%")


    per_type = {}
    if "attack_type" in df.columns:
        per_type = compute_per_type_metrics(df, preds)

        print(f"\n  Per-type over-grading rate:")
        for attack, stats in per_type.items():
            print(f"    {attack:<25}: {stats['over_grading_rate']*100:.1f}%")

    return {
    "system": system_name,
    "over_grading_rate": round(ogr, 4),
    "avg_predicted_score": round(avg_score, 4),
    "n_examples": len(preds),
    "per_attack_type": per_type
}

def print_comparison_table(comparison):
    print("\n" + "="*60)
    print("PER-ATTACK TYPE COMPARISON")
    print("="*60)

    systems = list(next(iter(comparison.values())).keys())
    header = f"{'Attack Type':<25}" + "".join([f"{s:<15}" for s in systems])
    print(header)

    for attack, system_vals in comparison.items():
        row = f"{attack:<25}"
        for s in systems:
            val = system_vals.get(s, 0)
            row += f"{val*100:<15.1f}"
        print(row)


def main():
    # load adversarial subset
    adv_path = Path("data/adversarial_dataset.csv")
    if not adv_path.exists():
        print("ERROR: data/adversarial_dataset.csv not found.")
        return

    df = pd.read_csv(adv_path)
    df = df.dropna(subset=["Question", "Student Answer", "Correct Answer"])
    print(f"Loaded {len(df)} adversarial examples")

    if "attack_type" in df.columns:
        print(f"Types: {df['attack_type'].value_counts().to_dict()}")

    # load optimized notes for GradeOpt
    grading_notes = {}
    notes_path = Path("results/gradeopt_notes.json")
    if notes_path.exists():
        with open(notes_path) as f:
            grading_notes = json.load(f)
        print(f"Loaded {len(grading_notes)} grading notes")

    Path("results").mkdir(exist_ok=True)
    all_metrics = []

    # ── Baseline ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Baseline Grader...")
    print(f"{'='*60}")

    baseline_metrics = evaluate_system(
        df=df,
        system_name="Baseline",
        grade_fn=lambda row: baseline_grade(
            question=str(row["Question"]),
            student_answer=str(row["Student Answer"])
        ),
        save_path="results/adversarial_baseline.csv"
    )
    all_metrics.append(baseline_metrics)

    # ── Static Rubric ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Static Rubric Grader...")
    print(f"{'='*60}")

    static_metrics = evaluate_system(
        df=df,
        system_name="Static Rubric",
        grade_fn=lambda row: static_grade(
            question=str(row["Question"]),
            correct_answer=str(row["Correct Answer"]),
            student_answer=str(row["Student Answer"])
        ),
        save_path="results/adversarial_static_rubric.csv"
    )
    all_metrics.append(static_metrics)

    # ── GradeOpt ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GradeOpt Grader...")
    print(f"{'='*60}")

    def gradeopt_fn(row):
        qid = str(row.get("Question_id", ""))
        notes = grading_notes.get(qid, "")
        result = grade_with_feedback(
            question=str(row["Question"]),
            student_answer=str(row["Student Answer"]),
            correct_answer=str(row["Correct Answer"]),
            grading_notes=notes,
        )
        return result["label"]

    gradeopt_metrics = evaluate_system(
        df=df,
        system_name="GradeOpt",
        grade_fn=gradeopt_fn,
        save_path="results/adversarial_gradeopt.csv"
    )
    all_metrics.append(gradeopt_metrics)

    # ── Summary ───────────────────────────────────────────────────────────────
    with open("results/adversarial_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    comparison = save_per_type_comparison(all_metrics)
    print_comparison_table(comparison)



if __name__ == "__main__":
    main()