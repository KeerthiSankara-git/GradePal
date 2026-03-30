"""
graders/gradeopt/pipeline.py

Orchestrates the full GradeOpt loop — no rubric JSON files needed.
Everything comes directly from the EngSAF dataset columns:
  - Question, Correct Answer, Student Answer, output_label, feedback

Loop per iteration:
  1. Grade all val examples using current grading_notes (grader_agent)
  2. Evaluate metrics (QWK, accuracy, weighted F1)
  3. Collect grading errors per question
  4. Reflect on errors using gold feedback as context (reflector_agent)
  5. Refine grading_notes for each question with errors (refiner_agent)

Usage:
    python graders/gradeopt/pipeline.py               # 1 iteration (default)
    python graders/gradeopt/pipeline.py --iters 3     # 3 iterations

Outputs:
    results/gradeopt_notes.json     — final per-question grading notes
    results/gradeopt_metrics.json   — per-iteration metrics
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from config import VAL_PATH, LABEL_MAP, BATCH_DELAY
from utils.metrics import evaluate
from graders.gradeopt import grader_agent, reflector_agent, refiner_agent

NOTES_PATH   = Path("results/gradeopt_notes.json")
METRICS_PATH = Path("results/gradeopt_metrics.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def grade_all(df: pd.DataFrame, grading_notes: dict) -> list[int]:
    """
    Grade every row in df. grading_notes is a dict of {question_id_str: notes_str}.
    On iteration 0 all notes are empty strings — grader uses correct_answer only.
    """
    preds = []
    for i, (_, row) in enumerate(df.iterrows()):
        qid = str(row["Question_id"])
        notes = grading_notes.get(qid, "")

        pred = grader_agent.grade(
            question=row["Question"],
            student_answer=row["Student Answer"],
            correct_answer=row["Correct Answer"],
            grading_notes=notes,
        )
        preds.append(pred)

        # Rate limit: pause every call to stay under free-tier quota
        if i < len(df) - 1:
            time.sleep(BATCH_DELAY)

    return preds


def collect_errors(df: pd.DataFrame, preds: list[int]) -> dict:
    """
    Returns { question_id_str: [ {student_answer, true_label, predicted_label}, ... ] }
    Only includes questions that had at least one wrong prediction.
    """
    errors_by_q: dict[str, list] = {}
    for (_, row), pred in zip(df.iterrows(), preds):
        true = int(row["output_label"])
        if pred != true:
            qid = str(row["Question_id"])
            errors_by_q.setdefault(qid, [])
            errors_by_q[qid].append({
                "student_answer": row["Student Answer"],
                "true_label": LABEL_MAP[true],
                "predicted_label": LABEL_MAP[pred],
            })
    return errors_by_q


def get_gold_feedback(df: pd.DataFrame, qid: str, n: int = 5) -> list[str]:
    """Pull up to n gold feedback strings for a question from the dataset."""
    subset = df[df["Question_id"].astype(str) == qid]
    return subset["feedback"].dropna().head(n).tolist()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_pipeline(n_iters: int = 1) -> None:
    print("=" * 60)
    print(f"GradeOpt Pipeline  |  {n_iters} iteration(s)")
    print("=" * 60)

    print(f"\nLoading validation data from {VAL_PATH} ...")
    df = pd.read_csv(VAL_PATH)
    y_true = df["output_label"].tolist()
    print(f"  {len(df)} rows | {df['Question_id'].nunique()} unique questions")

    # grading_notes: per-question string of refined criteria
    # starts empty — iteration 0 is baseline (correct_answer only)
    grading_notes: dict[str, str] = {}

    all_metrics = []
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for iteration in range(n_iters):
        print(f"\n{'─'*60}")
        print(f"ITERATION {iteration + 1} / {n_iters}")
        print(f"{'─'*60}")

        # ── Step 1: Grade ───────────────────────────────────────────────
        print(f"\n[1/3] Grading {len(df)} examples ...")
        preds = grade_all(df, grading_notes)

        # ── Step 2: Evaluate ────────────────────────────────────────────
        print("\n[2/3] Evaluating ...")
        metrics = evaluate(y_true, preds, split_name=f"iter_{iteration + 1}_val")
        metrics["iteration"] = iteration + 1
        all_metrics.append(metrics)

        with open(METRICS_PATH, "w") as f:
            json.dump(all_metrics, f, indent=2)

        if iteration == n_iters - 1:
            print("\nFinal iteration — skipping reflect/refine step.")
            break

        # ── Step 3: Reflect + Refine ────────────────────────────────────
        print("\n[3/3] Reflecting on errors and refining grading notes ...")
        errors_by_q = collect_errors(df, preds)
        print(f"  {len(errors_by_q)} question(s) had grading errors.")

        updated = 0
        for qid, errors in errors_by_q.items():
            row0 = df[df["Question_id"].astype(str) == qid].iloc[0]
            gold_fb = get_gold_feedback(df, qid)

            # Reflect
            critique = reflector_agent.reflect(
                question=row0["Question"],
                correct_answer=row0["Correct Answer"],
                errors=errors,
                gold_feedback_samples=gold_fb,
            )

            # Refine
            new_notes = refiner_agent.refine(
                question=row0["Question"],
                correct_answer=row0["Correct Answer"],
                critique=critique,
                previous_notes=grading_notes.get(qid, ""),
            )
            grading_notes[qid] = new_notes
            updated += 1

            time.sleep(BATCH_DELAY)  # respect free-tier rate limit

        print(f"  {updated} question(s) had their grading notes updated.")

        # Save grading notes after each iteration
        NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(NOTES_PATH, "w") as f:
            json.dump(grading_notes, f, indent=2)
        print(f"  Grading notes saved → {NOTES_PATH}")

    # ── Final saves ─────────────────────────────────────────────────────
    with open(NOTES_PATH, "w") as f:
        json.dump(grading_notes, f, indent=2)

    print(f"\nFinal grading notes → {NOTES_PATH}")
    print(f"All metrics         → {METRICS_PATH}")

    # Summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Iter':<6} {'Accuracy':<12} {'QWK':<10} {'Weighted F1'}")
    for m in all_metrics:
        print(
            f"{m['iteration']:<6} "
            f"{m.get('accuracy', 0):<12.4f} "
            f"{m.get('qwk', 0):<10.4f} "
            f"{m.get('weighted_f1', 0):.4f}"
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GradeOpt optimization loop")
    parser.add_argument(
        "--iters", type=int, default=1,
        help="Number of Grader→Reflector→Refiner iterations (default: 1)"
    )
    args = parser.parse_args()
    run_pipeline(n_iters=args.iters)