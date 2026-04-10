"""
evaluation/evaluate_unseen.py

Evaluates the optimized GradeOpt grader on unseen splits using
the final grading notes from results/gradeopt_notes.json.

Run:
    python -m evaluation.evaluate_unseen
"""

import json
import time
import pandas as pd
from pathlib import Path

from config import UNSEEN_ANS_PATH, UNSEEN_Q_PATH, BATCH_DELAY, LABEL_MAP
from utils.metrics import evaluate
from graders.gradeopt.grader_agent import grade_with_feedback


def evaluate_split(df: pd.DataFrame, grading_notes: dict, 
                   split_name: str, save_path: str) -> dict:
    """Grade all rows in df using optimized notes and evaluate."""
    
    # resume logic
    preds = []
    feedbacks = []
    start_index = 0
    
    if Path(save_path).exists():
        existing = pd.read_csv(save_path)
        if "predicted_label" in existing.columns:
            preds = existing["predicted_label"].tolist()
            feedbacks = existing.get("generated_feedback",
                       pd.Series([""] * len(preds))).tolist()
            start_index = len(preds)
            print(f"  Resuming from row {start_index}/{len(df)}")

    total = len(df)
    
    for i, (_, row) in enumerate(df.iterrows()):
        if i < start_index:
            continue

        qid = str(row["Question_id"])
        
        # for unseen_answers: use optimized notes if available
        # for unseen_question: no notes (questions never seen in training)
        notes = grading_notes.get(qid, "")

        result = grade_with_feedback(
            question=row["Question"],
            student_answer=row["Student Answer"],
            correct_answer=row["Correct Answer"],
            grading_notes=notes,
        )
        preds.append(result["label"])
        feedbacks.append(result["feedback"])

        # save progress after every row
        df_so_far = df.iloc[:len(preds)].copy()
        df_so_far["predicted_label"] = preds
        df_so_far["generated_feedback"] = feedbacks
        df_so_far.to_csv(save_path, index=False)

        if i % 10 == 0:
            print(f"    progress: {i}/{total} rows", end="\r")

        if i < total - 1:
            time.sleep(BATCH_DELAY)

    print(f"    progress: {total}/{total} rows ✓")

    y_true = df["output_label"].tolist()
    metrics = evaluate(y_true, preds, split_name=split_name)
    
    return metrics


def main():
    # load optimized grading notes
    notes_path = Path("results/gradeopt_notes.json")
    if not notes_path.exists():
        print("ERROR: results/gradeopt_notes.json not found.")
        print("Run the full pipeline first.")
        return

    with open(notes_path) as f:
        grading_notes = json.load(f)
    print(f"Loaded {len(grading_notes)} optimized grading notes.")

    Path("results").mkdir(exist_ok=True)
    all_metrics = []

    # ── Unseen Answers ────────────────────────────────────────────────────────
    # same questions as train but new student answers
    # optimized notes APPLY here
    print(f"\n{'='*60}")
    print("Evaluating on unseen_answers (same questions, new students)...")
    print(f"{'='*60}")
    
    unseen_ans_df = pd.read_csv(UNSEEN_ANS_PATH)
    before = len(unseen_ans_df)
    unseen_ans_df = unseen_ans_df.dropna(
        subset=["Question", "Student Answer", "Correct Answer"])
    if before - len(unseen_ans_df) > 0:
        print(f"  Dropped {before - len(unseen_ans_df)} rows with missing values")
    print(f"  {len(unseen_ans_df)} rows | "
          f"{unseen_ans_df['Question_id'].nunique()} unique questions")

    # check how many questions have optimized notes
    unseen_ans_qids = set(unseen_ans_df["Question_id"].astype(str).unique())
    notes_qids = set(grading_notes.keys())
    covered = unseen_ans_qids & notes_qids
    print(f"  Questions with optimized notes: {len(covered)}/{len(unseen_ans_qids)}")

    ans_metrics = evaluate_split(
        df=unseen_ans_df,
        grading_notes=grading_notes,
        split_name="gradeopt_unseen_answers",
        save_path="results/gradeopt_unseen_answers.csv"
    )
    ans_metrics["split"] = "unseen_answers"
    all_metrics.append(ans_metrics)

    # ── Unseen Questions ──────────────────────────────────────────────────────
    # completely new questions never seen during training
    # optimized notes DO NOT apply — graded with empty notes
    print(f"\n{'='*60}")
    print("Evaluating on unseen_question (brand new questions)...")
    print(f"{'='*60}")

    unseen_q_df = pd.read_csv(UNSEEN_Q_PATH)
    before = len(unseen_q_df)
    unseen_q_df = unseen_q_df.dropna(
        subset=["Question", "Student Answer", "Correct Answer"])
    if before - len(unseen_q_df) > 0:
        print(f"  Dropped {before - len(unseen_q_df)} rows with missing values")
    print(f"  {len(unseen_q_df)} rows | "
          f"{unseen_q_df['Question_id'].nunique()} unique questions")

    # check overlap with training questions
    unseen_q_qids = set(unseen_q_df["Question_id"].astype(str).unique())
    overlap = unseen_q_qids & notes_qids
    print(f"  Questions with optimized notes: {len(overlap)}/{len(unseen_q_qids)}")
    print(f"  Note: unseen_question has new questions — grading without notes")

    q_metrics = evaluate_split(
        df=unseen_q_df,
        grading_notes={},  # empty notes — brand new questions
        split_name="gradeopt_unseen_question",
        save_path="results/gradeopt_unseen_question.csv"
    )
    q_metrics["split"] = "unseen_question"
    all_metrics.append(q_metrics)

    # ── Save all metrics ──────────────────────────────────────────────────────
    with open("results/gradeopt_unseen_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("UNSEEN EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Split':<20} {'Accuracy':<12} {'QWK':<10} {'Weighted F1'}")
    for m in all_metrics:
        print(
            f"{m['split']:<20} "
            f"{m.get('accuracy', 0):<12.4f} "
            f"{m.get('quadratic_wk', 0):<10.4f} "
            f"{m.get('weighted_f1', 0):.4f}"
        )
    print("=" * 60)
    print(f"\nSaved → results/gradeopt_unseen_metrics.json")


if __name__ == "__main__":
    main()