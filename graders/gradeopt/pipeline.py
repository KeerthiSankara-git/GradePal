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

from config import VAL_PATH, LABEL_MAP, BATCH_DELAY, TRAIN_PATH
from utils import metrics
from utils.metrics import evaluate
from graders.gradeopt import grader_agent, reflector_agent, refiner_agent
from config import ADV_TRAIN_PATH, ADV_VAL_PATH


# NOTES_PATH   = Path("results/gradeopt_notes.json")
# METRICS_PATH = Path("results/gradeopt_metrics.json")

def get_paths(run_name):
    base = Path(f"results/{run_name}")
    base.mkdir(parents=True, exist_ok=True)

    return {
        "notes": base / "gradeopt_notes.json",
        "metrics": base / "gradeopt_metrics.json",
        "state": base / "gradeopt_state.json"
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def grade_all(df: pd.DataFrame, grading_notes: dict, 
              save_path: str = None) -> tuple[list[int], list[str]]:
    from graders.gradeopt.grader_agent import grade_with_feedback
    
    preds = []
    feedbacks = []
    start_index = 0
    total = len(df)

    # resume logic — load existing progress if file exists
    if save_path and Path(save_path).exists():
        existing = pd.read_csv(save_path)
        if "predicted_label" in existing.columns:
            preds = existing["predicted_label"].tolist()
            feedbacks = existing.get("generated_feedback", 
                       pd.Series([""] * len(preds))).tolist()
            start_index = len(preds)
            print(f"  Resuming from row {start_index}/{total}")

    for i, (_, row) in enumerate(df.iterrows()):
        if i < start_index:
            continue

        qid = str(row["Question_id"])
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
        if save_path:
            df_so_far = df.iloc[:len(preds)].copy()
            df_so_far["predicted_label"] = preds
            df_so_far["generated_feedback"] = feedbacks
            df_so_far.to_csv(save_path, index=False)

        if i % 10 == 0:
            print(f"    progress: {i}/{total} rows", end="\r")

        if i < total - 1:
            time.sleep(BATCH_DELAY)

    print(f"    progress: {total}/{total} rows ✓")
    return preds, feedbacks


def collect_errors(df: pd.DataFrame, preds: list[int]) -> dict:
    errors_by_q: dict[str, list] = {}
    for (_, row), pred in zip(df.iterrows(), preds):
        true = int(row["output_label"])
        if pred != true:
            qid = str(row["Question_id"])
            # skip nan question ids
            if qid == "nan" or not qid.strip():
                continue
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

def run_pipeline(n_iters: int = 1, sample: int = None) -> None:
    print("=" * 60)
    print(f"GradeOpt Pipeline  |  {n_iters} iteration(s)")
    print("=" * 60)

    # ── Resume from saved state if exists ─────────────────────────────────────
    # state_path = Path("results/gradeopt_state.json")
    
    paths = get_paths(args.run_name)

    NOTES_PATH = paths["notes"]
    METRICS_PATH = paths["metrics"]
    state_path = paths["state"]

    start_iteration = 0
    grading_notes: dict[str, str] = {}

    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        start_iteration = state.get("completed_iterations", 0)
        grading_notes = state.get("grading_notes", {})
        print(f"  Resuming from iteration {start_iteration + 1}")
    
    # ── Load TRAIN for optimization ───────────────────────────────────────────
    print(f"\nLoading TRAIN data from {TRAIN_PATH} ...")
    train_df = pd.read_csv(TRAIN_PATH)

    if args.use_adv_train:
        print(f"Loading adversarial training data from {ADV_TRAIN_PATH}")
        
        adv_train_df = pd.read_csv(ADV_TRAIN_PATH)
        train_df = pd.concat([train_df, adv_train_df], ignore_index=True)
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Training size: {len(train_df)}")

    if sample is not None:
        train_df = train_df.head(sample)
        print(f"  [DEV MODE] Using {sample} rows only")

    print(f"  {len(train_df)} rows | {train_df['Question_id'].nunique()} unique questions")

    # ── Load VAL for evaluation ───────────────────────────────────────────────
    print(f"\nLoading VAL data from {VAL_PATH} ...")
    val_df = pd.read_csv(VAL_PATH)
    if args.use_adv_train:
        print(f"Loading adversarial validation data from {ADV_VAL_PATH}")
        adv_val_df = pd.read_csv(ADV_VAL_PATH)
        val_df = pd.concat([val_df, adv_val_df], ignore_index=True)
        val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    if sample is not None:
        val_df = val_df.head(sample)
        print(f"  [DEV MODE] Using {sample} rows only")

    y_true_val = val_df["output_label"].tolist()
    print(f"  {len(val_df)} rows | {val_df['Question_id'].nunique()} unique questions")

    

    all_metrics = []
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    # load existing metrics if resuming
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            all_metrics = json.load(f)

    for iteration in range(start_iteration, n_iters):
        print(f"\n{'─'*60}")
        print(f"ITERATION {iteration + 1} / {n_iters}")
        print(f"{'─'*60}")

        # ── Step 1: Grade TRAIN set ───────────────────────────────────────────
        print(f"\n[1/3] Grading {len(train_df)} training examples ...")
        # train_save_path = f"results/gradeopt_train_iter{iteration+1}_progress.csv"
        train_save_path = f"results/{args.run_name}/train_iter{iteration+1}.csv"
        
        train_preds, train_feedbacks = grade_all(train_df, grading_notes,
                                                  save_path=train_save_path)

        # ── Step 2: Evaluate on VAL set ───────────────────────────────────────
        print(f"\n[2/3] Evaluating on val set ({len(val_df)} rows) ...")
        # val_save_path = f"results/gradeopt_val_iter{iteration+1}.csv"
        val_save_path   = f"results/{args.run_name}/val_iter{iteration+1}.csv"
        val_preds, val_feedbacks = grade_all(val_df, grading_notes,
                                              save_path=val_save_path)

        metrics = evaluate(y_true_val, val_preds, split_name=f"iter_{iteration+1}_val")
        metrics["iteration"] = iteration + 1
        all_metrics.append(metrics)

        # save metrics after every iteration
        with open(METRICS_PATH, "w") as f:
            json.dump(all_metrics, f, indent=2)

        # save val predictions + generated feedback for BERTScore later
        val_df_out = val_df.copy()
        val_df_out["predicted_label"] = val_preds
        val_df_out["generated_feedback"] = val_feedbacks
        val_df_out.to_csv(val_save_path, index=False)
        print(f"  Saved predictions + feedback → {val_save_path}")

        # skip reflect/refine on final iteration
        if iteration == n_iters - 1:
            print("\nFinal iteration — skipping reflect/refine step.")
            # save final state
            with open(state_path, "w") as f:
                json.dump({
                    "completed_iterations": iteration + 1,
                    "grading_notes": grading_notes
                }, f, indent=2)
            break

        # ── Step 3: Reflect + Refine on TRAIN errors ──────────────────────────
        print("\n[3/3] Reflecting on training errors and refining grading notes ...")
        errors_by_q = collect_errors(train_df, train_preds)
        print(f"  {len(errors_by_q)} question(s) had grading errors.")

        updated = 0
        for qid, errors in errors_by_q.items():
            if qid == "nan" or not qid.strip():
                continue
            
            matches = train_df[train_df["Question_id"].astype(str) == qid]
            if len(matches) == 0:
                continue
            row0 = matches.iloc[0]
            gold_fb = get_gold_feedback(train_df, qid)

            critique = reflector_agent.reflect(
                question=row0["Question"],
                correct_answer=row0["Correct Answer"],
                errors=errors,
                gold_feedback_samples=gold_fb,
            )

            new_notes = refiner_agent.refine(
                question=row0["Question"],
                correct_answer=row0["Correct Answer"],
                critique=critique,
                previous_notes=grading_notes.get(qid, ""),
            )
            grading_notes[qid] = new_notes
            updated += 1
            time.sleep(BATCH_DELAY)

        print(f"  {updated} question(s) had their grading notes updated.")

        # ── Save state after each completed iteration ─────────────────────────
        NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(NOTES_PATH, "w") as f:
            json.dump(grading_notes, f, indent=2)
        print(f"  Grading notes saved → {NOTES_PATH}")

        with open(state_path, "w") as f:
            json.dump({
                "completed_iterations": iteration + 1,
                "grading_notes": grading_notes
            }, f, indent=2)
        print(f"  State saved → {state_path}")

    # ── Final saves ───────────────────────────────────────────────────────────
    with open(NOTES_PATH, "w") as f:
        json.dump(grading_notes, f, indent=2)
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nFinal grading notes → {NOTES_PATH}")
    print(f"All metrics         → {METRICS_PATH}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Iter':<6} {'Accuracy':<12} {'QWK':<10} {'Weighted F1'}")
    for m in all_metrics:
        print(
            f"{m['iteration']:<6} "
            f"{m.get('accuracy', 0):<12.4f} "
            f"{m.get('quadratic_wk', 0):<10.4f} "
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
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Only use first N rows of train for optimization (for testing)"
    )
    parser.add_argument(
    "--use_adv_train",
    action="store_true",
    help="Include adversarial data in training"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="default",
        help="Name for saving results (e.g., gradeopt, adv, etc.)"
    )
    args = parser.parse_args()
    run_pipeline(n_iters=args.iters, sample=args.sample)
