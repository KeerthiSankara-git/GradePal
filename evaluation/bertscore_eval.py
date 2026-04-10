"""
evaluation/bertscore_eval.py

Evaluates the quality of GradeOpt-generated feedback against
gold EngSAF feedback annotations using BERTScore.

Run:
    python -m evaluation.bertscore_eval
"""

import json
import pandas as pd
from pathlib import Path
from bert_score import score as bert_score


def evaluate_bertscore(
    generated: list[str],
    references: list[str],
    split_name: str
) -> dict:
    """Compute BERTScore between generated and reference feedback."""
    
    print(f"\nComputing BERTScore for {split_name} ({len(generated)} examples)...")
    
    # compute BERTScore
    # lang="en" uses roberta-large by default
    P, R, F1 = bert_score(
        generated,
        references,
        lang="en",
        verbose=True
    )
    
    precision = P.mean().item()
    recall    = R.mean().item()
    f1        = F1.mean().item()
    
    print(f"\n── BERTScore: {split_name} ──")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    
    return {
        "split":     split_name,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "n_examples": len(generated)
    }


def load_feedback_pairs(csv_path: str, 
                        gen_col: str = "generated_feedback",
                        gold_col: str = "feedback") -> tuple[list, list]:
    """
    Load generated and gold feedback from a predictions CSV.
    Filters out empty or missing feedback.
    """
    df = pd.read_csv(csv_path)
    
    # check required columns exist
    if gen_col not in df.columns:
        raise ValueError(f"Column '{gen_col}' not found in {csv_path}")
    if gold_col not in df.columns:
        raise ValueError(f"Column '{gold_col}' not found in {csv_path}")
    
    # drop rows where either feedback is missing
    before = len(df)
    df = df.dropna(subset=[gen_col, gold_col])
    df = df[df[gen_col].str.strip() != ""]
    df = df[df[gold_col].str.strip() != ""]
    dropped = before - len(df)
    
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing feedback")
    
    print(f"  {len(df)} valid feedback pairs")
    
    return df[gen_col].tolist(), df[gold_col].tolist()


def main():
    Path("results").mkdir(exist_ok=True)
    all_metrics = []

    # ── Evaluate on val iterations ────────────────────────────────────────────
    for iteration in [1, 2, 3]:
        csv_path = f"results/gradeopt_val_iter{iteration}.csv"
        if not Path(csv_path).exists():
            print(f"Skipping iter {iteration} — {csv_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"GradeOpt Val — Iteration {iteration}")
        print(f"{'='*60}")

        try:
            generated, references = load_feedback_pairs(csv_path)
            metrics = evaluate_bertscore(
                generated=generated,
                references=references,
                split_name=f"gradeopt_val_iter{iteration}"
            )
            all_metrics.append(metrics)
        except Exception as e:
            print(f"  Error: {e}")

    # ── Evaluate on unseen splits ─────────────────────────────────────────────
    for split_name, csv_path in [
        ("unseen_answers",  "results/gradeopt_unseen_answers.csv"),
        ("unseen_question", "results/gradeopt_unseen_question.csv"),
    ]:
        if not Path(csv_path).exists():
            print(f"Skipping {split_name} — {csv_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"GradeOpt {split_name}")
        print(f"{'='*60}")

        try:
            generated, references = load_feedback_pairs(csv_path)
            metrics = evaluate_bertscore(
                generated=generated,
                references=references,
                split_name=f"gradeopt_{split_name}"
            )
            all_metrics.append(metrics)
        except Exception as e:
            print(f"  Error: {e}")

    # ── Save results ──────────────────────────────────────────────────────────
    output_path = "results/bertscore_metrics.json"
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BERTSCORE SUMMARY")
    print("=" * 60)
    print(f"{'Split':<30} {'Precision':<12} {'Recall':<10} {'F1'}")
    for m in all_metrics:
        print(
            f"{m['split']:<30} "
            f"{m['precision']:<12.4f} "
            f"{m['recall']:<10.4f} "
            f"{m['f1']:.4f}"
        )
    print("=" * 60)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    main()