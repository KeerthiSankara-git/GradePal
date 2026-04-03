"""
GradePal — System 1: Baseline Grader (No Rubric)
=================================================
Grades student short answers using only the question and the student's
answer. No rubric is provided — the LLM acts as the sole judge.

Output: one of {"incorrect", "partially correct", "correct"}
        mapped to {0, 1, 2} for metric computation.

Usage
-----
    python -m graders.baseline                         # all splits (default)
    python -m graders.baseline --split val             # val split only
    python -m graders.baseline --split unseen_answers  # unseen answers split
"""

import argparse
import sys
from pathlib import Path

# Make sure the project root (parent of this file's directory) is on sys.path
# so that 'config' and 'utils' are importable regardless of where you run from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    TRAIN_PATH,
    VAL_PATH,
    UNSEEN_ANS_PATH,
    UNSEEN_Q_PATH,
    LABEL_MAP,
    LABEL_MAP_INV,
    BATCH_DELAY,
)
from utils.llm import call_llm, batch_call
from utils.metrics import evaluate


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert teacher grading a short-answer exam question.
You will be given the exam question and the student's answer.

Classify the student's answer as exactly one of:
  incorrect         — The answer is wrong, irrelevant, or missing.
  partially correct — The answer shows some understanding but is incomplete or
                      contains a significant error.
  correct           — The answer is accurate and sufficiently complete.

Be strict:
- Do NOT award credit for keyword stuffing or answers that list terms without
  demonstrating real understanding.
- Do NOT award credit for fluent-sounding but factually wrong responses.
- Base your judgment solely on correctness and completeness.

Respond with ONLY one of these three labels (no punctuation, no explanation):
incorrect
partially correct
correct

IMPORTANT: Output ONLY the label word(s). No explanation. No punctuation. No other text.\
"""

USER_TEMPLATE = """\
Question: {question}

Student Answer: {student_answer}\
"""


# ---------------------------------------------------------------------------
# Core grading logic
# ---------------------------------------------------------------------------

def build_prompt(question: str, student_answer: str) -> str:
    """Format the user-side prompt for a single example."""
    return USER_TEMPLATE.format(
        question=question,
        student_answer=str(student_answer).strip() if str(student_answer).strip() not in ("", "nan") else "(no answer provided)",
    )


def parse_label(raw: str) -> int:
    """
    Convert raw LLM output to an integer label.

    Tries an exact match first, then a substring match as a fallback.
    Returns 0 (incorrect) if nothing matches.
    """
    cleaned = raw.strip().lower()

    # Exact match
    if cleaned in LABEL_MAP_INV:
        return LABEL_MAP_INV[cleaned]

    # Substring fallback (handles extra whitespace or trailing punctuation)
    for label_str, label_int in LABEL_MAP_INV.items():
        if label_str in cleaned:
            return label_int

    # Default to incorrect if unparseable
    print(f"  [WARN] Could not parse label from: {raw!r} — defaulting to 0")
    return 0


def grade_single(question: str, student_answer: str) -> int:
    """
    Grade one student answer.

    Parameters
    ----------
    question : str
    student_answer : str

    Returns
    -------
    int  — 0 (incorrect) | 1 (partially correct) | 2 (correct)
    """
    prompt = build_prompt(question, student_answer)
    raw = call_llm(
        prompt=prompt,
        system=SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=32,
    )
    return parse_label(raw)


def grade_dataframe(df: pd.DataFrame) -> list[int]:
    """
    Grade all rows in a DataFrame using batch_call for rate-limit safety.

    Expected columns: 'Question', 'Student Answer'

    Returns
    -------
    list[int]  — predicted labels, parallel to df rows
    """
    prompts = [
        {
            "prompt": build_prompt(row["Question"], row["Student Answer"]),
            "system": SYSTEM_PROMPT,
            "temperature": 0.0,
            "max_tokens": 32,
        }
        for _, row in df.iterrows()
    ]

    raw_outputs = batch_call(prompts, delay=BATCH_DELAY)
    return [parse_label(raw) for raw in raw_outputs]


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------

def run_evaluation(split: str = "val") -> dict:
    """
    Load a data split, grade every example, compute and print metrics.

    Parameters
    ----------
    split : str — "train" | "val" | "unseen_answers" | "unseen_question"

    Returns
    -------
    dict — metrics returned by evaluate()
    """
    paths = {
        "train":            TRAIN_PATH,
        "val":              VAL_PATH,
        "unseen_answers":   UNSEEN_ANS_PATH,
        "unseen_question":  UNSEEN_Q_PATH,
    }
    path = paths[split]
    df = pd.read_csv(path)

    # Drop rows with missing question or answer
    before = len(df)
    df = df.dropna(subset=["Question", "Student Answer"])
    if before - len(df) > 0:
        print(f"  Dropped {before - len(df)} rows with missing values")

    print(f"\n{'='*55}")
    print(f"  Baseline Grader — {split.upper()} split  ({len(df)} examples)")
    print(f"{'='*55}\n")

    y_pred = grade_dataframe(df)
    y_true = df["output_label"].tolist()

    metrics = evaluate(y_true, y_pred, split_name=f"Baseline ({split})")

    # Save predictions to results/
    Path("results").mkdir(exist_ok=True)
    df_out = df.copy()
    df_out["predicted_label"] = y_pred
    df_out.to_csv(f"results/baseline_{split}.csv", index=False)
    print(f"  Saved → results/baseline_{split}.csv")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description="Run the Baseline ASAG grader.")
    parser.add_argument(
        "--split",
        choices=["train", "val", "unseen_answers", "unseen_question", "all"],
        default="all",
        help="Which data split to evaluate (default: all)",
    )
    args = parser.parse_args()

    if args.split == "all":
        all_metrics = []
        for split in ["val", "unseen_answers", "unseen_question"]:
            metrics = run_evaluation(split=split)
            metrics["split"] = split
            all_metrics.append(metrics)

        Path("results").mkdir(exist_ok=True)
        with open("results/baseline_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)

        print("\n" + "=" * 55)
        print("BASELINE RESULTS SUMMARY")
        print("=" * 55)
        print(f"{'Split':<20} {'Accuracy':<12} {'QWK':<10} {'Weighted F1'}")
        for m in all_metrics:
            print(
                f"{m['split']:<20} "
                f"{m.get('accuracy', 0):<12.4f} "
                f"{m.get('quadratic_wk', 0):<10.4f} "
                f"{m.get('weighted_f1', 0):.4f}"
            )
    else:
        run_evaluation(split=args.split)