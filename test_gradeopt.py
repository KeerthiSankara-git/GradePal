"""
test_gradeopt.py

Smoke test for the GradeOpt pipeline.
Tests each agent individually on 1 hardcoded example, then runs the full
pipeline loop on a 5-row slice of val.csv.

Run from the project root:
    python test_gradeopt.py

Expected output:
    [1/4] grader_agent     ... PASS  (predicted: 2)
    [2/4] reflector_agent  ... PASS
    [3/4] refiner_agent    ... PASS
    [4/4] pipeline (5 rows)... PASS
    All checks passed!
"""

import sys
import pandas as pd
from config import VAL_PATH, LABEL_MAP, BATCH_DELAY
import time

# ── Hardcoded test row (matches full EngSAF schema: all 6 columns) ───────────
TEST_ROW = {
    "Question_id": 99.0,
    "Question": "What is photosynthesis?",
    "Student Answer": "Photosynthesis is the process by which plants use sunlight, water, and CO2 to produce glucose and oxygen.",
    "Correct Answer": "Photosynthesis is the process plants use to convert light energy into chemical energy (glucose), using CO2 and water and releasing oxygen.",
    "output_label": 2,   # correct
    "feedback": "The student correctly identified all inputs (sunlight, water, CO2) and outputs (glucose, oxygen).",
}

# Error row — same question, weaker answer, label = partially correct
ERROR_ROW = {
    "Question_id": 99.0,
    "Question": "What is photosynthesis?",
    "Student Answer": "Plants make food from sunlight.",
    "Correct Answer": TEST_ROW["Correct Answer"],
    "output_label": 1,   # partially correct
    "feedback": "The student identified sunlight as an input but missed CO2, water, and the glucose/oxygen outputs.",
}

# ── 1. grader_agent ──────────────────────────────────────────────────────────
print("[1/4] grader_agent (label + feedback)", end="", flush=True)
try:
    from graders.gradeopt.grader_agent import grade, grade_with_feedback

    # test grade_with_feedback (new combined function)
    result = grade_with_feedback(
        question=TEST_ROW["Question"],
        student_answer=TEST_ROW["Student Answer"],
        correct_answer=TEST_ROW["Correct Answer"],
        grading_notes="",
    )
    assert isinstance(result, dict), "Expected dict response"
    assert result["label"] in (0, 1, 2), f"Expected 0/1/2, got {result['label']}"
    assert isinstance(result["feedback"], str) and len(result["feedback"]) > 10, "Feedback too short"

    pred = result["label"]
    print(f"... PASS")
    print(f"  Label:    {pred} = '{LABEL_MAP[pred]}' (true = '{LABEL_MAP[TEST_ROW['output_label']]}')")
    print(f"  Feedback: {result['feedback']}")
    print(f"  Gold:     {TEST_ROW['feedback']}")

    # also verify backward-compatible grade() still returns int
    pred_int = grade(
        question=TEST_ROW["Question"],
        student_answer=TEST_ROW["Student Answer"],
        correct_answer=TEST_ROW["Correct Answer"],
        grading_notes="",
    )
    assert isinstance(pred_int, int), "grade() should return int"

except Exception as e:
    print(f"... FAIL\n  {e}")
    sys.exit(1)

# ── 2. reflector_agent ───────────────────────────────────────────────────────
print("[2/4] reflector_agent  ", end="", flush=True)
try:
    from graders.gradeopt.reflector_agent import reflect

    critique = reflect(
        question=ERROR_ROW["Question"],
        correct_answer=ERROR_ROW["Correct Answer"],
        errors=[
            {
                "student_answer": ERROR_ROW["Student Answer"],
                "true_label": LABEL_MAP[ERROR_ROW["output_label"]],
                "predicted_label": LABEL_MAP[0],  # simulating grader predicted incorrect
            }
        ],
        gold_feedback_samples=[ERROR_ROW["feedback"]],
    )
    assert isinstance(critique, str) and len(critique) > 20, "Critique too short or wrong type"
    print(f"... PASS  (critique length: {len(critique)} chars)")
except Exception as e:
    print(f"... FAIL\n  {e}")
    sys.exit(1)

# ── 3. refiner_agent ─────────────────────────────────────────────────────────
print("[3/4] refiner_agent    ", end="", flush=True)
try:
    from graders.gradeopt.refiner_agent import refine

    notes = refine(
        question=ERROR_ROW["Question"],
        correct_answer=ERROR_ROW["Correct Answer"],
        critique=critique,
        previous_notes="",
    )
    assert isinstance(notes, str) and len(notes) > 20, "Notes too short or wrong type"
    print(f"... PASS  (notes length: {len(notes)} chars)")
    print(f"\n  Sample grading notes generated:\n  {notes[:200]}...\n")
except Exception as e:
    print(f"... FAIL\n  {e}")
    sys.exit(1)

# ── 4. Full pipeline on 5 rows ────────────────────────────────────────────────
print("[4/4] pipeline (5 rows)", end="", flush=True)
try:
    df = pd.read_csv(VAL_PATH)
    sample = df.head(5)

    from graders.gradeopt.grader_agent import grade_with_feedback
    from config import LABEL_MAP, BATCH_DELAY

    preds = []
    feedbacks = []

    for i, (_, row) in enumerate(sample.iterrows()):
        result = grade_with_feedback(
            question=row["Question"],
            student_answer=row["Student Answer"],
            correct_answer=row["Correct Answer"],
            grading_notes="",
        )
        preds.append(result["label"])
        feedbacks.append(result["feedback"])

        if i < len(sample) - 1:
            time.sleep(BATCH_DELAY)  # use config delay, not hardcoded 13s

    assert len(preds) == 5
    assert all(p in (0, 1, 2) for p in preds)
    assert all(isinstance(f, str) for f in feedbacks)

    y_true = sample["output_label"].tolist()
    correct = sum(p == t for p, t in zip(preds, y_true))

    print(f"... PASS")
    print(f"\n  Results on 5 val rows:")
    print(f"  {'Question_id':<14} {'True':<18} {'Predicted':<20} {'Feedback snippet'}")
    for i, (_, row) in enumerate(sample.iterrows()):
        true_str = LABEL_MAP[int(row["output_label"])]
        pred_str = LABEL_MAP[preds[i]]
        match = "✓" if preds[i] == int(row["output_label"]) else "✗"
        fb_snippet = feedbacks[i][:60] + "..." if len(feedbacks[i]) > 60 else feedbacks[i]
        print(f"  {str(row['Question_id']):<14} {true_str:<18} {pred_str:<20} {match} {fb_snippet}")
    print(f"\n  Exact match: {correct}/5")

except Exception as e:
    print(f"... FAIL\n  {e}")
    sys.exit(1)

print("\n" + "=" * 45)
print("All checks passed! Pipeline is working.")
print("=" * 45)
print("\nNext step — run the full pipeline:")
print("  python graders/gradeopt/pipeline.py --iters 1")