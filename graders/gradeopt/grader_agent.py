"""
graders/gradeopt/grader_agent.py

Grades a student answer using the question's Correct Answer directly from
the dataset.

Returns:
    0 = incorrect
    1 = partially correct
    2 = correct
"""

from __future__ import annotations

import re

from utils.llm import call_llm

LABEL_MAP_INV = {
    "incorrect": 0,
    "partially correct": 1,
    "correct": 2,
}

SYSTEM = "You are a strict but fair short-answer exam grader."

GRADER_PROMPT = """Question:
{question}

Correct Answer:
{correct_answer}

Student Answer:
{student_answer}

{notes_block}
Grade the student's answer against the correct answer.

{notes_block}
Output ONLY one of these three options with no other text:
incorrect
partially correct
correct
"""


def _build_notes_block(grading_notes: str) -> str:
    notes = (grading_notes or "").strip()
    if not notes:
        return ""
    return f"Additional grading guidance:\n{notes}\n"


def _parse_label(raw: str) -> int:
    if raw is None:
        raise RuntimeError("LLM returned None for the grader response.")

    text = raw.strip().lower()
    if not text:
        raise RuntimeError("LLM returned an empty grader response.")

    # Remove markdown fences if present.
    text = re.sub(r"^```(?:text|json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip().lower()

    # Accept partial / truncated versions too.
    if text.startswith("part"):
        return LABEL_MAP_INV["partially correct"]

    if "partially correct" in text:
        return LABEL_MAP_INV["partially correct"]

    # Check incorrect before correct because "incorrect" contains "correct".
    if re.search(r"\bincorrect\b", text):
        return LABEL_MAP_INV["incorrect"]

    if re.search(r"\bcorrect\b", text):
        return LABEL_MAP_INV["correct"]

    # Fallback for numeric outputs.
    if re.search(r"\b2\b", text):
        return 2
    if re.search(r"\b1\b", text):
        return 1
    if re.search(r"\b0\b", text):
        return 0

    raise RuntimeError(f"Could not parse grader label from response: {raw!r}")


def grade(
    question: str,
    student_answer: str,
    correct_answer: str,
    grading_notes: str = "",
) -> int:
    """
    Grade a single student answer.

    Args:
        question: The exam question text.
        student_answer: The student's response.
        correct_answer: The reference answer from the dataset.
        grading_notes: Optional refined criteria built up by the GradeOpt loop.

    Returns:
        int: 0 (incorrect), 1 (partially correct), or 2 (correct)
    """
    prompt = GRADER_PROMPT.format(
        question=question.strip(),
        correct_answer=correct_answer.strip(),
        student_answer=student_answer.strip(),
        notes_block=_build_notes_block(grading_notes),
    )

    raw = call_llm(
        prompt,
        system=SYSTEM,
        temperature=0.0,
        max_tokens=128,
        retries=3,
        expect_json=False,
    )

    return _parse_label(raw)