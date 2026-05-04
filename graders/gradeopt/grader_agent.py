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
import json
from config import GEMINI_MODEL

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
Grade the student's answer and provide brief feedback.

Critical grading rules — apply strictly:
1. Keyword presence alone is NOT sufficient for credit.
   The answer must demonstrate logical understanding 
   and coherent explanation of the concept.
2. If the answer is a disorganized list of terms or 
   keywords without meaningful explanation, mark as incorrect.
3. If the answer is grammatically fluent but factually 
   wrong or contradicts the correct answer, mark as incorrect.
4. Only award partial or full credit if the student 
   demonstrates genuine understanding through coherent reasoning.

Return ONLY a JSON object in this exact format, nothing else:
{{
  "label": "incorrect" | "partially correct" | "correct",
  "feedback": "1-2 sentence explanation of what the student got right and what they missed"
}}

Rules:
- correct = student fully understood and answered the question
- partially correct = student showed some understanding but missed key points
- incorrect = student misunderstood or did not address the question
- feedback must be specific to the answer, not generic
- feedback must reference what the correct answer expects
"""


def _build_notes_block(grading_notes: str) -> str:
    notes = (grading_notes or "").strip()
    if not notes:
        return ""
    return f"Additional grading guidance:\n{notes}\n"


def _parse_response(raw: str) -> dict:
    """Parse JSON response containing label and feedback."""
    #print(f"\n  [DEBUG] raw response: {repr(raw)}")  # to see if LLM is returning what we expect before parsing
    if raw is None or not raw.strip():
        raise RuntimeError("LLM returned empty response.")

    # strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # fallback — try to extract label at minimum
        label = _parse_label_fallback(text)
        return {"label": label, "feedback": ""}

    # parse label
    label_str = parsed.get("label", "").strip().lower()

    if label_str.startswith("part"):
        label = LABEL_MAP_INV["partially correct"]
    elif re.search(r"\bincorrect\b", label_str):
        label = LABEL_MAP_INV["incorrect"]
    elif re.search(r"\bcorrect\b", label_str):
        label = LABEL_MAP_INV["correct"]
    else:
        label = 0  # default to incorrect

    feedback = parsed.get("feedback", "").strip()

    return {"label": label, "feedback": feedback}

def _parse_label_fallback(text: str) -> int:
    """Fallback label parser if JSON parsing fails."""
    text = text.lower()
    if text.startswith("part") or "partially correct" in text:
        return LABEL_MAP_INV["partially correct"]
    if re.search(r"\bincorrect\b", text):
        return LABEL_MAP_INV["incorrect"]
    if re.search(r"\bcorrect\b", text):
        return LABEL_MAP_INV["correct"]
    return 0

def grade(
    question: str,
    student_answer: str,
    correct_answer: str,
    grading_notes: str = "",
) -> int:
    """
    Grade a single student answer.
    Returns int label only — for backward compatibility with pipeline.py
    """
    result = grade_with_feedback(
        question=question,
        student_answer=student_answer,
        correct_answer=correct_answer,
        grading_notes=grading_notes,
    )
    return result["label"]


def grade_with_feedback(
    question: str,
    student_answer: str,
    correct_answer: str,
    grading_notes: str = "",
) -> dict:
    """
    Grade a single student answer and generate feedback.
    
    Returns:
        dict: {
            'label': int (0/1/2),
            'feedback': str
        }
    """

    # handle NaN values from pandas
    question       = str(question)       if question       and str(question)       != "nan" else ""
    student_answer = str(student_answer) if student_answer and str(student_answer) != "nan" else ""
    correct_answer = str(correct_answer) if correct_answer and str(correct_answer) != "nan" else ""
    
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
        max_tokens=2048,     # more tokens needed for JSON + feedback
        retries=3,
        expect_json=True,   # strip markdown fences automatically
        disable_thinking=False,
        model=GEMINI_MODEL
    )

    return _parse_response(raw)
