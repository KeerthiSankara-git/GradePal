"""
graders/gradeopt/refiner_agent.py

Takes the reflector's plain-text critique and produces updated grading_notes —
a concise set of bullet-point instructions that the grader will use on the
next iteration alongside the Correct Answer.

No rubric JSON involved. The "rubric" is just a growing string of grading
guidance that gets better each iteration.
"""

from utils.llm import call_llm

SYSTEM = "You are an expert rubric writer for short-answer exam grading."

REFINER_PROMPT = """Question:
{question}

Correct Answer:
{correct_answer}

A grading analyst reviewed errors and produced this critique:
{critique}

Previous grading notes (empty on first iteration):
{previous_notes}

Using the critique above, write an updated set of grading notes for this question.
These notes will be shown to the grader alongside the question and correct answer.

Your notes must:
- Be 3-6 bullet points, one sentence each
- Clearly state what counts as correct, partially correct, and incorrect
- Call out specific concepts, keywords, or phrasing to watch for
- Fix the errors described in the critique
- Be direct and grader-facing (e.g. "Award full credit if...", "Mark as partial if...", "Deduct for...")

Write ONLY the bullet points. No preamble, no explanation."""


def refine(
    question: str,
    correct_answer: str,
    critique: str,
    previous_notes: str = "",
) -> str:
    """
    Produce updated grading notes from the reflector's critique.

    Args:
        question:        The exam question text.
        correct_answer:  The reference answer from the dataset.
        critique:        Plain-text critique string from reflector_agent.reflect().
        previous_notes:  The grading_notes string from the previous iteration
                         (empty string on iteration 0).

    Returns:
        str: Updated grading notes to pass to grader_agent.grade() next iteration.
    """
    prompt = REFINER_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        critique=critique,
        previous_notes=previous_notes if previous_notes else "None yet.",
    )

    return call_llm(prompt, system=SYSTEM, temperature=0.3, max_tokens=300)
