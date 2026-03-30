"""
graders/gradeopt/reflector_agent.py

Analyzes grading errors for a question and produces a plain-text critique
that the refiner will use to write better grading guidance.

No rubric JSON — everything is derived from the dataset's Correct Answer
and gold feedback fields.
"""

from utils.llm import call_llm

SYSTEM = "You are an expert grading analyst who identifies patterns in grading mistakes."

REFLECTOR_PROMPT = """A grader is grading student answers for this question:

Question:
{question}

Correct Answer:
{correct_answer}

The grader made the following errors on the validation set:

{error_examples}

For context, here are some gold feedback comments from the dataset showing
how a human expert explained these grades:

{gold_feedback_samples}

Analyze the errors carefully. Write a concise critique (3-6 bullet points) that:
1. Identifies what the grader is getting wrong (over-grading, under-grading, or confusion)
2. Points out specific concepts or phrasing the grader is misjudging
3. Gives concrete advice on what to look for when grading this question

Be specific and actionable. Do not be vague. Write only the bullet points."""


def _format_errors(errors: list[dict]) -> str:
    lines = []
    for i, err in enumerate(errors, 1):
        lines.append(
            f"Error {i}: True={err['true_label']}, Predicted={err['predicted_label']}\n"
            f"  Student Answer: {err['student_answer']}"
        )
    return "\n\n".join(lines)


def _format_feedback_samples(samples: list[str]) -> str:
    if not samples:
        return "No gold feedback available."
    return "\n".join(f"- {fb}" for fb in samples[:5])


def reflect(
    question: str,
    correct_answer: str,
    errors: list[dict],
    gold_feedback_samples: list[str] = None,
) -> str:
    """
    Analyze grading errors and return a plain-text critique string.

    Args:
        question:              The exam question text.
        correct_answer:        The reference answer from the dataset.
        errors:                List of dicts — each has student_answer,
                               true_label (str), predicted_label (str).
        gold_feedback_samples: A few feedback strings from the dataset for
                               this question to ground the critique.

    Returns:
        str: Plain-text bullet-point critique with actionable grading advice.
    """
    if not errors:
        return "No errors found — grading looks good for this question."

    prompt = REFLECTOR_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        error_examples=_format_errors(errors),
        gold_feedback_samples=_format_feedback_samples(gold_feedback_samples or []),
    )

    return call_llm(prompt, system=SYSTEM, temperature=0.3, max_tokens=400)