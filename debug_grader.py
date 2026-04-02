from graders.gradeopt.grader_agent import grade_with_feedback

result = grade_with_feedback(
    question="What is photosynthesis?",
    student_answer="Photosynthesis is the process by which plants use sunlight, water, and CO2 to produce glucose and oxygen.",
    correct_answer="Photosynthesis is the process plants use to convert light energy into chemical energy (glucose), using CO2 and water and releasing oxygen.",
    grading_notes="",
)

print(f"Full result: {result}")
print(f"Label: {result['label']}")
print(f"Feedback: '{result['feedback']}'")
print(f"Feedback length: {len(result['feedback'])}")