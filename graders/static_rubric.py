import json
import pandas as pd
import sys
import os
import time  
from dotenv import load_dotenv

# 1. Load Environment
load_dotenv()  

# 2. ABSOLUTE PATH SETUP (Fixes the "file not found" or "no output" issue)
# This finds the 'GradePal' root folder regardless of where you run the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(ROOT_DIR)

# Now we can safely import from config and utils
from utils.llm import call_llm
from utils.metrics import evaluate
from config import (VAL_PATH, UNSEEN_ANS_PATH, UNSEEN_Q_PATH, 
                    BATCH_DELAY, GEMINI_MODEL)

# --- Configuration & Prompts ---
SYSTEM_PROMPT = (
    "You are a precise teaching assistant. Compare the student answer to the reference. "
    "Output ONLY the label: 'correct', 'partially correct', or 'incorrect'."
)

PROMPT_TEMPLATE = """Task: Grade the student answer based on the reference.

Example:
Q: What is photosynthesis? 
Ref: Process where plants use sunlight to make food.
Student: Plants making food from light.
Grade: correct

Current Task:
Question: {question}
Reference Answer: {rubric}
Student Answer: {student_answer}

Grade:"""

def parse_label(text: str) -> int:
    text = text.strip().lower().replace(".", "").replace(",", "")
    if "partially correct" in text or text.startswith("part"):
        return 1
    if "incorrect" in text:
        return 0
    if "correct" in text:
        return 2
    return 0


def grade(question: str, correct_answer: str, student_answer: str) -> int:
    formatted_prompt = PROMPT_TEMPLATE.format(
        question=question,
        rubric=correct_answer,
        student_answer=student_answer
    )

    try:
        raw_response = call_llm(
            prompt=formatted_prompt,
            system=SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=32,
            model=GEMINI_MODEL,
            disable_thinking=True
        )
        return parse_label(raw_response)
    except Exception as e:
        if "429" in str(e):
            raise e
        return 0


def main():
    # Define datasets using paths from config
    datasets = [
        {"name": "val", "path": os.path.join(ROOT_DIR, VAL_PATH)},
        {"name": "unseen_answers", "path": os.path.join(ROOT_DIR, UNSEEN_ANS_PATH)},
        {"name": "unseen_questions", "path": os.path.join(ROOT_DIR, UNSEEN_Q_PATH)}
    ]

    # Ensure results directory exists in the root
    results_dir = os.path.join(ROOT_DIR, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory: {results_dir}")

    for dataset in datasets:
        # Check if source data exists
        if not os.path.exists(dataset['path']):
            print(f"\n[!] Skipping {dataset['name']}: File not found at {dataset['path']}")
            continue

        print(f"\n--- NOW PROCESSING: {dataset['name'].upper()} ---")
        save_path = os.path.join(results_dir, f"system2_{dataset['name']}_predictions.json")

        df = pd.read_csv(dataset['path'])
        
        # Resume Logic
        results_list = []
        if os.path.exists(save_path):
            try:
                with open(save_path, 'r') as f:
                    results_list = json.load(f)
                start_index = len(results_list)
            except:
                start_index = 0
            
            if start_index >= len(df):
                print(f"Dataset '{dataset['name']}' already finished.")
                continue
            print(f"Resuming from row {start_index}...")
        else:
            start_index = 0

        # Grading Loop
        for index in range(start_index, len(df)):
            row = df.iloc[index]
            # formatted_prompt = PROMPT_TEMPLATE.format(
            #     question=row.get('Question', ''),
            #     rubric=row.get('Correct Answer', ''),  
            #     student_answer=row.get('Student Answer', '')
            # )
            
            # try:
            #     raw_response = call_llm(
            #         prompt=formatted_prompt, 
            #         system=SYSTEM_PROMPT, 
            #         temperature=0.0, 
            #         max_tokens=32, 
            #         model=GEMINI_MODEL,
            #         disable_thinking=True 
            #     )
            #     pred_int = parse_label(raw_response)
            # except Exception as e:
            #     if "429" in str(e):
            #         print("\n[!] Rate limit hit. Exiting.")
            #         return 
            #     pred_int = 0
            # 
            question=row.get('Question', ''),
            correct_answer=row.get('Correct Answer', ''),  
            student_answer=row.get('Student Answer', '') 
            pred_int = grade(question, correct_answer, student_answer)
            
            # Save progress
            result_entry = row.to_dict()
            result_entry['static_rubric_prediction'] = pred_int
            results_list.append(result_entry)
            
            with open(save_path, 'w') as f:
                json.dump(results_list, f, indent=4)
            
            if (index + 1) % 5 == 0:
                print(f"[{dataset['name']}] Progress: {index + 1}/{len(df)}")
            
            time.sleep(BATCH_DELAY)

        # Evaluation
        print(f"\nFinished {dataset['name']}!")
        y_true = [row.get('output_label', 0) for row in results_list]
        y_pred = [row.get('static_rubric_prediction', 0) for row in results_list]
        evaluate(y_true=y_true, y_pred=y_pred, split_name=f"Static {dataset['name']}")

if __name__ == "__main__":
    main()