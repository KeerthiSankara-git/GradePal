import pandas as pd
import sys
import os
import time  
from dotenv import load_dotenv

# 1. Load the API key
load_dotenv()  

# Ensure Python can find the config and utils folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm import call_llm
from utils.metrics import evaluate
from config import (VAL_PATH, LABEL_MAP_INV, BATCH_DELAY, GEMINI_MODEL)

# Strict instructions
SYSTEM_PROMPT = "You are an expert teaching assistant. Your task is to grade a student's answer based on a specific rubric. You must output the score label ONLY. Provide no explanations."

PROMPT_TEMPLATE = """Evaluate the student's answer. You must respond with EXACTLY ONE of the following score labels:
correct
partially correct
incorrect

Question: {question}
Reference Answer: {rubric}
Student Answer: {student_answer}

Grade:"""

def main():
    # Define paths
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "system2_static_predictions.csv")

    # Load the original data
    df = pd.read_csv(VAL_PATH)
    
    # CHECK FOR RESUME: If the file exists, load it to see where we left off
    if os.path.exists(save_path):
        processed_df = pd.read_csv(save_path)
        start_index = len(processed_df)
        print(f"Resuming from row {start_index}...")
        # We use the existing predictions and just append to them
        predicted_labels = processed_df['static_rubric_prediction'].tolist()
    else:
        start_index = 0
        predicted_labels = []

    print(f"Starting/Resuming grading loop using {GEMINI_MODEL}...")
    
    # 2. Loop only through the REMAINING rows
    for index in range(start_index, len(df)):
        row = df.iloc[index]
        
        formatted_prompt = PROMPT_TEMPLATE.format(
            question=row['Question'],
            rubric=row['Correct Answer'],  
            student_answer=row['Student Answer']
        )
        
        try:
            raw_response = call_llm(
                prompt=formatted_prompt, 
                system=SYSTEM_PROMPT, 
                temperature=0.0, 
                max_tokens=32, 
                model=GEMINI_MODEL  
            )
            clean_text = raw_response.strip().lower()
        except Exception as e:
            # If we hit a 429 error, we stop here so we don't save a bunch of errors
            if "429" in str(e):
                print(f"\n[!] Daily Quota Exhausted at row {index}. Stopping for today.")
                break
            print(f"  [!] Error on row {index}: {e}")
            clean_text = "error" 
        
        # Convert label to score
        pred_int = LABEL_MAP_INV.get(clean_text, 0)
        predicted_labels.append(pred_int)
        
        # 3. SAVE PROGRESS IMMEDIATELY
        # We create a temporary copy of the dataframe up to this point and save it
        current_progress_df = df.iloc[:len(predicted_labels)].copy()
        current_progress_df['static_rubric_prediction'] = predicted_labels
        current_progress_df.to_csv(save_path, index=False)
        
        print(f"Graded {index + 1}/{len(df)}...")
        time.sleep(BATCH_DELAY) 

    # 4. Final Evaluation (only runs if the whole file is finished)
    if len(predicted_labels) == len(df):
        print("\nAll rows finished! Calculating final metrics...")
        true_labels = df['output_label'].tolist()
        evaluate(y_true=true_labels, y_pred=predicted_labels, split_name="Static Rubric (System 2)")
    else:
        print(f"\nProgress saved to {save_path}. Run this again tomorrow to finish the remaining rows.")

if __name__ == "__main__":
    main()