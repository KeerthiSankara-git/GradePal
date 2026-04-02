from google import genai
import pandas as pd
import os
from dotenv import load_dotenv
import time

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

def load_splits(base_path="./data"):
    splits = {}
    
    for split in ["train", "val", "unseen_answers", "unseen_question"]:
        file_path = os.path.join(base_path, f"{split}.csv")
        
        if os.path.exists(file_path):
            splits[split] = pd.read_csv(file_path)
            # print(f"Loaded {split}: {splits[split].shape}")
        else:
            print(f"Warning: {file_path} not found")
    
    return splits


def call_gemini(prompt, client, model="gemini-2.5-flash"):
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        return response.text.strip()
    
    except Exception as e:
        print(f"Flash failed, switching to Pro: {e}")
        
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt
        )
        return response.text.strip()


def extract_keywords_llm(question, correct_answer, client):
    prompt = f"""
    Extract 3-5 key concepts from the correct answer.

    Question: {question}
    Correct Answer: {correct_answer}

    Return ONLY a comma-separated list.
    """
    
    return call_gemini(prompt, client)

def gen_keyword_stuffing(question, keywords, client):
    prompt = f"""
    Question: {question}
    Keywords: {keywords}

    Generate a nonsensical student answer that:
    - Uses ALL the keywords
    - Is incoherent or poorly structured
    - Does NOT correctly answer the question

    Output only the answer.
    """
    
    return call_gemini(prompt, client)


def gen_fluent_wrong(question, correct_answer, client):
    prompt = f"""
    Question: {question}
    Correct Answer: {correct_answer}

    Generate a fluent, confident, well-written answer that is incorrect.
    It should sound convincing but contain wrong reasoning.

    Output only the answer.
    """
    
    return call_gemini(prompt, client)


def gen_off_topic(question, client):
    prompt = f"""
    Question: {question}

    Generate an answer that is completely unrelated to the question.
    It should sound fluent but answer a different topic.

    Output only the answer.
    """
    
    return call_gemini(prompt, client)


def gen_flattery(question, client):
    prompt = f"""
    Question: {question}

    Generate a student answer that:
    - Does not answer the question
    - Appeals emotionally for marks
    - Mentions effort or hard work

    Output only the answer.
    """
    
    return call_gemini(prompt, client)



def generate_adv_dataset(df, client, max_questions=100):
    adv_rows = []
    
    df = df.dropna(subset=["Question", "Correct Answer"])
    
    for idx, row in df.head(max_questions).iterrows():
        question = row["Question"]
        correct = row["Correct Answer"]
        
        print(f"\nProcessing question {idx}")
        
        # Step 1: Extract keywords
        keywords = extract_keywords_llm(question, correct, client)
        
        # Step 2: Generate attacks
        attacks = [
            ("keyword", gen_keyword_stuffing(question, keywords, client), 0),
            ("fluent_wrong", gen_fluent_wrong(question, correct, client), 0),
            ("off_topic", gen_off_topic(question, client), 0),
            ("flattery", gen_flattery(question, client), 0),
        ]
        
        for attack_type, adv_answer, label in attacks:
            # if is_valid_adversarial(adv_answer):
            adv_rows.append({
                "Question": question,
                "Correct Answer": correct,
                "Adversarial Answer": adv_answer,
                "attack_type": attack_type,
                "output_label": label
            })
            print(f"\nAdversarial Type: {attack_type}, Answer: {adv_answer}")
            time.sleep(3)
        print(f"\nQuestion {idx} processed")
    return pd.DataFrame(adv_rows)


splits = load_splits("./data")
adversarial_df = generate_adv_dataset(splits["train"], client, max_questions=2)

adversarial_df.to_csv("data/adversarial/adversarial_dataset.csv", index=False)