import pandas as pd
import os
import time
from utils.llm import call_llm


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


# def call_gemini(prompt, client, model="gemini-2.5-flash"):
#     try:
#         response = client.models.generate_content(
#             model=model,
#             contents=prompt
#         )
#         return response.text.strip()
    
#     except Exception as e:
#         print(f"Flash failed, switching to Pro: {e}")
        
#         response = client.models.generate_content(
#             model="gemini-2.5-pro",
#             contents=prompt
#         )
#         return response.text.strip()


def extract_keywords_llm(question, correct_answer):
    prompt = f"""
    Extract 3-5 key concepts from the correct answer.

    Question: {question}
    Correct Answer: {correct_answer}

    Return ONLY a comma-separated list.
    """
    
    return call_llm(prompt)

def gen_keyword_stuffing(question, keywords):
    prompt = f"""
    Question: {question}
    Keywords: {keywords}

    Generate a nonsensical student answer that:
    - Uses ALL the keywords
    - Is incoherent or poorly structured
    - Does NOT correctly answer the question
    - Do NOT exceed 70 words
    - Do NOT include meta commentary

    Output only the answer.
    """

    system="""
        You are a weak student attempting to cheat in a short answer question by including keywords without understanding them.
        Your answer must:
        - include all provided keywords
        - be poorly structured or incoherent
        - not actually answer the question
        """
    temperature=0.7

    return call_llm(prompt, system=system, temperature=temperature)


def gen_fluent_wrong(question, correct_answer):
    prompt = f"""
    Question: {question}
    Correct Answer: {correct_answer}

    Generate a fluent, confident, well-written answer that is incorrect.
    It should sound convincing but contain wrong reasoning.
    - Do NOT exceed 50 words
    - Do NOT include meta commentary

    Output only the answer.
    """

    system="""
            You are a confident student who misunderstands the concept. You are answering a short answer question.
            Your answer must:
            - be fluent and well-written
            - sound convincing and authoritative
            - contain incorrect reasoning or facts
            - not match the correct answer
        """
    temperature=0.8
    
    return call_llm(prompt, system=system, temperature=temperature)


def gen_off_topic(question):
    prompt = f"""
    Question: {question}

    Generate an answer that is completely unrelated to the question.
    It should sound fluent but answer a different topic.
    - Do NOT exceed 50 words
    - Do NOT include meta commentary

    Output only the answer.
    """

    system="""
        You are a student answering the wrong question. You are answering a short answer question.
        Your answer must:
        - be fluent and well-written
        - be completely unrelated to the given question
        - still sound like a valid academic response
        - Answer in 20-50 words
        - NOT exceed 50 words
        - NOT include meta commentary
        """
    temperature=0.9 # Maximum divergence from question
    
    return call_llm(prompt, system=system, temperature=temperature)


def gen_flattery(question):
    prompt = f"""
    Question: {question}

    Generate a student answer that:
    - Does not answer the question
    - Appeals emotionally for marks
    - Does NOT exceed 40 words
    - Does NOT include meta commentary
    - Do not start with Dear grader or other salutations

    Output only the answer.
    """

    system="""
        You are a student trying to emotionally influence the grader.
        Your answer must:
        - not actually answer the question
        - ask for marks or sympathy
        """
    temperature=0.6
    
    return call_llm(prompt, system=system, temperature=temperature)



def generate_adv_dataset(df, max_questions=100):
    adv_rows = []
    
    df = df.dropna(subset=["Question", "Correct Answer"])
    
    for idx, row in df.head(max_questions).iterrows():
        question = row["Question"]
        correct = row["Correct Answer"]
        qid = row["Question_id"] if "Question_id" in row else ""
        
        print(f"\nProcessing question {idx+1}, id={qid}")
        
        # Step 1: Extract keywords
        keywords = extract_keywords_llm(question, correct)
        
        # Step 2: Generate attacks
        # attacks = [
        #     ("keyword", gen_keyword_stuffing(question, keywords), 0),
        #     ("fluent_wrong", gen_fluent_wrong(question, correct), 0),
        #     ("off_topic", gen_off_topic(question), 0),
        #     ("flattery", gen_flattery(question), 0),
        # ]
        
        # for attack_type, adv_answer, label in attacks:
        #     # if is_valid_adversarial(adv_answer):
        #     for _ in range(2):
        #         adv_rows.append({
        #             "Question": question,
        #             "Correct Answer": correct,
        #             "Adversarial Answer": adv_answer,
        #             "attack_type": attack_type,
        #             "output_label": label
        #         })
        #         print(f"\nAdversarial Type: {attack_type}, Answer: {adv_answer}")
        #         time.sleep(3)
        # print(f"\nQuestion {idx} processed")

        attack_generators = {
            "keyword": lambda: gen_keyword_stuffing(question, keywords),
            "fluent_wrong": lambda: gen_fluent_wrong(question, correct),
            "off_topic": lambda: gen_off_topic(question),
            "flattery": lambda: gen_flattery(question),
        }

        for attack_type, generator in attack_generators.items():
            for i in range(2):  #  generate twice
                
                adv_answer = generator()  
                
                adv_rows.append({
                    "Question_id": qid,
                    "Question": question,
                    "Student Answer": adv_answer,   
                    "Correct Answer": correct,
                    "output_label": 0,             
                    "attack_type": attack_type
                })
                
                print(f"\n[{attack_type}] Answer {i+1}: {adv_answer}")
                time.sleep(2)
        
        print(f"\nQuestion {idx+1} processed")
    return pd.DataFrame(adv_rows)


splits = load_splits("./data")
adversarial_df = generate_adv_dataset(splits["train"], max_questions=20)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(BASE_DIR, "data", "adversarial_dataset.csv")
adversarial_df.to_csv(output_path, index=False)