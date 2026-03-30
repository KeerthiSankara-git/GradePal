import os
from dotenv import load_dotenv
load_dotenv()

# API Keys and model names
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL   = "gemini-2.5-flash"

#Paths and constants
DATA_DIR    = "data/"
RUBRICS_DIR = "rubrics/"
RESULTS_DIR = "results/"
ADV_DIR     = "adversarial/"

TRAIN_PATH       = DATA_DIR + "train.csv"
VAL_PATH         = DATA_DIR + "val.csv"
UNSEEN_ANS_PATH  = DATA_DIR + "unseen_answers.csv"
UNSEEN_Q_PATH    = DATA_DIR + "unseen_question.csv"
RUBRICS_PATH     = RUBRICS_DIR + "rubrics.json"
RUBRICS_OPT_PATH = RUBRICS_DIR + "rubrics_optimized.json"

#Grading 
LABEL_MAP = {0: "incorrect", 1: "partially correct", 2: "correct"}
LABEL_MAP_INV = {"incorrect": 0, "partially correct": 1, "correct": 2}

#  Rate limiting 
BATCH_DELAY = 4.0  # seconds between calls on free tier
