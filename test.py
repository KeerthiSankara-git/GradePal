import pandas as pd
from config import TRAIN_PATH, LABEL_MAP
from utils.llm import call_llm

# test data loads
train = pd.read_csv(TRAIN_PATH)
print(f"Train loaded: {len(train)} rows")

# test API works
response = call_llm(
    prompt="Say 'API working' and nothing else.",
    system="You are a test assistant."
)
print(f"API response: {response}")