"""
debug_gemini.py
Prints the raw Gemini response object so we can see exactly what's coming back.
Run from project root: python debug_gemini.py
"""
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Reply with exactly one word: correct",
    config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=50,
        system_instruction="You are a grader. Reply with one word only."
    )
)

print("=== Full response object ===")
print(response)
print()
print("=== response.text ===")
print(repr(response.text))
print()
print("=== candidates ===")
for i, c in enumerate(response.candidates or []):
    print(f"  candidate[{i}]:")
    print(f"    finish_reason : {c.finish_reason}")
    print(f"    content       : {c.content}")
    if c.content:
        for j, p in enumerate(c.content.parts or []):
            print(f"    part[{j}]       : {repr(p)}")