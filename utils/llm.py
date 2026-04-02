from google import genai
from google.genai import types
import json
import time
import os

# Configure once at import time
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Model options
FLASH = "gemini-2.5-flash"
# PRO   = "gemini-2.5-pro"  # update when pro is available on free tier

def call_llm(
    prompt: str,
    system: str = None,
    model: str = FLASH,
    temperature: float = 0.0,
    max_tokens: int = 512,
    retries: int = 3,
    expect_json: bool = False
) -> str:
    """
    Core LLM call. Returns response text.
    All graders/agents should use this — never call genai directly.
    """
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system if system else "You are a helpful assistant."
    )

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )
            text = response.text.strip()

            if expect_json:
                text = _strip_json_fences(text)

            return text

        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  [llm] attempt {attempt+1} failed: {e} — retrying in {wait}s")
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM call failed after {retries} attempts: {e}")


def call_llm_json(prompt: str, system: str = None, **kwargs) -> dict:
    """
    Convenience wrapper — calls LLM and parses JSON response.
    Use for Reflector and Refiner agents.
    """
    kwargs["expect_json"] = True
    kwargs["temperature"] = kwargs.get("temperature", 0.0)
    text = call_llm(prompt, system=system, **kwargs)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON:\n{text}\n\nError: {e}")


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences that Gemini sometimes adds."""
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(lines).strip()
    return text


def batch_call(prompts: list[dict], delay: float = 1.5) -> list[str]:
    """
    Run a list of LLM calls with a delay between each.
    Set delay=4.0 on free tier to avoid rate limits.
    """
    results = []
    delay = 0.5
    for i, kwargs in enumerate(prompts):
        print(f"  [batch] {i+1}/{len(prompts)}", end="\r")
        result = call_llm(**kwargs)
        results.append(result)
        if i < len(prompts) - 1:
            time.sleep(delay)
    print()
    return results