from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Put it in your .env file as GEMINI_API_KEY=..."
    )

client = genai.Client(api_key=GEMINI_API_KEY)

FLASH = "gemini-2.5-flash"
PRO = "gemini-2.5-pro"


def _extract_text_from_response(response: Any) -> str:
    """
    Gemini may return:
      - response.text populated
      - response.text = None but text available in candidates[].content.parts[]
    This helper safely recovers the text.
    """
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    parts: list[str] = []
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                parts.append(part_text.strip())

    return "\n".join(parts).strip()


def _strip_json_fences(text: str) -> str:
    """
    Remove markdown fences if Gemini wraps JSON in ```json ... ``` blocks.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def call_llm(
    prompt: str,
    system: Optional[str] = None,
    model: str = FLASH,
    temperature: float = 0.0,
    max_tokens: int = 512,
    retries: int = 3,
    expect_json: bool = False,
) -> str:
    """
    Core LLM call. Returns response text.
    All graders/agents should use this — never call genai directly.
    """
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system if system else "You are a helpful assistant.",
    )

    last_error: Optional[Exception] = None

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )

            text = _extract_text_from_response(response)

            if not text:
                raise RuntimeError(
                    f"LLM returned empty response text. "
                    f"finish_reason={getattr(response.candidates[0], 'finish_reason', None) if getattr(response, 'candidates', None) else None}"
                )

            if expect_json:
                text = _strip_json_fences(text)

            return text

        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  [llm] attempt {attempt + 1} failed: {e} — retrying in {wait}s")
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM call failed after {retries} attempts: {e}") from e

    raise RuntimeError(f"LLM call failed: {last_error}")


def call_llm_json(prompt: str, system: Optional[str] = None, **kwargs) -> dict:
    """
    Convenience wrapper — calls LLM and parses JSON response.
    """
    kwargs["expect_json"] = True
    kwargs["temperature"] = kwargs.get("temperature", 0.0)

    text = call_llm(prompt, system=system, **kwargs)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON:\n{text}\n\nError: {e}") from e


def batch_call(prompts: list[dict], delay: float = 1.5) -> list[str]:
    """
    Run a list of LLM calls with a delay between each.
    Set delay=4.0 on free tier to avoid rate limits.
    """
    results: list[str] = []

    for i, kwargs in enumerate(prompts):
        print(f"  [batch] {i + 1}/{len(prompts)}", end="\r")
        result = call_llm(**kwargs)
        results.append(result)
        if i < len(prompts) - 1:
            time.sleep(delay)

    print()
    return results