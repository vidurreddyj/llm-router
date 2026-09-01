"""
Thin wrapper around the Gemini API that adds:
  - cost tracking per call (based on config.py pricing)
  - a confidence self-report from the cheap model, used for escalation
  - basic timing

Requires GOOGLE_API_KEY to be set in the environment. Get a free key (no
credit card) at https://aistudio.google.com -> "Get API key".

Uses the `google-genai` SDK (the current, actively-maintained package --
the older `google-generativeai` package is deprecated as of 2026).
"""

import os
import time
import re
import logging
from dataclasses import dataclass

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from . import config

# Google's SDK logs a noisy (non-blocking) recommendation to use the Chat
# API instead of one-shot generate_content calls. We're intentionally
# stateless per-call here (each request is independent, no conversation
# history), so generate_content is the right call -- silence the advisory.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))


@dataclass
class LLMResult:
    text: str
    model_key: str        # "cheap" or "strong"
    model_name: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    confidence: float | None = None  # only populated for cheap-model calls


def _cost(model_key: str, input_tokens: int, output_tokens: int) -> float:
    m = config.MODELS[model_key]
    return (
        input_tokens / 1_000_000 * m["price_in_per_m"]
        + output_tokens / 1_000_000 * m["price_out_per_m"]
    )


_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*([0-9.]+)", re.IGNORECASE)
_RETRY_DELAY_RE = re.compile(r"retry in ([0-9.]+)s", re.IGNORECASE)


def _call_with_quota_retry(model_name: str, prompt: str, max_retries: int = 2):
    """Call generate_content, and if we hit a 429 (rate/quota limit),
    parse Google's suggested wait time out of the error and retry once or
    twice instead of crashing the whole eval run over a transient limit."""
    for attempt in range(max_retries + 1):
        try:
            return _client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=config.MAX_TOKENS,
                ),
            )
        except genai_errors.ClientError as e:
            is_last = attempt == max_retries
            if "RESOURCE_EXHAUSTED" not in str(e) or is_last:
                raise
            match = _RETRY_DELAY_RE.search(str(e))
            wait_s = float(match.group(1)) + 1.0 if match else 30.0
            print(f"    (rate/quota limit hit on {model_name}, "
                  f"waiting {wait_s:.0f}s before retry {attempt + 1}/{max_retries}...)")
            time.sleep(wait_s)


def call_model(
    model_key: str,
    query: str,
    ask_confidence: bool = False,
) -> LLMResult:
    """Call either the 'cheap' or 'strong' model (as defined in config.py).

    If ask_confidence=True, the model is asked to self-report a confidence
    score (0-1) at the end of its answer, which the router uses to decide
    whether to escalate to the strong model. This is a cheap, dependency-free
    stand-in for calibrated confidence estimation.
    """
    model_cfg = config.MODELS[model_key]

    prompt = query
    if ask_confidence:
        prompt = (
            "Answer the user's question directly and concisely. "
            "After your answer, on a new line, output exactly: "
            "CONFIDENCE: <a number between 0 and 1 indicating how confident "
            "you are that your answer is correct and complete>.\n\n"
            f"Question: {query}"
        )

    start = time.perf_counter()
    resp = _call_with_quota_retry(model_cfg["name"], prompt)
    latency = time.perf_counter() - start

    text = resp.text or ""

    confidence = None
    if ask_confidence:
        match = _CONFIDENCE_RE.search(text)
        if match:
            confidence = float(match.group(1))
            text = _CONFIDENCE_RE.sub("", text).strip()

    usage = resp.usage_metadata
    input_tokens = usage.prompt_token_count if usage else 0
    output_tokens = usage.candidates_token_count if usage else 0

    return LLMResult(
        text=text,
        model_key=model_key,
        model_name=model_cfg["name"],
        input_tokens=input_tokens or 0,
        output_tokens=output_tokens or 0,
        cost_usd=_cost(model_key, input_tokens or 0, output_tokens or 0),
        latency_s=latency,
        confidence=confidence,
    )
