"""
client.py
Unified LLM client for Groq and Google AI Studio.

Routing:
  - Groq models          → OpenAI-compat endpoint (api.groq.com)
  - Google Gemini models → OpenAI-compat endpoint (generativelanguage.googleapis.com/v1beta/openai)
  - Google Gemma models  → Native generateContent endpoint
                           (OpenAI-compat does not support Gemma models)

Gemma models are detected by model_id prefix: "gemma-"
"""

import time
from typing import Optional

import requests

from core.src.api.rate_tracker import QuotaExhaustedError, RateTracker

# OpenAI-compatible endpoints
OPENAI_COMPAT_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
}

# Google native endpoint for Gemma (and any model that needs it)
GOOGLE_NATIVE_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

MAX_RETRIES = 3
RETRY_WAIT_SECS = 5
REQUEST_TIMEOUT = 90


def _is_gemma(model_id: str) -> bool:
    return model_id.lower().startswith("gemma-")


class LLMClient:
    def __init__(self, api_keys: dict, rate_tracker: RateTracker):
        self.api_keys = api_keys
        self.rate_tracker = rate_tracker

    def chat(
        self,
        provider: str,
        model_id: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        system: Optional[str] = None,
    ) -> str:
        provider = provider.lower()

        if provider not in OPENAI_COMPAT_URLS:
            raise ValueError(
                f"Unknown provider '{provider}'. Supported: {list(OPENAI_COMPAT_URLS)}"
            )

        keys = self.api_keys.get(provider, [])
        if not keys:
            raise ValueError(f"No API keys configured for provider '{provider}'.")

        # Prepend system message for OpenAI-compat path (Gemma handles it separately)
        full_messages = list(messages)
        if system and not _is_gemma(model_id):
            if not full_messages or full_messages[0].get("role") != "system":
                full_messages = [{"role": "system", "content": system}] + full_messages

        api_key, key_idx = self.rate_tracker.get_available_key(provider, model_id, keys)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if provider == "google" and _is_gemma(model_id):
                    response_text = self._call_gemma_native(
                        model_id, api_key, messages, temperature, max_tokens, system
                    )
                else:
                    response_text = self._call_openai_compat(
                        provider,
                        model_id,
                        api_key,
                        full_messages,
                        temperature,
                        max_tokens,
                    )
                self.rate_tracker.record_request(provider, key_idx, model_id)
                return response_text

            except _RateLimitError:
                if attempt < MAX_RETRIES:
                    try:
                        api_key, key_idx = self.rate_tracker.handle_429(
                            provider, key_idx, model_id, keys
                        )
                        time.sleep(RETRY_WAIT_SECS)
                        continue
                    except QuotaExhaustedError:
                        raise
                else:
                    raise

            except _RetryableError as e:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_WAIT_SECS * attempt)
                    continue
                raise RuntimeError(
                    f"API call failed after {MAX_RETRIES} attempts: {e}"
                ) from e

    # ── OpenAI-compatible call (Groq + Gemini) ─────────────────────────────────

    def _call_openai_compat(
        self,
        provider: str,
        model_id: str,
        api_key: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        url = f"{OPENAI_COMPAT_URLS[provider]}/chat/completions"

        resp = requests.post(
            url,
            json={
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=REQUEST_TIMEOUT,
        )

        self._raise_for_status(resp, provider)

        try:
            return resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"Unexpected response shape from {provider}: {resp.json()}"
            ) from e

    # ── Google native generateContent call (Gemma models) ─────────────────────

    def _call_gemma_native(
        self,
        model_id: str,
        api_key: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        system: Optional[str],
    ) -> str:
        url = f"{GOOGLE_NATIVE_BASE}/{model_id}:generateContent"

        # Convert OpenAI-style messages to Google contents format
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append(
                {
                    "role": role,
                    "parts": [{"text": m["content"]}],
                }
            )

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        resp = requests.post(
            url,
            json=payload,
            params={"key": api_key},
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )

        self._raise_for_status(resp, "google/gemma")

        try:
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"Unexpected response shape from Google Gemma native API: {resp.json()}"
            ) from e

    # ── shared error handler ───────────────────────────────────────────────────

    def _raise_for_status(self, resp: requests.Response, provider: str) -> None:
        if resp.status_code == 429:
            raise _RateLimitError(f"429 from {provider}")
        if resp.status_code in (500, 502, 503, 504):
            raise _RetryableError(f"HTTP {resp.status_code} from {provider}")
        if resp.status_code != 200:
            raise RuntimeError(
                f"API error {resp.status_code} from {provider}: {resp.text[:300]}"
            )


# ── private exception sentinels ────────────────────────────────────────────────


class _RateLimitError(Exception):
    pass


class _RetryableError(Exception):
    pass
