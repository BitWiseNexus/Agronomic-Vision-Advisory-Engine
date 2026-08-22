"""LLM providers — free tiers only, no vendor SDKs.

Both remote providers are plain `httpx` calls against a documented REST endpoint. That
keeps the dependency surface at zero extra packages and makes swapping providers a
15-line change rather than a rewrite.

    groq    -> llama-3.3-70b-versatile   (OpenAI-compatible chat completions)
    gemini  -> gemini-2.0-flash          (generateContent)
    none    -> NullProvider, which raises; the advisor then builds a deterministic plan

Selection order for LLM_PROVIDER=auto: Groq key, then Gemini key, then Null.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

import httpx

from app.config import Settings, get_settings
from app.core.logging import get_logger

log = get_logger(__name__)

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class LLMUnavailable(RuntimeError):
    """No usable LLM. Always recoverable — the advisor falls back to the template plan."""


class LLMProvider(ABC):
    name: str = "unknown"
    model: str = ""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, *, json_mode: bool = True) -> str:
        """Return the raw completion text. Raise LLMUnavailable on any failure."""


class NullProvider(LLMProvider):
    name = "template"

    def __init__(self, reason: str = "no API key configured") -> None:
        self.reason = reason

    def generate(self, system_prompt: str, user_prompt: str, *, json_mode: bool = True) -> str:
        raise LLMUnavailable(self.reason)


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.groq_model
        self._key = settings.groq_api_key

    def generate(self, system_prompt: str, user_prompt: str, *, json_mode: bool = True) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            response = httpx.post(
                GROQ_ENDPOINT,
                headers={"Authorization": f"Bearer {self._key}"},
                json=payload,
                timeout=self.settings.llm_timeout_seconds,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:400]
            raise LLMUnavailable(f"Groq HTTP {exc.response.status_code}: {detail}") from exc
        except Exception as exc:
            raise LLMUnavailable(f"Groq call failed: {exc}") from exc


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.gemini_model
        self._key = settings.gemini_api_key

    def generate(self, system_prompt: str, user_prompt: str, *, json_mode: bool = True) -> str:
        generation: dict = {
            "temperature": self.settings.llm_temperature,
            "maxOutputTokens": self.settings.llm_max_tokens,
        }
        if json_mode:
            generation["responseMimeType"] = "application/json"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": generation,
        }
        try:
            response = httpx.post(
                GEMINI_ENDPOINT.format(model=self.model),
                params={"key": self._key},
                json=payload,
                timeout=self.settings.llm_timeout_seconds,
            )
            response.raise_for_status()
            candidates = response.json().get("candidates") or []
            if not candidates:
                raise LLMUnavailable("Gemini returned no candidates (likely a safety block)")
            parts = candidates[0].get("content", {}).get("parts") or []
            text = "".join(part.get("text", "") for part in parts)
            if not text.strip():
                raise LLMUnavailable("Gemini returned an empty completion")
            return text
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:400]
            raise LLMUnavailable(f"Gemini HTTP {exc.response.status_code}: {detail}") from exc
        except LLMUnavailable:
            raise
        except Exception as exc:
            raise LLMUnavailable(f"Gemini call failed: {exc}") from exc


def resolve_provider(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    choice = (settings.llm_provider or "auto").strip().lower()

    if choice == "groq":
        if not settings.groq_api_key:
            return NullProvider("LLM_PROVIDER=groq but GROQ_API_KEY is empty")
        return GroqProvider(settings)
    if choice == "gemini":
        if not settings.gemini_api_key:
            return NullProvider("LLM_PROVIDER=gemini but GEMINI_API_KEY is empty")
        return GeminiProvider(settings)
    if choice == "template":
        return NullProvider("LLM_PROVIDER=template (deterministic advisor requested)")

    # auto
    if settings.groq_api_key:
        return GroqProvider(settings)
    if settings.gemini_api_key:
        return GeminiProvider(settings)
    return NullProvider("no GROQ_API_KEY or GEMINI_API_KEY found in .env")


def extract_json_object(text: str) -> dict:
    """Tolerant JSON extraction — strips ```json fences and trailing prose, which even
    JSON-mode models occasionally emit."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned[3:]
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError(f"No JSON object found in completion: {text[:200]!r}")
