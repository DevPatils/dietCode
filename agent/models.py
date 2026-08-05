"""Which models a provider will actually accept.

Any OpenAI-compatible endpoint exposes /v1/models, so the same call works for
all three providers -- but each returns a different mix, including embedding,
audio and vision-only ids that will 400 the moment the loop sends them a tool
schema. Filtering here is what makes "pick any of the three providers and it
just works" true: the picker only ever offers ids that can run the agent.

The listing is advisory. A model id the user types is still passed through,
because provider catalogues change faster than this file does.
"""

from __future__ import annotations

from typing import Any

# Substrings that mark an id as something the agentic loop cannot drive. Every
# one of these has been seen in a real /models response from groq, gemini or
# openai.
_NOT_CHAT = (
    "embed",       # embeddings: no chat completions endpoint at all
    "whisper",     # speech to text
    "tts",         # text to speech
    "audio",
    "guard",       # llama-guard / prompt-guard: classifiers, no tool calling
    "moderation",
    "image",
    "dall-e",
    "sora",
    "veo",
    "imagen",
    "rerank",
    "aqa",
    "-vision",
    "codex",       # responses-API only
    "realtime",
    "transcribe",
    "instruct",    # completions-style, not chat
    "davinci",
    "babbage",
)

# Shown when the network call fails -- offline, rate limited, or a provider
# that does not implement /models. Better a short known-good list than an empty
# picker.
FALLBACK_MODELS: dict[str, tuple[str, ...]] = {
    "groq": (
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3-32b",
    ),
    "gemini": (
        "gemini-flash-lite-latest",
        "gemini-flash-latest",
        "gemini-2.5-pro",
    ),
    "openai": (
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-4o-mini",
        "gpt-4o",
    ),
}


def is_chat_model(model_id: str) -> bool:
    low = model_id.lower()
    return not any(marker in low for marker in _NOT_CHAT)


def _normalize(model_id: str) -> str:
    """Gemini prefixes every id with 'models/'; its chat endpoint accepts both,
    but showing the prefix in a picker and then in the status bar is noise."""
    return model_id.removeprefix("models/")


def list_models(client: Any, provider: str) -> tuple[list[str], str | None]:
    """(model ids, error). Never raises -- a picker must always have options."""
    try:
        response = client.models.list()
        raw = [
            _normalize(str(getattr(item, "id", "") or ""))
            for item in getattr(response, "data", None) or response
        ]
    except Exception as exc:  # noqa: BLE001 - any failure falls back to the list
        return list(FALLBACK_MODELS.get(provider, ())), str(exc)

    usable = sorted({m for m in raw if m and is_chat_model(m)})
    if not usable:
        return list(FALLBACK_MODELS.get(provider, ())), "provider listed no chat models"
    return usable, None


def rank_models(models: list[str], preferred: str) -> list[str]:
    """Put the provider's default first, then the rest alphabetically.

    The default is the one that has been tested end to end, so it should be the
    option under the cursor when the picker opens.
    """
    rest = [m for m in models if m != preferred]
    return ([preferred] if preferred in models else []) + rest
