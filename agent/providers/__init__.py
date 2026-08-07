"""One transport per provider, each on that provider's own SDK.

The loop speaks exactly one message format -- the OpenAI shape -- and so does
the session transcript on disk. Everything a provider does differently is
converted at this boundary and nowhere else.

That rule is what keeps `/provider` switchable mid-conversation and keeps a
session recorded on one provider resumable on another. If Anthropic's content
blocks or Gemini's Parts leaked upward, a transcript would only be replayable
by the provider that wrote it.

Each transport turns a call into a `Completion(content, tool_calls, usage)`,
which is all the loop ever sees.
"""

from __future__ import annotations

from .base import (
    Completion,
    Transport,
    TransportError,
    normalize_tool_call,
    tool_call_id,
)

# Imported lazily by make_transport so a missing SDK is an error about that one
# provider, not an import failure that takes the whole CLI down.
_TRANSPORTS = {
    "groq": ("agent.providers.groq_api", "GroqTransport"),
    "openai": ("agent.providers.openai_api", "OpenAITransport"),
    "gemini": ("agent.providers.gemini_api", "GeminiTransport"),
    "anthropic": ("agent.providers.anthropic_api", "AnthropicTransport"),
}


def make_transport(
    provider: str, api_key: str, *, base_url: str | None = None, timeout: float = 120.0
) -> Transport:
    """Build the transport for a provider.

    `base_url` is honoured only by the OpenAI-protocol transports; it is how
    Ollama, vLLM and OpenRouter are reached.
    """
    import importlib

    try:
        module_name, class_name = _TRANSPORTS[provider]
    except KeyError:
        known = ", ".join(_TRANSPORTS)
        raise TransportError(
            f"unknown provider {provider!r}. Known providers: {known}"
        ) from None

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise TransportError(
            f"the {provider} SDK is not installed: {exc}. "
            f"Reinstall dietcode, or `pip install {_sdk_name(provider)}`."
        ) from None

    return getattr(module, class_name)(api_key=api_key, base_url=base_url, timeout=timeout)


def _sdk_name(provider: str) -> str:
    return {
        "groq": "groq",
        "openai": "openai",
        "gemini": "google-genai",
        "anthropic": "anthropic",
    }.get(provider, provider)


__all__ = [
    "Completion",
    "Transport",
    "TransportError",
    "make_transport",
    "normalize_tool_call",
    "tool_call_id",
]
