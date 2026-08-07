"""Groq, on the `groq` SDK.

Same wire protocol as OpenAI, different package and a different exception
hierarchy, which is the reason this is its own transport rather than a base_url.

Groq's distinguishing behaviour is server-side validation of the model's tool
arguments against our schema: a mismatch rejects the entire generation with a
400 rather than passing the bad call through. That is why the tool schemas keep
union types (`["integer", "string"]`) for this provider, and why a rejected
generation is salvaged from the error body instead of being lost.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from . import _openai_shape as shape
from .base import Completion, TransportError


class GroqTransport:
    provider = "groq"

    def __init__(
        self, api_key: str, base_url: str | None = None, timeout: float = 120.0
    ):
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise TransportError(f"the groq SDK is not installed: {exc}") from None

        # max_retries=0 because the loop owns retries and their backoff.
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": 0,
        }
        if base_url:
            kwargs["base_url"] = base_url
        self._client = Groq(**kwargs)
        self._include_usage = True

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        stream: bool = False,
        on_text: Callable[[str], None] | None = None,
    ) -> Completion:
        kwargs = shape.build_kwargs(
            model, shape.clean_messages(messages), tools, stream, self._include_usage
        )
        try:
            raw = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - classified by the caller
            # The 400 body carries what the model actually generated, so the
            # call can be recovered rather than the task lost.
            rejected = shape.failed_generation_text(exc)
            if rejected:
                return Completion(content=rejected)
            if self._include_usage and "stream_options" in str(exc).lower():
                self._include_usage = False
                kwargs = shape.build_kwargs(
                    model, shape.clean_messages(messages), tools, stream, False
                )
                raw = self._client.chat.completions.create(**kwargs)
            else:
                raise
        return shape.from_stream(raw, on_text) if stream else shape.from_response(raw)

    def list_models(self) -> list[str]:
        return [str(m.id) for m in self._client.models.list().data]

    def is_transient(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status == 429 or 500 <= status < 600
        try:
            import groq
        except ImportError:  # pragma: no cover
            return False
        return isinstance(
            exc, groq.APIConnectionError | groq.APITimeoutError | groq.InternalServerError
        )

    def is_quota_exhausted(self, exc: Exception) -> bool:
        """Groq words its daily cap as tokens-per-day."""
        text = str(exc).lower()
        return "per day" in text or "tpd" in text or "rpd" in text
