"""OpenAI, on the `openai` SDK.

Also the transport used for any custom `--base-url`: Ollama, vLLM and
OpenRouter all serve this protocol, and pointing the official client at them is
the whole integration.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from . import _openai_shape as shape
from .base import Completion, TransportError

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAITransport:
    provider = "openai"

    def __init__(
        self, api_key: str, base_url: str | None = None, timeout: float = 120.0
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise TransportError(f"the openai SDK is not installed: {exc}") from None

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or DEFAULT_BASE_URL,
            # The SDK defaults to 600s, so one hung request can stall a
            # benchmark task for ten minutes and then fail anyway.
            timeout=timeout,
            # The loop owns retries and their backoff; the SDK's own would
            # multiply both silently.
            max_retries=0,
        )
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
            # Not every OpenAI-compatible server knows stream_options. Losing
            # the token count beats losing the turn.
            if self._include_usage and "stream_options" in str(exc).lower():
                self._include_usage = False
                kwargs = shape.build_kwargs(
                    model, shape.clean_messages(messages), tools, stream, False
                )
                raw = self._client.chat.completions.create(**kwargs)
            else:
                rejected = shape.failed_generation_text(exc)
                if rejected:
                    return Completion(content=rejected)
                raise
        return shape.from_stream(raw, on_text) if stream else shape.from_response(raw)

    def list_models(self) -> list[str]:
        return [str(m.id) for m in self._client.models.list().data]

    def is_transient(self, exc: Exception) -> bool:
        return _openai_transient(exc)

    def is_quota_exhausted(self, exc: Exception) -> bool:
        return _daily_cap(exc)


def _openai_transient(exc: Exception) -> bool:
    """Classify by type and status, never by grepping the message.

    String-matching an error body is how the `tool_use_failed` bug hid, and a
    typed 4xx must never be retried.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or 500 <= status < 600
    try:
        import openai
    except ImportError:  # pragma: no cover
        return False
    return isinstance(
        exc, openai.APIConnectionError | openai.APITimeoutError | openai.InternalServerError
    )


def _daily_cap(exc: Exception) -> bool:
    """A cap that will not clear during this run.

    Both a burst limit and a daily one arrive as 429, but they need opposite
    handling: backing off for hours is just a slow failure.
    """
    text = str(exc).lower()
    if "per day" in text or "tpd" in text or "rpd" in text:
        return True
    return "exceeded your current quota" in text and "perday" in text.replace("-", "")
