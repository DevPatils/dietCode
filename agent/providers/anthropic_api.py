"""Anthropic, on the `anthropic` SDK.

The one provider that is not OpenAI-shaped, so this file is mostly translation:

- `system` is a top-level parameter, not a message with `role: "system"`;
- a tool call is a `tool_use` content block on the assistant turn, not a
  `tool_calls` array beside the content;
- a tool result is a `tool_result` block inside a **user** message, not a
  message with `role: "tool"`;
- tools declare `input_schema`, not `function.parameters`;
- arguments arrive already parsed as a dict, not as a JSON string;
- `max_tokens` is required, where every other provider defaults it.

All of that is converted here and nowhere else. The loop and the transcript
keep the OpenAI shape, so a session started on Claude can be resumed on Groq.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from .base import (
    Completion,
    TransportError,
    Usage,
    arguments_to_dict,
    json_schema_for,
    normalize_tool_call,
    split_system,
)

# Required by the API. Generous enough for a long file write, which is the
# biggest thing this agent generates in one turn.
DEFAULT_MAX_TOKENS = 8192

# Claude models, newest first. Used when the account cannot list models.
KNOWN_MODELS = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-haiku-4-5-20251001",
)


class AnthropicTransport:
    provider = "anthropic"

    def __init__(
        self, api_key: str, base_url: str | None = None, timeout: float = 120.0
    ):
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise TransportError(f"the anthropic SDK is not installed: {exc}") from None

        # max_retries=0 because the loop owns retries and their backoff.
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": 0,
        }
        if base_url:
            kwargs["base_url"] = base_url
        self._client = Anthropic(**kwargs)

    # -- outbound ------------------------------------------------------------

    def _tools(self, tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for tool in tools:
            name, description, parameters = json_schema_for(tool)
            if not name:
                continue
            converted.append(
                {"name": name, "description": description, "input_schema": parameters}
            )
        return converted

    def _messages(self, messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """OpenAI-shaped history into Anthropic content blocks.

        Consecutive tool results have to be merged into a single user message:
        the API pairs every `tool_use` in one assistant turn with the
        `tool_result` blocks in the next user turn, so emitting one message per
        result breaks the pairing and is rejected.
        """
        out: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []

        def flush() -> None:
            if pending_results:
                out.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for message in messages:
            role = message.get("role")

            if role == "tool":
                pending_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id") or "",
                        "content": _as_text(message.get("content")),
                    }
                )
                continue

            flush()

            if role == "assistant":
                blocks: list[dict[str, Any]] = []
                text = _as_text(message.get("content"))
                if text.strip():
                    blocks.append({"type": "text", "text": text})
                for call in message.get("tool_calls") or []:
                    function = call.get("function", {})
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.get("id") or "",
                            "name": function.get("name", ""),
                            "input": arguments_to_dict(function.get("arguments")),
                        }
                    )
                if blocks:
                    out.append({"role": "assistant", "content": blocks})
                continue

            text = _as_text(message.get("content"))
            if text.strip():
                out.append({"role": "user", "content": text})

        flush()
        return _ensure_alternating(out)

    # -- the call ------------------------------------------------------------

    def complete(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        stream: bool = False,
        on_text: Callable[[str], None] | None = None,
    ) -> Completion:
        system, rest = split_system(messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "messages": self._messages(rest),
        }
        if system:
            kwargs["system"] = system
        converted_tools = self._tools(tools)
        if converted_tools:
            kwargs["tools"] = converted_tools

        if stream:
            return self._stream(kwargs, on_text)
        return _from_message(self._client.messages.create(**kwargs))

    def _stream(
        self, kwargs: dict[str, Any], on_text: Callable[[str], None] | None
    ) -> Completion:
        """Streamed turns go through the SDK's own accumulator.

        It already reassembles content blocks, including tool input that
        arrives as partial JSON, so hand-rolling it would only add a second
        place for the fragment bugs to live.
        """
        with self._client.messages.stream(**kwargs) as stream:
            if on_text is not None:
                for text in stream.text_stream:
                    on_text(text)
            return _from_message(stream.get_final_message())

    # -- capabilities --------------------------------------------------------

    def list_models(self) -> list[str]:
        try:
            return [str(m.id) for m in self._client.models.list().data]
        except Exception:  # noqa: BLE001 - older keys cannot list models
            return list(KNOWN_MODELS)

    def is_transient(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status == 429 or status == 529 or 500 <= status < 600
        try:
            import anthropic
        except ImportError:  # pragma: no cover
            return False
        return isinstance(
            exc,
            anthropic.APIConnectionError
            | anthropic.APITimeoutError
            | anthropic.InternalServerError,
        )

    def is_quota_exhausted(self, exc: Exception) -> bool:
        """Credit exhaustion, which no amount of waiting fixes.

        Distinct from a rate limit: Anthropic has no free tier, so the failure
        that actually strands a run is an empty balance.
        """
        text = str(exc).lower()
        return "credit balance is too low" in text or "billing" in text


# -- inbound -----------------------------------------------------------------


def _from_message(message: Any) -> Completion:
    """An Anthropic response into the loop's Completion."""
    text_parts: list[str] = []
    calls: list[dict[str, Any]] = []

    for block in getattr(message, "content", None) or []:
        kind = getattr(block, "type", None)
        if kind == "text":
            text_parts.append(getattr(block, "text", "") or "")
        elif kind == "tool_use":
            calls.append(
                normalize_tool_call(
                    getattr(block, "name", ""),
                    # Arguments come parsed; the rest of the codebase expects
                    # the OpenAI convention of a JSON string.
                    json.dumps(getattr(block, "input", {}) or {}, default=str),
                    call_id=getattr(block, "id", None),
                )
            )

    raw = getattr(message, "usage", None)
    usage = None
    if raw is not None:
        prompt = getattr(raw, "input_tokens", 0) or 0
        completion = getattr(raw, "output_tokens", 0) or 0
        usage = Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            # Anthropic reports the two separately and never their sum, but the
            # metrics table and every budget check read total_tokens.
            total_tokens=prompt + completion,
        )
    return Completion(content="".join(text_parts), tool_calls=calls, usage=usage)


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, default=str)


def _ensure_alternating(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The API requires the first message to be from the user.

    A resumed transcript can begin with an assistant turn once trimming has
    dropped the oldest messages, and that is a 400 rather than a warning.
    """
    while messages and messages[0].get("role") != "user":
        messages.pop(0)
    if not messages:
        # Every request needs at least one message; an empty history means the
        # trim was aggressive, not that there is nothing to do.
        return [{"role": "user", "content": "Continue."}]
    return messages
