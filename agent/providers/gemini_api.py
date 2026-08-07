"""Gemini, on the `google-genai` SDK.

Native rather than Gemini's OpenAI-compatibility endpoint, which means:

- history is `Content(role, parts)`, and the assistant role is `"model"`;
- a tool call is a `function_call` part, a result is a `function_response` part
  inside a **user** Content;
- the system prompt is `config.system_instruction`, not a message;
- tools are `FunctionDeclaration`s wrapped in a `Tool`.

Schemas go through `parameters_json_schema`, which takes raw JSON Schema. The
older `parameters` field is OpenAPI-derived and cannot express a union type, so
it is what forced `tools_for()` to narrow `["integer", "string"]` down to one
type for this provider. Passing JSON Schema directly removes that constraint.
"""

from __future__ import annotations

import base64
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

KNOWN_MODELS = (
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-2.5-pro",
)


class GeminiTransport:
    provider = "gemini"

    def __init__(
        self, api_key: str, base_url: str | None = None, timeout: float = 120.0
    ):
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise TransportError(
                f"the google-genai SDK is not installed: {exc}"
            ) from None

        from google.genai import types

        options: dict[str, Any] = {
            "timeout": int(timeout * 1000),  # milliseconds, unlike the others
            # One attempt: the loop owns retries and their backoff.
            "retry_options": types.HttpRetryOptions(attempts=1),
        }
        if base_url:
            options["base_url"] = base_url
        self._genai = genai
        self._client = genai.Client(api_key=api_key, http_options=options)

    # -- outbound ------------------------------------------------------------

    def _tools(self, tools: Sequence[dict[str, Any]]) -> list[Any]:
        from google.genai import types

        declarations = []
        for tool in tools:
            name, description, parameters = json_schema_for(tool)
            if not name:
                continue
            declarations.append(
                types.FunctionDeclaration(
                    name=name,
                    description=description,
                    # Raw JSON Schema, so union types survive intact.
                    parameters_json_schema=parameters,
                )
            )
        return [types.Tool(function_declarations=declarations)] if declarations else []

    def _contents(self, messages: Sequence[dict[str, Any]]) -> list[Any]:
        """OpenAI-shaped history into Gemini Contents.

        Tool results are merged into one user Content per assistant turn, for
        the same reason Anthropic needs it: the call and its response are
        matched as a pair, and splitting them across messages breaks it.
        """
        from google.genai import types

        out: list[Any] = []
        pending: list[Any] = []
        names_by_id: dict[str, str] = {}

        def flush() -> None:
            if pending:
                out.append(types.Content(role="user", parts=list(pending)))
                pending.clear()

        for message in messages:
            role = message.get("role")

            if role == "tool":
                call_id = message.get("tool_call_id") or ""
                pending.append(
                    types.Part.from_function_response(
                        # Gemini matches on the function's name, not on an id.
                        name=names_by_id.get(call_id, message.get("name") or "tool"),
                        response={"result": _as_text(message.get("content"))},
                    )
                )
                continue

            flush()

            if role == "assistant":
                parts: list[Any] = []
                text = _as_text(message.get("content"))
                if text.strip():
                    parts.append(types.Part(text=text))
                for call in message.get("tool_calls") or []:
                    function = call.get("function", {})
                    name = function.get("name", "")
                    names_by_id[call.get("id") or ""] = name
                    part = types.Part(
                        function_call=types.FunctionCall(
                            name=name,
                            args=arguments_to_dict(function.get("arguments")),
                        )
                    )
                    # Gemini 3 refuses turn two with "Function call is missing a
                    # thought_signature" unless the one it issued comes back
                    # untouched. It lives on the Part, not on the FunctionCall.
                    signature = call.get("thought_signature")
                    if signature:
                        part.thought_signature = _decode_signature(signature)
                    parts.append(part)
                if parts:
                    out.append(types.Content(role="model", parts=parts))
                continue

            text = _as_text(message.get("content"))
            if text.strip():
                out.append(types.Content(role="user", parts=[types.Part(text=text)]))

        flush()
        return out

    # -- the call ------------------------------------------------------------

    def _config(self, system: str, tools: Sequence[dict[str, Any]]) -> Any:
        from google.genai import types

        options: dict[str, Any] = {}
        if system:
            options["system_instruction"] = system
        converted = self._tools(tools)
        if converted:
            options["tools"] = converted
            # The SDK will otherwise run tool calls itself and return only the
            # final text, which takes the loop out of the loop entirely.
            options["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                disable=True
            )
        return types.GenerateContentConfig(**options)

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
        contents = self._contents(rest)
        config = self._config(system, tools)

        if stream:
            return self._stream(model, contents, config, on_text)
        return _from_response(
            self._client.models.generate_content(
                model=model, contents=contents, config=config
            )
        )

    def _stream(
        self, model: str, contents: Any, config: Any, on_text: Callable[[str], None] | None
    ) -> Completion:
        text_parts: list[str] = []
        calls: list[dict[str, Any]] = []
        usage = None

        for chunk in self._client.models.generate_content_stream(
            model=model, contents=contents, config=config
        ):
            raw_usage = getattr(chunk, "usage_metadata", None)
            if raw_usage is not None:
                usage = raw_usage  # the last one is cumulative
            text, chunk_calls = _parts_of(chunk)
            if text:
                text_parts.append(text)
                if on_text:
                    on_text(text)
            calls.extend(chunk_calls)

        return Completion(
            content="".join(text_parts), tool_calls=calls, usage=_usage(usage)
        )

    # -- capabilities --------------------------------------------------------

    def list_models(self) -> list[str]:
        try:
            names = []
            for model in self._client.models.list():
                name = str(getattr(model, "name", "") or "")
                # Every id comes back prefixed; the API accepts either form,
                # but showing "models/..." in a picker is noise.
                names.append(name.removeprefix("models/"))
            return [n for n in names if n]
        except Exception:  # noqa: BLE001 - a picker must always have options
            return list(KNOWN_MODELS)

    def is_transient(self, exc: Exception) -> bool:
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if isinstance(code, int):
            return code == 429 or 500 <= code < 600
        text = str(exc)
        return "503" in text or "UNAVAILABLE" in text or "DEADLINE_EXCEEDED" in text

    def is_quota_exhausted(self, exc: Exception) -> bool:
        """The free tier is per model per day, and 20 requests on some models.

        Gemini never says "per day" in the message; the daily part is only in
        the quotaId, so missing it costs a full retry backoff per turn against
        a cap that clears tomorrow.
        """
        text = str(exc).lower()
        if "exceeded your current quota" in text and "perday" in text.replace("-", ""):
            return True
        return "per day" in text or "rpd" in text


# -- inbound -----------------------------------------------------------------


def _parts_of(response: Any) -> tuple[str, list[dict[str, Any]]]:
    text_parts: list[str] = []
    calls: list[dict[str, Any]] = []

    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                text_parts.append(text)
            call = getattr(part, "function_call", None)
            if call is not None and getattr(call, "name", None):
                extra = {}
                signature = getattr(part, "thought_signature", None)
                if signature:
                    # Base64 because the transcript is JSON and this arrives as
                    # raw bytes; it must survive a save/resume round trip.
                    extra["thought_signature"] = _encode_signature(signature)
                calls.append(
                    normalize_tool_call(
                        call.name,
                        json.dumps(dict(getattr(call, "args", None) or {}), default=str),
                        call_id=getattr(call, "id", None),
                        **extra,
                    )
                )
    return "".join(text_parts), calls


def _from_response(response: Any) -> Completion:
    text, calls = _parts_of(response)
    return Completion(
        content=text, tool_calls=calls, usage=_usage(getattr(response, "usage_metadata", None))
    )


def _usage(raw: Any) -> Usage | None:
    if raw is None:
        return None
    prompt = getattr(raw, "prompt_token_count", 0) or 0
    completion = getattr(raw, "candidates_token_count", 0) or 0
    total = getattr(raw, "total_token_count", 0) or (prompt + completion)
    return Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def _encode_signature(signature: Any) -> str:
    """Bytes from the API into something a JSON transcript can hold."""
    if isinstance(signature, bytes):
        return base64.b64encode(signature).decode("ascii")
    return str(signature)


def _decode_signature(signature: Any) -> Any:
    """The reverse, tolerant of a signature that was never bytes."""
    if isinstance(signature, bytes):
        return signature
    try:
        return base64.b64decode(signature, validate=True)
    except (ValueError, TypeError):
        return signature


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, default=str)
