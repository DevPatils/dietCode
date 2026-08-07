"""One transport per provider, each on that provider's own SDK.

The loop and the transcript speak one format: the OpenAI shape. What each
provider does differently is converted at the transport boundary and nowhere
else, so what is worth testing is the conversion, in both directions, without
touching the network.
"""

from __future__ import annotations

import json

import pytest

from agent.providers import Completion, TransportError, make_transport
from agent.providers.base import arguments_to_dict, normalize_tool_call, split_system

# A conversation with every shape the loop produces: a system prompt, a user
# turn, an assistant turn carrying a tool call, and the result of that call.
CONVERSATION = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "make a file"},
    {
        "role": "assistant",
        "content": "on it",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": '{"path": "a.py", "content": "x"}',
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "Wrote 1 byte"},
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": ["integer", "string"]},
                },
                "required": ["command"],
            },
        },
    }
]


def transport(provider: str):
    return make_transport(provider, "test-key")


# -- the registry ------------------------------------------------------------


@pytest.mark.parametrize(
    "provider,expected",
    [
        ("groq", "GroqTransport"),
        ("openai", "OpenAITransport"),
        ("gemini", "GeminiTransport"),
        ("anthropic", "AnthropicTransport"),
    ],
)
def test_each_provider_gets_its_own_transport(provider, expected):
    built = transport(provider)
    assert type(built).__name__ == expected
    assert built.provider == provider


def test_every_configured_provider_has_a_transport():
    """A provider in the picker with no transport is a crash on first use."""
    from agent.auth import PROVIDERS

    for name in PROVIDERS:
        assert transport(name).provider == name


def test_an_unknown_provider_says_what_is_known():
    with pytest.raises(TransportError) as excinfo:
        make_transport("mistral", "k")
    assert "groq" in str(excinfo.value)


def test_every_transport_satisfies_the_protocol():
    for name in ("groq", "openai", "gemini", "anthropic"):
        built = transport(name)
        for method in ("complete", "list_models", "is_transient", "is_quota_exhausted"):
            assert callable(getattr(built, method)), f"{name}.{method}"


# -- shared conversion helpers ----------------------------------------------


def test_the_system_prompt_is_pulled_out_for_providers_that_want_it():
    system, rest = split_system(CONVERSATION)
    assert system == "You are an agent."
    assert all(m["role"] != "system" for m in rest)


def test_several_system_messages_are_joined_not_dropped():
    system, _ = split_system(
        [{"role": "system", "content": "one"}, {"role": "system", "content": "two"}]
    )
    assert "one" in system and "two" in system


def test_tool_arguments_stay_a_json_string_in_the_canonical_shape():
    """The transcript, parse_arguments and every existing test assume it."""
    call = normalize_tool_call("run_shell", {"command": "ls"})
    assert isinstance(call["function"]["arguments"], str)
    assert json.loads(call["function"]["arguments"]) == {"command": "ls"}


def test_a_call_with_no_id_gets_one():
    """Every tool_call needs an answerable id or the next request is rejected."""
    assert normalize_tool_call("x", "{}")["id"]


def test_malformed_arguments_do_not_invent_a_parameter():
    """Models emit broken JSON routinely; the tool reports what it needed."""
    assert arguments_to_dict("{not json") == {}
    assert arguments_to_dict(None) == {}
    assert arguments_to_dict('["a"]') == {}
    assert arguments_to_dict('{"a": 1}') == {"a": 1}


# -- Anthropic ---------------------------------------------------------------


def anthropic_messages(conversation=CONVERSATION):
    system, rest = split_system(conversation)
    return system, transport("anthropic")._messages(rest)


def test_anthropic_turns_a_tool_call_into_a_tool_use_block():
    _system, messages = anthropic_messages()
    assistant = [m for m in messages if m["role"] == "assistant"][0]
    block = [b for b in assistant["content"] if b["type"] == "tool_use"][0]

    assert block["name"] == "write_file"
    assert block["id"] == "call_1"
    assert block["input"] == {"path": "a.py", "content": "x"}  # parsed, not a string


def test_anthropic_turns_a_tool_result_into_a_user_block():
    """Anthropic has no `tool` role; results ride in the next user turn."""
    _system, messages = anthropic_messages()
    assert all(m["role"] in ("user", "assistant") for m in messages)
    result = [b for m in messages if m["role"] == "user"
              for b in (m["content"] if isinstance(m["content"], list) else [])
              if b.get("type") == "tool_result"][0]
    assert result["tool_use_id"] == "call_1"


def test_anthropic_merges_results_from_one_batch_into_one_message():
    """The API pairs every tool_use in a turn with the tool_results in the
    next one; one message per result breaks the pairing and is rejected."""
    conversation = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "t", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "t", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "one"},
        {"role": "tool", "tool_call_id": "b", "content": "two"},
    ]
    messages = transport("anthropic")._messages(conversation)
    results = [m for m in messages if isinstance(m["content"], list)
               and any(b.get("type") == "tool_result" for b in m["content"])]
    assert len(results) == 1
    assert len(results[0]["content"]) == 2


def test_anthropic_history_must_start_with_a_user_turn():
    """Trimming can drop the oldest messages and leave an assistant turn
    first, which is a 400 rather than a warning."""
    messages = transport("anthropic")._messages(
        [{"role": "assistant", "content": "orphaned"}, {"role": "user", "content": "hi"}]
    )
    assert messages[0]["role"] == "user"


def test_anthropic_never_sends_an_empty_conversation():
    messages = transport("anthropic")._messages([])
    assert messages and messages[0]["role"] == "user"


def test_anthropic_tools_use_input_schema():
    tools = transport("anthropic")._tools(TOOLS)
    assert tools[0]["name"] == "run_shell"
    assert "input_schema" in tools[0]
    assert tools[0]["input_schema"]["properties"]["command"]["type"] == "string"


def test_anthropic_keeps_the_union_type_the_dispatcher_coerces():
    """Anthropic takes raw JSON Schema, so nothing has to be narrowed."""
    tools = transport("anthropic")._tools(TOOLS)
    assert tools[0]["input_schema"]["properties"]["timeout"]["type"] == [
        "integer",
        "string",
    ]


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _AnthropicUsage:
    input_tokens = 11
    output_tokens = 7


def test_anthropic_response_becomes_the_loop_s_completion():
    from agent.providers.anthropic_api import _from_message

    message = _Block(
        content=[
            _Block(type="text", text="working"),
            _Block(type="tool_use", id="tu_1", name="run_shell", input={"command": "ls"}),
        ],
        usage=_AnthropicUsage(),
    )
    completion = _from_message(message)

    assert isinstance(completion, Completion)
    assert completion.content == "working"
    assert completion.tool_calls[0]["function"]["name"] == "run_shell"
    # Arguments come back parsed; the rest of the codebase expects a string.
    assert completion.tool_calls[0]["function"]["arguments"] == '{"command": "ls"}'
    assert completion.tool_calls[0]["id"] == "tu_1"


def test_anthropic_reports_a_total_it_never_sends():
    """Anthropic gives input and output separately, but the metrics table and
    every budget check read total_tokens."""
    from agent.providers.anthropic_api import _from_message

    usage = _from_message(_Block(content=[], usage=_AnthropicUsage())).usage
    assert usage.total_tokens == 18


def test_an_empty_balance_is_not_retried():
    """No free tier here, so the failure that strands a run is no credit."""
    built = transport("anthropic")
    assert built.is_quota_exhausted(RuntimeError("Your credit balance is too low"))
    assert not built.is_quota_exhausted(RuntimeError("rate limit, try again in 2s"))


def test_anthropic_overload_is_retried():
    """529 is Anthropic's "overloaded", which clears on its own."""
    class Overloaded(Exception):
        status_code = 529

    assert transport("anthropic").is_transient(Overloaded())


# -- Gemini ------------------------------------------------------------------


def gemini_contents(conversation=CONVERSATION):
    system, rest = split_system(conversation)
    return system, transport("gemini")._contents(rest)


def test_gemini_calls_the_assistant_role_model():
    _system, contents = gemini_contents()
    assert {c.role for c in contents} == {"user", "model"}


def test_gemini_turns_a_tool_call_into_a_function_call_part():
    _system, contents = gemini_contents()
    model_turn = [c for c in contents if c.role == "model"][0]
    call = [p.function_call for p in model_turn.parts if p.function_call][0]

    assert call.name == "write_file"
    assert call.args == {"path": "a.py", "content": "x"}


def test_gemini_answers_a_call_by_name_not_by_id():
    """Gemini matches a function_response to its call on the name, so the id
    has to be resolved back to one."""
    _system, contents = gemini_contents()
    responses = [
        p.function_response
        for c in contents
        for p in c.parts
        if getattr(p, "function_response", None)
    ]
    assert responses[0].name == "write_file"


def test_gemini_tools_carry_raw_json_schema():
    """parameters_json_schema takes JSON Schema, so a union survives -- the
    OpenAPI-derived `parameters` field is what could not express one."""
    tools = transport("gemini")._tools(TOOLS)
    schema = tools[0].function_declarations[0].parameters_json_schema
    assert schema["properties"]["timeout"]["type"] == ["integer", "string"]


def test_gemini_does_not_let_the_sdk_run_the_tools():
    """Automatic function calling would return only the final text and take
    the loop out of the loop entirely."""
    config = transport("gemini")._config("sys", TOOLS)
    assert config.automatic_function_calling.disable is True


def test_gemini_puts_the_system_prompt_in_the_config():
    config = transport("gemini")._config("be careful", TOOLS)
    assert config.system_instruction == "be careful"


def test_the_gemini_daily_cap_is_not_retried():
    """It never says "per day"; the daily part is only in the quotaId."""
    built = transport("gemini")
    exc = RuntimeError(
        "429 RESOURCE_EXHAUSTED: You exceeded your current quota. quotaId: "
        "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    )
    assert built.is_quota_exhausted(exc)


# -- the OpenAI-protocol pair ------------------------------------------------


def test_groq_and_openai_are_separate_classes_on_separate_sdks():
    import groq
    import openai

    assert isinstance(transport("groq")._client, groq.Groq)
    assert isinstance(transport("openai")._client, openai.OpenAI)


def test_a_rejected_generation_is_salvaged_rather_than_lost():
    """Groq validates the model's arguments server-side and 400s the whole
    generation, but returns what it rejected."""
    from agent.providers._openai_shape import failed_generation_text

    class Rejected(Exception):
        body = {
            "error": {
                "code": "tool_use_failed",
                "failed_generation": '<function=run_shell>{"command": "ls"}',
            }
        }

    assert "run_shell" in failed_generation_text(Rejected())


def test_only_a_schema_rejection_is_salvaged():
    """A bad key or an unknown model must still surface as an error."""
    from agent.providers._openai_shape import failed_generation_text

    class BadKey(Exception):
        body = {"error": {"code": "invalid_api_key", "message": "nope"}}

    assert failed_generation_text(BadKey()) == ""


def test_provider_specific_fields_on_a_tool_call_are_handed_back():
    """Gemini 3 rejects turn two without its thought_signature."""
    from agent.providers._openai_shape import passthrough_fields

    assert passthrough_fields({"id": "1", "function": {}, "thought_signature": "abc"}) == {
        "thought_signature": "abc"
    }


def test_a_transcript_from_another_provider_is_cleaned_before_sending():
    """The transcript is canonical and may hold fields another transport
    wrote; sending them to Groq would 400."""
    from agent.providers._openai_shape import clean_messages

    cleaned = clean_messages(
        [{"role": "user", "content": "hi", "thought_signature": "x", "extra": 1}]
    )
    assert cleaned == [{"role": "user", "content": "hi"}]


def test_block_content_is_flattened_for_the_openai_protocol():
    from agent.providers._openai_shape import clean_messages

    cleaned = clean_messages([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
    assert isinstance(cleaned[0]["content"], str)


# -- the canonical format ----------------------------------------------------


def test_no_transport_mutates_the_conversation_it_is_given():
    """The loop reuses the list across turns and the transcript is written
    from it, so a transport that edited it in place would corrupt both."""
    before = json.dumps(CONVERSATION, sort_keys=True)

    system, rest = split_system(CONVERSATION)
    transport("anthropic")._messages(rest)
    transport("gemini")._contents(rest)

    assert json.dumps(CONVERSATION, sort_keys=True) == before


# -- the loop, driven through a transport ------------------------------------


class FakeTransport:
    """A transport that answers from a script and records what it was sent."""

    provider = "fake"

    def __init__(self, *completions, fail_with=None):
        self._completions = list(completions)
        self.fail_with = fail_with
        self.calls = []
        self.attempts = 0

    def complete(self, *, model, messages, tools, stream=False, on_text=None):
        self.attempts += 1
        self.calls.append([dict(m) for m in messages])
        if self.fail_with is not None:
            raise self.fail_with
        return self._completions.pop(0) if self._completions else Completion()

    def list_models(self):
        return ["fake-1"]

    def is_transient(self, exc):
        return True

    def is_quota_exhausted(self, exc):
        return False


def completion_calling(name, **arguments):
    return Completion(tool_calls=[normalize_tool_call(name, arguments)])


def test_the_loop_runs_on_a_transport(tmp_path):
    from agent.loop import agent_loop
    from agent.sandbox import LocalExecutor

    client = FakeTransport(
        completion_calling("write_file", path="a.py", content="print(1)"),
        completion_calling("task_complete", summary="done"),
    )
    result = agent_loop("make a file", LocalExecutor(tmp_path), client=client, model="m")

    assert result.status == "complete"
    assert (tmp_path / "a.py").read_text() == "print(1)"


def test_the_transport_decides_what_is_retryable(tmp_path):
    """Each SDK has its own exception types, so the loop must not classify."""
    from agent.loop import MAX_LLM_RETRIES, agent_loop
    from agent.sandbox import LocalExecutor

    class NeverRetry(FakeTransport):
        def is_transient(self, exc):
            return False

    client = NeverRetry(fail_with=RuntimeError("boom"))
    agent_loop("go", LocalExecutor(tmp_path), client=client, model="m")
    assert client.attempts == 1, "a transport saying no must stop the retries"

    retrying = FakeTransport(fail_with=RuntimeError("boom"))
    agent_loop("go", LocalExecutor(tmp_path), client=retrying, model="m")
    assert retrying.attempts == MAX_LLM_RETRIES


def test_an_exhausted_quota_stops_immediately(tmp_path):
    from agent.loop import agent_loop
    from agent.sandbox import LocalExecutor

    class Exhausted(FakeTransport):
        def is_quota_exhausted(self, exc):
            return True

    client = Exhausted(fail_with=RuntimeError("no credit"))
    result = agent_loop("go", LocalExecutor(tmp_path), client=client, model="m")

    assert client.attempts == 1
    assert result.status == "error"


def test_the_transcript_stays_openai_shaped_whatever_ran(tmp_path):
    """It is what makes /provider switchable mid-conversation, and a session
    recorded on one provider resumable on another."""
    from agent.loop import agent_loop
    from agent.sandbox import LocalExecutor

    client = FakeTransport(
        completion_calling("write_file", path="a.py", content="x"),
        completion_calling("task_complete", summary="done"),
    )
    result = agent_loop("go", LocalExecutor(tmp_path), client=client, model="m")

    roles = {m["role"] for m in result.messages}
    assert roles <= {"system", "user", "assistant", "tool"}
    assistant = [m for m in result.messages if m.get("tool_calls")][0]
    assert isinstance(assistant["tool_calls"][0]["function"]["arguments"], str)


def test_a_transcript_written_by_one_provider_replays_on_another(tmp_path):
    """The real portability check: take a finished conversation and hand it to
    every other transport's converter without it raising."""
    from agent.loop import agent_loop
    from agent.sandbox import LocalExecutor

    client = FakeTransport(
        completion_calling("write_file", path="a.py", content="x"),
        completion_calling("task_complete", summary="done"),
    )
    messages = agent_loop("go", LocalExecutor(tmp_path), client=client, model="m").messages

    _system, rest = split_system(messages)
    assert transport("anthropic")._messages(rest)
    assert transport("gemini")._contents(rest)
    from agent.providers._openai_shape import clean_messages

    assert clean_messages(messages)


# -- Gemini's thought_signature ----------------------------------------------
#
# Observed live, twice. Gemini 3 rejects turn two with "Function call is missing
# a thought_signature in functionCall parts" unless the signature it issued
# comes back untouched. It lives on the Part, not on the FunctionCall, and it
# arrives as raw bytes, so it has to survive a JSON transcript to be resumable.


class _FunctionCall:
    def __init__(self, name, args, ident=None):
        self.name, self.args, self.id = name, args, ident


class _Part:
    def __init__(self, function_call=None, text=None, thought_signature=None):
        self.function_call = function_call
        self.text = text
        self.thought_signature = thought_signature
        self.function_response = None


class _Candidate:
    def __init__(self, parts):
        self.content = type("C", (), {"parts": parts})()


class _Response:
    def __init__(self, parts):
        self.candidates = [_Candidate(parts)]
        self.usage_metadata = None


def test_a_gemini_signature_survives_onto_the_tool_call():
    from agent.providers.gemini_api import _from_response

    completion = _from_response(
        _Response([_Part(_FunctionCall("run_shell", {"command": "ls"}), thought_signature=b"\x00\x01sig")])
    )
    assert completion.tool_calls[0]["thought_signature"]


def test_a_signature_is_json_safe_so_a_session_can_be_resumed():
    """Raw bytes would make the transcript unwritable, and resume would lose
    the signature the next turn requires."""
    from agent.providers.gemini_api import _from_response

    completion = _from_response(
        _Response([_Part(_FunctionCall("t", {}), thought_signature=b"\xff\xfe binary")])
    )
    json.dumps(completion.tool_calls)  # must not raise


def test_a_signature_goes_back_out_on_the_part_it_came_from():
    from agent.providers.gemini_api import _from_response

    raw = b"\x00\x01sig"
    completion = _from_response(_Response([_Part(_FunctionCall("t", {}), thought_signature=raw)]))

    contents = transport("gemini")._contents(
        [{"role": "user", "content": "go"},
         {"role": "assistant", "tool_calls": completion.tool_calls}]
    )
    part = [p for c in contents if c.role == "model" for p in c.parts if p.function_call][0]
    assert part.thought_signature == raw, "the exact bytes must come back"


def test_a_tool_call_without_a_signature_is_left_alone():
    """Not every model issues one; inventing a field would be its own 400."""
    from agent.providers.gemini_api import _from_response

    completion = _from_response(_Response([_Part(_FunctionCall("t", {}))]))
    assert "thought_signature" not in completion.tool_calls[0]
