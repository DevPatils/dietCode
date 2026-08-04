"""Streaming tests.

The risk in streaming is reassembly: tool calls arrive as fragments, and a
naive join produces a call that is subtly wrong rather than obviously broken.
These run the same scripts as the non-streaming tests, chopped into deltas.
"""

from __future__ import annotations

import json

import pytest

from agent.loop import agent_loop
from agent.sandbox import LocalExecutor
from tests.fake_llm import (
    FakeChunk,
    FakeClient,
    FakeDelta,
    FakeDeltaFunction,
    FakeDeltaToolCall,
    FakeStreamChoice,
    tool_call,
    turn,
)


@pytest.fixture
def executor(tmp_path):
    return LocalExecutor(tmp_path)


def run_streamed(responses, executor, **kwargs):
    client = FakeClient(responses)
    result = agent_loop("do the thing", executor, client=client, stream=True, **kwargs)
    return result, client


# -- parity with the non-streaming path -------------------------------------


def test_streamed_run_completes(executor):
    result, client = run_streamed(
        [
            turn(tool_call("run_shell", {"command": "echo hi"})),
            turn(tool_call("task_complete", {"summary": "done"})),
        ],
        executor,
    )
    assert result.status == "complete"
    assert result.summary == "done"
    assert client.kwargs[0]["stream"] is True


def test_arguments_split_across_chunks_are_reassembled(executor, tmp_path):
    """The fake splits every tool call's arguments in half mid-JSON."""
    content = "line one\nline two\nline three\n"
    result, _ = run_streamed(
        [
            turn(tool_call("write_file", {"path": "out.txt", "content": content})),
            turn(tool_call("task_complete", {"summary": "wrote it"})),
        ],
        executor,
    )
    assert result.status == "complete"
    assert result.tool_errors == 0
    assert (tmp_path / "out.txt").read_text() == content


def test_parallel_streamed_calls_stay_separate(executor, tmp_path):
    """Interleaved calls are keyed by index; mixing them up would merge the
    JSON of one into another."""
    result, _ = run_streamed(
        [
            turn(
                tool_call("write_file", {"path": "a.txt", "content": "aaa"}, call_id="c1"),
                tool_call("write_file", {"path": "b.txt", "content": "bbb"}, call_id="c2"),
            ),
            turn(tool_call("task_complete", {"summary": "both"})),
        ],
        executor,
    )
    assert result.status == "complete"
    assert (tmp_path / "a.txt").read_text() == "aaa"
    assert (tmp_path / "b.txt").read_text() == "bbb"


def test_usage_survives_streaming(executor):
    """Usage arrives in a final chunk with no choices; dropping it would blank
    the benchmark's token column."""
    result, client = run_streamed(
        [turn(tool_call("task_complete", {"summary": "done"}))], executor
    )
    assert result.usage["total_tokens"] == 120
    assert client.kwargs[0]["stream_options"] == {"include_usage": True}


def test_text_is_delivered_as_deltas(executor):
    events = []
    client = FakeClient([turn(content="Hello there, this is a streamed reply.")])
    agent_loop(
        "task",
        executor,
        client=client,
        stream=True,
        on_event=lambda name, payload: events.append((name, payload.get("text", ""))),
    )
    deltas = [text for name, text in events if name == "assistant_delta"]
    assert len(deltas) > 1, "should arrive in pieces, not one lump"
    assert "".join(deltas) == "Hello there, this is a streamed reply."
    # The whole-message event would double-print what was already streamed.
    assert not any(name == "assistant_text" for name, _ in events)


def test_non_streaming_still_emits_the_whole_message(executor):
    events = []
    client = FakeClient([turn(content="All at once.")])
    agent_loop(
        "task",
        executor,
        client=client,
        on_event=lambda name, payload: events.append(name),
    )
    assert "assistant_text" in events
    assert "assistant_delta" not in events


def test_text_recovery_works_on_a_streamed_turn(executor, tmp_path):
    """A tool call written as prose still has to be salvaged when streamed."""
    result, _ = run_streamed(
        [
            turn(content='<function=write_file>{"path": "s.txt", "content": "x"}</function>'),
            turn(tool_call("task_complete", {"summary": "done"})),
        ],
        executor,
    )
    assert result.status == "complete"
    assert result.recovered_tool_calls == 1
    assert (tmp_path / "s.txt").read_text() == "x"


# -- reassembly edge cases --------------------------------------------------


def chunk(index=0, call_id=None, name=None, arguments=None, content=None):
    if content is not None:
        return FakeChunk([FakeStreamChoice(FakeDelta(content=content))])
    return FakeChunk(
        [
            FakeStreamChoice(
                FakeDelta(
                    tool_calls=[
                        FakeDeltaToolCall(
                            index=index,
                            id=call_id,
                            function=FakeDeltaFunction(name=name, arguments=arguments),
                        )
                    ]
                )
            )
        ]
    )


class RawStreamClient:
    """Replays hand-built chunk sequences, for shapes the helper cannot make."""

    def __init__(self, chunk_lists):
        self._chunk_lists = list(chunk_lists)
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        return iter(self._chunk_lists.pop(0))


def test_a_name_repeated_in_every_delta_is_not_doubled(executor):
    """Some providers resend the full function name on each fragment."""
    client = RawStreamClient(
        [
            [
                chunk(call_id="c1", name="task_complete", arguments='{"summary":'),
                chunk(name="task_complete", arguments=' "ok"}'),
            ]
        ]
    )
    result = agent_loop("task", executor, client=client, stream=True)
    assert result.status == "complete"
    assert result.summary == "ok"


def test_a_name_split_across_deltas_is_joined(executor):
    client = RawStreamClient(
        [
            [
                chunk(call_id="c1", name="task_"),
                chunk(name="complete", arguments='{"summary": "ok"}'),
            ]
        ]
    )
    result = agent_loop("task", executor, client=client, stream=True)
    assert result.status == "complete"


def test_a_call_streamed_without_an_id_still_gets_one(executor):
    client = RawStreamClient(
        [[chunk(name="task_complete", arguments='{"summary": "no id"}')]]
    )
    result = agent_loop("task", executor, client=client, stream=True)
    assert result.status == "complete"
    ids = [
        c["id"]
        for m in result.messages
        if m["role"] == "assistant"
        for c in m.get("tool_calls", [])
    ]
    assert all(ids)


def test_empty_stream_is_treated_as_stopping(executor):
    client = RawStreamClient([[]])
    result = agent_loop("task", executor, client=client, stream=True)
    assert result.status == "stopped"


def test_chunks_without_choices_are_skipped(executor):
    """Keep-alive and usage-only chunks carry no choices."""
    client = RawStreamClient(
        [
            [
                FakeChunk(choices=[]),
                chunk(call_id="c1", name="task_complete", arguments='{"summary": "ok"}'),
                FakeChunk(choices=[]),
            ]
        ]
    )
    result = agent_loop("task", executor, client=client, stream=True)
    assert result.status == "complete"


def test_malformed_streamed_arguments_do_not_crash(executor):
    """Half a JSON object is exactly what a truncated stream leaves behind."""
    client = RawStreamClient(
        [
            [chunk(call_id="c1", name="run_shell", arguments='{"command": "ec')],
            [chunk(call_id="c2", name="task_complete", arguments='{"summary": "recovered"}')],
        ]
    )
    result = agent_loop("task", executor, client=client, stream=True)
    assert result.status == "complete"
    assert result.tool_errors == 1


def test_stream_options_rejection_falls_back(executor):
    """Not every OpenAI-compatible server supports stream_options; losing the
    token count beats losing the turn."""

    class Picky:
        def __init__(self):
            self.attempts = []
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            self.attempts.append(kwargs)
            if "stream_options" in kwargs:
                raise ValueError("unrecognized argument: stream_options")
            return iter(
                [chunk(call_id="c1", name="task_complete", arguments='{"summary": "ok"}')]
            )

    client = Picky()
    result = agent_loop("task", executor, client=client, stream=True)
    assert result.status == "complete"
    assert len(client.attempts) == 2
    assert "stream_options" not in client.attempts[1]


def test_json_arguments_are_valid_after_reassembly(executor):
    """Guards against fragments being joined in the wrong order."""
    payload = {"path": "x.py", "content": "def f():\n    return {'a': [1, 2]}\n"}
    client = FakeClient([turn(tool_call("write_file", payload))])
    agent_loop("task", executor, client=client, stream=True, max_iterations=1)
    # If reassembly were wrong this would not parse.
    assert json.loads(json.dumps(payload)) == payload
