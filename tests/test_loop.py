"""Loop tests, driven by a scripted fake client -- no API key, no network."""

from __future__ import annotations

import json

import pytest

from agent.loop import agent_loop, failed_generation_text
from agent.sandbox import LocalExecutor
from tests.fake_llm import ExplodingClient, FakeClient, sdk_error, tool_call, turn


@pytest.fixture
def executor(tmp_path):
    return LocalExecutor(tmp_path)


def run(responses, executor, **kwargs):
    client = FakeClient(responses)
    result = agent_loop("do the thing", executor, client=client, **kwargs)
    return result, client


# -- termination ------------------------------------------------------------


def test_task_complete_ends_the_loop(executor):
    result, _ = run([turn(tool_call("task_complete", {"summary": "all done"}))], executor)
    assert result.status == "complete"
    assert result.ok
    assert result.summary == "all done"
    assert result.steps == 1


def test_loop_continues_until_task_complete(executor):
    result, client = run(
        [
            turn(tool_call("run_shell", {"command": "echo one"})),
            turn(tool_call("run_shell", {"command": "echo two"})),
            turn(tool_call("task_complete", {"summary": "done"})),
        ],
        executor,
    )
    assert result.status == "complete"
    assert result.steps == 3
    assert result.tool_calls == 3
    assert len(client.calls) == 3


def test_no_tool_calls_stops_the_loop(executor):
    result, _ = run([turn(content="I think we're done here")], executor)
    assert result.status == "stopped"
    assert result.summary == "I think we're done here"


# -- tool calls emitted as text --------------------------------------------


def test_a_tool_call_written_as_text_still_runs(executor, tmp_path):
    """The failure from the first real run: the model wrote the call as prose,
    the loop saw no tool calls and stopped at step 1."""
    result, _ = run(
        [
            turn(content='<function/write_file {"path": "made.txt", "content": "hi"} </function>'),
            turn(tool_call("task_complete", {"summary": "done"})),
        ],
        executor,
    )
    assert result.status == "complete"
    assert result.recovered_tool_calls == 1
    assert (tmp_path / "made.txt").read_text() == "hi"


def test_recovered_call_is_rewritten_into_the_transcript(executor):
    """The history should show the model a correctly-shaped version of its own
    call, not the malformed text that would reinforce the bad format."""
    result, _ = run(
        [
            turn(content='<function=run_shell>{"command": "echo hi"}</function>'),
            turn(tool_call("task_complete", {"summary": "done"})),
        ],
        executor,
    )
    first = result.messages[2]  # system, user, assistant
    assert first["role"] == "assistant"
    assert first["content"] == ""
    assert first["tool_calls"][0]["function"]["name"] == "run_shell"
    assert first["tool_calls"][0]["id"]
    answered = [m["tool_call_id"] for m in result.messages if m["role"] == "tool"]
    assert first["tool_calls"][0]["id"] in answered


def test_plain_text_with_no_call_still_stops(executor):
    """Recovery must not turn ordinary prose into a phantom tool call."""
    result, _ = run([turn(content="I'll go ahead and list the files now.")], executor)
    assert result.status == "stopped"
    assert result.recovered_tool_calls == 0


def test_max_iterations_is_respected(executor):
    responses = [turn(tool_call("run_shell", {"command": "echo x"})) for _ in range(10)]
    result, client = run(responses, executor, max_iterations=3)
    assert result.status == "max_iterations_reached"
    assert result.steps == 3
    assert len(client.calls) == 3  # did not overrun the budget


def test_llm_error_returns_error_status(executor):
    client = ExplodingClient(RuntimeError("invalid api key"))
    result = agent_loop("task", executor, client=client)
    assert result.status == "error"
    assert "invalid api key" in result.summary


TOOL_USE_FAILED = {
    "error": {
        "code": "tool_use_failed",
        "message": "Failed to call a function.",
        "failed_generation": (
            '<function=write_file>{"path": "salvaged.txt", '
            '"content": "recovered"}</function>'
        ),
    }
}


def test_server_side_schema_rejection_is_recovered(executor, tmp_path):
    """Groq validates tool args against our schema and 400s on a mismatch,
    killing the run. The body carries the rejected generation, so the call can
    be salvaged from it."""
    client = FakeClient(
        [sdk_error(TOOL_USE_FAILED), turn(tool_call("task_complete", {"summary": "done"}))]
    )
    result = agent_loop("task", executor, client=client)

    assert result.status == "complete"
    assert result.recovered_tool_calls == 1
    assert (tmp_path / "salvaged.txt").read_text() == "recovered"


@pytest.mark.parametrize(
    "body",
    [
        TOOL_USE_FAILED["error"],  # unwrapped, as openai>=2 exposes it
        TOOL_USE_FAILED,  # full envelope, as the wire carries it
    ],
)
def test_both_error_body_shapes_are_understood(body):
    """Neither shape may silently disable recovery."""

    class Exc(Exception):
        pass

    exc = Exc()
    exc.body = body
    assert "salvaged.txt" in (failed_generation_text(exc) or "")


def test_the_exact_generation_groq_rejected_live():
    """Verbatim from a real run -- note the `[]` between name and arguments."""
    payload = {
        "error": {
            "code": "tool_use_failed",
            "message": "Failed to call a function.",
            "failed_generation": (
                '<function=task_complete[]{"summary": "printed the first 20 primes"}'
                "</function>"
            ),
        }
    }
    text = failed_generation_text(sdk_error(payload))
    assert text is not None
    from agent.tools import extract_tool_calls_from_text

    calls = extract_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "task_complete"


def test_unrelated_400s_are_not_swallowed(executor):
    """Only tool_use_failed carries a recoverable generation."""
    err = sdk_error({"error": {"code": "invalid_api_key", "message": "bad key"}})
    result = agent_loop("task", executor, client=ExplodingClient(err))
    assert result.status == "error"


def test_transient_errors_are_retried(executor, monkeypatch):
    monkeypatch.setattr("agent.loop.time.sleep", lambda _: None)
    client = ExplodingClient(RuntimeError("rate limit exceeded (429)"))
    result = agent_loop("task", executor, client=client)
    assert result.status == "error"
    assert client.attempts == 3  # retried, then gave up


# -- the loop actually does work -------------------------------------------


def test_agent_can_write_and_verify_a_file(executor, tmp_path):
    result, _ = run(
        [
            turn(tool_call("write_file", {"path": "hello.py", "content": "print('hi')"})),
            turn(tool_call("run_shell", {"command": "python hello.py"})),
            turn(tool_call("task_complete", {"summary": "wrote and ran hello.py"})),
        ],
        executor,
    )
    assert result.status == "complete"
    assert (tmp_path / "hello.py").read_text() == "print('hi')"


# -- transcript shape -------------------------------------------------------


def test_every_tool_call_gets_a_matching_tool_message(executor):
    """The API rejects the next request if any tool_call id is unanswered."""
    result, _ = run(
        [
            turn(
                tool_call("run_shell", {"command": "echo a"}, call_id="c1"),
                tool_call("run_shell", {"command": "echo b"}, call_id="c2"),
            ),
            turn(tool_call("task_complete", {"summary": "done"}, call_id="c3")),
        ],
        executor,
    )
    assert result.status == "complete"

    requested = [
        call["id"]
        for msg in result.messages
        if msg["role"] == "assistant"
        for call in msg.get("tool_calls", [])
    ]
    answered = [m["tool_call_id"] for m in result.messages if m["role"] == "tool"]
    assert requested == ["c1", "c2", "c3"]
    assert sorted(answered) == sorted(requested)


def test_tool_calls_without_ids_get_synthesized_ones(executor):
    """Open models sometimes omit the id. An empty id breaks the next request."""
    result, _ = run(
        [
            turn(tool_call("run_shell", {"command": "echo a"}, call_id="")),
            turn(tool_call("task_complete", {"summary": "done"})),
        ],
        executor,
    )
    ids = [
        call["id"]
        for msg in result.messages
        if msg["role"] == "assistant"
        for call in msg.get("tool_calls", [])
    ]
    assert all(i for i in ids), "every tool call must carry a non-empty id"
    answered = [m["tool_call_id"] for m in result.messages if m["role"] == "tool"]
    assert sorted(answered) == sorted(ids)


def test_parallel_tool_calls_all_execute(executor, tmp_path):
    result, _ = run(
        [
            turn(
                tool_call("write_file", {"path": "x.txt", "content": "1"}, call_id="a"),
                tool_call("write_file", {"path": "y.txt", "content": "2"}, call_id="b"),
            ),
            turn(tool_call("task_complete", {"summary": "done"})),
        ],
        executor,
    )
    assert result.status == "complete"
    assert (tmp_path / "x.txt").read_text() == "1"
    assert (tmp_path / "y.txt").read_text() == "2"


# -- task_complete batched with the work ------------------------------------


def test_calls_after_task_complete_in_the_same_batch_still_run(executor, tmp_path):
    """Returning mid-batch would skip them and leave their ids unanswered."""
    result, _ = run(
        [
            turn(
                tool_call("task_complete", {"summary": "done"}, call_id="a"),
                tool_call("write_file", {"path": "after.txt", "content": "x"}, call_id="b"),
            ),
            turn(tool_call("task_complete", {"summary": "really done"})),
        ],
        executor,
    )
    assert (tmp_path / "after.txt").read_text() == "x"
    answered = [m["tool_call_id"] for m in result.messages if m["role"] == "tool"]
    assert "a" in answered and "b" in answered


def test_batched_completion_is_deferred_until_results_are_seen(executor):
    """The model cannot have verified output it had not received yet."""
    result, _ = run(
        [
            turn(
                tool_call("run_shell", {"command": "echo hi"}, call_id="a"),
                tool_call("task_complete", {"summary": "verified the output"}, call_id="b"),
            ),
            turn(tool_call("task_complete", {"summary": "now actually verified"})),
        ],
        executor,
    )
    assert result.status == "complete"
    assert result.steps == 2  # did not finish on the batched turn
    assert result.summary == "now actually verified"
    deferred = [m for m in result.messages if m.get("tool_call_id") == "b"]
    assert "Not recorded yet" in deferred[0]["content"]


def test_solo_task_complete_is_accepted_immediately(executor):
    result, _ = run(
        [
            turn(tool_call("run_shell", {"command": "echo hi"})),
            turn(tool_call("task_complete", {"summary": "done"})),
        ],
        executor,
    )
    assert result.status == "complete"
    assert result.steps == 2


def test_persistent_batching_still_terminates(executor):
    """A model that always batches must not loop until max_iterations."""
    batched = turn(
        tool_call("run_shell", {"command": "echo hi"}),
        tool_call("task_complete", {"summary": "done"}),
    )
    result, client = run([batched for _ in range(10)], executor, max_iterations=8)
    assert result.status == "complete"
    assert len(client.calls) == 3  # deferred twice, then accepted


# -- malformed calls do not kill the run ------------------------------------


def test_malformed_tool_call_is_fed_back_and_the_run_survives(executor):
    result, _ = run(
        [
            turn(tool_call("run_shell", "{not json at all")),
            turn(tool_call("task_complete", {"summary": "recovered"})),
        ],
        executor,
    )
    assert result.status == "complete"
    assert result.tool_errors == 1
    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    assert tool_msgs[0]["content"].startswith("Error:")


def test_hallucinated_tool_name_is_fed_back(executor):
    result, _ = run(
        [
            turn(tool_call("browse_web", {"url": "http://x"})),
            turn(tool_call("task_complete", {"summary": "ok"})),
        ],
        executor,
    )
    assert result.status == "complete"
    assert result.tool_errors == 1


def test_task_complete_with_malformed_arguments_still_completes(executor):
    result, _ = run([turn(tool_call("task_complete", "{broken"))], executor)
    assert result.status == "complete"


# -- conversation history ---------------------------------------------------


def test_history_is_continued_not_restarted(executor):
    prior = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first task"},
        {"role": "assistant", "content": "did it"},
    ]
    client = FakeClient([turn(tool_call("task_complete", {"summary": "ok"}))])
    result = agent_loop("second task", executor, client=client, history=prior)

    sent = client.calls[0]
    assert [m["content"] for m in sent[:3]] == ["sys", "first task", "did it"]
    assert sent[3]["content"] == "second task"
    assert result.status == "complete"


def test_history_is_copied_not_mutated(executor):
    """An interrupted turn must not leave the caller's transcript half-written
    with tool calls that were never answered."""
    prior = [{"role": "system", "content": "sys"}, {"role": "user", "content": "old"}]
    snapshot = [dict(m) for m in prior]
    client = FakeClient([turn(tool_call("task_complete", {"summary": "ok"}))])
    agent_loop("new", executor, client=client, history=prior)
    assert prior == snapshot


def test_history_without_a_system_prompt_gets_one(executor):
    client = FakeClient([turn(tool_call("task_complete", {"summary": "ok"}))])
    agent_loop(
        "task", executor, client=client, history=[{"role": "user", "content": "old"}]
    )
    assert client.calls[0][0]["role"] == "system"


# -- metrics ----------------------------------------------------------------


def test_usage_is_accumulated_across_steps(executor):
    result, _ = run(
        [
            turn(tool_call("run_shell", {"command": "echo x"})),
            turn(tool_call("task_complete", {"summary": "done"})),
        ],
        executor,
    )
    assert result.usage["api_calls"] == 2
    assert result.usage["prompt_tokens"] == 200
    assert result.usage["completion_tokens"] == 40
    assert result.metrics()["status"] == "complete"


def test_events_are_emitted(executor):
    events = []
    client = FakeClient(
        [
            turn(tool_call("run_shell", {"command": "echo x"})),
            turn(tool_call("task_complete", {"summary": "done"})),
        ]
    )
    agent_loop(
        "task",
        executor,
        client=client,
        on_event=lambda name, payload: events.append(name),
    )
    assert "step_start" in events
    assert "tool_call" in events
    assert "tool_result" in events
    assert "complete" in events


# -- extension hook (stretch goal: spawn_subagent) --------------------------


def test_extra_tool_handlers_are_dispatched(executor):
    seen = {}

    def handler(args):
        seen.update(args)
        return "subagent says: done"

    client = FakeClient(
        [
            turn(tool_call("spawn_subagent", {"task_description": "refactor"})),
            turn(tool_call("task_complete", {"summary": "done"})),
        ]
    )
    result = agent_loop(
        "task",
        executor,
        client=client,
        extra_tool_handlers={"spawn_subagent": handler},
    )
    assert result.status == "complete"
    assert seen == {"task_description": "refactor"}
    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    assert tool_msgs[0]["content"] == "subagent says: done"


def test_failing_extra_handler_does_not_kill_the_loop(executor):
    def handler(args):
        raise RuntimeError("subagent exploded")

    client = FakeClient(
        [
            turn(tool_call("spawn_subagent", {"task_description": "x"})),
            turn(tool_call("task_complete", {"summary": "done"})),
        ]
    )
    result = agent_loop(
        "task", executor, client=client, extra_tool_handlers={"spawn_subagent": handler}
    )
    assert result.status == "complete"
    assert result.tool_errors == 1
