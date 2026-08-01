"""Context trimming, spend ceilings, error classification, container limits."""

from __future__ import annotations

import pytest

from agent.loop import (
    _is_transient,
    agent_loop,
    estimate_tokens,
    is_context_error,
    trim_messages,
)
from agent.sandbox import DockerExecutor, LocalExecutor
from tests.fake_llm import FakeClient, sdk_error, tool_call, turn
from tests.test_sandbox import requires_docker


@pytest.fixture
def executor(tmp_path):
    return LocalExecutor(tmp_path)


def conversation(turns: int, size: int = 400) -> list[dict]:
    """A transcript shaped like a real one: assistant tool calls answered by
    tool results, which is what makes trimming non-trivial."""
    messages: list[dict] = [{"role": "system", "content": "system prompt"}]
    for i in range(turns):
        messages.append({"role": "user", "content": f"task {i}"})
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {"name": "run_shell", "arguments": '{"command":"ls"}'},
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * size})
    return messages


# -- trimming ---------------------------------------------------------------


def test_short_transcripts_are_untouched():
    messages = conversation(2)
    trimmed, dropped = trim_messages(messages, budget=100_000)
    assert dropped == 0
    assert trimmed == messages


def test_trimming_drops_oldest_and_keeps_recent():
    messages = conversation(30)
    trimmed, dropped = trim_messages(messages, budget=2_000)
    assert dropped > 0
    assert len(trimmed) < len(messages)
    assert estimate_tokens(trimmed) <= 2_000 * 1.5
    # The newest turn must survive -- it is what the model is answering.
    assert trimmed[-1]["content"].startswith("x") or "task 29" in str(trimmed)


def test_system_prompt_always_survives():
    trimmed, _ = trim_messages(conversation(40), budget=500)
    assert trimmed[0]["role"] == "system"


def test_tool_calls_and_their_results_are_never_split():
    """A tool_call with no matching tool message (or vice versa) is rejected by
    the API outright -- this is the failure trimming most easily causes."""
    trimmed, _ = trim_messages(conversation(30), budget=1_500)

    requested = {
        call["id"]
        for m in trimmed
        if m["role"] == "assistant"
        for call in m.get("tool_calls", [])
    }
    answered = {m["tool_call_id"] for m in trimmed if m["role"] == "tool"}
    assert requested == answered


def test_no_orphan_tool_message_at_the_front():
    for budget in (300, 700, 1_100, 2_500):
        trimmed, _ = trim_messages(conversation(25), budget=budget)
        body = [m for m in trimmed if m["role"] != "system"]
        assert not body or body[0]["role"] != "tool"


def test_the_model_is_told_that_history_was_dropped():
    trimmed, dropped = trim_messages(conversation(30), budget=1_000)
    assert dropped > 0
    assert any("dropped" in (m.get("content") or "") for m in trimmed)


def test_trimming_keeps_the_transcript_usable_end_to_end(executor):
    """Trim mid-run, then keep going: the trimmed transcript must be valid."""
    responses = [turn(tool_call("run_shell", {"command": "echo x"})) for _ in range(6)]
    responses.append(turn(tool_call("task_complete", {"summary": "done"})))
    client = FakeClient(responses)

    result = agent_loop(
        "task", executor, client=client, context_budget=300, max_iterations=10
    )
    assert result.status == "complete"
    for sent in client.calls:
        requested = {
            c["id"] for m in sent if m["role"] == "assistant" for c in m.get("tool_calls", [])
        }
        answered = {m["tool_call_id"] for m in sent if m["role"] == "tool"}
        assert requested == answered, "sent a transcript the API would reject"


def test_full_transcript_is_kept_even_when_trimmed(executor):
    """Trimming applies to what we send, not to what we record."""
    responses = [turn(tool_call("run_shell", {"command": "echo x"})) for _ in range(5)]
    responses.append(turn(tool_call("task_complete", {"summary": "done"})))
    client = FakeClient(responses)
    result = agent_loop("task", executor, client=client, context_budget=200)
    assert len(result.messages) > len(client.calls[-1])


# -- context_length_exceeded ------------------------------------------------


def test_context_error_is_recognised():
    assert is_context_error(
        sdk_error({"error": {"code": "context_length_exceeded", "message": "too long"}})
    )
    assert is_context_error(Exception("Please reduce the length of the messages"))
    assert not is_context_error(Exception("invalid api key"))


def test_a_context_rejection_triggers_a_harder_trim(executor):
    """Our estimate is approximate, so the server can still say no."""
    rejection = sdk_error(
        {"error": {"code": "context_length_exceeded", "message": "too long"}}
    )
    client = FakeClient([rejection, turn(tool_call("task_complete", {"summary": "ok"}))])
    result = agent_loop("task", executor, client=client, history=conversation(20))

    assert result.status == "complete"
    assert len(client.calls) == 2
    assert len(client.calls[1]) < len(client.calls[0])


# -- spend ceiling ----------------------------------------------------------


def test_token_budget_stops_the_run(executor):
    responses = [turn(tool_call("run_shell", {"command": "echo x"})) for _ in range(10)]
    client = FakeClient(responses)
    result = agent_loop(
        "task", executor, client=client, max_total_tokens=250, max_iterations=10
    )
    assert result.status == "budget_exhausted"
    assert len(client.calls) == 3  # 120 tokens per call; stops once over 250


def test_no_budget_means_no_ceiling(executor):
    client = FakeClient([turn(tool_call("task_complete", {"summary": "done"}))])
    result = agent_loop("task", executor, client=client)
    assert result.status == "complete"


# -- error classification ---------------------------------------------------


@pytest.mark.parametrize(
    "exc,expected",
    [
        (sdk_error({"error": {"code": "invalid_api_key"}}), False),  # typed 400
        (Exception("rate limit exceeded"), True),
        (Exception("503 service unavailable"), True),
        (Exception("something unexpected"), False),
    ],
)
def test_transient_classification(exc, expected):
    assert _is_transient(exc) is expected


def test_a_typed_400_is_not_retried(executor):
    """String-matching once made a 400 look retryable; status code cannot."""
    from tests.fake_llm import ExplodingClient

    client = ExplodingClient(sdk_error({"error": {"code": "invalid_request_error"}}))
    result = agent_loop("task", executor, client=client)
    assert result.status == "error"
    assert client.attempts == 1


# -- container limits -------------------------------------------------------


@requires_docker
def test_limits_are_applied_to_the_container():
    ex = DockerExecutor(image="python:3.11-slim", memory="512m", pids_limit=64, cpus="1")
    try:
        limit = ex.run_shell(
            "cat /sys/fs/cgroup/memory.max 2>/dev/null || "
            "cat /sys/fs/cgroup/memory/memory.limit_in_bytes"
        ).stdout.strip()
        assert limit and int(limit) <= 512 * 1024 * 1024
    finally:
        ex.close()


@requires_docker
def test_a_fork_bomb_cannot_take_down_the_host():
    """The pids limit is the difference between a failed command and a wedged
    Docker VM."""
    ex = DockerExecutor(image="python:3.11-slim", pids_limit=32, memory="256m")
    try:
        ex.run_shell("for i in $(seq 1 200); do sleep 30 & done", timeout=20)
        # The container survives and still answers.
        assert ex.run_shell("echo alive", timeout=15).stdout.strip() == "alive"
    finally:
        ex.close()


@requires_docker
def test_network_can_be_cut_off():
    ex = DockerExecutor(image="python:3.11-slim", network="none")
    try:
        result = ex.run_shell(
            "python -c \"import socket; socket.create_connection(('1.1.1.1', 80), 3)\"",
            timeout=20,
        )
        assert result.exit_code != 0
    finally:
        ex.close()


@requires_docker
def test_created_containers_are_labelled_for_cleanup():
    ex = DockerExecutor(image="python:3.11-slim")
    try:
        orphans = dict(DockerExecutor.list_orphans())
        assert orphans, "a container we created should be findable by label"
        assert all(age >= 0 for age in orphans.values())
    finally:
        ex.close()


@requires_docker
def test_sweep_leaves_young_containers_alone():
    """Another session's container must survive our startup sweep."""
    ex = DockerExecutor(image="python:3.11-slim")
    try:
        DockerExecutor.sweep_orphans(max_age_seconds=3600)
        assert ex.run_shell("echo alive").stdout.strip() == "alive"
    finally:
        ex.close()
