"""Sub-agent delegation and project context files.

The point of a sub-agent is context isolation: the child's transcript must not
reach the parent, or delegating costs exactly as much as doing the work inline
and there is nothing to measure.
"""

from __future__ import annotations

import pytest

from agent.loop import (
    SYSTEM_PROMPT,
    agent_loop,
    load_project_context,
    with_project_context,
)
from agent.sandbox import LocalExecutor
from agent.subagent import SPAWN_TOOL, make_spawn_handler
from agent.tools import TOOLS
from tests.fake_llm import FakeClient, tool_call, turn


@pytest.fixture
def executor(tmp_path):
    return LocalExecutor(tmp_path)


def spawning_client(child_script, parent_script):
    """One client serving both agents: the parent runs first, then the child."""
    return FakeClient(parent_script + child_script)


# -- isolation --------------------------------------------------------------


def test_only_the_summary_comes_back(executor):
    """The child does three steps; the parent must receive one sentence."""
    client = FakeClient(
        [
            turn(tool_call("spawn_subagent", {"task": "survey the repo"})),
            # child
            turn(tool_call("run_shell", {"command": "echo scanning"})),
            turn(tool_call("run_shell", {"command": "echo more scanning"})),
            turn(tool_call("task_complete", {"summary": "12 python files, all under src/"})),
            # parent again
            turn(tool_call("task_complete", {"summary": "done"})),
        ]
    )
    handler = make_spawn_handler(executor, client, "test-model")
    result = agent_loop(
        "big task",
        executor,
        client=client,
        tools=[*TOOLS, SPAWN_TOOL],
        extra_tool_handlers={"spawn_subagent": handler},
    )

    assert result.status == "complete"
    tool_msgs = [m["content"] for m in result.messages if m["role"] == "tool"]
    assert tool_msgs[0] == "12 python files, all under src/"
    assert "scanning" not in "".join(tool_msgs), "the child's transcript leaked"


def test_the_child_cannot_see_the_parent_conversation(executor):
    client = FakeClient(
        [
            turn(tool_call("spawn_subagent", {"task": "isolated errand"})),
            turn(tool_call("task_complete", {"summary": "ok"})),
            turn(tool_call("task_complete", {"summary": "done"})),
        ]
    )
    handler = make_spawn_handler(executor, client, "test-model")
    agent_loop(
        "parent task with secret detail",
        executor,
        client=client,
        tools=[*TOOLS, SPAWN_TOOL],
        extra_tool_handlers={"spawn_subagent": handler},
    )

    child_request = client.calls[1]
    flat = " ".join(str(m.get("content", "")) for m in child_request)
    assert "isolated errand" in flat
    assert "secret detail" not in flat


def test_the_child_shares_the_filesystem(executor, tmp_path):
    """Isolated context, same files -- that is the whole shape of it."""
    client = FakeClient(
        [
            turn(tool_call("spawn_subagent", {"task": "write the file"})),
            turn(tool_call("write_file", {"path": "made.txt", "content": "by child"})),
            turn(tool_call("task_complete", {"summary": "wrote made.txt"})),
            turn(tool_call("task_complete", {"summary": "done"})),
        ]
    )
    handler = make_spawn_handler(executor, client, "test-model")
    agent_loop(
        "task",
        executor,
        client=client,
        tools=[*TOOLS, SPAWN_TOOL],
        extra_tool_handlers={"spawn_subagent": handler},
    )
    assert (tmp_path / "made.txt").read_text() == "by child"


# -- bounds -----------------------------------------------------------------


def test_sub_agents_cannot_spawn_sub_agents(executor):
    """Unbounded recursion would spend the whole daily request quota."""
    handler = make_spawn_handler(executor, FakeClient([]), "m", depth=1, max_depth=1)
    out = handler({"task": "recurse"})
    assert out.startswith("Error:")
    assert "cannot spawn" in out


def test_a_missing_task_is_reported(executor):
    handler = make_spawn_handler(executor, FakeClient([]), "m")
    assert handler({}).startswith("Error:")
    assert handler({"task": "   "}).startswith("Error:")


def test_an_unfinished_child_is_flagged_not_passed_off_as_done(executor):
    """A partial result read as a finished one is how the parent builds on sand."""
    client = FakeClient(
        [
            turn(tool_call("spawn_subagent", {"task": "too big"})),
            *[turn(tool_call("run_shell", {"command": "echo x"})) for _ in range(8)],
            turn(tool_call("task_complete", {"summary": "done"})),
        ]
    )
    handler = make_spawn_handler(executor, client, "test-model", max_iterations=3)
    result = agent_loop(
        "task",
        executor,
        client=client,
        tools=[*TOOLS, SPAWN_TOOL],
        extra_tool_handlers={"spawn_subagent": handler},
    )
    first = [m["content"] for m in result.messages if m["role"] == "tool"][0]
    assert first.startswith("[sub-agent")
    assert "max_iterations_reached" in first


def test_the_child_gets_its_own_shorter_leash(executor):
    from agent.subagent import SUBAGENT_MAX_ITERATIONS

    assert SUBAGENT_MAX_ITERATIONS < 12


def test_spawn_is_not_in_the_default_toolset():
    """Opt-in: it doubles the conversations a task can cost."""
    from agent.tools import TOOL_NAMES

    assert "spawn_subagent" not in TOOL_NAMES


def test_subagent_events_are_emitted(executor):
    seen = []
    client = FakeClient(
        [
            turn(tool_call("spawn_subagent", {"task": "errand"})),
            turn(tool_call("task_complete", {"summary": "ok"})),
            turn(tool_call("task_complete", {"summary": "done"})),
        ]
    )
    handler = make_spawn_handler(
        executor, client, "m", on_event=lambda n, p: seen.append(n)
    )
    agent_loop(
        "task",
        executor,
        client=client,
        tools=[*TOOLS, SPAWN_TOOL],
        extra_tool_handlers={"spawn_subagent": handler},
    )
    assert seen == ["subagent_start", "subagent_done"]


# -- project context --------------------------------------------------------


def test_a_project_file_is_found(tmp_path):
    (tmp_path / "DIETCODE.md").write_text("Always run pytest before finishing.")
    text, source = load_project_context(tmp_path)
    assert source == "DIETCODE.md"
    assert "pytest" in text


@pytest.mark.parametrize("name", ["DIETCODE.md", "AGENTS.md", "CLAUDE.md", ".cursorrules"])
def test_the_known_context_filenames_are_all_read(tmp_path, name):
    (tmp_path / name).write_text("house rules")
    assert load_project_context(tmp_path)[1] == name


def test_the_first_match_wins(tmp_path):
    (tmp_path / "AGENTS.md").write_text("second")
    (tmp_path / "DIETCODE.md").write_text("first")
    assert "first" in load_project_context(tmp_path)[0]


def test_no_context_file_is_not_an_error(tmp_path):
    assert load_project_context(tmp_path) == ("", None)


def test_an_empty_context_file_is_ignored(tmp_path):
    (tmp_path / "AGENTS.md").write_text("   \n\n")
    assert load_project_context(tmp_path)[1] is None


def test_a_huge_context_file_is_truncated(tmp_path):
    (tmp_path / "AGENTS.md").write_text("x" * 40_000)
    text, _ = load_project_context(tmp_path)
    assert len(text) < 9_000
    assert "truncated" in text


def test_project_instructions_are_marked_as_outranking_the_defaults():
    prompt = with_project_context(SYSTEM_PROMPT, "Never touch main.", "AGENTS.md")
    assert "Never touch main." in prompt
    assert "AGENTS.md" in prompt
    assert prompt.index(SYSTEM_PROMPT) < prompt.index("Never touch main.")
    assert "follow these" in prompt


def test_no_context_leaves_the_prompt_alone():
    assert with_project_context(SYSTEM_PROMPT, "", None) == SYSTEM_PROMPT


def test_the_agent_actually_receives_the_project_instructions(executor):
    client = FakeClient([turn(tool_call("task_complete", {"summary": "done"}))])
    agent_loop(
        "task",
        executor,
        client=client,
        system_prompt=with_project_context(SYSTEM_PROMPT, "Never touch main.", "AGENTS.md"),
    )
    assert "Never touch main." in client.calls[0][0]["content"]
