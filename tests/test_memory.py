"""The agent's own notes, and the verify gate.

Memory is the half of the instructions split that the agent may write. The
verify gate is what turns "done" from the model's opinion into an exit code.
"""

from __future__ import annotations

import pytest

from agent.memory import (
    MAX_NOTE_CHARS,
    REMEMBER_TOOL,
    load_memory,
    make_remember_handler,
    memory_path,
    remember,
    with_memory,
)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("DIETCODE_HOME", str(tmp_path / "home"))


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    return root


# -- where notes live -------------------------------------------------------


def test_notes_are_not_written_into_the_project(project):
    remember(project, "the tests need Docker running")
    assert str(project) not in str(memory_path(project))
    assert not list(project.iterdir())


def test_a_note_survives_to_the_next_session(project):
    remember(project, "run the suite with python -m pytest")
    assert "python -m pytest" in load_memory(project)


def test_a_project_with_no_notes_reads_as_empty(project):
    assert load_memory(project) == ""


def test_notes_accumulate(project):
    remember(project, "first thing")
    remember(project, "second thing")

    notes = load_memory(project)
    assert "first thing" in notes and "second thing" in notes


def test_the_same_note_is_not_stored_twice(project):
    remember(project, "the parser is generated, do not edit it by hand")
    remember(project, "the parser is generated, do not edit it by hand")
    assert load_memory(project).count("do not edit it by hand") == 1


def test_a_rambling_note_is_cut_short(project):
    """Memory is prepended to every request, so it is paid for every turn."""
    remember(project, "x" * (MAX_NOTE_CHARS + 500))
    assert len(load_memory(project)) < MAX_NOTE_CHARS + 200


def test_an_empty_note_is_refused_not_stored(project):
    assert remember(project, "   ").startswith("Error:")
    assert load_memory(project) == ""


def test_a_note_is_flattened_to_one_line(project):
    remember(project, "line one\nline two")
    assert "line one line two" in load_memory(project)


def test_an_unwritable_disk_returns_an_error_the_model_can_read(project, monkeypatch):
    """Same contract as every other tool: never raise into the loop."""
    monkeypatch.setattr(
        "pathlib.Path.mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only"))
    )
    assert remember(project, "something").startswith("Error:")


def test_very_old_notes_are_trimmed_from_the_front(project):
    """Later notes supersede earlier ones, so the tail is what to keep."""
    for i in range(200):
        remember(project, f"note number {i} " + "padding " * 10)

    notes = load_memory(project)
    assert "note number 199" in notes
    assert "trimmed" in notes


# -- how it reaches the model -----------------------------------------------


def test_notes_are_added_to_the_system_prompt(project):
    remember(project, "the build needs node 20")
    prompt = with_memory("BASE", project)

    assert "BASE" in prompt
    assert "node 20" in prompt


def test_no_notes_means_no_change_to_the_prompt(project):
    assert with_memory("BASE", project) == "BASE"


def test_notes_are_marked_as_the_agents_own_not_the_users(project):
    """They must not outrank what the user actually said, or the code."""
    remember(project, "something I concluded once")
    prompt = with_memory("BASE", project).lower()

    assert "your own earlier notes" in prompt
    assert "not instructions from the user" in prompt


def test_the_handler_saves_and_reports(project):
    handler = make_remember_handler(project)
    reply = handler({"note": "use ruff, not flake8"})

    assert "Remembered" in reply
    assert "ruff" in load_memory(project)


def test_the_handler_survives_a_missing_argument(project):
    assert make_remember_handler(project)({}).startswith("Error:")


def test_the_tool_schema_is_the_shape_the_loop_expects(project):
    assert REMEMBER_TOOL["function"]["name"] == "remember"
    assert "note" in REMEMBER_TOOL["function"]["parameters"]["properties"]


def test_remember_is_not_in_the_default_toolset():
    """It is wired per project by the CLI, because it needs to know which one."""
    from agent.tools import TOOL_NAMES

    assert "remember" not in TOOL_NAMES


# -- the verify gate --------------------------------------------------------
#
# "Done" is the model's opinion. A command that has to exit 0 first is a fact,
# and when one is configured it is what ends the loop.


class Recording:
    """An executor that answers a scripted sequence of exit codes."""

    def __init__(self, *exit_codes):
        from agent.sandbox import ShellResult

        self._codes = list(exit_codes)
        self.commands = []
        self._ShellResult = ShellResult

    def run_shell(self, command, timeout=30):
        self.commands.append(command)
        code = self._codes.pop(0) if self._codes else 0
        return self._ShellResult(
            stdout="" if code == 0 else "2 failed", stderr="", exit_code=code
        )

    def read_file(self, path):
        raise OSError("no files here")

    def write_file(self, path, content):
        return None

    def list_files(self, root=".", limit=800):
        return []

    def search(self, pattern, root=".", glob=None, limit=200):
        return []

    def close(self):
        return None


def done(summary="finished"):
    from tests.fake_llm import tool_call, turn

    return turn(tool_call("task_complete", {"summary": summary}))


def test_a_passing_command_lets_the_agent_finish():
    from agent.loop import agent_loop
    from tests.fake_llm import FakeClient

    executor = Recording(0)
    result = agent_loop(
        "do it",
        executor,
        client=FakeClient([done()]),
        model="m",
        verify_command="pytest -q",
    )

    assert result.status == "complete"
    assert "pytest -q" in executor.commands


def test_a_failing_command_sends_the_agent_back_to_work():
    """The whole point: the model said done, the tests say otherwise."""
    from agent.loop import agent_loop
    from tests.fake_llm import FakeClient

    executor = Recording(1, 0)
    result = agent_loop(
        "do it",
        executor,
        client=FakeClient([done(), done()]),
        model="m",
        verify_command="pytest -q",
    )

    assert result.status == "complete"
    assert executor.commands.count("pytest -q") == 2


def test_the_failure_output_is_handed_back_to_the_model():
    from agent.loop import agent_loop
    from tests.fake_llm import FakeClient

    client = FakeClient([done(), done()])
    agent_loop(
        "do it",
        Recording(1, 0),
        client=client,
        model="m",
        verify_command="pytest -q",
    )

    sent = [m for call in client.calls for m in call if m.get("role") == "tool"]
    assert any("still fails" in str(m.get("content", "")) for m in sent)
    assert any("2 failed" in str(m.get("content", "")) for m in sent)


def test_a_model_that_cannot_fix_it_still_terminates():
    """Bounded, like the deferral cap: never burn the request quota looping."""
    from agent.loop import MAX_VERIFY_ATTEMPTS, agent_loop
    from tests.fake_llm import FakeClient

    executor = Recording(*([1] * 20))
    result = agent_loop(
        "do it",
        executor,
        client=FakeClient([done() for _ in range(20)]),
        model="m",
        max_iterations=10,
        verify_command="pytest -q",
    )

    assert result.status in ("complete", "max_iterations_reached")
    assert executor.commands.count("pytest -q") <= MAX_VERIFY_ATTEMPTS + 1


def test_no_verify_command_means_no_extra_command_is_run():
    from agent.loop import agent_loop
    from tests.fake_llm import FakeClient

    executor = Recording()
    result = agent_loop("do it", executor, client=FakeClient([done()]), model="m")

    assert result.status == "complete"
    assert executor.commands == []


def test_the_verify_run_is_reported_to_the_ui():
    from agent.loop import agent_loop
    from tests.fake_llm import FakeClient

    events = []
    agent_loop(
        "do it",
        Recording(1, 0),
        client=FakeClient([done(), done()]),
        model="m",
        verify_command="pytest -q",
        on_event=lambda name, payload: events.append(name),
    )

    assert "verifying" in events
    assert "verification_failed" in events
    assert "verified" in events
