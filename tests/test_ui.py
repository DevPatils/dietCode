"""Rendering and session tests. No terminal required -- Console writes to a buffer."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from agent.repl import Session
from agent.ui import Renderer, collapse, describe_tool_call
from tests.fake_llm import FakeClient, tool_call, turn


@pytest.fixture
def console():
    return Console(file=io.StringIO(), width=100, force_terminal=False, no_color=True)


def output(console: Console) -> str:
    return console.file.getvalue()


# -- describe_tool_call -----------------------------------------------------


def test_shell_calls_render_as_the_command():
    label, detail = describe_tool_call("run_shell", {"command": "ls -l", "timeout": 30})
    assert label == "shell"
    assert detail == "$ ls -l"


def test_write_calls_render_as_path_and_size():
    label, detail = describe_tool_call(
        "write_file", {"path": "a.py", "content": "one\ntwo\n"}
    )
    assert label == "write"
    assert "a.py" in detail
    assert "3 lines" in detail  # trailing newline counts as a line


def test_read_calls_render_as_the_path():
    assert describe_tool_call("read_file", {"path": "x.txt"}) == ("read", "x.txt")


def test_malformed_arguments_still_render():
    """Display must never be the thing that crashes a run."""
    label, detail = describe_tool_call("run_shell", "{not json")
    assert label == "run_shell"
    assert detail


def test_unknown_tool_renders():
    label, _detail = describe_tool_call("mystery", {"a": 1})
    assert label == "mystery"


# -- collapse ---------------------------------------------------------------


def test_short_output_is_untouched():
    assert collapse("one\ntwo") == "one\ntwo"


def test_long_output_keeps_head_and_tail():
    text = "\n".join(str(i) for i in range(100))
    out = collapse(text, max_lines=10)
    assert out.startswith("0\n1")
    assert out.rstrip().endswith("99")
    assert "more lines" in out


# -- Renderer ---------------------------------------------------------------


def test_renderer_draws_a_tool_call(console):
    Renderer(console).on_event(
        "tool_call", {"step": 1, "name": "run_shell", "arguments": {"command": "ls"}}
    )
    assert "$ ls" in output(console)


def test_renderer_ignores_unknown_events(console):
    Renderer(console).on_event("something_new", {"anything": 1})  # must not raise


def test_task_complete_is_not_drawn_twice(console):
    r = Renderer(console)
    r.on_event("tool_call", {"step": 1, "name": "task_complete", "arguments": {"summary": "hi"}})
    r.close()
    assert "task_complete" not in output(console)


def test_answering_a_question_is_not_flagged_as_a_failure(console):
    """Prose replies are a valid outcome; a warning there reads as a bug."""
    r = Renderer(console)
    r.on_event("stopped", {"step": 1, "text": "The file is notes.txt."})
    r.close()
    assert "stopped" not in output(console)


def test_a_silent_stop_is_flagged(console):
    r = Renderer(console)
    r.on_event("stopped", {"step": 1, "text": "   "})
    r.close()
    assert "stopped" in output(console)


def test_errors_are_visible(console):
    r = Renderer(console)
    r.on_event("tool_result", {"step": 1, "name": "read_file", "output": "Error: nope"})
    r.close()
    assert "Error: nope" in output(console)


# -- Windows console encoding ----------------------------------------------
#
# A cp1252 console cannot encode the arrow/tick glyphs, and printing one raises
# UnicodeEncodeError mid-render, killing the session. This crashed the first
# interactive run.


def test_glyphs_fall_back_to_ascii_on_a_narrow_encoding(monkeypatch):
    import agent.ui as ui

    monkeypatch.setattr(ui, "_glyphs", None)
    monkeypatch.setattr(ui.sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    assert ui.glyph("arrow") == "->"
    assert ui.glyph("tick").isascii()


def test_glyphs_use_unicode_when_supported(monkeypatch):
    import agent.ui as ui

    monkeypatch.setattr(ui, "_glyphs", None)
    monkeypatch.setattr(ui.sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="utf-8"))
    assert ui.glyph("arrow") == "→"


def test_everything_renders_on_a_cp1252_console(monkeypatch):
    """Full banner + every event, encoded as a legacy console would."""
    import agent.ui as ui

    monkeypatch.setattr(ui, "_glyphs", None)
    monkeypatch.setattr(ui.sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))

    buffer = io.StringIO()
    # legacy_windows makes rich swap its box-drawing characters for ASCII, as it
    # does on a real cp1252 console -- so this exercises our glyphs, not rich's.
    console = Console(file=buffer, width=90, no_color=True, legacy_windows=True)
    ui.banner(console, "test-model", "container-x", [("C:/proj", "/workspace")], local=False)

    renderer = ui.Renderer(console, show_steps=True)
    for event, payload in [
        ("step_start", {"step": 1, "max_steps": 12}),
        ("tool_call", {"step": 1, "name": "run_shell", "arguments": {"command": "ls"}}),
        ("tool_result", {"step": 1, "name": "run_shell", "output": "exit_code: 0"}),
        ("recovered_tool_calls", {"step": 1, "count": 1}),
        ("completion_deferred", {"step": 1, "summary": "x"}),
        ("complete", {"step": 1, "summary": "done"}),
        ("stopped", {"step": 1, "text": ""}),
        ("max_iterations", {"step": 12}),
        ("error", {"message": "boom"}),
    ]:
        renderer.on_event(event, payload)
    renderer.close()

    # The real failure mode: encoding the rendered output blew up.
    buffer.getvalue().encode("cp1252")


# -- Session ----------------------------------------------------------------


@pytest.fixture
def session(tmp_path, monkeypatch):
    from agent.sandbox import LocalExecutor

    s = Session(LocalExecutor(tmp_path), client=None, model="test-model")
    s.console = Console(file=io.StringIO(), width=100, no_color=True)
    s.renderer = Renderer(s.console)
    return s


def test_exit_command_ends_the_session(session):
    assert session.handle_command("/exit") is False
    assert session.handle_command("/quit") is False


def test_help_lists_commands(session):
    assert session.handle_command("/help") is True
    assert "/clear" in output(session.console)


def test_unknown_command_is_reported_not_run_as_a_task(session):
    assert session.handle_command("/nonsense") is True
    assert "unknown command" in output(session.console)


def test_clear_drops_history_but_not_the_sandbox(session):
    session.history = [{"role": "user", "content": "old"}]
    session.handle_command("/clear")
    assert session.history is None


def test_conversation_carries_across_turns(session):
    """The point of a session: turn two can see what turn one did."""
    session.client = FakeClient(
        [
            turn(tool_call("write_file", {"path": "a.txt", "content": "1"})),
            turn(tool_call("task_complete", {"summary": "made a.txt"})),
            turn(tool_call("task_complete", {"summary": "still here"})),
        ]
    )
    session.run_turn("make a file")
    first_length = len(session.history or [])
    assert first_length > 0

    session.run_turn("what did you just do?")
    assert len(session.history or []) > first_length
    assert session.turns == 2

    # The second turn was given the first turn's transcript.
    second_request = session.client.calls[-1]
    assert any(m.get("content") == "make a file" for m in second_request)


def test_session_accumulates_usage(session):
    session.client = FakeClient([turn(tool_call("task_complete", {"summary": "done"}))])
    session.run_turn("do a thing")
    assert session.total_tokens == 120
    assert session.total_steps == 1


def test_interrupted_turn_is_discarded(session):
    """A half-finished transcript has unanswered tool calls and would be
    rejected by the API on the next request."""

    class Interrupting:
        chat = completions = None

        def create(self, **kwargs):
            raise KeyboardInterrupt()

    client = Interrupting()
    client.chat = client
    client.completions = client
    session.client = client
    session.history = [{"role": "user", "content": "earlier"}]

    session.run_turn("this gets interrupted")

    assert session.history == [{"role": "user", "content": "earlier"}]
    assert session.turns == 0
    assert "interrupted" in output(session.console)
