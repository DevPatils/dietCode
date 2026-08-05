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


# -- error messages ---------------------------------------------------------


def test_daily_quota_error_is_readable():
    """The raw provider error is a wall of JSON with the point buried in it."""
    from agent.ui import humanize_error

    raw = (
        "LLM call failed: Error code: 429 - {'error': {'message': 'Rate limit reached "
        "for model `llama-3.3-70b-versatile` in organization `org_01k` service tier "
        "`on_demand` on tokens per day (TPD): Limit 100000, Used 99510, Requested 779. "
        "Please try again in 4m9.696s.', 'code': 'rate_limit_exceeded'}}"
    )
    headline, hint = humanize_error(raw)
    assert headline == "daily token quota exhausted"
    assert "99,510" in hint and "100,000" in hint
    assert len(headline) < 60


def test_per_minute_limit_reports_the_retry_delay():
    from agent.ui import humanize_error

    headline, hint = humanize_error(
        "Error code: 429 - rate limit reached on tokens per minute, "
        "please try again in 12.5s"
    )
    assert headline == "rate limited"
    assert "12.5s" in hint


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Error code: 401 - invalid api key", "API key rejected"),
        ("Connection error while reaching host", "could not reach the API"),
        ("model `nope` does not exist", "unknown model"),
    ],
)
def test_common_failures_get_a_headline(raw, expected):
    from agent.ui import humanize_error

    assert humanize_error(raw)[0] == expected


def test_unknown_errors_are_shown_but_bounded():
    from agent.ui import humanize_error

    headline, hint = humanize_error("something\nweird\n" + "x" * 500)
    assert len(headline) <= 201
    assert "\n" not in headline
    assert hint == ""


# -- startup screen ---------------------------------------------------------


def test_logo_falls_back_to_ascii(monkeypatch):
    import agent.ui as ui

    monkeypatch.setattr(ui, "_glyphs", None)
    monkeypatch.setattr(ui.sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    plain = ui.logo().plain
    assert plain.strip()
    plain.encode("cp1252")  # the block-drawing logo would raise here


LONG_PATH = "C:/Users/Someone/OneDrive/Desktop/Projects/dietCode/agent-work"


@pytest.mark.parametrize("width", [70, 78, 100, 110, 140, 200])
def test_banner_never_overflows_its_width(width):
    """A wrapped or over-wide banner reads as a rendering bug."""
    from agent.ui import banner

    console = Console(file=io.StringIO(), width=width, no_color=True)
    banner(console, "llama-3.3-70b", "cli-agent-abc123", [(LONG_PATH, "/workspace")], local=False)
    for line in console.file.getvalue().splitlines():
        assert len(line) <= width, f"line overflows at width {width}: {line!r}"


def test_wide_terminals_get_two_columns():
    from agent.ui import banner

    console = Console(file=io.StringIO(), width=120, no_color=True)
    banner(console, "llama-3.3-70b", "c", [(LONG_PATH, "/workspace")], local=False)
    rendered = console.file.getvalue()
    logo_row = next(ln for ln in rendered.splitlines() if "Tips for getting started" in ln)
    # Tips sit beside the wordmark, not under it.
    assert "█" in logo_row or "_" in logo_row


def test_narrow_terminals_stack_instead_of_squeezing():
    from agent.ui import banner

    console = Console(file=io.StringIO(), width=76, no_color=True)
    banner(console, "llama-3.3-70b", "c", [], local=False)
    logo_row = next(ln for ln in console.file.getvalue().splitlines() if "Tips" in ln)
    assert "█" not in logo_row


def test_the_wordmark_is_never_truncated():
    """rich shrinks a fixed column to fit a flexible neighbour, which cuts the
    logo mid-glyph."""
    from agent.ui import LOGO_WIDTH, banner, logo_lines

    console = Console(file=io.StringIO(), width=120, no_color=True)
    banner(console, "m", "c", [], local=False)
    rendered = console.file.getvalue()
    widest = max(logo_lines(), key=lambda t: len(t.plain)).plain.strip()
    assert widest in rendered
    assert LOGO_WIDTH >= len(widest)


def test_recent_activity_lists_newest_first(tmp_path):
    import os
    import time

    from agent.ui import recent_activity

    for i, name in enumerate(["old.txt", "mid.txt", "new.txt"]):
        p = tmp_path / name
        p.write_text("x" * (i + 1))
        os.utime(p, (time.time() + i, time.time() + i))

    entries = recent_activity([(str(tmp_path), "/workspace")])
    assert entries[0].startswith("new.txt")


def test_recent_activity_is_empty_without_a_mount():
    from agent.ui import recent_activity

    assert recent_activity([]) == []


def test_recent_activity_survives_a_missing_directory():
    from agent.ui import recent_activity

    assert recent_activity([("/definitely/not/here", "/workspace")]) == []


def test_context_percent():
    from agent.ui import context_percent

    assert context_percent(0, 1000) == 100
    assert context_percent(500, 1000) == 50
    assert context_percent(2000, 1000) == 0  # clamped, never negative
    assert context_percent(10, 0) == 100  # no budget configured


def test_status_bar_shows_location_sandbox_and_model():
    from agent.ui import sandbox_label, status_bar

    bar = status_bar(
        "~/proj", sandbox_label("c", [], local=False), "llama-3.3-70b", 73, width=90
    )
    assert "~/proj" in bar
    assert "sandboxed" in bar
    assert "llama-3.3-70b" in bar
    assert "(73%)" in bar


def test_the_status_bar_reports_the_protection_not_an_alarm():
    """Working in your own directory is the default now, so the bar states what
    is protecting you. A red warning on every single launch is one people stop
    reading, which is worse than no warning."""
    from agent.ui import sandbox_label

    label = sandbox_label(None, [], local=True)
    assert "asks first" in label
    assert "no sandbox" not in label


def test_a_mounted_sandbox_is_distinguishable_from_a_sealed_one():
    from agent.ui import sandbox_label

    assert "mounted" in sandbox_label("c", [("/host", "/workspace")], local=False)
    assert "mounted" not in sandbox_label("c", [], local=False)


def test_status_bar_survives_a_narrow_terminal():
    from agent.ui import sandbox_label, status_bar

    bar = status_bar(
        "x" * 30, sandbox_label("c", [], local=False), "some-long-model-name", 5, width=20
    )
    assert "(5%)" in bar  # no crash, no negative padding


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


def test_every_advertised_command_is_dispatchable(session, monkeypatch):
    """A command listed in /help that does nothing is worse than not listing it."""
    from agent.repl import COMMANDS

    for name in COMMANDS:
        if name in ("/exit",):
            continue
        # Stub the ones that would prompt or shell out.
        monkeypatch.setattr("agent.repl.login", lambda *a, **k: 0)
        monkeypatch.setattr("agent.repl.logout", lambda *a, **k: 0)
        monkeypatch.setattr("agent.repl.auth_status", lambda *a, **k: 0)
        monkeypatch.setattr("agent.repl.doctor", lambda *a, **k: 0)
        assert session.handle_command(name) is True, name
        assert "unknown command" not in output(session.console), name


def test_model_can_be_switched_mid_session(session):
    session.handle_command("/model llama-3.1-8b-instant")
    assert session.model == "llama-3.1-8b-instant"


def test_model_with_no_argument_reports_the_current_one(session):
    session.model = "some-model"
    session.handle_command("/model")
    assert "some-model" in output(session.console)


def test_switching_provider_without_a_key_is_refused(session, monkeypatch):
    """Leaving the session pointed at a provider it cannot authenticate with
    would fail every later turn."""
    monkeypatch.setattr("agent.repl.resolve_key", lambda name: (None, "not set"))
    before_provider, before_model = session.provider, session.model
    session.handle_command("/provider gemini")
    assert session.provider == before_provider
    assert session.model == before_model
    assert "no key" in output(session.console)


def test_switching_provider_updates_model_and_client(session, monkeypatch):
    monkeypatch.setattr("agent.repl.resolve_key", lambda name: ("k", "saved login"))
    monkeypatch.setattr("agent.repl.make_client", lambda **kw: "new-client")
    from agent.auth import PROVIDERS

    session.handle_command("/provider gemini")
    assert session.provider == "gemini"
    assert session.model == PROVIDERS["gemini"].default_model
    assert session.client == "new-client"


def test_switching_provider_keeps_the_conversation(session, monkeypatch):
    """Changing model mid-task is the point; the transcript is not provider
    specific."""
    monkeypatch.setattr("agent.repl.resolve_key", lambda name: ("k", "saved login"))
    monkeypatch.setattr("agent.repl.make_client", lambda **kw: "new-client")
    session.history = [{"role": "user", "content": "earlier"}]
    session.handle_command("/provider gemini")
    assert session.history == [{"role": "user", "content": "earlier"}]


def test_unknown_provider_is_reported(session):
    session.handle_command("/provider hotdog")
    assert "unknown provider" in output(session.console)


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
