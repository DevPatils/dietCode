"""Terminal-Bench adapter tests.

Skipped unless terminal-bench is importable. It needs Python >= 3.12, so on a
3.11 interpreter these will skip -- run them with the interpreter you installed
the harness on (e.g. `py -3.13 -m pytest tests/test_adapter.py`).
"""

from __future__ import annotations

import pytest

pytest.importorskip("terminal_bench", reason="terminal-bench not installed here")

from terminal_bench.agents.base_agent import BaseAgent  # noqa: E402

from adapters.terminal_bench import CliAgent, SessionExecutor  # noqa: E402


class FakeExecResult:
    def __init__(self, exit_code: int, output):
        self.exit_code = exit_code
        self.output = output


class FakeContainer:
    """Stands in for a docker-py Container from the harness."""

    name = "fake-harness-container"

    def __init__(self, exit_code: int = 0, output=(b"out", b"err")):
        self.calls: list[tuple[list[str], dict]] = []
        self._exit_code = exit_code
        self._output = output

    def exec_run(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        return FakeExecResult(self._exit_code, self._output)


class FakeSession:
    def __init__(self, container: FakeContainer | None = None):
        self.container = container or FakeContainer()
        self._user = ""


@pytest.fixture
def session():
    return FakeSession()


# -- harness contract -------------------------------------------------------


def test_agent_satisfies_the_base_agent_contract():
    assert issubclass(CliAgent, BaseAgent)
    assert not CliAgent.__abstractmethods__
    assert CliAgent.name() == "cli-agent"


def test_litellm_style_model_prefix_is_stripped():
    """tb passes --model through as e.g. groq/llama-3.3-70b; Groq's own API
    rejects the prefix."""
    assert CliAgent(model_name="groq/llama-3.3-70b-versatile")._model == (
        "llama-3.3-70b-versatile"
    )
    assert CliAgent(model_name="llama-3.3-70b-versatile")._model == (
        "llama-3.3-70b-versatile"
    )


# -- SessionExecutor --------------------------------------------------------


def test_session_executor_reuses_the_shared_tool_logic():
    """The whole point of the split: the benchmark runs the same code as the CLI."""
    assert SessionExecutor.run_shell is not SessionExecutor.__dict__.get("run_shell")
    assert SessionExecutor.run_shell.__qualname__ == "DockerExecutor.run_shell"
    assert SessionExecutor.write_file.__qualname__ == "DockerExecutor.write_file"
    assert SessionExecutor.read_file.__qualname__ == "DockerExecutor.read_file"


def test_run_shell_demuxes_stdout_and_stderr(session):
    result = SessionExecutor(session).run_shell("echo hi")
    assert result.exit_code == 0
    assert result.stdout == "out"
    assert result.stderr == "err"


def test_commands_execute_in_the_harness_container(session):
    ex = SessionExecutor(session, workdir="/app")
    ex.run_shell("echo hi")
    _argv, kwargs = session.container.calls[-1]
    assert kwargs["workdir"] == "/app"


def test_written_content_never_appears_as_shell_syntax(session):
    ex = SessionExecutor(session)
    ex.write_file("/app/x.txt", "rm -rf / $(whoami) `id` 'q' \"qq\"")
    argv, _kwargs = session.container.calls[-1]
    joined = " ".join(argv)
    assert "whoami" not in joined
    assert "rm -rf" not in joined


def test_missing_exit_code_is_treated_as_failure():
    container = FakeContainer(exit_code=None)
    ex = SessionExecutor(FakeSession(container))
    assert ex.run_shell("x").exit_code == 1


def test_undemuxed_output_does_not_crash():
    """demux=True normally yields a tuple; tolerate a plain bytes payload."""
    container = FakeContainer(output=b"plain-bytes")
    ex = SessionExecutor(FakeSession(container))
    assert ex.run_shell("x").stdout == "plain-bytes"


def test_close_never_touches_the_harness_container(session):
    ex = SessionExecutor(session)
    ex.close()
    assert ex._owns_container is False
    # No docker rm was issued.
    assert all("rm" not in argv for argv, _ in session.container.calls)


# -- perform_task -----------------------------------------------------------


def test_missing_api_key_is_reported_not_swallowed(session, tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = CliAgent().perform_task("do a thing", session, logging_dir=tmp_path)
    assert result.failure_mode.value == "unknown_agent_error"
    assert "GROQ_API_KEY" in (tmp_path / "error.txt").read_text(encoding="utf-8")


def test_completed_task_reports_tokens_and_no_failure(session, tmp_path, monkeypatch):
    from agent.loop import AgentResult as LoopResult

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr("adapters.terminal_bench.make_client", lambda: object())
    monkeypatch.setattr(
        "adapters.terminal_bench.agent_loop",
        lambda *a, **k: LoopResult(
            status="complete",
            summary="done",
            steps=3,
            usage={"prompt_tokens": 111, "completion_tokens": 22},
            messages=[{"role": "user", "content": "hi"}],
        ),
    )

    result = CliAgent().perform_task("do a thing", session, logging_dir=tmp_path)
    assert result.failure_mode.value == "none"
    assert result.total_input_tokens == 111
    assert result.total_output_tokens == 22
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "transcript.json").exists()


@pytest.mark.parametrize(
    "status,expected",
    [
        ("complete", "none"),
        ("stopped", "none"),
        ("max_iterations_reached", "agent_timeout"),
        ("error", "unknown_agent_error"),
    ],
)
def test_status_maps_to_failure_mode(session, monkeypatch, status, expected):
    from agent.loop import AgentResult as LoopResult

    monkeypatch.setattr("adapters.terminal_bench.make_client", lambda: object())
    monkeypatch.setattr(
        "adapters.terminal_bench.agent_loop",
        lambda *a, **k: LoopResult(status=status, usage={}),
    )
    result = CliAgent().perform_task("x", session)
    assert result.failure_mode.value == expected
