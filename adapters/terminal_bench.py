"""Terminal-Bench adapter.

Wraps the same `agent_loop` the CLI uses. The only difference is where commands
execute: instead of a container we create, they run in the container the harness
handed us via the TmuxSession.

Verified against terminal-bench 0.2.18, whose BaseAgent contract is:
    perform_task(instruction, session: TmuxSession, logging_dir: Path | None)
        -> AgentResult(total_input_tokens, total_output_tokens, failure_mode, ...)

Run it with:
    tb run --dataset terminal-bench-core --agent-import-path \
        adapters.terminal_bench:CliAgent --model llama-3.3-70b-versatile

Note: terminal-bench requires Python >= 3.12, while the agent itself runs on
3.11+. If `python --version` is 3.11 here, install the harness on a newer
interpreter (`py -3.13 -m pip install terminal-bench`) and run `tb` from that one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agent.auth import AuthError, default_provider, get_provider, resolve_key
from agent.loop import DEFAULT_MAX_ITERATIONS, DEFAULT_MODEL, agent_loop, make_client
from agent.sandbox import DockerExecutor, SandboxError

try:
    # `tb` does not load .env, so without this every task fails identically on a
    # missing key. Optional because the harness may run in its own isolated
    # environment where python-dotenv is not installed.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - depends on the harness's environment
    pass

try:  # only needed when actually running under the harness
    from terminal_bench.agents.base_agent import AgentResult, BaseAgent
    from terminal_bench.agents.failure_mode import FailureMode
    from terminal_bench.terminal.tmux_session import TmuxSession
except ImportError as exc:  # pragma: no cover - exercised only without the harness
    raise ImportError(
        "terminal-bench is not installed in this interpreter. "
        "Install it with `pip install terminal-bench` (requires Python >= 3.12)."
    ) from exc


def _credentials() -> tuple[str, str, str]:
    """(api_key, base_url, provider) for whichever provider is configured.

    The provider name is returned too, because each one is driven by its own
    SDK now and the base_url alone no longer identifies which.
    """
    provider = default_provider()
    spec = get_provider(provider)
    api_key, _source = resolve_key(provider)
    if not api_key:
        raise AuthError(
            f"no API key for {spec.label}. Run `dietcode login {spec.name}` "
            f"or set ${spec.env_var}."
        )
    return api_key, spec.base_url, provider


class SessionExecutor(DockerExecutor):
    """Runs commands in the harness's container instead of one we own.

    Uses docker-py's `exec_run` rather than typing into the tmux pane. A
    tool-calling agent needs stdout, stderr and an exit code as separate values;
    scraping a terminal pane gives you one blob of text with a prompt in it and
    no reliable exit status.

    Tradeoff: commands bypass tmux, so they will not appear in the task's
    asciinema recording. Grading reads the final container state, not the
    recording, so this does not affect scores -- but use the JSON transcript
    written to logging_dir for failure analysis, not the .cast file.
    """

    def __init__(self, session: TmuxSession, workdir: str = "/app"):
        self.session = session
        self.container = session.container
        self.image = ""
        self.workdir = workdir
        self.user = getattr(session, "_user", "") or None
        self._owns_container = False  # never tear down the harness's container

    @property
    def container_name(self) -> str:
        return getattr(self.container, "name", str(self.container))

    def _raw_exec(self, argv: list[str], timeout: int = 60) -> tuple[int, str, str]:
        # No client-side timeout here -- docker-py's exec_run has none. The
        # container-side `timeout` in the shell wrapper is what bounds run_shell.
        try:
            result = self.container.exec_run(
                argv,
                demux=True,
                workdir=self.workdir,
                user=self.user or "",
            )
        except Exception as exc:  # noqa: BLE001 - docker-py raises a wide range
            raise SandboxError(f"exec failed in container: {exc}") from None

        stdout_b, stderr_b = result.output if isinstance(result.output, tuple) else (result.output, b"")
        return (
            result.exit_code if result.exit_code is not None else 1,
            (stdout_b or b"").decode("utf-8", errors="replace"),
            (stderr_b or b"").decode("utf-8", errors="replace"),
        )

    def close(self) -> None:
        pass


# Maps our loop's exit status onto the harness's enum. `stopped` and
# `max_iterations_reached` are not harness failures -- the agent ran fine and
# just did not finish, which shows up as a failed test, not an error.
_FAILURE_MODES = {
    "complete": "NONE",
    "stopped": "NONE",
    "max_iterations_reached": "AGENT_TIMEOUT",
    "budget_exhausted": "AGENT_TIMEOUT",
    "error": "UNKNOWN_AGENT_ERROR",
}


class CliAgent(BaseAgent):
    """Our agent, wired into the Terminal-Bench harness."""

    def __init__(
        self,
        model_name: str | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        workdir: str = "/app",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        # tb passes --model through as model_name; it may carry a litellm-style
        # "provider/model" prefix that Groq's own API will not recognise.
        raw_model = model_name or DEFAULT_MODEL
        self._model = raw_model.split("/", 1)[-1] if raw_model.startswith("groq/") else raw_model
        self._max_iterations = int(os.environ.get("AGENT_MAX_ITERATIONS", max_iterations))
        self._workdir = workdir

    @staticmethod
    def name() -> str:
        return "cli-agent"

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        executor = SessionExecutor(session, workdir=self._workdir)

        try:
            # Resolve through the same credential path the CLI uses, so a
            # `dietcode login` is enough to run the benchmark -- tb does not
            # load .env and exporting a key per shell is easy to forget.
            api_key, base_url, provider = _credentials()
            client = make_client(
                api_key=api_key, base_url=base_url, provider=provider
            )
        except (AuthError, RuntimeError) as exc:
            # Almost always a missing GROQ_API_KEY. Surface it -- otherwise every
            # task in the run fails identically with no visible reason.
            if logging_dir is not None:
                try:
                    logging_dir.mkdir(parents=True, exist_ok=True)
                    (logging_dir / "error.txt").write_text(str(exc), encoding="utf-8")
                except OSError:
                    pass
            return AgentResult(
                total_input_tokens=0,
                total_output_tokens=0,
                failure_mode=FailureMode.UNKNOWN_AGENT_ERROR,
            )

        result = agent_loop(
            self._render_instruction(instruction),
            executor,
            client=client,
            model=self._model,
            max_iterations=self._max_iterations,
        )

        if logging_dir is not None:
            self._write_logs(logging_dir, instruction, result)

        return AgentResult(
            total_input_tokens=result.usage.get("prompt_tokens", 0),
            total_output_tokens=result.usage.get("completion_tokens", 0),
            failure_mode=getattr(
                FailureMode, _FAILURE_MODES.get(result.status, "UNKNOWN_AGENT_ERROR")
            ),
        )

    @staticmethod
    def _write_logs(logging_dir: Path, instruction: str, result: Any) -> None:
        """Per-task transcript and metrics. This is the input to the failure-mode
        breakdown in README.md, so it has to survive a crashed run."""
        try:
            logging_dir.mkdir(parents=True, exist_ok=True)
            (logging_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "instruction": instruction,
                        **result.metrics(),
                        # The reason, not just the status. On an `error` the
                        # summary holds the API failure; without it a failed
                        # task is a dead end for diagnosis.
                        "summary": result.summary,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (logging_dir / "transcript.json").write_text(
                json.dumps(result.messages, indent=2, default=str), encoding="utf-8"
            )
        except OSError:
            pass
