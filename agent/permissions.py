"""Consent gate for running against a real working directory.

Docker gives *containment*: the agent can do anything, and none of it reaches
you. Running in your own cwd has no containment, so the protection has to be
*consent* -- you see each action before it happens and approve it.

Be clear about what this is not. A shell command can `cd ..` or write an
absolute path, and no amount of string inspection reliably stops that. The gate
is a prompt, not a jail. Path containment here catches honest mistakes; the
approval prompt is what catches the rest. For untrusted work, use the sandbox.

Classification exists only to cut prompt fatigue: obviously read-only commands
run without asking, everything else asks. Misreading a command as read-only
would be the damaging direction, so the allowlist is small and, in a compound
command, the strictest segment decides.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .sandbox import DEFAULT_TIMEOUT, Executor, SandboxError, ShellResult

# Exit code shells use for "permission denied"; the model reads it as a normal
# failed command rather than a crash.
DENIED_EXIT_CODE = 126


class Risk(StrEnum):
    READ_ONLY = "read-only"
    MODIFIES = "modifies files"
    DANGEROUS = "dangerous"


class Decision(StrEnum):
    ONCE = "once"
    ALWAYS = "always"
    NO = "no"


@dataclass
class Request:
    """One action awaiting approval."""

    action: str  # run | write | read
    detail: str  # the command, or the path
    risk: Risk
    outside_root: bool = False
    root: str = ""

    @property
    def remember_key(self) -> str:
        """What "always allow" should cover.

        Keyed by program (or by action for file writes) rather than by the
        exact string: remembering a whole command line would mean approving
        `ls -la` and being asked again for `ls -l`.
        """
        if self.action == "run":
            return f"run:{primary_program(self.detail)}"
        return f"{self.action}:*"


Approver = Callable[[Request], Decision]


# Commands that cannot change anything. Deliberately short: `sed` and `awk` are
# absent because -i and redirection make them writers, and `python` is absent
# because it runs arbitrary code.
_READ_ONLY_PROGRAMS = {
    "ls", "cat", "head", "tail", "grep", "rg", "find", "pwd", "wc", "stat",
    "file", "tree", "du", "df", "which", "date", "whoami", "echo", "diff",
    "sort", "uniq", "cut", "basename", "dirname", "realpath", "env", "printenv",
    "true", "false", "test", "type", "id", "uname", "hostname", "ps", "man",
}

_READ_ONLY_GIT = {
    "status", "diff", "log", "show", "branch", "remote", "ls-files",
    "rev-parse", "describe", "blame", "config",
}

# Patterns worth naming explicitly in the prompt, so an approval is informed.
_DANGEROUS = [
    (re.compile(r"\brm\s+(-\w*[rf]\w*\s+)+"), "recursive or forced delete"),
    (re.compile(r"\b(sudo|su)\b"), "runs as another user"),
    (re.compile(r"\b(mkfs|fdisk|dd)\b"), "writes raw devices"),
    (re.compile(r"\b(shutdown|reboot|halt)\b"), "stops the machine"),
    (re.compile(r"(curl|wget)[^|;&]*\|\s*(ba)?sh"), "pipes the network into a shell"),
    (re.compile(r"\bchmod\s+(-\w+\s+)*777\b"), "makes files world-writable"),
    (re.compile(r"\bgit\s+push\b[^|;&]*--force"), "force-pushes"),
    (re.compile(r"\bgit\s+reset\b[^|;&]*--hard"), "discards local changes"),
    (re.compile(r":\(\)\s*\{.*\|.*&.*\}"), "fork bomb"),
    (re.compile(r">\s*/dev/[sh]d"), "writes to a disk device"),
]

_SEGMENT_SPLIT = re.compile(r"(?:\|\||&&|[;|&\n])")


def _segments(command: str) -> list[str]:
    return [s.strip() for s in _SEGMENT_SPLIT.split(command) if s.strip()]


def primary_program(command: str) -> str:
    """The program a command runs, for display and for 'always allow'."""
    segments = _segments(command)
    if not segments:
        return command.strip()[:20] or "?"
    try:
        parts = shlex.split(segments[0])
    except ValueError:  # unbalanced quotes
        parts = segments[0].split()
    for part in parts:
        # Skip leading VAR=value assignments.
        if "=" in part and not part.startswith("-") and part.split("=")[0].isidentifier():
            continue
        return Path(part).name
    return segments[0].split()[0] if segments[0].split() else "?"


def _segment_is_read_only(segment: str) -> bool:
    try:
        parts = shlex.split(segment)
    except ValueError:
        return False
    if not parts:
        return False
    # Any redirection makes it a writer regardless of the program.
    if ">" in segment:
        return False
    program = Path(parts[0]).name
    if program == "git":
        return len(parts) > 1 and parts[1] in _READ_ONLY_GIT
    return program in _READ_ONLY_PROGRAMS


def classify(command: str) -> tuple[Risk, str]:
    """(risk, why). The strictest segment of a compound command wins."""
    for pattern, reason in _DANGEROUS:
        if pattern.search(command):
            return Risk.DANGEROUS, reason

    segments = _segments(command)
    if segments and all(_segment_is_read_only(s) for s in segments):
        return Risk.READ_ONLY, "reads only"
    return Risk.MODIFIES, "can change files"


class Mode(StrEnum):
    """How much the agent may do before it has to stop and ask.

    A mode rather than a per-call setting, because the answer to "may I write
    this file" is almost never about the individual file -- it is about how
    much you currently trust the run. Snapshots are what make the looser two
    reasonable: every edit is recoverable with /undo.
    """

    MANUAL = "manual"            # ask before anything that changes state
    ACCEPT_EDITS = "accept-edits"  # file writes go through; commands still ask
    PLAN = "plan"                # nothing executes; the turn produces a plan
    AUTO = "auto"                # ask for nothing


MODE_HELP = {
    Mode.MANUAL: "ask before every command and every write",
    Mode.ACCEPT_EDITS: "write files freely, still ask before running commands",
    Mode.PLAN: "read and think, but change nothing",
    Mode.AUTO: "do everything without asking",
}


@dataclass
class Policy:
    """What to do without asking."""

    # Read-only commands and reads inside the working directory are the bulk of
    # what an agent does; prompting for each makes the tool unusable.
    auto_allow_read_only: bool = True
    auto_allow_reads: bool = True
    # Approve everything, no prompts. Named to be hard to type by accident.
    yes_to_everything: bool = False
    mode: Mode = Mode.MANUAL

    @classmethod
    def for_mode(cls, mode: Mode | str) -> Policy:
        mode = Mode(mode)
        return cls(yes_to_everything=mode is Mode.AUTO, mode=mode)

    @property
    def writes_need_approval(self) -> bool:
        return self.mode is Mode.MANUAL

    @property
    def read_only(self) -> bool:
        """Plan mode: nothing may change, however it is asked for."""
        return self.mode is Mode.PLAN


class PermissionGate:
    """Wraps an Executor and asks before anything that could change your files.

    Denials come back as ordinary tool failures -- an error string or a non-zero
    exit code -- so the model reads them as feedback and adapts, which is the
    same contract every other tool failure follows.
    """

    def __init__(
        self,
        inner: Executor,
        root: str | Path,
        approver: Approver,
        policy: Policy | None = None,
    ):
        self._inner = inner
        self.root = Path(root).resolve()
        self._approver = approver
        self.policy = policy or Policy()
        self._remembered: set[str] = set()
        self.denied_count = 0
        self.approved: list[Request] = []

    # -- decisions ----------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            return candidate.resolve()
        except OSError:
            return candidate

    def _outside_root(self, path: str) -> bool:
        try:
            return not self._resolve(path).is_relative_to(self.root)
        except (OSError, ValueError):
            return True

    def _permitted(self, request: Request) -> bool:
        if self.policy.read_only and request.risk is not Risk.READ_ONLY:
            # Plan mode. Refused without asking: the whole point is that this
            # turn cannot change anything, so a prompt would be theatre.
            self.denied_count += 1
            return False
        if self.policy.yes_to_everything:
            return True
        if (
            request.action == "write"
            and not request.outside_root
            and not self.policy.writes_need_approval
        ):
            # accept-edits: writes inside the working directory go through,
            # because snapshots make them recoverable. Outside it they do not
            # -- nothing about this mode says "edit the rest of the disk".
            self.approved.append(request)
            return True
        if request.remember_key in self._remembered:
            return True
        if (
            request.risk is Risk.READ_ONLY
            and not request.outside_root
            and self.policy.auto_allow_read_only
        ):
            return True

        decision = self._approver(request)
        if decision is Decision.ALWAYS:
            self._remembered.add(request.remember_key)
        if decision is Decision.NO:
            self.denied_count += 1
            return False
        self.approved.append(request)
        return True

    # -- Executor -----------------------------------------------------------

    def run_shell(self, command: str, timeout: int = DEFAULT_TIMEOUT) -> ShellResult:
        risk, why = classify(command)
        request = Request(
            action="run", detail=command, risk=risk, root=str(self.root)
        )
        if not self._permitted(request):
            return ShellResult(
                stdout="",
                stderr=(
                    "Plan mode: nothing may be run or changed. Describe what you "
                    "would do instead, then stop."
                    if self.policy.read_only
                    else "Denied by the user. Do not retry this command; either "
                    "explain why it is needed or try a different approach."
                ),
                exit_code=DENIED_EXIT_CODE,
            )
        del why
        return self._inner.run_shell(command, timeout=timeout)

    def read_file(self, path: str) -> str:
        outside = self._outside_root(path)
        if self.policy.auto_allow_reads and not outside:
            return self._inner.read_file(path)
        request = Request(
            action="read",
            detail=str(self._resolve(path)),
            risk=Risk.READ_ONLY,
            outside_root=outside,
            root=str(self.root),
        )
        if not self._permitted(request):
            raise SandboxError(f"reading {path} was denied by the user")
        return self._inner.read_file(path)

    def list_files(self, root: str = ".", limit: int = 800) -> list[str]:
        # Listing and searching are reads; they follow the same rule as
        # read_file rather than getting their own prompt.
        if self._outside_root(root):
            request = Request(
                action="read", detail=str(self._resolve(root)), risk=Risk.READ_ONLY,
                outside_root=True, root=str(self.root),
            )
            if not self._permitted(request):
                raise SandboxError(f"listing {root} was denied by the user")
        return self._inner.list_files(root, limit)

    def search(
        self, pattern: str, root: str = ".", glob: str | None = None, limit: int = 200
    ) -> list[str]:
        if self._outside_root(root):
            request = Request(
                action="read", detail=str(self._resolve(root)), risk=Risk.READ_ONLY,
                outside_root=True, root=str(self.root),
            )
            if not self._permitted(request):
                raise SandboxError(f"searching {root} was denied by the user")
        return self._inner.search(pattern, root, glob, limit)

    def write_file(self, path: str, content: str) -> None:
        outside = self._outside_root(path)
        request = Request(
            action="write",
            detail=str(self._resolve(path)),
            risk=Risk.DANGEROUS if outside else Risk.MODIFIES,
            outside_root=outside,
            root=str(self.root),
        )
        if not self._permitted(request):
            raise SandboxError(
                f"Plan mode: {path} was not written. Nothing may be changed this "
                f"turn -- describe the change instead."
                if self.policy.read_only
                else f"writing {path} was denied by the user"
            )
        self._inner.write_file(path, content)

    def close(self) -> None:
        self._inner.close()


def deny_all(request: Request) -> Decision:
    """Approver for when nobody can be asked -- piped input, or a script.

    Denying is the only safe default: silently approving because stdin is not a
    terminal is how an automated run quietly rewrites someone's home directory.
    """
    del request
    return Decision.NO
