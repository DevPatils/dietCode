"""Execution backends.

Every tool that touches the outside world goes through an Executor. That includes
read_file/write_file, not just run_shell -- if file tools hit the host filesystem
while the shell runs in a container, the agent reads one world and acts on another.

  LocalExecutor  -- subprocess + host filesystem. Tests and early development only.
  DockerExecutor -- everything inside a container, via `docker exec`. Either
                    creates its own (local dev) or attaches to one it was handed.

The Terminal-Bench adapter subclasses DockerExecutor and overrides only
`_raw_exec`, so the benchmark and the CLI share all the actual tool logic --
that is the point of the split.
"""

from __future__ import annotations

import base64
import os
import re
import shlex
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

DEFAULT_TIMEOUT = 30
DEFAULT_IMAGE = os.environ.get("SANDBOX_IMAGE", "python:3.11-slim")

# Resource caps. The agent runs shell commands an LLM wrote, so a fork bomb or a
# runaway `dd` is a normal Tuesday rather than an attack. Without these the
# container can exhaust the host's Docker VM.
DEFAULT_MEMORY = os.environ.get("SANDBOX_MEMORY", "2g")
DEFAULT_PIDS_LIMIT = int(os.environ.get("SANDBOX_PIDS", "512"))
DEFAULT_CPUS = os.environ.get("SANDBOX_CPUS", "2")

# Every container we create is labelled so orphans can be found later. close()
# only runs on a clean exit; a SIGKILL or a closed laptop leaves the container
# running forever otherwise.
CONTAINER_LABEL = "com.dietcode.agent"
CREATED_LABEL = "com.dietcode.agent.created"
DEFAULT_ORPHAN_AGE = 6 * 3600

# Where the shell wrapper stashes the working directory between calls.
CWD_STATE_FILE = "/tmp/.agent_cwd"

# Each `docker exec` is a fresh process, so `cd` in one command would be lost by
# the next -- the agent would run `cd /app && ls`, see the right thing, then have
# its next command silently execute somewhere else. These wrappers persist the cwd
# to a file and restore it, which is the behaviour a model expects from a shell.
# The agent's command always arrives as an argv element, never interpolated into
# the script, so no quoting inside it can break out.

# Restores the saved cwd; falls back to the workdir in "$2", then to /.
_RESTORE_CWD = f"""
CWD_FILE={CWD_STATE_FILE}
if [ -f "$CWD_FILE" ] && [ -d "$(cat "$CWD_FILE" 2>/dev/null)" ]; then
  cd "$(cat "$CWD_FILE")" || cd "$2" || cd /
else
  cd "$2" 2>/dev/null || cd /
fi
"""

# Runs in the same shell that executes the command, so a `cd` in the command is
# reflected by `pwd`. Capturing it in the outer shell instead would always record
# the outer shell's directory and silently lose every `cd` the agent makes.
_INNER_SHELL = f"""
eval "$1"
__ec=$?
pwd > {CWD_STATE_FILE} 2>/dev/null
exit $__ec
"""

# $1=command  $2=workdir  $3=timeout  $4=inner script
_SHELL_WRAPPER = f"""
{_RESTORE_CWD}
if command -v timeout >/dev/null 2>&1; then
  timeout "$3" sh -c "$4" sh "$1"
else
  sh -c "$4" sh "$1"
fi
"""

# read/write resolve relative paths against that same persisted cwd.
_READ_WRAPPER = f"""
{_RESTORE_CWD}
if [ -d "$1" ]; then echo "is a directory" >&2; exit 21; fi
cat -- "$1"
"""

# Content arrives base64-encoded as an argv element rather than on stdin: argv
# works identically through `docker exec` and through docker-py's exec_run (which
# has no clean stdin path), and base64 means no byte the model produced can be
# read as shell syntax. Cost is ARG_MAX -- writes above ~1MB will fail.
_WRITE_WRAPPER = f"""
{_RESTORE_CWD.replace('"$2"', '"$3"')}
command -v base64 >/dev/null 2>&1 || {{ echo "base64 not available in container" >&2; exit 22; }}
mkdir -p "$(dirname "$1")" || exit 1
printf '%s' "$2" | base64 -d > "$1"
"""


class SandboxError(RuntimeError):
    """Anything the executor could not do. tools.py turns these into tool-result
    strings; they never reach the loop."""


@dataclass
class ShellResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


# Never worth walking: huge, generated, and never what is being looked for.
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "target", ".next",
}
MAX_MATCHES = 200


class Executor(Protocol):
    def run_shell(self, command: str, timeout: int = DEFAULT_TIMEOUT) -> ShellResult: ...
    def read_file(self, path: str) -> str: ...
    def write_file(self, path: str, content: str) -> None: ...
    # Listing and searching live on the executor rather than in tools.py because
    # the two backends do them completely differently: the container has find
    # and grep, the host may be Windows and have neither.
    def list_files(self, root: str = ".", limit: int = MAX_MATCHES * 4) -> list[str]: ...
    def search(
        self, pattern: str, root: str = ".", glob: str | None = None,
        limit: int = MAX_MATCHES,
    ) -> list[str]: ...
    def close(self) -> None: ...


def _decode(raw: bytes | None) -> str:
    return (raw or b"").decode("utf-8", errors="replace")


class LocalExecutor:
    """Runs on the host. No isolation -- development and tests only."""

    def __init__(self, workdir: str | os.PathLike[str] | None = None):
        self.workdir = Path(workdir or Path.cwd()).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._cwd = self.workdir

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self._cwd / p)

    def run_shell(self, command: str, timeout: int = DEFAULT_TIMEOUT) -> ShellResult:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self._cwd),
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return ShellResult(
                stdout=_decode(exc.stdout),
                stderr=_decode(exc.stderr) + f"\n[timed out after {timeout}s]",
                exit_code=124,
                timed_out=True,
            )
        except OSError as exc:
            raise SandboxError(f"could not run command: {exc}") from None
        return ShellResult(
            stdout=_decode(proc.stdout),
            stderr=_decode(proc.stderr),
            exit_code=proc.returncode,
        )

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        try:
            return target.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            raise SandboxError(f"no such file: {path}") from None
        except IsADirectoryError:
            raise SandboxError(f"{path} is a directory, not a file") from None
        except PermissionError:
            raise SandboxError(f"permission denied reading {path}") from None
        except OSError as exc:
            raise SandboxError(f"could not read {path}: {exc}") from None

    def write_file(self, path: str, content: str) -> None:
        target = self._resolve(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise SandboxError(f"could not write {path}: {exc}") from None

    def list_files(self, root: str = ".", limit: int = MAX_MATCHES * 4) -> list[str]:
        """Walk in Python, not the shell: the host may be Windows, which has
        neither find nor a compatible grep."""
        base = self._resolve(root)
        if not base.is_dir():
            raise SandboxError(f"no such directory: {root}")

        found: list[str] = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for filename in sorted(filenames):
                rel = Path(dirpath, filename).relative_to(base)
                found.append(rel.as_posix())
                if len(found) >= limit:
                    return found
        return found

    def search(
        self,
        pattern: str,
        root: str = ".",
        glob: str | None = None,
        limit: int = MAX_MATCHES,
    ) -> list[str]:
        regex = re.compile(pattern)
        base = self._resolve(root)
        if not base.is_dir():
            raise SandboxError(f"no such directory: {root}")

        hits: list[str] = []
        for rel in self.list_files(root, limit=MAX_MATCHES * 20):
            if glob and not PurePosixPath(rel).match(glob):
                continue
            try:
                text = (base / rel).read_text(encoding="utf-8", errors="strict")
            except (OSError, UnicodeDecodeError):
                continue  # binary or unreadable; grep -I skips these too
            for number, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{rel}:{number}:{line.strip()[:200]}")
                    if len(hits) >= limit:
                        return hits
        return hits

    def close(self) -> None:
        pass


class DockerExecutor:
    """Everything runs inside a container.

    Pass ``container`` to attach to a container someone else owns; leave it None
    to create and own one.
    """

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        container: str | None = None,
        workdir: str = "/workspace",
        user: str | None = None,
        mounts: list[tuple[str, str]] | None = None,
        memory: str | None = DEFAULT_MEMORY,
        pids_limit: int | None = DEFAULT_PIDS_LIMIT,
        cpus: str | None = DEFAULT_CPUS,
        network: str | None = None,
    ):
        self.image = image
        self.workdir = workdir
        self.user = user
        self.mounts = mounts or []
        self.memory = memory
        self.pids_limit = pids_limit
        self.cpus = cpus
        self.network = network
        self._owns_container = container is None
        self.container = container or f"cli-agent-{uuid.uuid4().hex[:12]}"
        if self._owns_container:
            self._start()
        else:
            self._require_running()

    @staticmethod
    def parse_mount(spec: str, default_target: str = "/workspace") -> tuple[str, str]:
        """Parse HOSTPATH[:CONTAINERPATH].

        Windows paths make this fiddly: 'C:\\src' has a colon that is a drive
        letter, not a separator. Split from the right, and only when what
        follows looks like an absolute container path.
        """
        host, target = spec, default_target
        if ":" in spec:
            head, _, tail = spec.rpartition(":")
            if head and tail.startswith("/"):
                host, target = head, tail

        resolved = Path(host).expanduser().resolve()
        if not resolved.exists():
            raise SandboxError(f"mount source does not exist: {host}")
        if not resolved.is_dir():
            raise SandboxError(f"mount source is not a directory: {host}")
        # Docker Desktop accepts forward slashes on Windows; backslashes are
        # parsed inconsistently across versions.
        return str(resolved).replace("\\", "/"), target

    # -- container lifecycle ------------------------------------------------

    @staticmethod
    def _docker_path() -> str:
        docker = shutil.which("docker")
        if not docker:
            raise SandboxError("docker executable not found on PATH")
        return docker

    def _start(self) -> None:
        docker = self._docker_path()
        argv = [docker, "run", "-d", "--name", self.container, "-w", self.workdir]

        # Labels make orphans findable; see sweep_orphans().
        argv += [
            "--label", f"{CONTAINER_LABEL}=1",
            "--label", f"{CREATED_LABEL}={int(time.time())}",
        ]

        if self.memory:
            argv += ["--memory", self.memory, "--memory-swap", self.memory]
        if self.pids_limit:
            argv += ["--pids-limit", str(self.pids_limit)]
        if self.cpus:
            argv += ["--cpus", str(self.cpus)]
        if self.network:
            argv += ["--network", self.network]
        # Blocks setuid escalation inside the container. Costs nothing here --
        # the agent has no reason to gain privileges it was not started with.
        argv += ["--security-opt", "no-new-privileges"]

        for host, target in self.mounts:
            argv += ["-v", f"{host}:{target}"]
        argv += [
            # No --rm: close() removes it, so a crash leaves the container
            # inspectable instead of silently vanished.
            self.image, "sleep", "infinity",
        ]
        proc = subprocess.run(argv, capture_output=True, timeout=300)
        if proc.returncode != 0:
            raise SandboxError(
                f"could not start sandbox container from image {self.image!r}: "
                f"{_decode(proc.stderr).strip()}"
            )
        self.run_shell(f"mkdir -p {shlex.quote(self.workdir)}")

    @classmethod
    def list_orphans(cls) -> list[tuple[str, int]]:
        """Containers we created that are still around, with their age in
        seconds. Best effort -- returns [] if docker is unreachable."""
        try:
            docker = cls._docker_path()
            proc = subprocess.run(
                [
                    docker, "ps", "-a",
                    "--filter", f"label={CONTAINER_LABEL}",
                    "--format", "{{.ID}}\t{{.Labels}}",
                ],
                capture_output=True,
                timeout=30,
            )
        except (SandboxError, subprocess.SubprocessError, OSError):
            return []
        if proc.returncode != 0:
            return []

        now = int(time.time())
        found: list[tuple[str, int]] = []
        for line in _decode(proc.stdout).splitlines():
            if "\t" not in line:
                continue
            container_id, labels = line.split("\t", 1)
            created = now  # unlabelled: treat as brand new, never sweep
            for label in labels.split(","):
                key, _, value = label.partition("=")
                if key.strip() == CREATED_LABEL:
                    try:
                        created = int(value)
                    except ValueError:
                        pass
            found.append((container_id.strip(), max(0, now - created)))
        return found

    @classmethod
    def sweep_orphans(cls, max_age_seconds: int = DEFAULT_ORPHAN_AGE) -> int:
        """Remove containers we created that outlived their process.

        Age-based rather than liveness-based: there is no reliable way to ask
        whether the python process that owned a container is still alive, and
        killing a container out from under a running session would be far worse
        than leaving a stale one for a few hours. Pass 0 to remove all of them.
        """
        stale = [cid for cid, age in cls.list_orphans() if age >= max_age_seconds]
        if not stale:
            return 0
        try:
            docker = cls._docker_path()
            subprocess.run(
                [docker, "rm", "-f", *stale], capture_output=True, timeout=120
            )
        except (SandboxError, subprocess.SubprocessError, OSError):
            return 0
        return len(stale)

    def _require_running(self) -> None:
        docker = self._docker_path()
        proc = subprocess.run(
            [docker, "inspect", "-f", "{{.State.Running}}", self.container],
            capture_output=True,
            timeout=30,
        )
        if proc.returncode != 0 or _decode(proc.stdout).strip() != "true":
            raise SandboxError(f"container {self.container!r} is not running")

    def close(self) -> None:
        """Only tears down containers we created. An attached container belongs
        to the harness -- killing one mid-benchmark would fail the task."""
        if not self._owns_container:
            return
        try:
            subprocess.run(
                [self._docker_path(), "rm", "-f", self.container],
                capture_output=True,
                timeout=60,
            )
        except (SandboxError, subprocess.SubprocessError, OSError):
            pass

    def __enter__(self) -> DockerExecutor:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- exec ---------------------------------------------------------------

    def _raw_exec(self, argv: list[str], timeout: int = 60) -> tuple[int, str, str]:
        """Run argv inside the container. The single seam subclasses override."""
        docker_argv = [self._docker_path(), "exec"]
        if self.user:
            docker_argv += ["-u", self.user]
        docker_argv += ["-w", self.workdir, self.container, *argv]
        try:
            proc = subprocess.run(docker_argv, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            return 124, _decode(exc.stdout), _decode(exc.stderr)
        except OSError as exc:
            raise SandboxError(f"docker exec failed: {exc}") from None
        return proc.returncode, _decode(proc.stdout), _decode(proc.stderr)

    def run_shell(self, command: str, timeout: int = DEFAULT_TIMEOUT) -> ShellResult:
        exit_code, stdout, stderr = self._raw_exec(
            [
                "sh", "-c", _SHELL_WRAPPER, "sh",
                command, self.workdir, str(timeout), _INNER_SHELL,
            ],
            # Container-side `timeout` should fire first; this is the backstop
            # for a wedged docker client, not the primary mechanism.
            timeout=timeout + 15,
        )
        timed_out = exit_code == 124
        if timed_out:
            stderr = (stderr + f"\n[timed out after {timeout}s]").strip()
        return ShellResult(
            stdout=stdout, stderr=stderr, exit_code=exit_code, timed_out=timed_out
        )

    def read_file(self, path: str) -> str:
        exit_code, stdout, stderr = self._raw_exec(
            ["sh", "-c", _READ_WRAPPER, "sh", path, self.workdir], timeout=60
        )
        if exit_code == 21:
            raise SandboxError(f"{path} is a directory, not a file")
        if exit_code != 0:
            raise SandboxError(
                f"could not read {path}: {stderr.strip() or 'no such file'}"
            )
        return stdout

    def list_files(self, root: str = ".", limit: int = MAX_MATCHES * 4) -> list[str]:
        """find(1) inside the container, with the generated directories pruned."""
        prune = " -o ".join(f"-name {shlex.quote(d)}" for d in sorted(SKIP_DIRS))
        result = self.run_shell(
            f"cd {shlex.quote(root)} 2>/dev/null || exit 9; "
            f"find . \\( {prune} \\) -prune -o -type f -print 2>/dev/null "
            f"| sed 's|^\\./||' | sort | head -n {limit}",
            timeout=60,
        )
        if result.exit_code == 9:
            raise SandboxError(f"no such directory: {root}")
        return [line for line in result.stdout.splitlines() if line.strip()]

    def search(
        self,
        pattern: str,
        root: str = ".",
        glob: str | None = None,
        limit: int = MAX_MATCHES,
    ) -> list[str]:
        excludes = " ".join(f"--exclude-dir={shlex.quote(d)}" for d in sorted(SKIP_DIRS))
        include = f"--include={shlex.quote(glob)}" if glob else ""
        result = self.run_shell(
            f"cd {shlex.quote(root)} 2>/dev/null || exit 9; "
            # -I skips binaries, -E for the same regex flavour Python compiled.
            f"grep -rnI -E {shlex.quote(pattern)} . {excludes} {include} 2>/dev/null "
            f"| sed 's|^\\./||' | head -n {limit}",
            timeout=90,
        )
        if result.exit_code == 9:
            raise SandboxError(f"no such directory: {root}")
        return [line for line in result.stdout.splitlines() if line.strip()]

    def write_file(self, path: str, content: str) -> None:
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        exit_code, _stdout, stderr = self._raw_exec(
            ["sh", "-c", _WRITE_WRAPPER, "sh", path, encoded, self.workdir],
            timeout=120,
        )
        if exit_code == 22:
            raise SandboxError("cannot write files: base64 is not available in the container")
        if exit_code != 0:
            raise SandboxError(
                f"could not write {path}: {stderr.strip() or 'write failed'}"
            )
