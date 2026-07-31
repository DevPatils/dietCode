"""Executor tests. Docker tests skip automatically when the daemon is down."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from agent.sandbox import DockerExecutor, LocalExecutor, SandboxError


def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(
            ["docker", "info"], capture_output=True, timeout=20
        ).returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


requires_docker = pytest.mark.skipif(
    not docker_available(), reason="docker daemon not available"
)


# -- LocalExecutor ----------------------------------------------------------


def test_local_roundtrip(tmp_path):
    ex = LocalExecutor(tmp_path)
    ex.write_file("a/b.txt", "content")
    assert ex.read_file("a/b.txt") == "content"


def test_local_absolute_path(tmp_path):
    ex = LocalExecutor(tmp_path)
    target = tmp_path / "abs.txt"
    ex.write_file(str(target), "x")
    assert target.read_text() == "x"


def test_local_missing_file_raises_sandbox_error(tmp_path):
    with pytest.raises(SandboxError):
        LocalExecutor(tmp_path).read_file("nope")


def test_local_shell_exit_code(tmp_path):
    assert LocalExecutor(tmp_path).run_shell("exit 7").exit_code == 7


def test_local_shell_timeout(tmp_path):
    result = LocalExecutor(tmp_path).run_shell("python -c \"import time; time.sleep(5)\"", timeout=1)
    assert result.timed_out
    assert result.exit_code == 124


def test_unicode_roundtrip(tmp_path):
    ex = LocalExecutor(tmp_path)
    text = "héllo wörld — ünïcode ✓\n"
    ex.write_file("u.txt", text)
    assert ex.read_file("u.txt") == text


# -- DockerExecutor ---------------------------------------------------------


@pytest.fixture(scope="module")
def docker_executor():
    ex = DockerExecutor(image="python:3.11-slim")
    yield ex
    ex.close()


@requires_docker
def test_docker_shell_runs_in_container(docker_executor):
    result = docker_executor.run_shell("cat /etc/os-release")
    assert result.exit_code == 0
    assert "debian" in result.stdout.lower()


@requires_docker
def test_docker_file_roundtrip(docker_executor):
    docker_executor.write_file("/tmp/t/a.txt", "in-container\n")
    assert docker_executor.read_file("/tmp/t/a.txt") == "in-container\n"
    assert "in-container" in docker_executor.run_shell("cat /tmp/t/a.txt").stdout


@requires_docker
def test_docker_write_handles_shell_metacharacters(docker_executor):
    # The exact kind of content a model writes that naive quoting would mangle.
    nasty = "$(rm -rf /) `whoami` 'quoted' \"double\" \\backslash\n; echo pwned\n"
    docker_executor.write_file("/tmp/nasty.txt", nasty)
    assert docker_executor.read_file("/tmp/nasty.txt") == nasty


@requires_docker
def test_docker_working_directory_persists_between_commands(docker_executor):
    """cd in one call must be visible to the next, or the agent acts blind."""
    docker_executor.run_shell("mkdir -p /tmp/persist && cd /tmp/persist")
    assert docker_executor.run_shell("pwd").stdout.strip() == "/tmp/persist"
    docker_executor.write_file("rel.txt", "relative")
    assert docker_executor.read_file("/tmp/persist/rel.txt") == "relative"


@requires_docker
def test_docker_exit_code_survives_the_cwd_wrapper(docker_executor):
    """The wrapper runs the command through `eval`; it must not swallow status."""
    assert docker_executor.run_shell("exit 7").exit_code == 7
    assert docker_executor.run_shell("true").exit_code == 0
    assert docker_executor.run_shell("ls /definitely-not-here").exit_code != 0


@requires_docker
def test_docker_stdout_and_stderr_stay_separate(docker_executor):
    result = docker_executor.run_shell("echo to-out; echo to-err >&2")
    assert "to-out" in result.stdout
    assert "to-err" in result.stderr
    assert "to-err" not in result.stdout


@requires_docker
def test_docker_cd_to_missing_directory_does_not_wedge_the_session(docker_executor):
    docker_executor.run_shell("cd /tmp")
    docker_executor.run_shell("cd /no/such/dir")
    # A failed cd must not leave the saved cwd pointing at nothing.
    assert docker_executor.run_shell("pwd").exit_code == 0


@requires_docker
def test_docker_command_timeout_is_enforced(docker_executor):
    result = docker_executor.run_shell("sleep 30", timeout=2)
    assert result.timed_out
    assert result.exit_code == 124


@requires_docker
def test_docker_missing_file_raises(docker_executor):
    with pytest.raises(SandboxError):
        docker_executor.read_file("/definitely/not/here.txt")


@requires_docker
def test_docker_reading_a_directory_raises(docker_executor):
    with pytest.raises(SandboxError):
        docker_executor.read_file("/tmp")


@requires_docker
def test_attaching_to_a_missing_container_raises():
    with pytest.raises(SandboxError):
        DockerExecutor(container="no-such-container-xyz")


@requires_docker
def test_close_does_not_remove_an_attached_container(docker_executor):
    """Killing the harness's container mid-benchmark would fail the task."""
    attached = DockerExecutor(container=docker_executor.container)
    attached.close()
    assert docker_executor.run_shell("echo alive").exit_code == 0
