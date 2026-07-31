"""Full stack minus the model: real loop, real tools, real container.

The fake client scripts the tool calls a model would make, so this exercises
every layer the CLI uses without an API key.
"""

from __future__ import annotations

import pytest

from agent.loop import agent_loop
from agent.sandbox import DockerExecutor
from tests.fake_llm import FakeClient, tool_call, turn
from tests.test_sandbox import requires_docker

pytestmark = requires_docker


@pytest.fixture(scope="module")
def executor():
    ex = DockerExecutor(image="python:3.11-slim")
    yield ex
    ex.close()


def test_agent_writes_runs_and_verifies_a_program(executor):
    """The shape of a real task: write code, run it, check the output, finish."""
    client = FakeClient(
        [
            turn(
                tool_call(
                    "write_file",
                    {
                        "path": "/workspace/primes.py",
                        "content": (
                            "def primes(n):\n"
                            "    out = []\n"
                            "    c = 2\n"
                            "    while len(out) < n:\n"
                            "        if all(c % p for p in out):\n"
                            "            out.append(c)\n"
                            "        c += 1\n"
                            "    return out\n\n"
                            "print(*primes(10))\n"
                        ),
                    },
                )
            ),
            turn(tool_call("run_shell", {"command": "python /workspace/primes.py"})),
            turn(tool_call("task_complete", {"summary": "primes.py works"})),
        ]
    )

    result = agent_loop("print the first 10 primes", executor, client=client)

    assert result.status == "complete"
    assert result.tool_errors == 0
    # The program really ran in the container.
    assert "2 3 5 7 11 13 17 19 23 29" in executor.run_shell(
        "python /workspace/primes.py"
    ).stdout


def test_agent_recovers_from_a_failing_command(executor):
    """A failed command must come back as readable feedback, not end the run."""
    client = FakeClient(
        [
            turn(tool_call("run_shell", {"command": "python /nope/missing.py"})),
            turn(tool_call("write_file", {"path": "/workspace/ok.py", "content": "print('ok')"})),
            turn(tool_call("run_shell", {"command": "python /workspace/ok.py"})),
            turn(tool_call("task_complete", {"summary": "recovered"})),
        ]
    )

    result = agent_loop("run a script", executor, client=client)

    assert result.status == "complete"
    tool_outputs = [m["content"] for m in result.messages if m["role"] == "tool"]
    assert "exit_code: 0" not in tool_outputs[0]  # first command failed
    assert "ok" in tool_outputs[2]  # and the run continued


def test_directory_changes_carry_across_steps(executor):
    """A model that cds in one step and acts in the next must not act blind."""
    client = FakeClient(
        [
            turn(tool_call("run_shell", {"command": "mkdir -p /workspace/proj && cd /workspace/proj"})),
            turn(tool_call("write_file", {"path": "note.txt", "content": "relative write"})),
            turn(tool_call("run_shell", {"command": "pwd && cat note.txt"})),
            turn(tool_call("task_complete", {"summary": "done"})),
        ]
    )

    result = agent_loop("make a project dir", executor, client=client)

    assert result.status == "complete"
    assert executor.read_file("/workspace/proj/note.txt") == "relative write"
