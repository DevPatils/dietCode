"""Dispatch tests.

Most of these are malformed-input cases. The dispatcher's contract is that it
never raises, so every one of them asserts on a returned error string.
"""

from __future__ import annotations

import json

import pytest

from agent.sandbox import LocalExecutor, SandboxError
from agent.tools import (
    MAX_TOOL_RESULT_CHARS,
    TOOL_NAMES,
    execute_tool,
    extract_tool_calls_from_text,
    parse_arguments,
    truncate,
)


@pytest.fixture
def executor(tmp_path):
    return LocalExecutor(tmp_path)


# -- happy paths ------------------------------------------------------------


def test_write_then_read_roundtrip(executor):
    out = execute_tool("write_file", {"path": "a.txt", "content": "hello\nworld"}, executor)
    assert "Wrote" in out
    assert execute_tool("read_file", {"path": "a.txt"}, executor) == "hello\nworld"


def test_write_creates_parent_directories(executor):
    execute_tool("write_file", {"path": "deep/nested/a.txt", "content": "x"}, executor)
    assert execute_tool("read_file", {"path": "deep/nested/a.txt"}, executor) == "x"


def test_arguments_accepted_as_json_string(executor):
    out = execute_tool("write_file", '{"path": "b.txt", "content": "hi"}', executor)
    assert "Wrote" in out


def test_run_shell_reports_exit_code(executor):
    out = execute_tool("run_shell", {"command": "exit 3"}, executor)
    assert "exit_code: 3" in out


def test_run_shell_captures_stdout(executor):
    out = execute_tool("run_shell", {"command": "echo marker-value"}, executor)
    assert "marker-value" in out
    assert "exit_code: 0" in out


def test_task_complete_returns_summary(executor):
    assert execute_tool("task_complete", {"summary": "did it"}, executor) == "did it"


# -- malformed model output -------------------------------------------------


def test_unknown_tool_name_lists_valid_tools(executor):
    out = execute_tool("delete_everything", {}, executor)
    assert out.startswith("Error:")
    for name in TOOL_NAMES:
        assert name in out


def test_malformed_json_arguments(executor):
    out = execute_tool("read_file", '{"path": "a.txt"', executor)
    assert out.startswith("Error:")
    assert "valid JSON" in out


def test_json_arguments_that_are_not_an_object(executor):
    out = execute_tool("read_file", '"just a string"', executor)
    assert out.startswith("Error:")


def test_missing_required_argument(executor):
    out = execute_tool("read_file", {}, executor)
    assert out.startswith("Error:")
    assert "path" in out


def test_null_argument(executor):
    assert execute_tool("read_file", {"path": None}, executor).startswith("Error:")


def test_wrong_typed_argument_is_rejected(executor):
    out = execute_tool("write_file", {"path": ["a", "b"], "content": "x"}, executor)
    assert out.startswith("Error:")


def test_numeric_argument_is_coerced(executor):
    # A model sending 42 where a string belongs is recoverable; don't fail it.
    out = execute_tool("write_file", {"path": "42", "content": 42}, executor)
    assert "Wrote" in out


def test_non_string_tool_name(executor):
    assert execute_tool(None, {}, executor).startswith("Error:")


def test_missing_file_is_an_error_string_not_an_exception(executor):
    out = execute_tool("read_file", {"path": "nope.txt"}, executor)
    assert out.startswith("Error:")
    assert "no such file" in out


def test_reading_a_directory_is_an_error_string(executor, tmp_path):
    (tmp_path / "adir").mkdir()
    assert execute_tool("read_file", {"path": "adir"}, executor).startswith("Error:")


def test_executor_exceptions_never_escape():
    class BrokenExecutor:
        def read_file(self, path):
            raise SandboxError("disk on fire")

        def write_file(self, path, content):
            raise ValueError("unexpected non-sandbox error")

        def run_shell(self, command, timeout=30):
            raise RuntimeError("boom")

        def close(self):
            pass

    broken = BrokenExecutor()
    assert execute_tool("read_file", {"path": "x"}, broken).startswith("Error:")
    assert execute_tool("write_file", {"path": "x", "content": "y"}, broken).startswith("Error:")
    assert execute_tool("run_shell", {"command": "x"}, broken).startswith("Error:")


# -- timeouts and truncation ------------------------------------------------


def test_invalid_timeout_falls_back_to_default(executor):
    out = execute_tool("run_shell", {"command": "echo ok", "timeout": "not-a-number"}, executor)
    assert "exit_code: 0" in out


def test_absurd_timeout_is_clamped(executor):
    # A 1-hour timeout from the model would stall a whole benchmark task.
    out = execute_tool("run_shell", {"command": "echo ok", "timeout": 999999}, executor)
    assert "exit_code: 0" in out


def test_truncate_keeps_head_and_tail():
    text = "A" * 100 + "B" * 100
    out = truncate(text, limit=40)
    assert out.startswith("A")
    assert out.endswith("B")
    assert "omitted" in out
    assert len(out) < len(text)


def test_short_text_is_not_truncated():
    assert truncate("short", limit=100) == "short"


def test_large_tool_output_is_capped(executor):
    executor.write_file("big.txt", "x" * (MAX_TOOL_RESULT_CHARS * 3))
    out = execute_tool("read_file", {"path": "big.txt"}, executor)
    assert len(out) < MAX_TOOL_RESULT_CHARS * 2
    assert "omitted" in out


# -- argument parsing -------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ({"a": 1}, {"a": 1}),
        ('{"a": 1}', {"a": 1}),
        ("", {}),
        (None, {}),
    ],
)
def test_parse_arguments_accepts(raw, expected):
    args, err = parse_arguments(raw)
    assert err is None
    assert args == expected


@pytest.mark.parametrize("raw", ["{bad", "[1,2]", "null", "5", 5])
def test_parse_arguments_rejects(raw):
    args, err = parse_arguments(raw)
    assert err is not None
    assert args is None


# -- recovering tool calls written as text ----------------------------------
#
# Open models intermittently emit the call as prose instead of using the
# tool_calls field. Every format below has been seen in the wild.


def test_recovers_the_format_that_broke_the_first_real_run():
    """Verbatim from a llama-3.3-70b run: the call arrived as content text."""
    content = (
        '<function/run_shell {"command": "python -c \\"print(1)\\""} </function>'
    )
    calls = extract_tool_calls_from_text(content)
    assert len(calls) == 1
    assert calls[0]["name"] == "run_shell"
    assert json.loads(calls[0]["arguments"])["command"] == 'python -c "print(1)"'


@pytest.mark.parametrize(
    "content",
    [
        '<function=run_shell>{"command": "ls"}</function>',
        '<function/run_shell {"command": "ls"} </function>',
        '<function: run_shell> {"command": "ls"}',
        '<|python_tag|>{"name": "run_shell", "parameters": {"command": "ls"}}',
        '{"name": "run_shell", "arguments": {"command": "ls"}}',
        '```json\n{"name": "run_shell", "parameters": {"command": "ls"}}\n```',
        'Let me look around.\n<function=run_shell>{"command": "ls"}</function>',
    ],
)
def test_recovers_each_known_text_format(content):
    calls = extract_tool_calls_from_text(content)
    assert len(calls) == 1
    assert calls[0]["name"] == "run_shell"
    assert json.loads(calls[0]["arguments"]) == {"command": "ls"}


def test_recovers_content_containing_braces():
    """A regex would truncate at the first closing brace inside the string."""
    content = (
        '<function=write_file>{"path": "a.py", '
        '"content": "def f():\\n    return {\\"k\\": [1, 2]}\\n"}</function>'
    )
    calls = extract_tool_calls_from_text(content)
    assert len(calls) == 1
    args = json.loads(calls[0]["arguments"])
    assert args["path"] == "a.py"
    assert args["content"] == 'def f():\n    return {"k": [1, 2]}\n'


def test_recovers_multiple_calls_in_one_message():
    content = (
        '<function=write_file>{"path": "a", "content": "x"}</function>\n'
        '<function=run_shell>{"command": "cat a"}</function>'
    )
    calls = extract_tool_calls_from_text(content)
    assert [c["name"] for c in calls] == ["write_file", "run_shell"]


@pytest.mark.parametrize(
    "content",
    [
        "",
        None,
        "I will now run ls to see the files.",
        "The run_shell tool takes a command argument.",  # prose naming a tool
        '<function=nonexistent_tool>{"a": 1}</function>',  # not a real tool
        '<function=run_shell>not json at all</function>',
        '{"unrelated": "object"}',
        "Here is some JSON: {\"a\": 1}",
    ],
)
def test_does_not_invent_calls_from_ordinary_text(content):
    assert extract_tool_calls_from_text(content) == []
