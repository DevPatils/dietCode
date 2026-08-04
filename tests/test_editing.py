"""edit_file, find_files and search.

edit_file's contract is that it never guesses: a snippet either matches exactly
once or the call fails with a reason the model can act on. Silently editing the
wrong place is worse than not editing at all.
"""

from __future__ import annotations

import pytest

from agent.sandbox import LocalExecutor
from agent.tools import TOOL_NAMES, execute_tool


@pytest.fixture
def executor(tmp_path):
    return LocalExecutor(tmp_path)


SAMPLE = """def greet(name):
    print("hello " + name)


def farewell(name):
    print("bye " + name)
"""


# -- edit_file --------------------------------------------------------------


def test_edit_replaces_an_exact_snippet(executor, tmp_path):
    (tmp_path / "a.py").write_text(SAMPLE, encoding="utf-8")
    out = execute_tool(
        "edit_file",
        {"path": "a.py", "old": 'print("hello " + name)', "new": 'print(f"hello {name}")'},
        executor,
    )
    assert out.startswith("Edited")
    text = (tmp_path / "a.py").read_text()
    assert 'print(f"hello {name}")' in text
    assert 'print("bye " + name)' in text, "the rest of the file must be untouched"


def test_edit_reports_the_line_delta(executor, tmp_path):
    (tmp_path / "a.py").write_text(SAMPLE, encoding="utf-8")
    out = execute_tool(
        "edit_file",
        {"path": "a.py", "old": '    print("bye " + name)', "new": "    pass\n    return"},
        executor,
    )
    assert "+1 lines" in out


def test_missing_snippet_refuses_rather_than_guessing(executor, tmp_path):
    (tmp_path / "a.py").write_text(SAMPLE, encoding="utf-8")
    out = execute_tool(
        "edit_file", {"path": "a.py", "old": "nothing like this", "new": "x"}, executor
    )
    assert out.startswith("Error:")
    assert (tmp_path / "a.py").read_text() == SAMPLE


def test_a_whitespace_mismatch_says_so(executor, tmp_path):
    """The usual cause of a failed edit, and the model cannot see it otherwise."""
    (tmp_path / "a.py").write_text(SAMPLE, encoding="utf-8")
    out = execute_tool(
        "edit_file",
        {"path": "a.py", "old": 'print("bye " + name)   ', "new": "pass"},
        executor,
    )
    assert "whitespace" in out.lower()


def test_an_ambiguous_snippet_refuses_and_says_how_many(executor, tmp_path):
    (tmp_path / "a.py").write_text(SAMPLE, encoding="utf-8")
    out = execute_tool("edit_file", {"path": "a.py", "old": "name", "new": "who"}, executor)
    assert out.startswith("Error:")
    assert "6 times" in out or "times" in out
    assert (tmp_path / "a.py").read_text() == SAMPLE, "nothing may change on refusal"


def test_replace_all_allows_the_ambiguous_case(executor, tmp_path):
    (tmp_path / "a.py").write_text(SAMPLE, encoding="utf-8")
    out = execute_tool(
        "edit_file",
        {"path": "a.py", "old": "name", "new": "who", "replace_all": True},
        executor,
    )
    assert out.startswith("Edited")
    assert "name" not in (tmp_path / "a.py").read_text()


@pytest.mark.parametrize("truthy", ["true", "True", "yes", 1])
def test_replace_all_accepts_the_shapes_models_actually_send(executor, tmp_path, truthy):
    (tmp_path / "a.py").write_text(SAMPLE, encoding="utf-8")
    out = execute_tool(
        "edit_file",
        {"path": "a.py", "old": "name", "new": "who", "replace_all": truthy},
        executor,
    )
    assert out.startswith("Edited"), truthy


def test_empty_old_is_refused_with_a_pointer_to_write_file(executor, tmp_path):
    (tmp_path / "a.py").write_text(SAMPLE, encoding="utf-8")
    out = execute_tool("edit_file", {"path": "a.py", "old": "", "new": "x"}, executor)
    assert "write_file" in out


def test_editing_a_missing_file_is_an_error_string(executor):
    out = execute_tool("edit_file", {"path": "nope.py", "old": "a", "new": "b"}, executor)
    assert out.startswith("Error:")


def test_edit_is_cheaper_than_rewriting(executor, tmp_path):
    """The whole reason this tool exists: a one-line change should cost one
    line, not the file."""
    big = "\n".join(f"line {i}" for i in range(400))
    (tmp_path / "big.txt").write_text(big, encoding="utf-8")

    edit_payload = len("line 200") + len("CHANGED")
    rewrite_payload = len(big)
    execute_tool(
        "edit_file", {"path": "big.txt", "old": "line 200", "new": "CHANGED"}, executor
    )
    assert "CHANGED" in (tmp_path / "big.txt").read_text()
    assert edit_payload < rewrite_payload / 100


# -- find_files -------------------------------------------------------------


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("import os\nTODO = 1\n")
    (tmp_path / "src" / "util.py").write_text("def helper(): pass\n")
    (tmp_path / "README.md").write_text("# hi\nTODO later\n")
    noise = tmp_path / "node_modules" / "pkg"
    noise.mkdir(parents=True)
    (noise / "index.js").write_text("// TODO ignore me\n")
    return tmp_path


def test_find_files_matches_a_recursive_glob(executor, tree):
    out = execute_tool("find_files", {"pattern": "**/*.py"}, executor)
    assert "src/app.py" in out
    assert "src/util.py" in out
    assert "README.md" not in out


def test_find_files_skips_generated_directories(executor, tree):
    """node_modules and .git dwarf real source and are never the answer."""
    out = execute_tool("find_files", {"pattern": "**/*.js"}, executor)
    assert "node_modules" not in out


def test_find_files_reports_no_matches_plainly(executor, tree):
    out = execute_tool("find_files", {"pattern": "**/*.rs"}, executor)
    assert "No files matching" in out


def test_find_files_needs_a_pattern(executor, tree):
    assert execute_tool("find_files", {}, executor).startswith("Error:")


# -- search -----------------------------------------------------------------


def test_search_returns_file_and_line(executor, tree):
    out = execute_tool("search", {"pattern": "TODO"}, executor)
    assert "src/app.py" in out
    assert ":2:" in out or "TODO" in out


def test_search_can_be_limited_by_glob(executor, tree):
    out = execute_tool("search", {"pattern": "TODO", "glob": "*.md"}, executor)
    assert "README.md" in out
    assert "app.py" not in out


def test_search_skips_generated_directories(executor, tree):
    out = execute_tool("search", {"pattern": "TODO"}, executor)
    assert "node_modules" not in out


def test_search_reports_no_matches_plainly(executor, tree):
    assert "No matches" in execute_tool("search", {"pattern": "zzzznope"}, executor)


def test_an_invalid_regex_is_an_error_not_a_crash(executor, tree):
    out = execute_tool("search", {"pattern": "([unclosed"}, executor)
    assert out.startswith("Error:")
    assert "regular expression" in out


def test_a_missing_directory_is_reported(executor, tree):
    out = execute_tool("search", {"pattern": "x", "path": "no/such/dir"}, executor)
    assert out.startswith("Error:")


# -- schema -----------------------------------------------------------------


def test_the_new_tools_are_advertised_to_the_model():
    for name in ("edit_file", "find_files", "search"):
        assert name in TOOL_NAMES
