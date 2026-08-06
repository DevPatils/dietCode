"""Checkpoints before edits.

The gate decides whether an edit happens; a snapshot answers what if it should
not have. Together they are what makes a looser gate mode survivable.
"""

from __future__ import annotations

import pytest

from agent.sandbox import LocalExecutor
from agent.snapshots import MAX_SNAPSHOT_BYTES, SnapshotStore, Snapshotting


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("DIETCODE_HOME", str(tmp_path / "home"))


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    return root


@pytest.fixture
def executor(project):
    """A snapshotting executor over a real directory."""
    store = SnapshotStore(project=str(project), session_id="test")
    return Snapshotting(LocalExecutor(project), store), store, project


# -- capturing --------------------------------------------------------------


def test_overwriting_a_file_saves_what_was_there(executor):
    wrapped, store, project = executor
    (project / "a.py").write_text("original", encoding="utf-8")

    wrapped.write_file("a.py", "replaced")

    assert (project / "a.py").read_text() == "replaced"
    assert len(store.changes) == 1
    assert store.changes[0].backup.read_text(encoding="utf-8") == "original"


def test_creating_a_file_is_recorded_as_a_creation(executor):
    """Undoing a create means deleting it, so the absence has to be recorded."""
    wrapped, store, _project = executor
    wrapped.write_file("new.py", "hello")

    assert store.changes[0].existed is False
    assert store.changes[0].backup is None


def test_snapshots_do_not_land_in_the_project(executor):
    wrapped, store, project = executor
    (project / "a.py").write_text("original", encoding="utf-8")
    wrapped.write_file("a.py", "replaced")

    assert str(project) not in str(store.changes[0].backup)
    assert [p.name for p in project.iterdir()] == ["a.py"]


def test_a_huge_file_is_not_copied(executor):
    """Past a point the copy costs more than the safety is worth."""
    wrapped, store, project = executor
    (project / "big.bin").write_text("x" * (MAX_SNAPSHOT_BYTES + 10), encoding="utf-8")

    wrapped.write_file("big.bin", "small")
    assert store.changes[0].backup is None


def test_two_files_with_the_same_name_do_not_collide(executor):
    wrapped, store, project = executor
    (project / "one").mkdir()
    (project / "two").mkdir()
    (project / "one" / "x.py").write_text("first", encoding="utf-8")
    (project / "two" / "x.py").write_text("second", encoding="utf-8")

    wrapped.write_file("one/x.py", "a")
    wrapped.write_file("two/x.py", "b")

    saved = {c.backup.read_text(encoding="utf-8") for c in store.changes}
    assert saved == {"first", "second"}


def test_a_snapshot_that_cannot_be_written_does_not_block_the_edit(executor):
    """The user asked for the edit; the checkpoint is the extra."""
    wrapped, store, project = executor
    (project / "a.py").write_text("original", encoding="utf-8")
    # A plain file where the snapshot directory needs to be, so creating it
    # fails the way a full or read-only disk would.
    store.root.parent.mkdir(parents=True, exist_ok=True)
    store.root.write_text("in the way", encoding="utf-8")

    wrapped.write_file("a.py", "replaced")

    assert (project / "a.py").read_text() == "replaced"
    assert store.enabled is False


def test_snapshots_can_be_turned_off(project):
    store = SnapshotStore(project=str(project), session_id="t", enabled=False)
    wrapped = Snapshotting(LocalExecutor(project), store)
    (project / "a.py").write_text("original", encoding="utf-8")

    wrapped.write_file("a.py", "replaced")
    assert store.changes == []


def test_shell_commands_are_not_snapshotted(executor):
    """Guessing which paths a command will write is worse than being honest
    that undo covers file tools only."""
    wrapped, store, _project = executor
    wrapped.run_shell("echo hi")
    assert store.changes == []


def test_everything_else_reaches_the_wrapped_executor(executor):
    wrapped, _store, project = executor
    assert wrapped.read_file  # forwarded
    assert str(wrapped.workdir) == str(project)


# -- restoring --------------------------------------------------------------


def test_undo_puts_the_old_contents_back(executor):
    wrapped, store, project = executor
    (project / "a.py").write_text("original", encoding="utf-8")
    wrapped.write_file("a.py", "broken")

    store.undo_last(wrapped)
    assert (project / "a.py").read_text() == "original"


def test_undoing_a_created_file_removes_it(executor):
    wrapped, store, project = executor
    wrapped.write_file("new.py", "hello")

    store.undo_last(wrapped)
    assert not (project / "new.py").exists()


def test_undo_all_walks_back_to_the_oldest_state(executor):
    """A file written twice must end up as it was before the first write."""
    wrapped, store, project = executor
    (project / "a.py").write_text("v0", encoding="utf-8")
    wrapped.write_file("a.py", "v1")
    wrapped.write_file("a.py", "v2")

    store.undo_all(wrapped)
    assert (project / "a.py").read_text() == "v0"


def test_undo_all_reverts_every_file(executor):
    wrapped, store, project = executor
    (project / "a.py").write_text("a0", encoding="utf-8")
    (project / "b.py").write_text("b0", encoding="utf-8")
    wrapped.write_file("a.py", "a1")
    wrapped.write_file("b.py", "b1")
    wrapped.write_file("c.py", "new")

    store.undo_all(wrapped)
    assert (project / "a.py").read_text() == "a0"
    assert (project / "b.py").read_text() == "b0"
    assert not (project / "c.py").exists()


def test_undo_consumes_the_change(executor):
    wrapped, store, project = executor
    (project / "a.py").write_text("original", encoding="utf-8")
    wrapped.write_file("a.py", "broken")

    store.undo_last(wrapped)
    assert store.changes == []
    assert store.undo_last(wrapped) == []


def test_undo_reports_what_it_did(executor):
    wrapped, store, project = executor
    (project / "a.py").write_text("original", encoding="utf-8")
    wrapped.write_file("a.py", "broken")

    lines = store.undo_last(wrapped)
    assert len(lines) == 1 and "a.py" in lines[0]


def test_a_missing_backup_is_reported_not_crashed(executor):
    wrapped, store, project = executor
    (project / "a.py").write_text("original", encoding="utf-8")
    wrapped.write_file("a.py", "broken")
    store.changes[0].backup.unlink()

    lines = store.undo_last(wrapped)
    assert "no saved copy" in lines[0]
    assert (project / "a.py").read_text() == "broken"  # unchanged, not emptied


def test_the_edit_tool_is_covered_by_the_same_interception(executor):
    """edit_file reads, substitutes and writes back through write_file, so
    there is no second code path to snapshot."""
    from agent.tools import execute_tool

    wrapped, store, project = executor
    (project / "a.py").write_text("keep\nchange me\n", encoding="utf-8")

    execute_tool("edit_file", {"path": "a.py", "old": "change me", "new": "changed"}, wrapped)

    assert len(store.changes) == 1
    store.undo_last(wrapped)
    assert (project / "a.py").read_text() == "keep\nchange me\n"


# -- ordering against the permission gate -----------------------------------


def test_the_gate_asks_before_a_snapshot_is_taken(project):
    """Snapshotting sits inside the gate: a denied write must not leave a
    checkpoint behind, and taking one must never trigger a read prompt."""
    from agent.permissions import Decision, PermissionGate
    from agent.sandbox import SandboxError

    store = SnapshotStore(project=str(project), session_id="t")
    inner = Snapshotting(LocalExecutor(project), store)
    asked = []

    def approver(request):
        asked.append(request)
        return Decision.NO

    gate = PermissionGate(inner, root=project, approver=approver)
    (project / "a.py").write_text("original", encoding="utf-8")

    with pytest.raises(SandboxError):
        gate.write_file("a.py", "replaced")

    assert len(asked) == 1, "one prompt: for the write, not for reading it first"
    assert store.changes == [], "a denied write leaves no checkpoint"
    assert (project / "a.py").read_text() == "original"
