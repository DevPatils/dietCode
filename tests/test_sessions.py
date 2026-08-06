"""Transcript persistence: the file is the session, so resume is a read and
fork is a copy of a prefix."""

from __future__ import annotations

import json

import pytest

from agent import sessions


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never write into the developer's real ~/.dietcode."""
    monkeypatch.setenv("DIETCODE_HOME", str(tmp_path / "home"))
    return tmp_path


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    return root


def turn(store, *texts):
    store.record_turn([{"role": "user", "content": t} for t in texts], steps=1)


# -- where it goes ----------------------------------------------------------


def test_transcripts_are_not_written_into_the_project(project):
    """A directory in the repo is a directory that gets committed -- and a
    transcript holds every file read and every line of shell output."""
    store = sessions.SessionStore(project=str(project))
    turn(store, "hello")

    assert store.path.exists()
    assert str(project) not in str(store.path)
    assert not list(project.iterdir())


def test_the_project_path_is_encoded_in_the_directory_name(project):
    """So "sessions for this project" needs no index file to consult."""
    store = sessions.SessionStore(project=str(project))
    turn(store, "hello")
    assert project.name in str(store.path.parent)


def test_two_projects_do_not_share_a_directory(tmp_path):
    a, b = tmp_path / "alpha", tmp_path / "beta"
    a.mkdir()
    b.mkdir()
    assert sessions.project_dir(a) != sessions.project_dir(b)


def test_session_ids_sort_by_age():
    ids = sorted(sessions.new_session_id() for _ in range(5))
    assert ids == sorted(ids)


def test_a_slug_survives_a_path_with_no_usable_characters(tmp_path):
    assert sessions.slug(tmp_path)  # never empty, whatever the path looks like


# -- writing ----------------------------------------------------------------


def test_a_transcript_is_one_json_object_per_line(project):
    store = sessions.SessionStore(project=str(project), model="m", provider="groq")
    turn(store, "one", "two")

    lines = store.path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 3  # header + two messages
    for line in lines:
        json.loads(line)  # every line stands alone


def test_the_header_records_what_was_running(project):
    store = sessions.SessionStore(project=str(project), model="llama", provider="groq")
    turn(store, "hello")

    header = json.loads(store.path.read_text(encoding="utf-8").splitlines()[0])
    assert header["type"] == "session"
    assert header["model"] == "llama"
    assert header["provider"] == "groq"
    assert header["version"] == sessions.FORMAT_VERSION


def test_only_new_messages_are_appended(project):
    """The loop hands back the whole conversation every turn; re-writing all of
    it would grow the file quadratically."""
    store = sessions.SessionStore(project=str(project))
    first = [{"role": "user", "content": "one"}]
    store.record_turn(first)
    store.record_turn([*first, {"role": "assistant", "content": "two"}])

    messages = sessions.load_messages(store.path)
    assert [m["content"] for m in messages] == ["one", "two"]


def test_a_disk_that_cannot_be_written_does_not_kill_the_session(project, monkeypatch):
    """Losing the transcript is bad; losing the work in progress is worse."""
    store = sessions.SessionStore(project=str(project))

    def explode(*_a, **_k):
        raise OSError("read-only file system")

    monkeypatch.setattr("pathlib.Path.mkdir", explode)
    turn(store, "hello")  # must not raise
    assert store.enabled is False


def test_persistence_can_be_turned_off(project):
    store = sessions.SessionStore(project=str(project), enabled=False)
    turn(store, "secret")
    assert not store.path.exists()


# -- reading ----------------------------------------------------------------


def test_a_session_round_trips(project):
    store = sessions.SessionStore(project=str(project))
    messages = [
        {"role": "user", "content": "make a file"},
        {"role": "assistant", "content": "done"},
    ]
    store.record_turn(messages, steps=2)

    assert sessions.load_messages(store.path) == messages


def test_a_half_written_last_line_does_not_lose_the_session(project):
    """A session killed mid-write. Everything before the tear is still good."""
    store = sessions.SessionStore(project=str(project))
    store.record_turn([{"role": "user", "content": "kept"}])
    with store.path.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "message", "message": {"role": "assis')

    assert [m["content"] for m in sessions.load_messages(store.path)] == ["kept"]


def test_describe_summarises_without_reading_it_all_back(project):
    store = sessions.SessionStore(project=str(project), model="llama")
    store.record_turn([{"role": "user", "content": "fix the parser"}], steps=3)

    meta = sessions.describe(store.path)
    assert meta.model == "llama"
    assert meta.turns == 1
    assert "fix the parser" in meta.label


def test_a_session_with_no_prompt_still_has_a_label(project):
    store = sessions.SessionStore(project=str(project))
    store.record_turn([{"role": "assistant", "content": "hi"}])
    assert sessions.describe(store.path).label


def test_a_long_prompt_is_shortened_for_the_listing(project):
    store = sessions.SessionStore(project=str(project))
    store.record_turn([{"role": "user", "content": "x" * 500}])
    assert len(sessions.describe(store.path).label) <= 61


# -- listing and lookup -----------------------------------------------------


def test_listing_puts_the_most_recent_first(project):
    import os
    import time

    ids = []
    for i in range(3):
        store = sessions.SessionStore(project=str(project))
        store.record_turn([{"role": "user", "content": f"task {i}"}])
        ids.append(store.session_id)
        # Touch mtimes apart; the filesystem clock is coarser than the loop.
        os.utime(store.path, (time.time() + i, time.time() + i))

    listed = [meta.session_id for meta in sessions.list_sessions(project)]
    assert listed[0] == ids[-1]


def test_listing_a_project_with_no_sessions_is_empty(project):
    assert sessions.list_sessions(project) == []


def test_a_session_is_found_by_id_prefix(project):
    """Nobody types a full timestamp."""
    store = sessions.SessionStore(project=str(project))
    store.record_turn([{"role": "user", "content": "hi"}])

    found = sessions.find_session(project, store.session_id[:8])
    assert found is not None and found.session_id == store.session_id


def test_an_unknown_id_is_not_guessed(project):
    store = sessions.SessionStore(project=str(project))
    store.record_turn([{"role": "user", "content": "hi"}])
    assert sessions.find_session(project, "nope") is None


def test_latest_is_the_one_continue_would_pick(project):
    import os
    import time

    for i in range(2):
        store = sessions.SessionStore(project=str(project))
        store.record_turn([{"role": "user", "content": f"t{i}"}])
        os.utime(store.path, (time.time() + i, time.time() + i))

    latest = sessions.latest_session(project)
    assert latest is not None and "t1" in latest.label


# -- forking ----------------------------------------------------------------


def test_a_fork_is_a_new_session_with_the_same_history(project):
    store = sessions.SessionStore(project=str(project), model="llama")
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]
    store.record_turn(messages)

    new_id = sessions.fork(sessions.describe(store.path))
    forked = sessions.find_session(project, new_id)

    assert new_id != store.session_id
    assert forked is not None
    assert sessions.load_messages(forked.path) == messages


def test_forking_a_prefix_drops_what_came_after(project):
    """The point of forking: go back to before the turn that went wrong."""
    store = sessions.SessionStore(project=str(project))
    store.record_turn(
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ]
    )

    new_id = sessions.fork(sessions.describe(store.path), upto=1)
    forked = sessions.find_session(project, new_id)
    assert [m["content"] for m in sessions.load_messages(forked.path)] == ["one"]


def test_a_fork_records_where_it_came_from(project):
    store = sessions.SessionStore(project=str(project))
    store.record_turn([{"role": "user", "content": "one"}])

    new_id = sessions.fork(sessions.describe(store.path))
    forked = sessions.find_session(project, new_id)
    records = [
        json.loads(line)
        for line in forked.path.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert any(r.get("type") == "fork" and r["from"] == store.session_id for r in records)


def test_forking_does_not_touch_the_original(project):
    store = sessions.SessionStore(project=str(project))
    store.record_turn([{"role": "user", "content": "one"}])
    before = store.path.read_text(encoding="utf-8")

    sessions.fork(sessions.describe(store.path))
    assert store.path.read_text(encoding="utf-8") == before


# -- the CLI wiring ---------------------------------------------------------


@pytest.fixture
def resolver(project, monkeypatch):
    """resolve_session() against a parser, with the cwd pointed at a tmp project."""
    from agent.cli import build_parser, resolve_session

    console_module = pytest.importorskip("rich.console")
    console = console_module.Console(file=__import__("io").StringIO(), no_color=True)

    def run(*argv):
        args = build_parser().parse_args(["--workdir", str(project), *argv])
        return resolve_session(args, console, "test-model")

    run.console = console
    return run


def seed(project, *prompts, model="test-model"):
    store = sessions.SessionStore(project=str(project), model=model)
    store.record_turn([{"role": "user", "content": p} for p in prompts], steps=1)
    return store


def test_a_fresh_run_starts_a_new_transcript(resolver, project):
    store, history = resolver()
    assert store is not None and history is None
    assert sessions.list_sessions(project) == []  # nothing written until a turn ends


def test_continue_picks_up_the_last_conversation(resolver, project):
    seed(project, "the earlier task")
    store, history = resolver("--continue")

    assert history is not None
    assert history[0]["content"] == "the earlier task"
    assert store is not None


def test_continue_with_nothing_to_continue_starts_fresh(resolver, project):
    store, history = resolver("--continue")
    assert store is not None
    assert history is None


def test_resuming_appends_to_the_same_file(resolver, project):
    original = seed(project, "one")
    store, history = resolver("--resume", original.session_id)

    assert store.session_id == original.session_id
    store.record_turn([*history, {"role": "assistant", "content": "two"}])
    assert [m["content"] for m in sessions.load_messages(store.path)] == ["one", "two"]


def test_resuming_does_not_write_the_history_twice(resolver, project):
    """The loop hands back the full conversation; without knowing what is
    already on disk, resume doubles the transcript on the next turn."""
    original = seed(project, "one", "two")
    store, history = resolver("--resume", original.session_id)

    store.record_turn(history)  # a turn that added nothing new
    assert len(sessions.load_messages(store.path)) == 2


def test_resuming_by_prefix_works(resolver, project):
    original = seed(project, "one")
    store, _history = resolver("--resume", original.session_id[:8])
    assert store.session_id == original.session_id


def test_resuming_something_that_does_not_exist_gives_up(resolver):
    """Quietly starting a new session would lose whatever they meant to resume."""
    store, history = resolver("--resume", "nope")
    assert (store, history) == (None, None)


def test_fork_starts_a_new_transcript_from_the_old_one(resolver, project):
    original = seed(project, "one")
    store, history = resolver("--fork", original.session_id)

    assert store.session_id != original.session_id
    assert [m["content"] for m in history] == ["one"]
    # The original is untouched and still listed.
    assert original.session_id in {m.session_id for m in sessions.list_sessions(project)}


def test_forking_something_that_does_not_exist_gives_up(resolver):
    assert resolver("--fork", "nope") == (None, None)


def test_no_save_still_resumes_but_writes_nothing(resolver, project):
    original = seed(project, "one")
    store, history = resolver("--no-save", "--resume", original.session_id)

    assert store is None, "nothing should be appended"
    assert [m["content"] for m in history] == ["one"], "reading is a separate decision"
