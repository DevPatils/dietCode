"""Credential storage, provider resolution, and the login/logout commands."""

from __future__ import annotations

import io
import json
import stat
import sys

import pytest
from rich.console import Console

from agent import auth
from agent.cli import build_parser, resolve_model_config
from agent.commands import auth_status, login, logout


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never touch the developer's real keychain or ~/.dietcode."""
    monkeypatch.setattr(auth, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(auth, "CREDENTIALS_FILE", tmp_path / "credentials.json")
    monkeypatch.setattr(auth, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(auth, "_keyring", lambda: None)  # exercise the file store
    for spec in auth.PROVIDERS.values():
        monkeypatch.delenv(spec.env_var, raising=False)
    monkeypatch.delenv("DIETCODE_PROVIDER", raising=False)


@pytest.fixture
def console():
    return Console(file=io.StringIO(), width=100, no_color=True)


def output(console: Console) -> str:
    return console.file.getvalue()


# -- storage ----------------------------------------------------------------


def test_saved_key_round_trips():
    auth.store_key("groq", "gsk_secret")
    assert auth.stored_key("groq") == "gsk_secret"


def test_credentials_file_is_owner_only(tmp_path):
    auth.store_key("groq", "gsk_secret")
    mode = (tmp_path / "credentials.json").stat().st_mode
    if sys.platform != "win32":  # Windows uses ACLs, not mode bits
        assert not mode & stat.S_IRGRP
        assert not mode & stat.S_IROTH


def test_keys_are_not_stored_in_the_project_directory(tmp_path):
    """A key in the repo is a key that gets committed."""
    auth.store_key("groq", "gsk_secret")
    assert (tmp_path / "credentials.json").exists()
    assert str(auth.CREDENTIALS_FILE).startswith(str(tmp_path))


def test_logout_removes_the_key():
    auth.store_key("groq", "gsk_secret")
    assert auth.delete_key("groq") is True
    assert auth.stored_key("groq") is None


def test_logout_is_idempotent():
    assert auth.delete_key("groq") is False


def test_keys_are_kept_per_provider():
    auth.store_key("groq", "gsk_one")
    auth.store_key("gemini", "AIza_two")
    auth.delete_key("groq")
    assert auth.stored_key("gemini") == "AIza_two"


@pytest.mark.parametrize(
    "raw",
    [
        "﻿gsk_key",  # BOM — PowerShell adds one when piping
        "gsk_key\n",
        "  gsk_key  ",
        "​gsk_key",  # zero-width space, common when copying from a web page
        "\xa0gsk_key",  # non-breaking space
    ],
)
def test_invisible_characters_are_stripped_from_keys(raw):
    """str.strip() leaves BOMs and zero-width spaces in place, so the key is
    saved corrupted and every later request fails as an unexplained 401."""
    auth.store_key("groq", raw)
    assert auth.stored_key("groq") == "gsk_key"


def test_invisible_characters_are_stripped_from_env_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "﻿gsk_from_env")
    assert auth.resolve_key("groq")[0] == "gsk_from_env"


def test_empty_key_is_rejected():
    with pytest.raises(auth.AuthError):
        auth.store_key("groq", "   ")


def test_unknown_provider_is_rejected():
    with pytest.raises(auth.AuthError):
        auth.store_key("hotdog", "key")


def test_a_corrupt_credentials_file_does_not_crash(tmp_path):
    (tmp_path / "credentials.json").write_text("{not json", encoding="utf-8")
    assert auth.stored_key("groq") is None
    auth.store_key("groq", "gsk_new")  # and can be recovered from
    assert auth.stored_key("groq") == "gsk_new"


# -- resolution order -------------------------------------------------------


def test_env_var_wins_over_saved_login(monkeypatch):
    """CI and one-off overrides must not require touching a saved login."""
    auth.store_key("groq", "gsk_saved")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_from_env")
    key, source = auth.resolve_key("groq")
    assert key == "gsk_from_env"
    assert source == "$GROQ_API_KEY"


def test_saved_login_is_used_when_no_env_var():
    auth.store_key("groq", "gsk_saved")
    assert auth.resolve_key("groq") == ("gsk_saved", "saved login")


def test_missing_credentials_report_as_unset():
    assert auth.resolve_key("groq") == (None, "not set")


def test_default_provider_follows_the_saved_preference():
    auth.store_key("gemini", "AIza_x")
    auth.set_default_provider("gemini")
    assert auth.default_provider() == "gemini"


def test_default_provider_ignores_a_preference_with_no_key():
    """Preferring a provider you have since logged out of would fail every run
    with a confusing error."""
    auth.set_default_provider("gemini")
    auth.store_key("groq", "gsk_x")
    assert auth.default_provider() == "groq"


def test_default_provider_can_be_forced_by_env(monkeypatch):
    auth.store_key("groq", "gsk_x")
    auth.store_key("openai", "sk_y")
    monkeypatch.setenv("DIETCODE_PROVIDER", "openai")
    assert auth.default_provider() == "openai"


def test_mask_hides_the_middle():
    masked = auth.mask("gsk_abcdefghijklmnop")
    assert masked.startswith("gsk_")
    assert masked.endswith("mnop")
    assert "efghijkl" not in masked


def test_short_keys_are_fully_masked():
    assert set(auth.mask("abc")) == {"*"}


# -- wiring into the agent --------------------------------------------------


def test_resolve_model_config_uses_the_provider_defaults():
    auth.store_key("gemini", "AIza_x")
    auth.set_default_provider("gemini")
    args = build_parser().parse_args([])
    key, base_url, model = resolve_model_config(args)
    assert key == "AIza_x"
    assert "generativelanguage" in base_url
    assert model == auth.PROVIDERS["gemini"].default_model


def test_flags_override_the_provider_defaults():
    auth.store_key("groq", "gsk_x")
    args = build_parser().parse_args(
        ["--provider", "groq", "--model", "custom-model", "--base-url", "http://localhost:1234/v1"]
    )
    _key, base_url, model = resolve_model_config(args)
    assert base_url == "http://localhost:1234/v1"
    assert model == "custom-model"


def test_missing_credentials_explain_how_to_fix_it():
    args = build_parser().parse_args(["--provider", "groq"])
    with pytest.raises(auth.AuthError) as excinfo:
        resolve_model_config(args)
    message = str(excinfo.value)
    assert "dietcode login" in message
    assert "GROQ_API_KEY" in message


# -- commands ---------------------------------------------------------------


def test_login_saves_and_reports(console):
    assert login(console, "groq", "gsk_abcdefghijkl") == 0
    assert auth.stored_key("groq") == "gsk_abcdefghijkl"
    text = output(console)
    assert "gsk_" in text and "abcdefghijkl" not in text  # masked, not echoed


def test_login_sets_the_default_provider(console):
    login(console, "gemini", "AIza_x")
    assert auth.default_provider() == "gemini"


def test_login_warns_on_a_key_that_looks_like_another_provider(console):
    login(console, "groq", "sk-openai-style-key")
    assert "usually start with" in output(console)
    assert auth.stored_key("groq") == "sk-openai-style-key"  # saved anyway


def test_login_with_no_key_saves_nothing(console):
    assert login(console, "groq", "   ") == 1
    assert auth.stored_key("groq") is None


def test_logout_without_a_provider_clears_everything(console):
    auth.store_key("groq", "a")
    auth.store_key("gemini", "b")
    assert logout(console, None) == 0
    assert auth.stored_key("groq") is None
    assert auth.stored_key("gemini") is None


def test_logout_flags_a_key_still_live_in_the_environment(console, monkeypatch):
    """Otherwise `logout` looks like it worked and the next run still runs."""
    auth.store_key("groq", "gsk_saved")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env")
    logout(console, "groq")
    assert "still authenticated" in output(console)


def test_auth_status_lists_providers_and_fails_when_empty(console):
    assert auth_status(console) == 1
    assert "No credentials" in output(console)
    for name in auth.PROVIDERS:
        assert name in output(console)


def test_auth_status_never_prints_a_whole_key(console):
    auth.store_key("groq", "gsk_supersecretvalue")
    assert auth_status(console) == 0
    assert "supersecretvalue" not in output(console)


def test_config_file_is_valid_json(tmp_path):
    auth.set_default_provider("groq")
    assert json.loads((tmp_path / "config.json").read_text())["provider"] == "groq"


# -- the first run, with no key saved ---------------------------------------
#
# Dead-ending on "no credentials" made the user read the help, come back, and
# run a different command. The entrypoint offers to fix it in place instead.


@pytest.fixture
def first_run(monkeypatch):
    """main(), with .env ignored and everything past authentication stubbed.

    Disabling load_dotenv matters more than it looks: without it, main() reads
    the developer's own .env back into the environment that isolated_home just
    cleared, and the test makes a real API call.
    """
    import dotenv

    from agent import cli

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(
        cli,
        "make_executor",
        lambda *_a, **_k: (_ for _ in ()).throw(SystemExit(99)),
    )
    return cli


def test_a_missing_key_offers_to_set_one_up(first_run, monkeypatch):
    asked = {}

    def fake_login(_console, provider, _key):
        asked["provider"] = provider
        auth.store_key("groq", "gsk_saved_by_the_prompt")
        return 0

    monkeypatch.setattr(first_run, "confirm", lambda *_a, **_k: True)
    monkeypatch.setattr(first_run, "login", fake_login)

    with pytest.raises(SystemExit):  # got past auth, into executor setup
        first_run.main(["hello"])
    assert asked["provider"] is None, "login picks the provider when none was given"


def test_the_offer_names_the_provider_that_was_asked_for(first_run, monkeypatch):
    asked = {}

    def fake_login(_console, provider, _key):
        asked["provider"] = provider
        auth.store_key("gemini", "AIza_x")
        return 0

    monkeypatch.setattr(first_run, "confirm", lambda *_a, **_k: True)
    monkeypatch.setattr(first_run, "login", fake_login)

    with pytest.raises(SystemExit):
        first_run.main(["--provider", "gemini", "hello"])
    assert asked["provider"] == "gemini"


def test_declining_the_offer_exits_without_running(first_run, monkeypatch):
    monkeypatch.setattr(first_run, "confirm", lambda *_a, **_k: False)
    monkeypatch.setattr(
        first_run, "login", lambda *_a: pytest.fail("must not log in after a no")
    )
    assert first_run.main(["hello"]) == 2


def test_a_failed_login_does_not_start_a_run(first_run, monkeypatch):
    monkeypatch.setattr(first_run, "confirm", lambda *_a, **_k: True)
    monkeypatch.setattr(first_run, "login", lambda *_a: 1)
    assert first_run.main(["hello"]) == 2


def test_a_login_that_saves_nothing_is_not_treated_as_success(first_run, monkeypatch):
    """login exits 0 for a provider other than the one this run needs."""
    monkeypatch.setattr(first_run, "confirm", lambda *_a, **_k: True)
    monkeypatch.setattr(first_run, "login", lambda *_a: 0)
    assert first_run.main(["--provider", "groq", "hello"]) == 2


def test_a_piped_run_reports_the_error_instead_of_prompting(first_run, monkeypatch):
    """Nothing can answer a prompt in a script; hanging there would be worse."""
    monkeypatch.setattr(first_run.sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(first_run, "confirm", lambda *_a, **_k: pytest.fail("prompted"))
    assert first_run.main(["--provider", "groq", "hello"]) == 2
