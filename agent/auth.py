"""Providers and credential storage.

Keys live in the OS keychain when one is available, and in a 0600 file under
~/.dietcode otherwise. Not in the project directory: a key in a repo is a key
that gets committed, which nearly happened here already.

Resolution order is env var, then stored credential. The env var wins so CI and
one-off overrides work without touching a user's saved login.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("DIETCODE_HOME", Path.home() / ".dietcode"))
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"
CONFIG_FILE = CONFIG_DIR / "config.json"
KEYRING_SERVICE = "dietcode"


class AuthError(RuntimeError):
    """Something wrong with credentials that the user has to fix."""


@dataclass(frozen=True)
class Provider:
    name: str
    label: str
    base_url: str
    default_model: str
    env_var: str
    signup_url: str
    key_hint: str = ""


PROVIDERS: dict[str, Provider] = {
    "groq": Provider(
        name="groq",
        label="Groq",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        env_var="GROQ_API_KEY",
        signup_url="https://console.groq.com/keys",
        key_hint="gsk_",
    ),
    "gemini": Provider(
        name="gemini",
        label="Google Gemini",
        # Gemini speaks the OpenAI protocol at this path, so the same client works.
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        # An alias, not a pinned version. Google retires model ids and gates
        # older ones to existing accounts, so a hardcoded `gemini-2.5-flash`
        # started returning 404 "no longer available to new users" for anyone
        # signing up after it shipped. `-latest` follows the current model.
        #
        # Lite rather than plain flash, which is the better model: the free
        # tier allows 20 requests per day per model, and one loop step is one
        # request, so flash is spent after three or four tasks. A default that
        # 429s on someone's first session is worse than a smaller model.
        # `/model gemini-flash-latest` is one command away.
        default_model="gemini-flash-lite-latest",
        env_var="GEMINI_API_KEY",
        signup_url="https://aistudio.google.com/apikey",
    ),
    "openai": Provider(
        name="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4.1-mini",
        env_var="OPENAI_API_KEY",
        signup_url="https://platform.openai.com/api-keys",
        key_hint="sk-",
    ),
}

DEFAULT_PROVIDER = "groq"


def get_provider(name: str) -> Provider:
    try:
        return PROVIDERS[name]
    except KeyError:
        known = ", ".join(PROVIDERS)
        raise AuthError(f"unknown provider {name!r}. Known providers: {known}") from None


# -- storage ----------------------------------------------------------------


def _keyring():
    """The OS keychain, if this machine has a usable one."""
    try:
        import keyring
        from keyring.backends.fail import Keyring as FailKeyring

        # A headless box often resolves to the "fail" backend, which raises on
        # every call. Treat that as no keyring rather than discovering it later.
        if isinstance(keyring.get_keyring(), FailKeyring):
            return None
        return keyring
    except Exception:  # noqa: BLE001 - keyring raises broadly on odd platforms
        return None


def _read_file_store() -> dict[str, str]:
    try:
        data = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, str)}


def _write_file_store(store: dict[str, str]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CREDENTIALS_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")
    try:
        # Owner read/write only. No-op on Windows, which uses ACLs, but the
        # file is inside the user profile there anyway.
        CREDENTIALS_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


# Characters that ride along invisibly when a key is pasted or piped: a BOM
# (PowerShell adds one when piping), zero-width spaces from web pages, and a
# non-breaking space. str.strip() does not remove any of them, so the key is
# stored corrupted and every later request fails as an unexplained 401.
_INVISIBLE = "﻿​‌‍ "


def clean_key(api_key: str) -> str:
    return api_key.strip().strip(_INVISIBLE).strip()


def store_key(provider: str, api_key: str) -> str:
    """Save a key. Returns where it went, for the user to see."""
    get_provider(provider)
    api_key = clean_key(api_key)
    if not api_key:
        raise AuthError("no API key given")

    ring = _keyring()
    if ring is not None:
        try:
            ring.set_password(KEYRING_SERVICE, provider, api_key)
            return "system keychain"
        except Exception:  # noqa: BLE001 - fall back rather than lose the key
            pass

    store = _read_file_store()
    store[provider] = api_key
    _write_file_store(store)
    return str(CREDENTIALS_FILE)


def delete_key(provider: str) -> bool:
    """Forget a stored key. True if anything was removed."""
    removed = False
    ring = _keyring()
    if ring is not None:
        try:
            if ring.get_password(KEYRING_SERVICE, provider):
                ring.delete_password(KEYRING_SERVICE, provider)
                removed = True
        except Exception:  # noqa: BLE001
            pass

    store = _read_file_store()
    if store.pop(provider, None) is not None:
        _write_file_store(store)
        removed = True
    return removed


def stored_key(provider: str) -> str | None:
    ring = _keyring()
    if ring is not None:
        try:
            key = ring.get_password(KEYRING_SERVICE, provider)
            if key:
                return key
        except Exception:  # noqa: BLE001
            pass
    return _read_file_store().get(provider)


def resolve_key(provider: str) -> tuple[str | None, str]:
    """The key to use, and where it came from."""
    spec = get_provider(provider)
    from_env = os.environ.get(spec.env_var)
    if from_env and clean_key(from_env):
        # Cleaned here too: an exported key can carry the same invisible junk.
        return clean_key(from_env), f"${spec.env_var}"
    key = stored_key(provider)
    if key:
        return key, "saved login"
    return None, "not set"


def mask(api_key: str) -> str:
    """Enough to recognise a key, not enough to use it."""
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:4]}...{api_key[-4:]}"


# -- preferences ------------------------------------------------------------


def _read_config() -> dict[str, str]:
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, str)}


def set_default_provider(provider: str) -> None:
    get_provider(provider)
    config = _read_config()
    config["provider"] = provider
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")


def default_provider() -> str:
    """Which provider to use when the command line does not say.

    A provider is only default-worthy if it can actually authenticate, so a
    saved preference whose key has since been removed is ignored.
    """
    chosen = os.environ.get("DIETCODE_PROVIDER") or _read_config().get("provider")
    if chosen in PROVIDERS and resolve_key(chosen)[0]:
        return chosen
    for name in PROVIDERS:
        if resolve_key(name)[0]:
            return name
    return chosen if chosen in PROVIDERS else DEFAULT_PROVIDER


def credential_status() -> list[tuple[Provider, str | None, str]]:
    """(provider, masked key or None, source) for every known provider."""
    rows = []
    for spec in PROVIDERS.values():
        key, source = resolve_key(spec.name)
        rows.append((spec, mask(key) if key else None, source))
    return rows
