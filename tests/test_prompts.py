"""Pickers, completion and model listing.

None of this can be driven by a real keypress in a test, so what is asserted is
the part that decides behaviour: which option the cursor starts on, what
happens with no terminal attached, and which ids ever reach the list.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from agent import prompts
from agent.completion import SlashCompleter
from agent.models import FALLBACK_MODELS, is_chat_model, list_models, rank_models
from agent.prompts import Choice, choose, confirm


@pytest.fixture
def console():
    return Console(file=io.StringIO(), width=100, force_terminal=False, no_color=True)


def output(console: Console) -> str:
    return console.file.getvalue()


# -- choose -----------------------------------------------------------------


OPTIONS = [Choice("groq", "Groq"), Choice("gemini", "Gemini"), Choice("openai", "OpenAI")]


def test_no_terminal_cancels_rather_than_guessing(console, monkeypatch):
    """Picking the first option in a piped run would silently change a setting."""
    monkeypatch.setattr(prompts, "interactive", lambda: False)
    assert choose(console, "Which?", OPTIONS) is None


def test_empty_list_is_not_a_prompt(console):
    assert choose(console, "Which?", []) is None


def test_cursor_starts_on_what_is_already_selected():
    assert prompts._start_index(OPTIONS, 0, "openai") == 2


def test_cursor_falls_back_to_the_default_when_the_selection_is_unknown():
    assert prompts._start_index(OPTIONS, 1, "anthropic") == 1


def test_default_past_the_end_is_clamped():
    assert prompts._start_index(OPTIONS, 99, None) == 2


def test_typed_fallback_accepts_a_number(console, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt="": "2")
    assert prompts._typed_fallback(console, "Which?", OPTIONS, 0) == "gemini"


def test_typed_fallback_accepts_a_name(console, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt="": "OpenAI")
    assert prompts._typed_fallback(console, "Which?", OPTIONS, 0) == "openai"


def test_typed_fallback_empty_takes_the_default(console, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")
    assert prompts._typed_fallback(console, "Which?", OPTIONS, 1) == "gemini"


def test_typed_fallback_rejects_nonsense(console, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt="": "mistral")
    assert prompts._typed_fallback(console, "Which?", OPTIONS, 0) is None
    assert "not one of the options" in output(console)


def test_confirm_without_a_terminal_takes_the_default(console, monkeypatch):
    monkeypatch.setattr(prompts, "interactive", lambda: False)
    assert confirm(console, "Set one up now?", default=True) is True
    assert confirm(console, "Set one up now?", default=False) is False


# -- slash completion -------------------------------------------------------


class _Document:
    """Just enough of prompt_toolkit's Document for the completer."""

    def __init__(self, text: str):
        self.text_before_cursor = text


def completions(text: str) -> list[str]:
    completer = SlashCompleter({"/help": "show help", "/model": "switch model"})
    return [c.text for c in completer.get_completions(_Document(text), None)]


def test_slash_offers_every_command():
    assert completions("/") == ["/help", "/model"]


def test_partial_command_narrows():
    assert completions("/mo") == ["/model"]


def test_ordinary_prose_never_completes():
    """The bug: typing a task and hitting space popped the command list up."""
    assert completions("fix the ") == []
    assert completions("write a parser") == []


def test_a_path_in_the_middle_is_not_a_command():
    assert completions("read /tmp/x") == []


def test_nothing_completes_once_the_command_has_an_argument():
    assert completions("/model gemini") == []


# -- model listing ----------------------------------------------------------


class _Model:
    def __init__(self, ident: str):
        self.id = ident


class _Models:
    def __init__(self, ids, error=None):
        self._ids = ids
        self._error = error

    def list(self):
        if self._error:
            raise self._error
        return type("Response", (), {"data": [_Model(i) for i in self._ids]})()


class _Client:
    def __init__(self, ids=(), error=None):
        self.models = _Models(list(ids), error)


def test_chat_models_pass_and_the_rest_do_not():
    assert is_chat_model("llama-3.3-70b-versatile")
    assert is_chat_model("gpt-4.1-mini")
    assert not is_chat_model("text-embedding-3-small")
    assert not is_chat_model("whisper-large-v3")
    assert not is_chat_model("meta-llama/llama-guard-4-12b")


def test_listing_drops_what_cannot_run_the_loop():
    """An embedding id in the picker is a 400 waiting to happen."""
    models, error = list_models(
        _Client(["gpt-4.1-mini", "text-embedding-3-large", "tts-1"]), "openai"
    )
    assert models == ["gpt-4.1-mini"]
    assert error is None


def test_gemini_model_prefix_is_stripped():
    models, _ = list_models(_Client(["models/gemini-flash-latest"]), "gemini")
    assert models == ["gemini-flash-latest"]


def test_a_failed_call_falls_back_instead_of_raising():
    models, error = list_models(_Client(error=RuntimeError("offline")), "groq")
    assert models == list(FALLBACK_MODELS["groq"])
    assert "offline" in error


def test_a_provider_with_no_chat_models_falls_back():
    models, error = list_models(_Client(["text-embedding-3-small"]), "openai")
    assert models == list(FALLBACK_MODELS["openai"])
    assert error


def test_every_fallback_id_survives_its_own_filter():
    """A fallback the filter rejects would leave the picker empty."""
    for provider, ids in FALLBACK_MODELS.items():
        assert ids, provider
        for model_id in ids:
            assert is_chat_model(model_id), f"{provider}: {model_id}"


def test_every_provider_default_is_a_model_the_picker_would_show():
    """Otherwise switching provider lands on an id the filter just removed."""
    from agent.auth import PROVIDERS

    for spec in PROVIDERS.values():
        assert is_chat_model(spec.default_model), spec.name
        assert spec.default_model in FALLBACK_MODELS[spec.name], spec.name


def test_the_recommended_model_sorts_first():
    ranked = rank_models(["b-model", "a-model", "default"], "default")
    assert ranked[0] == "default"


def test_ranking_a_model_the_provider_did_not_list_does_not_invent_it():
    assert rank_models(["a", "b"], "missing") == ["a", "b"]


# -- tool schemas, per provider ---------------------------------------------
#
# The union types in TOOLS exist because Groq validates the model's arguments
# against them server-side. Gemini's schema layer cannot express a union at
# all, so the canonical schema is narrowed on the way out rather than weakened
# for everyone.


def union_typed(tools):
    return {
        name: schema["type"]
        for tool in tools
        for name, schema in tool["function"]["parameters"]["properties"].items()
        if isinstance(schema["type"], list)
    }


def test_groq_keeps_the_unions_it_needs():
    from agent.tools import tools_for

    assert union_typed(tools_for("groq")), "narrowing these breaks Groq runs"


def test_gemini_gets_a_single_type_per_argument():
    from agent.tools import tools_for

    assert union_typed(tools_for("gemini")) == {}


def test_narrowing_keeps_the_real_type_not_the_fallback():
    from agent.tools import tools_for

    schemas = {
        name: schema
        for tool in tools_for("gemini")
        for name, schema in tool["function"]["parameters"]["properties"].items()
    }
    assert schemas["timeout"]["type"] == "integer"


def test_narrowing_does_not_mutate_the_canonical_schema():
    """tools_for is called per run; a shared dict would narrow Groq too."""
    from agent.tools import TOOLS, tools_for

    tools_for("gemini")
    assert union_typed(TOOLS), "the module-level schema was edited in place"


def test_every_provider_gets_the_same_tools():
    from agent.auth import PROVIDERS
    from agent.tools import TOOL_NAMES, tools_for

    for name in PROVIDERS:
        assert [t["function"]["name"] for t in tools_for(name)] == TOOL_NAMES
