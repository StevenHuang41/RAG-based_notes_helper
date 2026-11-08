import pytest
from pydantic import ValidationError

from rag_notes_helper.core.config import LLMSettings, Settings


def make_settings(**overrides):
    values = {
        "provider": "ollama",
        "model": "llama3.1",
        "_env_file": None,
    }
    values.update(overrides)
    return LLMSettings(**values)


def test_unset_eval_pair_inherits_generation_provider_and_model():
    settings = make_settings()

    assert settings.eval_provider is None
    assert settings.eval_model is None
    assert settings.effective_eval_provider == "ollama"
    assert settings.effective_eval_model == "llama3.1"
    assert settings.effective_eval_api_key_str is None
    assert settings.context_window_tokens == 8192


def test_context_window_must_reserve_the_output_token_budget():
    with pytest.raises(ValidationError, match="LLM_CONTEXT_WINDOW_TOKENS"):
        make_settings(context_window_tokens=1024, max_tokens=1024)


def test_legacy_output_budget_does_not_require_a_new_context_setting():
    settings = make_settings(max_tokens=8192)

    assert settings.max_tokens == 8192
    assert settings.context_window_tokens == 8192


def test_legacy_output_budget_env_loads_without_context_window_setting(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama3.1")
    monkeypatch.setenv("LLM_MAX_TOKENS", "8192")
    monkeypatch.delenv("LLM_CONTEXT_WINDOW_TOKENS", raising=False)

    settings = LLMSettings(_env_file=None)

    assert settings.max_tokens == 8192
    assert settings.context_window_tokens == 8192


@pytest.mark.parametrize("setting", ["eval_provider", "eval_model"])
def test_partial_eval_pair_is_rejected(setting):
    value = "openai" if setting == "eval_provider" else "gpt-4o-mini"

    with pytest.raises(ValidationError, match="LLM_EVAL_PROVIDER and LLM_EVAL_MODEL"):
        make_settings(**{setting: value})


def test_matching_provider_reuses_generation_key():
    settings = make_settings(
        provider="openai",
        model="gpt-4o-mini",
        api_key="generation-secret",
        eval_provider="openai",
        eval_model="gpt-4o-mini",
    )

    assert settings.effective_eval_api_key_str == "generation-secret"


def test_explicit_eval_key_overrides_generation_key():
    settings = make_settings(
        provider="openai",
        model="gpt-4o-mini",
        api_key="generation-secret",
        eval_provider="openai",
        eval_model="gpt-4o-mini",
        eval_api_key="judge-secret",
    )

    assert settings.effective_eval_api_key_str == "judge-secret"


def test_different_non_ollama_provider_requires_eval_key():
    with pytest.raises(ValidationError, match="LLM_EVAL_API_KEY is required"):
        make_settings(
            provider="ollama",
            model="llama3.1",
            eval_provider="openai",
            eval_model="gpt-4o-mini",
        )


def test_different_provider_uses_explicit_eval_key():
    settings = make_settings(
        provider="ollama",
        model="llama3.1",
        eval_provider="openrouter",
        eval_model="openai/gpt-4o-mini",
        eval_api_key="judge-secret",
    )

    assert settings.effective_eval_api_key_str == "judge-secret"


def test_ollama_judge_does_not_require_an_api_key():
    settings = make_settings(
        provider="openai",
        model="gpt-4o-mini",
        api_key="generation-secret",
        eval_provider="ollama",
        eval_model="llama3.1",
    )

    assert settings.effective_eval_api_key_str is None


def test_cross_encoder_model_loads_from_env_file(tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CROSS_ENCODER_MODEL=fixture/custom-cross-encoder\n",
        encoding="utf-8",
    )

    settings = Settings(
        _env_file=env_file,
        llm=make_settings(),
        notes_dir=notes_dir,
        storage_dir=tmp_path / "storage",
        logs_dir=tmp_path / "logs",
    )

    assert settings.cross_encoder_model == "fixture/custom-cross-encoder"


def test_agentic_attempt_budgets_load_from_env_file(tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(
        "AGENTIC_MAX_SEARCH_ATTEMPTS=2\n"
        "AGENTIC_MAX_ANSWER_ATTEMPTS=4\n",
        encoding="utf-8",
    )

    settings = Settings(
        _env_file=env_file,
        llm=make_settings(),
        notes_dir=notes_dir,
        storage_dir=tmp_path / "storage",
        logs_dir=tmp_path / "logs",
    )

    assert settings.agentic_max_search_attempts == 2
    assert settings.agentic_max_answer_attempts == 4


def test_agentic_attempt_budgets_default_to_three(tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()

    settings = Settings(
        _env_file=None,
        llm=make_settings(),
        notes_dir=notes_dir,
        storage_dir=tmp_path / "storage",
        logs_dir=tmp_path / "logs",
    )

    assert settings.agentic_max_search_attempts == 3
    assert settings.agentic_max_answer_attempts == 3


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("AGENTIC_MAX_SEARCH_ATTEMPTS", 0),
        ("AGENTIC_MAX_SEARCH_ATTEMPTS", 6),
        ("AGENTIC_MAX_ANSWER_ATTEMPTS", 0),
        ("AGENTIC_MAX_ANSWER_ATTEMPTS", 6),
    ],
)
def test_agentic_attempt_budgets_reject_values_outside_one_to_five(
    tmp_path,
    name,
    value,
):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text(f"{name}={value}\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        Settings(
            _env_file=env_file,
            llm=make_settings(),
            notes_dir=notes_dir,
            storage_dir=tmp_path / "storage",
            logs_dir=tmp_path / "logs",
        )


@pytest.mark.parametrize("provider", ["hf", "openai", "ollama", "gemini", "openrouter"])
def test_judge_accepts_all_generation_providers(provider):
    values = {
        "eval_provider": provider,
        "eval_model": "llama3.1" if provider == "ollama" else "provider/model",
    }
    if provider != "ollama":
        values["eval_api_key"] = "judge-secret"

    settings = make_settings(**values)

    assert settings.effective_eval_provider == provider
