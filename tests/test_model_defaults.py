"""Tests for provider-aware LLM default resolution.

These lock in the behavior that replaced the old hardcoded ``gpt-5.5`` / ``OpenAI``
fallback: with no provider configured the app fails with a clear error, and with a
single provider's key set it defaults to that provider's model (not OpenAI's).
"""

import pytest

from src.llm.models import PROVIDER_ENV_KEYS, get_default_model
from src.utils.llm import get_agent_model_config

# Every env var that can make a provider "configured" (see is_provider_configured).
_ALL_KEY_ENV_VARS = [name for names in PROVIDER_ENV_KEYS.values() for name in names] + [
    "LLM_API_KEY",
    "LLM_PROVIDER",
    "GIGACHAT_USER",
    "GIGACHAT_PASSWORD",
]


@pytest.fixture
def clean_env(monkeypatch):
    """Remove every provider key so no cloud model is configured by default."""
    for name in _ALL_KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_no_provider_configured_raises_clear_error(clean_env):
    assert get_default_model() is None
    with pytest.raises(ValueError, match="No LLM provider configured"):
        get_agent_model_config({"metadata": {}}, "some_agent")


def test_single_provider_key_defaults_to_that_provider(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    default = get_default_model()
    assert default is not None
    assert default.provider.value == "Anthropic"

    # get_agent_model_config resolves to the same configured-provider default, not OpenAI.
    assert get_agent_model_config({"metadata": {}}, "some_agent") == (
        default.model_name,
        "Anthropic",
    )


def test_explicit_metadata_model_is_used(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    state = {"metadata": {"model_name": "claude-fable-5", "model_provider": "Anthropic"}}
    assert get_agent_model_config(state, "some_agent") == ("claude-fable-5", "Anthropic")
