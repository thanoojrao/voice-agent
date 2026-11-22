from pathlib import Path

import pytest
from pydantic import ValidationError

from voice_agent.app.config import AgentConfig, Settings, load_agent_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_repo_config_loads():
    cfg = load_agent_config(REPO_ROOT / "agent.config.yaml")
    assert cfg.owner_name == "Thanooj"
    t = cfg.greeting_templates()
    assert t.greeting.format(owner_name=t.owner_name).startswith("Hi, this is Thanooj's AI.")


def base(**overrides):
    data = {
        "owner_name": "T",
        "greeting": "Hi",
        "greeting_returning": "Hi {caller_name}",
        "name_question": "Who?",
    }
    return {**data, **overrides}


def test_out_of_range_rejected():
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(base(silence_prompt_s=2))


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(base(silence_promt_s=8))


def test_pronunciations_are_one_word_each_way():
    AgentConfig.model_validate(base(pronunciations={"Durgapur": "DOOR-gah-poor"}))
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(base(pronunciations={"Durgapur": "door gah poor"}))
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(base(pronunciations={"New York": "NY"}))


def test_blocked_numbers_must_be_e164():
    AgentConfig.model_validate(base(blocked_numbers=["+15551234567"]))
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(base(blocked_numbers=["555-1234"]))


def test_owner_numbers_from_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.setenv("OWNER_NUMBERS", " +15550100, +447700900123 ,")
    assert Settings().owner_number_set() == {"+15550100", "+447700900123"}
    monkeypatch.delenv("OWNER_NUMBERS")
    assert Settings().owner_number_set() == frozenset()


@pytest.mark.parametrize("bad", ["7770123", "+1 777 0123", "+17770123;+1777"])
def test_owner_numbers_must_be_e164_and_arent_echoed(monkeypatch, tmp_path, bad):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OWNER_NUMBERS", bad)
    with pytest.raises(ValidationError) as caught:
        Settings()
    assert "E.164" in str(caught.value)
    assert "777" not in str(caught.value)  # it's a personal number; errors get logged


def test_missing_config_file(tmp_path):
    with pytest.raises(RuntimeError):
        load_agent_config(tmp_path / "nope.yaml")


def test_settings_require_reports_missing_keys():
    s = Settings(_env_file=None, anthropic_api_key="x")  # pyright: ignore[reportCallIssue]
    s.require("anthropic_api_key")
    with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
        s.require("anthropic_api_key", "deepgram_api_key")


@pytest.mark.parametrize(
    "overrides",
    [
        {"greeting": "Hi {caller_name}"},  # plain greeting has no caller name to fill
        {"greeting": "Hi {ownr_name}"},
        {"greeting_returning": "Hi {owner_name}"},  # forgot the caller's name
        {"name_question": "Who is {"},
    ],
)
def test_bad_greeting_templates_rejected(overrides):
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(base(**overrides))
