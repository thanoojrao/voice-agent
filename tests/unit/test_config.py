from pathlib import Path

import pytest
from pydantic import ValidationError

from voice_agent.app.config import AgentConfig, Settings, load_agent_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_repo_config_loads():
    cfg = load_agent_config(REPO_ROOT / "agent.config.yaml")
    assert cfg.owner_name == "Thanooj"
    assert cfg.opening_line().startswith("Hi, this is Thanooj's AI.")


def base(**overrides):
    data = {"owner_name": "T", "greeting": "Hi", "name_question": "Who?"}
    return {**data, **overrides}


def test_out_of_range_rejected():
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(base(silence_prompt_s=2))


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(base(silence_promt_s=8))


def test_blocked_numbers_must_be_e164():
    AgentConfig.model_validate(base(blocked_numbers=["+15551234567"]))
    with pytest.raises(ValidationError):
        AgentConfig.model_validate(base(blocked_numbers=["555-1234"]))


def test_missing_config_file(tmp_path):
    with pytest.raises(RuntimeError):
        load_agent_config(tmp_path / "nope.yaml")


def test_settings_require_reports_missing_keys():
    s = Settings(_env_file=None, anthropic_api_key="x")  # pyright: ignore[reportCallIssue]
    s.require("anthropic_api_key")
    with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
        s.require("anthropic_api_key", "deepgram_api_key")
