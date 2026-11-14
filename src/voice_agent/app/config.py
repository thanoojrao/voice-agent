"""Configuration loading.

Two sources:
- Settings: secrets and deploy values from .env / environment variables.
- AgentConfig: the owner-editable product settings from agent.config.yaml.

Both are validated at startup; a bad config refuses to start (no hot reload in v0).
"""

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, StringConstraints, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from voice_agent.core.greeting import GreetingTemplates

E164 = Annotated[str, StringConstraints(pattern=r"^\+[1-9]\d{6,14}$")]


class Settings(BaseSettings):
    """Secrets and deploy values. Every field may be blank until the piece that uses it is
    wired up; call `require()` from that piece so a missing key fails loudly at startup."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: SecretStr | None = None
    deepgram_api_key: SecretStr | None = None
    cartesia_api_key: SecretStr | None = None
    twilio_auth_token: SecretStr | None = None
    owner_email: str | None = None
    gmail_address: str | None = None
    gmail_app_password: SecretStr | None = None
    admin_password: SecretStr | None = None
    public_base_url: str | None = None
    config_path: Path = Path("agent.config.yaml")

    def require(self, *names: str) -> None:
        missing = [n.upper() for n in names if not getattr(self, n)]
        if missing:
            raise RuntimeError(f"Missing in .env: {', '.join(missing)}")


class AgentConfig(BaseModel):
    """The spec's v0 Product variables. Unknown keys are rejected so typos don't pass silently."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    owner_name: str = Field(min_length=1)
    greeting: str = Field(min_length=1)
    greeting_returning: str = Field(min_length=1)
    name_question: str = Field(min_length=1)
    tone: str = "friendly, concise"
    voice_id: str = ""
    llm_model: str = "claude-haiku-5-5"

    silence_prompt_s: int = Field(8, ge=5, le=15)
    silence_hangup_s: int = Field(20, ge=10, le=60)
    max_session_min: int = Field(10, ge=3, le=30)
    daily_session_cap: int = Field(50, ge=1)
    per_caller_sessions_per_day: int = Field(3, ge=1, le=10)
    max_concurrent_sessions: int = Field(2, ge=1, le=10)

    status_stale_days: int = Field(14, ge=7, le=60)
    featured_projects: int = Field(3, ge=1, le=5)
    do_not_discuss: list[str] = ["salary", "address", "family", "health", "politics"]

    warm_intro: bool = True
    record_transcripts: bool = True
    blocked_numbers: list[E164] = []
    db_path: Path = Path("data/agent.db")

    @model_validator(mode="after")
    def _check_templates(self) -> "AgentConfig":
        # Fill every template now so a typo'd placeholder fails at startup, not mid-call.
        allowed = {
            "greeting": {"owner_name"},
            "greeting_returning": {"owner_name", "caller_name"},
            "name_question": {"owner_name"},
        }
        for field, names in allowed.items():
            try:
                getattr(self, field).format(**dict.fromkeys(names, "x"))
            except (KeyError, IndexError, ValueError) as e:
                raise ValueError(f"{field}: bad placeholder {e} (allowed: {sorted(names)})") from e
        if "{caller_name}" not in self.greeting_returning:
            raise ValueError("greeting_returning must include {caller_name}")
        return self

    def greeting_templates(self) -> GreetingTemplates:
        return GreetingTemplates(
            owner_name=self.owner_name,
            greeting=self.greeting,
            greeting_returning=self.greeting_returning,
            name_question=self.name_question,
        )


def load_agent_config(path: Path) -> AgentConfig:
    if not path.exists():
        raise RuntimeError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AgentConfig.model_validate(data)
