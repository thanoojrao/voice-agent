"""The app: builds every part once, wires them together, and serves Twilio.

    uv run uvicorn voice_agent.app.main:create_app --factory --port 8000

Startup reads .env and agent.config.yaml, loads the knowledge base and the audio clips, and
checks them, so a mistake stops the server before a call can reach it rather than in the
middle of one. Shutdown ends live calls the normal way (turns and summaries saved), then
closes the shared connections: the Cartesia socket, the Anthropic client, the database.

What's shared and what's per call:
    shared     storage, the three providers, prompt blocks 1-2, clips, the registry
    per call   the Twilio adapter (made by the stream route) and its Session (made by the
               registry through `new_session`, below)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from voice_agent.adapters.twilio.audio import SAMPLE_RATE
from voice_agent.adapters.twilio.routes import twilio_router
from voice_agent.app.config import AgentConfig, Settings, load_agent_config
from voice_agent.app.constants import (
    FILLER_AFTER_MS,
    MAX_ANSWER_SENTENCES,
    PROVIDER_TIMEOUT_S,
    SPEAKING_RATE,
)
from voice_agent.core.clips import load_clips
from voice_agent.core.interfaces import (
    Caller,
    ChannelAdapter,
    MessageRow,
    SessionCallbacks,
    SessionRow,
    SessionStart,
)
from voice_agent.core.prompt import Prompt, load_instructions, load_knowledge_base
from voice_agent.core.registry import Limits, Refusal, Registry, Slot
from voice_agent.core.session import Session, SessionDeps, SessionSettings
from voice_agent.core.turn import TurnSettings
from voice_agent.providers.anthropic_llm import AnthropicLLM, new_client
from voice_agent.providers.cartesia_tts import CartesiaSettings, CartesiaTTS
from voice_agent.providers.deepgram_flux import DeepgramFlux
from voice_agent.storage.sqlite import SqliteStorage

log = logging.getLogger(__name__)

KB_DIR = Path("kb")
CLIPS_DIR = Path("assets/audio")  # apology + fillers (loaded), busy (served to Twilio)
OWNER_TZ = ZoneInfo("America/New_York")  # "today" in the prompt is the owner's date
LLM_HTTP_TIMEOUT_S = 10.0  # the turn has its own, shorter per-event timeout
REQUIRED_ENV = (
    "anthropic_api_key",
    "deepgram_api_key",
    "cartesia_api_key",
    "twilio_auth_token",
    "public_base_url",
)


def owner_clock() -> datetime:
    return datetime.now(OWNER_TZ)


def create_app() -> FastAPI:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = Settings()
    settings.require(*REQUIRED_ENV)
    config = load_agent_config(settings.config_path)
    if not config.voice_id:
        raise RuntimeError("voice_id is empty in agent.config.yaml")
    secret = _secrets(settings)

    # Everything that can be checked without a network call is loaded now.
    prompt = Prompt(
        load_instructions(_instruction_values(config)),
        load_knowledge_base(KB_DIR, config.featured_projects),
        config.status_stale_days,
    )
    clips = load_clips(CLIPS_DIR, SAMPLE_RATE)
    client = new_client(secret("anthropic_api_key"), timeout_s=LLM_HTTP_TIMEOUT_S)
    tts = CartesiaTTS(
        secret("cartesia_api_key"),
        CartesiaSettings(speed=SPEAKING_RATE, pronunciations=config.pronunciations),
    )
    session_settings = SessionSettings(
        turn=TurnSettings(
            sample_rate=SAMPLE_RATE,
            voice_id=config.voice_id,
            filler_after_ms=FILLER_AFTER_MS,
            provider_timeout_s=PROVIDER_TIMEOUT_S,
        ),
        silence_prompt_s=config.silence_prompt_s,
        silence_hangup_s=config.silence_hangup_s,
        max_session_s=config.max_session_min * 60,
        record_transcripts=config.record_transcripts,
    )
    limits = Limits(
        max_concurrent=config.max_concurrent_sessions,
        daily_cap=config.daily_session_cap,
        per_caller_daily=config.per_caller_sessions_per_day,
        blocked=frozenset(config.blocked_numbers),
    )
    handle = _RegistryHandle()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        config.db_path.parent.mkdir(parents=True, exist_ok=True)
        storage = await SqliteStorage.open(config.db_path)
        notifier = _LogNotifier()
        llm = AnthropicLLM(client, config.llm_model)
        stt = DeepgramFlux(secret("deepgram_api_key"))
        templates = config.greeting_templates()

        def new_session(start: SessionStart, adapter: ChannelAdapter) -> Session:
            deps = SessionDeps(
                storage=storage,
                notifier=notifier,
                llm=llm,
                stt=stt,
                tts=tts,
                adapter=adapter,
                clips=clips,
                prompt=prompt,
                templates=templates,
                clock=owner_clock,
            )
            turn = replace(session_settings.turn, sample_rate=start.sample_rate)
            return Session(deps, replace(session_settings, turn=turn), start)

        handle.registry = Registry(storage, limits, new_session)
        log.info("ready: %s", settings.public_base_url)
        try:
            yield
        finally:
            await handle.registry.shutdown()
            await tts.aclose()
            await client.close()
            await storage.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(
        twilio_router(handle, secret("twilio_auth_token"), _required(settings.public_base_url))
    )
    app.mount("/static/audio", StaticFiles(directory=CLIPS_DIR), name="audio")

    @app.get("/healthz", response_class=PlainTextResponse)
    async def healthz() -> str:
        return "ok"

    return app


# -- helpers ------------------------------------------------------------------------------------


def _instruction_values(config: AgentConfig) -> dict[str, str]:
    return {
        "owner_name": config.owner_name,
        "tone": config.tone,
        "do_not_discuss": ", ".join(config.do_not_discuss),
        "max_sentences": str(MAX_ANSWER_SENTENCES),
    }


def _secrets(settings: Settings) -> Callable[[str], str]:
    def secret(name: str) -> str:
        value = getattr(settings, name)
        if value is None:  # require() already checked; this keeps the type checker happy
            raise RuntimeError(f"Missing in .env: {name.upper()}")
        return value.get_secret_value()

    return secret


def _required(value: str | None) -> str:
    if not value:
        raise RuntimeError("Missing in .env: PUBLIC_BASE_URL")
    return value


@dataclass
class _RegistryHandle:
    """The routes are added when the app is built, but the registry needs the database,
    which can only be opened inside the lifespan. The routes hold this handle instead."""

    registry: Registry | None = None

    def _live(self) -> Registry:
        if self.registry is None:
            raise RuntimeError("the app hasn't finished starting")
        return self.registry

    async def admit(self, channel: str, address: str | None) -> Slot | Refusal:
        return await self._live().admit(channel, address)

    def start_session(
        self, token: str, start: SessionStart, adapter: ChannelAdapter
    ) -> SessionCallbacks:
        return self._live().start_session(token, start, adapter)


class _LogNotifier:
    """Stands in until the Gmail notifier (build step 5). Messages are still saved; raising
    here leaves them marked not-notified, which is the truth."""

    async def message_left(self, msg: MessageRow, caller: Caller, session: SessionRow) -> None:
        raise RuntimeError("email isn't set up yet (build step 5); the message is saved")

    async def alert(self, subject: str, body: str) -> None:
        log.error("alert (email not set up yet): %s", subject)
