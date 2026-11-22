"""The contracts between the core and everything outside it.

The core imports only this module. Adapters (Twilio), providers (Deepgram, Claude, Cartesia),
storage (SQLite) and the notifier (Gmail) implement these Protocols, are created in app/main.py
and are handed to the core. Nothing here names a vendor.

Audio everywhere in the core is PCM16 little-endian mono at the rate the adapter declares.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Protocol

# ---------------------------------------------------------------------------
# Channel adapters (spec: Adapter interface)
# ---------------------------------------------------------------------------


class Capability(StrEnum):
    """Optional extras an adapter may support. Every adapter does audio in and out;
    the core must work when any of these is missing (spec D31)."""

    CALLER_ADDRESS = "caller_address"  # a stable caller ID (phone number) for warm intros
    FOLLOWUPS = "followups"  # can send a follow-up message (v1: SMS)
    END_SESSION = "end_session"  # the agent can end the session itself


@dataclass(frozen=True, slots=True)
class SessionStart:
    channel: str  # "twilio"
    channel_ref: str  # the channel's own ID for this session (Twilio CallSid)
    caller_address: str | None  # E.164 number, or None when withheld / not available
    sample_rate: int  # 8000 for Twilio


class ChannelAdapter(Protocol):
    """Implemented by adapters; called by the core."""

    capabilities: frozenset[Capability]

    async def send_audio(self, pcm: bytes) -> None: ...
    async def clear_audio(self) -> None: ...  # barge-in: drop audio not yet played
    async def mark(self, name: str) -> None: ...  # playback checkpoint, echoed when reached
    async def end_session(self) -> None: ...  # only if END_SESSION is supported
    async def send_followup(self, body: str, link: str | None) -> None: ...  # v1, FOLLOWUPS


class SessionCallbacks(Protocol):
    """Implemented by the core's Session; called by adapters."""

    async def on_audio(self, pcm: bytes) -> None: ...
    async def on_mark_played(self, name: str) -> None: ...
    async def on_session_end(self, reason: str) -> None: ...


# ---------------------------------------------------------------------------
# Speech to text
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TurnStarted:
    """The caller started speaking."""


@dataclass(frozen=True, slots=True)
class Words:
    """Words heard so far in the current caller turn (interim)."""

    text: str


@dataclass(frozen=True, slots=True)
class TurnEnded:
    """The caller finished their turn."""

    transcript: str


SttEvent = TurnStarted | Words | TurnEnded


class SttStream(Protocol):
    async def send(self, pcm: bytes) -> None: ...
    def events(self) -> AsyncIterator[SttEvent]: ...
    async def close(self) -> None: ...


class SpeechToText(Protocol):
    async def open(self, sample_rate: int) -> SttStream: ...


# ---------------------------------------------------------------------------
# Language model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SystemBlock:
    text: str
    cache: bool = False  # cache breakpoint after this block (knowledge base)


@dataclass(frozen=True, slots=True)
class TextPart:
    text: str


@dataclass(frozen=True, slots=True)
class ToolUsePart:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResultPart:
    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant"]
    parts: tuple[TextPart | ToolUsePart | ToolResultPart, ...]


@dataclass(frozen=True, slots=True)
class ToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Done:
    stop_reason: str  # "end_turn", "tool_use", "max_tokens", ...
    input_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class Refused:
    """The model declined to answer; the core speaks a fixed fallback line."""


LlmEvent = TextDelta | ToolCall | Done | Refused


class ProviderError(Exception):
    """A provider call failed. Providers translate their vendor's errors into this, so the
    core decides on retries without knowing any vendor (design §8.5).

    `retryable`: connection errors, rate limits and 5xx are; bad requests and auth are not.
    """

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class CallNotes:
    """What the end-of-call summary call reads from the transcript. The caller's details
    are None when they didn't say; the session only uses them to fill in what wasn't saved
    during the call, so the model forgetting save_caller_details doesn't lose them."""

    summary: str
    name: str | None = None
    company: str | None = None
    reason: str | None = None


class LanguageModel(Protocol):
    def stream_reply(
        self, system: list[SystemBlock], history: list[Message], tools: list[ToolDef]
    ) -> AsyncIterator[LlmEvent]: ...

    async def summarize(self, transcript: str) -> CallNotes: ...

    async def prewarm(self, system: list[SystemBlock], tools: list[ToolDef]) -> None:
        """Get the shared part of the prompt ready (e.g. cached) before the first reply,
        without generating anything. Called at the start of a session."""
        ...


# ---------------------------------------------------------------------------
# Text to speech
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WordTiming:
    word: str
    start_ms: int  # offset from the start of this utterance
    end_ms: int


@dataclass(frozen=True, slots=True)
class TtsChunk:
    pcm: bytes
    words: tuple[WordTiming, ...] = ()


class TtsStream(Protocol):
    """One per agent turn: sentences are appended to the same utterance."""

    async def say(self, sentence: str) -> None: ...
    async def finish(self) -> None: ...  # no more sentences for this turn
    def audio(self) -> AsyncIterator[TtsChunk]: ...
    async def cancel(self) -> None: ...


class TextToSpeech(Protocol):
    async def open(self, sample_rate: int, voice_id: str) -> TtsStream: ...


# ---------------------------------------------------------------------------
# Storage rows and Storage
# ---------------------------------------------------------------------------

Speaker = Literal["caller", "agent", "tool"]


@dataclass(frozen=True, slots=True)
class Caller:
    id: int
    channel: str
    address: str | None  # None = anonymous
    name: str | None
    company: str | None
    reason: str | None
    first_seen: datetime
    last_seen: datetime
    session_count: int  # sessions created so far (0 = first-time caller)


@dataclass(frozen=True, slots=True)
class SessionRow:
    id: int
    caller_id: int
    channel: str
    channel_ref: str | None
    started_at: datetime
    ended_at: datetime | None
    end_reason: str | None
    summary: str | None


@dataclass(frozen=True, slots=True)
class TurnRow:
    seq: int
    speaker: Speaker
    text: str
    started_at: datetime
    ended_at: datetime | None = None
    interrupted: bool = False
    latency_ms: int | None = None


@dataclass(frozen=True, slots=True)
class MessageRow:
    body: str
    created_at: datetime
    name: str | None = None
    contact: str | None = None
    id: int | None = None  # set once saved
    notified_at: datetime | None = field(default=None)


class Storage(Protocol):
    async def get_or_create_caller(self, channel: str, address: str | None) -> Caller: ...
    async def set_caller_details(
        self, caller_id: int, name: str | None, company: str | None, reason: str | None
    ) -> None: ...
    async def last_session(self, caller_id: int) -> SessionRow | None: ...
    async def create_session(self, caller_id: int, start: SessionStart) -> int: ...
    async def end_session(self, session_id: int, reason: str, summary: str | None) -> None: ...
    async def add_turn(self, session_id: int, turn: TurnRow) -> None: ...
    async def save_message(self, session_id: int, msg: MessageRow) -> int: ...  # or replace
    async def session_message(self, session_id: int) -> MessageRow | None: ...
    async def mark_message_notified(self, message_id: int) -> None: ...
    async def unsent_messages(self) -> list[tuple[MessageRow, Caller, SessionRow]]: ...
    async def count_sessions_since(self, since: datetime, caller_id: int | None = None) -> int: ...


# ---------------------------------------------------------------------------
# Notifier
# ---------------------------------------------------------------------------


class Notifier(Protocol):
    async def message_left(self, msg: MessageRow, caller: Caller, session: SessionRow) -> None: ...
    async def alert(self, subject: str, body: str) -> None: ...
