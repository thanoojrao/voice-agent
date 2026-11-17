"""Scriptable stand-ins for everything outside the core, so the core can be tested with no
keys, no network and no phone. Each fake implements a Protocol from core.interfaces and
records what the core did to it, so tests can assert on it.

Storage has no fake: tests use the real SqliteStorage on a temp file (it's fast and catches
real SQL bugs).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from voice_agent.core.interfaces import (
    Caller,
    Capability,
    LlmEvent,
    Message,
    MessageRow,
    SessionCallbacks,
    SessionRow,
    SttEvent,
    SystemBlock,
    ToolDef,
    TtsChunk,
    WordTiming,
)

SAMPLE_RATE = 8000
BYTES_PER_MS = SAMPLE_RATE * 2 // 1000  # PCM16 mono


def silence(ms: int) -> bytes:
    return b"\x00" * (BYTES_PER_MS * ms)


# ---------------------------------------------------------------------------
# Channel adapter
# ---------------------------------------------------------------------------


class FakeVoiceAdapter:
    """Records what the core plays and lets the test act as the caller's phone.

    `capabilities` defaults to none at all, which is the strictest case for the core.
    With `auto_play`, every mark is echoed back as soon as it's sent, as if the audio
    played instantly; turn it off to hold the agent "mid-sentence".
    """

    def __init__(self, capabilities: frozenset[Capability] = frozenset(), auto_play: bool = False):
        self.capabilities = capabilities
        self.auto_play = auto_play
        self.callbacks: SessionCallbacks | None = None
        self.audio: list[bytes] = []
        self.clears = 0
        self.marks: list[str] = []  # in the order the core sent them
        self.pending_marks: list[str] = []  # sent but not yet "played"
        self.ended = False
        self.followups: list[tuple[str, str | None]] = []

    def attach(self, callbacks: SessionCallbacks) -> None:
        self.callbacks = callbacks

    # ChannelAdapter
    async def send_audio(self, pcm: bytes) -> None:
        self.audio.append(pcm)

    async def clear_audio(self) -> None:
        self.clears += 1
        self.pending_marks.clear()  # Twilio drops unplayed marks on clear, so do we

    async def mark(self, name: str) -> None:
        self.marks.append(name)
        self.pending_marks.append(name)
        if self.auto_play and self.callbacks is not None:
            await self.play_marks()

    async def end_session(self) -> None:
        if Capability.END_SESSION not in self.capabilities:
            raise AssertionError("core called end_session on an adapter without END_SESSION")
        self.ended = True

    async def send_followup(self, body: str, link: str | None) -> None:
        if Capability.FOLLOWUPS not in self.capabilities:
            raise AssertionError("core called send_followup on an adapter without FOLLOWUPS")
        self.followups.append((body, link))

    # Test controls: act as the phone
    def _cb(self) -> SessionCallbacks:
        if self.callbacks is None:
            raise AssertionError("adapter not attached to a session")
        return self.callbacks

    async def caller_audio(self, pcm: bytes) -> None:
        await self._cb().on_audio(pcm)

    async def play_marks(self, count: int | None = None) -> None:
        """Report the oldest `count` pending marks (all by default) as played."""
        n = len(self.pending_marks) if count is None else count
        for name in self.pending_marks[:n]:
            await self._cb().on_mark_played(name)
        del self.pending_marks[:n]

    async def hang_up(self, reason: str = "caller_hangup") -> None:
        await self._cb().on_session_end(reason)

    @property
    def audio_bytes(self) -> int:
        return sum(len(a) for a in self.audio)


# ---------------------------------------------------------------------------
# Speech to text
# ---------------------------------------------------------------------------


class FakeSttStream:
    def __init__(self) -> None:
        self._events: asyncio.Queue[SttEvent | None] = asyncio.Queue()
        self.received_bytes = 0
        self.closed = False

    async def send(self, pcm: bytes) -> None:
        self.received_bytes += len(pcm)

    async def events(self) -> AsyncIterator[SttEvent]:
        while (event := await self._events.get()) is not None:
            yield event

    async def close(self) -> None:
        self.closed = True
        self._events.put_nowait(None)

    # Test control: "the caller said..."
    def emit(self, *events: SttEvent) -> None:
        for e in events:
            self._events.put_nowait(e)


class FakeSTT:
    def __init__(self) -> None:
        self.streams: list[FakeSttStream] = []

    async def open(self, sample_rate: int) -> FakeSttStream:
        stream = FakeSttStream()
        self.streams.append(stream)
        return stream

    @property
    def stream(self) -> FakeSttStream:
        """The most recently opened stream."""
        return self.streams[-1]


# ---------------------------------------------------------------------------
# Language model
# ---------------------------------------------------------------------------


@dataclass
class Reply:
    """One scripted model reply: the events to stream, with an optional pause before each
    event (to test fillers and barge-in), then an optional exception. An exception with no
    events fails before anything streams; after events, it fails mid-reply."""

    events: Sequence[LlmEvent] = ()
    delay_s: float = 0.0
    raises: Exception | None = None


@dataclass
class LlmCall:
    system: list[SystemBlock]
    history: list[Message]
    tools: list[ToolDef]


@dataclass
class FakeLLM:
    replies: list[Reply] = field(default_factory=list)
    summary: str = "Fake summary."
    calls: list[LlmCall] = field(default_factory=list)
    summarized: list[str] = field(default_factory=list)
    prewarmed: list[LlmCall] = field(default_factory=list)
    prewarm_raises: Exception | None = None

    async def stream_reply(
        self, system: list[SystemBlock], history: list[Message], tools: list[ToolDef]
    ) -> AsyncIterator[LlmEvent]:
        self.calls.append(LlmCall(list(system), list(history), list(tools)))
        if not self.replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        reply = self.replies.pop(0)
        for event in reply.events:
            if reply.delay_s:
                await asyncio.sleep(reply.delay_s)
            yield event
        if reply.raises:
            raise reply.raises

    async def summarize(self, transcript: str) -> str:
        self.summarized.append(transcript)
        return self.summary

    async def prewarm(self, system: list[SystemBlock], tools: list[ToolDef]) -> None:
        self.prewarmed.append(LlmCall(list(system), [], list(tools)))
        if self.prewarm_raises:
            raise self.prewarm_raises


# ---------------------------------------------------------------------------
# Text to speech
# ---------------------------------------------------------------------------

MS_PER_WORD = 300


class FakeTtsStream:
    """Each sentence becomes one chunk: MS_PER_WORD of silence per word, with word timings
    that continue across the whole utterance like a real TTS context."""

    def __init__(self, voice_id: str) -> None:
        self.voice_id = voice_id
        self.sentences: list[str] = []
        self.finished = False
        self.cancelled = False
        self._chunks: asyncio.Queue[TtsChunk | None] = asyncio.Queue()
        self._offset_ms = 0

    async def say(self, sentence: str) -> None:
        if self.finished or self.cancelled:
            raise AssertionError("say() after finish() or cancel()")
        self.sentences.append(sentence)
        words = []
        for w in sentence.split():
            words.append(WordTiming(w, self._offset_ms, self._offset_ms + MS_PER_WORD))
            self._offset_ms += MS_PER_WORD
        pcm = silence(MS_PER_WORD * len(words))
        self._chunks.put_nowait(TtsChunk(pcm, tuple(words)))

    async def finish(self) -> None:
        self.finished = True
        self._chunks.put_nowait(None)

    async def audio(self) -> AsyncIterator[TtsChunk]:
        while (chunk := await self._chunks.get()) is not None:
            yield chunk

    async def cancel(self) -> None:
        self.cancelled = True
        self._chunks.put_nowait(None)


class FakeTTS:
    def __init__(self) -> None:
        self.streams: list[FakeTtsStream] = []

    async def open(self, sample_rate: int, voice_id: str) -> FakeTtsStream:
        stream = FakeTtsStream(voice_id)
        self.streams.append(stream)
        return stream

    @property
    def spoken(self) -> list[str]:
        """Every sentence sent to TTS, across all turns."""
        return [s for stream in self.streams for s in stream.sentences]


# ---------------------------------------------------------------------------
# Notifier
# ---------------------------------------------------------------------------


@dataclass
class FakeNotifier:
    fail: bool = False  # simulate Gmail being down
    messages: list[tuple[MessageRow, Caller, SessionRow]] = field(default_factory=list)
    alerts: list[tuple[str, str]] = field(default_factory=list)

    async def message_left(self, msg: MessageRow, caller: Caller, session: SessionRow) -> None:
        if self.fail:
            raise ConnectionError("fake email failure")
        self.messages.append((msg, caller, session))

    async def alert(self, subject: str, body: str) -> None:
        if self.fail:
            raise ConnectionError("fake email failure")
        self.alerts.append((subject, body))
