"""Cartesia as the text-to-speech, over a raw WebSocket to /tts/websocket.

One WebSocket is shared by every turn. Each turn's sentences go into one Cartesia "context",
so they're spoken as one utterance with natural flow between sentences, and each turn's
audio is told apart from the others by its context id:

  say(sentence)  -> {"context_id": c, "transcript": "sentence ", "continue": true}
  finish()       -> {"context_id": c, "transcript": "", "continue": false}
  cancel()       -> {"context_id": c, "cancel": true}; audio() ends at once
  chunk          -> TtsChunk with the audio (raw PCM16 at the call's sample rate)
  timestamps     -> WordTiming, attached to the chunk where the word's audio has all arrived
  done           -> audio() ends
  error          -> ProviderError

Choices, and why:
- Raw WebSocket, not the SDK: the protocol is a few JSON messages, and barge-in needs the
  routing and timings the SDK hides.
- PCM16, not mu-law: the core works in PCM16; converting for the phone is the adapter's job.
- max_buffer_delay_ms = 0: the turn already sends whole sentences, so Cartesia shouldn't
  wait for more text before speaking.
- Word timings count from the start of the context, in seconds. They're held until the
  audio received covers the word, so a word's mark never goes out before its audio, whatever
  order Cartesia sends the two messages in.
- A context expires 1 s after its last audio. If a turn pauses that long between sentences
  (a tool running), the next sentence starts a new context, and its timings are shifted by
  the audio already received so the turn keeps one timeline.
- Cancel only stops text Cartesia hasn't started on; audio already being made still comes.
  So cancel() ends audio() itself, and anything that arrives later for the turn is dropped.
- Pronunciations (from the config) are swapped in just before a sentence is sent, so the
  model, the history and the transcripts keep the real spelling. Each word is replaced by
  one word, so Cartesia's word timings still count the same words as the history.
- A connect that fails on the network is tried once more over IPv4. Measured from home:
  about half of the IPv6 handshakes to Cartesia were reset, none over IPv4. A refusal (an
  HTTP status such as 401) isn't retried; asking again wouldn't change the answer.
"""

import asyncio
import base64
import json
import logging
import re
import socket
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import InvalidStatus, WebSocketException

from voice_agent.core.interfaces import ProviderError, TtsChunk, WordTiming

log = logging.getLogger(__name__)

URL = "wss://api.cartesia.ai/tts/websocket"
API_VERSION = "2026-08-14"
MODEL = "sonic-3.6"
BYTES_PER_SAMPLE = 2  # PCM16, mono
# Error codes worth one more try (besides 429 and 5xx).
_RETRYABLE_CODES = frozenset({"concurrency_limited"})


@dataclass(frozen=True, slots=True)
class CartesiaSettings:
    model: str = MODEL
    language: str = "en"
    speed: float = 1.0  # 0.6-1.5
    max_buffer_delay_ms: int = 0
    context_gap_s: float = 0.5  # renew a context this long after its last audio (expiry: 1 s)
    finish_timeout_s: float = 5.0  # after finish(), how long to wait for the rest of the audio
    connect_timeout_s: float = 5.0
    # word -> how to say it: a sounds-like spelling ("DOOR-gah-poor") or Cartesia IPA
    pronunciations: Mapping[str, str] = field(default_factory=dict)


class Socket(Protocol):
    """The part of a WebSocket connection this file uses."""

    async def send(self, message: bytes | str) -> None: ...
    async def recv(self) -> str | bytes: ...
    async def close(self) -> None: ...


class Connect(Protocol):
    def __call__(
        self, url: str, headers: dict[str, str], timeout_s: float, *, ipv4: bool = False
    ) -> Awaitable[Socket]: ...


async def _websocket(
    url: str, headers: dict[str, str], timeout_s: float, *, ipv4: bool = False
) -> Socket:
    return await ws_connect(
        url,
        additional_headers=headers,
        open_timeout=timeout_s,
        compression=None,
        # Passed on to the TCP connect. AF_UNSPEC is the default: IPv6 or IPv4.
        family=socket.AF_INET if ipv4 else socket.AF_UNSPEC,
    )


class CartesiaTTS:
    """Owns the shared connection: opens it on first use, reopens it after a drop, and hands
    each message to the turn whose context it belongs to."""

    def __init__(
        self,
        api_key: str,
        settings: CartesiaSettings | None = None,
        *,
        connect: Connect = _websocket,
    ) -> None:
        self._headers = {"X-API-Key": api_key, "Cartesia-Version": API_VERSION}
        self._settings = settings or CartesiaSettings()
        self._connect = connect
        self._socket: Socket | None = None
        self._reader: asyncio.Task[None] | None = None
        self._streams: dict[str, CartesiaStream] = {}  # context id -> its turn
        self._connecting = asyncio.Lock()
        self._respell = respeller(self._settings.pronunciations)

    async def open(self, sample_rate: int, voice_id: str) -> "CartesiaStream":
        await self._ensure_connected()
        return CartesiaStream(
            self,
            self._settings,
            sample_rate=sample_rate,
            voice_id=voice_id,
            respell=self._respell,
        )

    async def aclose(self) -> None:
        """At shutdown."""
        reader, socket = self._reader, self._socket
        self._reader = self._socket = None
        if reader is not None:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        if socket is not None:
            try:
                await socket.close()
            except (WebSocketException, OSError):
                pass

    # -- used by the streams --------------------------------------------------------------

    def register(self, context_id: str, stream: "CartesiaStream") -> None:
        self._streams[context_id] = stream

    def forget(self, context_id: str) -> None:
        self._streams.pop(context_id, None)

    async def send(self, message: dict[str, Any]) -> None:
        socket = self._socket
        if socket is None:
            raise ProviderError("cartesia connection lost", retryable=True)
        try:
            await socket.send(json.dumps(message))
        except (WebSocketException, OSError) as e:
            raise ProviderError(f"cartesia send failed: {type(e).__name__}", retryable=True) from e

    # -- connection -----------------------------------------------------------------------

    async def _ensure_connected(self) -> None:
        async with self._connecting:  # two turns opening at once share one connection
            if self._socket is not None:
                return
            self._socket = await self._open()
            self._reader = asyncio.create_task(self._read(self._socket))

    async def _open(self) -> Socket:
        timeout_s = self._settings.connect_timeout_s
        for ipv4 in (False, True):
            try:
                return await self._connect(URL, self._headers, timeout_s, ipv4=ipv4)
            except InvalidStatus as e:  # the server answered the handshake with an HTTP error
                status = e.response.status_code
                retryable = status == 429 or status >= 500
                raise ProviderError(
                    f"cartesia refused the connection: {status}", retryable=retryable
                ) from e
            except (WebSocketException, OSError, TimeoutError) as e:
                if ipv4:
                    raise ProviderError(
                        f"cartesia connect failed: {type(e).__name__}", retryable=True
                    ) from e
                log.warning("cartesia connect failed (%s); trying IPv4", type(e).__name__)
        raise AssertionError("unreachable")

    async def _read(self, socket: Socket) -> None:
        try:
            while True:
                self._route(await socket.recv())
        except (WebSocketException, OSError) as e:
            reason = type(e).__name__
        if self._socket is socket:
            self._socket = None  # the next open() reconnects
        streams = set(self._streams.values())
        self._streams.clear()
        if streams:
            log.warning("cartesia connection lost mid-turn (%s)", reason)
        else:  # e.g. closed for being idle between calls: nothing was lost
            log.info("cartesia connection closed while idle (%s)", reason)
        for stream in streams:  # turns in progress fail; their audio isn't coming
            stream.fail(ProviderError(f"cartesia connection lost: {reason}", retryable=True))

    def _route(self, raw: str | bytes) -> None:
        if isinstance(raw, bytes):
            return  # Cartesia only sends text messages
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("cartesia sent a message that isn't JSON")
            return
        context_id = message.get("context_id")
        stream = self._streams.get(context_id) if isinstance(context_id, str) else None
        if stream is not None:
            stream.handle(context_id, message)
        elif message.get("type") == "error":
            log.error("cartesia error outside a turn: %s", message.get("message"))
        # else: a cancelled or finished turn's late audio


def respeller(pronunciations: Mapping[str, str]) -> "Callable[[str], str]":
    """A function that swaps each listed word for how to say it, whole words only and in
    any case ("Durgapur", "DURGAPUR,"); text without them comes back unchanged."""
    if not pronunciations:
        return lambda text: text
    by_word = {word.lower(): said for word, said in pronunciations.items()}
    words = "|".join(re.escape(w) for w in sorted(by_word, key=len, reverse=True))
    pattern = re.compile(rf"\b(?:{words})\b", re.IGNORECASE)
    return lambda text: pattern.sub(lambda m: by_word[m.group(0).lower()], text)


class CartesiaStream:
    """One agent turn. Audio comes out of audio() in the order it arrives."""

    def __init__(
        self,
        owner: CartesiaTTS,
        settings: CartesiaSettings,
        *,
        sample_rate: int,
        voice_id: str,
        respell: "Callable[[str], str]" = lambda text: text,
    ) -> None:
        self._owner = owner
        self._respell = respell
        self._settings = settings
        self._bytes_per_ms = sample_rate * BYTES_PER_SAMPLE / 1000
        # Sent with every message of a context; Cartesia wants them the same each time.
        self._fixed: dict[str, Any] = {
            "model_id": settings.model,
            "voice": {"id": voice_id},
            "language": settings.language,
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": sample_rate,
            },
            "add_timestamps": True,
            "max_buffer_delay_ms": settings.max_buffer_delay_ms,
            "generation_config": {"speed": settings.speed},
        }
        self._queue: asyncio.Queue[TtsChunk | ProviderError | None] = asyncio.Queue()
        self._context: str | None = None
        self._offsets: dict[str, float] = {}  # context -> where its audio starts in the turn (ms)
        self._audio_ms = 0.0  # audio received so far this turn
        self._pending: list[WordTiming] = []  # words whose audio hasn't all arrived yet
        self._last_said_at = 0.0
        self._last_audio_at: float | None = None
        self._ended = False
        self._finish_timer: asyncio.TimerHandle | None = None

    async def say(self, sentence: str) -> None:
        if self._ended:
            return
        if self._context is None or self._expired():
            self._new_context()
        self._last_said_at = _now()
        # Cartesia joins transcripts exactly as sent; the space keeps sentences apart.
        await self._owner.send(self._message(self._respell(sentence) + " ", more=True))

    async def finish(self) -> None:
        if self._ended:
            return
        if self._context is None or self._expired():
            self._end()  # nothing said, or all of it has already arrived
            return
        await self._owner.send(self._message("", more=False))
        self._finish_timer = asyncio.get_running_loop().call_later(
            self._settings.finish_timeout_s, self._timed_out
        )

    async def audio(self) -> AsyncIterator[TtsChunk]:
        while (item := await self._queue.get()) is not None:
            if isinstance(item, ProviderError):
                raise item
            yield item

    async def cancel(self) -> None:
        if self._ended:
            return
        context = self._context
        self._end()
        if context is not None:
            try:
                await self._owner.send({"context_id": context, "cancel": True})
            except ProviderError:
                pass  # the connection is gone, and with it anything left to stop

    # -- called by CartesiaTTS --------------------------------------------------------------

    def handle(self, context_id: str, message: dict[str, Any]) -> None:
        match message.get("type"):
            case "chunk":
                if data := message.get("data"):
                    self._on_audio(base64.b64decode(data))
            case "timestamps":
                self._on_timestamps(context_id, message.get("word_timestamps") or {})
            case "error":
                self._on_error(message)
                return
            case _:
                pass  # done is handled below; flush_done and phoneme_timestamps aren't used
        if message.get("done") is True and context_id == self._context:
            self._on_done()

    def fail(self, error: ProviderError) -> None:
        self._end(error)

    # -- internals ------------------------------------------------------------------------

    def _expired(self) -> bool:
        """Has the context probably expired? Cartesia drops it 1 s after its last audio, so
        renew a little sooner, once both the text and the audio have gone quiet."""
        if self._last_audio_at is None:
            return False
        now = _now()
        gap = self._settings.context_gap_s
        return now - self._last_audio_at >= gap and now - self._last_said_at >= gap

    def _new_context(self) -> None:
        self._context = uuid.uuid4().hex
        self._offsets[self._context] = self._audio_ms
        self._owner.register(self._context, self)

    def _message(self, transcript: str, *, more: bool) -> dict[str, Any]:
        return {
            **self._fixed,
            "context_id": self._context,
            "transcript": transcript,
            "continue": more,
        }

    def _on_audio(self, pcm: bytes) -> None:
        self._audio_ms += len(pcm) / self._bytes_per_ms
        self._last_audio_at = _now()
        heard = 0
        while heard < len(self._pending) and self._pending[heard].end_ms <= self._audio_ms:
            heard += 1
        words, self._pending = tuple(self._pending[:heard]), self._pending[heard:]
        self._put(TtsChunk(pcm, words))

    def _on_timestamps(self, context_id: str, timestamps: dict[str, Any]) -> None:
        offset = self._offsets.get(context_id, 0.0)
        for word, start_s, end_s in zip(
            timestamps.get("words", ()),
            timestamps.get("start", ()),
            timestamps.get("end", ()),
            strict=False,
        ):
            start_ms = round(start_s * 1000 + offset)
            end_ms = round(end_s * 1000 + offset)
            self._pending.append(WordTiming(str(word), start_ms, end_ms))
        self._pending.sort(key=lambda w: w.end_ms)  # _on_audio hands them out from the front

    def _on_done(self) -> None:
        if self._pending:  # timings that came after the last audio
            self._put(TtsChunk(b"", tuple(self._pending)))
            self._pending = []
        self._end()

    def _on_error(self, message: dict[str, Any]) -> None:
        code = message.get("error_code")
        status = message.get("status_code")
        retryable = code in _RETRYABLE_CODES or (
            isinstance(status, int) and (status == 429 or status >= 500)
        )
        detail = f"cartesia {status} {code}: {message.get('message')}"
        log.error("%s (request %s)", detail, message.get("request_id"))
        self._end(ProviderError(detail, retryable=retryable))

    def _timed_out(self) -> None:
        self._end(ProviderError("cartesia: the audio never finished", retryable=True))

    def _put(self, chunk: TtsChunk) -> None:
        if not self._ended:
            self._queue.put_nowait(chunk)

    def _end(self, error: ProviderError | None = None) -> None:
        if self._ended:
            return
        self._ended = True
        if self._finish_timer is not None:
            self._finish_timer.cancel()
        for context_id in self._offsets:
            self._owner.forget(context_id)  # anything later for this turn is dropped
        if error is not None:
            self._queue.put_nowait(error)
        self._queue.put_nowait(None)  # ends audio()


def _now() -> float:
    return asyncio.get_running_loop().time()
