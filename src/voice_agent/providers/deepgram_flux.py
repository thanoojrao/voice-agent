"""Deepgram Flux as the speech-to-text, over a raw WebSocket to /v2/listen.

Flux transcribes and decides when the caller has finished their turn. This file speaks its
protocol directly (no Deepgram SDK) and maps it onto the core's events:

  TurnInfo StartOfTurn    -> TurnStarted, then Words with the first words heard
  TurnInfo Update         -> Words, when the turn's transcript has changed
  TurnInfo EndOfTurn      -> TurnEnded with the turn's transcript
  EagerEndOfTurn, TurnResumed: only sent when eager mode is on, which v0 leaves off
  Error                   -> logged; the server closes next, which is handled as a drop

Why a raw WebSocket: the protocol is one URL, binary audio out and a few JSON messages back.
The SDK's async example sends audio through a private method, and its Flux support is newer
than the API itself, so it would add a dependency and risk without saving much code.

Audio goes out in 80 ms chunks, as Deepgram recommends; the phone sends 20 ms frames, so a
few are collected first. If the connection drops, it reconnects once (design §10); a turn in
progress ends with the words heard so far. A second drop ends the stream with ProviderError,
which the session treats as a fatal provider failure.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode

from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import InvalidStatus, WebSocketException

from voice_agent.core.interfaces import ProviderError, SttEvent, TurnEnded, TurnStarted, Words

log = logging.getLogger(__name__)

URL = "wss://api.deepgram.com/v2/listen"
MODEL = "flux-general-en"
CHUNK_MS = 80  # Deepgram's recommended chunk size
BYTES_PER_SAMPLE = 2  # linear16, mono
CLOSE_STREAM = json.dumps({"type": "CloseStream"})


@dataclass(frozen=True, slots=True)
class FluxSettings:
    eot_threshold: float = 0.7  # how sure Flux must be that the caller is done (0.5-1.0)
    eot_timeout_ms: int = 5000  # the longest silence before the turn ends regardless
    keyterms: tuple[str, ...] = ()  # names and words to recognise, e.g. the owner's name
    model: str = MODEL
    connect_timeout_s: float = 5.0
    close_timeout_s: float = 2.0  # wait for the last transcripts after CloseStream


class Socket(Protocol):
    """The part of a WebSocket connection this file uses."""

    async def send(self, message: bytes | str) -> None: ...
    async def recv(self) -> str | bytes: ...
    async def close(self) -> None: ...


Connect = Callable[[str, dict[str, str], float], Awaitable[Socket]]


async def _websocket(url: str, headers: dict[str, str], timeout_s: float) -> Socket:
    # No compression: audio doesn't compress, and it would cost time on every frame.
    return await ws_connect(
        url, additional_headers=headers, open_timeout=timeout_s, compression=None
    )


class DeepgramFlux:
    def __init__(
        self, api_key: str, settings: FluxSettings | None = None, *, connect: Connect = _websocket
    ) -> None:
        self._headers = {"Authorization": f"Token {api_key}"}
        self._settings = settings or FluxSettings()
        self._connect = connect

    async def open(self, sample_rate: int) -> "FluxStream":
        s = self._settings
        url = flux_url(s, sample_rate)
        stream = FluxStream(
            lambda: self._connect(url, self._headers, s.connect_timeout_s),
            chunk_bytes=sample_rate * BYTES_PER_SAMPLE * CHUNK_MS // 1000,
            close_timeout_s=s.close_timeout_s,
        )
        await stream.start()
        return stream


def flux_url(settings: FluxSettings, sample_rate: int) -> str:
    params = [
        ("model", settings.model),
        ("encoding", "linear16"),
        ("sample_rate", str(sample_rate)),
        ("eot_threshold", str(settings.eot_threshold)),
        ("eot_timeout_ms", str(settings.eot_timeout_ms)),
        *(("keyterm", term) for term in settings.keyterms),  # repeated, one per term
    ]
    return f"{URL}?{urlencode(params)}"


class FluxStream:
    """One call's connection. Audio goes in through send(); events come out of events()."""

    def __init__(
        self,
        connect: Callable[[], Awaitable[Socket]],
        *,
        chunk_bytes: int,
        close_timeout_s: float,
    ) -> None:
        self._connect = connect
        self._chunk_bytes = chunk_bytes
        self._close_timeout_s = close_timeout_s
        self._socket: Socket | None = None
        self._buffer = bytearray()
        self._events: asyncio.Queue[SttEvent | ProviderError | None] = asyncio.Queue()
        self._reader: asyncio.Task[None] | None = None
        self._closing = False
        self._reconnects_left = 1
        self._in_turn = False
        self._transcript = ""  # the current turn's words so far
        self._last_error = ""  # the server's last Error message, for the failure report

    async def start(self) -> None:
        self._socket = await self._open()
        self._reader = asyncio.create_task(self._read())

    async def send(self, pcm: bytes) -> None:
        """Buffer `pcm` and send it in 80 ms chunks. Never raises: during a reconnect the
        audio is dropped, and a broken connection is noticed and handled by the reader."""
        if self._closing:
            return
        self._buffer += pcm
        while len(self._buffer) >= self._chunk_bytes:
            chunk = bytes(self._buffer[: self._chunk_bytes])
            del self._buffer[: self._chunk_bytes]
            await self._send(chunk)

    async def events(self) -> AsyncIterator[SttEvent]:
        while (item := await self._events.get()) is not None:
            if isinstance(item, ProviderError):
                raise item
            yield item

    async def close(self) -> None:
        """Send the audio still buffered, ask Flux to finish, and wait briefly for the last
        transcripts. Flux closes the connection itself once it has sent them."""
        if self._closing:
            return
        self._closing = True
        if self._buffer:
            await self._send(bytes(self._buffer))
            self._buffer.clear()
        await self._send(CLOSE_STREAM)
        if self._reader is not None:
            done, _ = await asyncio.wait({self._reader}, timeout=self._close_timeout_s)
            if not done:
                self._reader.cancel()
                await asyncio.gather(self._reader, return_exceptions=True)
        if self._socket is not None:
            try:
                await self._socket.close()
            except (WebSocketException, OSError):
                pass

    # -- connection -----------------------------------------------------------------------

    async def _open(self) -> Socket:
        try:
            return await self._connect()
        except InvalidStatus as e:  # the server answered the handshake with an HTTP error
            status = e.response.status_code
            retryable = status == 429 or status >= 500
            raise ProviderError(
                f"deepgram refused the connection: {status}", retryable=retryable
            ) from e
        except (WebSocketException, OSError, TimeoutError) as e:
            raise ProviderError(
                f"deepgram connect failed: {type(e).__name__}", retryable=True
            ) from e

    async def _send(self, message: bytes | str) -> None:
        socket = self._socket
        if socket is None:
            return
        try:
            await socket.send(message)
        except (WebSocketException, OSError):
            pass  # the reader sees the same drop on its next receive and reconnects

    async def _read(self) -> None:
        try:
            while True:
                assert self._socket is not None
                try:
                    raw = await self._socket.recv()
                except (WebSocketException, OSError) as e:
                    if self._closing:
                        break
                    await self._reconnect(e)
                    continue
                self._handle(raw)
        except ProviderError as e:
            self._events.put_nowait(e)
        finally:
            self._events.put_nowait(None)  # ends events()

    async def _reconnect(self, cause: Exception) -> None:
        reason = self._last_error or type(cause).__name__
        self._end_turn_early()
        if self._reconnects_left == 0:
            raise ProviderError(f"deepgram connection lost again: {reason}", retryable=True)
        self._reconnects_left -= 1
        log.warning("deepgram connection lost (%s); reconnecting", reason)
        self._socket = None  # audio arriving meanwhile is dropped
        self._buffer.clear()
        self._last_error = ""
        self._socket = await self._open()

    def _end_turn_early(self) -> None:
        """A drop mid-turn: answer what was heard rather than lose it."""
        if self._in_turn and self._transcript:
            self._events.put_nowait(TurnEnded(self._transcript))
        self._in_turn = False
        self._transcript = ""

    # -- messages -------------------------------------------------------------------------

    def _handle(self, raw: str | bytes) -> None:
        if isinstance(raw, bytes):
            return  # Flux only sends text messages
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("deepgram sent a message that isn't JSON")
            return
        match message.get("type"):
            case "TurnInfo":
                self._turn_info(message)
            case "Connected":
                log.info("deepgram connected (request %s)", message.get("request_id"))
            case "Error":
                self._last_error = f"{message.get('code')}: {message.get('description')}"
                log.error("deepgram error %s", self._last_error)
            case kind:
                log.debug("deepgram %s: %s", kind, message.get("description", ""))

    def _turn_info(self, message: dict[str, object]) -> None:
        event = message.get("event")
        text = str(message.get("transcript") or "").strip()
        match event:
            case "StartOfTurn":
                self._in_turn = True
                self._transcript = text
                self._events.put_nowait(TurnStarted())
                if text:
                    self._events.put_nowait(Words(text))
            case "Update":
                # Updates also come about every 0.25 s between turns; only a turn's count.
                if self._in_turn and text and text != self._transcript:
                    self._transcript = text
                    self._events.put_nowait(Words(text))
            case "EndOfTurn":
                self._in_turn = False
                self._transcript = ""
                self._events.put_nowait(TurnEnded(text))
            case _:
                pass  # EagerEndOfTurn / TurnResumed: eager mode is off in v0
