"""The Flux provider against a fake WebSocket.

The fake server plays back messages in the shapes from Deepgram's Flux API reference
(TurnInfo, Connected, Error, Warning) and records what the provider sends. No network, no
cost. Messages recorded from the live API come with the step 3 script.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from websockets.exceptions import ConnectionClosedError

from voice_agent.core.interfaces import ProviderError, SttEvent, TurnEnded, TurnStarted, Words
from voice_agent.providers.deepgram_flux import (
    CLOSE_STREAM,
    DeepgramFlux,
    FluxSettings,
    FluxStream,
)

RATE = 8000
FRAME = b"\x01\x00" * 160  # 20 ms of linear16 at 8 kHz, as the phone sends it
CHUNK = 1280  # 80 ms


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[bytes | str] = []
        self.closed = False
        self._incoming: asyncio.Queue[str | bytes | Exception] = asyncio.Queue()

    async def send(self, message: bytes | str) -> None:
        if self.closed:
            raise ConnectionClosedError(None, None)
        self.sent.append(message)
        if message == CLOSE_STREAM:
            self.drop()  # Flux closes once it has sent the last transcripts

    async def recv(self) -> str | bytes:
        item = await self._incoming.get()
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self) -> None:
        self.closed = True

    def push(self, *messages: dict[str, Any] | str | bytes) -> None:
        for m in messages:
            self._incoming.put_nowait(json.dumps(m) if isinstance(m, dict) else m)

    def drop(self) -> None:
        self._incoming.put_nowait(ConnectionClosedError(None, None))

    @property
    def audio(self) -> list[bytes]:
        return [m for m in self.sent if isinstance(m, bytes)]


class Server:
    """Hands out a fresh socket per connection, or raises what it's told to."""

    def __init__(self, *outcomes: FakeSocket | Exception) -> None:
        self._outcomes = list(outcomes)
        self.connections: list[tuple[str, dict[str, str], float]] = []

    async def connect(self, url: str, headers: dict[str, str], timeout_s: float) -> FakeSocket:
        self.connections.append((url, headers, timeout_s))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def turn(event: str, transcript: str = "", **extra: Any) -> dict[str, Any]:
    return {
        "type": "TurnInfo",
        "request_id": "req-1",
        "sequence_id": 1,
        "event": event,
        "turn_index": 0,
        "audio_window_start": 0.0,
        "audio_window_end": 1.0,
        "transcript": transcript,
        "words": [{"word": w, "confidence": 0.9} for w in transcript.split()],
        "end_of_turn_confidence": 0.8 if event == "EndOfTurn" else 0.1,
        **extra,
    }


async def open_stream(server: Server, settings: FluxSettings | None = None) -> FluxStream:
    flux = DeepgramFlux("dg-key", settings, connect=server.connect)
    return await flux.open(RATE)


async def take(events: AsyncIterator[SttEvent], n: int) -> list[SttEvent]:
    async with asyncio.timeout(1):
        return [await anext(events) for _ in range(n)]


# -- connecting -----------------------------------------------------------------------------


async def test_connects_to_flux_v2_with_the_settings_and_the_key_in_a_header():
    server = Server(FakeSocket())
    settings = FluxSettings(eot_threshold=0.8, eot_timeout_ms=3000, keyterms=("Alex", "Widget Co"))
    await open_stream(server, settings)

    [(url, headers, timeout_s)] = server.connections
    assert url == (
        "wss://api.deepgram.com/v2/listen?model=flux-general-en&encoding=linear16"
        "&sample_rate=8000&eot_threshold=0.8&eot_timeout_ms=3000&keyterm=Alex&keyterm=Widget+Co"
    )
    assert headers == {"Authorization": "Token dg-key"}
    assert "dg-key" not in url
    assert timeout_s == settings.connect_timeout_s


async def test_failing_to_connect_raises_provider_error():
    server = Server(OSError("no route to host"))
    with pytest.raises(ProviderError) as caught:
        await open_stream(server)
    assert caught.value.retryable


# -- audio ----------------------------------------------------------------------------------


async def test_audio_goes_out_in_80ms_chunks():
    socket = FakeSocket()
    stream = await open_stream(Server(socket))
    for _ in range(5):  # 100 ms of phone frames
        await stream.send(FRAME)
    assert socket.audio == [FRAME * 4]
    assert len(socket.audio[0]) == CHUNK


async def test_close_sends_the_rest_of_the_audio_then_close_stream_and_ends_events():
    socket = FakeSocket()
    stream = await open_stream(Server(socket))
    await stream.send(FRAME * 5)
    events = stream.events()

    await stream.close()

    assert socket.sent == [FRAME * 4, FRAME, CLOSE_STREAM]
    assert socket.closed
    with pytest.raises(StopAsyncIteration):
        await anext(events)
    await stream.send(FRAME)  # after close: ignored, no error
    assert len(socket.sent) == 3


async def test_close_gives_up_waiting_if_the_server_never_closes():
    socket = FakeSocket()
    socket.drop = lambda: None  # this server ignores CloseStream
    stream = await open_stream(Server(socket), FluxSettings(close_timeout_s=0.05))
    async with asyncio.timeout(1):
        await stream.close()
    assert socket.closed


# -- turns ----------------------------------------------------------------------------------


async def test_a_caller_turn_maps_to_started_words_and_ended():
    socket = FakeSocket()
    stream = await open_stream(Server(socket))
    socket.push(
        {"type": "Connected", "request_id": "req-1", "sequence_id": 0},
        turn("Update"),  # between turns: nothing said yet
        turn("StartOfTurn", "Hi"),
        turn("Update", "Hi"),  # unchanged: no event
        turn("Update", "Hi there"),
        turn("EagerEndOfTurn", "Hi there"),  # eager mode is off; ignored if it ever comes
        turn("TurnResumed", "Hi there"),
        turn("EndOfTurn", "Hi there, it's Sam.", trigger="model"),
    )
    events = stream.events()
    assert await take(events, 4) == [
        TurnStarted(),
        Words("Hi"),
        Words("Hi there"),
        TurnEnded("Hi there, it's Sam."),
    ]
    socket.push(turn("Update"), turn("StartOfTurn", "And"))  # nothing extra in between
    assert await take(events, 1) == [TurnStarted()]


async def test_unknown_and_malformed_messages_are_ignored():
    socket = FakeSocket()
    stream = await open_stream(Server(socket))
    socket.push(
        "not json",
        b"\x00\x01",
        {"type": "Warning", "code": "FORCE_END_TURN_NO_ACTIVE_TURN", "description": "x"},
        {"type": "SomethingNew"},
        turn("StartOfTurn", "Hello"),
    )
    assert await take(stream.events(), 2) == [TurnStarted(), Words("Hello")]


# -- drops ----------------------------------------------------------------------------------


async def test_a_drop_mid_turn_ends_it_with_the_words_so_far_and_reconnects_once():
    first, second = FakeSocket(), FakeSocket()
    server = Server(first, second)
    stream = await open_stream(server)
    events = stream.events()
    first.push(turn("StartOfTurn", "Can you"), turn("Update", "Can you tell"))
    first.drop()

    assert await take(events, 4) == [
        TurnStarted(),
        Words("Can you"),
        Words("Can you tell"),
        TurnEnded("Can you tell"),
    ]
    await stream.send(FRAME * 4)
    assert second.audio == [FRAME * 4]
    assert len(server.connections) == 2

    second.push(turn("StartOfTurn", "Sorry"))
    assert await take(events, 2) == [TurnStarted(), Words("Sorry")]


async def test_a_second_drop_ends_the_stream_with_the_servers_error():
    first, second = FakeSocket(), FakeSocket()
    stream = await open_stream(Server(first, second))
    events = stream.events()
    first.drop()
    second.push(
        {"type": "Error", "sequence_id": 3, "code": "INTERNAL_SERVER_ERROR", "description": "boom"}
    )
    second.drop()

    with pytest.raises(ProviderError) as caught:
        await take(events, 1)
    assert "INTERNAL_SERVER_ERROR" in str(caught.value)


async def test_a_failed_reconnect_ends_the_stream():
    first = FakeSocket()
    stream = await open_stream(Server(first, OSError("down")))
    first.drop()
    with pytest.raises(ProviderError):
        await take(stream.events(), 1)


async def test_sending_on_a_broken_connection_does_not_raise():
    socket = FakeSocket()
    stream = await open_stream(Server(socket, FakeSocket()))
    socket.closed = True  # the connection broke; the reader hasn't noticed yet
    await stream.send(FRAME * 4)
