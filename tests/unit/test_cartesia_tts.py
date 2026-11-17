"""The Cartesia provider against a fake WebSocket.

The fake server plays back messages in the shapes from Cartesia's TTS WebSocket reference
(chunk, timestamps, done, error) and records what the provider sends. No network, no cost.
Messages recorded from the live API come with the step 3 script.
"""

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError, InvalidStatus
from websockets.http11 import Response

from voice_agent.core.interfaces import ProviderError, TtsChunk, WordTiming
from voice_agent.providers.cartesia_tts import (
    API_VERSION,
    CartesiaSettings,
    CartesiaStream,
    CartesiaTTS,
    respeller,
)

RATE = 8000
VOICE = "voice-1"
MS_100 = b"\x01\x00" * 800  # 100 ms of PCM16 at 8 kHz


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self._incoming: asyncio.Queue[str | bytes | Exception] = asyncio.Queue()

    async def send(self, message: bytes | str) -> None:
        if self.closed:
            raise ConnectionClosedError(None, None)
        self.sent.append(json.loads(message))

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
        self.closed = True
        self._incoming.put_nowait(ConnectionClosedError(None, None))

    def context(self, n: int = -1) -> str:
        """The context id of the n-th message sent."""
        return self.sent[n]["context_id"]


class Server:
    """Hands out a fresh socket per connection, or raises what it's told to."""

    def __init__(self, *outcomes: FakeSocket | Exception) -> None:
        self._outcomes = list(outcomes)
        self.connections: list[tuple[str, dict[str, str], float]] = []
        self.ipv4: list[bool] = []

    async def connect(
        self, url: str, headers: dict[str, str], timeout_s: float, *, ipv4: bool = False
    ) -> FakeSocket:
        self.connections.append((url, headers, timeout_s))
        self.ipv4.append(ipv4)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def chunk(context_id: str, pcm: bytes = MS_100) -> dict[str, Any]:
    return {
        "type": "chunk",
        "data": base64.b64encode(pcm).decode(),
        "done": False,
        "status_code": 206,
        "step_time": 40,
        "context_id": context_id,
    }


def timestamps(context_id: str, *words: tuple[str, float, float]) -> dict[str, Any]:
    return {
        "type": "timestamps",
        "done": False,
        "status_code": 206,
        "context_id": context_id,
        "word_timestamps": {
            "words": [w for w, _, _ in words],
            "start": [s for _, s, _ in words],
            "end": [e for _, _, e in words],
        },
    }


def done(context_id: str) -> dict[str, Any]:
    return {"type": "done", "done": True, "status_code": 206, "context_id": context_id}


def error(context_id: str, code: str | None, status: int) -> dict[str, Any]:
    return {
        "type": "error",
        "done": True,
        "title": "Nope",
        "message": "something went wrong",
        "error_code": code,
        "status_code": status,
        "request_id": "req-1",
        "context_id": context_id,
    }


def make_tts(server: Server, settings: CartesiaSettings | None = None) -> CartesiaTTS:
    return CartesiaTTS("ct-key", settings, connect=server.connect)


async def take(audio: AsyncIterator[TtsChunk], n: int) -> list[TtsChunk]:
    async with asyncio.timeout(1):
        return [await anext(audio) for _ in range(n)]


async def rest(audio: AsyncIterator[TtsChunk]) -> list[TtsChunk]:
    async with asyncio.timeout(1):
        return [c async for c in audio]


async def opened(socket: FakeSocket, settings: CartesiaSettings | None = None) -> CartesiaStream:
    return await make_tts(Server(socket), settings).open(RATE, VOICE)


# -- connecting -----------------------------------------------------------------------------


async def test_connects_once_with_the_key_and_version_in_headers_and_reuses_it():
    server = Server(FakeSocket())
    tts = make_tts(server)
    await tts.open(RATE, VOICE)
    await tts.open(RATE, VOICE)  # the next turn

    [(url, headers, timeout_s)] = server.connections
    assert url == "wss://api.cartesia.ai/tts/websocket"
    assert headers == {"X-API-Key": "ct-key", "Cartesia-Version": API_VERSION}
    assert "ct-key" not in url
    assert timeout_s == CartesiaSettings().connect_timeout_s


async def test_a_network_failure_is_tried_once_more_over_ipv4():
    server = Server(ConnectionResetError(), FakeSocket())
    await make_tts(server).open(RATE, VOICE)
    assert server.ipv4 == [False, True]


async def test_failing_to_connect_twice_raises_provider_error():
    server = Server(OSError("no route to host"), OSError("no route to host"))
    with pytest.raises(ProviderError) as caught:
        await make_tts(server).open(RATE, VOICE)
    assert caught.value.retryable
    assert len(server.connections) == 2


async def test_a_refused_handshake_is_not_retried():
    refused = InvalidStatus(Response(401, "Unauthorized", Headers()))
    server = Server(refused, FakeSocket())
    with pytest.raises(ProviderError) as caught:
        await make_tts(server).open(RATE, VOICE)
    assert not caught.value.retryable
    assert "401" in str(caught.value)
    assert len(server.connections) == 1


# -- speaking -------------------------------------------------------------------------------


async def test_a_turns_sentences_go_into_one_context_then_it_is_closed():
    socket = FakeSocket()
    stream = await opened(socket)
    await stream.say("Sure, I can help.")
    await stream.say("Widget is a tool.")
    await stream.finish()

    first, second, last = socket.sent
    assert first == {
        "model_id": "sonic-3.6",
        "voice": {"id": VOICE},
        "language": "en",
        "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": RATE},
        "add_timestamps": True,
        "max_buffer_delay_ms": 0,
        "generation_config": {"speed": 1.0},
        "context_id": first["context_id"],
        "transcript": "Sure, I can help. ",
        "continue": True,
    }
    assert second == {**first, "transcript": "Widget is a tool. "}
    assert last == {**first, "transcript": "", "continue": False}


async def test_audio_comes_out_with_word_timings_once_their_audio_has_arrived():
    socket = FakeSocket()
    stream = await opened(socket)
    audio = stream.audio()
    await stream.say("Hi there, Sam.")
    await stream.finish()
    ctx = socket.context()
    socket.push(
        timestamps(ctx, ("Hi", 0.0, 0.08), ("there,", 0.1, 0.19), ("Sam.", 0.21, 0.5)),
        chunk(ctx),  # 0-100 ms: "Hi"
        chunk(ctx),  # 100-200 ms: "there,"
        chunk(ctx),  # 200-300 ms: "Sam." isn't over yet
        timestamps(ctx, ("late", 0.3, 0.35)),  # timings after their audio
        chunk(ctx),  # 300-400 ms: "late"
        done(ctx),
    )
    assert await rest(audio) == [
        TtsChunk(MS_100, (WordTiming("Hi", 0, 80),)),
        TtsChunk(MS_100, (WordTiming("there,", 100, 190),)),
        TtsChunk(MS_100, ()),
        TtsChunk(MS_100, (WordTiming("late", 300, 350),)),
        TtsChunk(b"", (WordTiming("Sam.", 210, 500),)),  # still owed when the audio ended
    ]


async def test_each_turn_gets_its_own_context_and_only_its_own_audio():
    socket = FakeSocket()
    tts = make_tts(Server(socket))
    one, two = await tts.open(RATE, VOICE), await tts.open(RATE, VOICE)
    await one.say("First.")
    await two.say("Second.")
    ctx_one, ctx_two = socket.context(0), socket.context(1)
    assert ctx_one != ctx_two

    socket.push(chunk(ctx_two, b"\x02\x00"), chunk(ctx_one, b"\x01\x00"))
    assert await take(one.audio(), 1) == [TtsChunk(b"\x01\x00")]
    assert await take(two.audio(), 1) == [TtsChunk(b"\x02\x00")]


async def test_finishing_a_turn_that_said_nothing_ends_audio_without_sending():
    socket = FakeSocket()
    stream = await opened(socket)
    await stream.finish()
    assert await rest(stream.audio()) == []
    assert socket.sent == []


async def test_pronunciations_are_swapped_in_only_on_the_way_to_cartesia():
    socket = FakeSocket()
    settings = CartesiaSettings(pronunciations={"Durgapur": "DOOR-gah-poor"})
    stream = await opened(socket, settings)
    await stream.say("I studied in Durgapur, at DURGAPUR's college; not durgapurx.")
    assert socket.sent[0]["transcript"] == (
        "I studied in DOOR-gah-poor, at DOOR-gah-poor's college; not durgapurx. "
    )


def test_respeller_leaves_text_alone_with_no_pronunciations():
    assert respeller({})("Durgapur") == "Durgapur"
    assert respeller({"ab": "X", "abc": "Y"})("ab abc") == "X Y"  # longest match wins


# -- pauses ---------------------------------------------------------------------------------


async def test_a_long_pause_starts_a_new_context_on_the_same_timeline():
    socket = FakeSocket()
    stream = await opened(socket, CartesiaSettings(context_gap_s=0.05))
    audio = stream.audio()
    await stream.say("Let me save that.")
    first = socket.context()
    socket.push(chunk(first), chunk(first))  # 200 ms of audio
    await take(audio, 2)

    await asyncio.sleep(0.1)  # a tool runs; the context would have expired
    await stream.say("Done.")
    await stream.finish()
    second = socket.context()
    assert second != first
    assert socket.sent[-2]["transcript"] == "Done. "

    socket.push(timestamps(second, ("Done.", 0.0, 0.1)), chunk(second), done(second))
    assert await rest(audio) == [TtsChunk(MS_100, (WordTiming("Done.", 200, 300),))]


async def test_finishing_after_a_long_pause_ends_without_reopening_the_context():
    socket = FakeSocket()
    stream = await opened(socket, CartesiaSettings(context_gap_s=0.05))
    audio = stream.audio()
    await stream.say("Let me save that.")
    socket.push(chunk(socket.context()))
    await take(audio, 1)

    await asyncio.sleep(0.1)  # a tool ran, and there was nothing more to say
    await stream.finish()
    assert await rest(audio) == []
    assert len(socket.sent) == 1


async def test_finish_gives_up_if_the_audio_never_ends():
    socket = FakeSocket()
    stream = await opened(socket, CartesiaSettings(finish_timeout_s=0.05))
    await stream.say("Hello.")
    await stream.finish()
    with pytest.raises(ProviderError) as caught:
        await rest(stream.audio())
    assert caught.value.retryable


# -- cancel ---------------------------------------------------------------------------------


async def test_cancel_ends_audio_at_once_and_drops_what_comes_after():
    socket = FakeSocket()
    tts = make_tts(Server(socket))
    stream = await tts.open(RATE, VOICE)
    audio = stream.audio()
    await stream.say("Widget is a tool for teams.")
    ctx = socket.context()
    socket.push(chunk(ctx))
    assert len(await take(audio, 1)) == 1

    await stream.cancel()
    assert socket.sent[-1] == {"context_id": ctx, "cancel": True}
    socket.push(chunk(ctx), done(ctx))  # already being made when the cancel arrived
    assert await rest(audio) == []

    nxt = await tts.open(RATE, VOICE)  # the connection is still good for the next turn
    await nxt.say("Sorry, go ahead.")
    socket.push(chunk(socket.context()))
    assert len(await take(nxt.audio(), 1)) == 1


# -- errors ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "status", "retryable"),
    [
        ("concurrency_limited", 429, True),
        (None, 500, True),
        ("voice_not_found", 404, False),
        ("quota_exceeded", 402, False),
    ],
)
async def test_an_error_message_ends_the_turn_with_provider_error(code, status, retryable):
    socket = FakeSocket()
    stream = await opened(socket)
    await stream.say("Hello.")
    socket.push(error(socket.context(), code, status))
    with pytest.raises(ProviderError) as caught:
        await rest(stream.audio())
    assert caught.value.retryable is retryable


async def test_a_dropped_connection_fails_the_turn_and_the_next_turn_reconnects():
    first, second = FakeSocket(), FakeSocket()
    server = Server(first, second)
    tts = make_tts(server)
    stream = await tts.open(RATE, VOICE)
    await stream.say("Hello.")
    first.drop()
    with pytest.raises(ProviderError) as caught:
        await rest(stream.audio())
    assert caught.value.retryable

    nxt = await tts.open(RATE, VOICE)
    await nxt.say("Hello again.")
    assert len(server.connections) == 2
    assert second.sent[0]["transcript"] == "Hello again. "


async def test_an_idle_close_between_calls_is_not_a_warning(caplog):
    socket = FakeSocket()
    tts = make_tts(Server(socket, FakeSocket()))
    await tts.open(RATE, VOICE)  # a turn that never spoke: nothing in progress
    with caplog.at_level("INFO", logger="voice_agent.providers.cartesia_tts"):
        socket.drop()
        await asyncio.sleep(0.01)
    [record] = caplog.records
    assert record.levelname == "INFO"
    assert "idle" in record.message


async def test_saying_on_a_broken_connection_raises():
    socket = FakeSocket()
    stream = await opened(socket)
    socket.closed = True  # broken; the reader hasn't noticed yet
    with pytest.raises(ProviderError):
        await stream.say("Hello.")


async def test_unknown_and_malformed_messages_are_ignored():
    socket = FakeSocket()
    stream = await opened(socket)
    await stream.say("Hi.")
    ctx = socket.context()
    socket.push(
        "not json",
        b"\x00\x01",
        {"type": "flush_done", "done": False, "flush_id": 1, "context_id": ctx},
        {"type": "SomethingNew", "context_id": ctx},
        chunk("someone-elses-context"),
        chunk(ctx),
    )
    assert await take(stream.audio(), 1) == [TtsChunk(MS_100)]
