import asyncio
import base64
import json
from typing import Any

import pytest
from starlette.websockets import WebSocketDisconnect

from voice_agent.adapters.twilio.adapter import TwilioAdapter
from voice_agent.core.interfaces import ChannelAdapter, SessionStart
from voice_agent.core.registry import AdmissionError

CLOSED = object()


class FakeSocket:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue[Any] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []
        self.closed = 0
        self.fail_sends = False

    def push(self, message: dict[str, Any] | str) -> None:
        self.incoming.put_nowait(message if isinstance(message, str) else json.dumps(message))

    def hang_up(self) -> None:
        self.incoming.put_nowait(CLOSED)

    async def receive_text(self) -> str:
        item = await self.incoming.get()
        if item is CLOSED:
            raise WebSocketDisconnect(1000)
        return item

    async def send_text(self, data: str) -> None:
        if self.fail_sends:
            raise RuntimeError("socket is gone")
        self.sent.append(json.loads(data))

    async def close(self, code: int = 1000) -> None:
        self.closed += 1
        self.hang_up()  # Twilio answers our close, so the read loop sees the disconnect


class FakeSession:
    def __init__(self) -> None:
        self.audio: list[bytes] = []
        self.marks: list[str] = []
        self.ended: list[str] = []

    async def on_audio(self, pcm: bytes) -> None:
        self.audio.append(pcm)

    async def on_mark_played(self, name: str) -> None:
        self.marks.append(name)

    async def on_session_end(self, reason: str) -> None:
        self.ended.append(reason)


class Starter:
    def __init__(self, refuse: bool = False) -> None:
        self.session = FakeSession()
        self.calls: list[tuple[str, SessionStart]] = []
        self.refuse = refuse

    def __call__(self, token: str, start: SessionStart, adapter: ChannelAdapter) -> FakeSession:
        self.calls.append((token, start))
        if self.refuse:
            raise AdmissionError("unknown, used or expired stream token")
        return self.session


def start_message(params: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "event": "start",
        "sequenceNumber": "1",
        "start": {
            "accountSid": "AC123",
            "streamSid": "MZ123",
            "callSid": "CA123",
            "tracks": ["inbound"],
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
            "customParameters": params
            if params is not None
            else {"from": "+15550100", "token": "tok"},
        },
        "streamSid": "MZ123",
    }


def media(mulaw: bytes) -> dict[str, Any]:
    payload = base64.b64encode(mulaw).decode()
    return {
        "event": "media",
        "streamSid": "MZ123",
        "media": {"track": "inbound", "payload": payload},
    }


def echo(name: str) -> dict[str, Any]:
    return {"event": "mark", "streamSid": "MZ123", "mark": {"name": name}}


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.fixture
async def call():
    """A running adapter that has received connected + start."""
    ws, starter = FakeSocket(), Starter()
    adapter = TwilioAdapter(ws, starter)
    task = asyncio.create_task(adapter.run())
    ws.push({"event": "connected", "protocol": "Call", "version": "1.0.0"})
    ws.push(start_message())
    await settle()
    yield ws, starter, adapter, task
    if not task.done():
        ws.hang_up()
    await asyncio.wait_for(task, 1)


# -- start ------------------------------------------------------------------------------------


async def test_start_begins_the_session_with_the_call_details(call):
    _, starter, _, _ = call
    [(token, start)] = starter.calls
    assert token == "tok"
    assert start == SessionStart(
        channel="twilio", channel_ref="CA123", caller_address="+15550100", sample_rate=8000
    )


async def test_withheld_number_becomes_none():
    ws, starter = FakeSocket(), Starter()
    task = asyncio.create_task(TwilioAdapter(ws, starter).run())
    ws.push(start_message({"from": "", "token": "tok"}))
    ws.hang_up()
    await asyncio.wait_for(task, 1)
    assert starter.calls[0][1].caller_address is None


async def test_refused_stream_is_closed_without_a_session():
    ws, starter = FakeSocket(), Starter(refuse=True)
    task = asyncio.create_task(TwilioAdapter(ws, starter).run())
    ws.push(start_message())
    await asyncio.wait_for(task, 1)
    assert ws.closed == 1
    assert starter.session.ended == []


async def test_start_missing_fields_closes():
    ws, starter = FakeSocket(), Starter()
    task = asyncio.create_task(TwilioAdapter(ws, starter).run())
    ws.push({"event": "start", "start": {"callSid": "CA123"}})
    await asyncio.wait_for(task, 1)
    assert ws.closed == 1
    assert starter.calls == []


# -- audio in ---------------------------------------------------------------------------------


async def test_media_is_decoded_to_pcm(call):
    ws, starter, _, _ = call
    ws.push(media(b"\xff\xff"))  # μ-law silence
    await settle()
    assert starter.session.audio == [b"\x00\x00\x00\x00"]


async def test_media_before_start_is_ignored():
    ws, starter = FakeSocket(), Starter()
    task = asyncio.create_task(TwilioAdapter(ws, starter).run())
    ws.push(media(b"\xff"))
    ws.push(start_message())
    ws.hang_up()
    await asyncio.wait_for(task, 1)
    assert starter.session.audio == []


async def test_bad_messages_are_skipped(call):
    ws, starter, _, task = call
    ws.push("not json")
    ws.push({"no": "event"})
    ws.push({"event": "media", "media": {"payload": "not base64!"}})
    ws.push({"event": "dtmf", "dtmf": {"track": "inbound", "digit": "1"}})
    ws.push({"event": "something-new"})
    ws.push(media(b"\xff"))
    await settle()
    assert not task.done()
    assert starter.session.audio == [b"\x00\x00"]


# -- audio out --------------------------------------------------------------------------------


async def test_send_audio_sends_mulaw_with_the_stream_sid(call):
    ws, _, adapter, _ = call
    await adapter.send_audio(b"\x00\x00\x00\x00")
    assert ws.sent == [{"event": "media", "media": {"payload": "//8="}, "streamSid": "MZ123"}]


async def test_empty_audio_sends_nothing(call):
    ws, _, adapter, _ = call
    await adapter.send_audio(b"")
    assert ws.sent == []


async def test_nothing_is_sent_before_start():
    ws = FakeSocket()
    adapter = TwilioAdapter(ws, Starter())
    await adapter.send_audio(b"\x00\x00")
    await adapter.mark("t1.end")
    assert ws.sent == []


async def test_mark_and_clear_messages(call):
    ws, _, adapter, _ = call
    await adapter.mark("t1.w3")
    await adapter.clear_audio()
    assert ws.sent == [
        {"event": "mark", "mark": {"name": "t1.w3"}, "streamSid": "MZ123"},
        {"event": "clear", "streamSid": "MZ123"},
    ]


# -- marks ------------------------------------------------------------------------------------


async def test_mark_echo_is_passed_on(call):
    ws, starter, adapter, _ = call
    await adapter.mark("t1.w3")
    ws.push(echo("t1.w3"))
    await settle()
    assert starter.session.marks == ["t1.w3"]


async def test_echoes_caused_by_clear_are_dropped(call):
    ws, starter, adapter, _ = call
    for name in ("t1.w3", "t1.w7", "t1.end"):
        await adapter.mark(name)
    ws.push(echo("t1.w3"))  # really played
    await settle()
    await adapter.clear_audio()
    ws.push(echo("t1.w7"))  # Twilio sends back the rest because of the clear
    ws.push(echo("t1.end"))
    await adapter.mark("t2.w2")  # the next turn
    ws.push(echo("t2.w2"))
    await settle()
    assert starter.session.marks == ["t1.w3", "t2.w2"]


async def test_clear_drops_only_one_echo_per_pending_mark(call):
    ws, starter, adapter, _ = call
    await adapter.mark("t3.apology")
    await adapter.clear_audio()
    await adapter.mark("t3.apology")  # same name sent again after the clear
    ws.push(echo("t3.apology"))  # the cleared one
    ws.push(echo("t3.apology"))  # the new one, really played
    await settle()
    assert starter.session.marks == ["t3.apology"]


# -- ending -----------------------------------------------------------------------------------


async def test_stop_tells_the_session_the_caller_hung_up(call):
    ws, starter, _, task = call
    ws.push({"event": "stop", "stop": {"accountSid": "AC123", "callSid": "CA123"}})
    await asyncio.wait_for(task, 1)
    assert starter.session.ended == ["caller_hangup"]


async def test_dropped_socket_counts_as_hang_up(call):
    ws, starter, _, task = call
    ws.hang_up()
    await asyncio.wait_for(task, 1)
    assert starter.session.ended == ["caller_hangup"]


async def test_end_session_closes_once_and_is_not_reported_back(call):
    ws, starter, adapter, task = call
    await adapter.end_session()
    await adapter.end_session()
    await asyncio.wait_for(task, 1)
    assert ws.closed == 1
    assert starter.session.ended == []


async def test_nothing_is_sent_after_end_session(call):
    ws, _, adapter, _ = call
    await adapter.end_session()
    await adapter.send_audio(b"\x00\x00")
    assert ws.sent == []


async def test_failed_send_stops_sending_but_hang_up_is_still_reported(call):
    ws, starter, adapter, task = call
    ws.fail_sends = True
    await adapter.send_audio(b"\x00\x00")  # doesn't raise
    ws.fail_sends = False
    await adapter.send_audio(b"\x00\x00")
    assert ws.sent == []
    ws.hang_up()
    await asyncio.wait_for(task, 1)
    assert starter.session.ended == ["caller_hangup"]


async def test_followups_are_not_supported_yet():
    with pytest.raises(NotImplementedError):
        await TwilioAdapter(FakeSocket(), Starter()).send_followup("hi", None)
