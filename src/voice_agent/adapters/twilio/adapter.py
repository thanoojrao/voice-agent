"""One Twilio Media Stream, connected to one session (design §7.2).

Twilio connects to our WebSocket after the webhook's TwiML. Then:

    Twilio → us   connected   ignored
                  start       stream and call IDs + our <Parameter>s → start the session
                  media       base64 μ-law → PCM16 → session.on_audio
                  mark        one of our marks has played → session.on_mark_played
                  dtmf        keypad press, ignored
                  stop        the call ended → session.on_session_end("caller_hangup")
    us → Twilio   media, mark, clear (the core's send_audio / mark / clear_audio)

Closing the socket ends the call: the TwiML has <Hangup/> after <Connect> (see twiml.py).

Marks after a clear. Twilio's docs: clear "empties all buffered audio and causes any mark
messages to be sent back". Those echoes are for audio the caller never heard, and passing
them on would tell the turn the whole reply was heard, which breaks trimming the history to
what was actually said (§6.3). So the adapter remembers which marks are still waiting, and
on clear drops the echoes of those. The core only ever sees marks for audio that played.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from collections.abc import Callable
from typing import Any, Protocol

from starlette.websockets import WebSocketDisconnect

from voice_agent.adapters.twilio.audio import SAMPLE_RATE, Encoder, decode
from voice_agent.core.interfaces import (
    Capability,
    ChannelAdapter,
    SessionCallbacks,
    SessionStart,
)
from voice_agent.core.registry import AdmissionError
from voice_agent.core.session import CALLER_HANGUP

log = logging.getLogger(__name__)

CHANNEL = "twilio"
# The <Parameter> names the webhook puts in the TwiML (routes.py) and we read back here.
PARAM_FROM = "from"  # the caller's E.164 number, or "" when withheld
PARAM_TOKEN = "token"  # the registry's one-time slot token


class Socket(Protocol):
    """The part of Starlette's WebSocket the adapter uses, so tests can pass a fake.
    `receive_text` raises WebSocketDisconnect once the connection is gone."""

    async def receive_text(self) -> str: ...
    async def send_text(self, data: str) -> None: ...
    async def close(self, code: int = 1000) -> None: ...


StartSession = Callable[[str, SessionStart, ChannelAdapter], SessionCallbacks]


class TwilioAdapter:
    """Implements ChannelAdapter for one call. `run()` reads Twilio's messages until the
    stream ends; the route awaits it."""

    capabilities = frozenset({Capability.CALLER_ADDRESS, Capability.END_SESSION})

    def __init__(self, ws: Socket, start_session: StartSession) -> None:
        self._ws = ws
        self._start_session = start_session
        self._session: SessionCallbacks | None = None
        self._stream_sid: str | None = None
        self._call_sid: str | None = None
        self._encoder = Encoder()
        self._waiting: Counter[str] = Counter()  # marks sent, echo not back yet
        self._dropping: Counter[str] = Counter()  # marks whose echo is from a clear
        self._send_lock = asyncio.Lock()  # media and marks must reach Twilio in order
        self._ended_by_us = False  # end_session() closed the socket
        self._gone = False  # the socket failed while sending

    # -- reading ---------------------------------------------------------------------------

    async def run(self) -> None:
        try:
            while not self._ended_by_us:
                try:
                    text = await self._ws.receive_text()
                except WebSocketDisconnect:
                    break
                if not await self._handle(text):
                    break
        finally:
            self._gone = True
            if self._session is not None and not self._ended_by_us:
                await self._session.on_session_end(CALLER_HANGUP)

    async def _handle(self, text: str) -> bool:
        """Handle one message. Returns False when the stream is over."""
        try:
            message = json.loads(text)
            event = message["event"]
        except (ValueError, KeyError, TypeError):
            log.warning("call %s: ignoring a message that isn't Twilio JSON", self._call_sid)
            return True

        if event == "media":
            await self._on_media(message)
        elif event == "mark":
            await self._on_mark(message)
        elif event == "start":
            return await self._on_start(message)
        elif event == "stop":
            log.info("call %s: stream stopped", self._call_sid)
            return False
        elif event not in ("connected", "dtmf"):
            log.debug("call %s: ignoring %r message", self._call_sid, event)
        return True

    async def _on_start(self, message: dict[str, Any]) -> bool:
        if self._session is not None:
            log.warning("call %s: second start message ignored", self._call_sid)
            return True
        try:
            start = message["start"]
            self._stream_sid = str(start["streamSid"])
            self._call_sid = str(start["callSid"])
            params = start.get("customParameters") or {}
            token = str(params.get(PARAM_TOKEN, ""))
            address = str(params.get(PARAM_FROM, "")) or None
        except (KeyError, TypeError):
            log.warning("start message is missing fields; closing")
            await self.end_session()
            return False

        session_start = SessionStart(
            channel=CHANNEL,
            channel_ref=self._call_sid,
            caller_address=address,
            sample_rate=SAMPLE_RATE,
        )
        try:
            self._session = self._start_session(token, session_start, self)
        except AdmissionError as e:
            log.warning("call %s: stream refused: %s", self._call_sid, e)
            await self.end_session()
            return False
        log.info("call %s: session started", self._call_sid)
        return True

    async def _on_media(self, message: dict[str, Any]) -> None:
        if self._session is None:
            return  # audio before start: nobody to give it to
        try:
            pcm = decode(message["media"]["payload"])
        except (KeyError, TypeError, ValueError):
            log.warning("call %s: skipping a bad media frame", self._call_sid)
            return
        await self._session.on_audio(pcm)

    async def _on_mark(self, message: dict[str, Any]) -> None:
        try:
            name = str(message["mark"]["name"])
        except (KeyError, TypeError):
            return
        if self._dropping[name]:
            self._dropping[name] -= 1  # echoed because of a clear: never played
            return
        if self._waiting[name]:
            self._waiting[name] -= 1
        if self._session is not None:
            await self._session.on_mark_played(name)

    # -- ChannelAdapter (called by the core) ---------------------------------------------------

    async def send_audio(self, pcm: bytes) -> None:
        payload = self._encoder.encode(pcm)
        if payload:  # b"" (a TTS chunk with only word timings) sends nothing
            await self._send({"event": "media", "media": {"payload": payload}})

    async def mark(self, name: str) -> None:
        self._waiting[name] += 1
        await self._send({"event": "mark", "mark": {"name": name}})

    async def clear_audio(self) -> None:
        self._dropping += self._waiting
        self._waiting.clear()
        self._encoder.reset()
        await self._send({"event": "clear"})

    async def end_session(self) -> None:
        if self._ended_by_us:
            return
        self._ended_by_us = True
        try:
            await self._ws.close()
        except Exception:  # already gone: the call is over either way
            log.debug("call %s: socket was already closed", self._call_sid)

    async def send_followup(self, body: str, link: str | None) -> None:
        raise NotImplementedError("follow-up texts are v1")

    # -- helpers ----------------------------------------------------------------------------

    async def _send(self, message: dict[str, Any]) -> None:
        """Send to Twilio. Does nothing before the stream starts or after it ends: a reply
        still finishing when the caller hangs up has nowhere to go, and that's fine."""
        if self._ended_by_us or self._gone or self._stream_sid is None:
            return
        message["streamSid"] = self._stream_sid
        async with self._send_lock:
            try:
                await self._ws.send_text(json.dumps(message))
            except Exception:
                log.info("call %s: socket gone while sending", self._call_sid)
                self._gone = True
