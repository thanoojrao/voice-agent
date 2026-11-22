"""The HTTP and WebSocket endpoints Twilio calls (design §7).

    POST /twilio/voice      incoming call: check the signature, admit, reply with TwiML
    WS   /twilio/stream     the call's audio, handed to a TwilioAdapter
    POST /twilio/fallback   Twilio's fallback when /voice fails: apology clip, hang up

Signatures. Twilio signs each request with the account auth token: HMAC-SHA1 of the exact URL
it called plus the sorted POST fields. Behind ngrok or a host's proxy our app sees a local
URL, so the check is made against PUBLIC_BASE_URL + the path, the address Twilio used.

The stream has two locks. Its handshake is signed too, over the stream URL exactly as the
TwiML gave it (wss://, no trailing slash; confirmed on a real call), and a bad signature is
refused. And the stream must present the registry's one-time token (single use, 60 s, only
handed out in a signed webhook reply).

The fallback isn't signature-checked on purpose: it must work when everything else is
broken, and all it returns is a public apology clip.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

from fastapi import APIRouter, Request, Response, WebSocket
from twilio.request_validator import RequestValidator

from voice_agent.adapters.twilio import twiml
from voice_agent.adapters.twilio.adapter import CHANNEL, PARAM_FROM, PARAM_TOKEN, TwilioAdapter
from voice_agent.core.interfaces import ChannelAdapter, SessionCallbacks, SessionStart
from voice_agent.core.registry import Refusal, Slot

log = logging.getLogger(__name__)

VOICE_PATH = "/twilio/voice"
STREAM_PATH = "/twilio/stream"
FALLBACK_PATH = "/twilio/fallback"
BUSY_CLIP_PATH = "/static/audio/busy.wav"
APOLOGY_CLIP_PATH = "/static/audio/apology.wav"
# Each refusal says why, so a capped caller doesn't think the line is broken. Anything not
# listed (a refusal added later) gets the busy clip.
REFUSAL_CLIP_PATHS = {
    Refusal.CONCURRENT: BUSY_CLIP_PATH,
    Refusal.CALLER_CAP: "/static/audio/limit_caller.wav",
    Refusal.DAILY_CAP: "/static/audio/limit_daily.wav",
}

_SIGNATURE_HEADER = "x-twilio-signature"
_POLICY_VIOLATION = 1008  # WebSocket close code for a refused handshake
# Twilio sends a withheld number as "anonymous", "unknown" or "other descriptions", so
# anything that isn't a real E.164 number is treated as no number at all.
_E164 = re.compile(r"\+[1-9]\d{6,14}")


class Admission(Protocol):
    """The two registry steps the routes need (core/registry.py)."""

    async def admit(self, channel: str, address: str | None) -> Slot | Refusal: ...
    def start_session(
        self, token: str, start: SessionStart, adapter: ChannelAdapter
    ) -> SessionCallbacks: ...


def caller_number(raw: str | None) -> str | None:
    """Twilio's `From` → an E.164 number, or None when withheld or not a phone number."""
    if raw is None:
        return None
    raw = raw.strip()
    return raw if _E164.fullmatch(raw) else None


def twilio_router(
    registry: Admission,
    auth_token: str,
    public_base_url: str,
) -> APIRouter:
    """The Twilio endpoints, bound to this app's registry and settings (built in main.py)."""
    base = public_base_url.rstrip("/")
    if not base.startswith("https://"):
        raise ValueError("PUBLIC_BASE_URL must be https:// (Twilio needs TLS for the stream)")
    stream_url = "wss://" + base.removeprefix("https://") + STREAM_PATH
    validator = RequestValidator(auth_token)
    router = APIRouter()

    @router.post(VOICE_PATH)
    async def voice(request: Request) -> Response:
        form = await request.form()
        url = base + request.url.path + (f"?{request.url.query}" if request.url.query else "")
        signature = request.headers.get(_SIGNATURE_HEADER, "")
        if not validator.validate(url, form, signature):
            log.warning("webhook with a bad or missing signature refused")
            return Response(status_code=403)

        call_sid = str(form.get("CallSid", ""))
        address = caller_number(_text(form.get("From")))
        admitted = await registry.admit(CHANNEL, address)
        if admitted is Refusal.BLOCKED:
            log.info("call %s: blocked number rejected", call_sid)
            return _twiml(twiml.reject())
        if isinstance(admitted, Refusal):
            log.info("call %s: refused (%s)", call_sid, admitted)
            clip = REFUSAL_CLIP_PATHS.get(admitted, BUSY_CLIP_PATH)
            return _twiml(twiml.play_then_hangup(base + clip))
        params = {PARAM_FROM: address or "", PARAM_TOKEN: admitted.token}
        return _twiml(twiml.stream(stream_url, params))

    @router.websocket(STREAM_PATH)
    async def stream(ws: WebSocket) -> None:
        signature = ws.headers.get(_SIGNATURE_HEADER, "")
        if not validator.validate(stream_url, {}, signature):
            log.warning("stream handshake with a bad or missing signature refused")
            await ws.close(code=_POLICY_VIOLATION)
            return
        await ws.accept()
        await TwilioAdapter(ws, registry.start_session).run()

    @router.post(FALLBACK_PATH)
    async def fallback() -> Response:
        log.error("Twilio used the fallback URL: the voice webhook failed")
        return _twiml(twiml.play_then_hangup(base + APOLOGY_CLIP_PATH))

    return router


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _twiml(xml: str) -> Response:
    return Response(content=xml, media_type="application/xml")
