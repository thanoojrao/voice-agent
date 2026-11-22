import json
from typing import Any
from xml.etree.ElementTree import fromstring

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from twilio.request_validator import RequestValidator

from voice_agent.adapters.twilio.routes import caller_number, twilio_router
from voice_agent.core.interfaces import ChannelAdapter, SessionStart
from voice_agent.core.registry import Refusal, Slot

TOKEN = "test-auth-token"
BASE = "https://agent.example.com"
VOICE_URL = BASE + "/twilio/voice"
STREAM_WSS = "wss://agent.example.com/twilio/stream"
SIGNER = RequestValidator(TOKEN)


class FakeSession:
    def __init__(self) -> None:
        self.ended: list[str] = []

    async def on_audio(self, pcm: bytes) -> None:
        pass

    async def on_mark_played(self, name: str) -> None:
        pass

    async def on_session_end(self, reason: str) -> None:
        self.ended.append(reason)


class FakeRegistry:
    def __init__(self, answer: Slot | Refusal | None = None) -> None:
        self.answer = answer or Slot(token="tok123", channel="twilio", address=None, expires_at=0)
        self.admitted: list[tuple[str, str | None]] = []
        self.started: list[tuple[str, SessionStart]] = []
        self.session = FakeSession()

    async def admit(self, channel: str, address: str | None) -> Slot | Refusal:
        self.admitted.append((channel, address))
        return self.answer

    def start_session(
        self, token: str, start: SessionStart, adapter: ChannelAdapter
    ) -> FakeSession:
        self.started.append((token, start))
        return self.session


def client_for(registry: FakeRegistry) -> TestClient:
    app = FastAPI()
    app.include_router(twilio_router(registry, TOKEN, BASE + "/"))
    return TestClient(app)


def call_form(**extra: str) -> dict[str, str]:
    return {"CallSid": "CA123", "From": "+15550100", "To": "+15550199", **extra}


def post_voice(client: TestClient, form: dict[str, str], signature: str | None = None) -> Any:
    sig = SIGNER.compute_signature(VOICE_URL, form) if signature is None else signature
    return client.post("/twilio/voice", data=form, headers={"X-Twilio-Signature": sig})


# -- caller number ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("+15550100", "+15550100"),
        (" +442071838750 ", "+442071838750"),
        ("anonymous", None),
        ("unknown", None),
        ("Restricted", None),
        ("+266696687", "+266696687"),  # looks like a number, so it's kept; caps still apply
        ("client:alice", None),
        ("", None),
        (None, None),
    ],
)
def test_caller_number(raw, want):
    assert caller_number(raw) == want


# -- webhook ----------------------------------------------------------------------------------


def test_admitted_call_gets_the_stream_twiml():
    registry = FakeRegistry()
    response = post_voice(client_for(registry), call_form())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert registry.admitted == [("twilio", "+15550100")]
    root = fromstring(response.text)
    stream_el = root.find("Connect/Stream")
    assert stream_el is not None
    assert stream_el.get("url") == STREAM_WSS
    params = {p.get("name"): p.get("value") for p in stream_el.findall("Parameter")}
    assert params == {"from": "+15550100", "token": "tok123"}


def test_withheld_number_is_admitted_as_anonymous():
    registry = FakeRegistry()
    response = post_voice(client_for(registry), call_form(From="anonymous"))
    assert registry.admitted == [("twilio", None)]
    param = fromstring(response.text).find("Connect/Stream/Parameter[@name='from']")
    assert param is not None
    assert param.get("value") == ""


def test_bad_signature_is_refused():
    registry = FakeRegistry()
    response = post_voice(client_for(registry), call_form(), signature="bogus")
    assert response.status_code == 403
    assert registry.admitted == []


def test_missing_signature_is_refused():
    registry = FakeRegistry()
    response = client_for(registry).post("/twilio/voice", data=call_form())
    assert response.status_code == 403


def test_signature_covers_the_form_fields():
    registry = FakeRegistry()
    signed = SIGNER.compute_signature(VOICE_URL, call_form())
    tampered = call_form(From="+15550123")
    response = post_voice(client_for(registry), tampered, signature=signed)
    assert response.status_code == 403


def test_blocked_number_is_rejected():
    response = post_voice(client_for(FakeRegistry(Refusal.BLOCKED)), call_form())
    assert [child.tag for child in fromstring(response.text)] == ["Reject"]


@pytest.mark.parametrize(
    ("refusal", "clip"),
    [
        (Refusal.CONCURRENT, "busy.wav"),
        (Refusal.CALLER_CAP, "limit_caller.wav"),
        (Refusal.DAILY_CAP, "limit_daily.wav"),
    ],
)
def test_refused_call_hears_why(refusal, clip):
    response = post_voice(client_for(FakeRegistry(refusal)), call_form())
    root = fromstring(response.text)
    assert [child.tag for child in root] == ["Play", "Hangup"]
    assert root[0].text == f"{BASE}/static/audio/{clip}"


def test_every_refusal_except_blocked_has_a_clip_the_script_renders():
    from scripts.generate_clips import clip_lines
    from voice_agent.adapters.twilio.routes import REFUSAL_CLIP_PATHS

    assert set(REFUSAL_CLIP_PATHS) == set(Refusal) - {Refusal.BLOCKED}
    rendered = {f"/static/audio/{name}.wav" for name in clip_lines("Alex")}
    assert set(REFUSAL_CLIP_PATHS.values()) <= rendered


def test_public_base_url_must_be_https():
    with pytest.raises(ValueError, match="https"):
        twilio_router(FakeRegistry(), TOKEN, "http://agent.example.com")


# -- fallback ---------------------------------------------------------------------------------


def test_fallback_plays_the_apology_without_a_signature():
    response = client_for(FakeRegistry()).post("/twilio/fallback", data=call_form())
    root = fromstring(response.text)
    assert [child.tag for child in root] == ["Play", "Hangup"]
    assert root[0].text == BASE + "/static/audio/apology.wav"


# -- stream -----------------------------------------------------------------------------------


def start_message() -> str:
    return json.dumps(
        {
            "event": "start",
            "start": {
                "streamSid": "MZ123",
                "callSid": "CA123",
                "customParameters": {"from": "+15550100", "token": "tok123"},
            },
        }
    )


def run_stream(client: TestClient, signature: str) -> None:
    with client.websocket_connect(
        "/twilio/stream", headers={"X-Twilio-Signature": signature}
    ) as ws:
        ws.send_text(start_message())
        ws.send_text(json.dumps({"event": "stop", "stop": {}}))


def test_stream_signed_over_the_wss_url_is_accepted():
    # The form a real call showed Twilio signs: the TwiML's URL, no trailing slash.
    registry = FakeRegistry()
    run_stream(client_for(registry), SIGNER.compute_signature(STREAM_WSS, {}))
    assert [token for token, _ in registry.started] == ["tok123"]
    assert registry.session.ended == ["caller_hangup"]


@pytest.mark.parametrize(
    "signature",
    [
        "bogus",
        "",
        SIGNER.compute_signature(STREAM_WSS + "/", {}),  # any other URL form doesn't count
        SIGNER.compute_signature(BASE + "/twilio/stream", {}),
        RequestValidator("someone-elses-token").compute_signature(STREAM_WSS, {}),
    ],
)
def test_stream_with_a_bad_signature_is_refused_before_the_token_is_used(signature):
    registry = FakeRegistry()
    with pytest.raises(WebSocketDisconnect):
        run_stream(client_for(registry), signature)
    assert registry.started == []
