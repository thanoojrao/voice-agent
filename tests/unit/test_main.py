"""The app builds and starts with fake settings, and makes no network calls doing so."""

import wave
from pathlib import Path
from xml.etree.ElementTree import fromstring

import pytest
from fastapi.testclient import TestClient
from twilio.request_validator import RequestValidator

from voice_agent.app import main

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://agent.example.com"
AUTH = "test-auth-token"
CONFIG = """\
owner_name: Alex
greeting: "Hi, this is {owner_name}'s AI."
greeting_returning: "Hi {caller_name}, this is {owner_name}'s AI."
name_question: "Who am I speaking with?"
voice_id: "test-voice"
db_path: "%s"
"""


def write_clip(path: Path) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 800)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A working directory with no .env, a config, clips, and the example KB."""
    monkeypatch.chdir(tmp_path)
    clips = tmp_path / "clips"
    clips.mkdir()
    for name in ("apology.wav", "filler_1.wav", "busy.wav"):
        write_clip(clips / name)
    config = tmp_path / "agent.config.yaml"
    config.write_text(CONFIG % (tmp_path / "db" / "agent.db").as_posix(), encoding="utf-8")
    monkeypatch.setattr(main, "CLIPS_DIR", clips)
    monkeypatch.setattr(main, "KB_DIR", ROOT / "kb.example")
    for name, value in {
        "ANTHROPIC_API_KEY": "test",
        "DEEPGRAM_API_KEY": "test",
        "CARTESIA_API_KEY": "test",
        "TWILIO_AUTH_TOKEN": AUTH,
        "PUBLIC_BASE_URL": BASE,
        "CONFIG_PATH": str(config),
    }.items():
        monkeypatch.setenv(name, value)
    return tmp_path


def test_app_starts_and_answers(env):
    with TestClient(main.create_app()) as client:
        assert client.get("/healthz").text == "ok"
        assert client.get("/static/audio/busy.wav").status_code == 200

        form = {"CallSid": "CA123", "From": "+15550100", "To": "+15550199"}
        signature = RequestValidator(AUTH).compute_signature(BASE + "/twilio/voice", form)
        reply = client.post("/twilio/voice", data=form, headers={"X-Twilio-Signature": signature})
        stream_el = fromstring(reply.text).find("Connect/Stream")
        assert stream_el is not None
        assert stream_el.get("url") == "wss://agent.example.com/twilio/stream"
    assert (env / "db" / "agent.db").exists()  # created on startup


def test_missing_key_stops_startup(env, monkeypatch):
    monkeypatch.delenv("TWILIO_AUTH_TOKEN")
    with pytest.raises(RuntimeError, match="TWILIO_AUTH_TOKEN"):
        main.create_app()


def test_missing_clips_stop_startup(env, monkeypatch):
    monkeypatch.setattr(main, "CLIPS_DIR", env / "nowhere")
    with pytest.raises(RuntimeError, match="clip folder not found"):
        main.create_app()


def test_owner_clock_is_new_york_time():
    assert main.owner_clock().tzinfo == main.OWNER_TZ
    assert str(main.OWNER_TZ) == "America/New_York"
