"""The app builds and starts with fake settings, and makes no network calls doing so."""

import wave
from datetime import UTC, datetime
from pathlib import Path
from xml.etree.ElementTree import fromstring

import pytest
from fastapi.testclient import TestClient
from twilio.request_validator import RequestValidator

from tests.fakes import FakeNotifier
from voice_agent.app import main
from voice_agent.app.config import Settings
from voice_agent.core.interfaces import MessageRow, SessionStart
from voice_agent.notify.gmail import GmailNotifier
from voice_agent.storage.sqlite import SqliteStorage

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
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
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


def test_no_admin_pages_without_a_password(env, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "")  # as .env.example leaves it
    with TestClient(main.create_app()) as client:
        assert client.get("/admin", auth=("admin", "")).status_code == 404


def test_admin_pages_read_the_database(env, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "test admin password")
    with TestClient(main.create_app()) as client:
        assert client.get("/admin").status_code == 401
        page = client.get("/admin", auth=("admin", "test admin password"))
        assert page.status_code == 200 and "No calls yet." in page.text


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


# -- email --------------------------------------------------------------------------------------

GMAIL_ENV = {
    "GMAIL_ADDRESS": "bot@example.com",
    "GMAIL_APP_PASSWORD": "abcd efgh ijkl mnop",
    "OWNER_EMAIL": "owner@example.com",
}


def settings_with(monkeypatch, tmp_path, **env) -> Settings:
    monkeypatch.chdir(tmp_path)  # no .env here, so only what's set below
    for name in GMAIL_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return Settings()


def test_email_is_off_until_set_up(monkeypatch, tmp_path):
    assert main.gmail_notifier(settings_with(monkeypatch, tmp_path)) is None


def test_gmail_is_used_when_all_three_are_set(monkeypatch, tmp_path):
    notifier = main.gmail_notifier(settings_with(monkeypatch, tmp_path, **GMAIL_ENV))
    assert isinstance(notifier, GmailNotifier)


def test_email_half_set_up_stops_startup(monkeypatch, tmp_path):
    env = {k: v for k, v in GMAIL_ENV.items() if k != "OWNER_EMAIL"}
    with pytest.raises(RuntimeError, match="OWNER_EMAIL"):
        main.gmail_notifier(settings_with(monkeypatch, tmp_path, **env))


def test_app_starts_with_email_set_up(env, monkeypatch):
    for name, value in GMAIL_ENV.items():
        monkeypatch.setenv(name, value)
    with TestClient(main.create_app()) as client:  # nothing unsent, so no SMTP connection
        assert client.get("/healthz").text == "ok"


async def test_unsent_messages_are_sent_and_marked_and_a_failure_stays_unsent(tmp_path):
    store = await SqliteStorage.open(tmp_path / "agent.db")
    try:
        caller = await store.get_or_create_caller("twilio", "+15550100")
        start = SessionStart("twilio", "CA1", "+15550100", 8000)
        old = await store.create_session(caller.id, start)
        await store.save_message(old, MessageRow("call me back", datetime.now(UTC)))
        await store.set_caller_details(caller.id, "Sam", None, None)  # named after the message

        await main.send_unsent_messages(store, FakeNotifier(fail=True))
        assert len(await store.unsent_messages()) == 1  # still waiting

        notifier = FakeNotifier()
        await main.send_unsent_messages(store, notifier)
        [(msg, sent_caller, session)] = notifier.messages
        assert (msg.body, msg.name, sent_caller.name, session.id) == (
            "call me back",
            "Sam",
            "Sam",
            old,
        )
        assert await store.unsent_messages() == []
    finally:
        await store.close()
