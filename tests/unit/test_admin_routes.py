"""The admin pages' sign-in, headers and missing items. What the pages show is tested
once the templates exist; the labels themselves are in test_admin_views.py."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from voice_agent.admin.routes import admin_router
from voice_agent.core.interfaces import Caller, MessageRow, SessionRow, TurnRow
from voice_agent.storage.sqlite import SessionListing

PASSWORD = "correct horse battery staple"
NOW = datetime(2026, 11, 24, 23, 17, tzinfo=UTC)


class EmptyReads:
    """No calls, messages or callers yet."""

    async def recent_sessions(self, limit=100, caller_id=None):
        return []

    async def get_session(self, session_id):
        return None

    async def session_turns(self, session_id):
        return []

    async def session_message(self, session_id):
        return None

    async def recent_messages(self, limit=100):
        return []

    async def recent_callers(self, limit=100):
        return []

    async def get_caller(self, caller_id):
        return None

    async def unsent_messages(self):
        return []


@pytest.fixture
def client():
    app = FastAPI()
    reads = EmptyReads()
    app.include_router(
        admin_router(lambda: reads, SecretStr(PASSWORD), ZoneInfo("America/New_York"), lambda: NOW)
    )
    return TestClient(app)


PAGES = ["/admin", "/admin/calls/1", "/admin/messages", "/admin/callers/1", "/admin/admin.css"]


@pytest.mark.parametrize("path", PAGES)
def test_every_page_needs_the_password(client, path):
    r = client.get(path)
    assert r.status_code == 401
    assert r.headers["WWW-Authenticate"].startswith("Basic")
    assert r.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("auth", [("admin", "wrong"), ("admin", ""), ("admin", PASSWORD + "x")])
def test_a_wrong_password_is_refused(client, auth):
    assert client.get("/admin/calls/1", auth=auth).status_code == 401


def test_any_username_with_the_right_password_gets_in(client):
    # The call doesn't exist, so it's a 404: past the sign-in, before any template.
    r = client.get("/admin/calls/1", auth=("anything", PASSWORD))
    assert r.status_code == 404
    assert r.headers["X-Robots-Tag"] == "noindex, nofollow"
    assert "script-src" not in r.headers["Content-Security-Policy"]  # default-src 'none'


def test_a_missing_caller_is_not_found(client):
    assert client.get("/admin/callers/7", auth=("admin", PASSWORD)).status_code == 404


def test_an_empty_password_refuses_to_start():
    with pytest.raises(ValueError, match="empty"):
        admin_router(EmptyReads, SecretStr(""), UTC, lambda: NOW)


# -- the pages, rendered --------------------------------------------------------------


def _rows():
    t = NOW - timedelta(minutes=5)
    sam = Caller(1, "twilio", "+15550100142", "Sam Rivera", "Acme", "The backend role", t, t, 2)
    hidden = Caller(2, "twilio", None, None, None, None, t, t, 1)
    call = SessionRow(
        41, 1, "twilio", "CA1", t, t + timedelta(seconds=190), "goodbye", "Asked about a role."
    )
    failed = SessionRow(
        40, 2, "twilio", "CA2", t - timedelta(days=1), t - timedelta(days=1), "error", None
    )
    msg = MessageRow("Call me <b>back</b> on Friday.", t, "Sam", None, 9, None)
    turns = [
        TurnRow(1, "agent", "Hi Sam, this is the AI.", t, latency_ms=600),
        TurnRow(2, "caller", "Can you pass a message on?", t),
        TurnRow(3, "tool", 'save_message {"body": "x", "contact": "this number"}', t),
        TurnRow(4, "agent", "Done. Anything else?", t, latency_ms=1400, interrupted=True),
    ]
    return sam, hidden, call, failed, msg, turns, SessionListing


class DataReads:
    """A named caller with a call and a message, and a withheld caller whose call failed."""

    def __init__(self):
        sam, hidden, call, failed, msg, turns, listing = _rows()
        self.sam, self.hidden, self.call, self.failed, self.msg, self.turns = (
            sam, hidden, call, failed, msg, turns,
        )  # fmt: skip
        self.listing = [listing(call, sam, 1, True), listing(failed, hidden, 0, False)]

    async def recent_sessions(self, limit=100, caller_id=None):
        return [x for x in self.listing if caller_id in (None, x.caller.id)]

    async def get_session(self, session_id):
        return {41: (self.call, self.sam), 40: (self.failed, self.hidden)}.get(session_id)

    async def session_turns(self, session_id):
        return self.turns if session_id == 41 else []

    async def session_message(self, session_id):
        return self.msg if session_id == 41 else None

    async def recent_messages(self, limit=100):
        return [(self.msg, self.sam, self.call)]

    async def recent_callers(self, limit=100):
        return [self.sam, self.hidden]

    async def get_caller(self, caller_id):
        return {1: self.sam, 2: self.hidden}.get(caller_id)

    async def unsent_messages(self):
        return [(self.msg, self.sam, self.call)]


@pytest.fixture
def signed_in():
    app = FastAPI()
    reads = DataReads()
    app.include_router(
        admin_router(lambda: reads, SecretStr(PASSWORD), ZoneInfo("America/New_York"), lambda: NOW)
    )
    client = TestClient(app)
    client.auth = ("admin", PASSWORD)
    return client


def test_calls_page(signed_in):
    r = signed_in.get("/admin")
    assert r.status_code == 200 and r.headers["Cache-Control"] == "no-store"
    html = r.text
    assert "Today" in html and "Yesterday" in html
    assert "Sam Rivera" in html and "6:12 PM" in html and "3m 10s" in html
    assert "Left a message" in html
    assert 'class="row-title danger">Number withheld' in html  # the failed call, in red
    assert "1 today · 1 message" in html
    assert 'class="count"' in html  # one message not emailed yet
    assert "has-detail" not in html


def test_a_call_with_its_transcript(signed_in):
    html = signed_in.get("/admin/calls/41").text
    assert 'class="app has-detail"' in html and 'aria-current="true"' in html
    assert "Asked about a role." in html
    assert "Call me &lt;b&gt;back&lt;/b&gt; on Friday." in html  # caller words are escaped
    assert "Reach them: This number" in html and "Not emailed yet" in html
    assert "Saved message" in html
    assert "replied in 1.4 s" in html and "Cut off" in html
    assert "replied in 0.6 s" not in html  # the greeting answers nothing
    assert "avg 1.4 s · slowest 1.4 s" in html and ">Once<" in html
    assert 'href="tel:+15550100142"' in html
    assert html.index("Summary") < html.index("Message</div>")


def test_messages_page_keeps_the_messages_tab(signed_in):
    html = signed_in.get("/admin/messages/41").text
    assert 'href="/admin/messages" aria-current="page"' in html
    assert 'href="/admin/messages">' in html  # the back link on a phone
    assert "1 message · 1 not emailed yet" in html


def test_a_caller_with_their_calls(signed_in):
    html = signed_in.get("/admin/callers/1").text
    assert "Acme" in html and "The backend role" in html
    assert 'href="/admin/calls/41"' in html
    assert 'href="/admin/calls/40"' not in html  # someone else's call


def test_a_withheld_caller(signed_in):
    html = signed_in.get("/admin/callers/2").text
    assert "Withheld, so each call is a new caller" in html
    assert "tel:" not in html


def test_empty_lists(client):
    auth = ("admin", PASSWORD)
    assert "No calls yet." in client.get("/admin", auth=auth).text
    assert "No messages yet." in client.get("/admin/messages", auth=auth).text


def test_the_stylesheet(signed_in):
    r = signed_in.get("/admin/admin.css")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/css")
