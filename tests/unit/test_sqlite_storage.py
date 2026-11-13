"""SqliteStorage against a real database file in a temp folder, with a controllable clock."""

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from voice_agent.core.interfaces import MessageRow, SessionStart, TurnRow
from voice_agent.storage.sqlite import SqliteStorage

T0 = datetime(2026, 11, 13, 15, 0, tzinfo=UTC)
NUMBER = "+15551234567"


class Clock:
    def __init__(self, at: datetime):
        self.at = at

    def __call__(self) -> datetime:
        return self.at

    def advance(self, **kw) -> None:
        self.at += timedelta(**kw)


@pytest.fixture
def clock():
    return Clock(T0)


@pytest.fixture
async def store(tmp_path, clock):
    s = await SqliteStorage.open(tmp_path / "agent.db", now=clock)
    yield s
    await s.close()


def start(ref="CA1", number=NUMBER):
    return SessionStart(channel="twilio", channel_ref=ref, caller_address=number, sample_rate=8000)


async def new_session(store, number=NUMBER, ref="CA1"):
    caller = await store.get_or_create_caller("twilio", number)
    return caller, await store.create_session(caller.id, start(ref, number))


# -- callers ---------------------------------------------------------------


async def test_new_caller_starts_empty(store):
    c = await store.get_or_create_caller("twilio", NUMBER)
    assert c.address == NUMBER
    assert c.session_count == 0
    assert c.name is None
    assert c.first_seen == T0


async def test_returning_caller_gets_same_row_with_count(store, clock):
    first, _ = await new_session(store)
    clock.advance(days=2)
    again = await store.get_or_create_caller("twilio", NUMBER)
    assert again.id == first.id
    assert again.session_count == 1
    assert again.first_seen == T0
    assert again.last_seen == T0  # last_seen moves when a session is created, not on lookup


async def test_same_number_on_another_channel_is_another_caller(store):
    a = await store.get_or_create_caller("twilio", NUMBER)
    b = await store.get_or_create_caller("web", NUMBER)
    assert a.id != b.id


async def test_withheld_numbers_are_never_merged(store):
    a = await store.get_or_create_caller("twilio", None)
    b = await store.get_or_create_caller("twilio", None)
    assert a.id != b.id
    assert a.address is None and b.address is None


async def test_caller_details_keep_values_not_given(store):
    c = await store.get_or_create_caller("twilio", NUMBER)
    await store.set_caller_details(c.id, "Ann", None, "hiring")
    await store.set_caller_details(c.id, None, "Acme", None)
    c = await store.get_or_create_caller("twilio", NUMBER)
    assert (c.name, c.company, c.reason) == ("Ann", "Acme", "hiring")


# -- sessions --------------------------------------------------------------


async def test_create_session_bumps_count_and_last_seen(store, clock):
    caller, _ = await new_session(store)
    clock.advance(hours=3)
    await store.create_session(caller.id, start("CA2"))
    c = await store.get_or_create_caller("twilio", NUMBER)
    assert c.session_count == 2
    assert c.last_seen == T0 + timedelta(hours=3)


async def test_last_session_is_most_recent_with_summary(store, clock):
    caller, s1 = await new_session(store)
    await store.end_session(s1, "caller_hangup", "asked about the voice agent")
    clock.advance(days=1)
    s2 = await store.create_session(caller.id, start("CA2"))
    clock.advance(minutes=5)
    await store.end_session(s2, "agent_end", "left a message")

    last = await store.last_session(caller.id)
    assert last is not None
    assert (last.id, last.channel_ref, last.summary) == (s2, "CA2", "left a message")
    assert last.started_at == T0 + timedelta(days=1)
    assert last.ended_at == T0 + timedelta(days=1, minutes=5)
    assert last.end_reason == "agent_end"


async def test_last_session_none_for_first_time_caller(store):
    c = await store.get_or_create_caller("twilio", NUMBER)
    assert await store.last_session(c.id) is None


async def test_count_sessions_since(store, clock):
    ann, _ = await new_session(store)  # T0
    clock.advance(hours=1)
    await store.create_session(ann.id, start("CA2"))  # T0+1h
    clock.advance(hours=1)
    await new_session(store, "+15559876543", "CA3")  # T0+2h, another caller

    assert await store.count_sessions_since(T0) == 3
    assert await store.count_sessions_since(T0 + timedelta(minutes=30)) == 2
    assert await store.count_sessions_since(T0, caller_id=ann.id) == 2
    assert await store.count_sessions_since(T0 + timedelta(hours=1), caller_id=ann.id) == 1


# -- turns and messages ----------------------------------------------------


async def test_turns_saved_in_order(store, tmp_path):
    _, sid = await new_session(store)
    await store.add_turn(sid, TurnRow(1, "caller", "what are you working on?", T0))
    await store.add_turn(
        sid,
        TurnRow(2, "agent", "Right now I'm building", T0, T0, interrupted=True, latency_ms=640),
    )
    await store.add_turn(sid, TurnRow(3, "tool", 'end_session {"reason":"done"}', T0))

    rows = sqlite3.connect(tmp_path / "agent.db").execute(
        "SELECT seq, speaker, interrupted, latency_ms FROM turns WHERE session_id=? ORDER BY seq",
        (sid,),
    )
    assert rows.fetchall() == [(1, "caller", 0, None), (2, "agent", 1, 640), (3, "tool", 0, None)]


async def test_duplicate_turn_number_rejected(store):
    _, sid = await new_session(store)
    await store.add_turn(sid, TurnRow(1, "caller", "hi", T0))
    with pytest.raises(sqlite3.IntegrityError):
        await store.add_turn(sid, TurnRow(1, "agent", "hello", T0))


async def test_turn_for_missing_session_rejected(store):
    with pytest.raises(sqlite3.IntegrityError):
        await store.add_turn(999, TurnRow(1, "caller", "hi", T0))


async def test_naive_datetime_rejected(store):
    _, sid = await new_session(store)
    with pytest.raises(ValueError):
        await store.add_turn(sid, TurnRow(1, "caller", "hi", datetime(2026, 11, 13, 15, 0)))


async def test_message_saved_then_marked_notified(store, clock, tmp_path):
    _, sid = await new_session(store)
    mid = await store.save_message(sid, MessageRow("call me back", T0, "Ann", "ann@example.com"))
    db = sqlite3.connect(tmp_path / "agent.db")
    assert db.execute("SELECT notified_at FROM messages WHERE id=?", (mid,)).fetchone() == (None,)

    clock.advance(seconds=4)
    await store.mark_message_notified(mid)
    (notified,) = db.execute("SELECT notified_at FROM messages WHERE id=?", (mid,)).fetchone()
    assert datetime.fromisoformat(notified) == T0 + timedelta(seconds=4)


# -- persistence -----------------------------------------------------------


async def test_data_survives_reopen(tmp_path, clock):
    path = tmp_path / "agent.db"
    s = await SqliteStorage.open(path, now=clock)
    caller, sid = await new_session(s)
    await s.end_session(sid, "caller_hangup", "talked about projects")
    await s.close()

    s = await SqliteStorage.open(path, now=clock)  # migrations already applied: no-op
    try:
        again = await s.get_or_create_caller("twilio", NUMBER)
        assert again.id == caller.id and again.session_count == 1
        last = await s.last_session(caller.id)
        assert last is not None and last.summary == "talked about projects"
    finally:
        await s.close()
