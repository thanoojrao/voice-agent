"""The admin pages' labels, times and counts, from made-up rows (no HTML)."""

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from voice_agent.admin.views import (
    Bubble,
    day_label,
    end_label,
    format_number,
    how_to_reach,
    is_problem,
    length_label,
    message_status,
    person,
    reply_stats,
    time_label,
    tool_label,
    transcript,
    when_label,
)
from voice_agent.core.interfaces import Caller, MessageRow, SessionRow, TurnRow

NY = ZoneInfo("America/New_York")
# 6:17 PM in New York on Tue 24 Nov 2026 (UTC-5)
T = datetime(2026, 11, 24, 23, 17, tzinfo=UTC)
TODAY = date(2026, 11, 24)


def caller(name: str | None = None, address: str | None = "+15550100142") -> Caller:
    return Caller(1, "twilio", address, name, None, None, T, T, 1)


def session(seconds=190, reason="goodbye", ended=True) -> SessionRow:
    end = T + timedelta(seconds=seconds) if ended else None
    return SessionRow(1, 1, "twilio", "CA1", T, end, reason if ended else None, None)


# -- people ---------------------------------------------------------------------------


def test_named_caller():
    p = person(caller("  sam   rivera "))
    assert (p.title, p.initials, p.number) == ("sam rivera", "SR", "+1 555 010 0142")


def test_unnamed_caller_shows_the_number_and_no_initials():
    p = person(caller())
    assert (p.title, p.initials, p.tint) == ("+1 555 010 0142", "", 0)


def test_withheld_number():
    p = person(caller(address=None))
    assert (p.title, p.number) == ("Number withheld", None)


def test_a_name_always_gets_the_same_colour():
    assert person(caller("Sam")).tint == person(caller("sam")).tint
    assert 0 <= person(caller("Priya Nair")).tint < 6


@pytest.mark.parametrize(
    ("address", "shown"),
    [("+15550100142", "+1 555 010 0142"), ("+447700900123", "+447700900123"), ("+1555", "+1555")],
)
def test_number_formatting(address, shown):
    assert format_number(address) == shown


# -- times ----------------------------------------------------------------------------


def test_days_are_in_the_owners_time_zone():
    late = datetime(2026, 11, 25, 3, 0, tzinfo=UTC)  # 10 PM on the 24th in New York
    assert day_label(late, TODAY, NY) == "Today"
    assert day_label(T - timedelta(days=1), TODAY, NY) == "Yesterday"
    assert day_label(T - timedelta(days=2), TODAY, NY) == "Sun 22 Nov"
    assert day_label(T - timedelta(days=365), TODAY, NY) == "Mon 24 Nov 2025"


def test_time_labels():
    assert time_label(T, NY) == "6:17 PM"
    assert time_label(datetime(2026, 11, 24, 5, 5, tzinfo=UTC), NY) == "12:05 AM"
    assert when_label(T - timedelta(days=2), TODAY, NY) == "Sun 22 Nov, 6:17 PM"


@pytest.mark.parametrize(("seconds", "shown"), [(40, "40s"), (190, "3m 10s"), (600, "10m 00s")])
def test_call_length(seconds, shown):
    assert length_label(session(seconds)) == shown


def test_a_call_still_going():
    s = session(ended=False)
    assert (length_label(s), end_label(s)) == ("In progress", "In progress")


def test_end_labels_and_problems():
    assert end_label(session(reason="silence")) == "Went quiet"
    assert end_label(session(reason="something_new")) == "something_new"
    assert is_problem(session(reason="error")) and not is_problem(session(reason="goodbye"))


# -- messages -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("contact", "address", "shown"),
    [
        (None, "+15550100142", "This number"),
        ("sam@example.com", "+15550100142", "sam@example.com"),
        ("declined", "+15550100142", "They didn't say"),
        (None, None, "Not given"),
    ],
)
def test_how_to_reach(contact, address, shown):
    msg = MessageRow("Call me.", T, contact=contact)
    assert how_to_reach(msg, caller(address=address)) == shown


def test_message_status():
    assert message_status(MessageRow("x", T), NY, TODAY) is None  # not emailed yet
    sent = MessageRow("x", T, notified_at=T + timedelta(minutes=3))
    assert message_status(sent, NY, TODAY) == "Emailed 6:20 PM"
    assert message_status(sent, NY, TODAY + timedelta(days=1)) == "Emailed Yesterday, 6:20 PM"


# -- transcript -----------------------------------------------------------------------


def turn(seq, speaker, text="...", latency=None, cut=False) -> TurnRow:
    return TurnRow(seq, speaker, text, T, interrupted=cut, latency_ms=latency)


CALL = [
    turn(1, "agent", "Hi, this is the AI.", 600),  # greeting: nothing to answer
    turn(2, "agent", "Are you still there?", 150),  # silence prompt: same
    turn(3, "caller", "Yes. Pass a message on?"),
    turn(4, "agent", "Sure, what should it say?", 1200),
    turn(5, "caller", "Call me back."),
    turn(6, "tool", 'save_message {"body": "Call me back.", "contact": "this number"}'),
    turn(7, "agent", "Done. Anything else?", 2600),
    turn(8, "caller", "Also"),
    turn(9, "agent", "Sure, go on", 1000, cut=True),
]


def test_transcript_bubbles_time_only_real_replies():
    bubbles = transcript(CALL)
    assert bubbles[0] == Bubble("me", "Hi, this is the AI.")
    assert bubbles[1].reply_s is None
    assert bubbles[3] == Bubble("me", "Sure, what should it say?", 1.2)
    assert bubbles[5] == Bubble("tool", "Saved message")
    assert bubbles[6].reply_s == 2.6  # the tool call between doesn't hide the reply
    assert bubbles[8] == Bubble("me", "Sure, go on", 1.0, cut_off=True)


def test_reply_stats_skip_the_greeting_and_prompts():
    stats = reply_stats(CALL)
    assert stats is not None
    assert (round(stats.average_s, 2), stats.slowest_s, stats.cut_offs) == (1.6, 2.6, 1)


def test_no_replies_no_stats():
    assert reply_stats(CALL[:2]) is None


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ('save_caller_details {"name": "Sam", "reason": "a role"}', "Saved name Sam"),
        ('save_caller_details {"name": "Sam", "company": "Acme"}', "Saved name Sam, company Acme"),
        ('save_caller_details {"reason": "a role"}', "Saved details"),
        ("save_caller_details {not json", "Saved details"),
        ('end_session {"reason": "goodbye"}', "Ended the call"),
        ("lookup_thing {}", "lookup_thing"),
    ],
)
def test_tool_labels(text, label):
    assert tool_label(text) == label
