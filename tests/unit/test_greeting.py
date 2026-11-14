from datetime import UTC, datetime

import pytest

from voice_agent.core.greeting import GreetingTemplates, build_opening, usable_name
from voice_agent.core.interfaces import Caller

T = GreetingTemplates(
    owner_name="Alex",
    greeting="Hi, this is {owner_name}'s AI.",
    greeting_returning="Hi {caller_name}, this is {owner_name}'s AI.",
    name_question="Who am I speaking with?",
)
NOW = datetime(2026, 11, 14, tzinfo=UTC)


def caller(name: str | None = None, sessions: int = 0, address: str | None = "+15551234567"):
    return Caller(1, "twilio", address, name, None, None, NOW, NOW, sessions)


def opening(c, summary=None, warm_intro=True):
    return build_opening(c, T, last_summary=summary, warm_intro=warm_intro)


def test_first_session_gets_greeting_and_name_question():
    o = opening(caller())
    assert o.text == "Hi, this is Alex's AI. Who am I speaking with?"
    assert o.warm_intro_summary is None


def test_returning_with_name_is_greeted_by_name():
    o = opening(caller("Sam", sessions=2))
    assert o.text == "Hi Sam, this is Alex's AI."


def test_returning_without_name_is_not_asked_again():
    assert opening(caller(None, sessions=2)).text == "Hi, this is Alex's AI."


def test_returning_with_mangled_name_gets_plain_greeting():
    o = opening(caller("um it's sam from acme corp", sessions=2))
    assert o.text == "Hi, this is Alex's AI."


def test_anonymous_is_asked_every_time():
    o = opening(caller("Sam", sessions=5, address=None), summary="the voice agent")
    assert o.text == "Hi, this is Alex's AI. Who am I speaking with?"
    assert o.warm_intro_summary is None


def test_warm_intro_for_returning_caller_with_summary():
    o = opening(caller("Sam", sessions=1), summary="Talked about the voice agent.")
    assert o.warm_intro_summary == "Talked about the voice agent."


def test_no_warm_intro_without_summary_or_when_off():
    assert opening(caller("Sam", sessions=1)).warm_intro_summary is None
    o = opening(caller("Sam", sessions=1), summary="x", warm_intro=False)
    assert o.warm_intro_summary is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Sam", "Sam"),
        ("  mary   jane  ", "mary jane"),
        ("Anne-Marie O'Brien", "Anne-Marie O'Brien"),
        ("D’Angelo", "D’Angelo"),
        ("José García", "José García"),
        (None, None),
        ("", None),
        ("   ", None),
        ("um it's sam from acme", None),  # too many words
        ("A" * 41, None),
        ("Sam2", None),
        ("sam@acme.com", None),
        ("-Sam", None),
        ("Sam.", None),
    ],
)
def test_usable_name(raw, expected):
    assert usable_name(raw) == expected
