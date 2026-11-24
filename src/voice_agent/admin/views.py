"""What the admin pages show, worked out from stored rows (design doc §13).

The templates only lay things out; every label, time, length and count they print is
made here, so it can be tested without rendering HTML. Times are shown in the owner's
time zone, the way a phone's call log shows them.

Everything here except the labels came from callers (through speech recognition and the
model), and the templates escape all of it.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo
from statistics import mean

from voice_agent.core.interfaces import Caller, MessageRow, SessionRow, TurnRow
from voice_agent.core.tools import DECLINED

# How a call ended, in the words the call list uses.
END_LABELS = {
    "goodbye": "Said goodbye",
    "caller_hangup": "Caller hung up",
    "silence": "Went quiet",
    "limit": "Reached the time limit",
    "abuse": "Ended for abuse",
    "error": "Something went wrong",
}
# Ends that are shown in red, like a missed call.
PROBLEM_ENDS = frozenset({"error", "abuse"})
AVATAR_TINTS = 6  # .tint-0 to .tint-5 in the stylesheet

# What a tool call looks like in the transcript.
_TOOL_LABELS = {
    "save_message": "Saved message",
    "save_caller_details": "Saved details",
    "end_session": "Ended the call",
}


@dataclass(frozen=True, slots=True)
class Person:
    """A caller as the lists and headers show them."""

    title: str  # name, else number, else "Number withheld"
    initials: str  # "" when there's no name: the template shows a person icon
    tint: int  # which avatar colour, stable per name
    number: str | None  # formatted for reading ("+1 555 010 0142"); None if withheld


@dataclass(frozen=True, slots=True)
class Bubble:
    kind: str  # "caller", "me" or "tool"
    text: str
    reply_s: float | None = None  # my replies: seconds from the caller stopping to my voice
    cut_off: bool = False


@dataclass(frozen=True, slots=True)
class ReplyStats:
    average_s: float
    slowest_s: float
    cut_offs: int


def person(caller: Caller) -> Person:
    number = format_number(caller.address) if caller.address else None
    name = " ".join(caller.name.split()) if caller.name else None
    title = name or number or "Number withheld"
    return Person(title, initials(name) if name else "", tint(name) if name else 0, number)


def initials(name: str) -> str:
    return "".join(w[0] for w in name.split()[:2]).upper()


def tint(name: str) -> int:
    """The same name always gets the same colour (crc32, unlike hash(), is stable)."""
    return zlib.crc32(name.casefold().encode()) % AVATAR_TINTS


def format_number(address: str) -> str:
    """A US number in E.164 ("+15550100142") as "+1 555 010 0142"; anything else as is."""
    digits = address[2:]
    if address.startswith("+1") and len(digits) == 10 and digits.isdigit():
        return f"+1 {digits[:3]} {digits[3:6]} {digits[6:]}"
    return address


def day_label(when: datetime, today: date, tz: tzinfo) -> str:
    """The heading a call is grouped under: "Today", "Yesterday", or "Sat 22 Nov"."""
    day = when.astimezone(tz).date()
    if day == today:
        return "Today"
    if day == today - timedelta(days=1):
        return "Yesterday"
    label = f"{day:%a} {day.day} {day:%b}"
    return label if day.year == today.year else f"{label} {day.year}"


def time_label(when: datetime, tz: tzinfo) -> str:
    local = when.astimezone(tz)
    return f"{local.hour % 12 or 12}:{local:%M %p}"


def when_label(when: datetime, today: date, tz: tzinfo) -> str:
    """ "Today, 6:17 PM" or "Sat 22 Nov, 6:17 PM": a call's date and time."""
    return f"{day_label(when, today, tz)}, {time_label(when, tz)}"


def length_label(session: SessionRow) -> str:
    """How long the call lasted: "40s", "3m 10s". "In progress" if it hasn't ended."""
    if session.ended_at is None:
        return "In progress"
    seconds = max(0, round((session.ended_at - session.started_at).total_seconds()))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


def end_label(session: SessionRow) -> str:
    if session.end_reason is None:
        return "In progress" if session.ended_at is None else "Unknown"
    return END_LABELS.get(session.end_reason, session.end_reason)


def is_problem(session: SessionRow) -> bool:
    return session.end_reason in PROBLEM_ENDS


def how_to_reach(msg: MessageRow, caller: Caller) -> str:
    """Where to get back to them, as the message card says it."""
    if msg.contact == DECLINED:
        return "They didn't say"
    if msg.contact:
        return msg.contact
    return "This number" if caller.address else "Not given"


def message_status(msg: MessageRow, tz: tzinfo, today: date) -> str | None:
    """ "Emailed 6:20 PM" once it's gone out; None while it's not emailed yet."""
    if msg.notified_at is None:
        return None
    if msg.notified_at.astimezone(tz).date() == today:
        return f"Emailed {time_label(msg.notified_at, tz)}"
    return f"Emailed {when_label(msg.notified_at, today, tz)}"


def transcript(turns: list[TurnRow]) -> list[Bubble]:
    replies = {id(t) for t in _replies(turns)}
    bubbles = []
    for t in turns:
        if t.speaker == "tool":
            bubbles.append(Bubble("tool", tool_label(t.text)))
        elif t.speaker == "agent":
            reply_s = _seconds(t) if id(t) in replies else None
            bubbles.append(Bubble("me", t.text, reply_s, t.interrupted))
        else:
            bubbles.append(Bubble("caller", t.text))
    return bubbles


def tool_label(text: str) -> str:
    """A stored tool turn ("save_caller_details {"name": "Sam"}") as a short label."""
    name, _, raw = text.partition(" ")
    label = _TOOL_LABELS.get(name, name)
    if name == "save_caller_details":
        try:
            args = json.loads(raw) if raw else {}
        except ValueError:
            args = {}
        saved = [f"{k} {args[k]}" for k in ("name", "company") if isinstance(args.get(k), str)]
        if saved:
            label = "Saved " + ", ".join(saved)
    return label


def reply_stats(turns: list[TurnRow]) -> ReplyStats | None:
    """How quickly I answered the caller, and how often I was cut off. None if I never
    answered them."""
    timed = [s for t in _replies(turns) if (s := _seconds(t)) is not None]
    if not timed:
        return None
    cut_offs = sum(t.interrupted for t in turns if t.speaker == "agent")
    return ReplyStats(mean(timed), max(timed), cut_offs)


def _replies(turns: list[TurnRow]) -> list[TurnRow]:
    """My turns that answer the caller. The greeting and "Are you still there?" also have
    a latency, but nothing to answer, so they'd skew the numbers. Tool calls sit between
    the caller's turn and my reply, so they're skipped over."""
    found, last = [], None
    for t in turns:
        if t.speaker == "tool":
            continue
        if t.speaker == "agent" and last == "caller":
            found.append(t)
        last = t.speaker
    return found


def _seconds(turn: TurnRow) -> float | None:
    return turn.latency_ms / 1000 if turn.latency_ms is not None else None
