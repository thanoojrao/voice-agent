"""Whole calls through the core, wired the way app/main.py will wire it.

Registry -> Session -> Turn -> tools, with the real prompt (built from kb.example), real
SQLite, and fakes only at the edges (phone, speech-to-text, model, text-to-speech, email).
Unit tests cover each module; these cover what only shows up when they're joined: a name
saved in one call greeting the caller in the next, a message reaching the email, the
registry freeing a place when a call ends.
"""

import asyncio
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.fakes import FakeLLM, FakeNotifier, FakeSTT, FakeTTS, FakeVoiceAdapter, Reply
from voice_agent.core.clips import Clip, Clips
from voice_agent.core.greeting import GreetingTemplates
from voice_agent.core.history import CUT_OFF
from voice_agent.core.interfaces import (
    Capability,
    ChannelAdapter,
    Done,
    LlmEvent,
    ProviderError,
    SessionStart,
    TextDelta,
    TextPart,
    ToolCall,
    ToolResultPart,
    TurnEnded,
    TurnStarted,
    Words,
)
from voice_agent.core.prompt import Prompt, load_instructions, load_knowledge_base
from voice_agent.core.registry import Limits, Refusal, Registry, Slot
from voice_agent.core.session import Session, SessionDeps, SessionSettings
from voice_agent.core.turn import TurnSettings
from voice_agent.storage.sqlite import SqliteStorage

NOW = datetime(2026, 11, 15, 15, 0, tzinfo=UTC)
SAM = "+15550000001"
KB = Path(__file__).parents[2] / "kb.example"
FILLER = Clip("filler_1", b"F" * 160, 8000)
APOLOGY = Clip("apology", b"A" * 160, 8000)
TEMPLATES = GreetingTemplates(
    owner_name="Alex",
    greeting="Hi, this is Alex's AI.",
    greeting_returning="Hi {caller_name}, this is Alex's AI.",
    name_question="Who am I speaking with?",
)
SETTINGS = SessionSettings(
    turn=TurnSettings(
        sample_rate=8000,
        voice_id="voice",
        filler_after_ms=1000,
        provider_timeout_s=1.0,
        mark_grace_ms=50,
    ),
    silence_prompt_s=5.0,
    silence_hangup_s=5.0,
    max_session_s=60.0,
    goodbye_hold_s=0.05,
)
LIMITS = Limits(max_concurrent=2, daily_cap=50, per_caller_daily=2)


def build_prompt() -> Prompt:
    instructions = load_instructions(
        {
            "owner_name": "Alex",
            "tone": "friendly, concise",
            "do_not_discuss": "salary, address",
            "max_sentences": "3",
        }
    )
    return Prompt(instructions, load_knowledge_base(KB, featured_limit=3), status_stale_days=14)


class Call:
    def __init__(self, line: "Line", adapter: FakeVoiceAdapter, session: Session) -> None:
        self.line = line
        self.adapter = adapter
        self.session = session
        self.stream_index = len(line.stt.streams)  # the STT stream this session will open

    async def answered(self) -> None:
        """Wait for the greeting to finish."""
        await self.idle(turn=1)

    async def idle(self, turn: int) -> None:
        """Wait for turn `turn` to finish playing and the session to be listening again."""
        await until(lambda: f"t{turn}.end" in self.adapter.marks and self.session._turn is None)

    def say(self, text: str) -> None:
        self.stream.emit(TurnStarted(), Words(text), TurnEnded(text))

    @property
    def stream(self):
        return self.line.stt.streams[self.stream_index]

    async def ended(self) -> None:
        await until(lambda: self.line.registry.live_count == 0)


class Line:
    """The phone line: one registry, shared providers, a session per call."""

    def __init__(self, store: SqliteStorage, replies: list[Reply], settings=SETTINGS) -> None:
        self.store = store
        self.llm = FakeLLM(replies)
        self.stt = FakeSTT()
        self.tts = FakeTTS()
        self.notifier = FakeNotifier()
        self.settings = settings
        self.prompt = build_prompt()
        self.registry = Registry(store, LIMITS, self._new_session, clock=lambda: NOW)
        self.calls = 0

    def _new_session(self, start: SessionStart, adapter: ChannelAdapter) -> Session:
        deps = SessionDeps(
            storage=self.store,
            notifier=self.notifier,
            llm=self.llm,
            stt=self.stt,
            tts=self.tts,
            adapter=adapter,
            clips=Clips(apology=APOLOGY, fillers=(FILLER,)),
            prompt=self.prompt,
            templates=TEMPLATES,
            clock=lambda: NOW,
        )
        return Session(deps, self.settings, start)

    async def dial(self, number: str | None) -> Call:
        slot = await self.registry.admit("twilio", number)
        assert isinstance(slot, Slot), slot
        self.calls += 1
        adapter = FakeVoiceAdapter(frozenset({Capability.END_SESSION}), auto_play=True)
        start = SessionStart("twilio", f"CA{self.calls}", number, 8000)
        session = self.registry.start_session(slot.token, start, adapter)
        assert isinstance(session, Session)
        adapter.attach(session)
        call = Call(self, adapter, session)
        await call.answered()
        return call

    async def rows(self, sql: str) -> list[tuple]:
        cur = await self.store._db.execute(sql)
        return [tuple(r) for r in await cur.fetchall()]


@pytest.fixture
async def store(tmp_path):
    s = await SqliteStorage.open(tmp_path / "agent.db", now=lambda: NOW)
    yield s
    await s.close()


def says(*sentences: str) -> Reply:
    return Reply([TextDelta(s) for s in sentences] + [Done("end_turn")])


def uses(tool: str, text: str = "", **args: str) -> Reply:
    events: list[LlmEvent] = [TextDelta(text)] if text else []
    return Reply([*events, ToolCall(f"tu_{tool}", tool, args), Done("tool_use")])


async def until(check: Callable[[], bool], timeout: float = 3.0) -> None:
    async with asyncio.timeout(timeout):
        while not check():
            await asyncio.sleep(0.005)


# ---------------------------------------------------------------------------------


async def test_a_caller_is_remembered_from_one_call_to_the_next(store):
    line = Line(
        store,
        [
            # call 1
            uses("save_caller_details", name="Sam", company="Acme"),
            says("Nice to meet you, Sam. How can I help?"),
            uses("save_message", "I'll pass that on. ", body="Please call Sam back."),
            says("Done. Anything else?"),
            uses("end_session", "Thanks for calling, bye!", reason="goodbye"),
            # call 2
            says("Good to hear from you again."),
        ],
    )
    line.llm.summary = "Sam from Acme asked for a call back."

    first = await line.dial(SAM)
    assert line.tts.spoken[-1] == "Who am I speaking with?"
    first.say("It's Sam from Acme.")
    await first.idle(turn=2)
    first.say("Can you ask Alex to call me back?")
    await first.idle(turn=3)
    first.say("No, that's all. Bye.")
    await first.ended()
    assert first.adapter.ended

    # The real prompt went out: instructions, cached KB, then this call's context.
    system = line.llm.calls[0].system
    assert "Alex" in system[0].text
    assert system[1].cache and "<knowledge_base>" in system[1].text
    assert "name is unknown" in system[2].text

    # The message reached the owner with the caller's saved name and number.
    [(message, caller, _)] = line.notifier.messages
    assert (message.name, message.body, caller.address) == ("Sam", "Please call Sam back.", SAM)
    assert await line.rows("SELECT notified_at IS NOT NULL FROM messages") == [(1,)]

    second = await line.dial(SAM)
    assert line.tts.spoken[-1] == "Hi Sam, this is Alex's AI."  # by name, no name question
    second.say("Hi, me again.")
    await second.idle(turn=2)
    context = line.llm.calls[-1].system[2].text
    assert "The caller is Sam" in context
    assert "Sam from Acme asked for a call back." in context
    await second.adapter.hang_up()
    await second.ended()

    assert await line.rows("SELECT end_reason FROM sessions ORDER BY id") == [
        ("goodbye",),
        ("caller_hangup",),
    ]
    # Two calls today is Sam's limit.
    assert await line.registry.admit("twilio", SAM) is Refusal.CALLER_CAP


async def test_anonymous_caller_is_asked_how_to_reach_them_before_a_message_saves(store):
    line = Line(
        store,
        [
            uses("save_message", body="Interested in the widget."),  # refused: no contact
            says("Sure. What's the best way to reach you?"),
            uses("save_message", body="Interested in the widget.", contact="sam@example.com"),
            says("Got it, I'll pass that on."),
        ],
    )
    call = await line.dial(None)
    call.say("Can I leave a message? I'm interested in the widget.")
    await call.idle(turn=2)
    call.say("Email me at sam@example.com.")
    await call.idle(turn=3)

    refused = line.llm.calls[1].history[-1].parts[-1]
    assert isinstance(refused, ToolResultPart) and refused.is_error
    [(message, caller, _)] = line.notifier.messages
    assert (message.contact, caller.address) == ("sam@example.com", None)

    await call.adapter.hang_up()
    await call.ended()
    tools = await line.rows("SELECT text FROM turns WHERE speaker = 'tool' ORDER BY seq")
    assert len(tools) == 2


async def test_slow_answer_gets_a_filler_and_a_barge_in_trims_it(store):
    settings = replace(SETTINGS, turn=replace(SETTINGS.turn, filler_after_ms=30))
    slow = Reply(
        [TextDelta("Widget is a tool. "), TextDelta("It sorts things fast."), Done("end_turn")],
        delay_s=0.06,
    )
    line = Line(store, [slow, says("Sure, what would you like to know?")], settings)
    call = await line.dial(SAM)
    call.adapter.auto_play = False

    call.say("What's the widget?")
    await until(lambda: "t2.end" in call.adapter.marks)
    assert FILLER.pcm in call.adapter.audio
    await call.adapter.play_marks(1)  # the caller heard "Widget is a tool."
    call.stream.emit(TurnStarted(), Words("hold on"))
    await until(lambda: call.adapter.clears == 1)

    call.adapter.auto_play = True
    call.stream.emit(TurnEnded("hold on, a question"))
    await call.idle(turn=3)

    history = line.llm.calls[-1].history
    assert history[-2].parts == (TextPart("Widget is a tool." + CUT_OFF),)
    rows = await line.rows("SELECT text, interrupted, latency_ms FROM turns WHERE seq = 3")
    [(text, interrupted, latency_ms)] = rows
    assert (text, interrupted) == ("Widget is a tool." + CUT_OFF, 1)
    assert latency_ms >= 60

    await call.adapter.hang_up()
    await call.ended()


async def test_provider_outage_apologises_alerts_and_frees_the_line(store):
    down = ProviderError("overloaded", retryable=True)
    line = Line(store, [Reply(raises=down), Reply(raises=down)])
    call = await line.dial(SAM)
    call.say("Hello?")
    await call.ended()

    assert call.adapter.audio[-1] == APOLOGY.pcm
    assert call.adapter.ended
    [(_, body)] = line.notifier.alerts
    assert "overloaded" in body
    assert await line.rows("SELECT end_reason FROM sessions") == [("error",)]
    assert isinstance(await line.registry.admit("twilio", None), Slot)
