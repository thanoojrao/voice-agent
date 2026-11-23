"""Whole calls against the fakes and real SQLite: greeting, replies, barge-in, timers, endings."""

import asyncio
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from tests.fakes import FakeLLM, FakeNotifier, FakeSTT, FakeTTS, FakeVoiceAdapter, Reply
from voice_agent.core.clips import Clip, Clips
from voice_agent.core.greeting import GreetingTemplates
from voice_agent.core.interfaces import (
    CallNotes,
    Capability,
    Done,
    MessageRow,
    ProviderError,
    SessionStart,
    TextDelta,
    TextPart,
    ToolCall,
    TurnEnded,
    TurnStarted,
    Words,
)
from voice_agent.core.prompt import KnowledgeBase, Prompt
from voice_agent.core.session import (
    LIMIT_NOTE,
    SILENCE_GOODBYE,
    STILL_THERE,
    Session,
    SessionDeps,
    SessionSettings,
)
from voice_agent.core.turn import TurnSettings
from voice_agent.storage.sqlite import SqliteStorage

NOW = datetime(2026, 11, 15, 15, 0, tzinfo=UTC)
NUMBER = "+15551234567"
APOLOGY = Clip("apology", b"A" * 160, 8000)
TEMPLATES = GreetingTemplates(
    owner_name="Alex",
    greeting="Hi, this is Alex's AI.",
    greeting_returning="Hi {caller_name}, this is Alex's AI.",
    name_question="Who am I speaking with?",
)
PROMPT = Prompt("Be Alex.", KnowledgeBase("<knowledge_base/>", None, (), {}), 14)
TURN = TurnSettings(
    sample_rate=8000,
    voice_id="voice",
    filler_after_ms=1000,
    provider_timeout_s=1.0,
    mark_grace_ms=50,
)
SETTINGS = SessionSettings(
    turn=TURN,
    silence_prompt_s=5.0,
    silence_hangup_s=5.0,
    max_session_s=30.0,
    limit_grace_s=5.0,
    goodbye_hold_s=0.05,
)


@pytest.fixture
async def store(tmp_path):
    s = await SqliteStorage.open(tmp_path / "agent.db", now=lambda: NOW)
    yield s
    await s.close()


class Call:
    def __init__(
        self,
        store: SqliteStorage,
        replies: list[Reply],
        settings: SessionSettings = SETTINGS,
        address: str | None = NUMBER,
    ) -> None:
        self.store = store
        self.llm = FakeLLM(replies)
        self.stt = FakeSTT()
        self.tts = FakeTTS()
        self.adapter = FakeVoiceAdapter(frozenset({Capability.END_SESSION}), auto_play=True)
        self.notifier = FakeNotifier()
        deps = SessionDeps(
            storage=store,
            notifier=self.notifier,
            llm=self.llm,
            stt=self.stt,
            tts=self.tts,
            adapter=self.adapter,
            clips=Clips(apology=APOLOGY, fillers=(Clip("filler_1", b"F" * 160, 8000),)),
            prompt=PROMPT,
            templates=TEMPLATES,
            clock=lambda: NOW,
        )
        self.session = Session(deps, settings, SessionStart("twilio", "CA1", address, 8000))
        self.adapter.attach(self.session)
        self.task: asyncio.Task[str] | None = None

    async def connect(self) -> None:
        """Start the call and wait for the greeting to finish."""
        self.task = asyncio.create_task(self.session.run())
        await self.played(1)

    async def played(self, turn: int) -> None:
        """Wait for turn `turn` to finish playing and the session to be idle again."""
        await until(lambda: f"t{turn}.end" in self.adapter.marks and self.session._turn is None)

    def say(self, text: str) -> None:
        self.stt.stream.emit(TurnStarted(), Words(text), TurnEnded(text))

    async def ended(self) -> str:
        assert self.task is not None
        return await asyncio.wait_for(self.task, 3)

    async def turns(self) -> list[tuple]:
        cur = await self.store._db.execute(
            "SELECT seq, speaker, text, interrupted FROM turns ORDER BY seq"
        )
        return [tuple(r) for r in await cur.fetchall()]

    async def session_row(self) -> tuple:
        cur = await self.store._db.execute("SELECT end_reason, summary FROM sessions")
        return tuple(await cur.fetchone())  # type: ignore[arg-type]


def reply(*sentences: str, delay_s: float = 0.0) -> Reply:
    return Reply([TextDelta(s) for s in sentences] + [Done("end_turn")], delay_s=delay_s)


def goodbye(line: str = "Bye now.", reason: str = "goodbye") -> Reply:
    call = ToolCall("tu_end", "end_session", {"reason": reason})
    return Reply([TextDelta(line), call, Done("tool_use")])


async def until(check: Callable[[], bool], timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not check():
            await asyncio.sleep(0.005)


# -- greeting and replies --------------------------------------------------------


async def test_first_time_caller_gets_greeting_and_name_question(store):
    call = Call(store, [])
    await call.connect()
    assert call.tts.spoken == ["Hi, this is Alex's AI.", "Who am I speaking with?"]
    assert call.llm.calls == []  # the greeting is never generated
    await call.adapter.hang_up()
    assert await call.ended() == "caller_hangup"


async def test_the_prompt_is_prewarmed_once_at_the_start_with_the_sessions_prompt(store):
    call = Call(store, [reply("I build agents.")])
    await call.connect()
    call.say("What do you do?")
    await call.played(2)

    [warm] = call.llm.prewarmed
    [real] = call.llm.calls
    assert warm.system == real.system
    assert warm.tools == real.tools
    await call.adapter.hang_up()
    await call.ended()


async def test_a_failed_prewarm_does_not_affect_the_call(store):
    call = Call(store, [reply("I build agents.")])
    call.llm.prewarm_raises = ProviderError("overloaded", retryable=True)
    await call.connect()
    call.say("What do you do?")
    await call.played(2)
    assert call.tts.spoken[-1] == "I build agents."
    await call.adapter.hang_up()
    assert await call.ended() == "caller_hangup"


async def test_returning_caller_is_greeted_by_name_and_the_model_gets_the_summary(store):
    caller = await store.get_or_create_caller("twilio", NUMBER)
    await store.set_caller_details(caller.id, "Sam", None, None)
    old = await store.create_session(caller.id, SessionStart("twilio", "CA0", NUMBER, 8000))
    await store.end_session(old, "goodbye", "Asked about the voice agent.")

    call = Call(store, [reply("Sure.")])
    await call.connect()
    assert call.tts.spoken == ["Hi Sam, this is Alex's AI."]
    call.say("Tell me more.")
    await call.played(2)
    context = call.llm.calls[0].system[2].text
    assert "Sam" in context and "Asked about the voice agent." in context
    await call.adapter.hang_up()
    await call.ended()


async def test_the_model_knows_the_last_calls_message_was_passed_on(store):
    caller = await store.get_or_create_caller("twilio", NUMBER)
    old = await store.create_session(caller.id, SessionStart("twilio", "CA0", NUMBER, 8000))
    await store.save_message(old, MessageRow("Call me back on Friday.", NOW))
    await store.end_session(old, "goodbye", "Left a callback request.")

    call = Call(store, [reply("Sure.")])
    await call.connect()
    call.say("Hi again.")
    await call.played(2)
    context = call.llm.calls[0].system[2].text
    assert 'left a message for the real me, and it was passed on: "Call me back on Friday."' in (
        context
    )
    await call.adapter.hang_up()
    await call.ended()


async def test_anonymous_caller_is_always_asked_their_name(store):
    call = Call(store, [], address=None)
    await call.connect()
    assert call.tts.spoken[-1] == "Who am I speaking with?"
    await call.adapter.hang_up()
    await call.ended()


async def test_a_conversation_is_stored_and_summarized(store):
    call = Call(store, [reply("I build agents."), goodbye()])
    await call.connect()
    call.say("What do you do?")
    await call.played(2)
    call.say("Thanks, bye.")

    assert await call.ended() == "goodbye"
    assert call.adapter.ended
    turns = await call.turns()
    assert [(speaker, text) for _, speaker, text, _ in turns] == [
        ("agent", "Hi, this is Alex's AI. Who am I speaking with?"),
        ("caller", "What do you do?"),
        ("agent", "I build agents."),
        ("caller", "Thanks, bye."),
        ("tool", 'end_session {"reason": "goodbye"}'),
        ("agent", "Bye now."),
    ]
    assert [seq for seq, *_ in turns] == [1, 2, 3, 4, 5, 6]
    assert await call.session_row() == ("goodbye", "Fake summary.")
    assert "Caller: What do you do?" in call.llm.summarized[0]


async def test_details_the_model_forgot_to_save_are_filled_in_from_the_summary(store):
    # The model said "Nice to meet you, Sam" but never called save_caller_details.
    call = Call(store, [reply("Nice to meet you, Sam.")])
    call.llm.notes = CallNotes("The caller said hello.", name="Sam", company="Acme")
    await call.connect()
    call.say("Sam.")
    await call.played(2)
    await call.adapter.hang_up()
    await call.ended()

    caller = await store.get_or_create_caller("twilio", NUMBER)
    assert (caller.name, caller.company, caller.reason) == ("Sam", "Acme", None)
    assert await call.session_row() == ("caller_hangup", "The caller said hello.")


async def test_the_summary_never_overwrites_details_saved_during_the_call(store):
    caller = await store.get_or_create_caller("twilio", NUMBER)
    await store.set_caller_details(caller.id, "Priya", None, None)
    call = Call(store, [reply("Thanks, Priya.")])
    call.llm.notes = CallNotes("x", name="Sam", reason="a backend role")
    await call.connect()
    call.say("I'm calling about a backend role.")
    await call.played(2)
    await call.adapter.hang_up()
    await call.ended()

    caller = await store.get_or_create_caller("twilio", NUMBER)
    assert (caller.name, caller.reason) == ("Priya", "a backend role")


async def test_a_summary_name_that_isnt_a_name_is_ignored(store):
    call = Call(store, [reply("Sorry, could you repeat that?")])
    call.llm.notes = CallNotes("x", name="um it's sam from acme corp, the recruiting team")
    await call.connect()
    call.say("Um it's Sam from Acme Corp, the recruiting team.")
    await call.played(2)
    await call.adapter.hang_up()
    await call.ended()

    caller = await store.get_or_create_caller("twilio", NUMBER)
    assert caller.name is None


async def test_call_with_no_caller_words_is_not_summarized(store):
    call = Call(store, [])
    await call.connect()
    await call.adapter.hang_up()
    await call.ended()
    assert call.llm.summarized == []
    assert await call.session_row() == ("caller_hangup", None)


# -- barge-in --------------------------------------------------------------------------


async def test_real_words_interrupt_the_agent(store):
    call = Call(store, [reply("One two three four five six."), reply("Sure, go ahead.")])
    await call.connect()
    call.adapter.auto_play = False
    call.say("Question.")
    await until(lambda: "t2.end" in call.adapter.marks)  # all sent, still "playing"

    call.stt.stream.emit(TurnStarted(), Words("wait actually"))
    await until(lambda: call.adapter.clears == 1)
    call.adapter.auto_play = True
    call.stt.stream.emit(TurnEnded("wait actually"))
    await call.played(3)

    assert call.tts.spoken[-1] == "Sure, go ahead."
    await call.adapter.hang_up()
    await call.ended()
    rows = await call.turns()
    assert ("agent", "", 1) not in [(s, t, i) for _, s, t, i in rows]  # nothing heard: no row
    assert [s for _, s, _, _ in rows] == ["agent", "caller", "caller", "agent"]


async def test_backchannel_does_not_interrupt_and_is_not_answered(store):
    call = Call(store, [reply("One two three four five six.")])
    await call.connect()
    call.adapter.auto_play = False
    call.say("Question.")
    await until(lambda: "t2.end" in call.adapter.marks)

    call.stt.stream.emit(TurnStarted(), Words("mm-hm"), TurnEnded("mm-hm"))
    await asyncio.sleep(0.05)
    assert call.adapter.clears == 0
    assert len(call.llm.calls) == 1
    await call.adapter.hang_up()
    await call.ended()


async def test_one_real_word_interrupts_when_the_turn_ends_not_mid_sentence(store):
    call = Call(store, [reply("One two three four five six."), reply("Okay, stopping.")])
    await call.connect()
    call.adapter.auto_play = False
    call.say("Question.")
    await until(lambda: "t2.end" in call.adapter.marks)

    call.stt.stream.emit(TurnStarted(), Words("Stop."))
    await asyncio.sleep(0.05)
    assert call.adapter.clears == 0  # one word: could be a stray word or noise
    call.adapter.auto_play = True
    call.stt.stream.emit(TurnEnded("Stop."))
    await call.played(3)

    assert call.adapter.clears == 1
    assert call.tts.spoken[-1] == "Okay, stopping."
    await call.adapter.hang_up()
    await call.ended()


async def test_a_hum_after_the_goodbye_does_not_carry_the_call_on(store):
    settings = replace(SETTINGS, goodbye_hold_s=0.3)
    call = Call(store, [goodbye()], settings=settings)
    await call.connect()
    call.say("Thanks, bye.")
    await call.played(2)
    call.say("Mhmm.")  # spelled the way Flux spelled it on a real call
    assert await call.ended() == "goodbye"
    assert len(call.llm.calls) == 1


async def test_interrupting_the_goodbye_keeps_the_call_going(store):
    call = Call(store, [goodbye("Bye for now, take care."), reply("Of course.")])
    await call.connect()
    call.adapter.auto_play = False
    call.say("That's all.")
    await until(lambda: "t2.end" in call.adapter.marks)
    call.adapter.auto_play = True
    call.say("Oh wait, one more thing.")
    await call.played(3)
    assert not call.adapter.ended
    await call.adapter.hang_up()
    assert await call.ended() == "caller_hangup"


async def test_after_a_goodbye_the_line_stays_open_for_the_hold_then_hangs_up(store):
    settings = replace(SETTINGS, goodbye_hold_s=0.3)
    call = Call(store, [goodbye()], settings=settings)
    await call.connect()
    call.say("Thanks, bye.")
    await call.played(2)

    await asyncio.sleep(0.1)
    assert not call.adapter.ended  # still holding
    assert await call.ended() == "goodbye"
    assert call.adapter.ended


async def test_speaking_during_the_hold_carries_the_call_on(store):
    settings = replace(SETTINGS, goodbye_hold_s=0.3)
    call = Call(store, [goodbye(), reply("Of course, go ahead.")], settings=settings)
    await call.connect()
    call.say("Okay, thanks.")
    await call.played(2)
    call.say("Oh wait, can I leave a message?")
    await call.played(3)

    assert call.tts.spoken[-1] == "Of course, go ahead."
    await asyncio.sleep(0.4)  # longer than the hold: the goodbye no longer counts
    assert not call.adapter.ended
    await call.adapter.hang_up()
    assert await call.ended() == "caller_hangup"


async def test_a_backchannel_during_the_hold_is_not_answered(store):
    settings = replace(SETTINGS, goodbye_hold_s=0.1)
    call = Call(store, [goodbye()], settings=settings)
    await call.connect()
    call.say("Thanks, bye.")
    await call.played(2)
    call.say("Okay.")
    assert await call.ended() == "goodbye"
    assert len(call.llm.calls) == 1


async def test_an_abusive_caller_is_hung_up_on_without_a_hold(store):
    settings = replace(SETTINGS, goodbye_hold_s=10.0)
    call = Call(store, [goodbye("I'm ending this call now.", reason="abuse")], settings=settings)
    await call.connect()
    call.say("Something abusive.")
    assert await call.ended() == "abuse"  # well inside the 10 s hold


async def test_hanging_up_during_the_hold_ends_the_call(store):
    settings = replace(SETTINGS, goodbye_hold_s=5.0)
    call = Call(store, [goodbye()], settings=settings)
    await call.connect()
    call.say("Thanks, bye.")
    await call.played(2)
    await call.adapter.hang_up()
    assert await call.ended() == "caller_hangup"


# -- timers --------------------------------------------------------------------------


async def test_silence_prompts_once_then_hangs_up(store):
    settings = SessionSettings(
        turn=TURN, silence_prompt_s=0.05, silence_hangup_s=0.05, max_session_s=30.0,
        goodbye_hold_s=0.05,
    )  # fmt: skip
    call = Call(store, [], settings=settings)
    await call.connect()
    assert await call.ended() == "silence"
    assert " ".join(call.tts.spoken).endswith(f"{STILL_THERE} {SILENCE_GOODBYE}")
    assert call.adapter.ended


async def test_speaking_resets_the_silence_prompt(store):
    settings = SessionSettings(
        turn=TURN, silence_prompt_s=0.1, silence_hangup_s=5.0, max_session_s=30.0
    )
    call = Call(store, [reply("Hello.")], settings=settings)
    await call.connect()
    await until(lambda: STILL_THERE in call.tts.spoken)
    call.say("Yes, sorry.")
    await until(lambda: "Hello." in call.tts.spoken)
    await call.adapter.hang_up()
    await call.ended()


async def test_time_limit_tells_the_model_to_wrap_up(store):
    settings = SessionSettings(
        turn=TURN, silence_prompt_s=5.0, silence_hangup_s=5.0, max_session_s=0.1,
        goodbye_hold_s=0.05,
    )  # fmt: skip
    call = Call(store, [goodbye("We're out of time, bye!", reason="limit")], settings=settings)
    await call.connect()
    assert await call.ended() == "limit"
    last = call.llm.calls[0].history[-1]
    assert last.role == "user" and last.parts[-1] == TextPart(LIMIT_NOTE)


async def test_hard_limit_ends_the_call_even_mid_turn(store):
    settings = SessionSettings(
        turn=TURN, silence_prompt_s=5.0, silence_hangup_s=5.0, max_session_s=0.05,
        limit_grace_s=0.05,
    )  # fmt: skip
    call = Call(store, [reply("Too late.", delay_s=2.0)], settings=settings)
    await call.connect()
    assert await call.ended() == "limit"
    assert call.adapter.ended


# -- failures ------------------------------------------------------------------------


async def test_provider_failure_apologises_alerts_and_ends(store):
    err = ProviderError("down", retryable=True)
    call = Call(store, [Reply(raises=err), Reply(raises=err)])
    await call.connect()
    call.say("Hello?")
    assert await call.ended() == "error"
    assert call.adapter.audio[-1] == APOLOGY.pcm
    [(subject, body)] = call.notifier.alerts
    assert "down" in body
    assert call.adapter.ended


async def test_speech_to_text_dropping_ends_the_call(store):
    call = Call(store, [])
    await call.connect()
    await call.stt.stream.close()
    assert await call.ended() == "error"
    assert call.adapter.audio[-1] == APOLOGY.pcm
    assert len(call.notifier.alerts) == 1


async def test_hang_up_mid_reply_stores_what_was_heard_without_clearing(store):
    call = Call(store, [reply("One two three four.")])
    await call.connect()
    call.adapter.auto_play = False
    call.say("Question.")
    await until(lambda: "t2.end" in call.adapter.marks)
    await call.adapter.play_marks(1)  # "t2.w4" played, "t2.end" not yet
    await call.adapter.hang_up()
    assert await call.ended() == "caller_hangup"
    assert call.adapter.clears == 0
    assert not call.adapter.ended  # nothing to hang up: the caller already did
    *_, last = await call.turns()
    assert last[1:] == ("agent", "One two three four.", 1)
