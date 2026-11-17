"""One agent turn against the fakes: scripted LLM, fake TTS with word timings, fake phone."""

import asyncio
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from tests.fakes import FakeLLM, FakeNotifier, FakeTTS, FakeVoiceAdapter, Reply
from voice_agent.core.clips import Clip, Clips
from voice_agent.core.history import CUT_OFF, History
from voice_agent.core.interfaces import (
    Done,
    Message,
    ProviderError,
    Refused,
    SessionRow,
    SessionStart,
    TextDelta,
    TextPart,
    ToolCall,
    ToolResultPart,
    ToolUsePart,
)
from voice_agent.core.tools import ToolContext, tool_defs
from voice_agent.core.turn import (
    MAX_TOOL_ROUNDS,
    REFUSAL_LINE,
    Outcome,
    Turn,
    TurnDeps,
    TurnSettings,
)
from voice_agent.storage.sqlite import SqliteStorage

NOW = datetime(2026, 11, 15, 15, 0, tzinfo=UTC)
FILLER = Clip("filler_1", b"F" * 160, 8000)
APOLOGY = Clip("apology", b"A" * 160, 8000)
SETTINGS = TurnSettings(
    sample_rate=8000,
    voice_id="voice",
    filler_after_ms=1000,
    provider_timeout_s=1.0,
    mark_grace_ms=50,
)


class EchoAdapter(FakeVoiceAdapter):
    """Echoes each mark straight back, as if the audio played instantly."""

    def __init__(self) -> None:
        super().__init__()
        self.echo = True
        self.turn: Turn | None = None

    async def mark(self, name: str) -> None:
        await super().mark(name)
        if self.echo and self.turn is not None:
            self.turn.mark_played(name)


@pytest.fixture
async def ctx(tmp_path):
    store = await SqliteStorage.open(tmp_path / "agent.db", now=lambda: NOW)
    caller = await store.get_or_create_caller("twilio", "+15551234567")
    session_id = await store.create_session(
        caller.id, SessionStart("twilio", "CA1", caller.address, 8000)
    )
    session = SessionRow(session_id, caller.id, "twilio", "CA1", NOW, None, None, None)
    yield ToolContext(store, FakeNotifier(), session, caller, clock=lambda: NOW)
    await store.close()


class Rig:
    def __init__(self, ctx: ToolContext, replies: list[Reply], settings=SETTINGS) -> None:
        self.llm = FakeLLM(replies)
        self.tts = FakeTTS()
        self.adapter = EchoAdapter()
        self.history = History()
        self.history.add_caller("Hi, what do you work on?")
        self.ctx = ctx
        self.deps = TurnDeps(
            llm=self.llm,
            tts=self.tts,
            adapter=self.adapter,
            clips=Clips(apology=APOLOGY, fillers=(FILLER,)),
            tools=tool_defs(can_end_session=True),
            tool_ctx=ctx,
        )
        self.settings = settings

    def turn(self, number: int = 1) -> Turn:
        turn = Turn(self.deps, self.settings, self.history, [], number)
        self.adapter.turn = turn
        return turn

    async def run(self):
        turn = self.turn()
        turn.start()
        return await turn.result()

    @property
    def last(self) -> Message:
        return self.history.messages[-1]


def text(*parts: str) -> list:
    return [TextDelta(p) for p in parts] + [Done("end_turn")]


async def until(check: Callable[[], bool], timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not check():
            await asyncio.sleep(0.005)


# -- the normal path ----------------------------------------------------------


async def test_reply_is_spoken_sentence_by_sentence_and_kept_in_history(ctx):
    rig = Rig(ctx, [Reply(text("I build AI agents. ", "Mostly at Dottr."))])
    result = await rig.run()

    assert result.outcome is Outcome.COMPLETED
    assert rig.tts.spoken == ["I build AI agents.", "Mostly at Dottr."]
    assert rig.tts.streams[0].finished
    assert result.spoken == "I build AI agents. Mostly at Dottr."
    assert rig.last == Message("assistant", (TextPart(result.spoken),))
    assert result.latency_ms is not None and result.latency_ms >= 0
    assert result.filler is None


async def test_tagged_text_is_spelled_out_in_speech_and_history(ctx):
    rig = Rig(ctx, [Reply(text("Email me. ", "That's <spell>ab@gmail.com</spell>."))])
    result = await rig.run()
    assert rig.tts.spoken == ["Email me.", "That's A B, at gmail dot com."]
    assert result.spoken == "Email me. That's A B, at gmail dot com."


async def test_marks_count_words_and_end_with_an_end_mark(ctx):
    rig = Rig(ctx, [Reply(text("One two. ", "Three four five."))])
    await rig.run()
    assert rig.adapter.marks == ["t1.w2", "t1.w5", "t1.end"]
    assert len(rig.adapter.audio) == 2


async def test_turn_waits_for_the_final_mark_but_not_forever(ctx):
    rig = Rig(ctx, [Reply(text("Short."))])
    rig.adapter.echo = False  # an adapter that never echoes marks
    result = await rig.run()  # fake audio is 300 ms, plus 50 ms grace
    assert result.outcome is Outcome.COMPLETED
    assert result.spoken == "Short."


async def test_marks_from_another_turn_are_ignored(ctx):
    rig = Rig(ctx, [Reply(text("One two three."))])
    rig.adapter.echo = False
    turn = rig.turn(number=2)
    turn.start()
    await until(lambda: "t2.end" in rig.adapter.marks)
    turn.mark_played("t1.w3")  # a late echo from turn 1
    turn.mark_played("t1.end")
    result = await turn.interrupt()
    assert result.spoken == ""


# -- filler ---------------------------------------------------------------------


async def test_filler_plays_when_the_reply_is_slow(ctx):
    slow = replace(SETTINGS, filler_after_ms=20)
    rig = Rig(ctx, [Reply(text("Here you go."), delay_s=0.1)], settings=slow)
    result = await rig.run()
    assert result.filler == FILLER
    assert rig.adapter.audio[0] == FILLER.pcm  # before the reply
    assert len(rig.adapter.audio) == 2
    assert result.latency_ms is not None and result.latency_ms >= 100


async def test_no_filler_when_the_reply_is_fast(ctx):
    rig = Rig(ctx, [Reply(text("Quick."))])
    result = await rig.run()
    assert result.filler is None
    assert FILLER.pcm not in rig.adapter.audio


# -- tools ----------------------------------------------------------------------


async def test_tool_call_runs_between_requests_and_the_reply_continues(ctx):
    call = ToolCall("tu_1", "save_message", {"body": "Please call me back."})
    rig = Rig(
        ctx,
        [
            Reply([TextDelta("Sure, I'll pass that on. "), call, Done("tool_use")]),
            Reply(text("Done. Anything else?")),
        ],
    )
    result = await rig.run()

    assert rig.tts.spoken == ["Sure, I'll pass that on.", "Done.", "Anything else?"]
    assert len(ctx.notifier.messages) == 1  # type: ignore[attr-defined]
    [run] = result.tool_runs
    assert run.call == ToolUsePart("tu_1", "save_message", {"body": "Please call me back."})
    assert not run.result.is_error
    # The second request saw the tool call and its result.
    second = rig.llm.calls[1].history
    assert second[-2].parts[-1] == run.call
    assert second[-1].parts == (run.result,)
    assert result.spoken == "Sure, I'll pass that on. Done. Anything else?"


async def test_parallel_tool_calls_share_one_message_each(ctx):
    calls = [
        ToolCall("tu_1", "save_caller_details", {"name": "Sam"}),
        ToolCall("tu_2", "save_caller_details", {"company": "Acme"}),
    ]
    rig = Rig(ctx, [Reply([*calls, Done("tool_use")]), Reply(text("Thanks, Sam."))])
    await rig.run()
    assistant, results = rig.llm.calls[1].history[-2:]
    assert [p.id for p in assistant.parts if isinstance(p, ToolUsePart)] == ["tu_1", "tu_2"]
    assert [p.tool_use_id for p in results.parts if isinstance(p, ToolResultPart)] == [
        "tu_1",
        "tu_2",
    ]


async def test_end_session_is_reported(ctx):
    call = ToolCall("tu_1", "end_session", {"reason": "goodbye"})
    rig = Rig(ctx, [Reply([TextDelta("Bye for now."), call, Done("tool_use")]), Reply(text())])
    result = await rig.run()
    assert result.end_reason == "goodbye"
    assert result.spoken == "Bye for now."


async def test_tool_loop_is_capped(ctx):
    def again(i: int) -> Reply:
        return Reply(
            [ToolCall(f"tu_{i}", "save_caller_details", {"reason": "x"}), Done("tool_use")]
        )

    rig = Rig(ctx, [again(i) for i in range(MAX_TOOL_ROUNDS + 1)])
    result = await rig.run()
    assert len(rig.llm.calls) == MAX_TOOL_ROUNDS + 1
    assert [r.result.is_error for r in result.tool_runs] == [False] * MAX_TOOL_ROUNDS + [True]
    assert result.outcome is Outcome.COMPLETED


# -- refusal and failures ----------------------------------------------------------


async def test_refusal_speaks_the_fixed_line(ctx):
    rig = Rig(ctx, [Reply([Refused()])])
    result = await rig.run()
    assert rig.tts.spoken == [REFUSAL_LINE]
    assert result.spoken == REFUSAL_LINE


async def test_failure_before_speaking_is_retried_once(ctx):
    rig = Rig(
        ctx,
        [Reply(raises=ProviderError("rate limited", retryable=True)), Reply(text("Hello."))],
    )
    result = await rig.run()
    assert result.outcome is Outcome.COMPLETED
    assert len(rig.llm.calls) == 2
    assert rig.tts.spoken == ["Hello."]


async def test_slow_first_token_counts_as_a_retryable_failure(ctx):
    rig = Rig(
        ctx,
        [Reply(text("Too slow."), delay_s=0.2), Reply(text("Hello."))],
        settings=replace(SETTINGS, provider_timeout_s=0.05),
    )
    result = await rig.run()
    assert result.outcome is Outcome.COMPLETED
    assert rig.tts.spoken == ["Hello."]


async def test_second_failure_plays_the_apology(ctx):
    err = ProviderError("down", retryable=True)
    rig = Rig(ctx, [Reply(raises=err), Reply(raises=err)])
    result = await rig.run()
    assert result.outcome is Outcome.FAILED
    assert "down" in (result.error or "")
    assert rig.adapter.audio == [APOLOGY.pcm]
    assert rig.adapter.marks[-1] == "t1.apology"
    assert rig.history.messages[-1].role == "user"  # nothing was said, so no agent message


async def test_non_retryable_failure_is_not_retried(ctx):
    rig = Rig(ctx, [Reply(raises=ProviderError("bad request", retryable=False))])
    result = await rig.run()
    assert result.outcome is Outcome.FAILED
    assert len(rig.llm.calls) == 1


async def test_failure_mid_reply_is_not_retried(ctx):
    err = ProviderError("dropped", retryable=True)
    rig = Rig(ctx, [Reply([TextDelta("First part. "), TextDelta("second")], raises=err)])
    result = await rig.run()
    assert result.outcome is Outcome.FAILED
    assert len(rig.llm.calls) == 1  # a retry would repeat "First part."
    assert rig.adapter.audio[-1] == APOLOGY.pcm


# -- interruption ----------------------------------------------------------------------


async def test_interrupt_keeps_only_the_words_heard(ctx):
    rig = Rig(ctx, [Reply(text("One two three. ", "Four five six."))])
    rig.adapter.echo = False
    turn = rig.turn()
    turn.start()
    await until(lambda: "t1.end" in rig.adapter.marks)
    turn.mark_played("t1.w3")

    result = await turn.interrupt()
    assert result.outcome is Outcome.INTERRUPTED
    assert result.spoken == "One two three." + CUT_OFF
    assert rig.last == Message("assistant", (TextPart("One two three." + CUT_OFF),))
    assert rig.adapter.clears == 1
    assert rig.tts.streams[0].cancelled


async def test_interrupt_while_thinking_leaves_no_agent_message(ctx):
    rig = Rig(ctx, [Reply(text("Never heard."), delay_s=0.5)])
    turn = rig.turn()
    turn.start()
    await asyncio.sleep(0.02)
    result = await turn.interrupt()
    assert result.outcome is Outcome.INTERRUPTED
    assert result.spoken == ""
    assert rig.last.role == "user"
    assert not rig.history.in_agent_turn


async def test_interrupt_after_hang_up_does_not_clear(ctx):
    rig = Rig(ctx, [Reply(text("Hello."), delay_s=0.5)])
    turn = rig.turn()
    turn.start()
    await asyncio.sleep(0.02)
    await turn.interrupt(clear_audio=False)
    assert rig.adapter.clears == 0


async def test_interrupt_after_the_turn_finished_returns_its_result(ctx):
    rig = Rig(ctx, [Reply(text("All done."))])
    turn = rig.turn()
    turn.start()
    first = await turn.result()
    assert await turn.interrupt() == first
    assert rig.adapter.clears == 0
