"""One call, from the greeting to the hang-up (design §5, §6.1).

    session = Session(deps, settings, start)
    adapter.attach(session)          # the adapter calls on_audio / on_mark_played / on_session_end
    reason = await session.run()     # returns once the session is over and stored

Everything that can change the call's state goes through one inbox and is handled in order
by `run()`: speech-to-text events, a turn finishing, the caller hanging up. Timers (silence,
time limit) are deadlines checked by the same loop. So there are no races between, say, a
barge-in and a turn ending at the same moment.

    idle        no agent turn: waiting for the caller; silence timers run
    agent turn  a Turn is thinking or speaking; the caller's words are checked for barge-in

Fixed lines (the greeting, "Are you still there?", the silence goodbye) are spoken as turns
too, from a one-line scripted model, so they get the same marks, barge-in and history.

After a goodbye the line stays open for `goodbye_hold_s` before the session hangs up, so a
caller who remembers one more thing isn't cut off: real words in that time carry the call
on, and silence or a "mm-hm" lets it end. Most callers hang up themselves first. An abusive
caller is hung up on straight away.

Turn rows are written as they happen, so a crash loses at most the summary.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

from voice_agent.core.bargein import Heard, classify
from voice_agent.core.clips import Clip, Clips
from voice_agent.core.greeting import GreetingTemplates, build_opening
from voice_agent.core.history import History
from voice_agent.core.interfaces import (
    Capability,
    ChannelAdapter,
    Done,
    LanguageModel,
    LlmEvent,
    Message,
    Notifier,
    SessionRow,
    SessionStart,
    SpeechToText,
    Storage,
    SttStream,
    SystemBlock,
    TextDelta,
    TextToSpeech,
    ToolDef,
    TurnEnded,
    TurnRow,
    TurnStarted,
    Words,
)
from voice_agent.core.prompt import CallContext, Prompt
from voice_agent.core.tools import ToolContext, tool_defs
from voice_agent.core.turn import Outcome, Turn, TurnDeps, TurnResult, TurnSettings

log = logging.getLogger(__name__)

STILL_THERE = "Are you still there?"
SILENCE_GOODBYE = "I'll let you go for now. Feel free to call back anytime. Bye!"
# Added to the caller's side of the history; instructions.md says what to do with it.
LIMIT_NOTE = "(Note: the call has reached its time limit.)"
SUMMARY_MAX_AGE = timedelta(days=90)  # older summaries aren't mentioned (design §5)

# End reasons, besides the end_session tool's own (goodbye, abuse, limit).
CALLER_HANGUP = "caller_hangup"
SILENCE = "silence"
LIMIT = "limit"
ERROR = "error"
# Ended at once, with no goodbye hold.
_NO_HOLD = frozenset({"abuse"})


@dataclass(frozen=True, slots=True)
class SessionSettings:
    turn: TurnSettings
    silence_prompt_s: float  # silence before "Are you still there?"
    silence_hangup_s: float  # further silence before hanging up
    max_session_s: float  # the model is told to wrap up
    limit_grace_s: float = 60.0  # then hang up regardless
    goodbye_hold_s: float = 3.0  # line kept open after a goodbye, for "oh, one more thing"
    record_transcripts: bool = True


@dataclass(frozen=True, slots=True)
class SessionDeps:
    """Shared by every session; built once in app/main.py."""

    storage: Storage
    notifier: Notifier
    llm: LanguageModel
    stt: SpeechToText
    tts: TextToSpeech
    adapter: ChannelAdapter
    clips: Clips
    prompt: Prompt
    templates: GreetingTemplates
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)


# Inbox items besides SttEvents.
@dataclass(frozen=True, slots=True)
class _TurnDone:
    turn: Turn
    result: TurnResult


@dataclass(frozen=True, slots=True)
class _HungUp:
    reason: str


@dataclass(frozen=True, slots=True)
class _SttClosed:
    pass


_Item = TurnStarted | Words | TurnEnded | _TurnDone | _HungUp | _SttClosed


class Session:
    def __init__(self, deps: SessionDeps, settings: SessionSettings, start: SessionStart):
        self._deps = deps
        self._settings = settings
        self._start = start
        self._inbox: asyncio.Queue[_Item] = asyncio.Queue()
        self._history = History()
        self._stt: SttStream | None = None
        self._end_reason: str | None = None

        # Set up in run().
        self._session_id = 0
        self._system: list[SystemBlock] = []
        self._tools: list[ToolDef] = []
        self._tool_ctx: ToolContext | None = None

        # Turns.
        self._turn: Turn | None = None
        self._turn_started_at: datetime | None = None
        self._turn_count = 0
        self._seq = 0
        self._last_filler: Clip | None = None
        self._transcript: list[str] = []  # for the summary
        self._end_after_turn: str | None = None  # hang up once this turn has played
        self._holding: str | None = None  # the end reason waiting out the goodbye hold
        self._hang_up_at = 0.0

        # Timers, in loop time.
        self._loop_start = 0.0
        self._idle_since = 0.0
        self._caller_speaking = False
        self._caller_started_at: datetime | None = None
        self._prompted = False  # "Are you still there?" said in this silence
        self._limit_reached = False  # max_session_s has passed
        self._limit_noted = False  # LIMIT_NOTE is in the history

    # -- SessionCallbacks (called by the adapter) -----------------------------------

    async def on_audio(self, pcm: bytes) -> None:
        if self._stt is not None and self._end_reason is None:
            await self._stt.send(pcm)

    async def on_mark_played(self, name: str) -> None:
        if self._turn is not None:
            self._turn.mark_played(name)

    async def on_session_end(self, reason: str) -> None:
        self._inbox.put_nowait(_HungUp(reason))

    # -- the session -------------------------------------------------------------------

    async def run(self) -> str:
        """Run the call to the end. Returns the end reason."""
        loop = asyncio.get_running_loop()
        self._loop_start = self._idle_since = loop.time()
        pump: asyncio.Task[None] | None = None
        prewarm: asyncio.Task[None] | None = None
        try:
            opening = await self._set_up()
            prewarm = asyncio.create_task(self._prewarm())
            pump = asyncio.create_task(self._pump_stt())
            self._speak(opening)
            while self._end_reason is None:
                try:
                    async with asyncio.timeout(self._next_timer_in()):
                        item = await self._inbox.get()
                except TimeoutError:
                    await self._on_timer()
                    continue
                await self._handle(item)
        except Exception as e:
            log.exception("session %s crashed", self._session_id)
            await self._fatal(repr(e))
        finally:
            if prewarm is not None:
                prewarm.cancel()  # a very short call: no need to finish warming
                await asyncio.wait([prewarm])
            await self._finish(pump)
        return self._end_reason or ERROR

    async def _prewarm(self) -> None:
        """Warm the model's prompt cache while the greeting plays, so the first answer
        doesn't pay for it. A failure only means a slower first answer."""
        try:
            await self._deps.llm.prewarm(self._system, self._tools)
        except Exception:
            log.warning("prewarm failed (session %s)", self._session_id, exc_info=True)

    async def _set_up(self) -> str:
        """Find the caller, create the session, build the prompt. Returns the opening."""
        d, start = self._deps, self._start
        now = d.clock()
        caller = await d.storage.get_or_create_caller(start.channel, start.caller_address)
        last = await d.storage.last_session(caller.id)
        recent = None
        if last and last.summary and now - last.started_at <= SUMMARY_MAX_AGE:
            recent = last
        self._session_id = await d.storage.create_session(caller.id, start)

        # Warm intro (design §8.4) comes with build step 7; until then the model just
        # has the last summary in its context.
        opening = build_opening(
            caller, d.templates, last_summary=recent.summary if recent else None, warm_intro=False
        )
        self._system = d.prompt.system(
            CallContext(
                today=now.date(),
                caller_name=caller.name,
                anonymous=start.caller_address is None,
                last_summary=recent.summary if recent else None,
                last_call_date=recent.started_at.date() if recent else None,
            )
        )
        self._tools = tool_defs(Capability.END_SESSION in d.adapter.capabilities)
        row = SessionRow(
            self._session_id, caller.id, start.channel, start.channel_ref, now, None, None, None
        )
        self._tool_ctx = ToolContext(d.storage, d.notifier, row, caller, clock=d.clock)
        self._stt = await d.stt.open(start.sample_rate)
        return opening.text

    async def _pump_stt(self) -> None:
        assert self._stt is not None
        try:
            async for event in self._stt.events():
                self._inbox.put_nowait(event)
        except Exception:
            log.exception("speech-to-text failed (session %s)", self._session_id)
        self._inbox.put_nowait(_SttClosed())

    # -- events ---------------------------------------------------------------------------

    async def _handle(self, item: _Item) -> None:
        match item:
            case TurnStarted():
                self._caller_speaking = True
                self._caller_started_at = self._deps.clock()
            case Words(text=text):
                if self._turn is not None and classify(text) is Heard.SPEECH:
                    await self._interrupt()
            case TurnEnded(transcript=transcript):
                await self._caller_turn_ended(transcript)
            case _TurnDone(turn=turn, result=result):
                if turn is self._turn:  # else: already settled by interrupt()
                    self._turn = None
                    await self._settle(result)
            case _HungUp(reason=reason):
                self._end_reason = reason
            case _SttClosed():
                if self._end_reason is None:
                    await self._fatal("speech-to-text stream closed")

    async def _caller_turn_ended(self, transcript: str) -> None:
        self._caller_speaking = False
        self._idle_since = self._now()
        heard = classify(transcript)
        if self._holding is not None:
            if heard is not Heard.SPEECH:  # a cough or "okay" after the goodbye
                self._hang_up_at = self._now() + self._settings.goodbye_hold_s
                return
            log.info("session %s: caller spoke after the goodbye; carrying on", self._session_id)
            self._cancel_ending()
        if heard is Heard.NOTHING:
            return
        if self._turn is not None:
            if heard is Heard.BACKCHANNEL:
                return  # "mm-hm" while the agent talks: not a turn to answer
            await self._interrupt()

        started = self._caller_started_at or self._deps.clock()
        self._caller_started_at = None
        self._prompted = False
        self._history.add_caller(transcript)
        await self._store(TurnRow(0, "caller", transcript, started, self._deps.clock()))
        self._transcript.append(f"Caller: {transcript}")
        self._add_limit_note_if_due()
        self._reply()

    async def _interrupt(self) -> None:
        turn, self._turn = self._turn, None
        if turn is None:
            return
        result = await turn.interrupt()
        if result.outcome is Outcome.INTERRUPTED:
            self._cancel_ending()  # the caller cut in on a goodbye: keep talking
        await self._settle(result)

    def _cancel_ending(self) -> None:
        self._end_after_turn = None
        self._holding = None
        if self._tool_ctx is not None:
            self._tool_ctx.end_reason = None

    async def _settle(self, result: TurnResult) -> None:
        """Store a finished turn and decide what happens next."""
        self._last_filler = result.filler or self._last_filler
        for run in result.tool_runs:
            text = f"{run.call.name} {json.dumps(run.call.input)}"
            await self._store(TurnRow(0, "tool", text, self._deps.clock()))
        if result.spoken:
            await self._store(
                TurnRow(
                    0,
                    "agent",
                    result.spoken,
                    self._turn_started_at or self._deps.clock(),
                    self._deps.clock(),
                    interrupted=result.outcome is Outcome.INTERRUPTED,
                    latency_ms=result.latency_ms,
                )
            )
            self._transcript.append(f"Agent: {result.spoken}")
        self._idle_since = self._now()

        if result.outcome is Outcome.FAILED:
            await self._alert(result.error or "unknown error")
            self._end_reason = ERROR
        elif result.outcome is Outcome.COMPLETED and (
            reason := self._end_after_turn or result.end_reason
        ):
            self._end_after_turn = None
            self._end_or_hold(reason)
        elif self._add_limit_note_if_due():
            self._reply()  # the limit passed mid-turn: let the model wrap up now

    def _end_or_hold(self, reason: str) -> None:
        if reason in _NO_HOLD or self._settings.goodbye_hold_s <= 0:
            self._end_reason = reason
            return
        self._holding = reason
        self._hang_up_at = self._now() + self._settings.goodbye_hold_s

    # -- timers ------------------------------------------------------------------------------

    def _next_timer_in(self) -> float:
        s = self._settings
        deadlines = [self._loop_start + s.max_session_s + s.limit_grace_s]
        if self._holding is not None:  # only the hold, and the hard limit, apply
            if not self._caller_speaking:
                deadlines.append(self._hang_up_at)
            return max(0.0, min(deadlines) - self._now())
        if not self._limit_reached:
            deadlines.append(self._loop_start + s.max_session_s)
        if self._turn is None and not self._caller_speaking:
            wait = s.silence_hangup_s if self._prompted else s.silence_prompt_s
            deadlines.append(self._idle_since + wait)
        return max(0.0, min(deadlines) - self._now())

    async def _on_timer(self) -> None:
        s, now = self._settings, self._now()
        if now >= self._loop_start + s.max_session_s + s.limit_grace_s:
            log.info("session %s hit the hard time limit", self._session_id)
            self._end_reason = LIMIT
            return
        if self._holding is not None:
            if not self._caller_speaking and now >= self._hang_up_at:
                self._end_reason = self._holding
            return
        if not self._limit_reached and now >= self._loop_start + s.max_session_s:
            self._limit_reached = True
            if self._turn is None and not self._caller_speaking:
                self._add_limit_note_if_due()
                self._reply()
            return  # otherwise the note goes in after the current turn
        if self._turn is None and not self._caller_speaking:
            if not self._prompted:
                self._prompted = True
                self._speak(STILL_THERE)
            else:
                self._end_after_turn = SILENCE
                self._speak(SILENCE_GOODBYE)

    def _add_limit_note_if_due(self) -> bool:
        if self._limit_noted or not self._limit_reached:
            return False
        self._limit_noted = True
        self._history.add_caller(LIMIT_NOTE)
        return True

    # -- starting turns -------------------------------------------------------------------------

    def _reply(self) -> None:
        """An agent turn answered by the model."""
        self._start_turn(self._deps.llm, self._settings.turn)

    def _speak(self, line: str) -> None:
        """A fixed line, with no LLM call and no filler."""
        settings = replace(self._settings.turn, filler_after_ms=10**9)
        self._start_turn(_Scripted(line), settings)

    def _start_turn(self, llm: LanguageModel, settings: TurnSettings) -> None:
        assert self._tool_ctx is not None and self._turn is None
        self._turn_count += 1
        deps = TurnDeps(
            llm=llm,
            tts=self._deps.tts,
            adapter=self._deps.adapter,
            clips=self._deps.clips,
            tools=self._tools,
            tool_ctx=self._tool_ctx,
        )
        turn = Turn(
            deps,
            settings,
            self._history,
            self._system,
            self._turn_count,
            last_filler=self._last_filler,
        )
        self._turn = turn
        self._turn_started_at = self._deps.clock()
        turn.start()
        turn.on_done(lambda result: self._inbox.put_nowait(_TurnDone(turn, result)))

    # -- ending ------------------------------------------------------------------------------

    async def _fatal(self, error: str) -> None:
        """A failure outside a turn: apologise if we can, alert, and end."""
        self._end_reason = ERROR
        if self._turn is not None:
            turn, self._turn = self._turn, None
            await turn.interrupt()
        adapter = self._deps.adapter
        await _quietly(adapter.clear_audio())
        await _quietly(adapter.send_audio(self._deps.clips.apology.pcm))
        await asyncio.sleep(self._deps.clips.apology.duration_ms / 1000)
        await self._alert(error)

    async def _finish(self, pump: asyncio.Task[None] | None) -> None:
        reason = self._end_reason or ERROR
        self._end_reason = reason
        hung_up = reason == CALLER_HANGUP
        if self._turn is not None:
            turn, self._turn = self._turn, None
            await _quietly(self._settle_last(turn, clear_audio=not hung_up))
        if self._stt is not None:
            await _quietly(self._stt.close())
        if pump is not None:
            pump.cancel()
            await asyncio.wait([pump])
        adapter = self._deps.adapter
        if not hung_up and Capability.END_SESSION in adapter.capabilities:
            await _quietly(adapter.end_session())
        if self._session_id:
            summary = await self._summarize()
            await _quietly(self._deps.storage.end_session(self._session_id, reason, summary))
        log.info("session %s ended: %s", self._session_id, reason)

    async def _settle_last(self, turn: Turn, clear_audio: bool) -> None:
        """Stop and store the turn still running at the end, keeping the end reason."""
        reason = self._end_reason
        await self._settle(await turn.interrupt(clear_audio=clear_audio))
        self._end_reason = reason

    async def _summarize(self) -> str | None:
        if not any(line.startswith("Caller:") for line in self._transcript):
            return None
        try:
            return await self._deps.llm.summarize("\n".join(self._transcript))
        except Exception:
            log.exception("summary failed (session %s)", self._session_id)
            return None

    # -- helpers --------------------------------------------------------------------------------

    async def _store(self, row: TurnRow) -> None:
        if not self._settings.record_transcripts:
            return
        self._seq += 1
        await _quietly(self._deps.storage.add_turn(self._session_id, replace(row, seq=self._seq)))

    async def _alert(self, error: str) -> None:
        body = f"Session {self._session_id} ended with an error: {error}"
        await _quietly(self._deps.notifier.alert("Voice agent: call failed", body))

    def _now(self) -> float:
        return asyncio.get_running_loop().time()


class _Scripted:
    """A model that says one fixed line. Used for lines that must never be generated."""

    def __init__(self, line: str) -> None:
        self._line = line

    async def stream_reply(
        self, system: list[SystemBlock], history: list[Message], tools: list[ToolDef]
    ) -> AsyncIterator[LlmEvent]:
        yield TextDelta(self._line)
        yield Done("end_turn")

    async def summarize(self, transcript: str) -> str:
        raise NotImplementedError

    async def prewarm(self, system: list[SystemBlock], tools: list[ToolDef]) -> None:
        pass


async def _quietly(op: Awaitable[object]) -> None:
    """Await `op`, logging instead of raising: storage and phone hiccups must not end a
    call, and cleanup must run to the end."""
    try:
        await op
    except Exception:
        log.warning("step failed", exc_info=True)
