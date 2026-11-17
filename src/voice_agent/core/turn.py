"""One agent turn: from the caller finishing their turn to the reply being heard, or cut off.

    turn = Turn(deps, settings, history, system, number=3)
    turn.start()                     # the caller's words are already in `history`
    turn.mark_played(name)           # the session forwards every mark echo from the adapter
    result = await turn.result()     # the reply finished playing, or a provider failed
    result = await turn.interrupt()  # or: the caller barged in / hung up

Three things run at once, so the caller hears sentence 1 while the model writes sentence 3:

    think:   LLM stream -> sentence splitter -> speakable() -> tts.say(); tool calls run
             between requests
    play:    TTS audio  -> adapter.send_audio(), then a mark after each chunk's words
    filler:  no reply audio after `filler_after_ms` -> play one filler clip

Marks are named `t{number}.w{n}`, where n is the words sent so far, plus `t{number}.end` after
the last chunk. When a mark echoes back, the caller has heard n words, so an interruption
trims the history to exactly that (design §6.3). The turn number keeps a late echo from an
earlier turn from counting here.

Failures (design §8.5): a failed LLM request is retried once, but only if that request hasn't
said anything yet, since a retry would repeat what the caller already heard. A second
failure, or any TTS or adapter failure, plays the apology clip and returns FAILED; the session
then alerts the owner and hangs up.

The turn doesn't touch storage. It returns a TurnResult and the session writes the rows, so
one place owns turn numbering.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from voice_agent.core.clips import Clip, Clips
from voice_agent.core.history import History
from voice_agent.core.interfaces import (
    ChannelAdapter,
    Done,
    LanguageModel,
    ProviderError,
    Refused,
    SystemBlock,
    TextDelta,
    TextToSpeech,
    ToolCall,
    ToolDef,
    ToolResultPart,
    ToolUsePart,
    TtsStream,
)
from voice_agent.core.sentences import SentenceSplitter
from voice_agent.core.speech import speakable
from voice_agent.core.tools import ToolContext, run_tool

log = logging.getLogger(__name__)

# Spoken when the model refuses. Fixed text, so a refusal can never produce something odd.
REFUSAL_LINE = "Sorry, I can't help with that one, but I can take a message for the real me."
# Requests that may run tools in one turn. Normal turns use 0 or 1; this stops a model stuck
# in a loop from keeping the caller waiting.
MAX_TOOL_ROUNDS = 3
TOOL_LIMIT_RESULT = "Too many tool calls this turn. Answer the caller without tools."


class Outcome(StrEnum):
    COMPLETED = "completed"  # the whole reply played (or the adapter stopped echoing marks)
    INTERRUPTED = "interrupted"  # cut off by interrupt()
    FAILED = "failed"  # a provider failed; the apology clip was played


@dataclass(frozen=True, slots=True)
class TurnSettings:
    sample_rate: int
    voice_id: str
    filler_after_ms: int
    provider_timeout_s: float  # max wait for each LLM event
    mark_grace_ms: int = 2000  # extra wait for the final mark after the audio should be done


@dataclass(frozen=True, slots=True)
class TurnDeps:
    """Everything that stays the same across a session's turns."""

    llm: LanguageModel
    tts: TextToSpeech
    adapter: ChannelAdapter
    clips: Clips
    tools: list[ToolDef]
    tool_ctx: ToolContext


@dataclass(frozen=True, slots=True)
class ToolRun:
    call: ToolUsePart
    result: ToolResultPart


@dataclass(frozen=True, slots=True)
class TurnResult:
    outcome: Outcome
    spoken: str  # what the caller heard, for the stored transcript
    latency_ms: int | None  # turn start to first reply audio; None if none was sent
    tool_runs: tuple[ToolRun, ...]
    filler: Clip | None  # the filler played, so the next turn can pick a different one
    end_reason: str | None  # set when the model called end_session
    error: str | None = None  # what failed, for FAILED


class Turn:
    def __init__(
        self,
        deps: TurnDeps,
        settings: TurnSettings,
        history: History,
        system: list[SystemBlock],
        number: int,
        *,
        last_filler: Clip | None = None,
    ) -> None:
        self._deps = deps
        self._settings = settings
        self._history = history
        self._system = system
        self._prefix = f"t{number}."
        self._last_filler = last_filler

        self._task: asyncio.Task[TurnResult] | None = None
        self._tts: TtsStream | None = None
        self._send_lock = asyncio.Lock()  # filler and reply audio must never interleave
        self._first_audio = asyncio.Event()
        self._end_played = asyncio.Event()
        self._end_mark: str | None = None
        self._started = 0.0
        self._playback_until = 0.0  # loop time when the audio sent so far should finish
        self._latency_ms: int | None = None
        self._words_sent = 0
        self._words_heard = 0
        self._said_this_request = False
        self._tool_runs: list[ToolRun] = []
        self._filler: Clip | None = None
        self._spoken: str | None = None  # set once the history turn is closed

    # -- called by the session --------------------------------------------------

    def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("turn already started")
        loop = asyncio.get_running_loop()
        self._started = self._playback_until = loop.time()
        self._history.begin_agent_turn()
        self._task = asyncio.create_task(self._run())

    async def result(self) -> TurnResult:
        return await self._require_task()

    def on_done(self, callback: Callable[[TurnResult], None]) -> None:
        """Call `callback` when the turn finishes by itself. Not called after interrupt(),
        which returns its result directly. Lets the session hear about the turn without
        awaiting its task (cancelling an awaiter would cancel the turn mid-cleanup)."""

        def done(task: asyncio.Task[TurnResult]) -> None:
            if task.cancelled():
                return
            if (error := task.exception()) is not None:  # a bug in the turn itself
                log.error("turn crashed: %r", error, exc_info=error)
                spoken = self._close_history(self._words_heard)
                callback(self._result(Outcome.FAILED, spoken, error=repr(error)))
                return
            callback(task.result())

        self._require_task().add_done_callback(done)

    async def interrupt(self, clear_audio: bool = True) -> TurnResult:
        """Stop now and keep only what the caller heard. Pass clear_audio=False when the
        caller has hung up and there's no phone to clear."""
        task = self._require_task()
        if not task.done():
            task.cancel()
            await asyncio.wait([task])
        if not task.cancelled():
            return task.result()  # it had already finished
        await self._cancel_tts()
        if clear_audio:
            await self._quietly(self._deps.adapter.clear_audio())
        return self._result(Outcome.INTERRUPTED, self._close_history(self._words_heard))

    def mark_played(self, name: str) -> None:
        if not name.startswith(self._prefix):
            return  # an earlier turn's mark
        if name == self._end_mark:
            self._end_played.set()
        elif name.startswith(self._prefix + "w"):
            self._words_heard = max(self._words_heard, int(name.removeprefix(self._prefix + "w")))

    # -- the turn -----------------------------------------------------------------

    async def _run(self) -> TurnResult:
        filler = asyncio.create_task(self._play_filler())
        error: BaseException | None = None
        try:
            self._tts = await self._deps.tts.open(
                self._settings.sample_rate, self._settings.voice_id
            )
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._think(self._tts))
                tg.create_task(self._play(self._tts))
        except Exception as e:
            error = _first_error(e)
        finally:
            filler.cancel()
            await asyncio.wait([filler])

        if error is not None:
            log.error("turn failed: %r", error, exc_info=error)
            return await self._fail(error)
        if self._end_mark is not None:
            await self._wait_for_playback()
        return self._result(Outcome.COMPLETED, self._close_history(None))

    async def _think(self, tts: TtsStream) -> None:
        retry_left = True
        rounds = 0
        while True:
            try:
                calls, refused = await self._request(tts)
            except Exception as e:
                if retry_left and _retryable(e) and not self._said_this_request:
                    retry_left = False
                    log.warning("LLM request failed, retrying once: %r", e)
                    continue
                raise
            if refused:
                await self._say(tts, REFUSAL_LINE)
                break
            if not calls:
                break
            rounds += 1
            await self._run_tools(calls, over_limit=rounds > MAX_TOOL_ROUNDS)
            if rounds > MAX_TOOL_ROUNDS or self._deps.tool_ctx.end_reason:
                break  # end_session comes after the closing line: nothing more to say
        await tts.finish()

    async def _request(self, tts: TtsStream) -> tuple[list[ToolCall], bool]:
        """One LLM request. Returns its tool calls and whether the model refused."""
        self._said_this_request = False
        splitter = SentenceSplitter()
        calls: list[ToolCall] = []
        stream = self._deps.llm.stream_reply(self._system, self._history.messages, self._deps.tools)
        events = aiter(stream)
        try:
            while True:
                try:
                    async with asyncio.timeout(self._settings.provider_timeout_s):
                        event = await anext(events)
                except StopAsyncIteration:
                    break
                match event:
                    case TextDelta(text=text):
                        for sentence in splitter.feed(text):
                            await self._say(tts, sentence)
                    case ToolCall():
                        calls.append(event)
                    case Refused():
                        return [], True
                    case Done():
                        break
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()
        if rest := splitter.flush():
            await self._say(tts, rest)
        return calls, False

    async def _say(self, tts: TtsStream, sentence: str) -> None:
        # The history keeps what's spoken (spelled out), so the words heard, counted from
        # the audio, match the words in the history when a reply is cut off.
        spoken = speakable(sentence)
        if not spoken.strip():
            return
        self._history.add_agent_text(spoken)
        self._said_this_request = True
        await tts.say(spoken)

    async def _run_tools(self, calls: list[ToolCall], *, over_limit: bool) -> None:
        # All calls first, then all results: the API wants every result for one assistant
        # message in the next user message.
        results = []
        for call in calls:
            if over_limit:
                result = ToolResultPart(call.id, TOOL_LIMIT_RESULT, is_error=True)
            else:
                result = await run_tool(call, self._deps.tool_ctx, self._deps.tools)
            use = ToolUsePart(call.id, call.name, call.input)
            self._history.add_tool_call(use)
            self._tool_runs.append(ToolRun(use, result))
            results.append(result)
        for result in results:
            self._history.add_tool_result(result)

    async def _play(self, tts: TtsStream) -> None:
        async for chunk in tts.audio():
            if self._latency_ms is None:
                now = asyncio.get_running_loop().time()
                self._latency_ms = round((now - self._started) * 1000)
                self._first_audio.set()
            await self._send(chunk.pcm)
            if chunk.words:
                self._words_sent += len(chunk.words)
                await self._deps.adapter.mark(f"{self._prefix}w{self._words_sent}")
        if self._latency_ms is not None:
            self._end_mark = f"{self._prefix}end"
            await self._deps.adapter.mark(self._end_mark)

    async def _play_filler(self) -> None:
        try:
            async with asyncio.timeout(self._settings.filler_after_ms / 1000):
                await self._first_audio.wait()
            return
        except TimeoutError:
            pass
        self._filler = self._deps.clips.filler(self._last_filler)
        await self._quietly(self._send(self._filler.pcm))

    async def _send(self, pcm: bytes) -> None:
        async with self._send_lock:
            await self._deps.adapter.send_audio(pcm)
        now = asyncio.get_running_loop().time()
        self._playback_until = max(now, self._playback_until) + _duration_s(
            pcm, self._settings.sample_rate
        )

    async def _wait_for_playback(self) -> None:
        """Wait for the final mark, but not forever: an adapter that never echoes marks
        (or a lost echo) must not hang the call."""
        now = asyncio.get_running_loop().time()
        timeout = max(0.0, self._playback_until - now) + self._settings.mark_grace_ms / 1000
        try:
            async with asyncio.timeout(timeout):
                await self._end_played.wait()
        except TimeoutError:
            log.warning("final mark %s never came back; assuming it played", self._end_mark)

    async def _fail(self, error: BaseException) -> TurnResult:
        await self._cancel_tts()
        spoken = self._close_history(self._words_heard)
        adapter = self._deps.adapter
        await self._quietly(adapter.clear_audio())
        self._playback_until = asyncio.get_running_loop().time()
        self._end_played.clear()
        self._end_mark = f"{self._prefix}apology"
        if await self._quietly(self._send(self._deps.clips.apology.pcm)):
            await self._quietly(adapter.mark(self._end_mark))
            await self._wait_for_playback()
        return self._result(Outcome.FAILED, spoken, error=repr(error))

    # -- helpers --------------------------------------------------------------------

    def _close_history(self, words_heard: int | None) -> str:
        if self._spoken is None:
            self._spoken = self._history.end_agent_turn(words_heard)
        return self._spoken

    async def _cancel_tts(self) -> None:
        if self._tts is not None:
            await self._quietly(self._tts.cancel())

    async def _quietly(self, op: Awaitable[object]) -> bool:
        """Await `op`, logging instead of raising: used on paths that are already
        cleaning up, where the phone may be gone."""
        try:
            await op
        except Exception:
            log.warning("cleanup step failed", exc_info=True)
            return False
        return True

    def _result(self, outcome: Outcome, spoken: str, error: str | None = None) -> TurnResult:
        return TurnResult(
            outcome=outcome,
            spoken=spoken,
            latency_ms=self._latency_ms,
            tool_runs=tuple(self._tool_runs),
            filler=self._filler,
            end_reason=self._deps.tool_ctx.end_reason,
            error=error,
        )

    def _require_task(self) -> asyncio.Task[TurnResult]:
        if self._task is None:
            raise RuntimeError("turn not started; call start() first")
        return self._task


def _retryable(error: BaseException) -> bool:
    if isinstance(error, ProviderError):
        return error.retryable
    return isinstance(error, TimeoutError)  # no event within provider_timeout_s


def _first_error(error: BaseException) -> BaseException:
    """TaskGroup wraps failures in an ExceptionGroup; report the one that started it."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return error


def _duration_s(pcm: bytes, sample_rate: int) -> float:
    return len(pcm) / (sample_rate * 2)  # PCM16 mono
