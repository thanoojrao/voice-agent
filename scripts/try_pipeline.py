"""Talk to the real providers with no phone: Flux -> Claude -> Cartesia, timed.

    uv run python scripts/try_pipeline.py --voice <cartesia voice id> \
        --say "What are you working on these days?" "Can you tell me more?"
    uv run python scripts/try_pipeline.py --voice <id> --wav question.wav --record

Each question is streamed to Flux like phone audio (20 ms frames, in real time, silence in
between), the transcript goes to Claude with the real prompt and tools, and the reply is
spoken by Cartesia a sentence at a time. Every reply is saved as a WAV to listen to, and
each turn prints where the time went. Two or more questions show the prompt cache: the
first writes it, the rest should read it.

--record saves every message the providers send and receive (audio left out) to
messages.jsonl, to check their real shapes against the docs and use them as test data.
--probe-cancel and --probe-gap try two things Cartesia's docs leave open.

This calls the real APIs and costs a fraction of a cent per run. Keys come from .env.
The providers are called directly rather than through the turn engine: this step measures
the providers; the full engine runs on a real call with the Twilio adapter (step 4).
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import audioop

from voice_agent.app.config import Settings, load_agent_config
from voice_agent.app.constants import MAX_ANSWER_SENTENCES
from voice_agent.core.history import History
from voice_agent.core.interfaces import (
    Done,
    ProviderError,
    SystemBlock,
    TextDelta,
    ToolCall,
    ToolDef,
    ToolResultPart,
    ToolUsePart,
    TurnEnded,
)
from voice_agent.core.prompt import CallContext, Prompt, load_instructions, load_knowledge_base
from voice_agent.core.sentences import SentenceSplitter
from voice_agent.core.speech import speakable
from voice_agent.core.tools import tool_defs
from voice_agent.providers import cartesia_tts, deepgram_flux
from voice_agent.providers.anthropic_llm import AnthropicLLM, new_client
from voice_agent.providers.cartesia_tts import CartesiaSettings, CartesiaStream, CartesiaTTS
from voice_agent.providers.deepgram_flux import DeepgramFlux, FluxStream

RATE = 8000  # what the phone sends
FRAME_MS = 20  # Twilio sends 20 ms frames
FRAME_BYTES = RATE * 2 * FRAME_MS // 1000  # PCM16 mono
LEAD_IN_S = 0.5  # silence before each question, like a caller drawing breath
TURN_TIMEOUT_S = 15.0  # after the question ends, how long to wait for Flux to end the turn
PROBE_TEXT = (
    "Here is a longer answer so there is something to cancel. It goes on for a few "
    "sentences. Each one adds a little more audio. By now the first audio has arrived."
)

# -- audio helpers (pure; unit tested) ------------------------------------------------------


def to_phone_pcm(raw: bytes, *, rate: int, width: int, channels: int) -> bytes:
    """Convert raw WAV frames to what the phone sends: PCM16, mono, 8 kHz."""
    if width != 2:
        raw = audioop.lin2lin(raw, width, 2)
    if channels == 2:
        raw = audioop.tomono(raw, 2, 0.5, 0.5)
    elif channels != 1:
        raise ValueError(f"unsupported channel count: {channels}")
    if rate != RATE:
        raw, _ = audioop.ratecv(raw, 2, 1, rate, RATE, None)
    return raw


def read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
        return to_phone_pcm(
            raw, rate=w.getframerate(), width=w.getsampwidth(), channels=w.getnchannels()
        )


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


def frames(pcm: bytes) -> list[bytes]:
    """Split into 20 ms frames, padding the last one with silence."""
    out = [pcm[i : i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]
    if out and len(out[-1]) < FRAME_BYTES:
        out[-1] = out[-1].ljust(FRAME_BYTES, b"\x00")
    return out


def silence(seconds: float) -> bytes:
    return b"\x00" * (int(RATE * seconds) * 2)


# -- the caller's microphone ---------------------------------------------------------------


class Mic:
    """Sends a 20 ms frame to Flux every 20 ms, like a phone line: the queued question
    while there is one, silence otherwise. Paced against the clock so it doesn't drift."""

    def __init__(self, stt: FluxStream) -> None:
        self._stt = stt
        self._frames: list[bytes] = []
        self._spoken: asyncio.Event = asyncio.Event()
        self.speech_ended_at = 0.0

    async def say(self, pcm: bytes) -> None:
        """Queue a question and wait until its last frame has gone out."""
        self._spoken.clear()
        self._frames = frames(silence(LEAD_IN_S) + pcm)
        await self._spoken.wait()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        start, sent = loop.time(), 0
        quiet = bytes(FRAME_BYTES)
        while True:
            if self._frames:
                await self._stt.send(self._frames.pop(0))
                if not self._frames:
                    self.speech_ended_at = loop.time()
                    self._spoken.set()
            else:
                await self._stt.send(quiet)
            sent += 1
            await asyncio.sleep(max(0.0, start + sent * FRAME_MS / 1000 - loop.time()))


# -- recording raw messages ----------------------------------------------------------------


@dataclass
class Recorder:
    """Writes every provider message to a JSONL file, with audio left out."""

    path: Path
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, provider: str, direction: str, message: str | bytes) -> None:
        if isinstance(message, bytes):
            return  # Flux audio frames: too many and not interesting
        try:
            body: Any = json.loads(message)
        except json.JSONDecodeError:
            body = message
        if isinstance(body, dict) and isinstance(body.get("data"), str):
            body = {**body, "data": f"<{len(body['data'])} base64 chars>"}
        entry = {"t": round(time.monotonic(), 3), "provider": provider, "dir": direction}
        self.entries.append({**entry, "message": body})
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.entries[-1]) + "\n")


class RecordingSocket:
    def __init__(self, socket: Any, provider: str, recorder: Recorder) -> None:
        self._socket, self._provider, self._recorder = socket, provider, recorder

    async def send(self, message: bytes | str) -> None:
        self._recorder.add(self._provider, "sent", message)
        await self._socket.send(message)

    async def recv(self) -> str | bytes:
        message = await self._socket.recv()
        self._recorder.add(self._provider, "recv", message)
        return message

    async def close(self) -> None:
        await self._socket.close()


Connect = Callable[..., Awaitable[Any]]


def recording(connect: Connect, provider: str, recorder: Recorder | None) -> Connect:
    if recorder is None:
        return connect

    async def connect_and_record(
        url: str, headers: dict[str, str], timeout_s: float, **options: Any
    ) -> Any:
        # The URL holds settings only; the keys are in the headers, which aren't written.
        recorder.add(provider, "connect", json.dumps({"url": url, **options}))
        socket = await connect(url, headers, timeout_s, **options)
        return RecordingSocket(socket, provider, recorder)

    return connect_and_record


# -- one turn ------------------------------------------------------------------------------


@dataclass
class TurnTimes:
    speech_ended: float
    turn_ended: float = 0.0
    first_text: float = 0.0
    first_sentence: float = 0.0
    first_audio: float = 0.0
    done: Done | None = None


async def speak_reply(
    llm: AnthropicLLM,
    tts: CartesiaStream,
    system: list[SystemBlock],
    history: History,
    tools: list[ToolDef],
    times: TurnTimes,
) -> bytes:
    """One LLM request, spoken as it streams. Returns the reply audio."""
    loop = asyncio.get_running_loop()
    audio = bytearray()

    async def think() -> None:
        splitter = SentenceSplitter()
        history.begin_agent_turn()
        calls: list[ToolCall] = []

        async def say(sentence: str) -> None:
            spoken = speakable(sentence)  # as the turn does: <spell> tags become letters
            if not spoken.strip():
                return
            if not times.first_sentence:
                times.first_sentence = loop.time()
            history.add_agent_text(spoken)
            print(f"    agent: {sentence}")
            if spoken != sentence:
                print(f"      spoken as: {spoken}")
            await tts.say(spoken)

        async for event in llm.stream_reply(system, history.messages, tools):
            match event:
                case TextDelta(text=text):
                    times.first_text = times.first_text or loop.time()
                    for sentence in splitter.feed(text):
                        await say(sentence)
                case ToolCall():
                    calls.append(event)
                case Done():
                    times.done = event
                case _:
                    print("    (refused)")
        if rest := splitter.flush():
            await say(rest)
        for call in calls:  # shown, not run: this script has no storage or email
            print(f"    tool call: {call.name} {json.dumps(call.input)}")
            history.add_tool_call(ToolUsePart(call.id, call.name, call.input))
        for call in calls:
            history.add_tool_result(ToolResultPart(call.id, "Done (dev script: not run)."))
        history.end_agent_turn()
        await tts.finish()

    async def listen() -> None:
        async for chunk in tts.audio():
            times.first_audio = times.first_audio or loop.time()
            audio.extend(chunk.pcm)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(think())
        tg.create_task(listen())
    return bytes(audio)


def report(n: int, t: TurnTimes) -> None:
    def ms(a: float, b: float) -> str:
        return f"{round((b - a) * 1000):>6} ms" if a and b else "     -   "

    print(f"  speech ended -> turn ended       {ms(t.speech_ended, t.turn_ended)}  (Flux)")
    print(f"  turn ended   -> first text       {ms(t.turn_ended, t.first_text)}  (Claude)")
    print(f"  turn ended   -> first sentence   {ms(t.turn_ended, t.first_sentence)}")
    print(f"  first sentence -> first audio    {ms(t.first_sentence, t.first_audio)}  (Cartesia)")
    print(
        f"  speech ended -> first audio      {ms(t.speech_ended, t.first_audio)}"
        "  <- what the caller waits (plus phone network)"
    )
    if t.done:
        d = t.done
        print(
            f"  tokens: in {d.input_tokens}, cache read {d.cache_read_tokens}, "
            f"out {d.output_tokens}"
        )


# -- probes --------------------------------------------------------------------------------


async def probe_cancel(tts: CartesiaTTS, voice: str, recorder: Recorder) -> None:
    """What does Cartesia send after a cancel, if anything?"""
    print("\nprobe: cancel after the first audio")
    stream = await tts.open(RATE, voice)
    await stream.say(PROBE_TEXT)
    await anext(aiter(stream.audio()))
    cancelled_at = time.monotonic()
    await stream.cancel()
    await asyncio.sleep(2)  # the recorder sees what our stream now ignores
    after = [
        e["message"].get("type")
        for e in recorder.entries
        if e["provider"] == "cartesia" and e["dir"] == "recv" and e["t"] >= cancelled_at
    ]
    print(f"  messages in the 2 s after cancel: {after or 'none'}")


async def probe_gap(tts_key: str, voice: str, recorder: Recorder, out: Path) -> None:
    """Does a context still work 1.5 s after its last audio? (Docs: it expires after 1 s.)"""
    print("\nprobe: a second sentence 1.5 s after the first one's audio, same context")
    settings = CartesiaSettings(context_gap_s=60.0)  # turn our own renewal off
    tts = CartesiaTTS(
        tts_key, settings, connect=recording(cartesia_tts._websocket, "cartesia", recorder)
    )
    stream = await tts.open(RATE, voice)
    audio = bytearray()
    first_done = asyncio.Event()

    async def listen() -> None:
        async for chunk in stream.audio():
            audio.extend(chunk.pcm)
            if len(audio) >= RATE:  # some of the first sentence has arrived
                first_done.set()

    listener = asyncio.create_task(listen())
    await stream.say("This is the first sentence.")
    await first_done.wait()
    await asyncio.sleep(1.5)
    try:
        await stream.say("And this is the second one, after a pause.")
        await stream.finish()
        await listener
        print(f"  ok: {len(audio) / (RATE * 2):.1f} s of audio in total")
    except ProviderError as e:
        print(f"  failed: {e}")
    write_wav(out / "probe_gap.wav", bytes(audio))
    await tts.aclose()


# -- main ----------------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> None:
    settings = Settings()
    settings.require("anthropic_api_key", "deepgram_api_key", "cartesia_api_key")
    assert settings.anthropic_api_key and settings.deepgram_api_key and settings.cartesia_api_key
    config = load_agent_config(settings.config_path)
    voice = args.voice or config.voice_id
    if not voice:
        sys.exit("No voice: pass --voice <Cartesia voice id> (or set voice_id in the config).")

    out = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    probing = args.probe_cancel or args.probe_gap
    recorder = Recorder(out / "messages.jsonl") if args.record or probing else None

    instructions = load_instructions(
        {
            "owner_name": config.owner_name,
            "tone": config.tone,
            "do_not_discuss": ", ".join(config.do_not_discuss),
            "max_sentences": str(MAX_ANSWER_SENTENCES),
        }
    )
    kb = load_knowledge_base(args.kb, config.featured_projects)
    prompt = Prompt(instructions, kb, config.status_stale_days)
    # The app will use a New York clock; for a dev run the local date is close enough.
    system = prompt.system(CallContext(date.today(), None, anonymous=False, last_summary=None))
    tools = tool_defs(can_end_session=True)

    cartesia_key = settings.cartesia_api_key.get_secret_value()
    tts = CartesiaTTS(
        cartesia_key,
        CartesiaSettings(pronunciations=config.pronunciations),
        connect=recording(cartesia_tts._websocket, "cartesia", recorder),
    )
    client = new_client(settings.anthropic_api_key.get_secret_value(), timeout_s=10.0)
    llm = AnthropicLLM(client, config.llm_model)
    flux = DeepgramFlux(
        settings.deepgram_api_key.get_secret_value(),
        connect=recording(deepgram_flux._websocket, "deepgram", recorder),
    )

    try:
        if args.wav:
            questions = [(str(p), read_wav(p)) for p in args.wav]
        else:  # Cartesia reads the questions out (not timed)
            questions = []
            for text in args.say:
                stream = await tts.open(RATE, args.caller_voice or voice)
                await stream.say(text)
                await stream.finish()
                questions.append((text, b"".join([c.pcm async for c in stream.audio()])))

        if not args.cold:  # as a real call does while the greeting plays
            await llm.prewarm(system, tools)
        history = History()
        history.add_agent(config.greeting.format(owner_name=config.owner_name))
        stt = await flux.open(RATE)
        transcripts: asyncio.Queue[str] = asyncio.Queue()

        async def hear() -> None:
            async for event in stt.events():
                if isinstance(event, TurnEnded):
                    transcripts.put_nowait(event.transcript)

        mic = Mic(stt)
        background = [asyncio.create_task(mic.run()), asyncio.create_task(hear())]
        try:
            loop = asyncio.get_running_loop()
            for n, (label, pcm) in enumerate(questions, start=1):
                print(f"\nturn {n}: {label}")
                await mic.say(pcm)
                times = TurnTimes(speech_ended=mic.speech_ended_at)
                async with asyncio.timeout(TURN_TIMEOUT_S):
                    transcript = await transcripts.get()
                times.turn_ended = loop.time()
                print(f"    caller (as heard): {transcript}")
                history.add_caller(transcript)
                stream = await tts.open(RATE, voice)
                reply = await speak_reply(llm, stream, system, history, tools, times)
                write_wav(out / f"reply_{n}.wav", reply)
                report(n, times)
        finally:
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)
            await stt.close()

        if recorder is not None and args.probe_cancel:
            await probe_cancel(tts, voice, recorder)
        if recorder is not None and args.probe_gap:
            await probe_gap(cartesia_key, voice, recorder, out)
    finally:
        await tts.aclose()
        await client.close()
    print(f"\nsaved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--say", nargs="+", metavar="TEXT", help="questions for Cartesia to read")
    source.add_argument("--wav", nargs="+", type=Path, metavar="FILE", help="recorded questions")
    parser.add_argument("--voice", help="Cartesia voice id for the agent (default: config)")
    parser.add_argument(
        "--caller-voice", help="voice that reads --say questions (default: --voice)"
    )
    parser.add_argument("--kb", type=Path, default=Path("kb"), help="knowledge base folder")
    parser.add_argument("--out", type=Path, default=Path("data/dev"), help="output folder")
    parser.add_argument("--record", action="store_true", help="save raw provider messages")
    parser.add_argument("--cold", action="store_true", help="skip the prompt cache prewarm")
    parser.add_argument("--probe-cancel", action="store_true", help="what follows a cancel")
    parser.add_argument("--probe-gap", action="store_true", help="a context after 1.5 s quiet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="    [%(name)s] %(message)s")
    logging.getLogger("voice_agent.providers.anthropic_llm").setLevel(logging.DEBUG)
    try:
        asyncio.run(run(args))
    except ProviderError as e:
        sys.exit(f"provider error: {e}")


if __name__ == "__main__":
    main()
