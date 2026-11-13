"""The fakes must match the core's Protocols and behave the way later tests rely on."""

import pytest

from tests.fakes import (
    FakeLLM,
    FakeNotifier,
    FakeSTT,
    FakeTTS,
    FakeVoiceAdapter,
    Reply,
    silence,
)
from voice_agent.core.interfaces import (
    Capability,
    ChannelAdapter,
    Done,
    LanguageModel,
    Notifier,
    SpeechToText,
    TextDelta,
    TextToSpeech,
    TurnEnded,
    TurnStarted,
    Words,
)

# Static check (pyright): each fake is assignable to its Protocol.
_adapter: ChannelAdapter = FakeVoiceAdapter()
_stt: SpeechToText = FakeSTT()
_llm: LanguageModel = FakeLLM()
_tts: TextToSpeech = FakeTTS()
_notifier: Notifier = FakeNotifier()


class Recorder:
    def __init__(self):
        self.audio, self.marks, self.ended = [], [], []

    async def on_audio(self, pcm):
        self.audio.append(pcm)

    async def on_mark_played(self, name):
        self.marks.append(name)

    async def on_session_end(self, reason):
        self.ended.append(reason)


async def test_adapter_plays_marks_and_clear_drops_pending():
    a, rec = FakeVoiceAdapter(), Recorder()
    a.attach(rec)
    for m in ("m1", "m2", "m3"):
        await a.mark(m)
    await a.play_marks(1)
    await a.clear_audio()
    await a.play_marks()
    assert rec.marks == ["m1"]  # m2, m3 were cleared before playing
    await a.caller_audio(silence(20))
    await a.hang_up()
    assert len(rec.audio[0]) == 320 and rec.ended == ["caller_hangup"]


async def test_adapter_rejects_missing_capabilities():
    a = FakeVoiceAdapter()
    with pytest.raises(AssertionError):
        await a.end_session()
    a = FakeVoiceAdapter(frozenset({Capability.END_SESSION}))
    await a.end_session()
    assert a.ended


async def test_stt_emits_scripted_events_until_closed():
    stt = FakeSTT()
    s = await stt.open(8000)
    s.emit(TurnStarted(), Words("what are"), TurnEnded("what are you building?"))
    await s.close()
    assert [e async for e in s.events()][-1] == TurnEnded("what are you building?")


async def test_llm_replays_script_and_records_calls():
    llm = FakeLLM([Reply([TextDelta("Hi."), Done("end_turn")]), Reply(raises=TimeoutError())])
    out = [e async for e in llm.stream_reply([], [], [])]
    assert out == [TextDelta("Hi."), Done("end_turn")]
    with pytest.raises(TimeoutError):
        [e async for e in llm.stream_reply([], [], [])]
    assert len(llm.calls) == 2


async def test_tts_word_timings_continue_across_sentences():
    tts = FakeTTS()
    s = await tts.open(8000, "voice")
    await s.say("Hello there.")
    await s.say("I build things.")
    await s.finish()
    chunks = [c async for c in s.audio()]
    starts = [w.start_ms for c in chunks for w in c.words]
    assert starts == [0, 300, 600, 900, 1200]
    assert tts.spoken == ["Hello there.", "I build things."]


async def test_notifier_can_fail():
    with pytest.raises(ConnectionError):
        await FakeNotifier(fail=True).alert("x", "y")
