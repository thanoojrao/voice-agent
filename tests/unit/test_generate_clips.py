from scripts.generate_clips import FILLERS, clip_lines, generate
from tests.fakes import FakeTTS
from voice_agent.core.clips import load_clips


def test_lines_cover_every_clip_the_app_needs():
    lines = clip_lines("Alex")
    assert {"apology", "busy"} <= set(lines)
    assert [k for k in lines if k.startswith("filler_")] == [
        f"filler_{n}" for n in range(1, len(FILLERS) + 1)
    ]


def test_busy_clip_discloses_the_ai():
    assert clip_lines("Alex")["busy"].startswith("Hi, this is Alex's AI.")


async def test_generate_writes_clips_the_app_can_load(tmp_path):
    (tmp_path / "filler_9.wav").write_bytes(b"stale")  # from an older, longer list
    tts = FakeTTS()
    await generate(tts, "voice-1", "Alex", tmp_path)

    assert {s.voice_id for s in tts.streams} == {"voice-1"}
    assert sorted(tts.spoken) == sorted(clip_lines("Alex").values())
    assert not (tmp_path / "filler_9.wav").exists()
    assert (tmp_path / "busy.wav").is_file()
    clips = load_clips(tmp_path, 8000)
    assert len(clips.fillers) == len(FILLERS)
    assert clips.apology.duration_ms > 0
