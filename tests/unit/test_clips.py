import random
import wave
from pathlib import Path

import pytest

from voice_agent.core.clips import Clip, ClipError, Clips, load_clips, read_clip

RATE = 8000


def write_wav(path: Path, ms: int = 200, rate: int = RATE, channels: int = 1, width: int = 2):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"\x00" * (rate * ms // 1000) * channels * width)


@pytest.fixture
def clip_dir(tmp_path: Path) -> Path:
    write_wav(tmp_path / "apology.wav", ms=1500)
    write_wav(tmp_path / "filler_1.wav", ms=400)
    write_wav(tmp_path / "filler_2.wav", ms=600)
    write_wav(tmp_path / "busy.wav")  # not loaded by the core
    return tmp_path


def test_loads_apology_and_fillers(clip_dir):
    clips = load_clips(clip_dir, RATE)
    assert clips.apology.name == "apology"
    assert clips.apology.duration_ms == 1500
    assert [f.name for f in clips.fillers] == ["filler_1", "filler_2"]
    assert clips.fillers[1].duration_ms == 600


def test_missing_folder(tmp_path):
    with pytest.raises(ClipError, match="clip folder not found"):
        load_clips(tmp_path / "nope", RATE)


def test_missing_apology(clip_dir):
    (clip_dir / "apology.wav").unlink()
    with pytest.raises(ClipError, match="clip not found"):
        load_clips(clip_dir, RATE)


def test_no_fillers(clip_dir):
    for p in clip_dir.glob("filler_*.wav"):
        p.unlink()
    with pytest.raises(ClipError, match="no filler"):
        load_clips(clip_dir, RATE)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rate": 16000}, "16000 Hz, need 8000 Hz"),
        ({"channels": 2}, "2 channels, need mono"),
        ({"width": 1}, "8-bit samples, need 16-bit"),
        ({"ms": 0}, "no audio"),
    ],
)
def test_wrong_format_rejected(tmp_path, kwargs, message):
    write_wav(tmp_path / "bad.wav", **kwargs)
    with pytest.raises(ClipError, match=message):
        read_clip(tmp_path / "bad.wav", RATE)


def test_not_a_wav(tmp_path):
    (tmp_path / "bad.wav").write_bytes(b"not audio")
    with pytest.raises(ClipError, match="not a readable WAV"):
        read_clip(tmp_path / "bad.wav", RATE)


def test_filler_never_repeats_back_to_back(clip_dir):
    clips = load_clips(clip_dir, RATE)
    rng = random.Random(0)
    previous = None
    for _ in range(50):
        pick = clips.filler(previous, rng)
        assert pick != previous
        previous = pick


def test_single_filler_can_repeat():
    only = Clip("filler_1", b"\x00\x00", RATE)
    clips = Clips(apology=only, fillers=(only,))
    assert clips.filler(only) == only


def test_clips_need_a_filler():
    apology = Clip("apology", b"\x00\x00", RATE)
    with pytest.raises(ClipError):
        Clips(apology=apology, fillers=())
