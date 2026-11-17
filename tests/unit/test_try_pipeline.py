"""The dev script's pure helpers: converting audio to what the phone sends, and recording
provider messages. The script itself calls real APIs and isn't run here."""

import json
import struct
from pathlib import Path

from scripts.try_pipeline import (
    FRAME_BYTES,
    Recorder,
    frames,
    read_wav,
    silence,
    to_phone_pcm,
    write_wav,
)


def pcm16(*samples: int) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def test_stereo_16khz_becomes_mono_8khz():
    one_second_stereo = pcm16(1000, 3000) * 16000
    out = to_phone_pcm(one_second_stereo, rate=16000, width=2, channels=2)
    assert abs(len(out) - 16000) <= 4  # 8000 samples of 2 bytes, give or take the resampler
    assert struct.unpack("<h", out[100:102])[0] == 2000  # the channels averaged


def test_8bit_audio_is_widened_to_16bit():
    out = to_phone_pcm(bytes(800), rate=8000, width=1, channels=1)
    assert len(out) == 1600


def test_frames_are_20ms_with_the_last_one_padded():
    pcm = silence(0.05)  # 50 ms -> 20 + 20 + 10 padded to 20
    out = frames(b"\x01" * len(pcm))
    assert [len(f) for f in out] == [FRAME_BYTES] * 3
    assert out[-1].endswith(b"\x00" * (FRAME_BYTES // 2))


def test_a_phone_rate_wav_round_trips(tmp_path: Path):
    pcm = pcm16(1, -1, 300, -300) * 100
    write_wav(tmp_path / "a.wav", pcm)
    assert read_wav(tmp_path / "a.wav") == pcm


def test_recorder_leaves_out_audio(tmp_path: Path):
    recorder = Recorder(tmp_path / "messages.jsonl")
    recorder.add("deepgram", "sent", b"\x00\x01")  # an audio frame: skipped
    recorder.add("cartesia", "recv", json.dumps({"type": "chunk", "data": "QUJD" * 100}))

    [line] = (tmp_path / "messages.jsonl").read_text().splitlines()
    entry = json.loads(line)
    assert entry["provider"] == "cartesia"
    assert entry["message"] == {"type": "chunk", "data": "<400 base64 chars>"}
