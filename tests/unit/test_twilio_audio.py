import base64
import struct

import audioop
import pytest

from voice_agent.adapters.twilio.audio import Encoder, decode


def pcm(*samples: int) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def samples(data: bytes) -> list[int]:
    return list(struct.unpack(f"<{len(data) // 2}h", data))


def test_decode_turns_mulaw_into_pcm16():
    # 0xFF is μ-law silence; one byte becomes one 16-bit sample
    assert decode(base64.b64encode(b"\xff\xff\xff").decode()) == pcm(0, 0, 0)


def test_decode_rejects_bad_base64():
    with pytest.raises(ValueError, match="base64"):
        decode("not base64!")


def test_encode_then_decode_round_trips_within_mulaw_precision():
    original = [0, 1000, -1000, 8000, -8000, 30000, -30000]
    payload = Encoder().encode(pcm(*original))
    assert len(base64.b64decode(payload)) == len(original)  # one byte per sample
    for got, want in zip(samples(decode(payload)), original, strict=True):
        assert abs(got - want) <= max(16, abs(want) // 16)  # μ-law is lossy, about 4%


def test_encode_matches_audioop_for_whole_samples():
    data = pcm(0, 123, -456, 7890)
    assert Encoder().encode(data) == base64.b64encode(audioop.lin2ulaw(data, 2)).decode()


def test_odd_byte_is_held_for_the_next_chunk():
    data = pcm(100, 200, 300)  # 6 bytes
    enc = Encoder()
    first, second = enc.encode(data[:3]), enc.encode(data[3:])  # split mid-sample
    sent = base64.b64decode(first) + base64.b64decode(second)
    assert sent == audioop.lin2ulaw(data, 2)


def test_a_single_byte_gives_nothing_yet():
    enc = Encoder()
    assert enc.encode(b"\x01") == ""
    assert enc.encode(b"") == ""


def test_empty_chunk_gives_nothing():
    assert Encoder().encode(b"") == ""


def test_reset_drops_the_held_byte():
    enc = Encoder()
    enc.encode(pcm(5)[:1])
    enc.reset()
    assert enc.encode(pcm(7)) == Encoder().encode(pcm(7))
