"""Audio conversion between Twilio Media Streams and the core.

On the wire, Twilio sends and expects base64 μ-law, 8000 Hz, mono, with no file header.
The core works in PCM16 little-endian at the adapter's sample rate (design §4.1), so this
adapter is the only place that knows about μ-law.

`audioop` left the standard library in Python 3.13; the `audioop-lts` package provides the
same module.
"""

from __future__ import annotations

import base64
import binascii

import audioop

SAMPLE_RATE = 8000  # Twilio's mediaFormat is always audio/x-mulaw, 8000 Hz, 1 channel
_WIDTH = 2  # bytes per PCM16 sample


def decode(payload: str) -> bytes:
    """A `media.payload` from Twilio → PCM16. Raises ValueError on bad base64."""
    try:
        mulaw = base64.b64decode(payload, validate=True)
    except binascii.Error as e:
        raise ValueError(f"media payload is not valid base64: {e}") from None
    return audioop.ulaw2lin(mulaw, _WIDTH)


class Encoder:
    """PCM16 → `media.payload` for Twilio, one chunk at a time.

    TTS chunks arrive as raw bytes and nothing guarantees a chunk ends on a sample
    boundary, so an odd trailing byte is held and joined to the next chunk instead of
    being dropped (which would shift every later sample and turn the audio to noise).
    """

    def __init__(self) -> None:
        self._held = b""

    def encode(self, pcm: bytes) -> str:
        """Return the payload to send, or "" when there's no whole sample yet."""
        data = self._held + pcm
        cut = len(data) - len(data) % _WIDTH
        self._held = data[cut:]
        if cut == 0:
            return ""
        return base64.b64encode(audioop.lin2ulaw(data[:cut], _WIDTH)).decode("ascii")

    def reset(self) -> None:
        """Forget a held byte. Called on clear, when the rest of that audio is dropped."""
        self._held = b""
