"""Pre-recorded audio clips the core plays without calling TTS.

- **Fillers** ("Hmm, let me think."): played when no reply audio is ready after
  FILLER_AFTER_MS, so the caller isn't left in silence. Never added to the history.
- **Apology** ("Sorry, I'm having trouble right now..."): played when a provider fails
  twice, right before the session ends. It must work when Cartesia is the thing that broke,
  so it can't be generated on the fly.

The busy clip lives in the same folder but isn't loaded here: Twilio plays it straight from
a static URL before a session ever starts.

Clips are WAV files rendered once with the current voice by scripts/generate_clips.py:

    assets/audio/apology.wav
    assets/audio/filler_*.wav      (at least one)

They must be PCM16 mono at the adapter's sample rate (8 kHz for Twilio), the same format
the core sends to adapters, so they play without conversion. A wrong format fails at
startup, not in the middle of a call.
"""

from __future__ import annotations

import random
import wave
from dataclasses import dataclass
from pathlib import Path

SAMPLE_WIDTH = 2  # bytes per sample: PCM16


class ClipError(RuntimeError):
    """A clip is missing or in the wrong format."""


@dataclass(frozen=True, slots=True)
class Clip:
    name: str
    pcm: bytes
    sample_rate: int

    @property
    def duration_ms(self) -> int:
        return len(self.pcm) * 1000 // (self.sample_rate * SAMPLE_WIDTH)


@dataclass(frozen=True, slots=True)
class Clips:
    apology: Clip
    fillers: tuple[Clip, ...]

    def __post_init__(self) -> None:
        if not self.fillers:
            raise ClipError("at least one filler clip is needed")

    def filler(self, previous: Clip | None = None, rng: random.Random | None = None) -> Clip:
        """A random filler, never the same one twice in a row (when there's a choice)."""
        choices = [c for c in self.fillers if c != previous] or list(self.fillers)
        return (rng or random).choice(choices)


def load_clips(directory: Path, sample_rate: int) -> Clips:
    """Load and check every clip the core needs. Raises ClipError on any problem."""
    if not directory.is_dir():
        raise ClipError(f"clip folder not found: {directory}")
    filler_paths = sorted(directory.glob("filler_*.wav"))
    if not filler_paths:
        raise ClipError(f"no filler_*.wav clips in {directory}")
    return Clips(
        apology=read_clip(directory / "apology.wav", sample_rate),
        fillers=tuple(read_clip(p, sample_rate) for p in filler_paths),
    )


def read_clip(path: Path, sample_rate: int) -> Clip:
    """Read one WAV file, checking it is PCM16 mono at `sample_rate`."""
    if not path.is_file():
        raise ClipError(f"clip not found: {path}")
    try:
        with wave.open(str(path), "rb") as w:
            channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            pcm = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as e:
        raise ClipError(f"{path.name}: not a readable WAV file ({e})") from e

    problems = []
    if channels != 1:
        problems.append(f"{channels} channels, need mono")
    if width != SAMPLE_WIDTH:
        problems.append(f"{width * 8}-bit samples, need 16-bit")
    if rate != sample_rate:
        problems.append(f"{rate} Hz, need {sample_rate} Hz")
    if not pcm:
        problems.append("no audio")
    if problems:
        raise ClipError(f"{path.name}: {', '.join(problems)}")
    return Clip(name=path.stem, pcm=pcm, sample_rate=rate)
