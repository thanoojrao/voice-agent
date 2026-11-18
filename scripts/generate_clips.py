"""Render the pre-recorded clips in the agent's voice: apology, fillers and busy.

    uv run python scripts/generate_clips.py              # voice_id from agent.config.yaml
    uv run python scripts/generate_clips.py --voice <id> --out assets/audio

Run it once before the first call, and again after changing the voice (say, to a clone).
The clips are WAV files, PCM16 mono at 8 kHz: what the core plays to the phone without
conversion (core/clips.py checks this at startup), and a format Twilio's <Play> accepts for
the busy and apology clips it fetches from /static/audio.

    apology.wav     a provider failed mid-call, or Twilio used the fallback URL
    filler_*.wav    a reply is slow to start (never added to the transcript)
    busy.wav        refused at the webhook: too many calls, or a daily cap

Calls Cartesia once per clip; the whole set costs a fraction of a cent. CARTESIA_API_KEY
comes from .env.
"""

import argparse
import asyncio
import sys
import wave
from pathlib import Path

from voice_agent.adapters.twilio.audio import SAMPLE_RATE
from voice_agent.app.config import Settings, load_agent_config
from voice_agent.app.constants import SPEAKING_RATE
from voice_agent.core.clips import load_clips
from voice_agent.core.interfaces import TextToSpeech
from voice_agent.providers.cartesia_tts import CartesiaSettings, CartesiaTTS

FILLERS = ("Hmm, let me think.", "Let me see.", "One sec.", "Okay, so.")


def clip_lines(owner_name: str) -> dict[str, str]:
    """File name (no extension) → what the voice says."""
    lines = {
        "apology": "Sorry, I'm having some trouble on my end right now. "
        "Please try calling back in a little while.",
        # Heard instead of the greeting, so it carries the AI disclosure too.
        "busy": f"Hi, this is {owner_name}'s AI. I can't take a call right now. "
        "Please try again a bit later.",
    }
    for n, text in enumerate(FILLERS, start=1):
        lines[f"filler_{n}"] = text
    return lines


async def render(tts: TextToSpeech, voice_id: str, text: str) -> bytes:
    stream = await tts.open(SAMPLE_RATE, voice_id)
    await stream.say(text)
    await stream.finish()
    return b"".join([chunk.pcm async for chunk in stream.audio()])


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


async def generate(tts: TextToSpeech, voice_id: str, owner_name: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("filler_*.wav"):  # a shorter filler list mustn't leave strays
        old.unlink()
    for name, text in clip_lines(owner_name).items():
        pcm = await render(tts, voice_id, text)
        write_wav(out / f"{name}.wav", pcm)
        print(f"  {name}.wav  {len(pcm) / (SAMPLE_RATE * 2):4.1f} s  {text}")
    load_clips(out, SAMPLE_RATE)  # the same check the app makes at startup


async def run(args: argparse.Namespace) -> None:
    settings = Settings()
    settings.require("cartesia_api_key")
    assert settings.cartesia_api_key
    config = load_agent_config(settings.config_path)
    voice = args.voice or config.voice_id
    if not voice:
        sys.exit("No voice: pass --voice <Cartesia voice id> (or set voice_id in the config).")
    tts = CartesiaTTS(
        settings.cartesia_api_key.get_secret_value(),
        CartesiaSettings(speed=SPEAKING_RATE, pronunciations=config.pronunciations),
    )
    try:
        await generate(tts, voice, config.owner_name, args.out)
    finally:
        await tts.aclose()
    print(f"saved to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--voice", help="Cartesia voice id (default: voice_id in the config)")
    parser.add_argument("--out", type=Path, default=Path("assets/audio"), help="output folder")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
