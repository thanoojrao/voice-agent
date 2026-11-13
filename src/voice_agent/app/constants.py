"""Fixed values for v0 (spec: "Constants in code (v0)").

Each of these can become a setting in agent.config.yaml later with a one-line change.
"""

SPEAKING_RATE = 1.0  # TTS speed multiplier
MAX_ANSWER_SENTENCES = 3  # spoken answers stay short
LANGUAGE = "en"  # English only
CHANNELS = ("twilio",)  # the only channel in v0
FILLER_AFTER_MS = 1500  # play a filler clip if no audio is ready by then
PROVIDER_TIMEOUT_S = 5.0  # per provider call before the apology clip
BLOCK_ANONYMOUS = False  # withheld caller IDs are allowed
