"""Barge-in: is the caller interrupting, or just saying "mm-hm"?

While the agent is speaking, the caller often makes small listening noises ("yeah", "okay",
"uh-huh"). Those must not stop the agent. Real words must.

Two decisions use this, at two moments:

- While the caller is still talking, each time new words arrive: `interrupts_now()`. It
  needs MIN_LIVE_INTERRUPT_WORDS real words, so a single stray word (or noise the
  speech-to-text heard as one) doesn't cut the agent off. "No, wait" stops it at once.
- When the caller's turn ends: `classify()`.

      NOTHING      no words (a cough, line noise)  -> ignore
      BACKCHANNEL  only listening noises           -> ignore while the agent talks or after
                                                      its goodbye; otherwise it's an answer
      SPEECH       at least one real word          -> interrupt and answer

  So a lone "Stop." still stops the agent, just when the turn ends rather than mid-word.

Speech-to-text spells listening noises many ways ("mhm", "Mhmm.", "mmhmm", "hmmmm"), so
words are matched by sound, not spelling: repeated letters are collapsed first ("mhmmm" ->
"mhm", "okaaay" -> "okay"), and any word made only of m, h, n and u with an "m" in it is a
hum ("hmm", "mmhm", "umm").
"""

from __future__ import annotations

import re
from enum import StrEnum

# Real words needed to interrupt while the caller is still talking (see above).
MIN_LIVE_INTERRUPT_WORDS = 2

_REPEATS = re.compile(r"(.)\1+")
_HUM = re.compile(r"[hnu]*m[mhnu]*")
_WORD = re.compile(r"[a-z']+")


def _squash(word: str) -> str:
    """Collapse repeated letters, so spellings of the same sound compare equal."""
    return _REPEATS.sub(r"\1", word)


# Single words that only signal "I'm listening", compared after _squash.
BACKCHANNEL_WORDS = frozenset(
    _squash(w)
    for w in (
        "uh", "uhhuh", "ah", "oh",
        "yeah", "yep", "yup", "yes", "ok", "okay", "alright",
        "right", "sure", "cool", "nice", "wow",
    )
)  # fmt: skip
# Two-word listening phrases, after splitting on spaces and hyphens ("uh-huh" -> uh, huh).
# "huh" and "see" are only backchannels as part of these; "huh?" alone means "what?".
BACKCHANNEL_PHRASES = frozenset(
    (_squash(a), _squash(b))
    for a, b in (("uh", "huh"), ("got", "it"), ("i", "see"), ("all", "right"))
)


class Heard(StrEnum):
    NOTHING = "nothing"
    BACKCHANNEL = "backchannel"
    SPEECH = "speech"


def classify(transcript: str) -> Heard:
    """What the caller said, from the barge-in point of view."""
    words, real = _count(transcript)
    if not words:
        return Heard.NOTHING
    return Heard.SPEECH if real else Heard.BACKCHANNEL


def interrupts_now(transcript: str) -> bool:
    """Whether the caller's words so far should cut the agent off mid-sentence."""
    return _count(transcript)[1] >= MIN_LIVE_INTERRUPT_WORDS


def _count(transcript: str) -> tuple[int, int]:
    """(all words, real words): real words are the ones that aren't listening noises."""
    words = [_squash(w) for w in _WORD.findall(transcript.lower().replace("-", " "))]
    real = i = 0
    while i < len(words):
        if tuple(words[i : i + 2]) in BACKCHANNEL_PHRASES:
            i += 2
            continue
        if words[i] not in BACKCHANNEL_WORDS and not _HUM.fullmatch(words[i]):
            real += 1
        i += 1
    return len(words), real
