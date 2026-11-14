"""Barge-in: is the caller interrupting, or just saying "mm-hm"?

While the agent is speaking, the caller often makes small listening noises ("yeah", "okay",
"uh-huh"). Those must not stop the agent. Real words must, quickly.

The session calls `classify()` with the caller's transcript so far each time new words
arrive while the agent is speaking:

    NOTHING      no words yet (a cough, line noise) -> keep speaking
    BACKCHANNEL  only listening noises so far        -> keep speaking, keep checking
    SPEECH       at least one real word              -> interrupt now

Transcripts grow while the caller talks ("yeah" -> "yeah but wait"), so a BACKCHANNEL result
is never final: the next update may turn it into SPEECH. If the caller's turn ends as only
a backchannel, the session drops it instead of answering it.
"""

from __future__ import annotations

import re
from enum import StrEnum

# Single words that only signal "I'm listening".
BACKCHANNEL_WORDS = frozenset(
    {
        "mm", "mmm", "hm", "hmm", "mhm", "uh", "um",
        "yeah", "yep", "yup", "yes", "ok", "okay", "alright",
        "right", "sure", "cool", "nice", "wow", "ah", "oh",
    }
)  # fmt: skip
# Two-word listening phrases, after splitting on spaces and hyphens ("uh-huh" -> uh, huh).
# "huh" and "see" are only backchannels as part of these; "huh?" alone means "what?".
BACKCHANNEL_PHRASES = frozenset(
    {("uh", "huh"), ("mm", "hm"), ("mm", "hmm"), ("got", "it"), ("i", "see"), ("all", "right")}
)

_WORD = re.compile(r"[a-z']+")


class Heard(StrEnum):
    NOTHING = "nothing"
    BACKCHANNEL = "backchannel"
    SPEECH = "speech"


def classify(transcript: str) -> Heard:
    """What the caller has said so far, from the barge-in point of view."""
    words = _WORD.findall(transcript.lower().replace("-", " "))
    if not words:
        return Heard.NOTHING
    i = 0
    while i < len(words):
        if tuple(words[i : i + 2]) in BACKCHANNEL_PHRASES:
            i += 2
        elif words[i] in BACKCHANNEL_WORDS:
            i += 1
        else:
            return Heard.SPEECH
    return Heard.BACKCHANNEL
