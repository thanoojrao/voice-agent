"""Turning what the model writes into what the caller should hear.

The model marks anything to be spelled out with a tag:

    You can email me at jane.doe@gmail.com. That's <spell>jane.doe@gmail.com</spell>.

and `speakable()` replaces the tag with the letters, written the way a person reads them
out over the phone:

    You can email me at jane.doe@gmail.com. That's J A N E, dot, D O E, at gmail dot com.

The model decides when to spell (first time an address comes up, the caller asking to hear
it again, reading back the caller's own address); this file decides how, so the letters are
always right. Nothing is spelled without a tag.

How it reads:
- Letters in capitals, one at a time, with a short pause (a comma) every few letters, so
  the caller can write them down.
- A repeated letter is said as "double": "double O".
- Symbols are said as words: dot, dash, underscore, plus; "@" becomes "at".
- After the "@", well-known names (gmail, com, ...) are said as words; anything else is
  spelled too.

Any tag left over (unclosed, or stray) is removed, so the caller never hears "spell".
The turn puts the result in the history too, so the words the caller heard (counted from
the audio) line up with the words the model sees it said.
"""

from __future__ import annotations

import re

_TAG = re.compile(r"<spell>(.*?)</spell>", re.IGNORECASE | re.DOTALL)
_STRAY_TAG = re.compile(r"</?spell\s*/?>", re.IGNORECASE)

# Domain parts said as words rather than spelled.
SAY_AS_WORDS = frozenset(
    {
        "gmail", "googlemail", "outlook", "hotmail", "live", "yahoo", "icloud", "me", "aol",
        "proton", "protonmail", "com", "net", "org", "edu", "gov", "io", "co", "ai", "dev",
        "uk", "us", "in", "ca",
    }
)  # fmt: skip
SYMBOLS = {".": "dot", "-": "dash", "_": "underscore", "+": "plus"}
GROUP = 4  # letters between pauses


def speakable(text: str) -> str:
    """`text` with each <spell>...</spell> spelled out and any stray tag removed."""
    spoken = _TAG.sub(lambda m: spell(m.group(1)), text)
    return _STRAY_TAG.sub("", spoken)


def spell(value: str) -> str:
    """Spell an email address, or any code, the way it's read out over the phone."""
    value = value.strip()
    local, at, domain = value.partition("@")
    if not at:
        return _spell_chars(value)
    parts = [p if p.lower() in SAY_AS_WORDS else _spell_chars(p) for p in domain.split(".")]
    return f"{_spell_chars(local)}, at {' dot '.join(parts)}"


def _spell_chars(text: str) -> str:
    """Letters and digits one by one, "double" for a pair, symbols as words, a pause
    every GROUP letters and around each symbol."""
    groups: list[list[str]] = [[]]
    letters = 0  # letters in the current group; "double O" counts as two
    i = 0
    while i < len(text):
        char = text[i]
        if char in SYMBOLS:
            groups += [[SYMBOLS[char]], []]
            letters = 0
            i += 1
            continue
        if not char.isalnum():  # spaces and anything unexpected: a pause, not a sound
            groups.append([])
            letters = 0
            i += 1
            continue
        pair = i + 1 < len(text) and text[i + 1].lower() == char.lower()
        size = 2 if pair else 1
        if letters + size > GROUP:
            groups.append([])
            letters = 0
        groups[-1].append(f"double {char.upper()}" if pair else char.upper())
        letters += size
        i += size
    return ", ".join(" ".join(g) for g in groups if g)
