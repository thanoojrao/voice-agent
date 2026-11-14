"""Streaming sentence splitter.

The LLM streams text in small pieces. Sending each complete sentence to TTS as soon as it
exists lets the caller hear the first sentence while the rest is still being written.

    splitter = SentenceSplitter()
    for delta in llm_text:
        for sentence in splitter.feed(delta):
            await tts.say(sentence)
    if rest := splitter.flush():
        await tts.say(rest)

A sentence ends at . ! ? (plus any closing quote or bracket) followed by whitespace, or at a
line break. A boundary is only decided once the next character has arrived, so "3." waiting
for "5" never splits a number. Not boundaries: abbreviations ("Dr.", "e.g."), initials
("J. Smith"), an ellipsis or "etc." before a lowercase word, and anything without a
following space ("example.com", "v1.2").
"""

from __future__ import annotations

ENDERS = ".!?…"
CLOSERS = "\"')]}”’"
# Never end a sentence after these (lowercased, without the final dot).
ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "e.g", "i.e", "approx", "inc", "ltd"}
)
# End a sentence after these only if the next word starts with a capital letter.
MAYBE_ABBREVIATIONS = frozenset({"etc"})
# A sentence with no boundary in sight is split at a clause break once it gets this long,
# so TTS can start speaking.
SOFT_LIMIT = 200
CLAUSE_BREAKS = (", ", "; ", ": ", " - ", " — ")


class SentenceSplitter:
    def __init__(self, soft_limit: int = SOFT_LIMIT) -> None:
        self._buf = ""
        self._soft_limit = soft_limit

    def feed(self, text: str) -> list[str]:
        """Add streamed text; return any sentences now complete."""
        self._buf += text
        out: list[str] = []
        while (cut := self._next_boundary()) is not None:
            self._emit(cut, out)
        if len(self._buf) > self._soft_limit and (cut := self._clause_break()) is not None:
            self._emit(cut, out)
        return out

    def flush(self) -> str | None:
        """End of the reply: return whatever is left, if anything."""
        rest = self._buf.strip()
        self._buf = ""
        return rest or None

    # -- internals ------------------------------------------------------------

    def _emit(self, cut: int, out: list[str]) -> None:
        sentence = self._buf[:cut].strip()
        self._buf = self._buf[cut:].lstrip()
        if sentence:
            out.append(sentence)

    def _next_boundary(self) -> int | None:
        """Index just past the first sentence end in the buffer, or None if there isn't one
        yet (including 'maybe, but need to see the next character')."""
        buf = self._buf
        i = 0
        while i < len(buf):
            ch = buf[i]
            if ch == "\n":
                return i + 1
            if ch not in ENDERS:
                i += 1
                continue
            j = i + 1  # past the whole run of punctuation and closing quotes
            while j < len(buf) and (buf[j] in ENDERS or buf[j] in CLOSERS):
                j += 1
            if j == len(buf):
                return None  # can't tell yet: "3." might become "3.5"
            if is_boundary(buf, i, j):
                return j
            if needs_next_word(buf, i, j) and not buf[j:].strip():
                return None
            i = j  # the run is decided; never re-examine its tail
        return None

    def _clause_break(self) -> int | None:
        """Last clause break within the soft limit, as a cut index."""
        window = self._buf[: self._soft_limit]
        cuts = [
            pos + len(b.rstrip())  # keep the comma or dash with the first part
            for b in CLAUSE_BREAKS
            if (pos := window.rfind(b)) > 0
        ]
        return max(cuts, default=None)


# -- boundary rules: pure functions of the text -------------------------------


def is_boundary(buf: str, i: int, j: int) -> bool:
    """Does the punctuation run buf[i:j] (followed by buf[j]) end a sentence?"""
    if not buf[j].isspace():
        return False  # "example.com", "v1.2"
    if needs_next_word(buf, i, j):
        nxt = buf[j:].lstrip()
        return bool(nxt) and nxt[0].isupper()
    if buf[i] == ".":
        return ends_sentence(buf, i)
    return True


def needs_next_word(buf: str, i: int, j: int) -> bool:
    """Ellipses and "etc." end a sentence only before a capitalised word."""
    run = buf[i:j]
    if "..." in run or "…" in run:
        return True
    return buf[i] == "." and word_before(buf, i).lower() in MAYBE_ABBREVIATIONS


def ends_sentence(buf: str, dot: int) -> bool:
    """Does the "." at buf[dot] end a sentence, given the word before it?"""
    word = word_before(buf, dot)
    if word.lower() in ABBREVIATIONS:
        return False
    if len(word) == 1 and word.isupper():
        return False  # an initial: "J. Smith"
    return True


def word_before(buf: str, dot: int) -> str:
    """The word ending at buf[dot], without any opening quote or bracket."""
    start = dot
    while start > 0 and not buf[start - 1].isspace():
        start -= 1
    return buf[start:dot].lstrip("\"'([{“‘")
