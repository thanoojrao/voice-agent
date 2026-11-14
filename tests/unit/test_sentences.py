import pytest

from voice_agent.core.sentences import SentenceSplitter


def split_whole(text: str, **kw) -> list[str]:
    s = SentenceSplitter(**kw)
    out = s.feed(text)
    if rest := s.flush():
        out.append(rest)
    return out


def split_streamed(text: str, step: int = 1, **kw) -> list[str]:
    """Feed a few characters at a time, like an LLM stream."""
    s = SentenceSplitter(**kw)
    out = []
    for i in range(0, len(text), step):
        out += s.feed(text[i : i + step])
    if rest := s.flush():
        out.append(rest)
    return out


CASES = [
    ("Hi there. I build voice agents! Want to hear more?",
     ["Hi there.", "I build voice agents!", "Want to hear more?"]),
    ("It cut latency by 3.5 times. Nice.", ["It cut latency by 3.5 times.", "Nice."]),
    ("I met Dr. Patel there. It shipped.", ["I met Dr. Patel there.", "It shipped."]),
    ("Things like e.g. parsing work. Yes.", ["Things like e.g. parsing work.", "Yes."]),
    ("Ask J. R. Smith about it. Thanks.", ["Ask J. R. Smith about it.", "Thanks."]),
    ("See example.com for details. Bye.", ["See example.com for details.", "Bye."]),
    ("Email me at a.b@c.io today. Bye.", ["Email me at a.b@c.io today.", "Bye."]),
    ("Version v1.2 is out. Try it.", ["Version v1.2 is out.", "Try it."]),
    ('She said "it works." Then left.', ['She said "it works."', "Then left."]),
    ("Really?! Yes.", ["Really?!", "Yes."]),
    ("Well... maybe later. Sure.", ["Well... maybe later.", "Sure."]),
    ("Hmm… let me think. Okay.", ["Hmm… let me think.", "Okay."]),
    ("I tried that... It failed.", ["I tried that...", "It failed."]),
    ("Python, Go, etc. are fine. Next.", ["Python, Go, etc. are fine.", "Next."]),
    ("I use Python, Go, etc. Mostly Python.", ["I use Python, Go, etc.", "Mostly Python."]),
    ("First line\nSecond line", ["First line", "Second line"]),
    ("No ending punctuation", ["No ending punctuation"]),
    ("", []),
]  # fmt: skip


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_whole_text(text, expected):
    assert split_whole(text) == expected


@pytest.mark.parametrize(("text", "expected"), CASES)
@pytest.mark.parametrize("step", [1, 3, 7])
def test_streaming_matches_whole(text, expected, step):
    assert split_streamed(text, step) == expected


def test_sentence_released_as_soon_as_next_char_arrives():
    s = SentenceSplitter()
    assert s.feed("Hello there.") == []  # might be "there.com"
    assert s.feed(" ") == ["Hello there."]
    assert s.feed("More") == []
    assert s.flush() == "More"


def test_number_not_split_while_waiting():
    s = SentenceSplitter()
    assert s.feed("It took 3.") == []
    assert s.feed("5 seconds. ") == ["It took 3.5 seconds."]


def test_long_sentence_split_at_clause_break():
    text = (
        "I have been working on a voice agent that answers my phone, "
        "and it uses streaming speech recognition, a language model and streaming speech "
        "synthesis so that replies start quickly even when the full answer is long"
    )
    out = split_streamed(text, step=5, soft_limit=100)
    assert len(out) > 1
    assert all(len(part) <= 100 for part in out[:-1])
    assert out[0].endswith(",")
    assert " ".join(out) == text


def test_dash_stays_with_first_part():
    out = split_whole("x" * 60 + " — and then the rest of it", soft_limit=70)
    assert out == ["x" * 60 + " —", "and then the rest of it"]


def test_long_text_without_breaks_waits_for_boundary():
    out = split_whole("a" * 300 + ". Done.", soft_limit=100)
    assert out == ["a" * 300 + ".", "Done."]


def test_flush_resets():
    s = SentenceSplitter()
    s.feed("Partial")
    assert s.flush() == "Partial"
    assert s.flush() is None
