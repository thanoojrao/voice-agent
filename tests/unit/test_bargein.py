import pytest

from voice_agent.core.bargein import Heard, classify


@pytest.mark.parametrize("text", ["", "   ", "...", "?"])
def test_no_words_is_nothing(text):
    assert classify(text) is Heard.NOTHING


@pytest.mark.parametrize(
    "text",
    [
        "yeah",
        "Yeah.",
        "Okay, okay.",
        "mm-hm",
        "Mm-hmm.",
        "mhm",
        "uh-huh",
        "uh huh",
        "right right",
        "Got it.",
        "I see.",
        "all right",
        "Oh, cool.",
        "yeah, sure",
    ],
)
def test_listening_noises_are_backchannel(text):
    assert classify(text) is Heard.BACKCHANNEL


@pytest.mark.parametrize(
    "text",
    [
        "wait",
        "no",
        "Huh?",  # "what?", not "uh-huh"
        "yeah but what about the cost",
        "okay so how does it work",
        "I have a question",
        "Sorry, can you repeat that?",
        "see",
    ],
)
def test_real_words_are_speech(text):
    assert classify(text) is Heard.SPEECH


def test_growing_transcript_turns_into_speech():
    # Flux sends the transcript so far each time; the decision is re-made on each update.
    updates = ["uh", "uh yeah", "uh yeah but", "uh yeah but how"]
    assert [classify(t) for t in updates] == [
        Heard.BACKCHANNEL,
        Heard.BACKCHANNEL,
        Heard.SPEECH,
        Heard.SPEECH,
    ]
