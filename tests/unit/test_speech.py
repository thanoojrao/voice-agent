"""Spelling out what the model tags with <spell>."""

import pytest

from voice_agent.core.speech import speakable, spell


def test_text_without_tags_is_unchanged():
    text = "You can email me at jane.doe@gmail.com, or leave a message."
    assert speakable(text) == text


def test_a_tagged_email_is_spelled_in_place():
    assert speakable("That's <spell>jane.doe@gmail.com</spell>.") == (
        "That's J A N E, dot, D O E, at gmail dot com."
    )


def test_letters_pause_every_four_and_pairs_are_double():
    assert spell("bookkeeper") == "B double O, double K double E, P E R"


def test_an_unusual_domain_is_spelled_too():
    assert spell("sam@acme.io") == "S A M, at A C M E dot io"


@pytest.mark.parametrize(
    ("value", "spoken"),
    [
        ("j_doe-2+x", "J, underscore, D O E, dash, 2, plus, X"),
        ("AB12", "A B 1 2"),
        ("ab 12", "A B, 1 2"),  # a space is a pause
        ("aaa", "double A A"),
        ("  ok  ", "O K"),
    ],
)
def test_symbols_digits_and_spaces(value, spoken):
    assert spell(value) == spoken


def test_several_tags_and_any_case():
    assert speakable("<spell>ab</spell> and <SPELL>cd</Spell>") == "A B and C D"


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("Email <spell>jane@gmail.com", "Email jane@gmail.com"),  # never closed
        ("Done.</spell>", "Done."),
        ("A <spell/> b", "A  b"),
    ],
)
def test_stray_tags_are_removed(text, spoken):
    assert speakable(text) == spoken
