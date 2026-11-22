"""The eval runner's grading code, with made-up replies (no API calls).

Only the checks that are easy to get subtly wrong are tested here: a check that wrongly
passes would hide a real failure behind a "5/5".
"""

from pathlib import Path

import pytest

from scripts.run_evals import (
    CASES_PATH,
    Reply,
    check_never_says,
    check_no_tool,
    check_speakable,
    check_tool,
    history_for,
    load_cases,
    parse_verdict,
    sentence_count,
)
from voice_agent.app.config import load_agent_config
from voice_agent.core.interfaces import ToolCall, ToolResultPart, ToolUsePart
from voice_agent.core.tools import tool_defs

TOOL_NAMES = {t.name for t in tool_defs(can_end_session=True)}
FIRST_PERSON = r"\b(he|him|his)\b"


def reply_with(*calls: ToolCall, text: str = "") -> Reply:
    return Reply(text=text, calls=list(calls))


# -- the case file ------------------------------------------------------------------------------


def test_case_file_loads_and_names_only_real_tools():
    cases = load_cases(CASES_PATH, "Alex", TOOL_NAMES)
    assert len(cases) >= 20


def test_case_file_never_contains_the_owners_name():
    # The repo is public: the name comes from agent.config.yaml at run time, via {owner}.
    owner = load_agent_config(Path("agent.config.yaml")).owner_name
    assert owner.casefold() not in CASES_PATH.read_text(encoding="utf-8").casefold()


def test_owner_name_is_filled_in_as_a_literal(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text(
        "- id: x\n  conversation: [{caller: 'hi {owner}'}]\n  expect: {says: \"{owner}'s AI\"}\n"
    )
    [case] = load_cases(path, "A.J.", TOOL_NAMES)
    assert case.conversation == (("caller", "hi A.J."),)
    assert check_never_says("I'm AxJx's AI", case.expect["says"]) is None  # dots aren't wildcards


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ("  expect: {tool: {name: send_fax}}", "unknown tool"),
        ("  expect: {tool: {name: save_message, args_has: {}}}", "unknown field"),
        ("  expect: {says_maybe: x}", "unknown field"),
        ("  expect: {never_says: '(unclosed'}", "missing"),
    ],
)
def test_bad_case_fails_before_any_call(tmp_path, body, error):
    path = tmp_path / "cases.yaml"
    path.write_text(f"- id: x\n  conversation: [{{caller: hi}}]\n{body}\n")
    with pytest.raises(Exception, match=error):
        load_cases(path, "Alex", TOOL_NAMES)


# -- checks -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "fails"),
    [
        ("I built the backend there.", False),
        ("This is the thing they asked about.", False),  # "the", "this" aren't "he", "his"
        ("He built the backend.", True),
        ("That was his first job.", True),
    ],
)
def test_first_person_check_matches_whole_words_only(text, fails):
    assert (check_never_says(text, FIRST_PERSON) is not None) == fails


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("Sure. Email me at sam.lee@example.com.", 2),
        ("I use tools, e.g. Python and SQL, every day.", 1),
        ("It took 2.5 seconds", 1),
        ("Hi! How are you? Good.", 3),
    ],
)
def test_sentence_count(text, count):
    assert sentence_count(text) == count


def test_speakable_allows_spell_tags_and_emails():
    assert check_speakable("That's <spell>sam@example.com</spell>. Is that right?") is None


@pytest.mark.parametrize(
    "text",
    [
        "One. Two. Three. Four.",
        "- first point",
        "I'm **really** glad.",
        "It went up 20% this year.",
        "Python/SQL",
    ],
)
def test_unspeakable_replies_fail(text):
    assert check_speakable(text, max_sentences=3) is not None


def test_tool_check_compares_args_loosely_but_needs_the_right_value():
    spec = {"name": "save_message", "args": {"contact": "this number"}}
    assert (
        check_tool(reply_with(ToolCall("1", "save_message", {"contact": " This Number"})), spec)
        is None
    )
    assert check_tool(reply_with(ToolCall("1", "save_message", {"contact": "declined"})), spec)
    assert check_tool(reply_with(), spec)


def test_tool_check_args_contain_needs_every_word():
    spec = {"name": "save_message", "args_contain": {"body": ["backend", "afternoon"]}}
    both = ToolCall("1", "save_message", {"body": "Backend role; free any Afternoon."})
    only_new = ToolCall("1", "save_message", {"body": "Free any afternoon."})
    assert check_tool(reply_with(both), spec) is None
    assert check_tool(reply_with(only_new), spec)


def test_a_past_tool_call_sits_inside_my_turn(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text(
        "- id: x\n"
        "  conversation:\n"
        "    - caller: hi\n"
        "    - me: Got it.\n"
        "    - tool: {name: save_message, args: {contact: this number, body: hi}}\n"
        "    - me: Saved.\n"
        "    - caller: thanks\n"
        "  expect: {no_tool: true}\n"
    )
    [case] = load_cases(path, "Alex", TOOL_NAMES)
    messages = history_for(case, load_agent_config(Path("agent.config.yaml")))
    roles = [m.role for m in messages]
    # greeting, "hi", "Got it." + tool_use, tool_result, "Saved.", "thanks"
    assert roles[-5:] == ["user", "assistant", "user", "assistant", "user"]
    [use] = [p for p in messages[-4].parts if isinstance(p, ToolUsePart)]
    [result] = messages[-3].parts
    assert isinstance(result, ToolResultPart)
    assert result.tool_use_id == use.id and use.name == "save_message"


def test_no_tool_check():
    call = ToolCall("1", "save_caller_details", {"name": "Sam"})
    assert check_no_tool(reply_with(call), True)
    assert check_no_tool(reply_with(call), "save_message") is None
    assert check_no_tool(reply_with(call), "save_caller_details")


# -- judge --------------------------------------------------------------------------------------


def test_parse_verdict():
    assert parse_verdict('{"reason": "gave a recipe", "passed": false}') == (False, "gave a recipe")
    with pytest.raises(ValueError):
        parse_verdict('{"reason": "fine"}')
