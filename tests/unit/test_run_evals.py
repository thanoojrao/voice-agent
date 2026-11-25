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
    get_reply,
    history_for,
    load_cases,
    parse_verdict,
    sentence_count,
    system_for,
)
from voice_agent.app.config import load_agent_config
from voice_agent.core.interfaces import (
    Done,
    Message,
    TextDelta,
    TextPart,
    ToolCall,
    ToolResultPart,
    ToolUsePart,
)
from voice_agent.core.tools import tool_defs
from voice_agent.core.turn import MAX_TOOL_ROUNDS

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


def test_a_case_can_give_the_last_calls_summary_and_message(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text(
        "- id: x\n"
        "  caller: {name: Sam, last_summary: 'Asked {owner} for a call.', "
        "last_message: 'Call me.'}\n"
        "  conversation: [{caller: hi}]\n"
        "  expect: {no_tool: true}\n",
        encoding="utf-8",
    )
    [case] = load_cases(path, "Alex", TOOL_NAMES)
    assert (case.last_summary, case.last_message) == ("Asked Alex for a call.", "Call me.")

    class Prompt:  # records the call context instead of building the whole prompt
        def system(self, call):
            self.call = call
            return []

    prompt = Prompt()
    system_for(case, prompt)  # type: ignore[arg-type]
    assert (prompt.call.last_summary, prompt.call.last_message) == (
        "Asked Alex for a call.",
        "Call me.",
    )
    assert prompt.call.last_call_date is not None


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


def test_a_closing_question_doesnt_count_but_a_fourth_sentence_does():
    assert check_speakable("One. Two. Three. Want to hear more?", max_sentences=3) is None
    assert check_speakable("One. Two. Three. Four?", max_sentences=3) is None
    assert check_speakable("One? Two. Three. Four.", max_sentences=3)
    assert check_speakable("One. Two. Three. Four. More?", max_sentences=3)


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


# -- the bot's whole turn -----------------------------------------------------------------------


class ScriptedLLM:
    """Plays back one list of events per request, and keeps the messages each one saw."""

    def __init__(self, *rounds):
        self.rounds = list(rounds)
        self.seen = []

    async def stream_reply(self, system, history, tools):
        self.seen.append(history)
        for event in self.rounds.pop(0):
            yield event

    async def summarize(self, transcript):
        raise NotImplementedError

    async def prewarm(self, system, tools):
        pass


def _done():
    return Done("end_turn", input_tokens=10, output_tokens=5, cache_read_tokens=100)


async def test_a_turn_carries_on_after_a_tool_call():
    save = ToolCall("t1", "save_caller_details", {"name": "Sam"})
    llm = ScriptedLLM(
        [TextDelta("Thanks, Sam."), save, _done()],
        [TextDelta("Thanks, Sam. What can I help with?"), _done()],
    )
    reply = await get_reply(llm, [], [Message("user", (TextPart("Sam."),))], [])
    # Both requests are what the caller hears: the repeat shows.
    assert reply.text == "Thanks, Sam. Thanks, Sam. What can I help with?"
    assert reply.calls == [save]
    assert (reply.input_tokens, reply.cached_tokens, reply.output_tokens) == (20, 200, 10)
    said, results = llm.seen[1][-2:]
    assert said.parts == (
        TextPart("Thanks, Sam."),
        ToolUsePart("t1", "save_caller_details", {"name": "Sam"}),
    )
    [result] = results.parts
    assert isinstance(result, ToolResultPart) and result.tool_use_id == "t1"


async def test_a_turn_stops_at_end_session():
    end = ToolCall("t1", "end_session", {"reason": "goodbye"})
    llm = ScriptedLLM([TextDelta("Bye!"), end, _done()])
    reply = await get_reply(llm, [], [], [])
    assert (reply.text, reply.calls, len(llm.seen)) == ("Bye!", [end], 1)


async def test_a_turn_that_keeps_calling_tools_is_cut_short():
    again = [ToolCall("t", "save_caller_details", {"name": "Sam"}), _done()]
    llm = ScriptedLLM(*[again] * 10)
    reply = await get_reply(llm, [], [], [])
    assert len(llm.seen) == MAX_TOOL_ROUNDS + 1 and reply.text == ""
