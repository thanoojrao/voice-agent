import pytest

from voice_agent.core.history import CALL_CONNECTED, CUT_OFF, History
from voice_agent.core.interfaces import Message, TextPart, ToolResultPart, ToolUsePart

CALL = ToolUsePart(id="t1", name="save_message", input={"body": "call me"})
RESULT = ToolResultPart(tool_use_id="t1", content='{"saved": true}')


def user(*parts) -> Message:
    return Message("user", tuple(TextPart(p) if isinstance(p, str) else p for p in parts))


def agent(*parts) -> Message:
    return Message("assistant", tuple(TextPart(p) if isinstance(p, str) else p for p in parts))


def assert_alternates(h: History) -> None:
    msgs = h.messages
    assert msgs[0].role == "user"
    assert all(a.role != b.role for a, b in zip(msgs, msgs[1:], strict=False))


def test_greeting_starts_with_call_connected_placeholder():
    h = History()
    h.add_agent("Hi, this is Thanooj's AI.")
    assert h.messages == [user(CALL_CONNECTED), agent("Hi, this is Thanooj's AI.")]


def test_two_agent_lines_in_a_row_merge():
    h = History()
    h.add_agent("Hi, this is Thanooj's AI.")
    h.add_agent("Who am I speaking with?")
    assert h.messages[1:] == [agent("Hi, this is Thanooj's AI. Who am I speaking with?")]


def test_full_turn_kept_as_said():
    h = History()
    h.add_caller("What are you working on?")
    h.begin_agent_turn()
    h.add_agent_text("A voice agent.")
    h.add_agent_text("It answers my phone.")
    spoken = h.end_agent_turn()
    assert spoken == "A voice agent. It answers my phone."
    assert h.messages == [
        user("What are you working on?"),
        agent("A voice agent. It answers my phone."),
    ]


def test_interrupted_turn_trimmed_to_words_heard():
    h = History()
    h.add_caller("What are you working on?")
    h.begin_agent_turn()
    h.add_agent_text("A voice agent.")
    h.add_agent_text("It answers my phone.")
    spoken = h.end_agent_turn(words_heard=4)
    assert spoken == "A voice agent. It" + CUT_OFF
    assert h.messages[-1] == agent("A voice agent. It" + CUT_OFF)


def test_words_heard_equal_to_total_is_not_interrupted():
    h = History()
    h.add_caller("Hi")
    h.begin_agent_turn()
    h.add_agent_text("Hello there.")
    assert h.end_agent_turn(words_heard=2) == "Hello there."


def test_nothing_heard_drops_the_reply_and_merges_caller_turns():
    h = History()
    h.add_caller("What are you working on?")
    h.begin_agent_turn()
    h.add_agent_text("A voice agent.")
    assert h.end_agent_turn(words_heard=0) == ""
    h.add_caller("Actually, wait.")
    assert h.messages[0].parts == (TextPart("What are you working on? Actually, wait."),)


def test_tool_exchange_kept_in_order():
    h = History()
    h.add_caller("Tell him to call me.")
    h.begin_agent_turn()
    h.add_agent_text("Sure, I'll pass that on.")
    h.add_tool_call(CALL)
    h.add_tool_result(RESULT)
    h.add_agent_text("Done. Anything else?")
    h.end_agent_turn()
    assert h.messages == [
        user("Tell him to call me."),
        agent("Sure, I'll pass that on.", CALL),
        user(RESULT),
        agent("Done. Anything else?"),
    ]
    assert_alternates(h)


def test_llm_sees_tool_exchange_before_turn_ends():
    h = History()
    h.add_caller("Tell him to call me.")
    h.begin_agent_turn()
    h.add_tool_call(CALL)
    h.add_tool_result(RESULT)
    assert h.messages[-2:] == [agent(CALL), user(RESULT)]


def test_interrupt_after_tool_keeps_tool_and_drops_unheard_text():
    h = History()
    h.add_caller("Tell him to call me.")
    h.begin_agent_turn()
    h.add_agent_text("Sure, I'll pass that on.")
    h.add_tool_call(CALL)
    h.add_tool_result(RESULT)
    h.add_agent_text("Done. Anything else?")
    spoken = h.end_agent_turn(words_heard=2)
    assert spoken == "Sure, I'll" + CUT_OFF
    assert h.messages[1:] == [agent("Sure, I'll" + CUT_OFF, CALL), user(RESULT)]
    h.add_caller("No, that's all.")
    assert h.messages[-1] == user(RESULT, "No, that's all.")
    assert_alternates(h)


def test_interrupt_before_any_words_keeps_tool_call():
    h = History()
    h.add_caller("Bye!")
    h.begin_agent_turn()
    h.add_tool_call(CALL)
    h.add_tool_result(RESULT)
    h.add_agent_text("Talk soon.")
    assert h.end_agent_turn(words_heard=0) == ""
    assert h.messages[1:] == [agent(CALL), user(RESULT)]


def test_messages_is_a_copy():
    h = History()
    h.add_caller("Hi")
    h.messages.clear()
    assert len(h.messages) == 1


def test_misuse_raises():
    h = History()
    with pytest.raises(RuntimeError):
        h.add_agent_text("no turn")
    with pytest.raises(RuntimeError):
        h.end_agent_turn()
    h.begin_agent_turn()
    with pytest.raises(RuntimeError):
        h.begin_agent_turn()
    with pytest.raises(RuntimeError):
        h.add_caller("talking over")
