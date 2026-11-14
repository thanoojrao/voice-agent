"""Conversation history sent to the LLM on every turn.

The history must match what was actually *said on the call*, not what the LLM generated.
When the caller interrupts, the agent's reply is cut down to the words the caller heard
(spec: Interruptions), so the model doesn't think it said things the caller never heard.

One agent turn can span several messages when a tool is used:

    assistant: "Sure, I'll pass that on."  + tool_use(save_message)
    user:      tool_result
    assistant: "Done. Anything else?"

so the turn is built up while it streams (the LLM needs to see the tool exchange before it
continues) and trimmed once at the end:

    history.begin_agent_turn()
    history.add_agent_text("Sure, I'll pass that on.")
    history.add_tool_call(call); history.add_tool_result(result)
    history.add_agent_text("Done. Anything else?")
    spoken = history.end_agent_turn(words_heard=4)   # None = heard it all

Trimming counts whitespace-separated words, the same unit as the TTS word marks. Tool
calls are never removed: they ran, whether or not the caller heard the words around them.

The history also keeps the shape the LLM API expects: it starts with a user message and
roles alternate (two caller turns in a row are merged into one message).
"""

from __future__ import annotations

from typing import Literal

from voice_agent.core.interfaces import Message, TextPart, ToolResultPart, ToolUsePart

# The greeting is the agent's first line, but the history must start with a user message.
CALL_CONNECTED = "(The call has connected.)"
# Appended to a reply the caller cut off, so the model knows it was interrupted.
CUT_OFF = "—"

Part = TextPart | ToolUsePart | ToolResultPart
Role = Literal["user", "assistant"]


class History:
    def __init__(self) -> None:
        self._messages: list[Message] = []
        self._turn_start: int | None = None  # index of the current agent turn's first message

    @property
    def messages(self) -> list[Message]:
        """A copy, safe to hand to the LLM while the history keeps changing."""
        return list(self._messages)

    @property
    def in_agent_turn(self) -> bool:
        return self._turn_start is not None

    # -- caller ---------------------------------------------------------------

    def add_caller(self, text: str) -> None:
        if self.in_agent_turn:
            raise RuntimeError("end the agent turn before adding the caller's turn")
        self._append("user", TextPart(text))

    # -- agent ----------------------------------------------------------------

    def add_agent(self, text: str) -> None:
        """A complete line the caller hears in full, such as the greeting."""
        self.begin_agent_turn()
        self.add_agent_text(text)
        self.end_agent_turn()

    def begin_agent_turn(self) -> None:
        if self.in_agent_turn:
            raise RuntimeError("agent turn already started")
        if not self._messages:
            self._append("user", TextPart(CALL_CONNECTED))
        self._turn_start = len(self._messages)

    def add_agent_text(self, text: str) -> None:
        """Text sent to TTS, a sentence at a time."""
        self._require_turn()
        self._append("assistant", TextPart(text))

    def add_tool_call(self, call: ToolUsePart) -> None:
        self._require_turn()
        self._append("assistant", call)

    def add_tool_result(self, result: ToolResultPart) -> None:
        self._require_turn()
        self._append("user", result)

    def end_agent_turn(self, words_heard: int | None = None) -> str:
        """Finish the turn, trimmed to `words_heard` words if the caller interrupted.

        Returns the text the caller heard, for the stored transcript.
        """
        start = self._require_turn()
        self._turn_start = None
        turn = self._messages[start:]
        del self._messages[start:]

        budget = words_heard if words_heard is not None else _count_words(turn)
        interrupted = budget < _count_words(turn)
        heard: list[str] = []
        for msg in turn:
            parts: list[Part] = []
            for part in msg.parts:
                if isinstance(part, TextPart):
                    words = part.text.split()[:budget]
                    budget -= len(words)
                    if words:
                        heard.append(" ".join(words))
                        parts.append(TextPart(" ".join(words)))
                else:
                    parts.append(part)
            for part in parts:
                self._append(msg.role, part)

        if interrupted and heard:
            self._mark_cut_off()
            heard[-1] += CUT_OFF
        return " ".join(heard)

    # -- internals ------------------------------------------------------------

    def _require_turn(self) -> int:
        if self._turn_start is None:
            raise RuntimeError("no agent turn in progress; call begin_agent_turn() first")
        return self._turn_start

    def _append(self, role: Role, part: Part) -> None:
        """Add a part, merging into the last message when it has the same role.

        During an agent turn, never merge into a message from before the turn, so the
        turn's messages stay separate until end_agent_turn() trims them.
        """
        floor = self._turn_start if self._turn_start is not None else 0
        last = self._messages[-1] if len(self._messages) > floor else None
        if last is None or last.role != role:
            self._messages.append(Message(role=role, parts=(part,)))
            return
        parts = list(last.parts)
        if isinstance(part, TextPart) and parts and isinstance(parts[-1], TextPart):
            parts[-1] = TextPart(f"{parts[-1].text} {part.text}")
        else:
            parts.append(part)
        self._messages[-1] = Message(role=last.role, parts=tuple(parts))

    def _mark_cut_off(self) -> None:
        """Add the cut-off mark to the last text the agent said."""
        for i in range(len(self._messages) - 1, -1, -1):
            msg = self._messages[i]
            if msg.role != "assistant":
                continue
            parts = list(msg.parts)
            for j in range(len(parts) - 1, -1, -1):
                part = parts[j]
                if isinstance(part, TextPart):
                    parts[j] = TextPart(part.text + CUT_OFF)
                    self._messages[i] = Message(role="assistant", parts=tuple(parts))
                    return


def _count_words(messages: list[Message]) -> int:
    return sum(
        len(part.text.split())
        for msg in messages
        for part in msg.parts
        if isinstance(part, TextPart)
    )
