"""The opening line of a session: fixed config text, spoken with no LLM call.

The first audio has to be instant, and the AI disclosure must be worded the same for every
caller, so the opening is never generated. Which template is used depends on the caller:

| Caller                                   | Opening                                     |
| ---------------------------------------- | ------------------------------------------- |
| First session, or anonymous              | greeting, then name_question                |
| Returning, no usable name saved          | greeting                                    |
| Returning, with a usable saved name      | greeting_returning ("Hi Sam, this is ...")  |

A returning caller who declined to give their name isn't asked again. Anonymous callers
can't be recognised next time, so they're asked every time.

The warm intro ("Last time we talked about X. Want to pick that up?") is the one generated
part. This module only decides whether to ask for it; the session makes the LLM request
while the greeting plays.
"""

from __future__ import annotations

from dataclasses import dataclass

from voice_agent.core.interfaces import Caller

MAX_NAME_CHARS = 40
MAX_NAME_WORDS = 3
NAME_PUNCTUATION = frozenset(" -'’")  # ’ because speech recognition sometimes uses it


@dataclass(frozen=True, slots=True)
class GreetingTemplates:
    """The owner's wording from agent.config.yaml (checked there when the config loads).

    `greeting` and `name_question` may use {owner_name}; `greeting_returning` also uses
    {caller_name}.
    """

    owner_name: str
    greeting: str
    greeting_returning: str
    name_question: str


@dataclass(frozen=True, slots=True)
class Opening:
    text: str  # spoken as the first agent turn
    warm_intro_summary: str | None  # last session's summary to build the warm intro from


def build_opening(
    caller: Caller,
    templates: GreetingTemplates,
    *,
    last_summary: str | None,
    warm_intro: bool,
) -> Opening:
    """Choose the opening for this caller.

    `caller` is the row as read before this session was created, so session_count == 0
    means a first-time caller. `last_summary` is the previous session's summary if it's
    recent enough to mention (the caller of this function applies the age limit).
    """
    anonymous = caller.address is None
    first_session = caller.session_count == 0
    name = None if anonymous else usable_name(caller.name)

    if name:
        text = templates.greeting_returning.format(
            owner_name=templates.owner_name, caller_name=name
        )
    else:
        text = templates.greeting.format(owner_name=templates.owner_name)

    if anonymous or first_session:
        question = templates.name_question.format(owner_name=templates.owner_name)
        text = f"{text} {question}"

    offer_last_topic = warm_intro and not anonymous and not first_session and last_summary
    return Opening(text=text, warm_intro_summary=last_summary if offer_last_topic else None)


def usable_name(name: str | None) -> str | None:
    """The name tidied up if it looks like a real name, else None.

    Names come from speech recognition, so a saved one can be mangled ("um it's sam from
    acme corp"). Greeting someone with that is worse than not using their name at all.
    """
    if not name:
        return None
    name = " ".join(name.split())
    if not name or len(name) > MAX_NAME_CHARS or len(name.split()) > MAX_NAME_WORDS:
        return None
    if not name[0].isalpha():
        return None
    if not all(c.isalpha() or c in NAME_PUNCTUATION for c in name):
        return None
    return name
