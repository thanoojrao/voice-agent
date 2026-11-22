"""The tools the model can call during a session, and what happens when it does.

- save_message:        take a message for the owner, or replace it; emailed when the call ends
- save_caller_details: save or correct the caller's name, company and reason for calling
- end_session:         hang up once the closing line has been spoken

The model only sends the fields below. Who the caller is and which session this is come from
`ToolContext`, which the session fills in, so the model can never touch another caller's row.

The core doesn't rely on how a provider sends tools (the Claude provider uses strict tool
use, which can't express length limits). So every input is checked against the tool's own
schema here, limits included, before its handler runs, and any failure goes back to the
model as an `is_error` result it can recover from in conversation. Nothing in here raises
into the turn loop.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from voice_agent.core.greeting import usable_name
from voice_agent.core.interfaces import (
    Caller,
    MessageRow,
    SessionRow,
    Storage,
    ToolCall,
    ToolDef,
    ToolResultPart,
)

log = logging.getLogger(__name__)

END_REASONS = ("goodbye", "abuse", "limit")
# save_message `contact` values that aren't an address. Required, so the model has to decide
# every time: on a real call it left out an email the caller had spelled and confirmed.
THIS_NUMBER = "this number"  # call back on the number they're calling from
DECLINED = "declined"  # they wouldn't say how to reach them


@dataclass
class ToolContext:
    """What the handlers need about this session. One per session, owned by the session."""

    storage: Storage
    session: SessionRow
    caller: Caller  # kept up to date when save_caller_details runs
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    end_reason: str | None = None  # set by end_session; the session hangs up after the turn


# ---------------------------------------------------------------------------
# Definitions (what the model sees)
# ---------------------------------------------------------------------------


def _schema(properties: dict[str, dict[str, Any]], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


SAVE_MESSAGE = ToolDef(
    name="save_message",
    description=(
        "Save the caller's message for the real me. Call it once the message is confirmed and "
        "you know how to reach them. If they add to or change the message afterwards, call it "
        "again with the full updated message: it replaces the earlier one. "
        "`body` is the message in the caller's own words, tidied up. "
        "The caller's saved name and number are attached automatically: if they tell you "
        "their name now, call save_caller_details first. "
        "`contact` is how the real me should reach them: the email address or phone number "
        "they gave during this call, exactly as they confirmed it; or "
        f"\"{THIS_NUMBER}\" if they're happy to be called back on the number they're calling "
        f'from; or "{DECLINED}" if they won\'t say. When the caller is anonymous I have no '
        f'number, so "{THIS_NUMBER}" doesn\'t work: ask how to reach them.'
    ),
    input_schema=_schema(
        {
            "contact": {"type": "string", "maxLength": 200},
            "body": {"type": "string", "maxLength": 2000},
        },
        required=["contact", "body"],
    ),
)

SAVE_CALLER_DETAILS = ToolDef(
    name="save_caller_details",
    description=(
        "Save or correct who the caller is. Call it when they tell you their name, company or "
        "reason for calling, or when they say the name I have for them is wrong. Send only "
        "what they told you; fields you leave out keep what's already saved. "
        '`name` is just their name ("Sam Lee"), not a sentence.'
    ),
    input_schema=_schema(
        {
            "name": {"type": "string", "maxLength": 100},
            "company": {"type": "string", "maxLength": 100},
            "reason": {"type": "string", "maxLength": 300},
        },
        required=[],
    ),
)

END_SESSION = ToolDef(
    name="end_session",
    description=(
        "Hang up. Say your closing line first; the call ends once it has finished playing. "
        "`goodbye`: the caller is done. `abuse`: the caller is abusive after one warning. "
        "`limit`: the call has reached its time limit."
    ),
    input_schema=_schema(
        {"reason": {"type": "string", "enum": list(END_REASONS)}},
        required=["reason"],
    ),
)


def tool_defs(can_end_session: bool) -> list[ToolDef]:
    """The tools for a session, in a fixed order so they stay inside the prompt cache.

    end_session is left out when the adapter can't hang up by itself (spec D31).
    """
    tools = [SAVE_MESSAGE, SAVE_CALLER_DETAILS]
    if can_end_session:
        tools.append(END_SESSION)
    return tools


# ---------------------------------------------------------------------------
# Running a tool call
# ---------------------------------------------------------------------------


class ToolInputError(ValueError):
    """The input doesn't match the schema. The message is shown to the model."""


Handler = Callable[[dict[str, str], ToolContext], Awaitable[dict[str, Any]]]


async def run_tool(call: ToolCall, ctx: ToolContext, tools: list[ToolDef]) -> ToolResultPart:
    """Check the input, run the handler, and turn the outcome into a tool result.

    `tools` is the list offered this session; a call to anything else is refused.
    """
    tool = next((t for t in tools if t.name == call.name), None)
    if tool is None:
        return _error(call, f"unknown tool {call.name!r}")
    try:
        args = check_input(tool.input_schema, call.input)
        result = await _HANDLERS[tool.name](args, ctx)
    except ToolInputError as e:
        return _error(call, str(e))
    except Exception:
        log.exception("tool %s failed (session %s)", call.name, ctx.session.id)
        return _error(call, "something went wrong on my side")
    return ToolResultPart(tool_use_id=call.id, content=json.dumps(result))


def _error(call: ToolCall, message: str) -> ToolResultPart:
    return ToolResultPart(tool_use_id=call.id, content=message, is_error=True)


def check_input(schema: dict[str, Any], raw: object) -> dict[str, str]:
    """Validate against the small subset of JSON Schema the tools use: an object of string
    properties with required, enum and maxLength. Strings are trimmed, and a blank string
    counts as missing. Returns only the fields that have a value."""
    if not isinstance(raw, dict):
        raise ToolInputError("input must be an object")
    properties: dict[str, dict[str, Any]] = schema["properties"]
    unknown = sorted(set(raw) - set(properties))
    if unknown:
        raise ToolInputError(f"unknown field(s): {', '.join(unknown)}")

    args: dict[str, str] = {}
    for key, rules in properties.items():
        value = raw.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ToolInputError(f"{key} must be a string")
        value = value.strip()
        if not value:
            continue
        if len(value) > rules.get("maxLength", len(value)):
            raise ToolInputError(f"{key} is too long (max {rules['maxLength']} characters)")
        if "enum" in rules and value not in rules["enum"]:
            raise ToolInputError(f"{key} must be one of: {', '.join(rules['enum'])}")
        args[key] = value

    missing = [k for k in schema["required"] if k not in args]
    if missing:
        raise ToolInputError(f"missing: {', '.join(missing)}")
    return args


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def save_message(args: dict[str, str], ctx: ToolContext) -> dict[str, Any]:
    """Save the session's message, or replace it: one message per session, emailed by the
    session when the call ends (core/session.py). So the caller can add to it or correct it
    without a second email, and sending never slows the call down. A message is scoped to
    this session by `ctx.session`; the model never names one, so it can't touch another.

    Who it's from comes from the caller record, never from the model: the name saved via
    save_caller_details, and the number the call came from (the email shows it from
    `ctx.caller`). `contact` is required: an address the caller gave, THIS_NUMBER (stored
    as no extra contact, since the number is on the caller record) or DECLINED. An
    anonymous caller has no number, so THIS_NUMBER is refused until the model has asked
    ("declined" is accepted, so the caller can still say no).
    """
    contact: str | None = args["contact"]
    if contact.lower() == DECLINED:
        contact = DECLINED
    elif contact.lower() == THIS_NUMBER:
        if ctx.caller.address is None:
            raise ToolInputError(
                "The caller is anonymous, so I have no number for them. Ask how the real me "
                f'can reach them (phone or email). If they won\'t say, use "{DECLINED}".'
            )
        contact = None
    msg = MessageRow(
        body=args["body"],
        created_at=ctx.clock(),
        name=ctx.caller.name,
        contact=contact,
    )
    await ctx.storage.save_message(ctx.session.id, msg)
    return {"saved": True}


async def save_caller_details(args: dict[str, str], ctx: ToolContext) -> dict[str, Any]:
    if not args:
        raise ToolInputError("nothing to save: give at least one of name, company, reason")
    name = args.get("name")
    if name is not None:
        name = usable_name(name)
        if name is None:
            raise ToolInputError(
                f"{args['name']!r} doesn't look like a name. Ask the caller for just their name."
            )
    company, reason = args.get("company"), args.get("reason")

    await ctx.storage.set_caller_details(ctx.caller.id, name, company, reason)
    ctx.caller = replace(
        ctx.caller,
        name=name or ctx.caller.name,
        company=company or ctx.caller.company,
        reason=reason or ctx.caller.reason,
    )
    return {"ok": True}


async def end_session(args: dict[str, str], ctx: ToolContext) -> dict[str, Any]:
    ctx.end_reason = args["reason"]
    return {"ok": True}


_HANDLERS: dict[str, Handler] = {
    SAVE_MESSAGE.name: save_message,
    SAVE_CALLER_DETAILS.name: save_caller_details,
    END_SESSION.name: end_session,
}
