"""Tool handlers against real SQLite storage and the fake notifier."""

import json
from datetime import UTC, datetime

import pytest

from tests.fakes import FakeNotifier
from voice_agent.core.interfaces import SessionRow, SessionStart, ToolCall, ToolDef
from voice_agent.core.tools import (
    END_SESSION,
    SAVE_CALLER_DETAILS,
    SAVE_MESSAGE,
    ToolContext,
    run_tool,
    tool_defs,
)
from voice_agent.storage.sqlite import SqliteStorage

NOW = datetime(2026, 11, 15, 15, 0, tzinfo=UTC)
NUMBER = "+15551234567"
ALL_TOOLS = tool_defs(can_end_session=True)


@pytest.fixture
async def store(tmp_path):
    s = await SqliteStorage.open(tmp_path / "agent.db", now=lambda: NOW)
    yield s
    await s.close()


async def make_ctx(store, number: str | None = NUMBER, notifier=None) -> ToolContext:
    caller = await store.get_or_create_caller("twilio", number)
    start = SessionStart("twilio", "CA1", number, 8000)
    session_id = await store.create_session(caller.id, start)
    session = SessionRow(session_id, caller.id, "twilio", "CA1", NOW, None, None, None)
    return ToolContext(store, notifier or FakeNotifier(), session, caller, clock=lambda: NOW)


async def call(ctx, tool_name, tools: list[ToolDef] = ALL_TOOLS, **args):
    return await run_tool(ToolCall("tu_1", tool_name, args), ctx, tools)


async def stored_caller(store, ctx):
    return await store.get_or_create_caller("twilio", ctx.caller.address)


def message_rows(store):
    return store._db.execute("SELECT name, contact, body, notified_at FROM messages")


# -- definitions -------------------------------------------------------------


def test_tool_list_is_fixed_and_end_session_needs_the_capability():
    assert [t.name for t in ALL_TOOLS] == ["save_message", "save_caller_details", "end_session"]
    assert END_SESSION not in tool_defs(can_end_session=False)


@pytest.mark.parametrize("tool", [SAVE_MESSAGE, SAVE_CALLER_DETAILS, END_SESSION])
def test_schemas_are_strict_objects(tool):
    assert tool.input_schema["additionalProperties"] is False
    assert set(tool.input_schema["required"]) <= set(tool.input_schema["properties"])


# -- save_message --------------------------------------------------------------


async def test_save_message_saves_emails_and_marks_notified(store):
    notifier = FakeNotifier()
    ctx = await make_ctx(store, notifier=notifier)
    await call(ctx, "save_caller_details", name="Sam")
    result = await call(ctx, "save_message", body="Call me about the role.")
    assert not result.is_error
    assert json.loads(result.content) == {"saved": True}

    [(msg, caller, session)] = notifier.messages
    assert (msg.name, msg.contact, msg.body) == ("Sam", None, "Call me about the role.")
    assert caller.address == NUMBER  # the email shows the number from the caller record
    assert msg.id is not None and caller.id == ctx.caller.id and session == ctx.session
    rows = await (await message_rows(store)).fetchall()
    assert rows[0][3] is not None  # notified_at set


async def test_name_comes_from_the_record_not_the_model(store):
    ctx = await make_ctx(store)
    result = await call(ctx, "save_message", name="Somebody Else", body="Hello")
    assert result.is_error and "unknown field" in result.content


async def test_message_without_a_name_is_still_saved(store):
    ctx = await make_ctx(store)
    result = await call(ctx, "save_message", body="Liked your talk.")
    assert not result.is_error
    rows = await (await message_rows(store)).fetchall()
    assert rows[0][:3] == (None, None, "Liked your talk.")


async def test_known_caller_can_add_another_contact(store):
    ctx = await make_ctx(store)
    await call(ctx, "save_message", contact="sam@example.com", body="Hello")
    rows = await (await message_rows(store)).fetchall()
    assert rows[0][1] == "sam@example.com"


async def test_anonymous_caller_must_be_asked_for_contact(store):
    ctx = await make_ctx(store, number=None)
    result = await call(ctx, "save_message", body="Hello")
    assert result.is_error and "anonymous" in result.content
    assert await (await message_rows(store)).fetchall() == []

    await call(ctx, "save_message", contact="sam@example.com", body="Hello")
    rows = await (await message_rows(store)).fetchall()
    assert rows[0][:2] == (None, "sam@example.com")


async def test_anonymous_caller_can_decline_contact(store):
    ctx = await make_ctx(store, number=None)
    result = await call(ctx, "save_message", contact="declined", body="Hello")
    assert not result.is_error
    rows = await (await message_rows(store)).fetchall()
    assert rows[0][1] == "declined"


async def test_email_failure_still_saves_message(store):
    ctx = await make_ctx(store, notifier=FakeNotifier(fail=True))
    result = await call(ctx, "save_message", body="Hello")
    assert not result.is_error  # the caller's message is safe; the owner sees it unsent
    rows = await (await message_rows(store)).fetchall()
    assert rows[0][2:] == ("Hello", None)


async def test_storage_failure_is_an_error_result(store):
    ctx = await make_ctx(store)
    await store.close()
    result = await call(ctx, "save_message", body="Hello")
    assert result.is_error and "went wrong" in result.content


# -- save_caller_details -------------------------------------------------------


async def test_details_saved_and_context_updated(store):
    ctx = await make_ctx(store)
    result = await call(ctx, "save_caller_details", name="  Sam  Lee ", company="Acme")
    assert not result.is_error
    saved = await stored_caller(store, ctx)
    assert (saved.name, saved.company) == ("Sam Lee", "Acme")
    assert (ctx.caller.name, ctx.caller.company) == ("Sam Lee", "Acme")


async def test_correcting_name_keeps_other_fields(store):
    ctx = await make_ctx(store)
    await call(ctx, "save_caller_details", name="Sam", company="Acme", reason="internships")
    await call(ctx, "save_caller_details", name="Priya")
    saved = await stored_caller(store, ctx)
    assert (saved.name, saved.company, saved.reason) == ("Priya", "Acme", "internships")
    assert ctx.caller.company == "Acme"


async def test_bad_name_rejected_and_nothing_saved(store):
    ctx = await make_ctx(store)
    result = await call(ctx, "save_caller_details", name="um it's sam from acme", company="Acme")
    assert result.is_error and "doesn't look like a name" in result.content
    assert (await stored_caller(store, ctx)).company is None


async def test_empty_details_rejected(store):
    ctx = await make_ctx(store)
    result = await call(ctx, "save_caller_details", name="   ")
    assert result.is_error and "nothing to save" in result.content


# -- end_session ---------------------------------------------------------------


async def test_end_session_sets_reason(store):
    ctx = await make_ctx(store)
    result = await call(ctx, "end_session", reason="goodbye")
    assert not result.is_error and ctx.end_reason == "goodbye"


async def test_end_session_not_offered_cannot_be_called(store):
    ctx = await make_ctx(store)
    result = await call(
        ctx, "end_session", tools=tool_defs(can_end_session=False), reason="goodbye"
    )
    assert result.is_error and ctx.end_reason is None


# -- input checks --------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "args", "error"),
    [
        ("end_session", {"reason": "bored"}, "must be one of"),
        ("end_session", {}, "missing: reason"),
        ("save_message", {"body": ""}, "missing: body"),
        ("save_message", {"body": 42}, "must be a string"),
        ("save_message", {"body": "x" * 2001}, "too long"),
        ("save_message", {"body": "hi", "phone": "123"}, "unknown field"),
        ("transfer_call", {}, "unknown tool"),
    ],
)
async def test_bad_input_is_an_error_result(store, name, args, error):
    ctx = await make_ctx(store)
    result = await call(ctx, name, **args)
    assert result.is_error and error in result.content
    assert result.tool_use_id == "tu_1"


async def test_input_that_is_not_an_object(store):
    ctx = await make_ctx(store)
    result = await run_tool(ToolCall("tu_1", "end_session", "oops"), ctx, ALL_TOOLS)  # type: ignore[arg-type]
    assert result.is_error and "must be an object" in result.content
