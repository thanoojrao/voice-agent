"""The admin pages: calls, messages and callers, read-only, behind a password (design §13).

Laid out like a phone's call app. On a phone each page shows one screen: a list, or one
call or caller with a back link. On a wide screen the same page shows the list beside the
open item. So every page carries both, and the stylesheet decides; there's no JavaScript.

    /admin                       calls
    /admin/calls/{id}            calls, with that call open
    /admin/messages              messages
    /admin/messages/{call id}    messages, with that message's call open
    /admin/callers               callers
    /admin/callers/{id}          callers, with that caller open

Sign-in is HTTP Basic: the browser asks once and remembers it until it's closed. Any
username works; the password is checked against ADMIN_PASSWORD in constant time. That's
enough for one owner over HTTPS (the host forces it). Every response also tells browsers
and proxies not to keep a copy, search engines not to index it, and other sites not to
frame it, and allows no scripts at all.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, tzinfo
from pathlib import Path
from typing import Annotated, Protocol

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from pydantic import SecretStr

from voice_agent.admin import views
from voice_agent.admin.views import Bubble, Person, ReplyStats
from voice_agent.core.interfaces import Caller, MessageRow, SessionRow, TurnRow
from voice_agent.storage.sqlite import SessionListing

PREFIX = "/admin"
LIST_LIMIT = 100  # newest first; enough for one person's calls
TEMPLATES_DIR = Path(__file__).parent / "templates"
STYLESHEET = TEMPLATES_DIR / "admin.css"

REALM = "Voice agent admin"
_BASIC = HTTPBasic(realm=REALM, auto_error=False)  # auto_error=False: we answer 401 ourselves
_HEADERS = {
    "Cache-Control": "no-store",
    "X-Robots-Tag": "noindex, nofollow",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'self'; img-src 'self' data:; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
}


class AdminReads(Protocol):
    """The storage reads the pages use (storage.sqlite.SqliteStorage)."""

    async def recent_sessions(
        self, limit: int = ..., caller_id: int | None = ...
    ) -> list[SessionListing]: ...
    async def get_session(self, session_id: int) -> tuple[SessionRow, Caller] | None: ...
    async def session_turns(self, session_id: int) -> list[TurnRow]: ...
    async def session_message(self, session_id: int) -> MessageRow | None: ...
    async def recent_messages(
        self, limit: int = ...
    ) -> list[tuple[MessageRow, Caller, SessionRow]]: ...
    async def recent_callers(self, limit: int = ...) -> list[Caller]: ...
    async def get_caller(self, caller_id: int) -> Caller | None: ...
    async def unsent_messages(self) -> list[tuple[MessageRow, Caller, SessionRow]]: ...


# -- what the templates get ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Row:
    """One row of a list: a call, a message or a caller."""

    href: str
    person: Person | None  # None: no avatar (a caller's own call list)
    title: str
    sub: str = ""
    side: str = ""  # the time, on the right
    preview: str = ""  # a message's first lines
    status: str | None = None  # "Emailed 6:20 PM"; None for a message not emailed yet
    has_message: bool = False
    problem: bool = False  # shown in red
    current: bool = False  # the open item, highlighted on a wide screen


@dataclass(frozen=True, slots=True)
class Group:
    label: str  # "Today", "Yesterday", "Sun 22 Nov"
    rows: list[Row]


@dataclass(frozen=True, slots=True)
class CallDetail:
    person: Person
    caller_href: str
    when: str
    length: str
    ended: str
    problem: bool
    summary: str | None
    message: MessageRow | None
    reach: str | None
    message_status: str | None
    stats: ReplyStats | None
    bubbles: list[Bubble]


@dataclass(frozen=True, slots=True)
class CallerDetail:
    person: Person
    company: str | None
    facts: list[tuple[str, str]]
    calls: list[Row]


@dataclass(slots=True)
class Page:
    tab: str  # "calls", "messages" or "callers"
    title: str
    subtitle: str
    unsent: int  # messages not emailed yet: the badge on the Messages tab
    groups: list[Group] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)
    call: CallDetail | None = None
    caller: CallerDetail | None = None
    back: tuple[str, str] | None = None  # (href, label) on a phone, when an item is open


# -- routes -------------------------------------------------------------------------------


def admin_router(
    reads: Callable[[], AdminReads],
    password: SecretStr,
    tz: tzinfo,
    now: Callable[[], datetime],
) -> APIRouter:
    """The admin pages (built in main.py). `reads` returns the storage, which only exists
    once the app has started."""
    if not password.get_secret_value():
        raise ValueError("ADMIN_PASSWORD is empty")
    templates = Jinja2Templates(directory=TEMPLATES_DIR)  # autoescapes .html
    router = APIRouter(prefix=PREFIX, dependencies=[Depends(_PasswordCheck(password))])

    async def render(request: Request, page: Page) -> HTMLResponse:
        response = templates.TemplateResponse(request, "page.html", {"page": page})
        response.headers.update(_HEADERS)
        return response

    def today() -> date:
        return now().astimezone(tz).date()

    @router.get("", response_class=HTMLResponse)
    async def calls(request: Request) -> HTMLResponse:
        return await render(request, await _calls_page(reads(), tz, today(), None))

    @router.get("/calls/{call_id}", response_class=HTMLResponse)
    async def call(request: Request, call_id: int) -> HTMLResponse:
        page = await _calls_page(reads(), tz, today(), call_id)
        page.call = await _call_detail(reads(), call_id, tz, today())
        page.back = (PREFIX, "Calls")
        return await render(request, page)

    @router.get("/messages", response_class=HTMLResponse)
    async def messages(request: Request) -> HTMLResponse:
        return await render(request, await _messages_page(reads(), tz, today(), None))

    @router.get("/messages/{call_id}", response_class=HTMLResponse)
    async def message(request: Request, call_id: int) -> HTMLResponse:
        page = await _messages_page(reads(), tz, today(), call_id)
        page.call = await _call_detail(reads(), call_id, tz, today())
        page.back = (f"{PREFIX}/messages", "Messages")
        return await render(request, page)

    @router.get("/callers", response_class=HTMLResponse)
    async def callers(request: Request) -> HTMLResponse:
        return await render(request, await _callers_page(reads(), None))

    @router.get("/callers/{caller_id}", response_class=HTMLResponse)
    async def caller(request: Request, caller_id: int) -> HTMLResponse:
        page = await _callers_page(reads(), caller_id)
        page.caller = await _caller_detail(reads(), caller_id, tz, today())
        page.back = (f"{PREFIX}/callers", "Callers")
        return await render(request, page)

    @router.get("/admin.css")
    async def stylesheet() -> Response:
        css = STYLESHEET.read_text(encoding="utf-8")
        return Response(css, media_type="text/css", headers=_HEADERS)

    return router


class _PasswordCheck:
    """The sign-in check, as a dependency of every admin route."""

    def __init__(self, password: SecretStr) -> None:
        self._expected = password.get_secret_value().encode()

    def __call__(
        self, credentials: Annotated[HTTPBasicCredentials | None, Depends(_BASIC)]
    ) -> None:
        given = credentials.password.encode() if credentials else b""
        if not secrets.compare_digest(given, self._expected):
            raise HTTPException(
                status_code=401, headers={"WWW-Authenticate": f'Basic realm="{REALM}"', **_HEADERS}
            )


# -- pages --------------------------------------------------------------------------------


async def _calls_page(reads: AdminReads, tz: tzinfo, today: date, open_id: int | None) -> Page:
    listing = await reads.recent_sessions(LIST_LIMIT)
    groups: list[Group] = []
    for item in listing:
        label = views.day_label(item.session.started_at, today, tz)
        row = _call_row(item, tz, open_id, f"{PREFIX}/calls")
        if groups and groups[-1].label == label:
            groups[-1].rows.append(row)
        else:
            groups.append(Group(label, [row]))
    on_today = sum(len(g.rows) for g in groups if g.label == "Today")
    with_message = sum(item.has_message for item in listing)
    subtitle = f"{on_today} today · {_count(with_message, 'message')}"
    return Page("calls", "Calls", subtitle, await _unsent(reads), groups=groups)


async def _messages_page(reads: AdminReads, tz: tzinfo, today: date, open_id: int | None) -> Page:
    found = await reads.recent_messages(LIST_LIMIT)
    rows = []
    for msg, caller, session in found:
        day = views.day_label(session.started_at, today, tz)
        rows.append(
            Row(
                href=f"{PREFIX}/messages/{session.id}",
                person=views.person(caller),
                title=views.person(caller).title,
                side=views.time_label(session.started_at, tz) if day == "Today" else day,
                preview=msg.body,
                status=views.message_status(msg, tz, today),
                current=session.id == open_id,
            )
        )
    unsent = await _unsent(reads)
    subtitle = f"{_count(len(rows), 'message')} · {unsent} not emailed yet"
    return Page("messages", "Messages", subtitle, unsent, rows=rows)


async def _callers_page(reads: AdminReads, open_id: int | None) -> Page:
    found = await reads.recent_callers(LIST_LIMIT)
    rows = []
    for c in found:
        p = views.person(c)
        sub = " · ".join(x for x in (c.company, _count(c.session_count, "call")) if x)
        rows.append(Row(f"{PREFIX}/callers/{c.id}", p, p.title, sub, current=c.id == open_id))
    subtitle = f"{_count(len(rows), 'caller')} · most recent first"
    return Page("callers", "Callers", subtitle, await _unsent(reads), rows=rows)


async def _call_detail(reads: AdminReads, call_id: int, tz: tzinfo, today: date) -> CallDetail:
    found = await reads.get_session(call_id)
    if found is None:
        raise HTTPException(status_code=404, detail="That call isn't here.", headers=_HEADERS)
    session, caller = found
    turns = await reads.session_turns(call_id)
    msg = await reads.session_message(call_id)
    return CallDetail(
        person=views.person(caller),
        caller_href=f"{PREFIX}/callers/{caller.id}",
        when=views.when_label(session.started_at, today, tz),
        length=views.length_label(session),
        ended=views.end_label(session),
        problem=views.is_problem(session),
        summary=session.summary,
        message=msg,
        reach=views.how_to_reach(msg, caller) if msg else None,
        message_status=views.message_status(msg, tz, today) if msg else None,
        stats=views.reply_stats(turns),
        bubbles=views.transcript(turns),
    )


async def _caller_detail(
    reads: AdminReads, caller_id: int, tz: tzinfo, today: date
) -> CallerDetail:
    caller = await reads.get_caller(caller_id)
    if caller is None:
        raise HTTPException(status_code=404, detail="That caller isn't here.", headers=_HEADERS)
    facts = [
        ("Calling about", caller.reason),
        ("Calls", str(caller.session_count)),
        ("First call", views.when_label(caller.first_seen, today, tz)),
        ("Last call", views.when_label(caller.last_seen, today, tz)),
    ]
    if caller.address is None:
        facts.append(("Number", "Withheld, so each call is a new caller"))
    listing = await reads.recent_sessions(LIST_LIMIT, caller_id=caller_id)
    calls = [
        Row(
            href=f"{PREFIX}/calls/{item.session.id}",
            person=None,
            title=views.day_label(item.session.started_at, today, tz),
            sub=_call_sub(item),
            side=views.time_label(item.session.started_at, tz),
            has_message=item.has_message,
            problem=views.is_problem(item.session),
        )
        for item in listing
    ]
    return CallerDetail(
        person=views.person(caller),
        company=caller.company,
        facts=[(k, v) for k, v in facts if v],
        calls=calls,
    )


def _call_row(item: SessionListing, tz: tzinfo, open_id: int | None, base: str) -> Row:
    p = views.person(item.caller)
    return Row(
        href=f"{base}/{item.session.id}",
        person=p,
        title=p.title,
        sub=_call_sub(item),
        side=views.time_label(item.session.started_at, tz),
        has_message=item.has_message,
        problem=views.is_problem(item.session),
        current=item.session.id == open_id,
    )


def _call_sub(item: SessionListing) -> str:
    """The line under a call: its length, then how it ended if that's worth a look. A
    message is added by the template, as a coloured "Left a message"."""
    length = views.length_label(item.session)
    if item.has_message:
        return length
    s = item.session
    if views.is_problem(s) or s.end_reason in ("silence", "limit"):
        return f"{length} · {views.end_label(s)}"
    return length


async def _unsent(reads: AdminReads) -> int:
    return len(await reads.unsent_messages())


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"
