"""Email through Gmail: the message a caller left (sent when the call ends), and alerts when
a call fails. Implements core.interfaces.Notifier.

Gmail's SMTP server on port 465, encrypted from the first byte, signed in with an app
password (design doc §2). smtplib blocks, so each send runs in a worker thread and never
holds up a live call. There's no retry here: a failed message email leaves the message
saved and marked unsent, and it's sent again at the next startup.

The app password is only read when signing in. smtplib's errors carry the server's reply,
never the password, so they're safe to log.

Message emails are plain text plus an HTML version: the message in a box, the details in a
table, and the caller's number and email as tap-to-call and tap-to-email links. Alerts are
plain text.
"""

from __future__ import annotations

import asyncio
import html
import re
import smtplib
import ssl
from collections.abc import Callable
from dataclasses import dataclass
from datetime import tzinfo
from email.message import EmailMessage
from email.utils import formataddr
from typing import Any

from pydantic import SecretStr

from voice_agent.core.interfaces import Caller, MessageRow, SessionRow
from voice_agent.core.tools import DECLINED

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465  # TLS from the start (587 would be STARTTLS)
SMTP_TIMEOUT_S = 30.0
SENDER_NAME = "Voice agent"  # shown as the sender in the inbox
SUBJECT_PREFIX = "Voice agent: "  # makes a Gmail filter easy; alerts already start with it

_EMAIL_ADDRESS = re.compile(r"[\w.%+-]+@[\w-]+(\.[\w-]+)+")
_PHONE = re.compile(r"\+?[\d\s().-]{7,20}")


@dataclass(frozen=True, slots=True)
class Email:
    subject: str
    text: str
    html: str | None = None  # sent alongside the text, for mail apps that show it


@dataclass(frozen=True, slots=True)
class _Detail:
    label: str
    value: str
    link: str | None = None  # tel: or mailto:, so it's one tap on a phone


def message_email(msg: MessageRow, caller: Caller, session: SessionRow, tz: tzinfo) -> Email:
    """The email for a message a caller left: plain text, plus an HTML version.

    Everything here except the labels came from the caller (through speech recognition
    and the model), so the HTML escapes all of it, and only builds links from values
    that are plainly an email address or a phone number."""
    name = msg.name or caller.name
    number = caller.address
    who = name or number or "an anonymous caller"
    subject = _one_line(f"{SUBJECT_PREFIX}message from {who}")

    reach, reach_link = _how_to_reach(msg.contact, number)
    details = [
        _Detail("From", name or "Name not given"),
        _Detail("Phone", number, _link(number)) if number else _Detail("Phone", "withheld"),
        _Detail("Reach them", reach, reach_link),
    ]
    if caller.company:
        details.append(_Detail("Company", caller.company))
    if caller.reason:
        details.append(_Detail("Calling about", caller.reason))
    details.append(_Detail("Called", _when(session, tz)))
    ended = session.end_reason or "unknown"
    footer = f"Call #{session.id}, ended: {ended}. Sent by {SENDER_NAME}."

    heading = f"New message from {who}"
    lines = [heading, "-" * len(heading), f'"{msg.body}"', ""]
    lines += [f"{d.label + ':':<15}{d.value}" for d in details]
    if session.summary:
        lines += ["", f"Call summary: {session.summary}"]
    lines += ["", footer]
    text = "\n".join(lines) + "\n"
    return Email(subject, text, _html(heading, msg.body, details, session.summary, footer))


def _html(heading: str, body: str, details: list[_Detail], summary: str | None, footer: str) -> str:
    e = html.escape
    rows = "".join(
        f'<tr><td style="padding:4px 16px 4px 0;color:#666;white-space:nowrap;'
        f'vertical-align:top">{e(d.label)}</td><td style="padding:4px 0">'
        + (f'<a href="{e(d.link)}">{e(d.value)}</a>' if d.link else e(d.value))
        + "</td></tr>"
        for d in details
    )
    summary_html = (
        f'<p style="margin:16px 0 0;color:#444"><b>Call summary:</b> {e(summary)}</p>'
        if summary
        else ""
    )
    return (
        '<div style="font-family:Arial,Helvetica,sans-serif;font-size:15px;color:#222;'
        'max-width:560px">'
        f'<h2 style="font-size:18px;margin:0 0 12px">{e(heading)}</h2>'
        '<div style="background:#f3f6fb;border-left:4px solid #3b6fd8;padding:12px 14px;'
        f'margin:0 0 16px;white-space:pre-wrap">{e(body)}</div>'
        f'<table style="border-collapse:collapse">{rows}</table>'
        f"{summary_html}"
        f'<p style="margin:20px 0 0;font-size:12px;color:#888">{e(footer)}</p>'
        "</div>"
    )


def _how_to_reach(contact: str | None, number: str | None) -> tuple[str, str | None]:
    if contact == DECLINED:
        return "they didn't want to say", None
    if contact:
        return contact, _link(contact)
    if number:
        return f"call back on {number}", _link(number)
    return "not given", None


def _link(value: str) -> str | None:
    """A mailto: or tel: link, only for a value that is clearly one or the other."""
    value = value.strip()
    if _EMAIL_ADDRESS.fullmatch(value):
        return f"mailto:{value}"
    if _PHONE.fullmatch(value):
        return "tel:" + "".join(c for c in value if c.isdigit() or c == "+")
    return None


def _when(session: SessionRow, tz: tzinfo) -> str:
    started = session.started_at.astimezone(tz)
    return started.strftime("%a %d %b %Y, %I:%M %p %Z").replace(" 0", " ")


def _one_line(text: str) -> str:
    """Headers can't contain line breaks: one would start a new header (say, a Bcc)."""
    return " ".join(text.split())


class GmailNotifier:
    def __init__(
        self,
        address: str,
        app_password: SecretStr,
        to: str,
        tz: tzinfo,
        *,
        smtp: Callable[..., Any] = smtplib.SMTP_SSL,  # an SMTP_SSL-like class; a fake in tests
    ) -> None:
        self._address = address
        self._password = app_password
        self._to = to
        self._tz = tz
        self._smtp = smtp

    async def message_left(self, msg: MessageRow, caller: Caller, session: SessionRow) -> None:
        await self._send(message_email(msg, caller, session, self._tz))

    async def alert(self, subject: str, body: str) -> None:
        await self._send(Email(_one_line(subject), body))

    async def _send(self, mail: Email) -> None:
        email = EmailMessage()
        email["From"] = formataddr((SENDER_NAME, self._address))
        email["To"] = self._to
        email["Subject"] = mail.subject
        email.set_content(mail.text)
        if mail.html:
            email.add_alternative(mail.html, subtype="html")
        await asyncio.to_thread(self._send_now, email)

    def _send_now(self, email: EmailMessage) -> None:
        # Google shows app passwords in groups of four ("abcd efgh ..."); the spaces aren't
        # part of it.
        password = "".join(self._password.get_secret_value().split())
        context = ssl.create_default_context()
        with self._smtp(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT_S, context=context) as smtp:
            smtp.login(self._address, password)
            smtp.send_message(email)
