"""Email through Gmail: the message a caller left (sent when the call ends), and alerts when
a call fails. Implements core.interfaces.Notifier.

Gmail's SMTP server on port 465, encrypted from the first byte, signed in with an app
password (design doc §2). smtplib blocks, so each send runs in a worker thread and never
holds up a live call. There's no retry here: a failed message email leaves the message
saved and marked unsent, and it's sent again at the next startup.

The app password is only read when signing in. smtplib's errors carry the server's reply,
never the password, so they're safe to log.

The email is plain text: it's read on a phone, and there's nothing to format.
"""

from __future__ import annotations

import asyncio
import smtplib
import ssl
from collections.abc import Callable
from datetime import tzinfo
from email.message import EmailMessage
from typing import Any

from pydantic import SecretStr

from voice_agent.core.interfaces import Caller, MessageRow, SessionRow
from voice_agent.core.tools import DECLINED

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465  # TLS from the start (587 would be STARTTLS)
SMTP_TIMEOUT_S = 30.0


def message_email(
    msg: MessageRow, caller: Caller, session: SessionRow, tz: tzinfo
) -> tuple[str, str]:
    """The subject and body of the email for a message a caller left."""
    name = msg.name or caller.name
    number = caller.address
    who = name or number or "an anonymous caller"
    subject = " ".join(f"Message from {who}".split())  # no line breaks in a header

    from_line = name or "Name not given"
    from_line += f" ({number})" if number else " (number withheld)"
    lines = [
        msg.body,
        "",
        f"From: {from_line}",
        f"How to reach them: {_how_to_reach(msg.contact, number)}",
    ]
    if caller.company:
        lines.append(f"Company: {caller.company}")
    if caller.reason:
        lines.append(f"Calling about: {caller.reason}")
    lines.append(f"Called: {_when(session, tz)}")
    if session.summary:
        lines += ["", f"Call summary: {session.summary}"]
    lines += ["", f"(Call #{session.id}, ended: {session.end_reason or 'unknown'})"]
    return subject, "\n".join(lines) + "\n"


def _how_to_reach(contact: str | None, number: str | None) -> str:
    if contact == DECLINED:
        return "they didn't want to say"
    if contact:
        return contact
    if number:
        return f"call back on {number}"
    return "not given"


def _when(session: SessionRow, tz: tzinfo) -> str:
    started = session.started_at.astimezone(tz)
    return started.strftime("%a %d %b %Y, %I:%M %p %Z").replace(" 0", " ")


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
        subject, body = message_email(msg, caller, session, self._tz)
        await self._send(subject, body)

    async def alert(self, subject: str, body: str) -> None:
        await self._send(" ".join(subject.split()), body)

    async def _send(self, subject: str, body: str) -> None:
        email = EmailMessage()
        email["From"] = self._address
        email["To"] = self._to
        email["Subject"] = subject
        email.set_content(body)
        await asyncio.to_thread(self._send_now, email)

    def _send_now(self, email: EmailMessage) -> None:
        # Google shows app passwords in groups of four ("abcd efgh ..."); the spaces aren't
        # part of it.
        password = "".join(self._password.get_secret_value().split())
        context = ssl.create_default_context()
        with self._smtp(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT_S, context=context) as smtp:
            smtp.login(self._address, password)
            smtp.send_message(email)
