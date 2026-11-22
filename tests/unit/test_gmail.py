"""The Gmail notifier, against a fake SMTP server (nothing is sent)."""

import smtplib
from datetime import UTC, datetime
from email.message import EmailMessage
from zoneinfo import ZoneInfo

import pytest
from pydantic import SecretStr

from voice_agent.core.interfaces import Caller, MessageRow, SessionRow
from voice_agent.core.tools import DECLINED
from voice_agent.notify.gmail import SMTP_HOST, SMTP_PORT, GmailNotifier, message_email

TZ = ZoneInfo("America/New_York")
NUMBER = "+15550100"
PASSWORD = "abcd efgh ijkl mnop"
STARTED = datetime(2026, 11, 20, 19, 5, tzinfo=UTC)  # 2:05 PM in New York


def caller(
    name: str | None = None,
    address: str | None = NUMBER,
    company: str | None = None,
    reason: str | None = None,
) -> Caller:
    return Caller(1, "twilio", address, name, company, reason, STARTED, STARTED, 1)


def session(summary: str | None = "The caller asked about the backend role.") -> SessionRow:
    return SessionRow(7, 1, "twilio", "CA1", STARTED, STARTED, "goodbye", summary)


def message(body="Would like to chat about the backend role.", name=None, contact=None):
    return MessageRow(body=body, created_at=STARTED, name=name, contact=contact, id=3)


class FakeSmtp:
    """Stands in for smtplib.SMTP_SSL and records what it was asked to do."""

    instances: list["FakeSmtp"] = []

    def __init__(self, host, port, *, timeout, context, fail_login=False):
        self.host, self.port, self.timeout, self.context = host, port, timeout, context
        self.logins: list[tuple[str, str]] = []
        self.sent: list[EmailMessage] = []
        self.closed = False
        self.fail_login = fail_login
        FakeSmtp.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True

    def login(self, user, password):
        if self.fail_login:
            raise smtplib.SMTPAuthenticationError(535, b"5.7.8 Username and Password not accepted")
        self.logins.append((user, password))

    def send_message(self, email):
        self.sent.append(email)


@pytest.fixture(autouse=True)
def _reset():
    FakeSmtp.instances.clear()


def notifier(**smtp_kwargs) -> GmailNotifier:
    return GmailNotifier(
        "bot@example.com",
        SecretStr(PASSWORD),
        "owner@example.com",
        TZ,
        smtp=lambda *a, **k: FakeSmtp(*a, **k, **smtp_kwargs),
    )


# -- what the email says ------------------------------------------------------------------------


def test_message_email_has_everything_the_owner_needs_to_follow_up():
    email = message_email(
        message(contact="sam@example.com", name="Sam"),
        caller(name="Sam", company="Acme", reason="a backend role"),
        session(),
        TZ,
    )
    assert email.subject == "Voice agent: message from Sam"
    text = email.text
    assert text.startswith("New message from Sam\n")
    assert '"Would like to chat about the backend role."' in text
    for line in (
        "From:          Sam",
        f"Phone:         {NUMBER}",
        "Reach them:    sam@example.com",
        "Company:       Acme",
        "Calling about: a backend role",
        "Called:        Fri 20 Nov 2026, 2:05 PM EST",
        "Call summary: The caller asked about the backend role.",
        "Call #7, ended: goodbye. Sent by Voice agent.",
    ):
        assert line in text


def test_html_has_tap_to_call_and_tap_to_email_links():
    email = message_email(message(contact="sam@example.com"), caller(name="Sam"), session(), TZ)
    assert email.html is not None
    assert f'<a href="tel:{NUMBER}">{NUMBER}</a>' in email.html
    assert '<a href="mailto:sam@example.com">sam@example.com</a>' in email.html
    assert "Would like to chat about the backend role." in email.html


def test_html_escapes_everything_the_caller_said():
    # A caller (or speech recognition) can put anything in a message or a name.
    sneaky = '<a href="https://evil.example">click</a><script>x()</script>'
    email = message_email(
        message(body=sneaky, contact='"><img src=x>'),
        caller(name="Sam", company="<b>Acme</b>"),
        session(summary="<i>hi</i>"),
        TZ,
    )
    assert email.html is not None
    for raw in ("<script>", '<a href="https://evil', "<img", "<b>Acme", "<i>hi"):
        assert raw not in email.html
    assert "&lt;script&gt;" in email.html
    assert 'href="mailto:' not in email.html  # the odd contact isn't an address: no link


@pytest.mark.parametrize(
    ("contact", "address", "reach", "link"),
    [
        (None, NUMBER, f"call back on {NUMBER}", f"tel:{NUMBER}"),  # "this number"
        (DECLINED, NUMBER, "they didn't want to say", None),
        ("+1 (555) 019-9", NUMBER, "+1 (555) 019-9", "tel:+15550199"),
        (None, None, "not given", None),
    ],
)
def test_how_to_reach_them(contact, address, reach, link):
    email = message_email(message(contact=contact), caller(address=address), session(), TZ)
    assert f"Reach them:    {reach}" in email.text
    assert email.html is not None
    if link:
        assert f'<a href="{link}">' in email.html


def test_without_a_name_the_number_or_anonymous_is_used():
    email = message_email(message(), caller(), session(summary=None), TZ)
    assert email.subject == f"Voice agent: message from {NUMBER}"
    assert "From:          Name not given" in email.text
    assert "Company:" not in email.text and "Call summary:" not in email.text

    email = message_email(message(), caller(address=None), session(), TZ)
    assert email.subject == "Voice agent: message from an anonymous caller"
    assert "Phone:         withheld" in email.text


def test_the_name_on_the_message_wins_over_the_callers():
    email = message_email(message(name="Priya"), caller(name="Sam"), session(), TZ)
    assert email.subject == "Voice agent: message from Priya"


# -- sending ------------------------------------------------------------------------------------


async def test_sends_over_tls_signed_in_with_the_app_password():
    await notifier().message_left(message(), caller(name="Sam"), session())

    [smtp] = FakeSmtp.instances
    assert (smtp.host, smtp.port) == (SMTP_HOST, SMTP_PORT) == ("smtp.gmail.com", 465)
    assert smtp.context is not None
    assert smtp.logins == [("bot@example.com", "abcdefghijklmnop")]  # spaces removed
    [email] = smtp.sent
    assert (email["From"], email["To"]) == ("Voice agent <bot@example.com>", "owner@example.com")
    assert email["Subject"] == "Voice agent: message from Sam"
    text_part = email.get_body(preferencelist=("plain",))
    html_part = email.get_body(preferencelist=("html",))
    assert text_part is not None and html_part is not None  # both, for any mail app
    assert "backend role" in text_part.get_content()
    assert "backend role" in html_part.get_content()
    assert smtp.closed


async def test_alert_is_sent_as_given():
    await notifier().alert("Voice agent: call failed", "Session 7 ended with an error.")
    [email] = FakeSmtp.instances[0].sent
    assert email["Subject"] == "Voice agent: call failed"
    assert email.get_content().strip() == "Session 7 ended with an error."


async def test_a_line_break_in_a_subject_cant_add_a_header():
    await notifier().alert("call failed\nBcc: someone@example.com", "x")
    [email] = FakeSmtp.instances[0].sent
    assert email["Bcc"] is None
    assert email["Subject"] == "call failed Bcc: someone@example.com"


async def test_a_failed_sign_in_raises_without_the_password():
    with pytest.raises(smtplib.SMTPAuthenticationError) as caught:
        await notifier(fail_login=True).message_left(message(), caller(), session())
    assert "abcd" not in str(caught.value)
    assert FakeSmtp.instances[0].sent == []
    assert FakeSmtp.instances[0].closed
