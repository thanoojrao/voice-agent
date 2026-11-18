"""The TwiML replies to Twilio's webhooks (design §7.1, §7.3).

TwiML is the XML that tells Twilio what to do with a call. We only need four replies:

- stream:       connect the call to our WebSocket (the normal case)
- reject:       refuse without answering (a blocked number)
- play + hangup: play a clip, then hang up (busy, and the fallback apology)

Built with ElementTree rather than f-strings so every value is escaped: the caller's number
goes into an attribute, and a raw caller-ID string from Twilio could hold anything.
"""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import urlsplit
from xml.etree.ElementTree import Element, SubElement, tostring

# Twilio: "The combined length of each <Parameter> name and value attributes must be under
# 500 characters."
_MAX_PARAMETER_CHARS = 500


def stream(url: str, parameters: Mapping[str, str]) -> str:
    """Connect the call to our WebSocket at `url`, passing `parameters` in its start message.

    `<Connect><Stream>` holds the call until our server closes the socket; then Twilio runs
    the next verb. `<Hangup/>` is that verb, so closing the socket ends the call (how
    `end_session()` works). It is also what Twilio would do with nothing left, but saying it
    keeps the behaviour visible here.
    """
    parts = urlsplit(url)
    if parts.scheme != "wss":
        raise ValueError(f"stream url must be wss://, got {parts.scheme!r}")
    if parts.query:
        # Twilio: "The url does not support query string parameters." Use parameters instead.
        raise ValueError("stream url can't have a query string")

    response = Element("Response")
    connect = SubElement(response, "Connect")
    stream_el = SubElement(connect, "Stream", url=url)
    for name, value in parameters.items():
        if len(name) + len(value) >= _MAX_PARAMETER_CHARS:
            raise ValueError(f"parameter {name!r} is too long for Twilio")
        SubElement(stream_el, "Parameter", name=name, value=value)
    SubElement(response, "Hangup")
    return _xml(response)


def reject() -> str:
    """Refuse the call without answering it; the caller hears a busy tone."""
    response = Element("Response")
    SubElement(response, "Reject")
    return _xml(response)


def play_then_hangup(audio_url: str) -> str:
    """Play one clip (the busy message, or the apology on the fallback URL), then hang up."""
    response = Element("Response")
    SubElement(response, "Play").text = audio_url
    SubElement(response, "Hangup")
    return _xml(response)


def _xml(response: Element) -> str:
    return '<?xml version="1.0" encoding="UTF-8"?>' + tostring(response, encoding="unicode")
