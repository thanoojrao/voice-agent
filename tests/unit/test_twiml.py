from xml.etree.ElementTree import fromstring

import pytest

from voice_agent.adapters.twilio import twiml

URL = "wss://agent.example.com/twilio/stream"


def test_stream_connects_then_hangs_up():
    root = fromstring(twiml.stream(URL, {"from": "+15550100", "token": "abc"}))
    assert root.tag == "Response"
    assert [child.tag for child in root] == ["Connect", "Hangup"]
    stream_el = root.find("Connect/Stream")
    assert stream_el is not None
    assert stream_el.get("url") == URL
    params = {p.get("name"): p.get("value") for p in stream_el.findall("Parameter")}
    assert params == {"from": "+15550100", "token": "abc"}


def test_stream_starts_with_an_xml_declaration():
    assert twiml.reject() == '<?xml version="1.0" encoding="UTF-8"?><Response><Reject /></Response>'


def test_stream_escapes_parameter_values():
    nasty = '"/><Hangup/><x a="'
    xml = twiml.stream(URL, {"from": nasty})
    root = fromstring(xml)
    assert [child.tag for child in root] == ["Connect", "Hangup"]  # nothing injected
    param = root.find("Connect/Stream/Parameter")
    assert param is not None
    assert param.get("value") == nasty


def test_stream_needs_wss():
    with pytest.raises(ValueError, match="wss"):
        twiml.stream("https://agent.example.com/twilio/stream", {})


def test_stream_url_cannot_have_a_query_string():
    with pytest.raises(ValueError, match="query"):
        twiml.stream(URL + "?token=abc", {})


def test_stream_parameter_must_fit_twilios_limit():
    twiml.stream(URL, {"from": "x" * 495})  # 4 + 495 = 499, just under
    with pytest.raises(ValueError, match="too long"):
        twiml.stream(URL, {"from": "x" * 496})


def test_reject():
    root = fromstring(twiml.reject())
    assert [child.tag for child in root] == ["Reject"]


def test_play_then_hangup():
    root = fromstring(twiml.play_then_hangup("https://agent.example.com/static/audio/busy.wav"))
    assert [child.tag for child in root] == ["Play", "Hangup"]
    assert root[0].text == "https://agent.example.com/static/audio/busy.wav"
