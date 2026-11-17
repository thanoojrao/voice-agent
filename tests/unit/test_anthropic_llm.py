"""The Claude provider, run through the real SDK against canned HTTP responses.

A mock transport stands in for the network: it records each request body and answers with
server-sent events in the Messages API wire format. So these cover the SDK's own parsing and
error types too, with no API calls and no cost. Responses recorded from the live API come
with the step 3 latency script.
"""

import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2
import pytest

from voice_agent.core.interfaces import (
    Done,
    LlmEvent,
    Message,
    ProviderError,
    Refused,
    SystemBlock,
    TextDelta,
    TextPart,
    ToolCall,
    ToolResultPart,
    ToolUsePart,
)
from voice_agent.core.tools import SAVE_MESSAGE, tool_defs
from voice_agent.providers.anthropic_llm import AnthropicLLM

SYSTEM = [
    SystemBlock("You are Alex's AI."),
    SystemBlock("<knowledge_base>...</knowledge_base>", cache=True),
    SystemBlock("The caller's name is unknown."),
]
HELLO = [Message("user", (TextPart("Hi, who's this?"),))]
TOOLS = tool_defs(can_end_session=True)
USAGE = {"input_tokens": 40, "cache_read_input_tokens": 3000, "output_tokens": 12}


# -- canned responses -----------------------------------------------------------------------


def sse(*events: dict[str, Any]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def start() -> dict[str, Any]:
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {**USAGE, "output_tokens": 1},
    }
    return {"type": "message_start", "message": message}


def text_block(index: int, *pieces: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        },
        *(
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": p},
            }
            for p in pieces
        ),
        {"type": "content_block_stop", "index": index},
    ]


def tool_block(index: int, name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    block = {"type": "tool_use", "id": f"toolu_{name}", "name": name, "input": {}}
    raw = json.dumps(args)
    return [
        {"type": "content_block_start", "index": index, "content_block": block},
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": raw[:10]},
        },
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": raw[10:]},
        },
        {"type": "content_block_stop", "index": index},
    ]


def end(stop_reason: str, **delta: Any) -> list[dict[str, Any]]:
    return [
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None, **delta},
            "usage": {"output_tokens": USAGE["output_tokens"]},
        },
        {"type": "message_stop"},
    ]


def streamed(*blocks: list[dict[str, Any]], stop: str = "end_turn", **delta: Any) -> bytes:
    return sse(start(), *(e for b in blocks for e in b), *end(stop, **delta))


def error_body(kind: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": kind, "message": "nope"}}


# -- harness --------------------------------------------------------------------------------


class Api:
    """The fake server: answers each request with the next response and keeps the bodies."""

    def __init__(self, *responses: httpx2.Response | Callable[[], httpx2.Response]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(self._handle))
        sdk = anthropic.AsyncAnthropic(api_key="test", max_retries=0, http_client=client)
        self.llm = AnthropicLLM(sdk, "claude-haiku-5-5")

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        response = self._responses.pop(0)
        return response() if callable(response) else response

    @property
    def body(self) -> dict[str, Any]:
        [body] = self.requests
        return body


def ok_stream(body: bytes) -> httpx2.Response:
    return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})


async def reply(api: Api, history: list[Message] = HELLO) -> list[LlmEvent]:
    return [e async for e in api.llm.stream_reply(SYSTEM, history, TOOLS)]


# -- streaming ------------------------------------------------------------------------------


async def test_text_streams_as_it_arrives_then_done_with_token_counts():
    api = Api(ok_stream(streamed(text_block(0, "Hi, I'm ", "Alex's AI."))))
    assert await reply(api) == [
        TextDelta("Hi, I'm "),
        TextDelta("Alex's AI."),
        Done("end_turn", input_tokens=40, cache_read_tokens=3000, output_tokens=12),
    ]


async def test_request_has_thinking_off_the_kb_cached_and_strict_tools():
    api = Api(ok_stream(streamed(text_block(0, "Hi."))))
    await reply(api)
    body = api.body

    assert body["model"] == "claude-haiku-5-5"
    assert body["stream"] is True
    assert body["thinking"] == {"type": "disabled"}
    assert body["output_config"] == {"effort": "low"}
    assert [b.get("cache_control") for b in body["system"]] == [None, {"type": "ephemeral"}, None]
    assert [b["text"] for b in body["system"]] == [b.text for b in SYSTEM]

    assert [t["name"] for t in body["tools"]] == [t.name for t in TOOLS]
    assert all(t["strict"] for t in body["tools"])
    save = next(t for t in body["tools"] if t["name"] == SAVE_MESSAGE.name)
    assert "maxLength" not in json.dumps(save["input_schema"])
    assert save["input_schema"]["additionalProperties"] is False
    assert "maxLength" in json.dumps(SAVE_MESSAGE.input_schema)  # tools.py keeps its limits


async def test_history_maps_to_messages_api_blocks():
    history = [
        Message("user", (TextPart("Leave a message for Alex."),)),
        Message(
            "assistant",
            (
                TextPart("Sure."),
                TextPart(""),  # nothing said before a tool call
                ToolUsePart("toolu_1", "save_message", {"body": "Call me"}),
            ),
        ),
        Message("user", (ToolResultPart("toolu_1", "need a contact", is_error=True),)),
    ]
    api = Api(ok_stream(streamed(text_block(0, "How can Alex reach you?"))))
    await reply(api, history)

    assert api.body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Leave a message for Alex."}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Sure."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "save_message",
                    "input": {"body": "Call me"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "need a contact",
                    "is_error": True,
                }
            ],
        },
    ]


async def test_tool_calls_come_after_the_text_and_before_done():
    args = {"name": "Sam", "company": "Acme"}
    api = Api(
        ok_stream(
            streamed(
                text_block(0, "Nice to meet you."),
                tool_block(1, "save_caller_details", args),
                stop="tool_use",
            )
        )
    )
    assert await reply(api) == [
        TextDelta("Nice to meet you."),
        ToolCall("toolu_save_caller_details", "save_caller_details", args),
        Done("tool_use", input_tokens=40, cache_read_tokens=3000, output_tokens=12),
    ]


async def test_refusal_after_some_text_is_refused_and_runs_no_tool():
    api = Api(
        ok_stream(
            streamed(
                text_block(0, "Well, "),
                tool_block(1, "end_session", {"reason": "abuse"}),
                stop="refusal",
                stop_details={"type": "refusal", "category": "general_harms", "explanation": None},
            )
        )
    )
    assert await reply(api) == [TextDelta("Well, "), Refused()]


async def test_reply_cut_off_at_max_tokens_drops_its_tool_call():
    api = Api(
        ok_stream(
            streamed(
                text_block(0, "Saving that."),
                tool_block(1, "save_message", {"body": "Call me"}),
                stop="max_tokens",
            )
        )
    )
    events = await reply(api)
    assert not any(isinstance(e, ToolCall) for e in events)
    assert events[-1] == Done(
        "max_tokens", input_tokens=40, cache_read_tokens=3000, output_tokens=12
    )


# -- errors ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "kind", "retryable"),
    [
        (529, "overloaded_error", True),
        (500, "api_error", True),
        (429, "rate_limit_error", True),
        (400, "invalid_request_error", False),
        (401, "authentication_error", False),
    ],
)
async def test_http_errors_become_provider_errors_with_no_sdk_retry(status, kind, retryable):
    api = Api(httpx2.Response(status, json=error_body(kind)))
    with pytest.raises(ProviderError) as caught:
        await reply(api)
    assert caught.value.retryable is retryable
    assert kind in str(caught.value)
    assert len(api.requests) == 1  # the turn decides on retries, not the SDK


async def test_error_mid_stream_is_retryable_even_though_the_response_was_200():
    body = sse(
        start(), *text_block(0, "Let me "), {"type": "error", **error_body("overloaded_error")}
    )
    api = Api(ok_stream(body))
    events: list[LlmEvent] = []
    with pytest.raises(ProviderError) as caught:
        async for event in api.llm.stream_reply(SYSTEM, HELLO, TOOLS):
            events.append(event)
    assert events == [TextDelta("Let me ")]
    assert caught.value.retryable


async def test_connection_failure_is_retryable():
    def refuse() -> httpx2.Response:
        raise httpx2.ConnectError("no route to host")

    api = Api(refuse)
    with pytest.raises(ProviderError) as caught:
        await reply(api)
    assert caught.value.retryable


# -- prewarm --------------------------------------------------------------------------------


def prewarm_response() -> httpx2.Response:
    message = {
        "id": "msg_0",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [],
        "stop_reason": "max_tokens",
        "stop_sequence": None,
        "usage": {"input_tokens": 8, "cache_creation_input_tokens": 3000, "output_tokens": 0},
    }
    return httpx2.Response(200, json=message)


async def test_prewarm_caches_the_same_prefix_as_a_reply_and_generates_nothing():
    api = Api(prewarm_response(), ok_stream(streamed(text_block(0, "Hi."))))
    await api.llm.prewarm(SYSTEM, TOOLS)
    await reply(api)
    warm, real = api.requests

    assert warm["max_tokens"] == 0
    assert "stream" not in warm
    # Everything up to the cache breakpoint must match, or the reply misses the cache.
    for key in ("model", "system", "tools", "thinking", "output_config"):
        assert warm[key] == real[key], key


async def test_prewarm_failure_is_a_provider_error():
    api = Api(httpx2.Response(529, json=error_body("overloaded_error")))
    with pytest.raises(ProviderError) as caught:
        await api.llm.prewarm(SYSTEM, TOOLS)
    assert caught.value.retryable


# -- summary --------------------------------------------------------------------------------


def summary_response(text: str, stop_reason: str = "end_turn") -> httpx2.Response:
    message = {
        "id": "msg_2",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 200, "output_tokens": 20},
    }
    return httpx2.Response(200, json=message)


async def test_summary_is_one_short_request_over_the_transcript():
    api = Api(summary_response(" The caller asked for a call back. "))
    transcript = "Caller: It's Sam from Acme.\nAgent: Nice to meet you, Sam."
    assert await api.llm.summarize(transcript) == "The caller asked for a call back."

    body = api.body
    assert "stream" not in body
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] <= 200
    [message] = body["messages"]
    assert message["content"] == f"<transcript>\n{transcript}\n</transcript>"
    assert "tools" not in body


async def test_summary_refusal_and_errors_raise():
    api = Api(
        summary_response("", stop_reason="refusal"),
        httpx2.Response(529, json=error_body("overloaded_error")),
    )
    with pytest.raises(ProviderError):
        await api.llm.summarize("Caller: hi")
    with pytest.raises(ProviderError) as caught:
        await api.llm.summarize("Caller: hi")
    assert caught.value.retryable
