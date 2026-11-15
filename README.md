# Voice Agent

A phone agent that answers calls as an AI version of me. Callers can ask about my work, my projects and what I'm up to right now. They can also leave a message, which reaches me by email. It always says up front that it's an AI.

> **Status:** early development. The project skeleton is done; the conversation engine is next. See the [build order](docs/technical-design.md#16-build-order-thin-slices).

## How it works

A cascaded speech pipeline, built without a voice framework:

```
caller ──► Twilio ──► speech-to-text ──► LLM ──► text-to-speech ──► Twilio ──► caller
            (phone)    Deepgram Flux     Claude    Cartesia
```

- **Streaming end to end.** The reply is spoken sentence by sentence while the model is still writing, and the knowledge base sits in a cached prompt prefix, to keep the gap after the caller stops talking short.
- **Interruptions.** Real words from the caller stop the agent mid-sentence; "mm-hm" and "yeah" don't. The transcript keeps only the words the caller actually heard.
- **Channel-agnostic core.** The conversation engine only talks to small interfaces. Twilio is one adapter behind them, and a website voice widget can be another. The speech and model vendors sit behind interfaces too, and a test fails the build if the core imports anything outside itself or the standard library.
- **Knowledge base as Markdown.** What the agent knows about me lives in `kb/`, which is kept out of the repo. `kb.example/` shows the layout with made-up content, and the whole thing goes into a cached prompt prefix rather than a retrieval index.

## Stack

Python 3.13 · asyncio · FastAPI · SQLite (aiosqlite, plain SQL migrations) · Twilio Media Streams · Deepgram Flux · Claude · Cartesia · pytest · ruff · uv

## Project layout

```
src/voice_agent/
  core/        conversation engine and the interfaces everything else implements
  adapters/    channels (Twilio)
  providers/   speech-to-text, LLM, text-to-speech
  storage/     SQLite storage and migrations
  notify/      email
  admin/       read-only admin pages
  app/         config and startup wiring
kb.example/    made-up knowledge base showing the layout (the real kb/ is gitignored)
migrations/    numbered SQL migrations
tests/         unit and integration tests, with fakes for every outside service
docs/          technical design
```

## Running locally

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # install Python 3.13 and dependencies
cp .env.example .env                      # then fill in the keys
uv run python -m voice_agent.storage.migrate
uv run pytest
```

Settings the agent uses (greeting, timers, call limits) are in `agent.config.yaml`; secrets go in `.env`, which is never committed.

## Documentation

[`docs/technical-design.md`](docs/technical-design.md) covers the architecture, interfaces, turn-taking, storage schema and build plan.
