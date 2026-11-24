# Voice Agent

A phone agent that answers calls as an AI version of me. Callers can ask about my work, my projects and what I'm up to right now. They can also leave a message, which reaches me by email. It always says up front that it's an AI.

> **Status:** live on a real phone number, hosted on Fly.io. Calls, answers, interruptions, messages and the end-of-call email all work end to end. The read-only admin page is next, then a warm intro for returning callers. See the [build order](docs/technical-design.md#16-build-order-thin-slices).

## How it works

A cascaded speech pipeline, built without a voice framework:

```
caller ──► Twilio ──► speech-to-text ──► LLM ──► text-to-speech ──► Twilio ──► caller
            (phone)    Deepgram Flux     Claude    Cartesia
```

- **Streaming end to end.** The reply is spoken sentence by sentence while the model is still writing, and the knowledge base sits in a cached prompt prefix, to keep the gap after the caller stops talking short.
- **Interruptions.** Two real words from the caller stop the agent mid-sentence; listening noises ("mm-hm", "yeah", however speech-to-text spells them) don't. A lone word like "Stop" still lands when the caller's turn ends. The transcript keeps only the words the caller actually heard.
- **Messages.** The agent reads a message back, asks how to reach the caller, and saves it with a tool call. It's emailed once when the call ends, and anything left unsent (say, after a crash) is sent at the next startup.
- **Remembers callers.** A returning caller is greeted by name, and the agent knows a short summary of their last call and any message they left on it. After each call, a structured-output summary fills in details the model forgot to save.
- **Channel-agnostic core.** The conversation engine only talks to small interfaces. Twilio is one adapter behind them, and a website voice widget can be another. The speech and model vendors sit behind interfaces too, and a test fails the build if the core imports anything outside itself or the standard library.
- **Limits and safety.** Twilio's signature is checked on the webhook and the media stream, each caller gets a few calls a day, only a couple of calls run at once, and every call has a time limit.
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
evals/         behaviour eval cases (made-up callers only)
scripts/       dev tools: run the pipeline without a phone, run the evals, generate voice clips
docs/          technical design
Dockerfile, fly.toml   the container and its Fly.io deployment
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

To take real calls on your machine, run the server and a tunnel (ngrok or Cloudflare Tunnel), set `PUBLIC_BASE_URL` to the tunnel's address, and point your Twilio number's voice webhook at `<PUBLIC_BASE_URL>/twilio/voice`:

```bash
uv run uvicorn voice_agent.app.main:create_app --factory --port 8080
```

Without a phone:

```bash
uv run python scripts/try_pipeline.py --say "What are you working on?"   # real Flux -> Claude -> Cartesia, prints latency
uv run python scripts/run_evals.py                                        # behaviour evals against the real model (costs a few cents)
```

## Deploying

The app runs as one always-on Fly.io Machine with a volume for the SQLite database (see `fly.toml`). Secrets are Fly secrets, and `PUBLIC_BASE_URL` is the app's `https://<app>.fly.dev` address.

```bash
fly secrets import < .env      # then set PUBLIC_BASE_URL to the fly.dev address
fly deploy --ha=false          # one Machine only: the call registry lives in memory
```

A deploy drops any call in progress.

## Documentation

[`docs/technical-design.md`](docs/technical-design.md) covers the architecture, interfaces, turn-taking, storage schema, hosting and build plan.
