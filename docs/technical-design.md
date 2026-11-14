# Voice Agent — Technical Design (v0)

Status: draft for v0 · Last updated: 2026-11-13 (build step 1 done)
The product spec (use cases, decisions, settings) is kept separately and is the source of truth for *what*; this doc covers *how*. Spec references such as "D30" or "UC-8" point to it.

This document turns the product spec into code structure. When the two disagree, the spec wins and this doc gets fixed.

---

## 1. Decisions this design rests on

| Area | Choice |
| --- | --- |
| Language / runtime | Python 3.13, asyncio throughout (3.13 because `audioop-lts` needs it) |
| Packaging | `uv` (Python version, deps, lockfile) |
| Web framework | FastAPI + uvicorn (single worker — see §14) |
| Voice pipeline | Custom, no framework. Cascaded STT → LLM → TTS |
| STT | Deepgram Flux (streaming, turn detection built in) |
| LLM | Claude Haiku 5.5 (`claude-haiku-5-5`) via the official `anthropic` Python SDK |
| TTS | Cartesia (streaming, cloned voice with stock fallback) |
| Channel (v0) | Twilio Programmable Voice + bidirectional Media Streams |
| Database | SQLite via `aiosqlite`, hand-written SQL, numbered migration files |
| Admin UI | Server-rendered Jinja2 templates in the same FastAPI app, plain CSS |
| Email | Gmail SMTP (app password) via stdlib `smtplib` in a worker thread |
| Tests | `pytest` + `pytest-asyncio`; lint/format with `ruff` |

---

## 2. Dependency rule (the one rule that matters)

```
app (FastAPI wiring, startup)
 ├── adapters/        Twilio today, web later      ─┐
 ├── providers/       Deepgram, Anthropic, Cartesia │  all depend on core.interfaces,
 ├── storage/         SQLite                        │  never on each other
 ├── notify/          Gmail                        ─┘
 └── core/            sessions, turn-taking, prompts, tools
       └── interfaces.py   Protocols every outer piece implements
```

- `core/` imports **nothing** from `adapters/`, `providers/`, `storage/` or `notify/`, and no vendor SDK. It only knows the Protocols in `core/interfaces.py`.
- Outer pieces are created in `app/main.py` and injected into the core.
- This applies to providers as well as channels. Swapping Cartesia or Deepgram is the same kind of change as adding the web channel, and tests run the whole core on fakes.
- `tests/unit/test_core_imports.py` enforces this: it parses every file in `core/` (imports inside functions and relative imports included) and fails unless each import is the standard library or `voice_agent.core.*`. It runs with every `pytest`.

---

## 3. Repository layout

```
voice_agent/
├── pyproject.toml / uv.lock
├── .env.example                 # every secret, no values
├── agent.config.yaml            # owner-editable settings (spec: Product variables)
├── kb/                          # knowledge base (spec: Knowledge base)
│   ├── profile.md  experience.md  status.md  skills.md  faq.md  boundaries.md
│   └── projects/*.md
├── assets/audio/                # generated clips: apology, fillers, busy (PCM16)
├── migrations/                  # 0001_init.sql, 0002_…sql
├── scripts/
│   └── generate_clips.py        # render clips with the current voice_id
├── src/voice_agent/
│   ├── app/
│   │   ├── main.py              # FastAPI app, startup wiring, routes mounting
│   │   ├── config.py            # Settings (.env) + AgentConfig (yaml) + constants
│   │   └── constants.py         # spec: "Constants in code (v0)"
│   ├── core/
│   │   ├── interfaces.py        # ChannelAdapter, SessionCallbacks, STT/LLM/TTS, Storage, Notifier
│   │   ├── session.py           # Session: one conversation, owns the turn state machine
│   │   ├── registry.py          # active sessions, concurrency + daily caps
│   │   ├── turn.py              # one agent turn: LLM stream → sentences → TTS → audio
│   │   ├── sentences.py         # streaming sentence splitter
│   │   ├── bargein.py           # backchannel filter
│   │   ├── history.py           # conversation history + trimming on interruption
│   │   ├── prompt.py            # system prompt assembly (KB + instructions + session context)
│   │   ├── tools.py             # save_message, end_session definitions + handlers
│   │   ├── greeting.py          # opening line, warm intro, name question
│   │   └── clips.py             # loads apology/filler/busy clips
│   ├── adapters/
│   │   └── twilio/
│   │       ├── routes.py        # POST /twilio/voice, /twilio/fallback, WS /twilio/stream
│   │       ├── adapter.py       # TwilioAdapter implements ChannelAdapter
│   │       ├── audio.py         # μ-law ↔ PCM16
│   │       └── twiml.py         # TwiML builders
│   ├── providers/
│   │   ├── deepgram_flux.py     # implements SpeechToText
│   │   ├── anthropic_llm.py     # implements LanguageModel
│   │   └── cartesia_tts.py      # implements TextToSpeech
│   ├── storage/
│   │   ├── migrate.py           # apply pending migrations (also: python -m voice_agent.storage.migrate)
│   │   └── sqlite.py            # implements Storage
│   ├── notify/gmail.py          # implements Notifier
│   └── admin/
│       ├── routes.py            # /admin pages, HTTP Basic auth
│       └── templates/*.html
└── tests/
    ├── fakes/                   # FakeVoiceAdapter, FakeSTT, FakeLLM, FakeTTS, MemoryStorage
    ├── unit/
    └── integration/             # full sessions through the core on fakes
```

---

## 4. Core interfaces

Sketches, not final signatures. All methods are `async` unless noted.

### 4.1 Audio format

Core audio is **PCM16 little-endian, mono**, at the sample rate the adapter declares. Twilio declares 8000 Hz, and its adapter converts μ-law ↔ PCM16. Deepgram and Cartesia are configured with the adapter's rate, so v0 never resamples.

> `audioop` was removed in Python 3.13. Use the `audioop-lts` package or a 256-entry μ-law lookup table in `adapters/twilio/audio.py`.

### 4.2 Channel adapter (spec: Adapter interface)

```python
class Capability(StrEnum):
    CALLER_ADDRESS = "caller_address"
    FOLLOWUPS = "followups"  # v1 (SMS on Twilio)
    END_SESSION = "end_session"


@dataclass(frozen=True)
class SessionStart:
    channel: str  # "twilio"
    channel_ref: str  # Twilio CallSid
    caller_address: str | None  # E.164, or None when withheld
    sample_rate: int  # 8000 for Twilio


class ChannelAdapter(Protocol):  # implemented by adapters
    capabilities: frozenset[Capability]

    async def send_audio(self, pcm: bytes) -> None: ...
    async def clear_audio(self) -> None: ...  # barge-in: drop queued audio
    async def mark(self, name: str) -> None: ...  # playback checkpoint, see §6.3
    async def end_session(self) -> None: ...
    async def send_followup(self, body: str, link: str | None) -> None: ...  # v1


class SessionCallbacks(Protocol):  # implemented by core.Session, called by adapters
    async def on_audio(self, pcm: bytes) -> None: ...
    async def on_mark_played(self, name: str) -> None: ...
    async def on_session_end(self, reason: str) -> None: ...
```

The adapter calls `registry.start_session(SessionStart, adapter)` and gets back the `SessionCallbacks` to feed. These correspond to the spec's `on_session_start` and `on_audio`.

### 4.3 Providers

```python
class SpeechToText(Protocol):
    async def open(self, sample_rate: int) -> "SttStream": ...


class SttStream(Protocol):
    async def send(self, pcm: bytes) -> None: ...
    def events(self) -> AsyncIterator[SttEvent]: ...  # TurnStarted, Words, TurnEnded
    async def close(self) -> None: ...


class LanguageModel(Protocol):
    def stream_reply(
        self, system: list[SystemBlock], history: list[Message], tools: list[ToolDef]
    ) -> AsyncIterator[LlmEvent]: ...  # TextDelta, ToolCall, Done, Refused
    async def summarize(self, transcript: str) -> str: ...


class TextToSpeech(Protocol):
    async def open(self, sample_rate: int, voice_id: str) -> "TtsStream": ...


class TtsStream(Protocol):  # one per agent turn
    async def say(self, sentence: str) -> None: ...  # append text to the same utterance
    async def finish(self) -> None: ...  # no more sentences this turn
    def audio(self) -> AsyncIterator[TtsChunk]: ...  # pcm + word timestamps
    async def cancel(self) -> None: ...
```

`SttEvent` uses our own names. The Flux adapter maps Deepgram's events onto them (§6.1). Likewise `Message` is our own type (text, tool-use and tool-result parts), not the Anthropic SDK's, so the core never sees a vendor type. The real definitions are in `src/voice_agent/core/interfaces.py`; where this sketch and the code differ, the code wins.

### 4.4 Storage and notifier

```python
class Storage(Protocol):
    async def get_or_create_caller(self, channel: str, address: str | None) -> Caller: ...
    async def set_caller_details(
        self, caller_id: int, name: str | None, company: str | None, reason: str | None
    ) -> None: ...  # None keeps the stored value
    async def last_session(self, caller_id: int) -> SessionRow | None: ...
    async def create_session(self, caller_id: int, start: SessionStart) -> int: ...
    async def end_session(self, session_id: int, reason: str, summary: str | None) -> None: ...
    async def add_turn(self, session_id: int, turn: TurnRow) -> None: ...
    async def save_message(self, session_id: int, msg: MessageRow) -> int: ...
    async def mark_message_notified(self, message_id: int) -> None: ...
    async def count_sessions_since(
        self, since: datetime, caller_id: int | None = None
    ) -> int: ...  # the core decides when "today" starts


class Notifier(Protocol):
    async def message_left(self, msg: MessageRow, caller: Caller, session: SessionRow) -> None: ...
    async def alert(self, subject: str, body: str) -> None: ...
```

---

## 5. Session lifecycle

```mermaid
sequenceDiagram
    participant Caller
    participant Twilio
    participant API as FastAPI (Twilio adapter)
    participant Core as core.Session
    Caller->>Twilio: dials number
    Twilio->>API: POST /twilio/voice (From, CallSid, signature)
    API->>API: verify signature, blocklist, caps, concurrency
    API-->>Twilio: TwiML <Connect><Stream url=wss…/twilio/stream> + <Parameter> from, token
    Twilio->>API: WS connect → "connected", "start" (customParameters)
    API->>Core: registry.start_session(SessionStart, adapter)
    Core-->>Caller: opening line (TTS → adapter.send_audio)
    loop each turn
        Twilio->>API: "media" frames (μ-law)
        API->>Core: on_audio(pcm)
        Core-->>Caller: reply audio, "mark" checkpoints
    end
    Core->>API: adapter.end_session() (close WS → call ends)
    Core->>Core: summarize (background), store, done
```

**Start of a session (`registry.start_session`)**
1. `get_or_create_caller(channel, address)`. A `None` address creates a one-off anonymous caller row (spec D30).
2. `create_session(...)`, then open the STT stream.
3. Build the session context: caller name if known, last session's summary if under 90 days old, anonymous flag, today's date.
4. Speak the greeting (§8.4). Name question only when no name is saved, and every time for anonymous callers.

**End of a session.** Triggered by the `end_session` tool, the silence timeout, the `max_session_min` limit, the caller hanging up (WS `stop`), or a fatal provider failure.
1. Cancel in-flight LLM/TTS, close STT.
2. Close the adapter if it's still open.
3. In the background: `summarize` the transcript (Haiku, non-streaming), then `storage.end_session(reason, summary)`.

Turns are already stored live, so a crash during step 3 loses only the summary.

---

## 6. The turn engine

### 6.1 States

```mermaid
stateDiagram-v2
    [*] --> Speaking: greeting
    Listening --> Thinking: STT TurnEnded(transcript)
    Thinking --> Speaking: first TTS audio ready
    Thinking --> Speaking: filler timer (1.5 s) fires → play filler clip
    Speaking --> Listening: last mark played
    Speaking --> Interrupted: STT real word during playback
    Thinking --> Interrupted: STT real word before audio
    Interrupted --> Listening: cancel LLM+TTS, clear_audio, trim history
    Listening --> Listening: silence 8 s → "Still there?" (once)
    Listening --> Ending: silence 20 s more / end_session tool / limit
    Speaking --> Ending: end_session tool after closing line played
    Ending --> [*]
```

**Mapping Flux events.** Flux emits turn events (StartOfTurn, EagerEndOfTurn, TurnResumed, EndOfTurn, transcript updates). v0 maps:

| Flux | Our `SttEvent` | Used for |
| --- | --- | --- |
| StartOfTurn + interim words | `TurnStarted`, `Words(text)` | barge-in check (§6.4), silence timer reset |
| EndOfTurn | `TurnEnded(transcript)` | start the agent turn |
| EagerEndOfTurn / TurnResumed | ignored in v0 | later latency optimization (spec) |

> Verify the exact event names and payload fields against Deepgram's current Flux docs when writing `deepgram_flux.py`. Nothing outside that file depends on them.

### 6.2 One agent turn (`core/turn.py`)

```
TurnEnded(transcript)
  → history.append(user: transcript); storage.add_turn(caller turn)
  → start filler timer (FILLER_AFTER_MS)
  → llm.stream_reply(system, history, tools)
       TextDelta  → sentences.feed(delta) → each full sentence → tts.say(sentence)
       ToolCall   → handle (§8.3), feed result back, continue the stream
       Refused    → speak a fixed safe line, end turn
       Done       → sentences.flush() → tts.say(rest)
  → tts.audio() chunks → adapter.send_audio(pcm); adapter.mark(f"w{n}") at word boundaries
  → first chunk: cancel filler timer, record latency_ms
  → last mark played → history.append(assistant: spoken text); storage.add_turn(agent turn)
```

- The LLM, TTS and playback steps run as concurrent asyncio tasks, connected by queues. Cancelling the turn cancels the task group.
- **Sentence splitter.** Splits on `.?!` followed by whitespace, but not after common abbreviations ("e.g.", "Dr.", version numbers). It flushes a long clause at a comma or dash after about 200 characters, so the first audio is never stuck behind a long sentence.
- **Filler.** If the timer fires before the first TTS chunk, play a random filler clip. TTS audio queued meanwhile plays after it. Fillers never enter `history`.
- **Latency.** `latency_ms` is measured from `TurnEnded` to the first audio chunk sent to the adapter, and stored on the agent turn.

### 6.3 Knowing what the caller actually heard

Twilio plays audio asynchronously. We know what was *sent*, not what was *heard*. The Twilio adapter sends `mark` messages after each word's audio, and Twilio echoes them back as each one finishes playing. Cartesia's word timestamps tell us which word ends at which byte offset.

- `on_mark_played("w17")` means the caller has heard up to word 17.
- On interruption, the spoken text is the words up to the last played mark. History stores only that text, and the turn row gets `interrupted = 1` (spec: Interruptions).

### 6.4 Barge-in (`core/bargein.py`)

- On `TurnStarted` while Speaking or Thinking: become *tentatively interrupted*. Keep playing, and wait for words.
- On each `Words(text)` while tentative, `classify(text)` returns `NOTHING`, `BACKCHANNEL` or `SPEECH`:
  - `BACKCHANNEL` (only listening noises such as `mm-hm, uh-huh, yeah, okay, right, sure, got it`): keep speaking. The transcript keeps growing, so the next update is checked again ("yeah" → "yeah but wait").
  - `SPEECH`: interrupt. `adapter.clear_audio()` (Twilio `clear`), cancel the LLM and TTS tasks, trim history (§6.3), go to Listening.
- A noise with no transcribed words (`NOTHING`) never interrupts.
- If the caller's turn ends as only a backchannel while the agent is speaking, the turn is dropped, not answered.
- The 300 ms acceptance target is measured from Flux's first-word event to the `clear` being sent. It's the first number to check on a real call.

---

## 7. Twilio adapter

### 7.1 HTTP webhook `POST /twilio/voice`
1. Validate `X-Twilio-Signature` with the Twilio helper library's request validator against the public URL. Invalid → 403.
2. Normalize `From`. Empty, `anonymous`, `restricted` or `unknown` → `None`.
3. Reject or decline:
   - Number in `blocked_numbers` → `<Reject/>`.
   - Daily cap, per-caller cap or concurrency limit hit → `<Play>` the busy clip, then `<Hangup/>`. The busy clip is served as a static WAV at `/static/audio/busy.wav`.
4. Mint a one-time stream token (random, 60 s TTL, kept in memory).
5. Respond with TwiML:
   ```xml
   <Response>
     <Connect>
       <Stream url="wss://HOST/twilio/stream">
         <Parameter name="from" value="+15551234567"/>
         <Parameter name="token" value="…"/>
       </Stream>
     </Connect>
   </Response>
   ```
   With nothing after `<Connect>`, closing the WebSocket ends the call. That's how `end_session()` works, with no REST call needed.

### 7.2 WebSocket `/twilio/stream`
- `connected` → ignore.
- `start` → read `callSid`, `streamSid`, `customParameters.from`, `customParameters.token`. Validate and consume the token, otherwise close the socket. Then `registry.start_session(...)`.
- `media` → base64 μ-law → PCM16 → `session.on_audio`.
- `mark` → `session.on_mark_played(name)`.
- `stop` / socket closed → `session.on_session_end("caller_hangup")`.
- Outgoing: `media` (PCM16 → μ-law → base64, 20 ms frames), `mark`, `clear`.

> The one-time token is how the WebSocket is authenticated. Before relying on signature headers on the WS handshake, check Twilio's current docs.

### 7.3 Fallback `POST /twilio/fallback`
This is configured as the number's fallback URL in the Twilio console. It returns `<Play>` of the apology clip (static WAV) and then `<Hangup/>`. It needs no database or providers, so it works even when the core is broken.

---

## 8. Claude integration (`providers/anthropic_llm.py`)

### 8.1 Request shape
- Client: `anthropic.AsyncAnthropic()`. The key comes from `ANTHROPIC_API_KEY`.
- `model="claude-haiku-5-5"`, streaming via `client.messages.stream(...)`.
- **Thinking off for live turns.** Use `thinking={"type": "disabled"}`, which Haiku 5.5 accepts at effort `high` or below. A spoken one-to-three-sentence answer from a small knowledge base doesn't benefit from thinking, and thinking adds latency before the first token. If evals show quality gaps, try adaptive thinking at `effort: "low"` and measure the latency cost.
- `max_tokens=1024`. Answers are deliberately short, and this leaves room for a tool call plus a sentence.
- No `temperature`/`top_p` (rejected on current models).
- Read response content by block `type`, never by position. Check `stop_reason`: `"refusal"` → speak a fixed safe line ("I can't help with that one — anything else about my work?"). Haiku 5.5 has no server-side refusal fallback.

### 8.2 Prompt layout and caching
`system` is a list of blocks in this fixed order:

1. **Instructions:** persona, guardrails, voice style rules, tool usage. A static file, `src/voice_agent/core/prompts/instructions.md`.
2. **Knowledge base:** all `kb/` files concatenated in a fixed order with headers, read once at startup. **`cache_control: {"type": "ephemeral"}` goes on this block.**
3. **Session context:** today's date, caller name or "unknown", last-session summary or none, anonymous yes/no. Not cached, and placed after the breakpoint so it can change per session without invalidating the cache.

- Tools (`save_message`, `end_session`) render before `system`. The list is static and in a fixed order, so it stays inside the cached prefix.
- Silent cache-breakers to avoid: timestamps or caller data in blocks 1–2, and non-deterministic KB file ordering. Log `usage.cache_read_input_tokens` on every turn, and treat a zero after the first turn of a session as a bug.
- The minimum cacheable prefix on Haiku 5.5 is 512 tokens, which the instructions plus knowledge base easily exceed.

### 8.3 Tools
Both tools use `strict: true` and `eager_input_streaming: true`, with `tool_choice` left as `auto`. Forced tool choice isn't used.

| Tool | Input schema | Handler |
| --- | --- | --- |
| `save_message` | `{name: string, contact: string, body: string}` | Validate the input. `storage.save_message`, then `notifier.message_left(...)` immediately, then `mark_message_notified`. Return `{"saved": true}`, or an `is_error` result on failure. |
| `end_session` | `{reason: "goodbye" \| "abuse" \| "limit"}` | Set the session to end once the current spoken text finishes playing. Return `{"ok": true}`. |

- Because of eager input streaming, the accumulated tool input can be incomplete. Parse it, validate against the schema, and on failure return an `is_error` tool result rather than running the handler.
- The tool loop runs inside one agent turn. After a `tool_use`, append the assistant content and the `tool_result`, then continue streaming. Text spoken before and after the tool call flows to TTS as usual.
- The instructions tell the model to read the message back and get a "yes" before calling `save_message`, and to say its closing line before calling `end_session`.

### 8.4 Greeting
The opening line comes from `agent.config.yaml → greeting` and is spoken directly, with no LLM call, so the first audio is instant. It's added to `history` as the first assistant turn.

The warm intro is the exception. When a returning caller has a summary from the last 90 days, a short LLM turn produces the line ("Welcome back, Sam — last time we talked about the voice agent. Want to pick that up?") and is spoken right after the fixed disclosure line. The disclosure itself is never generated.

### 8.5 Errors and the fallback clip
- Retry once on `anthropic.APIConnectionError`, `RateLimitError` and 5xx `APIStatusError`, and on a first-token timeout (`PROVIDER_TIMEOUT_S`).
- On a second failure, play the apology clip, end the session, and send `notifier.alert`. Deepgram and Cartesia failures take the same path.
- The SDK has its own default retries (`max_retries=2`, a 10 min timeout), which are wrong for live voice. Construct the client with `max_retries=0` and a short timeout so `turn.py` owns the retry policy.

---

## 9. Cartesia (`providers/cartesia_tts.py`)
- One WebSocket connection per session, reused across turns. One *context* per agent turn, so `say()` appends sentences to the same utterance with natural prosody across sentence breaks.
- Output: raw PCM16 at the adapter's sample rate (8000 for Twilio), with word timestamps on.
- `cancel()` cancels the context on interruption.
- `voice_id` comes from config. The cloned voice is the default and the stock voice is the fallback (spec D12).

> Confirm the streaming/context API, the timestamp field names and support for 8 kHz PCM output in Cartesia's current docs when writing this file.

## 10. Deepgram Flux (`providers/deepgram_flux.py`)
- One streaming connection per session, encoding `linear16`, `sample_rate` from the adapter.
- Use the official SDK if it supports Flux; otherwise use the raw WebSocket API through `websockets`. Either way it stays inside this file.
- Reconnect once on drop. A second drop goes down the fatal-provider-failure path (§8.5).

---

## 11. Storage

### 11.1 Schema (`migrations/0001_init.sql`)

```sql
CREATE TABLE callers (
  id            INTEGER PRIMARY KEY,
  channel       TEXT NOT NULL,
  address       TEXT,                       -- NULL = anonymous
  name          TEXT, company TEXT, reason TEXT,
  first_seen    TEXT NOT NULL, last_seen TEXT NOT NULL,
  session_count INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX callers_channel_address ON callers(channel, address) WHERE address IS NOT NULL;

CREATE TABLE sessions (
  id          INTEGER PRIMARY KEY,
  caller_id   INTEGER NOT NULL REFERENCES callers(id),
  channel     TEXT NOT NULL,
  channel_ref TEXT,                         -- Twilio CallSid
  started_at  TEXT NOT NULL, ended_at TEXT,
  end_reason  TEXT,
  summary     TEXT
);

CREATE TABLE turns (
  id          INTEGER PRIMARY KEY,
  session_id  INTEGER NOT NULL REFERENCES sessions(id),
  seq         INTEGER NOT NULL,
  speaker     TEXT NOT NULL CHECK (speaker IN ('caller','agent','tool')),
  text        TEXT NOT NULL,
  started_at  TEXT NOT NULL, ended_at TEXT,
  interrupted INTEGER NOT NULL DEFAULT 0,
  latency_ms  INTEGER
);

CREATE TABLE messages (
  id          INTEGER PRIMARY KEY,
  session_id  INTEGER NOT NULL REFERENCES sessions(id),
  name TEXT, contact TEXT, body TEXT NOT NULL,
  created_at  TEXT NOT NULL,
  notified_at TEXT
);

-- plus indexes: sessions(caller_id, started_at), sessions(started_at), messages(session_id),
-- and UNIQUE(session_id, seq) on turns
```

- Timestamps are ISO-8601 UTC text.
- Tool calls are stored as `speaker='tool'` turns (name plus JSON input), so the admin page shows them in place.
- Set `PRAGMA journal_mode=WAL` and `foreign_keys=ON` on connect.
- `storage/migrate.py` applies `migrations/NNNN_*.sql` files in order, each file plus its version bump in one transaction, and owns the `schema_version` table. `SqliteStorage.open()` runs it at startup. v1 adds `0002_sms.sql` (the spec's texting columns plus a `texts` table).

### 11.2 Writes during a call
Every turn is written as it completes (spec: dropped calls keep their transcript). With one connection and WAL mode, writes take about a millisecond, which is no concern at 2 concurrent sessions.

---

## 12. Configuration

| Source | Holds | Loaded by |
| --- | --- | --- |
| `.env` | `ANTHROPIC_API_KEY`, `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`, `TWILIO_AUTH_TOKEN` (signature check), `OWNER_EMAIL`, `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`, `ADMIN_PASSWORD`, `PUBLIC_BASE_URL` | `pydantic-settings` |
| `agent.config.yaml` | the spec's Product variables (greeting, voice_id, caps, silence timers, do_not_discuss, …) | pydantic model, validated at startup; bad config refuses to start |
| `app/constants.py` | the spec's "Constants in code (v0)" | import |

v0 has no hot reload. Changing config or the knowledge base means a restart or redeploy (spec D28).

---

## 13. Admin page

- Routes: `GET /admin` (sessions list), `GET /admin/sessions/{id}` (transcript with latency, interruptions and tool calls), `GET /admin/messages`, `GET /admin/callers`.
- Auth: HTTP Basic over HTTPS, checking `ADMIN_PASSWORD` with `secrets.compare_digest`. That's enough for a single owner.
- Read-only in v0. Jinja2 templates with one small CSS file and no JavaScript.
- Latency view: per session, the average and p95 of `latency_ms` across agent turns, computed in SQL or Python at render time.

---

## 14. Running and deploying

- **Single process, single uvicorn worker.** The session registry, stream tokens and concurrency count live in memory. Two workers would split them. One worker easily handles 2 concurrent sessions, since the work is I/O-bound.
- **Docker.** One `Dockerfile` (python:3.13-slim + uv) and a volume mounted at `/app/data` for `agent.db`. The host must be always on, with a persistent disk, in a US region (spec D26).
- **Local development.** `uv run uvicorn …` plus a tunnel (ngrok or Cloudflare Tunnel). Point the Twilio number's voice webhook at `https://<tunnel>/twilio/voice` and set `PUBLIC_BASE_URL` to match, since signature validation needs the exact public URL.
- **Backups.** A daily `sqlite3 agent.db ".backup …"` copy off-box (spec: Proposed).
- **Logging.** Structured JSON lines carrying `session_id` on every line: per-turn latency, cache read tokens, provider errors, state transitions.

---

## 15. Testing strategy

| Level | What | How |
| --- | --- | --- |
| Unit | sentence splitter, backchannel filter, history trimming from marks, greeting/warm-intro rules, TwiML builders, μ-law codec, caps logic | plain pytest |
| Integration (core) | full sessions: greeting → question → answer; barge-in mid-answer; filler; `save_message` + email; `end_session`; anonymous caller; provider failure → apology clip | `FakeVoiceAdapter` + `FakeSTT` (scripted events) + `FakeLLM` (scripted deltas/tool calls) + `FakeTTS` (timestamps) + in-memory SQLite + `FakeNotifier` |
| Contract | each provider maps vendor events to our types correctly | recorded vendor messages replayed from fixtures |
| Manual | real calls against the spec's acceptance list | the checklist, run on a phone |
| Eval (later) | 30 KB questions, 10 out-of-scope, 10 injection | separate script against the real LLM with the real prompt |

The fake adapter is also run with `caller_address=None`, which covers spec D30.

---

## 16. Build order (thin slices)

1. ✅ **Skeleton:** uv project, config loading, constants, migrations, storage, and the interfaces with their fakes. Tests pass on an empty core.
2. **Core turn engine on fakes:** state machine, sentence splitter, history, barge-in, filler, tools. The integration tests are the definition of done.
3. **Real providers, no phone:** a dev script feeds a WAV file through Flux → Haiku → Cartesia and writes the reply WAV. Measure latency and the cache hit rate.
4. **Twilio adapter:** webhook, stream, marks and clear, fallback URL. First real call.
5. **Messages and email:** `save_message` → Gmail.
6. **Admin page.**
7. **Warm intro, session summary, caps, anonymous handling, clips script.**
8. **Run the acceptance checklist on real calls.** Tune latency.

---

## 17. Open technical items (resolve while building)

| Item | Where it bites | Resolve by |
| --- | --- | --- |
| Exact Flux event names and whether interim words arrive before EndOfTurn | §6.1, §6.4 | Deepgram docs + a recorded session (slice 3) |
| Cartesia context/continuation API, word timestamps, 8 kHz PCM output | §9, §6.3 | Cartesia docs + slice 3 |
| Whether Twilio signs the Media Stream WS handshake | §7.2 (token covers it either way) | Twilio docs |
| Hosting provider | §14 | Owner decision before the first real-number test |
| Thinking off vs adaptive at low effort for answer quality | §8.1 | the eval set once the KB exists |
