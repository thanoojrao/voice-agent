-- 0001_init: v0 tables. Timestamps are ISO-8601 UTC text.
-- schema_version is created and updated by scripts/migrate.py, not here.

CREATE TABLE callers (
  id            INTEGER PRIMARY KEY,
  channel       TEXT NOT NULL,                 -- "twilio"
  address       TEXT,                          -- E.164 number; NULL = withheld / anonymous
  name          TEXT,
  company       TEXT,
  reason        TEXT,
  first_seen    TEXT NOT NULL,
  last_seen     TEXT NOT NULL,
  session_count INTEGER NOT NULL DEFAULT 0
);
-- One row per known caller per channel. Anonymous callers get a new row each time,
-- so the uniqueness only applies when address is present.
CREATE UNIQUE INDEX callers_channel_address
  ON callers(channel, address) WHERE address IS NOT NULL;

CREATE TABLE sessions (
  id          INTEGER PRIMARY KEY,
  caller_id   INTEGER NOT NULL REFERENCES callers(id),
  channel     TEXT NOT NULL,
  channel_ref TEXT,                            -- Twilio CallSid
  started_at  TEXT NOT NULL,
  ended_at    TEXT,
  end_reason  TEXT,                            -- caller_hangup, agent_end, silence, time_limit, error
  summary     TEXT                             -- short summary used by the warm intro
);
CREATE INDEX sessions_caller ON sessions(caller_id, started_at);
CREATE INDEX sessions_started ON sessions(started_at);

CREATE TABLE turns (
  id          INTEGER PRIMARY KEY,
  session_id  INTEGER NOT NULL REFERENCES sessions(id),
  seq         INTEGER NOT NULL,                -- order within the session
  speaker     TEXT NOT NULL CHECK (speaker IN ('caller', 'agent', 'tool')),
  text        TEXT NOT NULL,                   -- for 'tool': tool name + JSON input
  started_at  TEXT NOT NULL,
  ended_at    TEXT,
  interrupted INTEGER NOT NULL DEFAULT 0,      -- 1 = caller cut the agent off; text is what was heard
  latency_ms  INTEGER,                         -- agent turns: end of caller speech -> first audio
  UNIQUE (session_id, seq)
);

CREATE TABLE messages (
  id          INTEGER PRIMARY KEY,
  session_id  INTEGER NOT NULL REFERENCES sessions(id),
  name        TEXT,
  contact     TEXT,                            -- how to reach them, if they gave one
  body        TEXT NOT NULL,
  created_at  TEXT NOT NULL,
  notified_at TEXT                             -- NULL until the email to the owner is sent
);
CREATE INDEX messages_session ON messages(session_id);
