"""SQLite implementation of core.interfaces.Storage.

One aiosqlite connection for the whole process (single uvicorn worker, design doc §14).
aiosqlite runs queries on its own thread; an asyncio.Lock keeps each write's statements
together so two live sessions can't interleave inside one transaction.

Timestamps are stored as ISO-8601 UTC text and returned as timezone-aware datetimes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from voice_agent.core.interfaces import Caller, MessageRow, SessionRow, SessionStart, TurnRow
from voice_agent.storage.migrate import apply_migrations


def utc_now() -> datetime:
    return datetime.now(UTC)


def _ts(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        raise ValueError("naive datetime; use timezone-aware UTC")
    return dt.astimezone(UTC).isoformat(timespec="milliseconds")


def _dt(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _dt_opt(text: str | None) -> datetime | None:
    return datetime.fromisoformat(text) if text else None


async def _one(cur: aiosqlite.Cursor) -> aiosqlite.Row:
    """Fetch a row that must exist (we just wrote it or looked it up by key)."""
    row = await cur.fetchone()
    if row is None:
        raise RuntimeError("expected a row, got none")
    return row


def _new_id(cur: aiosqlite.Cursor) -> int:
    if cur.lastrowid is None:
        raise RuntimeError("insert returned no row id")
    return cur.lastrowid


def _caller(r: aiosqlite.Row) -> Caller:
    return Caller(
        id=r["id"], channel=r["channel"], address=r["address"],
        name=r["name"], company=r["company"], reason=r["reason"],
        first_seen=_dt(r["first_seen"]), last_seen=_dt(r["last_seen"]),
        session_count=r["session_count"],
    )  # fmt: skip


def _session(r: aiosqlite.Row) -> SessionRow:
    return SessionRow(
        id=r["id"], caller_id=r["caller_id"], channel=r["channel"],
        channel_ref=r["channel_ref"], started_at=_dt(r["started_at"]),
        ended_at=_dt_opt(r["ended_at"]), end_reason=r["end_reason"], summary=r["summary"],
    )  # fmt: skip


def _message(r: aiosqlite.Row) -> MessageRow:
    return MessageRow(
        body=r["body"], created_at=_dt(r["created_at"]), name=r["name"],
        contact=r["contact"], id=r["id"], notified_at=_dt_opt(r["notified_at"]),
    )  # fmt: skip


class SqliteStorage:
    def __init__(self, conn: aiosqlite.Connection, now: Callable[[], datetime] = utc_now):
        self._db = conn
        self._now = now
        self._write = asyncio.Lock()

    @classmethod
    async def open(
        cls, db_path: Path | str, *, migrate: bool = True, now: Callable[[], datetime] = utc_now
    ) -> SqliteStorage:
        if migrate:
            apply_migrations(db_path)
        conn = await aiosqlite.connect(db_path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=5000")
        return cls(conn, now)

    async def close(self) -> None:
        await self._db.close()

    # -- callers -----------------------------------------------------------

    async def get_or_create_caller(self, channel: str, address: str | None) -> Caller:
        """Known number -> its existing row. New number -> new row.
        No number (withheld) -> a new anonymous row every time (spec D30)."""
        now = _ts(self._now())
        async with self._write:
            if address is None:
                cur = await self._db.execute(
                    "INSERT INTO callers (channel, address, first_seen, last_seen) "
                    "VALUES (?, NULL, ?, ?)",
                    (channel, now, now),
                )
                caller_id = _new_id(cur)
            else:
                await self._db.execute(
                    "INSERT INTO callers (channel, address, first_seen, last_seen) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT (channel, address) WHERE address IS NOT NULL DO NOTHING",
                    (channel, address, now, now),
                )
                cur = await self._db.execute(
                    "SELECT id FROM callers WHERE channel = ? AND address = ?", (channel, address)
                )
                caller_id = (await _one(cur))["id"]
            await self._db.commit()
        cur = await self._db.execute("SELECT * FROM callers WHERE id = ?", (caller_id,))
        return _caller(await _one(cur))

    async def set_caller_details(
        self, caller_id: int, name: str | None, company: str | None, reason: str | None
    ) -> None:
        """Fill in what the caller told us. A None argument keeps the stored value."""
        async with self._write:
            await self._db.execute(
                "UPDATE callers SET name = COALESCE(?, name), company = COALESCE(?, company), "
                "reason = COALESCE(?, reason) WHERE id = ?",
                (name, company, reason, caller_id),
            )
            await self._db.commit()

    # -- sessions ----------------------------------------------------------

    async def last_session(self, caller_id: int) -> SessionRow | None:
        cur = await self._db.execute(
            "SELECT * FROM sessions WHERE caller_id = ? ORDER BY started_at DESC, id DESC LIMIT 1",
            (caller_id,),
        )
        row = await cur.fetchone()
        return _session(row) if row else None

    async def create_session(self, caller_id: int, start: SessionStart) -> int:
        """Insert the session and bump the caller's count and last_seen in one transaction."""
        now = _ts(self._now())
        async with self._write:
            cur = await self._db.execute(
                "INSERT INTO sessions (caller_id, channel, channel_ref, started_at) "
                "VALUES (?, ?, ?, ?)",
                (caller_id, start.channel, start.channel_ref, now),
            )
            await self._db.execute(
                "UPDATE callers SET session_count = session_count + 1, last_seen = ? WHERE id = ?",
                (now, caller_id),
            )
            await self._db.commit()
            return _new_id(cur)

    async def end_session(self, session_id: int, reason: str, summary: str | None) -> None:
        async with self._write:
            await self._db.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ?, summary = ? WHERE id = ?",
                (_ts(self._now()), reason, summary, session_id),
            )
            await self._db.commit()

    async def count_sessions_since(self, since: datetime, caller_id: int | None = None) -> int:
        sql = "SELECT COUNT(*) FROM sessions WHERE started_at >= ?"
        args: tuple = (_ts(since),)
        if caller_id is not None:
            sql += " AND caller_id = ?"
            args += (caller_id,)
        cur = await self._db.execute(sql, args)
        return (await _one(cur))[0]

    # -- turns and messages --------------------------------------------------

    async def add_turn(self, session_id: int, turn: TurnRow) -> None:
        async with self._write:
            await self._db.execute(
                "INSERT INTO turns (session_id, seq, speaker, text, started_at, ended_at, "
                "interrupted, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id, turn.seq, turn.speaker, turn.text, _ts(turn.started_at),
                    _ts(turn.ended_at), int(turn.interrupted), turn.latency_ms,
                ),
            )  # fmt: skip
            await self._db.commit()

    async def save_message(self, session_id: int, msg: MessageRow) -> int:
        """Save the session's message, or replace it if it already has one (one per
        session). `created_at` stays the time of the first save. Returns the message id."""
        async with self._write:
            cur = await self._db.execute(
                "INSERT INTO messages (session_id, name, contact, body, created_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (session_id) DO UPDATE SET "
                "name = excluded.name, contact = excluded.contact, body = excluded.body "
                "RETURNING id",
                (session_id, msg.name, msg.contact, msg.body, _ts(msg.created_at)),
            )
            (message_id,) = await _one(cur)
            await self._db.commit()
            return message_id

    async def session_message(self, session_id: int) -> MessageRow | None:
        cur = await self._db.execute(
            "SELECT id, name, contact, body, created_at, notified_at FROM messages "
            "WHERE session_id = ?",
            (session_id,),
        )
        r = await cur.fetchone()
        return _message(r) if r else None

    async def mark_message_notified(self, message_id: int) -> None:
        async with self._write:
            await self._db.execute(
                "UPDATE messages SET notified_at = ? WHERE id = ?", (_ts(self._now()), message_id)
            )
            await self._db.commit()

    async def unsent_messages(self) -> list[tuple[MessageRow, Caller, SessionRow]]:
        """Messages whose email never went out (it failed, or the process stopped before the
        call ended), oldest first, each with its caller and session for the email."""
        cur = await self._db.execute("SELECT * FROM messages WHERE notified_at IS NULL ORDER BY id")
        found = []
        for r in await cur.fetchall():
            cur = await self._db.execute("SELECT * FROM sessions WHERE id = ?", (r["session_id"],))
            session = _session(await _one(cur))
            cur = await self._db.execute("SELECT * FROM callers WHERE id = ?", (session.caller_id,))
            found.append((_message(r), _caller(await _one(cur)), session))
        return found
