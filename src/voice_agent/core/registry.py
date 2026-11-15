"""Who's on the line: admission checks, live sessions, and shutdown.

A call reaches us in two steps, and the registry is involved in both:

    1. webhook:  slot = await registry.admit("twilio", from_number)
                 Refusal -> the adapter rejects (BLOCKED) or plays the busy clip (the rest)
                 Slot    -> the adapter puts slot.token in the stream URL
    2. stream:   callbacks = registry.start_session(token, start, adapter)
                 (the adapter feeds audio and mark echoes to `callbacks`)

The slot is held from step 1, so two calls ringing at once can't both squeeze into the last
free place. It doubles as the stream token (design §7.1): random, single use, and void after
`slot_ttl_s`, so nobody can open a stream without going through the webhook first.

Limits (spec: caps):
    max_concurrent    live sessions + slots waiting for their stream
    daily_cap         sessions in the last 24 hours, all callers
    per_caller_daily  sessions in the last 24 hours from one number (withheld numbers can't
                      be told apart, so only the daily cap applies to them)

"Daily" is a rolling 24 hours: no timezone to get wrong, and no burst allowed at midnight.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol

from voice_agent.core.interfaces import ChannelAdapter, SessionCallbacks, SessionStart, Storage

log = logging.getLogger(__name__)

DAY = timedelta(hours=24)
SLOT_TTL_S = 60.0  # webhook to stream start is normally under a second
SHUTDOWN = "shutdown"  # end reason for sessions still live when the server stops


class RunnableSession(SessionCallbacks, Protocol):
    async def run(self) -> str: ...


SessionFactory = Callable[[SessionStart, ChannelAdapter], RunnableSession]


class Refusal(StrEnum):
    BLOCKED = "blocked"  # on the blocklist: reject without answering
    CONCURRENT = "concurrent"  # the rest: answer with the busy clip
    DAILY_CAP = "daily_cap"
    CALLER_CAP = "caller_cap"


class AdmissionError(RuntimeError):
    """A stream arrived with an unknown, used or expired token, or not matching its slot."""


@dataclass(frozen=True, slots=True)
class Limits:
    max_concurrent: int
    daily_cap: int
    per_caller_daily: int
    blocked: frozenset[str] = frozenset()  # E.164 numbers


@dataclass(frozen=True, slots=True)
class Slot:
    token: str
    channel: str
    address: str | None
    expires_at: float  # monotonic time


@dataclass(slots=True)
class _Live:
    session: RunnableSession
    task: asyncio.Task[str]


class Registry:
    def __init__(
        self,
        storage: Storage,
        limits: Limits,
        new_session: SessionFactory,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
        slot_ttl_s: float = SLOT_TTL_S,
    ) -> None:
        self._storage = storage
        self._limits = limits
        self._new_session = new_session
        self._clock = clock
        self._monotonic = monotonic
        self._slot_ttl_s = slot_ttl_s
        self._slots: dict[str, Slot] = {}
        self._live: dict[str, _Live] = {}  # by channel_ref
        self._admit_lock = asyncio.Lock()  # check-then-reserve must not interleave
        self._closing = False

    @property
    def live_count(self) -> int:
        return len(self._live)

    # -- step 1: the webhook ------------------------------------------------------------

    async def admit(self, channel: str, address: str | None) -> Slot | Refusal:
        if address is not None and address in self._limits.blocked:
            return Refusal.BLOCKED
        async with self._admit_lock:
            self._drop_expired()
            if self._closing or len(self._live) + len(self._slots) >= self._limits.max_concurrent:
                return Refusal.CONCURRENT

            since = self._clock() - DAY
            pending = list(self._slots.values())
            today = await self._storage.count_sessions_since(since)
            if today + len(pending) >= self._limits.daily_cap:
                return Refusal.DAILY_CAP

            if address is not None:
                caller = await self._storage.get_or_create_caller(channel, address)
                theirs = await self._storage.count_sessions_since(since, caller.id)
                theirs += sum(1 for s in pending if (s.channel, s.address) == (channel, address))
                if theirs >= self._limits.per_caller_daily:
                    return Refusal.CALLER_CAP

            slot = Slot(
                token=secrets.token_urlsafe(24),
                channel=channel,
                address=address,
                expires_at=self._monotonic() + self._slot_ttl_s,
            )
            self._slots[slot.token] = slot
            return slot

    def release(self, token: str) -> None:
        """Give a slot back without starting a session (the call ended before its stream)."""
        self._slots.pop(token, None)

    # -- step 2: the stream -----------------------------------------------------------------

    def start_session(
        self, token: str, start: SessionStart, adapter: ChannelAdapter
    ) -> SessionCallbacks:
        """Use up the slot and start the session. Raises AdmissionError if the token isn't a
        live slot for this caller; the adapter should then close the stream."""
        self._drop_expired()
        slot = self._slots.pop(token, None)
        if slot is None:
            raise AdmissionError("unknown, used or expired stream token")
        if (slot.channel, slot.address) != (start.channel, start.caller_address):
            raise AdmissionError("stream doesn't match the call that was admitted")
        if start.channel_ref in self._live:
            raise AdmissionError(f"session {start.channel_ref} is already running")

        session = self._new_session(start, adapter)
        task = asyncio.create_task(self._run(start.channel_ref, session))
        self._live[start.channel_ref] = _Live(session, task)
        return session

    async def _run(self, ref: str, session: RunnableSession) -> str:
        try:
            return await session.run()
        except Exception:
            log.exception("session %s crashed outside its own error handling", ref)
            return "error"
        finally:
            self._live.pop(ref, None)

    # -- shutdown ---------------------------------------------------------------------------

    async def shutdown(self, timeout_s: float = 10.0) -> None:
        """Stop admitting, end every live session the normal way (so turns, summaries and
        end reasons are saved), and cancel any that don't finish within `timeout_s`."""
        self._closing = True
        self._slots.clear()
        live = list(self._live.values())
        for entry in live:
            await entry.session.on_session_end(SHUTDOWN)
        tasks = [entry.task for entry in live]
        if not tasks:
            return
        _, stuck = await asyncio.wait(tasks, timeout=timeout_s)
        for task in stuck:
            task.cancel()
        if stuck:
            log.warning(
                "%d session(s) didn't end in %.0f s and were cancelled", len(stuck), timeout_s
            )
            await asyncio.wait(stuck)

    # -- helpers ----------------------------------------------------------------------------

    def _drop_expired(self) -> None:
        now = self._monotonic()
        for token in [t for t, s in self._slots.items() if s.expires_at <= now]:
            del self._slots[token]
