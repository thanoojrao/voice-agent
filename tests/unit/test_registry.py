"""Admission limits, slots as stream tokens, live sessions and shutdown."""

import asyncio
from datetime import UTC, datetime

import pytest

from tests.fakes import FakeVoiceAdapter
from voice_agent.core.interfaces import ChannelAdapter, SessionStart
from voice_agent.core.registry import (
    SHUTDOWN,
    AdmissionError,
    Limits,
    Refusal,
    Registry,
    Slot,
)
from voice_agent.storage.sqlite import SqliteStorage

NOW = datetime(2026, 11, 15, 15, 0, tzinfo=UTC)
SAM, ALEX, BLOCKED = "+15550000001", "+15550000002", "+15550000666"
LIMITS = Limits(max_concurrent=2, daily_cap=5, per_caller_daily=2, blocked=frozenset({BLOCKED}))


class StubSession:
    """Runs until told the session ended; records how it ended."""

    def __init__(self, start: SessionStart, stubborn: bool = False) -> None:
        self.start = start
        self.stubborn = stubborn  # ignores on_session_end, to test the shutdown timeout
        self.ended = asyncio.Event()
        self.reason: str | None = None
        self.cancelled = False

    async def run(self) -> str:
        try:
            await self.ended.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return self.reason or "?"

    async def on_audio(self, pcm: bytes) -> None:
        pass

    async def on_mark_played(self, name: str) -> None:
        pass

    async def on_session_end(self, reason: str) -> None:
        if not self.stubborn:
            self.reason = reason
            self.ended.set()


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
async def store(tmp_path):
    s = await SqliteStorage.open(tmp_path / "agent.db", now=lambda: NOW)
    yield s
    await s.close()


class Rig:
    def __init__(self, store: SqliteStorage, limits: Limits = LIMITS, stubborn=False) -> None:
        self.store = store
        self.sessions: list[StubSession] = []
        self.mono = Clock()
        self.stubborn = stubborn
        self.registry = Registry(
            store, limits, self.make, clock=lambda: NOW, monotonic=self.mono, slot_ttl_s=60
        )
        self.refs = 0

    def make(self, start: SessionStart, adapter: ChannelAdapter) -> StubSession:
        session = StubSession(start, self.stubborn)
        self.sessions.append(session)
        return session

    async def admit(self, address: str | None = SAM) -> Slot:
        slot = await self.registry.admit("twilio", address)
        assert isinstance(slot, Slot), slot
        return slot

    def start(self, slot: Slot) -> StubSession:
        self.refs += 1
        start = SessionStart("twilio", f"CA{self.refs}", slot.address, 8000)
        self.registry.start_session(slot.token, start, FakeVoiceAdapter())
        return self.sessions[-1]

    async def past_sessions(self, address: str | None, count: int) -> None:
        """Sessions already in the database from earlier today."""
        for i in range(count):
            caller = await self.store.get_or_create_caller("twilio", address)
            await self.store.create_session(
                caller.id, SessionStart("twilio", f"OLD{address}{i}", address, 8000)
            )


async def settle() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


# -- admission ---------------------------------------------------------------------


async def test_blocked_number_is_refused_first(store):
    rig = Rig(store)
    assert await rig.registry.admit("twilio", BLOCKED) is Refusal.BLOCKED


async def test_slot_tokens_are_long_and_unique(store):
    rig = Rig(store)
    a, b = await rig.admit(SAM), await rig.admit(ALEX)
    assert a.token != b.token and len(a.token) >= 32


async def test_concurrency_counts_live_sessions_and_waiting_slots(store):
    rig = Rig(store)
    rig.start(await rig.admit(SAM))  # live
    await rig.admit(ALEX)  # admitted, stream not started yet
    assert await rig.registry.admit("twilio", None) is Refusal.CONCURRENT


async def test_finished_session_frees_its_place(store):
    rig = Rig(store)
    first = rig.start(await rig.admit(SAM))
    rig.start(await rig.admit(ALEX))
    assert rig.registry.live_count == 2
    await first.on_session_end("caller_hangup")
    await settle()
    assert rig.registry.live_count == 1
    assert isinstance(await rig.registry.admit("twilio", None), Slot)


async def test_expired_and_released_slots_free_their_place(store):
    rig = Rig(store)
    stale = await rig.admit(SAM)
    released = await rig.admit(ALEX)
    rig.registry.release(released.token)
    rig.mono.t = 61  # `stale` has expired
    assert isinstance(await rig.registry.admit("twilio", None), Slot)
    assert isinstance(await rig.registry.admit("twilio", None), Slot)
    with pytest.raises(AdmissionError):
        rig.start(stale)


async def test_daily_cap_counts_stored_sessions_and_waiting_slots(store):
    rig = Rig(store)
    await rig.past_sessions(None, 4)
    await rig.admit(SAM)  # the 5th
    assert await rig.registry.admit("twilio", ALEX) is Refusal.DAILY_CAP


async def test_per_caller_cap_counts_stored_sessions_and_waiting_slots(store):
    rig = Rig(store, Limits(max_concurrent=5, daily_cap=50, per_caller_daily=2))
    await rig.past_sessions(SAM, 1)
    await rig.admit(SAM)  # Sam's 2nd
    assert await rig.registry.admit("twilio", SAM) is Refusal.CALLER_CAP
    assert isinstance(await rig.registry.admit("twilio", ALEX), Slot)


async def test_the_owners_number_skips_the_per_caller_cap_but_not_the_shared_ones(store):
    rig = Rig(
        store, Limits(max_concurrent=5, daily_cap=4, per_caller_daily=1, exempt=frozenset({ALEX}))
    )
    await rig.past_sessions(ALEX, 2)
    assert isinstance(await rig.registry.admit("twilio", ALEX), Slot)  # 3rd today: fine
    assert isinstance(await rig.registry.admit("twilio", SAM), Slot)  # 4th overall
    assert await rig.registry.admit("twilio", ALEX) is Refusal.DAILY_CAP  # shared cap still holds
    await rig.past_sessions(SAM, 1)
    assert await rig.registry.admit("twilio", SAM) is Refusal.DAILY_CAP


async def test_withheld_numbers_only_meet_the_shared_caps(store):
    rig = Rig(store, Limits(max_concurrent=5, daily_cap=50, per_caller_daily=1))
    await rig.admit(None)
    assert isinstance(await rig.registry.admit("twilio", None), Slot)


async def test_simultaneous_calls_cannot_both_take_the_last_place(store):
    rig = Rig(store, Limits(max_concurrent=1, daily_cap=50, per_caller_daily=5))
    results = await asyncio.gather(*(rig.registry.admit("twilio", n) for n in (SAM, ALEX)))
    assert sorted(isinstance(r, Slot) for r in results) == [False, True]


# -- starting sessions ---------------------------------------------------------------


async def test_a_token_works_once(store):
    rig = Rig(store)
    slot = await rig.admit()
    rig.start(slot)
    with pytest.raises(AdmissionError):
        rig.start(slot)


async def test_unknown_token_is_refused(store):
    rig = Rig(store)
    with pytest.raises(AdmissionError):
        rig.registry.start_session(
            "made-up", SessionStart("twilio", "CA1", SAM, 8000), FakeVoiceAdapter()
        )


async def test_stream_must_match_the_admitted_call(store):
    rig = Rig(store)
    slot = await rig.admit(SAM)
    with pytest.raises(AdmissionError):
        rig.registry.start_session(
            slot.token, SessionStart("twilio", "CA1", ALEX, 8000), FakeVoiceAdapter()
        )


async def test_start_returns_the_session_as_callbacks(store):
    rig = Rig(store)
    slot = await rig.admit()
    callbacks = rig.registry.start_session(
        slot.token, SessionStart("twilio", "CA9", SAM, 8000), FakeVoiceAdapter()
    )
    assert callbacks is rig.sessions[-1]


# -- shutdown ------------------------------------------------------------------------


async def test_shutdown_ends_live_sessions_and_stops_admitting(store):
    rig = Rig(store)
    a, b = rig.start(await rig.admit(SAM)), rig.start(await rig.admit(ALEX))
    await rig.registry.shutdown()
    assert (a.reason, b.reason) == (SHUTDOWN, SHUTDOWN)
    assert rig.registry.live_count == 0
    assert await rig.registry.admit("twilio", None) is Refusal.CONCURRENT


async def test_shutdown_cancels_sessions_that_do_not_end(store):
    rig = Rig(store, stubborn=True)
    session = rig.start(await rig.admit())
    await rig.registry.shutdown(timeout_s=0.05)
    assert session.cancelled
    assert rig.registry.live_count == 0
