"""Three 3.0.2-era defects of the rolling wrap's save, each pinned by the failure that was
reproduced on a scratch store before the fix (two processes for the first two; the tests
here drive the other party from a thread or a second connection, which the same locks
serialise):

- a baton take could land between the in-batch baton check and the commit;
- the save read its wrap before taking the write lock and compared only the token, so a
  wrap cancelled and prepared again under the same caller-supplied token took the save;
- a keyed prepare's replay re-planned the step from its frozen episodes alone, so a slice
  of a long day came back as a whole day.
"""

from __future__ import annotations

import json
import sqlite3
import threading

import pytest

from anneal_memory import Store, register_writer_schema, rolling, sessions
from anneal_memory import continuity
from anneal_memory.continuity import prepare_wrap, validated_save_continuity
from anneal_memory.store import IdempotencyConflictError

try:
    import fcntl  # noqa: F401
    HAS_FLOCK = True
except ImportError:  # the Windows job: the baton lock is a no-op there
    HAS_FLOCK = False


def _record(store: Store, n: int = 2, day: str = "2026-03-01", pad: int = 0) -> None:
    for i in range(n):
        store.record(
            f"note {i} " + "x" * pad, "observation", timestamp=f"{day}T12:0{i}:00.000000Z"
        )


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "m.db", project_name="T")
    yield s
    s.close()


@pytest.mark.skipif(not HAS_FLOCK, reason="the baton lock is an flock")
def test_a_baton_take_waits_for_a_save_in_its_commit_window(store, monkeypatch):
    """The in-batch check passes for s1; a take for s2 started right after must not finish
    until the commit, so no save commits under a baton its session no longer holds."""
    _record(store)
    path = store.continuity_path
    sessions.register_session(path, "s1")
    sessions.claim_baton(path, "s1")
    prep = prepare_wrap(store, tz="UTC", session_id="s1")
    real = continuity._check_save_authority
    seen: dict[str, object] = {"calls": 0}

    def take_after_the_in_batch_check(*args, **kwargs):
        real(*args, **kwargs)
        seen["calls"] = int(seen["calls"]) + 1  # type: ignore[call-overload]
        if seen["calls"] == 2:
            taker = threading.Thread(
                target=lambda: sessions.claim_baton(path, "s2", take=True), daemon=True)
            taker.start()
            taker.join(timeout=0.5)
            seen["taker"] = taker
            seen["finished_before_commit"] = not taker.is_alive()

    monkeypatch.setattr(continuity, "_check_save_authority", take_after_the_in_batch_check)
    validated_save_continuity(store, "memory", wrap_token=prep["wrap_token"], session_id="s1")
    taker = seen["taker"]
    assert isinstance(taker, threading.Thread)
    taker.join(timeout=10)
    assert seen["finished_before_commit"] is False
    assert not taker.is_alive()
    assert sessions.baton_holder(path) == "s2"  # the take happened, after the commit
    assert store.load_continuity() == "memory"


@pytest.mark.skipif(not HAS_FLOCK, reason="the baton lock is an flock")
def test_a_save_releases_the_baton_lock_when_it_is_refused(store):
    """A refused gated save must not leave the baton locked for every later claim."""
    _record(store)
    path = store.continuity_path
    sessions.register_session(path, "s1")
    sessions.claim_baton(path, "s1")
    prep = prepare_wrap(store, tz="UTC", session_id="s1")
    with pytest.raises(ValueError, match="empty|at least|non-empty|characters"):
        validated_save_continuity(
            store, "   ", wrap_token=prep["wrap_token"], session_id="s1")
    done = threading.Event()
    threading.Thread(
        target=lambda: (sessions.claim_baton(path, "s3", take=True), done.set()),
        daemon=True).start()
    assert done.wait(timeout=10)


def _replace_wrap_under_the_same_token(db_path, token: str) -> None:
    other = Store(db_path, project_name="T")
    try:
        other.wrap_cancelled(expect_token=token)
        r = prepare_wrap(other, tz="UTC", wrap_token=token, max_chars=1000)
        assert r["status"] == "ready"
    finally:
        other.close()


def test_a_wrap_replaced_under_the_same_token_is_not_closed_by_the_old_reply(
    store, monkeypatch,
):
    """The save checked a 3000-character reply against the first wrap's limit. Another
    connection cancels that wrap and prepares another with the same caller token and a
    1000-character limit: the save must refuse, and the new wrap must stay open."""
    _record(store)
    token = "ab" * 16
    assert prepare_wrap(store, tz="UTC", wrap_token=token)["status"] == "ready"
    real = rolling.check_reply
    state = {"replaced": False}

    def replace_after_the_check(text, limit):
        body = real(text, limit)
        if not state["replaced"]:
            state["replaced"] = True
            _replace_wrap_under_the_same_token(store.path, token)
        return body

    monkeypatch.setattr(rolling, "check_reply", replace_after_the_check)
    with pytest.raises(ValueError, match="different wrap"):
        validated_save_continuity(store, "x" * 3000, wrap_token=token)
    snapshot = store.load_wrap_snapshot()
    assert snapshot is not None and snapshot["token"] == token
    assert store.wrap_max_chars() == 1000
    assert store.last_wrap_id() == 0  # no wrap was recorded
    assert store.load_continuity() in (None, "")


def test_a_save_whose_wrap_is_unchanged_still_commits(store):
    """The identity check must not refuse an ordinary save, caller token or minted."""
    _record(store)
    token = "cd" * 16
    prep = prepare_wrap(store, tz="UTC", wrap_token=token)
    out = validated_save_continuity(store, "memory", wrap_token=prep["wrap_token"])
    assert out["episodes_compressed"] == 2 and store.load_wrap_snapshot() is None


@pytest.fixture
def long_day(store, monkeypatch):
    """A day of five 130-character episodes, cut into five slices of one."""
    monkeypatch.setitem(rolling.plan_steps.__kwdefaults__, "chunk_max", 300)  # type: ignore[index]
    _record(store, n=5, pad=120)
    return store


def test_a_keyed_prepare_replay_returns_the_slice_numbers_of_the_first_call(long_day):
    first = prepare_wrap(long_day, tz="UTC", idempotency_key="k1")
    again = prepare_wrap(long_day, tz="UTC", idempotency_key="k1")
    assert (first["step"]["chunk"], first["step"]["chunks"]) == (0, 5)
    assert (again["step"]["chunk"], again["step"]["chunks"]) == (0, 5)
    assert again["step"]["steps_remaining"] == first["step"]["steps_remaining"] == 4
    assert again["step"]["prompt"] == first["step"]["prompt"]


def test_a_replay_after_the_second_slice_keeps_that_slices_numbers(long_day):
    prep = prepare_wrap(long_day, tz="UTC", idempotency_key="k1")
    validated_save_continuity(long_day, "one", wrap_token=prep["wrap_token"])
    second = prepare_wrap(long_day, tz="UTC", idempotency_key="k2")
    again = prepare_wrap(long_day, tz="UTC", idempotency_key="k2")
    assert (second["step"]["chunk"], second["step"]["chunks"]) == (
        again["step"]["chunk"], again["step"]["chunks"])


def _set_key_row(store, key: str, mutate) -> None:
    with sqlite3.connect(store.path) as conn:
        register_writer_schema(conn)
        (raw,) = conn.execute(
            "SELECT result_json FROM idempotency_keys WHERE key = ?", (key,)).fetchone()
        row = json.loads(raw)
        mutate(row)
        conn.execute(
            "UPDATE idempotency_keys SET result_json = ? WHERE key = ?",
            (json.dumps(row), key))


def test_a_key_row_from_before_the_slice_numbers_were_kept_still_replays(long_day):
    first = prepare_wrap(long_day, tz="UTC", idempotency_key="k1")
    _set_key_row(long_day, "k1", lambda row: (row.pop("chunk"), row.pop("chunks")))
    again = prepare_wrap(long_day, tz="UTC", idempotency_key="k1")
    assert again["wrap_token"] == first["wrap_token"]
    assert (again["step"]["chunk"], again["step"]["chunks"]) == (0, 1)  # re-planned, as before


@pytest.mark.parametrize("chunk,chunks", [(5, 5), (-1, 5), ("0", 5), (0, True), (0, None)])
def test_a_malformed_slice_in_a_key_row_is_refused_not_returned(long_day, chunk, chunks):
    prepare_wrap(long_day, tz="UTC", idempotency_key="k1")

    def spoil(row):
        row["chunk"], row["chunks"] = chunk, chunks

    _set_key_row(long_day, "k1", spoil)
    with pytest.raises(IdempotencyConflictError) as err:
        prepare_wrap(long_day, tz="UTC", idempotency_key="k1")
    assert err.value.reason == "malformed_row"
