"""spore-1233: an episode recorded while a wrap is open must reach a later window.

Both tests were built from failures reproduced on 0.9.17.dev0 before the fix:
the sequential case stranded the late episode on the second wrap, and a real
two-process run (recorder vs prepare/save loop) stranded 2,517 of 3,459.
"""

from __future__ import annotations

import datetime
import sys
import threading

import pytest

from anneal_memory import Store
from anneal_memory.store import StoreError
from anneal_memory.continuity import prepare_wrap, validated_save_continuity

TODAY = datetime.date.today().isoformat()


def _text(n: int) -> str:
    return (
        f"# T — Memory (v1)\n\n## State\nWorking {n}.\n\n"
        f"## Patterns\nthought: p | 1x ({TODAY})\n\n"
        f"## Decisions\nNone.\n\n## Context\nPass {n}.\n"
    )


def test_episode_recorded_during_an_open_wrap_survives_every_cycle(tmp_path):
    store = Store(tmp_path / "m.db", project_name="t")
    try:
        for cycle in (1, 2, 3):
            store.record(f"in-window {cycle}", "observation")
            prep = prepare_wrap(store)
            late = store.record(f"late {cycle}", "observation")
            validated_save_continuity(
                store, _text(cycle), wrap_token=prep["wrap_token"], today=TODAY
            )
            window = {e.id for e in store.episodes_since_wrap()}
            assert late.id in window, f"cycle {cycle}: late episode stranded"
    finally:
        store.close()


def test_record_cannot_stamp_a_session_a_concurrent_wrap_just_closed(tmp_path):
    """Force the interleaving: a second connection commits a wrap between
    record()'s session read and its INSERT."""
    db = tmp_path / "m.db"
    store = Store(db, project_name="t")
    errors: list[BaseException] = []
    snapshot: list[str] = []

    def wrap_on_another_connection() -> None:
        # A sqlite connection is bound to its thread, so the whole second
        # store lives here.
        try:
            other = Store(db, project_name="t")
            try:
                prep = prepare_wrap(other)
                snapshot.extend(other.load_wrap_snapshot()["episode_ids"])
                validated_save_continuity(
                    other, _text(1), wrap_token=prep["wrap_token"], today=TODAY
                )
            finally:
                other.close()
        except BaseException as exc:  # surfaced by the assert below
            errors.append(exc)

    try:
        # Get past the first wrap, so records are session-stamped, not NULL.
        store.record("seed", "observation")
        first = prepare_wrap(store)
        validated_save_continuity(store, _text(0), wrap_token=first["wrap_token"], today=TODAY)
        store.record("in-window", "observation")

        saver = threading.Thread(target=wrap_on_another_connection)
        real = store._current_session_id

        def read_then_let_the_wrap_run():
            sid = real()
            saver.start()
            saver.join(timeout=2)  # fixed: the second Store() blocks on record's write lock
            return sid

        store._current_session_id = read_then_let_the_wrap_run  # type: ignore[method-assign]
        late = store.record("late", "observation")
        store._current_session_id = real  # type: ignore[method-assign]
        saver.join(timeout=10)
        assert not saver.is_alive()
        assert errors == []
        assert store.status().total_wraps == 2

        # Compressed by the wrap, or waiting in the open window; anything else
        # is stranded.
        window = {e.id for e in store.episodes_since_wrap()}
        assert late.id in window or late.id in snapshot, (
            "late episode stamped into the just-closed session"
        )
    finally:
        store.close()


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="sqlite3.Connection.setlimit is Python 3.11+"
)
def test_snapshot_at_the_variable_guard_fits_every_statement(tmp_path):
    """L1 (spore-1233 review), reproduced: the carry-over UPDATE binds three
    parameters beside the snapshot ids, so a snapshot the guard admitted could
    still overflow SQLite's 999-variable build limit inside the transaction.
    With JSON1 a snapshot past that limit completes (a big dream day, codex
    L3 1018); without it the guard still holds at its boundary."""
    import sqlite3

    from anneal_memory import _sqlkeys

    for json1, size in ((True, 2000), (False, Store._MAX_SQL_VARS_IN_CLAUSE)):
        store = Store(tmp_path / f"m{json1}.db", project_name="t")
        if json1 and not _sqlkeys.json_each_available(store._conn):
            store.close()
            continue
        saved = _sqlkeys._json_each
        _sqlkeys._json_each = json1
        try:
            store._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
            for cycle in (1, 2):  # the second wrap has a previous wrap, so the carry-over runs
                store.record(f"in-window {cycle}", "observation")
                ids = [f"{i:08x}" for i in range(size)]
                store.wrap_completed(episodes_compressed=1, continuity_chars=10, episode_ids=ids)
            assert store.status().total_wraps == 2
            if not json1:
                with pytest.raises(StoreError):
                    store.wrap_completed(
                        episodes_compressed=1, continuity_chars=10,
                        episode_ids=ids + ["one-more"],
                    )
        finally:
            _sqlkeys._json_each = saved
            store.close()


def test_a_wrap_completed_during_prepare_is_not_overwritten(tmp_path):
    """L2 (spore-1233 review), reproduced: prepare_wrap read its window, a whole
    wrap completed on another connection while it was still building the
    package (here inside the prepare-time re-derive), and the stale prepare
    then opened a wrap whose save overwrote the completed one. The episode
    only that wrap had compressed was lost."""
    from anneal_memory import continuity as C
    from anneal_memory.schema import PROJECT_SCHEMA

    def text(tag: str) -> str:
        return "\n".join([
            "# t — Memory (v1)", "", "## Plan", f"- plan {tag}", "",
            "## State", f"- state {tag} [judged: t, {TODAY}, x]", "",
            "## Decisions", "- d", "", "## Open", "- o", "",
            "## Lessons", "- none yet", "", "## History", f"- {tag}", "",
        ])

    db = tmp_path / "m.db"
    a = Store(db, project_name="t", section_schema=PROJECT_SCHEMA)
    errors: list[BaseException] = []

    def other_wrap() -> None:
        try:
            b = Store(db, project_name="t", section_schema=PROJECT_SCHEMA)
            try:
                b.record("late, only the other wrap sees it", "observation")
                prep = C.prepare_wrap(b)
                b.record("after the other wrap's snapshot", "observation")
                C.validated_save_continuity(b, text("B"), wrap_token=prep["wrap_token"], today=TODAY)
            finally:
                b.close()
        except BaseException as exc:
            errors.append(exc)

    real = C.rederive_text
    fired: list[int] = []

    def rederive_while_another_wrap_completes(*args, **kwargs):
        if not fired:
            fired.append(1)
            t = threading.Thread(target=other_wrap)
            t.start()
            t.join()
        return real(*args, **kwargs)

    try:
        a.record("e0", "observation")
        first = C.prepare_wrap(a)
        C.validated_save_continuity(a, text("W1"), wrap_token=first["wrap_token"], today=TODAY)
        a.record("early", "observation")
        C.rederive_text = rederive_while_another_wrap_completes  # type: ignore[assignment]
        try:
            result = C.prepare_wrap(a)
        finally:
            C.rederive_text = real  # type: ignore[assignment]
        assert errors == []
        assert result["status"] != "ready"  # told to retry, no wrap opened
        # codex L3 r1: the retry result must count what is still pending, not 0.
        assert result["episode_count"] == len(a.episodes_since_wrap()) == 1
        assert a.load_wrap_snapshot() is None
        assert "- B" in (a.load_continuity() or "")
        assert a.status().total_wraps == 2
    finally:
        a.close()


def test_record_inside_a_batch_reads_its_session_under_the_write_lock(tmp_path):
    """L2 (spore-1233 review), reproduced: inside _batch() record() skipped the
    BEGIN IMMEDIATE, so its session read ran unlocked and a peer wrap could
    commit before the INSERT. The lock must be held from the read on."""
    import sqlite3

    db = tmp_path / "m.db"
    store = Store(db, project_name="t")
    peer = sqlite3.connect(db, timeout=0, isolation_level=None)
    try:
        real = store._current_session_id
        seen: list[bool] = []

        def read_and_probe():
            try:
                peer.execute("BEGIN IMMEDIATE")
                peer.execute("ROLLBACK")
                seen.append(False)  # the peer got the write lock: the read was unlocked
            except sqlite3.OperationalError:
                seen.append(True)
            return real()

        store._current_session_id = read_and_probe  # type: ignore[method-assign]
        with store._batch():
            store.record("in a batch", "observation")
        assert seen == [True]
    finally:
        peer.close()
        store.close()


@pytest.mark.parametrize("offset_days", [-40, 3], ids=["replayed-past", "future"])
def test_a_mid_wrap_episode_with_a_replayed_timestamp_reaches_the_next_window(
    tmp_path, offset_days
):
    """A caller that replays episodes passes the timestamp each was first written
    under, so one recorded mid-wrap can sort anywhere in time. A real run (a
    second process recording across a whole wrap, replayed and future
    timestamps among them) lost none: the wrap's membership is the frozen id
    snapshot, never a timestamp, so the late episode stays out of this wrap's
    compression and is carried to the next one. Mutation-checked: with the
    carry-over UPDATE in wrap_completed disabled, both cases fail."""
    store = Store(tmp_path / "m.db", project_name="t")
    try:
        # Past the first wrap, so records are session-stamped, not NULL.
        store.record("seed", "observation")
        first = prepare_wrap(store)
        validated_save_continuity(store, _text(0), wrap_token=first["wrap_token"], today=TODAY)

        store.record("in-window", "observation")
        prep = prepare_wrap(store)
        snapshot = set(store.load_wrap_snapshot()["episode_ids"])
        when = (
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=offset_days)
        ).isoformat()
        in_window = next(e for e in store.episodes_since_wrap() if e.content == "in-window")
        late = store.record("late, replayed", "observation", timestamp=when)
        assert late.id not in snapshot
        # The offset must survive storage, or the two cases collapse into one.
        def parsed(ts: str) -> datetime.datetime:  # fromisoformat takes "Z" only on 3.11+
            return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))

        before = parsed(late.timestamp) < parsed(in_window.timestamp)
        assert before == (offset_days < 0)

        validated_save_continuity(store, _text(1), wrap_token=prep["wrap_token"], today=TODAY)

        assert late.id in {e.id for e in store.episodes_since_wrap()}
        prepare_wrap(store)
        assert late.id in set(store.load_wrap_snapshot()["episode_ids"])
    finally:
        store.close()
