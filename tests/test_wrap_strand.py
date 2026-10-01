"""spore-1233: an episode recorded while a wrap is open must reach a later window.

Both tests were built from failures reproduced on 0.9.17.dev0 before the fix:
the sequential case stranded the late episode on the second wrap, and a real
two-process run (recorder vs prepare/save loop) stranded 2,517 of 3,459.
"""

from __future__ import annotations

import datetime
import threading

from anneal_memory import Store
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
            saver.join(timeout=0.5)  # fixed: blocks on record's write lock
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
