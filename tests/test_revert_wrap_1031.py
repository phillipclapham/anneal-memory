"""Store.revert_wrap: a rolled-back wrap's episodes are offered again.

Built from the residue run in project_memory/seat_1031_37/revert_run.py (a real
prepare -> save wrap, a revert, a second prepare and save, a recorder process
throughout). Before revert_wrap the second prepare offered 2 of the 17 episodes
the rolled-back wrap had consumed; after it, all of them.
"""

from __future__ import annotations

import datetime
import json
import threading
import time

import pytest

from anneal_memory import Store, WrapRevertRefused
from anneal_memory.audit import AuditTrail
from anneal_memory.continuity import prepare_wrap, validated_save_continuity

TODAY = datetime.date.today().isoformat()


def _text(n: int) -> str:
    return (
        f"# T — Memory (v1)\n\n## State\nWorking {n}.\n\n"
        f"## Patterns\nthought: p{n} | 1x ({TODAY})\n\n"
        f"## Decisions\nNone.\n\n## Context\nPass {n}.\n"
    )


def _wrap(store: Store, n: int) -> list[str]:
    """One real prepare -> save; returns the episode ids the wrap consumed."""
    prep = prepare_wrap(store)
    ids = list(store.load_wrap_snapshot()["episode_ids"])
    validated_save_continuity(store, _text(n), wrap_token=prep["wrap_token"], today=TODAY)
    return ids


def _seeded(tmp_path, per_day: int = 6) -> tuple[Store, list[str], list[str]]:
    store = Store(tmp_path / "m.db", project_name="t")
    for i in range(per_day):
        store.record(f"day one {i}", "observation")
    day1 = _wrap(store, 1)
    for i in range(per_day):
        store.record(f"day two {i}", "observation")
    day2 = _wrap(store, 2)
    return store, day1, day2


def test_the_reverted_days_episodes_are_offered_again_and_a_new_wrap_takes_them(tmp_path):
    store, _day1, day2 = _seeded(tmp_path)
    try:
        assert not store.episodes_since_wrap()
        receipt = store.revert_wrap(2, reason="dream rollback", actor="test")
        assert receipt.episodes_reoffered == len(day2) == 6
        assert not receipt.already_reverted and receipt.last_wrap_id_after == 1
        assert {e.id for e in store.episodes_since_wrap()} == set(day2)

        prep = prepare_wrap(store)
        assert set(store.load_wrap_snapshot()["episode_ids"]) == set(day2)
        validated_save_continuity(store, _text(2), wrap_token=prep["wrap_token"], today=TODAY)
        # The reverted id is never handed to a later wrap, and the retry of the old revert is told so.
        assert [w.id for w in store.get_wrap_history()] == [1, 3]
        again = store.revert_wrap(2, reason="retry", actor="test")
        assert again.already_reverted and again.episodes_reoffered == 0
        assert again.revert_id == receipt.revert_id
        assert [w.id for w in store.get_wrap_history()] == [1, 3]
        assert not store.episodes_since_wrap()
    finally:
        store.close()


def test_a_revert_that_is_not_of_the_newest_wrap_or_is_unsafe_changes_nothing(tmp_path):
    store, _day1, _day2 = _seeded(tmp_path)
    try:
        before = (store.get_wrap_history(), store.load_continuity())
        for wrap_id, code in ((1, "not_latest"), (9, "unknown_wrap")):
            with pytest.raises(WrapRevertRefused) as exc:
                store.revert_wrap(wrap_id, reason="r")
            assert exc.value.code == code
        store.record("pending", "observation")
        prepare_wrap(store)  # a wrap is now open
        with pytest.raises(WrapRevertRefused) as exc:
            store.revert_wrap(2, reason="r")
        assert exc.value.code == "wrap_open"
        store.wrap_cancelled(expect_token=store.load_wrap_snapshot()["token"])
        assert (store.get_wrap_history(), store.load_continuity()) == before
        with pytest.raises(WrapRevertRefused) as exc:  # the identity moved after the wrap
            store.save_continuity(_text(7))
            store.revert_wrap(2, reason="r")
        assert exc.value.code == "identity_moved"
        assert store._conn.execute("SELECT COUNT(*) FROM wrap_reverts").fetchone()[0] == 0
    finally:
        store.close()


def test_the_revert_is_archived_audited_and_restores_the_prior_identity(tmp_path):
    store, _day1, _day2 = _seeded(tmp_path)
    try:
        receipt = store.revert_wrap(2, reason="dream rollback occ-7", actor="levain-dream")
        row = store._conn.execute("SELECT * FROM wrap_reverts WHERE wrap_id = 2").fetchone()
        assert (row["reason"], row["actor"], row["episodes_reoffered"]) == (
            "dream rollback occ-7", "levain-dream", 6)
        assert json.loads(row["wrap_row_json"])["id"] == 2
        assert store.load_continuity() == _text(1)  # the files are projected from the restored version
        assert receipt.identity_version == row["identity_version_restored"]
        events = [json.loads(line) for line in store._audit._active_path.read_text().splitlines()
                  if line.strip()]
        reverted = [e for e in events if e["event"] == "wrap_reverted"]
        assert len(reverted) == 1
        assert reverted[0]["data"]["reason"] == "dream rollback occ-7"
        assert reverted[0]["data"]["actor"] == "levain-dream"
        assert reverted[0]["data"]["wrap_id"] == 2
        assert AuditTrail.verify(store.path).valid
    finally:
        store.close()


def _wait(done) -> bool:
    deadline = time.monotonic() + 20
    while not done() and time.monotonic() < deadline:
        time.sleep(0.005)
    return bool(done())


def test_an_episode_recorded_on_another_connection_during_the_revert_is_not_lost(tmp_path):
    store, _day1, day2 = _seeded(tmp_path)
    db = store.path
    stop = threading.Event()
    recorded: list[str] = []
    errors: list[BaseException] = []

    def recorder() -> None:
        try:
            other = Store(db, project_name="t")
            try:
                while not stop.is_set():
                    recorded.append(other.record(f"live {len(recorded)}", "observation").id)
            finally:
                other.close()
        except BaseException as exc:  # surfaced by the assert below
            errors.append(exc)

    thread = threading.Thread(target=recorder)
    thread.start()
    try:
        assert _wait(lambda: len(recorded) >= 5 or errors), "the recorder made no progress"
        store.revert_wrap(2, reason="r")
        assert _wait(lambda: len(recorded) >= 12 or errors), "the recorder stopped after the revert"
    finally:
        stop.set()
        thread.join(timeout=30)
    try:
        assert not errors, errors
        window = {e.id for e in store.episodes_since_wrap()}
        assert set(day2) <= window
        assert set(recorded) <= window, "an episode recorded around the revert fell outside every window"
        assert len(set(recorded)) == len(recorded)
    finally:
        store.close()


def _graduating(store: Store, ep_id: str, quote: str) -> str:
    return (
        "## State\nw.\n\n## Patterns\n"
        f'- alpha_proven | 2x ({TODAY}) [evidence: {ep_id} "{quote}"]\n\n'
        "## Decisions\n- d.\n\n## Context\n- c.\n"
    )


def _graduation_store(tmp_path) -> Store:
    store = Store(tmp_path / "m.db", project_name="t")
    for i in range(4):
        store.record(
            f"baseline episode {i}: substrate observation about discipline rotation "
            f"memory tracking topic {i}.", "observation")
    prep = prepare_wrap(store)
    validated_save_continuity(
        store, f"## State\nw1.\n\n## Patterns\n- alpha_proven | 1x ({TODAY})\n\n"
        "## Decisions\n- d.\n\n## Context\n- c.\n",
        wrap_token=prep["wrap_token"], today=TODAY)
    for i in range(4):
        store.record(
            f"baseline episode d{i}: substrate observation about discipline rotation "
            f"memory tracking topic d{i}.", "observation")
    return store


def test_the_pattern_record_is_put_back_so_the_same_episodes_graduate_on_reworded_evidence(tmp_path):
    """Reviewers L1 and L2 reproduced: a revert that left pattern_levels and the
    pattern_history corpus as the wrap wrote them demoted the redo (cross-session-overlap)."""
    store = _graduation_store(tmp_path)
    try:
        prep = prepare_wrap(store)
        ep = store.load_wrap_snapshot()["episode_ids"][0]
        validated_save_continuity(
            store, _graduating(store, ep, "baseline substrate observation about discipline rotation memory tracking"),
            wrap_token=prep["wrap_token"], today=TODAY)
        assert store.get_pattern_history("alpha_proven") is not None
        assert store.saved_pattern_levels()[("name", "alpha_proven")] == 2

        receipt = store.revert_wrap(2, reason="r")
        assert store.get_pattern_history("alpha_proven") is None
        assert store.saved_pattern_levels()[("name", "alpha_proven")] == 1
        undone = json.loads(store._conn.execute(
            "SELECT undone_json FROM wrap_reverts WHERE id = ?", (receipt.revert_id,)).fetchone()[0])
        assert undone["pattern_history_exact"] == 1 and undone["pattern_levels_exact"] >= 1

        prep = prepare_wrap(store)
        ep = store.load_wrap_snapshot()["episode_ids"][0]
        result = validated_save_continuity(
            store, _graduating(store, ep, "episode substrate observation discipline rotation memory tracking topic"),
            wrap_token=prep["wrap_token"], today=TODAY)
        assert result["graduations_validated"] == 1 and result["graduations_demoted"] == 0
        assert f"alpha_proven | 2x ({TODAY})" in store.load_continuity()
    finally:
        store.close()


def test_reverting_two_wraps_in_a_row_and_a_save_retry_after_a_revert(tmp_path):
    store, _day1, _day2 = _seeded(tmp_path)
    try:
        store.record("day three", "observation")
        prep = prepare_wrap(store)
        validated_save_continuity(
            store, _text(3), wrap_token=prep["wrap_token"], today=TODAY, idempotency_key="save-3")
        store.revert_wrap(3, reason="r")
        store.revert_wrap(2, reason="r")  # was identity_moved: the latest version carries wrap 2's own text
        assert store.load_continuity() == _text(1)
        assert [w.id for w in store.get_wrap_history()] == [1]
        prep = prepare_wrap(store)
        retried = validated_save_continuity(
            store, _text(3), wrap_token=prep["wrap_token"], today=TODAY, idempotency_key="save-3")
        assert not retried.get("replayed"), "a retry after the revert was told the reverted wrap succeeded"
    finally:
        store.close()


def test_the_cli_reverts_refuses_with_exit_3_and_retries_cleanly(tmp_path, capsys, monkeypatch):
    from anneal_memory import cli

    store, _day1, day2 = _seeded(tmp_path)
    db = str(store.path)
    store.close()

    def run(*argv: str) -> tuple[int, str]:
        try:
            monkeypatch.setattr("sys.argv", ["anneal-memory", "--db", db, "revert-wrap", *argv, "--json"])
            cli.main()
            code = 0
        except SystemExit as exc:
            code = int(exc.code or 0)
        return code, capsys.readouterr().out

    code, out = run("1", "--reason", "r")
    assert code == 3 and json.loads(out)["refused"] == "not_latest"
    code, out = run("2", "--reason", "dream rollback", "--actor", "levain-dream")
    body = json.loads(out)
    assert code == 0 and body["status"] == "reverted" and body["episodes_reoffered"] == len(day2)
    code, out = run("2", "--reason", "retry")
    assert code == 0 and json.loads(out)["status"] == "already_reverted"


def test_a_later_meta_save_with_the_same_text_and_pruned_episodes_both_refuse(tmp_path):
    """L3 r1 (codex HIGH, complement MED): a save_meta after the wrap carries the wrap's
    text, so a text-hash gate let the revert discard the new meta; and a wrap whose episodes
    were pruned since would be reverted although its summary is all that is left of them."""
    store, _day1, day2 = _seeded(tmp_path)
    try:
        store.save_meta({**store.load_meta(), "operator_note": "set after the wrap"})
        with pytest.raises(WrapRevertRefused) as exc:
            store.revert_wrap(2, reason="r")
        assert exc.value.code == "identity_moved"
    finally:
        store.close()
    (tmp_path / "two").mkdir()
    store, _day1, day2 = _seeded(tmp_path / "two")
    try:
        assert store.delete(day2[0])
        with pytest.raises(WrapRevertRefused) as exc:
            store.revert_wrap(2, reason="r")
        assert exc.value.code == "episodes_gone"
        assert [w.id for w in store.get_wrap_history()] == [1, 2]
        receipt = store.revert_wrap(2, reason="r", allow_missing=True)
        assert receipt.episodes_reoffered == len(day2) - 1
    finally:
        store.close()


def test_a_supersession_link_the_wrap_recorded_goes_with_it(tmp_path):
    """L3 r1 (codex MED): the wrap's [supersedes: A by B] decision outlived its rollback."""
    store = Store(tmp_path / "m.db", project_name="t")
    try:
        store.record("seed", "observation")
        _wrap(store, 1)
        old = store.record("the old note", "observation").id
        new = store.record("the newer note", "observation").id
        prep = prepare_wrap(store)
        validated_save_continuity(
            store, _text(2) + f"\n[supersedes: {old} by {new}]\n",
            wrap_token=prep["wrap_token"], today=TODAY)
        assert store.supersession_exists(old_id=old, new_id=new)
        store.revert_wrap(2, reason="r")
        assert not store.supersession_exists(old_id=old, new_id=new)
        assert {e.id for e in store.episodes_since_wrap()} >= {old, new}
    finally:
        store.close()


def test_a_revert_that_loses_the_race_still_repairs_the_projection(tmp_path):
    """L3 r3 (codex MED): the in-lock already-reverted return skipped the projection repair."""
    store, _day1, _day2 = _seeded(tmp_path)
    try:
        store.revert_wrap(2, reason="r")
        store.continuity_path.write_text("stale projection a failed write left\n", encoding="utf-8")
        real = store._already_reverted_receipt
        calls = []

        def first_call_sees_nothing(wrap_id):
            calls.append(wrap_id)
            return None if len(calls) == 1 else real(wrap_id)

        store._already_reverted_receipt = first_call_sees_nothing  # the race: the revert commits after the unlocked check
        again = store.revert_wrap(2, reason="retry")
        assert again.already_reverted and len(calls) == 2
        assert store.continuity_path.read_text(encoding="utf-8") == _text(1)
    finally:
        store.close()
