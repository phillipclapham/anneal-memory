"""The audit outbox's claim/ack state machine (store half).

Design: project_memory/outbox_claim_ack_design_1028.md, sections 3 and 4.
T2 (ids never reused) and the store half of T3 (a poison row is bounded, never
silent) are built from the reproductions the design records: an emptied table
reused id 1 (run 10-28), and a row that is never delivered was retried and
counted on every flush with no terminal state.
"""

from __future__ import annotations

import json

import pytest

from anneal_memory import store as store_mod
from anneal_memory.store import Store


def _event(name: str) -> str:
    return json.dumps({"event": name, "payload": {}, "kwargs": {},
                       "method": "t", "committed": "t"})


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "ob.db")
    yield s
    s.close()


def _insert(store: Store, *names: str) -> list[int]:
    ids = []
    for name in names:
        cur = store._conn.execute(
            "INSERT INTO audit_outbox (event_json, created_at) VALUES (?, 'x')",
            (_event(name),))
        ids.append(cur.lastrowid)
    store._conn.commit()
    return ids


def _row(store: Store, row_id: int):
    return store._conn.execute(
        "SELECT state, attempts, claim_token, claim_tail_seq, not_before, "
        "last_error, dead_at, gap_logged FROM audit_outbox WHERE id = ?",
        (row_id,)).fetchone()


def _expire(store: Store, row_id: int) -> None:
    """Make a held lease or a backoff already over."""
    store._conn.execute(
        "UPDATE audit_outbox SET claim_expires_at = '2000-01-01T00:00:00.000000Z', "
        "not_before = NULL WHERE id = ?", (row_id,))
    store._conn.commit()


class TestOutboxIds:
    def test_ids_strictly_increase_across_an_emptied_table(self, store):
        """T2 (design R2): insert, delete, insert returned 1 then 1 before."""
        (first,) = _insert(store, "a")
        store._conn.execute("DELETE FROM audit_outbox")
        store._conn.commit()
        (second,) = _insert(store, "b")
        assert second > first


class TestOutboxClaimAck:
    def test_claim_counts_the_attempt_and_ack_deletes_by_token(self, store):
        ids = _insert(store, "a", "b")
        claims = store._outbox_claim(tail_seq=7)
        assert [c.id for c in claims] == ids
        assert {c.kind for c in claims} == {"event"}
        assert [c.attempts for c in claims] == [1, 1]
        assert {_row(store, i)[0] for i in ids} == {"claimed"}
        assert all(c.tail_seq == 7 for c in claims)
        assert store._outbox_ack(claims[0]) is True
        assert [r[0] for r in store._conn.execute(
            "SELECT id FROM audit_outbox")] == [ids[1]]

    def test_a_second_claim_finds_nothing_while_the_lease_is_live(self, store):
        _insert(store, "a")
        assert len(store._outbox_claim()) == 1
        assert store._outbox_claim() == []

    def test_an_expired_lease_is_reclaimed_and_the_attempt_counted(self, store):
        (row_id,) = _insert(store, "a")
        (first,) = store._outbox_claim(tail_seq=3)
        _expire(store, row_id)
        (second,) = store._outbox_claim(tail_seq=99)
        assert second.token != first.token
        assert second.attempts == 2
        # the scan window of the FIRST claim covers every attempt
        assert second.tail_seq == 3
        assert _row(store, row_id)[3] == 3

    def test_a_stale_holder_cannot_ack_or_nack_after_a_reclaim(self, store):
        (row_id,) = _insert(store, "a")
        (stale,) = store._outbox_claim()
        _expire(store, row_id)
        (fresh,) = store._outbox_claim()
        assert store._outbox_ack(stale) is False
        assert store._outbox_nack(stale, RuntimeError("late")) is False
        assert _row(store, row_id)[0] == "claimed"
        assert store._outbox_ack(fresh) is True
        assert store._outbox_ack(fresh) is False  # row gone: a no-op, not an error

    def test_nack_releases_with_a_backoff_and_the_head_blocks(self, store):
        first, second = _insert(store, "a", "b")
        claims = store._outbox_claim()
        assert store._outbox_nack(claims[0], RuntimeError("sink down")) is True
        # the row 1 backoff is live: row 2 is NOT handed out ahead of it
        assert store._outbox_claim() == []
        # row 2 stays claimed under its own live lease; the head is pending
        state, attempts, token, _tail, not_before, error, _dead, _gap = _row(
            store, first)
        assert (state, attempts, token) == ("pending", 1, None)
        assert not_before is not None and "sink down" in error
        _expire(store, first)
        (again,) = store._outbox_claim()
        assert again.id == first and again.attempts == 2

    def test_a_claim_is_capped_at_the_batch_limit(self, store):
        _insert(store, *[f"e{i}" for i in range(5)])
        assert len(store._outbox_claim(limit=2)) == 2

    def test_a_claim_inside_a_callers_transaction_takes_nothing(self, store):
        _insert(store, "a")
        store._conn.execute("BEGIN IMMEDIATE")
        try:
            assert store._outbox_claim() == []
        finally:
            store._conn.rollback()
        assert _row(store, 1)[0] == "pending"


class TestOutboxPoisonBound:
    def test_a_row_never_acked_goes_dead_after_max_claims_and_the_queue_moves(
            self, store):
        """T3, store half (design R3): a row that never delivers was retried
        and counted forever; now it is dead after MAX attempts, retained, and
        the row behind it is delivered."""
        poison, later = _insert(store, "poison", "later")
        for attempt in range(1, store_mod._OUTBOX_MAX_ATTEMPTS + 1):
            claims = store._outbox_claim()
            assert [(c.id, c.kind, c.attempts) for c in claims][0] == (
                poison, "event", attempt)
            # the flusher failed on the head: the untried row behind it is
            # released with its attempt uncounted, the head's lease lapses
            assert store._outbox_release(claims[1]) is True
            _expire(store, poison)
        claims = store._outbox_claim()
        gap = claims[0]
        assert (gap.id, gap.kind) == (poison, "gap")
        assert "attempts exhausted" in gap.reason
        state, attempts, _t, _tail, _nb, _err, dead_at, gap_logged = _row(
            store, poison)
        assert (state, attempts, gap_logged) == ("dead", 5, 0)
        assert dead_at is not None
        # the later row is claimed in the same pass, after the gap
        assert [c.id for c in claims] == [poison, later]
        assert store._outbox_ack(gap) is True
        assert _row(store, poison)[0] == "dead"       # retained
        assert _row(store, poison)[7] == 1            # gap marker logged
        assert claims[1].attempts == 1  # never aged by the poison head's claims
        assert store._outbox_ack(claims[1]) is True
        # a dead, marked row is never claimed again
        assert store._outbox_claim() == []

    def test_a_dead_row_without_its_marker_is_the_head_until_it_is_logged(
            self, store):
        dead, later = _insert(store, "dead", "later")
        (claim,) = store._outbox_claim(limit=1)
        marked = store._outbox_mark_dead(claim, "event_json does not parse")
        assert marked is not None and marked.kind == "gap"
        # the marker was never acked: it comes back first, ahead of the later row
        _expire(store, dead)
        claims = store._outbox_claim()
        assert [(c.id, c.kind) for c in claims] == [(dead, "gap"), (later, "event")]

    def test_mark_dead_by_a_lost_claim_is_a_noop(self, store):
        (row_id,) = _insert(store, "a")
        (stale,) = store._outbox_claim()
        _expire(store, row_id)
        store._outbox_claim()
        assert store._outbox_mark_dead(stale, "late") is None
        assert _row(store, row_id)[0] == "claimed"


# --------------------------------------------------------------------------
# S2: AuditTrail.append_once (design sec. 5, crash cases C3 and C5)
# --------------------------------------------------------------------------

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from anneal_memory.audit import AuditTrail

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)


def _entries(db_path) -> list[dict]:
    path = db_path.parent / f"{db_path.stem}.audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _with_ref(db_path, ref: str) -> list[dict]:
    return [e for e in _entries(db_path)
            if (e.get("data") or {}).get("outbox_ref") == ref]


class TestAppendOnce:
    def test_a_killed_process_between_append_and_ack_logs_the_event_once(
            self, tmp_path, monkeypatch):
        """T1 (design R1, C3): a subprocess is killed with os._exit after its
        append and before its ack. Before append_once the reclaim appended the
        event a second time (the log held 2 entries for it). Written at S2 on
        the claim/append_once primitives with the old open-time flush stubbed
        out in both processes; S3 reruns it through the real flush."""
        db = tmp_path / "t1.db"
        script = tmp_path / "killed_after_append.py"
        script.write_text(textwrap.dedent('''
            import os, sys
            from anneal_memory.store import Store
            Store._flush_audit_outbox = lambda self, **kw: None
            store = Store(sys.argv[1])
            store._conn.execute(
                "INSERT INTO audit_outbox (event_json, created_at) VALUES (?, 'x')",
                ('{"event": "crash_event"}',))
            store._conn.commit()
            (claim,) = store._outbox_claim(tail_seq=store._audit._tail_seq())
            store._audit._append_once("crash_event", {}, "agent",
                                     ref=f"{claim.id}:abc", since_seq=claim.tail_seq)
            os._exit(0)   # killed between the append and the ack
        '''))
        # the tree under test, not an installed copy of another generation
        env = {**os.environ, "PYTHONPATH": _REPO_ROOT}
        subprocess.run([sys.executable, str(script), str(db)], check=True, env=env)
        row_id = 1
        assert len(_with_ref(db, f"{row_id}:abc")) == 1

        monkeypatch.setattr(Store, "_flush_audit_outbox", lambda self, **kw: None)
        s = Store(db)
        try:
            _expire(s, row_id)
            (claim,) = s._outbox_claim(tail_seq=s._audit._tail_seq())
            assert claim.attempts == 2
            entry, appended = s._audit._append_once(
                "crash_event", {}, "agent", ref=f"{row_id}:abc",
                since_seq=claim.tail_seq)
            assert appended is False and entry["event"] == "crash_event"
            assert s._outbox_ack(claim) is True
        finally:
            s.close()
        assert len(_with_ref(db, f"{row_id}:abc")) == 1
        assert _entries(db)[-1]["event"] == "crash_event"

    def test_a_stalled_second_claimant_logs_the_event_once(self, store, tmp_path):
        """T5 (C5): P stalls past its lease, Q reclaims, appends and acks; P
        resumes, appends: the log still holds one entry, and P's ack is a no-op."""
        (row_id,) = _insert(store, "stall_event")
        (p,) = store._outbox_claim(tail_seq=store._audit._tail_seq())
        _expire(store, row_id)
        (q,) = store._outbox_claim(tail_seq=store._audit._tail_seq())
        ref = f"{row_id}:h"
        e_q, new_q = store._audit._append_once(
            "stall_event", {}, "agent", ref=ref, since_seq=q.tail_seq)
        assert store._outbox_ack(q) is True
        e_p, new_p = store._audit._append_once(
            "stall_event", {}, "agent", ref=ref, since_seq=p.tail_seq)
        assert (new_q, new_p) == (True, False)
        assert e_p["seq"] == e_q["seq"]
        assert store._outbox_ack(p) is False
        assert len(_with_ref(tmp_path / "ob.db", ref)) == 1

    def test_the_scan_stops_at_since_seq_and_other_refs_do_not_match(self, store):
        trail = store._audit
        first, _ = trail._append_once("e", {}, "agent", ref="1:a", since_seq=None)
        trail.log("filler", {})
        window = trail._tail_seq()
        # the first copy lies before the window: not seen, so it appends again
        again, appended = trail._append_once("e", {}, "agent", ref="1:a",
                                            since_seq=window)
        assert appended is True and again["seq"] > first["seq"]
        # inside the window the same ref is found
        same, appended = trail._append_once("e", {}, "agent", ref="1:a",
                                           since_seq=window)
        assert appended is False and same["seq"] == again["seq"]
        # a different hash on the same id is a different event
        other, appended = trail._append_once("e", {}, "agent", ref="1:b",
                                            since_seq=window)
        assert appended is True and other["seq"] > again["seq"]

    def test_the_chain_stays_valid_and_a_skipped_duplicate_leaves_it_untouched(
            self, store, tmp_path):
        trail = store._audit
        trail._append_once("e", {"k": 1}, "agent", ref="9:z", since_seq=None)
        before = _entries(tmp_path / "ob.db")
        trail._append_once("e", {"k": 1}, "agent", ref="9:z", since_seq=None)
        assert _entries(tmp_path / "ob.db") == before
        assert AuditTrail.verify(tmp_path / "ob.db").valid

    def test_append_once_does_not_fire_the_callback_and_notify_does(self, tmp_path):
        seen: list[str] = []
        s = Store(tmp_path / "cb.db", on_audit_event=lambda e: seen.append(e["event"]))
        try:
            entry, appended = s._audit._append_once(
                "cb_event", {}, "agent", ref="1:c", since_seq=None)
            assert appended is True and "cb_event" not in seen
            s._audit._notify(entry)
            assert seen.count("cb_event") == 1
        finally:
            s.close()

    def test_a_torn_final_line_is_skipped_by_the_scan(self, store, tmp_path):
        trail = store._audit
        trail._append_once("e", {}, "agent", ref="2:t", since_seq=None)
        active = tmp_path / "ob.audit.jsonl"
        with open(active, "ab") as f:
            f.write(b'{"v":1,"seq":')   # a kill mid-write
        entry, appended = trail._append_once("e", {}, "agent", ref="2:t",
                                            since_seq=None)
        assert appended is False and entry["data"]["outbox_ref"] == "2:t"

    def test_a_line_longer_than_one_read_chunk_is_joined(self, store):
        trail = store._audit
        trail._append_once("big", {"pad": "x" * 200_000}, "agent", ref="3:b",
                          since_seq=None)
        trail.log("tail", {})
        entry, appended = trail._append_once("big", {"pad": "x" * 200_000}, "agent",
                                            ref="3:b", since_seq=None)
        assert appended is False and entry["data"]["outbox_ref"] == "3:b"

    def test_the_data_key_is_owned_by_append_once(self, store):
        with pytest.raises(ValueError):
            store._audit._append_once("e", {"outbox_ref": "mine"}, "agent",
                                     ref="1:a", since_seq=None)
