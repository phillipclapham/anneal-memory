"""0.9.31: reads that must describe one committed state, and two routed LOWs.

Each test is built from a failure reproduced on 0.9.30 (2026-10-04, 1004+2):
a concurrent writer made ``recall``'s ``total_matching`` disagree with its rows on
65 of 65 reads; ``WrapWindowMovedError`` failed to unpickle; a durable warning line
reached 150,138 characters; ``wrap-status`` could pair one wrap's token with a
replacement wrap's gated session.
"""

from __future__ import annotations

import dataclasses
import pickle
import sqlite3
import uuid

from anneal_memory import Store, prepare_wrap
from anneal_memory.durable import DurableReport, report_warnings
from anneal_memory.store import WrapWindowMovedError
from anneal_memory.types import EpisodeType


class _CommitBetween:
    """A connection proxy that, just before the first statement starting with
    ``trigger``, has ``action`` commit through ANOTHER connection: a concurrent
    writer landing between two reads, deterministically."""

    def __init__(self, conn, trigger, action):
        self._conn, self._trigger, self._action, self._fired = conn, trigger, action, False

    def execute(self, sql, *a):
        if not self._fired and sql.lstrip().startswith(self._trigger):
            self._fired = True
            self._action()
        return self._conn.execute(sql, *a)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _seed(path, n):
    with Store(str(path), project_name="Snap") as s:
        for i in range(n):
            s.record(f"zqx snapshot episode {i}", EpisodeType.OBSERVATION)


def test_counts_and_rows_are_read_in_one_snapshot(tmp_path, monkeypatch):
    db = tmp_path / "m.db"
    _seed(db, 5)

    def peer_insert():
        with Store(str(db)) as peer:
            peer.record("zqx inserted by a concurrent writer", EpisodeType.OBSERVATION)

    with Store(str(db)) as s:
        real = s._conn
        s._conn = _CommitBetween(real, "SELECT * FROM episodes", peer_insert)
        r = s.recall(keyword="zqx", limit=100)
        assert s._conn._fired and r.total_matching == len(r.episodes)

        s._conn = _CommitBetween(real, "SELECT id,", peer_insert)
        eps, doc_freq, corpus_n = s.keyword_candidates(["zqx"], limit_per_keyword=100)
        assert s._conn._fired and doc_freq["zqx"] == len(eps) <= corpus_n
        s._conn = real
        # Any number of keywords: one OR chain of 1,100 LIKEs failed in review.
        many = [f"zqk{i}" for i in range(1100)] + ["zqx"]
        one = s.keyword_candidates(["zqx"], limit_per_keyword=5)
        assert s.keyword_candidates(many, limit_per_keyword=5)[1]["zqx"] == one[1]["zqx"]
        s.record("ZQX upper-case writer episode", EpisodeType.OBSERVATION)

    # Matching ignores ASCII case even on a SQLite whose LIKE is case-sensitive
    # (SQLITE_CASE_SENSITIVE_LIKE builds): the match lowers the content (codex L3, run).
    real_connect = sqlite3.connect

    def case_sensitive_connect(*a, **k):
        conn = real_connect(*a, **k)
        conn.execute("PRAGMA case_sensitive_like=ON")
        return conn

    monkeypatch.setattr(sqlite3, "connect", case_sensitive_connect)
    with Store(str(db), read_only=True) as reader:
        assert reader.recall(keyword="zqx").total_matching == one[1]["zqx"] + 1


def test_wrap_window_moved_error_pickles():
    e = WrapWindowMovedError(3, 4)
    again = pickle.loads(pickle.dumps(e))
    assert (again.expected, again.actual, str(again)) == (3, 4, str(e))


def test_durable_warning_lines_are_bounded():
    big = "- " + "x" * 50_000
    empty = {f.name: ([] if f.type.startswith("list") else 0)
             for f in dataclasses.fields(DurableReport)}
    report = DurableReport(**{
        **empty, "heading": "Durable Facts", "recreated": False,
        "pair_scan_stopped": False, "dropped": [big], "reinserted": [big],
        "own_lines_dropped": [big], "unknown_drops": [(big, big)],
        "near_duplicates": [(big, big)], "multi_drops": [(big, [big, big])],
        "contradictions": [(big, big)], "untracked": [big], "stray_markers": [big],
        "cue_sprawl": [big], "pattern_shaped": [big],
        "shared_facts": [(big, [big, big])], "near_miss_headers": [big],
    })
    warnings = report_warnings(report)
    assert warnings and max(len(w) for w in warnings) < 5_000
    # A suggested marker is pasted back by the writer, so it is never a clipped one
    # (L1 0.9.31, run: a clipped marker named no line and the fact came back).
    for w in warnings:
        for part in w.split("[drop-durable: ")[1:]:
            assert "… (+" not in part.split("]", 1)[0], w[:200]


def test_wrap_status_snapshot_does_not_mix_two_wraps(tmp_path, monkeypatch):
    db = tmp_path / "m.db"
    _seed(db, 2)
    with Store(str(db)) as s:
        a = prepare_wrap(s)["wrap_token"]

        def replace_with_a_gated_wrap():
            with Store(str(db)) as peer:
                peer.wrap_cancelled(expect_token=a)
                peer.wrap_started(token=uuid.uuid4().hex, episode_ids=[],
                                  gated_session_id="peer-session")

        real = Store.wrap_gated_session
        fired = []

        def gated_after_peer(self):
            if not fired:
                fired.append(1)
                replace_with_a_gated_wrap()
            return real(self)

        monkeypatch.setattr(Store, "wrap_gated_session", gated_after_peer)
        status = s.wrap_status_snapshot()
        assert status.snapshot["token"] == a
        assert status.gated_session is None  # wrap A was ungated
