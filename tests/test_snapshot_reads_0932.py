"""0.9.32: the remaining multi-statement reads describe one committed state.

Each test fails on 0.9.31 (2026-10-04, 1004+14). ``status`` was reproduced under a
second writer PROCESS: total episodes != the sum of the by-type counts on 335 of
1,500 reads, and the since-wrap count exceeded the total on 309. The other three
reads run the same shape of statements, so each is pinned by a peer commit landing
between two of them (the harness of ``test_snapshot_reads_0931``).
"""

from __future__ import annotations

from anneal_memory import Store, prepare_wrap, validated_save_continuity
from anneal_memory.types import EpisodeType

from .test_snapshot_reads_0931 import _CommitBetween

_WRAP = """# Snap — Memory (v1)

## State
s

## Patterns
- p_{n} | 1x (2026-10-04)

## Decisions
[decided(rationale: "x", on: "2026-10-04")] y

## Context
c
"""


def _wrap(store, n):
    assert prepare_wrap(store)["status"] == "ready"
    validated_save_continuity(store, _WRAP.format(n=n))


def test_status_counts_describe_one_state(tmp_path):
    db = tmp_path / "m.db"
    with Store(str(db), project_name="Snap") as s:
        for i in range(4):
            s.record(f"zqx seed {i}", EpisodeType.OBSERVATION)

    def peer():
        with Store(str(db)) as p:
            p.record("zqx peer", EpisodeType.DECISION)

    with Store(str(db), project_name="Snap") as s:
        real = s._conn
        s._conn = _CommitBetween(real, "SELECT type,", peer)
        st = s.status()
        assert s._conn._fired
        assert st.total_episodes == sum(st.episodes_by_type.values())


def test_association_stats_describe_one_state(tmp_path):
    db = tmp_path / "m.db"
    with Store(str(db), project_name="Snap") as s:
        ids = [getattr(s.record(f"zqx e{i}", EpisodeType.OBSERVATION), "id", None)
               for i in range(4)]
        s.record_associations({(ids[0], ids[1])})

    def peer():
        with Store(str(db)) as p:
            p.record_associations({(ids[2], ids[3])})

    with Store(str(db), project_name="Snap") as s:
        real = s._conn
        s._conn = _CommitBetween(real, "SELECT episode_a,", peer)
        a = s.association_stats()
        assert s._conn._fired
        assert len(a.strongest_pairs) == min(5, a.total_links)


def test_compression_window_is_one_session(tmp_path):
    db = tmp_path / "m.db"
    with Store(str(db), project_name="Snap") as s:
        s.record("zqx first", EpisodeType.OBSERVATION)
        _wrap(s, 1)
        s.record("zqx second", EpisodeType.OBSERVATION)

    def peer():
        with Store(str(db), project_name="Snap") as p:
            _wrap(p, 2)
            p.record("zqx third", EpisodeType.OBSERVATION)

    with Store(str(db), project_name="Snap") as s:
        real = s._conn
        s._conn = _CommitBetween(real, "SELECT * FROM episodes", peer)
        eps = s.episodes_since_wrap()
        assert s._conn._fired
        assert len({e.session_id for e in eps}) == 1
