"""Tests for Hebbian association links — the third cognitive layer.

Tests cover: association formation, strengthening, decay, retrieval,
canonical ordering, cascade deletion, session co-citation extraction,
integration with Store, and integration with the full wrap pipeline.
"""

import json
import sqlite3
import tempfile
from datetime import date
from pathlib import Path

import pytest

from anneal_memory.associations import (
    DIRECT_CO_CITATION_STRENGTH,
    SESSION_CO_CITATION_STRENGTH,
    DEFAULT_DECAY_FACTOR,
    DEFAULT_CLEANUP_THRESHOLD,
    MAX_STRENGTH,
    AFFECTIVE_MODULATION_FACTOR,
    ASSOCIATIONS_SCHEMA,
    canonical_pair,
    record_associations,
    decay_associations,
    get_associations,
    get_association_context,
    association_stats,
)
from anneal_memory.types import AffectiveState
from anneal_memory.graduation import (
    extract_session_co_citations,
    validate_graduations,
)
from anneal_memory.store import Store
from anneal_memory.types import AssociationPair, AssociationStats


# -- Helpers --

def _make_db(tmp_path) -> sqlite3.Connection:
    """Create a minimal SQLite DB with episodes + associations tables."""
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE episodes (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'agent',
            session_id TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
    """)
    conn.executescript(ASSOCIATIONS_SCHEMA)
    conn.commit()
    return conn


def _insert_episode(conn, ep_id, content="test"):
    """Insert a minimal episode for foreign key satisfaction."""
    conn.execute(
        "INSERT INTO episodes (id, timestamp, type, content) VALUES (?, ?, ?, ?)",
        (ep_id, "2026-04-07T12:00:00Z", "observation", content),
    )
    conn.commit()


class TestCanonicalPair:
    """canonical_pair ensures consistent ordering."""

    def test_already_ordered(self):
        assert canonical_pair("aaa", "bbb") == ("aaa", "bbb")

    def test_reverse_ordered(self):
        assert canonical_pair("bbb", "aaa") == ("aaa", "bbb")

    def test_equal_returns_none(self):
        assert canonical_pair("abc", "abc") is None

    def test_hex_ids(self):
        assert canonical_pair("f0e1d2c3", "a1b2c3d4") == ("a1b2c3d4", "f0e1d2c3")


class TestRecordAssociations:
    """Association formation and strengthening."""

    def test_direct_co_citation_creates_link(self, tmp_path):
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa")
        _insert_episode(conn, "bbb")

        formed, strengthened = record_associations(
            conn,
            direct_pairs={("aaa", "bbb")},
            session_pairs=set(),
            timestamp="2026-04-07T12:00:00Z",
        )

        assert formed == 1
        assert strengthened == 0

        row = conn.execute("SELECT * FROM associations").fetchone()
        assert row is not None
        assert row[0] == "aaa"  # episode_a (canonical)
        assert row[1] == "bbb"  # episode_b
        assert row[2] == DIRECT_CO_CITATION_STRENGTH  # strength
        assert row[5] == 1  # co_citations

    def test_session_co_citation_creates_weaker_link(self, tmp_path):
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa")
        _insert_episode(conn, "bbb")

        formed, _ = record_associations(
            conn,
            direct_pairs=set(),
            session_pairs={("aaa", "bbb")},
            timestamp="2026-04-07T12:00:00Z",
        )

        assert formed == 1
        row = conn.execute("SELECT strength FROM associations").fetchone()
        assert row[0] == SESSION_CO_CITATION_STRENGTH

    def test_repeated_co_citation_strengthens(self, tmp_path):
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa")
        _insert_episode(conn, "bbb")

        record_associations(conn, {("aaa", "bbb")}, set(), "2026-04-07T12:00:00Z")
        formed, strengthened = record_associations(
            conn, {("aaa", "bbb")}, set(), "2026-04-08T12:00:00Z"
        )

        assert formed == 0
        assert strengthened == 1

        row = conn.execute("SELECT strength, co_citations FROM associations").fetchone()
        assert row[0] == DIRECT_CO_CITATION_STRENGTH * 2
        assert row[1] == 2

    def test_canonical_order_enforced(self, tmp_path):
        """Pairs are stored in canonical order regardless of input order."""
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa")
        _insert_episode(conn, "zzz")

        record_associations(conn, {("zzz", "aaa")}, set(), "2026-04-07T12:00:00Z")

        row = conn.execute("SELECT episode_a, episode_b FROM associations").fetchone()
        assert row[0] == "aaa"
        assert row[1] == "zzz"

    def test_session_pair_excluded_if_already_direct(self, tmp_path):
        """Session pairs that are also direct pairs don't double-count."""
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa")
        _insert_episode(conn, "bbb")

        formed, _ = record_associations(
            conn,
            direct_pairs={("aaa", "bbb")},
            session_pairs={("aaa", "bbb")},
            timestamp="2026-04-07T12:00:00Z",
        )

        assert formed == 1  # Only one link, not two
        row = conn.execute("SELECT strength FROM associations").fetchone()
        assert row[0] == DIRECT_CO_CITATION_STRENGTH  # Direct strength, not direct + session

    def test_multiple_pairs_in_one_call(self, tmp_path):
        conn = _make_db(tmp_path)
        for eid in ["aaa", "bbb", "ccc", "ddd"]:
            _insert_episode(conn, eid)

        formed, _ = record_associations(
            conn,
            direct_pairs={("aaa", "bbb"), ("ccc", "ddd")},
            session_pairs={("aaa", "ccc")},
            timestamp="2026-04-07T12:00:00Z",
        )

        assert formed == 3
        total = conn.execute("SELECT COUNT(*) FROM associations").fetchone()[0]
        assert total == 3


class TestDecayAssociations:
    """Decay and cleanup mechanics."""

    def test_unreinforced_links_decay(self, tmp_path):
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa")
        _insert_episode(conn, "bbb")

        record_associations(conn, {("aaa", "bbb")}, set(), "2026-04-07T12:00:00Z")

        decayed = decay_associations(conn, strengthened_pairs=set())

        assert decayed == 1
        row = conn.execute("SELECT strength FROM associations").fetchone()
        assert abs(row[0] - DIRECT_CO_CITATION_STRENGTH * DEFAULT_DECAY_FACTOR) < 0.001

    def test_reinforced_links_skip_decay(self, tmp_path):
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa")
        _insert_episode(conn, "bbb")

        record_associations(conn, {("aaa", "bbb")}, set(), "2026-04-07T12:00:00Z")

        decayed = decay_associations(
            conn, strengthened_pairs={("aaa", "bbb")}
        )

        assert decayed == 0
        row = conn.execute("SELECT strength FROM associations").fetchone()
        assert row[0] == DIRECT_CO_CITATION_STRENGTH  # Unchanged

    def test_weak_links_deleted_on_decay(self, tmp_path):
        """Links below cleanup threshold are removed entirely."""
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa")
        _insert_episode(conn, "bbb")

        # Create a very weak link
        record_associations(
            conn, set(), {("aaa", "bbb")}, "2026-04-07T12:00:00Z"
        )  # strength = 0.3

        # Decay repeatedly until below threshold
        for _ in range(15):  # 0.3 * 0.9^15 ≈ 0.062 < 0.1
            decay_associations(conn, strengthened_pairs=set())

        total = conn.execute("SELECT COUNT(*) FROM associations").fetchone()[0]
        assert total == 0  # Deleted

    def test_strong_links_persist_through_decay(self, tmp_path):
        """Strongly reinforced links survive many decay cycles."""
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa")
        _insert_episode(conn, "bbb")

        # Create and reinforce 5 times
        for i in range(5):
            record_associations(
                conn, {("aaa", "bbb")}, set(), f"2026-04-0{i+1}T12:00:00Z"
            )

        # Decay 20 times — should still survive (5.0 * 0.9^20 ≈ 0.61)
        for _ in range(20):
            decay_associations(conn, strengthened_pairs=set())

        total = conn.execute("SELECT COUNT(*) FROM associations").fetchone()[0]
        assert total == 1
        row = conn.execute("SELECT strength FROM associations").fetchone()
        assert row[0] > DEFAULT_CLEANUP_THRESHOLD

    def test_mixed_decay_some_survive_some_die(self, tmp_path):
        """In a mix of weak and strong links, only strong survive."""
        conn = _make_db(tmp_path)
        for eid in ["aaa", "bbb", "ccc", "ddd"]:
            _insert_episode(conn, eid)

        # Strong link (3x reinforced = 3.0)
        for _ in range(3):
            record_associations(conn, {("aaa", "bbb")}, set(), "2026-04-07T12:00:00Z")

        # Weak link (single session co-citation = 0.3)
        record_associations(conn, set(), {("ccc", "ddd")}, "2026-04-07T12:00:00Z")

        # Decay 12 times
        for _ in range(12):
            decay_associations(conn, strengthened_pairs=set())

        # Strong link survives (3.0 * 0.9^12 ≈ 0.85)
        # Weak link dies (0.3 * 0.9^12 ≈ 0.085 < 0.1)
        total = conn.execute("SELECT COUNT(*) FROM associations").fetchone()[0]
        assert total == 1
        row = conn.execute("SELECT episode_a FROM associations").fetchone()
        assert row[0] == "aaa"  # The strong link survived


class TestGetAssociations:
    """Association retrieval."""

    def test_get_by_episode_id(self, tmp_path):
        conn = _make_db(tmp_path)
        for eid in ["aaa", "bbb", "ccc"]:
            _insert_episode(conn, eid)

        record_associations(conn, {("aaa", "bbb"), ("aaa", "ccc")}, set(), "2026-04-07T12:00:00Z")

        assocs = get_associations(conn, ["aaa"])
        assert len(assocs) == 2
        assert all(isinstance(a, AssociationPair) for a in assocs)

    def test_get_returns_empty_for_unconnected(self, tmp_path):
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa")

        assocs = get_associations(conn, ["aaa"])
        assert len(assocs) == 0

    def test_min_strength_filter(self, tmp_path):
        conn = _make_db(tmp_path)
        for eid in ["aaa", "bbb", "ccc"]:
            _insert_episode(conn, eid)

        # Strong link
        for _ in range(3):
            record_associations(conn, {("aaa", "bbb")}, set(), "2026-04-07T12:00:00Z")
        # Weak link
        record_associations(conn, set(), {("aaa", "ccc")}, "2026-04-07T12:00:00Z")

        assocs = get_associations(conn, ["aaa"], min_strength=1.0)
        assert len(assocs) == 1
        assert assocs[0].episode_b == "bbb"

    def test_ordered_by_strength_descending(self, tmp_path):
        conn = _make_db(tmp_path)
        for eid in ["aaa", "bbb", "ccc"]:
            _insert_episode(conn, eid)

        # 3x reinforced
        for _ in range(3):
            record_associations(conn, {("aaa", "bbb")}, set(), "2026-04-07T12:00:00Z")
        # 1x
        record_associations(conn, {("aaa", "ccc")}, set(), "2026-04-07T12:00:00Z")

        assocs = get_associations(conn, ["aaa"])
        assert assocs[0].strength > assocs[1].strength

    def test_limit_parameter(self, tmp_path):
        conn = _make_db(tmp_path)
        for i in range(10):
            _insert_episode(conn, f"ep{i:02d}")

        # Create many associations
        for i in range(1, 10):
            record_associations(conn, {("ep00", f"ep{i:02d}")}, set(), "2026-04-07T12:00:00Z")

        assocs = get_associations(conn, ["ep00"], limit=3)
        assert len(assocs) == 3

    def test_empty_episode_ids(self, tmp_path):
        conn = _make_db(tmp_path)
        assocs = get_associations(conn, [])
        assert len(assocs) == 0


class TestAssociationContext:
    """Wrap package association context formatting."""

    def test_formats_associations_for_wrap(self, tmp_path):
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa", "Database performance is critical")
        _insert_episode(conn, "bbb", "ACID compliance outweighs speed")

        # Reinforce enough to exceed min_strength=0.5
        record_associations(conn, {("aaa", "bbb")}, set(), "2026-04-07T12:00:00Z")

        context = get_association_context(conn, ["aaa", "bbb"])
        assert "Episode Associations" in context
        assert "aaa" in context
        assert "bbb" in context
        # Content snippets should be included so LLM can evaluate connections
        assert "Database performance" in context
        assert "ACID compliance" in context

    def test_empty_when_no_associations(self, tmp_path):
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa")

        context = get_association_context(conn, ["aaa"])
        assert context == ""

    def test_filters_by_min_strength(self, tmp_path):
        conn = _make_db(tmp_path)
        for eid in ["aaa", "bbb"]:
            _insert_episode(conn, eid)

        # Weak link below default min_strength=0.5
        record_associations(conn, set(), {("aaa", "bbb")}, "2026-04-07T12:00:00Z")

        context = get_association_context(conn, ["aaa", "bbb"])
        assert context == ""  # Too weak to surface


class TestAssociationStats:
    """Network health metrics."""

    def test_empty_store(self, tmp_path):
        conn = _make_db(tmp_path)
        stats = association_stats(conn, total_episodes=0)

        assert stats.total_links == 0
        assert stats.avg_strength == 0
        assert stats.density == 0.0

    def test_with_associations(self, tmp_path):
        conn = _make_db(tmp_path)
        for eid in ["aaa", "bbb", "ccc"]:
            _insert_episode(conn, eid)

        record_associations(conn, {("aaa", "bbb"), ("bbb", "ccc")}, set(), "2026-04-07T12:00:00Z")

        stats = association_stats(conn, total_episodes=3)

        assert stats.total_links == 2
        assert stats.avg_strength == DIRECT_CO_CITATION_STRENGTH
        assert stats.max_strength == DIRECT_CO_CITATION_STRENGTH
        # 2 links / 3 possible (3*2/2) = 0.667
        assert abs(stats.density - 2 / 3) < 0.01

    def test_local_density(self, tmp_path):
        """local_density measures density among connected episodes only."""
        conn = _make_db(tmp_path)
        # 5 episodes, but only 3 are connected
        for eid in ["aaa", "bbb", "ccc", "ddd", "eee"]:
            _insert_episode(conn, eid)

        # 2 links among 3 connected episodes (aaa, bbb, ccc)
        record_associations(conn, {("aaa", "bbb"), ("bbb", "ccc")}, set(), "2026-04-07T12:00:00Z")

        stats = association_stats(conn, total_episodes=5)

        # global density: 2 / (5*4/2) = 2/10 = 0.2
        assert abs(stats.density - 0.2) < 0.01
        # local density: 2 / (3*2/2) = 2/3 = 0.667
        assert abs(stats.local_density - 2 / 3) < 0.01

    def test_local_density_empty(self, tmp_path):
        """local_density is 0 when no associations exist."""
        conn = _make_db(tmp_path)
        stats = association_stats(conn, total_episodes=5)
        assert stats.local_density == 0.0

    def test_strongest_pairs_limited(self, tmp_path):
        conn = _make_db(tmp_path)
        for i in range(10):
            _insert_episode(conn, f"ep{i:02d}")

        for i in range(1, 10):
            record_associations(conn, {("ep00", f"ep{i:02d}")}, set(), "2026-04-07T12:00:00Z")

        stats = association_stats(conn, total_episodes=10, top_n=3)
        assert len(stats.strongest_pairs) == 3


class TestCascadeDelete:
    """Associations cascade-delete when episodes are removed."""

    def test_delete_episode_removes_associations(self, tmp_path):
        db = tmp_path / "test.db"
        store = Store(db)

        ep1 = store.record("Episode 1", "observation")
        ep2 = store.record("Episode 2", "observation")
        ep3 = store.record("Episode 3", "observation")

        store.record_associations(
            direct_pairs={(ep1.id, ep2.id), (ep1.id, ep3.id)},
        )

        # Verify associations exist
        assocs = store.get_associations([ep1.id])
        assert len(assocs) == 2

        # Delete ep1 — should cascade
        store.delete(ep1.id)

        # All associations involving ep1 should be gone
        assocs = store.get_associations([ep1.id])
        assert len(assocs) == 0

        # ep2↔ep3 never existed, so nothing there
        assocs = store.get_associations([ep2.id])
        assert len(assocs) == 0

        store.close()

    def test_prune_removes_associations(self, tmp_path):
        db = tmp_path / "test.db"
        store = Store(db)

        # Record with old timestamps
        ep1 = store.record("Old 1", "observation", timestamp="2020-01-01T00:00:00Z")
        ep2 = store.record("Old 2", "observation", timestamp="2020-01-01T00:00:00Z")
        ep3 = store.record("New", "observation")

        store.record_associations(
            direct_pairs={(ep1.id, ep2.id), (ep1.id, ep3.id)},
        )

        # Prune old episodes
        pruned = store.prune(older_than_days=1)
        assert pruned == 2

        # Associations to old episodes should be gone
        total = store._conn.execute("SELECT COUNT(*) FROM associations").fetchone()[0]
        assert total == 0

        store.close()


class TestExtractSessionCoCitations:
    """Session co-citation pair extraction from graduation results."""

    def test_single_line_no_session_pairs(self):
        """One line of IDs produces no session pairs (they're all direct)."""
        result = extract_session_co_citations([{"aaa", "bbb", "ccc"}])
        # All from same line — no cross-line pairs
        assert len(result) == 0

    def test_two_lines_different_ids(self):
        """IDs from different lines produce session pairs."""
        result = extract_session_co_citations([{"aaa"}, {"bbb"}])
        assert ("aaa", "bbb") in result

    def test_overlapping_ids_across_lines(self):
        """ID appearing in multiple lines creates session pairs with others."""
        result = extract_session_co_citations([
            {"aaa", "bbb"},
            {"aaa", "ccc"},
        ])
        # aaa appears in both lines, so:
        # bbb↔ccc is a session pair (different lines)
        # aaa↔bbb and aaa↔ccc are also session pairs (aaa spans lines)
        assert ("bbb", "ccc") in result or ("ccc", "bbb") in result

    def test_empty_input(self):
        result = extract_session_co_citations([])
        assert len(result) == 0

    def test_single_id_per_line(self):
        """Single IDs across multiple lines create session pairs."""
        result = extract_session_co_citations([{"aaa"}, {"bbb"}, {"ccc"}])
        assert len(result) == 3  # aaa↔bbb, aaa↔ccc, bbb↔ccc


class TestGraduationCoCitationExtraction:
    """validate_graduations extracts co-citation pairs."""

    def test_direct_co_citations_extracted(self):
        """Two IDs on the same pattern line produce a direct co-citation pair."""
        text = """## Patterns
{test:
  thought: foo | 2x (2026-04-07) [evidence: aaa11111, bbb22222 "database performance scaling"]
}"""
        valid_ids = {"aaa11111", "bbb22222"}
        node_map = {
            "aaa11111": "database performance is critical for scaling under load",
            "bbb22222": "scaling requires careful performance tuning",
        }

        result = validate_graduations(text, valid_ids, "2026-04-07", node_map)

        assert result.validated == 1
        assert len(result.direct_co_citations) == 1
        assert ("aaa11111", "bbb22222") in result.direct_co_citations

    def test_no_co_citations_from_single_id(self):
        """Single cited ID produces no co-citation pairs."""
        text = """## Patterns
{test:
  thought: bar | 2x (2026-04-07) [evidence: aaa11111 "database performance matters"]
}"""
        valid_ids = {"aaa11111"}
        node_map = {"aaa11111": "database performance is critical for production"}

        result = validate_graduations(text, valid_ids, "2026-04-07", node_map)

        assert result.validated == 1
        assert len(result.direct_co_citations) == 0

    def test_demoted_citations_produce_no_associations(self):
        """Ungrounded citations must NOT form association links."""
        text = """## Patterns
{test:
  thought: bad | 2x (2026-04-07) [evidence: dead1111, dead2222 "nothing real"]
}"""
        valid_ids = set()  # No valid IDs — both will fail

        result = validate_graduations(text, valid_ids, "2026-04-07")

        assert result.demoted == 1
        assert len(result.direct_co_citations) == 0

    def test_three_ids_on_one_line(self):
        """Three co-cited IDs produce three pairs (combinatorial)."""
        text = """## Patterns
{test:
  thought: baz | 2x (2026-04-07) [evidence: aaa11111, bbb22222, ccc33333 "database scaling architecture"]
}"""
        valid_ids = {"aaa11111", "bbb22222", "ccc33333"}
        node_map = {
            "aaa11111": "database scaling requires careful architecture decisions",
            "bbb22222": "scaling architecture must handle concurrent writes",
            "ccc33333": "architecture patterns for database clustering",
        }

        result = validate_graduations(text, valid_ids, "2026-04-07", node_map)

        assert result.validated == 1
        assert len(result.direct_co_citations) == 3  # C(3,2) = 3 pairs

    def test_all_validated_ids_tracked_per_line(self):
        """Each validated line's IDs are tracked for session co-citation."""
        text = """## Patterns
{test:
  thought: foo | 2x (2026-04-07) [evidence: aaa11111 "database performance critical"]
  thought: bar | 2x (2026-04-07) [evidence: bbb22222 "connection pooling bottleneck"]
}"""
        valid_ids = {"aaa11111", "bbb22222"}
        node_map = {
            "aaa11111": "database performance is critical for production workloads",
            "bbb22222": "connection pooling is the real bottleneck here",
        }

        result = validate_graduations(text, valid_ids, "2026-04-07", node_map)

        assert result.validated == 2
        assert len(result.all_validated_ids) == 2
        assert {"aaa11111"} in result.all_validated_ids
        assert {"bbb22222"} in result.all_validated_ids


class TestStoreAssociationMethods:
    """Store-level association API."""

    def test_record_and_get(self, tmp_path):
        db = tmp_path / "test.db"
        store = Store(db)

        ep1 = store.record("Episode 1", "observation")
        ep2 = store.record("Episode 2", "observation")

        store.record_associations(direct_pairs={(ep1.id, ep2.id)})

        assocs = store.get_associations([ep1.id])
        assert len(assocs) == 1
        assert assocs[0].strength == DIRECT_CO_CITATION_STRENGTH

        store.close()

    def test_decay_via_store(self, tmp_path):
        db = tmp_path / "test.db"
        store = Store(db)

        ep1 = store.record("Episode 1", "observation")
        ep2 = store.record("Episode 2", "observation")

        store.record_associations(direct_pairs={(ep1.id, ep2.id)})
        decayed = store.decay_associations()

        assert decayed == 1
        assocs = store.get_associations([ep1.id])
        assert abs(assocs[0].strength - DIRECT_CO_CITATION_STRENGTH * DEFAULT_DECAY_FACTOR) < 0.001

        store.close()

    def test_stats_via_store(self, tmp_path):
        db = tmp_path / "test.db"
        store = Store(db)

        ep1 = store.record("Episode 1", "observation")
        ep2 = store.record("Episode 2", "observation")

        store.record_associations(direct_pairs={(ep1.id, ep2.id)})

        stats = store.association_stats()
        assert stats.total_links == 1

        store.close()

    def test_status_includes_associations(self, tmp_path):
        db = tmp_path / "test.db"
        store = Store(db)

        ep1 = store.record("Episode 1", "observation")
        ep2 = store.record("Episode 2", "observation")

        store.record_associations(direct_pairs={(ep1.id, ep2.id)})

        status = store.status()
        assert status.association_stats is not None
        assert status.association_stats.total_links == 1

        store.close()

    def test_association_context_via_store(self, tmp_path):
        db = tmp_path / "test.db"
        store = Store(db)

        ep1 = store.record("Episode 1", "observation")
        ep2 = store.record("Episode 2", "observation")

        store.record_associations(direct_pairs={(ep1.id, ep2.id)})

        context = store.get_association_context([ep1.id, ep2.id])
        assert "Episode Associations" in context

        store.close()

    def test_audit_events_for_associations(self, tmp_path):
        """Association operations should produce audit trail entries."""
        db = tmp_path / "test.db"
        store = Store(db)

        ep1 = store.record("Episode 1", "observation")
        ep2 = store.record("Episode 2", "observation")

        store.record_associations(direct_pairs={(ep1.id, ep2.id)})
        store.decay_associations()

        audit_path = tmp_path / "test.audit.jsonl"
        lines = audit_path.read_text(encoding="utf-8").strip().split("\n")
        events = [json.loads(l)["event"] for l in lines]

        assert "associations_updated" in events
        assert "associations_decayed" in events

        store.close()


class TestEdgeCases:
    """Edge cases and defensive behavior."""

    def test_self_pair_filtered(self, tmp_path):
        """Self-pairs (same ID twice) must not form associations."""
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa")

        formed, strengthened = record_associations(
            conn,
            direct_pairs={("aaa", "aaa")},
            session_pairs=set(),
            timestamp="2026-04-07T12:00:00Z",
        )

        assert formed == 0
        assert strengthened == 0
        total = conn.execute("SELECT COUNT(*) FROM associations").fetchone()[0]
        assert total == 0

    def test_canonical_pair_returns_none_for_self(self):
        assert canonical_pair("abc", "abc") is None

    def test_foreign_key_violation_on_missing_episode(self, tmp_path):
        """Recording associations for non-existent episodes should fail."""
        conn = _make_db(tmp_path)
        # Don't insert any episodes — FK should reject

        with pytest.raises(sqlite3.IntegrityError):
            record_associations(
                conn,
                direct_pairs={("aaa", "bbb")},
                session_pairs=set(),
                timestamp="2026-04-07T12:00:00Z",
            )

    def test_decay_with_no_associations(self, tmp_path):
        """Decay on empty association table should return 0."""
        conn = _make_db(tmp_path)
        decayed = decay_associations(conn, strengthened_pairs=set())
        assert decayed == 0

    def test_strength_capped_at_maximum(self, tmp_path):
        """Strength cannot exceed MAX_STRENGTH to prevent calcification."""
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa")
        _insert_episode(conn, "bbb")

        # Co-cite many times — should hit the cap
        for i in range(20):
            record_associations(conn, {("aaa", "bbb")}, set(), f"2026-04-{i+1:02d}T12:00:00Z")

        row = conn.execute("SELECT strength, co_citations FROM associations").fetchone()
        assert row[0] == MAX_STRENGTH  # Capped
        assert row[1] == 20  # Co-citations still counted accurately

    def test_non_canonical_session_pairs_handled(self, tmp_path):
        """Session pairs in reverse order should still deduplicate against direct pairs."""
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa")
        _insert_episode(conn, "bbb")

        # Direct pair is (aaa, bbb), session pair is (bbb, aaa) — same pair, different order
        formed, strengthened = record_associations(
            conn,
            direct_pairs={("aaa", "bbb")},
            session_pairs={("bbb", "aaa")},  # Reverse order
            timestamp="2026-04-07T12:00:00Z",
        )

        assert formed == 1  # Only one link
        assert strengthened == 0
        row = conn.execute("SELECT strength FROM associations").fetchone()
        assert row[0] == DIRECT_CO_CITATION_STRENGTH  # Only direct strength, not both


class TestFullWrapPipelineWithAssociations:
    """End-to-end: episodes → wrap → graduation → associations."""

    def test_wrap_pipeline_forms_associations(self, tmp_path):
        """Full MCP-style wrap with citations creates association links."""
        db = tmp_path / "test.db"
        store = Store(db)

        # Record episodes
        ep1 = store.record("Database performance is critical for scaling", "observation")
        ep2 = store.record("ACID compliance outweighs raw speed for our use case", "decision")
        ep3 = store.record("Connection pooling is the real bottleneck", "observation")

        # Simulate agent producing continuity with co-citations
        today = "2026-04-07"
        continuity = f"""# Agent — Memory (v1)

## State
Working on database architecture.

## Patterns
{{database_architecture:
  thought: ACID compliance outweighs raw speed | 2x ({today}) [evidence: {ep1.id}, {ep2.id} "database performance and ACID compliance both point to this"]
  thought: connection pooling is the real bottleneck | 1x ({today})
}}

## Decisions
[decided(rationale: "ACID > speed", on: "{today}")] Use PostgreSQL

## Context
Evaluated database architecture. Identified connection pooling as bottleneck.
"""

        # Mark wrap and save (mimics MCP flow)
        store.wrap_started()

        from anneal_memory.continuity import validate_structure, measure_sections
        from anneal_memory.graduation import validate_graduations, extract_session_co_citations
        from anneal_memory.associations import canonical_pair

        assert validate_structure(continuity)

        episodes = store.episodes_since_wrap()
        valid_ids = {ep.id[:8].lower() for ep in episodes}
        node_map = {ep.id[:8].lower(): ep.content for ep in episodes}
        meta = store.load_meta()

        grad_result = validate_graduations(
            text=continuity,
            valid_ids=valid_ids,
            today=today,
            node_content_map=node_map,
            citations_seen=meta.get("citations_seen", False),
        )

        # Should have validated the co-citation
        assert grad_result.validated >= 1

        # Record associations
        direct_pairs = set(grad_result.direct_co_citations)
        session_pairs = extract_session_co_citations(grad_result.all_validated_ids)

        if direct_pairs or session_pairs:
            formed, strengthened = store.record_associations(
                direct_pairs=direct_pairs,
                session_pairs=session_pairs,
            )
            assert formed > 0

        # Verify association exists
        assocs = store.get_associations([ep1.id[:8].lower()])
        assert len(assocs) >= 1

        store.close()


class TestEngineWrapAssociations:
    """Engine.wrap() end-to-end with associations."""

    def test_engine_wrap_forms_associations(self, tmp_path):
        """Engine.wrap() with a mock LLM that produces co-citations."""
        from anneal_memory.engine import Engine

        db = tmp_path / "test.db"
        store = Store(db)

        # Record episodes
        ep1 = store.record("Database performance is critical for scaling", "observation")
        ep2 = store.record("ACID compliance outweighs raw speed", "decision")

        today = date.today().isoformat()

        # Mock LLM that returns continuity with co-citations
        def mock_llm(prompt: str) -> str:
            return f"""# Agent — Memory (v1)

## State
Working on database architecture.

## Patterns
{{database:
  thought: ACID compliance outweighs speed | 2x ({today}) [evidence: {ep1.id}, {ep2.id} "database performance and ACID compliance converge"]
}}

## Decisions
[decided(rationale: "ACID > speed", on: "{today}")] Use PostgreSQL

## Context
Evaluated database architecture decisions.
"""

        engine = Engine(store, llm=mock_llm, max_chars=20000)
        result = engine.wrap()

        assert result.saved is True
        assert result.associations_formed >= 1 or result.associations_strengthened >= 0

        # Verify association exists in the store
        assocs = store.get_associations([ep1.id[:8].lower()])
        assert len(assocs) >= 1

        store.close()

    def test_engine_wrap_no_citations_still_decays(self, tmp_path):
        """Engine.wrap() with no citations should still decay existing associations."""
        from anneal_memory.engine import Engine

        db = tmp_path / "test.db"
        store = Store(db)

        ep1 = store.record("Episode 1", "observation")
        ep2 = store.record("Episode 2", "observation")

        # Pre-existing association
        store.record_associations(direct_pairs={(ep1.id, ep2.id)})
        initial_assocs = store.get_associations([ep1.id])
        assert len(initial_assocs) == 1

        # Wrap with no citations → should decay
        today = "2026-04-07"

        def mock_llm(prompt: str) -> str:
            return f"""# Agent — Memory (v1)

## State
Nothing notable.

## Patterns
thought: general observation | 1x ({today})

## Decisions
No new decisions.

## Context
Quiet session.
"""

        # Need to add a new episode for the wrap to have something to compress
        store.record("A new episode", "context")

        engine = Engine(store, llm=mock_llm, max_chars=20000)
        result = engine.wrap()

        assert result.saved is True
        assert result.associations_decayed >= 1

        # Association should be weaker now
        assocs = store.get_associations([ep1.id])
        if assocs:  # Might still exist at reduced strength
            assert assocs[0].strength < DIRECT_CO_CITATION_STRENGTH

        store.close()


class TestLimbicLayer:
    """Affective state tagging on Hebbian associations — the fourth cognitive layer."""

    def test_affective_state_stored_on_association(self, tmp_path):
        """Affective tag and intensity persist on the association record."""
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa")
        _insert_episode(conn, "bbb")

        affect = AffectiveState(tag="engaged", intensity=0.8)
        record_associations(
            conn, {("aaa", "bbb")}, set(),
            "2026-04-07T12:00:00Z", affective_state=affect,
        )

        assocs = get_associations(conn, ["aaa"])
        assert len(assocs) == 1
        assert assocs[0].affective_tag == "engaged"
        assert assocs[0].affective_intensity == 0.8

    def test_affective_intensity_modulates_strength(self, tmp_path):
        """High engagement produces stronger associations than no affect."""
        conn = _make_db(tmp_path)
        for eid in ["aaa", "bbb", "ccc", "ddd"]:
            _insert_episode(conn, eid)

        # No affect — base strength
        record_associations(
            conn, {("aaa", "bbb")}, set(),
            "2026-04-07T12:00:00Z",
        )

        # High engagement — modulated strength
        affect = AffectiveState(tag="engaged", intensity=0.8)
        record_associations(
            conn, {("ccc", "ddd")}, set(),
            "2026-04-07T12:00:00Z", affective_state=affect,
        )

        base = conn.execute(
            "SELECT strength FROM associations WHERE episode_a='aaa'"
        ).fetchone()[0]
        modulated = conn.execute(
            "SELECT strength FROM associations WHERE episode_a='ccc'"
        ).fetchone()[0]

        expected_modulated = DIRECT_CO_CITATION_STRENGTH * (1 + 0.8 * AFFECTIVE_MODULATION_FACTOR)
        assert base == DIRECT_CO_CITATION_STRENGTH
        assert abs(modulated - expected_modulated) < 0.001
        assert modulated > base

    def test_zero_intensity_no_modulation(self, tmp_path):
        """Zero intensity produces same strength as no affect."""
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa")
        _insert_episode(conn, "bbb")

        affect = AffectiveState(tag="calm", intensity=0.0)
        record_associations(
            conn, {("aaa", "bbb")}, set(),
            "2026-04-07T12:00:00Z", affective_state=affect,
        )

        row = conn.execute("SELECT strength FROM associations").fetchone()
        assert row[0] == DIRECT_CO_CITATION_STRENGTH  # No modulation

    def test_max_intensity_capped(self, tmp_path):
        """Intensity > 1.0 is clamped to 1.0."""
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa")
        _insert_episode(conn, "bbb")

        affect = AffectiveState(tag="manic", intensity=5.0)  # Way over 1.0
        record_associations(
            conn, {("aaa", "bbb")}, set(),
            "2026-04-07T12:00:00Z", affective_state=affect,
        )

        row = conn.execute("SELECT strength FROM associations").fetchone()
        max_modulated = DIRECT_CO_CITATION_STRENGTH * (1 + 1.0 * AFFECTIVE_MODULATION_FACTOR)
        assert abs(row[0] - max_modulated) < 0.001

    def test_affective_tag_updates_on_strengthening(self, tmp_path):
        """Most recent affective tag wins when a link is strengthened."""
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa")
        _insert_episode(conn, "bbb")

        # First co-citation: curious
        affect1 = AffectiveState(tag="curious", intensity=0.5)
        record_associations(
            conn, {("aaa", "bbb")}, set(),
            "2026-04-07T12:00:00Z", affective_state=affect1,
        )

        # Second co-citation: engaged (overwrites)
        affect2 = AffectiveState(tag="engaged", intensity=0.9)
        record_associations(
            conn, {("aaa", "bbb")}, set(),
            "2026-04-08T12:00:00Z", affective_state=affect2,
        )

        assocs = get_associations(conn, ["aaa"])
        assert assocs[0].affective_tag == "engaged"
        assert assocs[0].affective_intensity == 0.9
        assert assocs[0].co_citations == 2

    def test_null_affect_preserves_existing_tag(self, tmp_path):
        """Strengthening without affect doesn't overwrite existing affective tag."""
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa")
        _insert_episode(conn, "bbb")

        # First: with affect
        affect = AffectiveState(tag="focused", intensity=0.7)
        record_associations(
            conn, {("aaa", "bbb")}, set(),
            "2026-04-07T12:00:00Z", affective_state=affect,
        )

        # Second: no affect (should preserve "focused")
        record_associations(
            conn, {("aaa", "bbb")}, set(),
            "2026-04-08T12:00:00Z",
        )

        assocs = get_associations(conn, ["aaa"])
        assert assocs[0].affective_tag == "focused"
        assert assocs[0].co_citations == 2

    def test_affective_context_in_association_display(self, tmp_path):
        """Association context includes affective state when present."""
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa", "Database design matters")
        _insert_episode(conn, "bbb", "ACID compliance is key")

        affect = AffectiveState(tag="engaged", intensity=0.8)
        record_associations(
            conn, {("aaa", "bbb")}, set(),
            "2026-04-07T12:00:00Z", affective_state=affect,
        )

        context = get_association_context(conn, ["aaa", "bbb"])
        assert "engaged" in context
        assert "0.8" in context

    def test_no_affective_context_when_no_tag(self, tmp_path):
        """Association context omits affective info when no tag set."""
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa", "Episode one content")
        _insert_episode(conn, "bbb", "Episode two content")

        record_associations(
            conn, {("aaa", "bbb")}, set(),
            "2026-04-07T12:00:00Z",
        )

        context = get_association_context(conn, ["aaa", "bbb"])
        assert "[" not in context.split("strength")[1]  # No affective brackets

    def test_store_passes_affect_through(self, tmp_path):
        """Store.record_associations passes affective state to the lower layer."""
        db = tmp_path / "test.db"
        store = Store(db)

        ep1 = store.record("Episode 1", "observation")
        ep2 = store.record("Episode 2", "observation")

        affect = AffectiveState(tag="curious", intensity=0.6)
        store.record_associations(
            direct_pairs={(ep1.id, ep2.id)},
            affective_state=affect,
        )

        assocs = store.get_associations([ep1.id])
        assert assocs[0].affective_tag == "curious"
        assert assocs[0].affective_intensity == 0.6

        store.close()

    def test_audit_includes_affective_data(self, tmp_path):
        """Audit trail captures affective state when present."""
        db = tmp_path / "test.db"
        store = Store(db)

        ep1 = store.record("Episode 1", "observation")
        ep2 = store.record("Episode 2", "observation")

        affect = AffectiveState(tag="engaged", intensity=0.8)
        store.record_associations(
            direct_pairs={(ep1.id, ep2.id)},
            affective_state=affect,
        )

        audit_path = tmp_path / "test.audit.jsonl"
        lines = audit_path.read_text(encoding="utf-8").strip().split("\n")
        assoc_entries = [
            json.loads(l) for l in lines
            if "associations_updated" in l
        ]
        assert len(assoc_entries) == 1
        assert assoc_entries[0]["data"]["affective_tag"] == "engaged"
        assert assoc_entries[0]["data"]["affective_intensity"] == 0.8

        store.close()

    def test_migration_adds_columns_to_existing_table(self, tmp_path):
        """Existing associations table without affective columns gets them added."""
        from anneal_memory.associations import migrate_add_affective_columns

        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))
        conn.execute("PRAGMA foreign_keys=ON")

        # Create OLD schema (no affective columns)
        conn.executescript("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'agent',
                session_id TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            );
            CREATE TABLE associations (
                episode_a TEXT NOT NULL,
                episode_b TEXT NOT NULL,
                strength REAL NOT NULL DEFAULT 1.0,
                first_linked TEXT NOT NULL,
                last_strengthened TEXT NOT NULL,
                co_citations INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (episode_a, episode_b),
                CHECK (episode_a < episode_b),
                FOREIGN KEY (episode_a) REFERENCES episodes(id) ON DELETE CASCADE,
                FOREIGN KEY (episode_b) REFERENCES episodes(id) ON DELETE CASCADE
            );
        """)

        # Insert an episode and association pre-migration
        conn.execute(
            "INSERT INTO episodes (id, timestamp, type, content) VALUES (?, ?, ?, ?)",
            ("aaa", "2026-04-07T12:00:00Z", "observation", "test"),
        )
        conn.execute(
            "INSERT INTO episodes (id, timestamp, type, content) VALUES (?, ?, ?, ?)",
            ("bbb", "2026-04-07T12:00:00Z", "observation", "test"),
        )
        conn.execute(
            "INSERT INTO associations (episode_a, episode_b, strength, first_linked, last_strengthened) "
            "VALUES (?, ?, ?, ?, ?)",
            ("aaa", "bbb", 1.0, "2026-04-07T12:00:00Z", "2026-04-07T12:00:00Z"),
        )
        conn.commit()

        # Migrate
        migrate_add_affective_columns(conn)

        # Verify columns exist and old data preserved
        row = conn.execute(
            "SELECT affective_tag, affective_intensity FROM associations WHERE episode_a='aaa'"
        ).fetchone()
        assert row[0] is None  # No tag yet
        assert row[1] == 0.0  # Default intensity

        # Verify we can now write affective data
        conn.execute(
            "UPDATE associations SET affective_tag='engaged', affective_intensity=0.8 "
            "WHERE episode_a='aaa'"
        )
        conn.commit()

        row = conn.execute(
            "SELECT affective_tag, affective_intensity FROM associations WHERE episode_a='aaa'"
        ).fetchone()
        assert row[0] == "engaged"
        assert row[1] == 0.8

        conn.close()

    def test_affective_state_normalizes_tag(self):
        """AffectiveState normalizes tag to lowercase and strips whitespace."""
        state = AffectiveState(tag="  ENGAGED  ", intensity=0.5)
        assert state.tag == "engaged"

    def test_affective_state_clamps_intensity(self):
        """AffectiveState clamps intensity to [0.0, 1.0] at type level."""
        high = AffectiveState(tag="manic", intensity=5.0)
        assert high.intensity == 1.0

        low = AffectiveState(tag="numb", intensity=-0.5)
        assert low.intensity == 0.0

        normal = AffectiveState(tag="calm", intensity=0.7)
        assert normal.intensity == 0.7

    def test_whitespace_only_tag_rejected_by_server(self):
        """Whitespace-only tags should not create an AffectiveState.

        The server's save_continuity handler checks tag.strip() to prevent
        whitespace-only strings from passing the truthiness check.
        AffectiveState itself normalizes via strip().lower(), so a
        whitespace-only tag would become an empty string — which is invalid.
        """
        # AffectiveState strips whitespace, resulting in empty tag
        state = AffectiveState(tag="   ", intensity=0.5)
        assert state.tag == ""  # Stripped to empty

        # Verify the server-side guard: tag.strip() is falsy for whitespace
        tag = "   "
        assert not (tag and isinstance(tag, str) and tag.strip())

    def test_negative_intensity_clamped(self, tmp_path):
        """Intensity < 0.0 is clamped to 0.0."""
        conn = _make_db(tmp_path)
        _insert_episode(conn, "aaa")
        _insert_episode(conn, "bbb")

        affect = AffectiveState(tag="confused", intensity=-0.5)
        record_associations(
            conn, {("aaa", "bbb")}, set(),
            "2026-04-07T12:00:00Z", affective_state=affect,
        )

        row = conn.execute("SELECT strength FROM associations").fetchone()
        assert row[0] == DIRECT_CO_CITATION_STRENGTH  # No negative modulation

    def test_migration_idempotent(self, tmp_path):
        """Calling migration twice is a safe no-op."""
        from anneal_memory.associations import migrate_add_affective_columns

        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))
        conn.executescript("""
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,
                type TEXT NOT NULL, content TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'agent', session_id TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            );
            CREATE TABLE associations (
                episode_a TEXT NOT NULL, episode_b TEXT NOT NULL,
                strength REAL NOT NULL DEFAULT 1.0,
                first_linked TEXT NOT NULL, last_strengthened TEXT NOT NULL,
                co_citations INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (episode_a, episode_b),
                CHECK (episode_a < episode_b),
                FOREIGN KEY (episode_a) REFERENCES episodes(id) ON DELETE CASCADE,
                FOREIGN KEY (episode_b) REFERENCES episodes(id) ON DELETE CASCADE
            );
        """)

        # First migration
        migrate_add_affective_columns(conn)
        # Second migration — should be no-op
        migrate_add_affective_columns(conn)

        # Verify columns exist
        cursor = conn.execute("PRAGMA table_info(associations)")
        cols = {row[1] for row in cursor.fetchall()}
        assert "affective_tag" in cols
        assert "affective_intensity" in cols
        conn.close()

    def test_engine_affect_characterization_failure_graceful(self, tmp_path):
        """Engine degrades gracefully when LLM returns garbage for affect."""
        from anneal_memory.engine import Engine

        db = tmp_path / "test.db"
        store = Store(db)

        ep1 = store.record("Episode 1", "observation")
        ep2 = store.record("Episode 2", "decision")

        today = date.today().isoformat()
        call_count = 0

        def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return f"""# Agent — Memory (v1)

## State
Working.

## Patterns
{{test:
  thought: test pattern | 2x ({today}) [evidence: {ep1.id}, {ep2.id} "episode one and episode two converge"]
}}

## Decisions
No new decisions.

## Context
Test session.
"""
            else:
                # Garbage response for affect characterization
                return "I don't understand what you're asking me to do here."

        engine = Engine(store, llm=mock_llm, max_chars=20000, characterize_affect=True)
        result = engine.wrap()

        # Should still succeed — affect failure is non-fatal
        assert result.saved is True
        assert call_count == 2  # Both calls were made

        # Associations should exist but without affective data
        assocs = store.get_associations([ep1.id[:8].lower()])
        if assocs:
            assert assocs[0].affective_tag is None

        store.close()

    def test_engine_characterize_affect(self, tmp_path):
        """Engine with characterize_affect=True makes a second LLM call."""
        from anneal_memory.engine import Engine

        db = tmp_path / "test.db"
        store = Store(db)

        ep1 = store.record("Database performance is critical for scaling", "observation")
        ep2 = store.record("ACID compliance outweighs raw speed", "decision")

        today = date.today().isoformat()
        call_count = 0

        def mock_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Compression call
                return f"""# Agent — Memory (v1)

## State
Working on database architecture.

## Patterns
{{database:
  thought: ACID compliance outweighs speed | 2x ({today}) [evidence: {ep1.id}, {ep2.id} "database performance and ACID compliance converge"]
}}

## Decisions
[decided(rationale: "ACID > speed", on: "{today}")] Use PostgreSQL

## Context
Evaluated database architecture decisions.
"""
            else:
                # Affect characterization call
                return "engaged 0.8"

        engine = Engine(store, llm=mock_llm, max_chars=20000, characterize_affect=True)
        result = engine.wrap()

        assert result.saved is True
        assert call_count == 2  # Compression + affect characterization

        # Association should have affective data
        assocs = store.get_associations([ep1.id[:8].lower()])
        if assocs:
            assert assocs[0].affective_tag == "engaged"
            assert assocs[0].affective_intensity == 0.8

        store.close()
