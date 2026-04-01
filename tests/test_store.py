"""Tests for the SQLite episodic store."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from anneal_memory.store import Store
from anneal_memory.types import EpisodeType, StoreStatus, WrapResult


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database path."""
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def store(tmp_db):
    """Create a store instance with cleanup."""
    s = Store(tmp_db, project_name="TestProject")
    yield s
    s.close()


# -- Initialization --


class TestInit:
    def test_creates_database_file(self, tmp_db):
        s = Store(tmp_db)
        assert Path(tmp_db).exists()
        s.close()

    def test_creates_parent_directories(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "memory.db"
        s = Store(str(deep_path))
        assert deep_path.exists()
        s.close()

    def test_default_metadata_set(self, store):
        row = store._conn.execute(
            "SELECT value FROM metadata WHERE key = 'format_version'"
        ).fetchone()
        assert row["value"] == "1"

    def test_project_name_stored(self, store):
        row = store._conn.execute(
            "SELECT value FROM metadata WHERE key = 'project_name'"
        ).fetchone()
        assert row["value"] == "TestProject"

    def test_wal_mode_enabled(self, store):
        row = store._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"

    def test_context_manager(self, tmp_db):
        with Store(tmp_db) as s:
            ep = s.record("test", EpisodeType.OBSERVATION)
            assert ep.id

    def test_reopening_existing_db(self, tmp_db):
        s1 = Store(tmp_db)
        s1.record("episode one", EpisodeType.OBSERVATION)
        s1.close()

        s2 = Store(tmp_db)
        result = s2.recall()
        assert result.total_matching == 1
        assert result.episodes[0].content == "episode one"
        s2.close()


# -- Recording --


class TestRecord:
    def test_basic_record(self, store):
        ep = store.record("An observation", EpisodeType.OBSERVATION)
        assert len(ep.id) == 8
        assert ep.type == EpisodeType.OBSERVATION
        assert ep.content == "An observation"
        assert ep.source == "agent"
        assert ep.timestamp  # not empty

    def test_record_all_types(self, store):
        for ep_type in EpisodeType:
            ep = store.record(f"Test {ep_type.value}", ep_type)
            assert ep.type == ep_type

    def test_record_with_string_type(self, store):
        ep = store.record("A decision", "decision")
        assert ep.type == EpisodeType.DECISION

    def test_record_with_source(self, store):
        ep = store.record("From daemon", EpisodeType.OBSERVATION, source="daemon")
        assert ep.source == "daemon"

    def test_record_with_metadata(self, store):
        meta = {"confidence": 0.9, "tags": ["important"]}
        ep = store.record("Meta episode", EpisodeType.OBSERVATION, metadata=meta)
        assert ep.metadata == meta

        # Verify persisted
        retrieved = store.get(ep.id)
        assert retrieved.metadata == meta

    def test_record_with_custom_timestamp(self, store):
        ts = "2026-03-30T12:00:00Z"
        ep = store.record("Past episode", EpisodeType.OBSERVATION, timestamp=ts)
        assert ep.timestamp == ts

    def test_record_persists(self, store):
        ep = store.record("Persistent", EpisodeType.DECISION)
        retrieved = store.get(ep.id)
        assert retrieved is not None
        assert retrieved.content == "Persistent"
        assert retrieved.type == EpisodeType.DECISION

    def test_unique_ids_for_different_content(self, store):
        ep1 = store.record("Content A", EpisodeType.OBSERVATION)
        ep2 = store.record("Content B", EpisodeType.OBSERVATION)
        assert ep1.id != ep2.id

    def test_invalid_type_raises(self, store):
        with pytest.raises(ValueError):
            store.record("Bad type", "not_a_type")

    def test_empty_content_raises(self, store):
        with pytest.raises(ValueError, match="cannot be empty"):
            store.record("", EpisodeType.OBSERVATION)

    def test_whitespace_only_content_raises(self, store):
        with pytest.raises(ValueError, match="cannot be empty"):
            store.record("   ", EpisodeType.OBSERVATION)

    def test_empty_dict_metadata_round_trips(self, store):
        ep = store.record("With empty meta", EpisodeType.OBSERVATION, metadata={})
        retrieved = store.get(ep.id)
        assert retrieved.metadata == {}


# -- Get --


class TestGet:
    def test_get_existing(self, store):
        ep = store.record("Find me", EpisodeType.QUESTION)
        found = store.get(ep.id)
        assert found is not None
        assert found.content == "Find me"

    def test_get_nonexistent(self, store):
        assert store.get("00000000") is None

    def test_get_preserves_all_fields(self, store):
        meta = {"key": "value"}
        ep = store.record(
            "Full episode",
            EpisodeType.TENSION,
            source="test_agent",
            metadata=meta,
        )
        found = store.get(ep.id)
        assert found.type == EpisodeType.TENSION
        assert found.source == "test_agent"
        assert found.metadata == meta


# -- Recall --


class TestRecall:
    def test_recall_all(self, store):
        store.record("One", EpisodeType.OBSERVATION)
        store.record("Two", EpisodeType.DECISION)
        store.record("Three", EpisodeType.QUESTION)
        result = store.recall()
        assert result.total_matching == 3
        assert len(result.episodes) == 3

    def test_recall_by_type(self, store):
        store.record("Obs 1", EpisodeType.OBSERVATION)
        store.record("Dec 1", EpisodeType.DECISION)
        store.record("Obs 2", EpisodeType.OBSERVATION)
        result = store.recall(episode_type=EpisodeType.OBSERVATION)
        assert result.total_matching == 2
        assert all(ep.type == EpisodeType.OBSERVATION for ep in result.episodes)

    def test_recall_by_string_type(self, store):
        store.record("Dec", EpisodeType.DECISION)
        result = store.recall(episode_type="decision")
        assert result.total_matching == 1

    def test_recall_by_source(self, store):
        store.record("Agent ep", EpisodeType.OBSERVATION, source="agent")
        store.record("Daemon ep", EpisodeType.OBSERVATION, source="daemon")
        result = store.recall(source="daemon")
        assert result.total_matching == 1
        assert result.episodes[0].source == "daemon"

    def test_recall_by_keyword(self, store):
        store.record("The database is slow", EpisodeType.OBSERVATION)
        store.record("The API is fast", EpisodeType.OBSERVATION)
        result = store.recall(keyword="database")
        assert result.total_matching == 1
        assert "database" in result.episodes[0].content

    def test_recall_by_time_range(self, store):
        store.record("Old", EpisodeType.OBSERVATION, timestamp="2026-03-01T00:00:00Z")
        store.record("Mid", EpisodeType.OBSERVATION, timestamp="2026-03-15T00:00:00Z")
        store.record("New", EpisodeType.OBSERVATION, timestamp="2026-03-30T00:00:00Z")

        result = store.recall(since="2026-03-10T00:00:00Z")
        assert result.total_matching == 2

        result = store.recall(until="2026-03-20T00:00:00Z")
        assert result.total_matching == 2

        result = store.recall(
            since="2026-03-10T00:00:00Z", until="2026-03-20T00:00:00Z"
        )
        assert result.total_matching == 1

    def test_recall_with_limit(self, store):
        for i in range(10):
            store.record(f"Episode {i}", EpisodeType.OBSERVATION)
        result = store.recall(limit=3)
        assert len(result.episodes) == 3
        assert result.total_matching == 10

    def test_recall_with_offset(self, store):
        for i in range(10):
            store.record(f"Episode {i}", EpisodeType.OBSERVATION)
        result = store.recall(limit=3, offset=7)
        assert len(result.episodes) == 3
        assert result.total_matching == 10

    def test_recall_combined_filters(self, store):
        store.record("DB observation", EpisodeType.OBSERVATION, source="agent")
        store.record("DB decision", EpisodeType.DECISION, source="agent")
        store.record("API observation", EpisodeType.OBSERVATION, source="daemon")
        result = store.recall(episode_type=EpisodeType.OBSERVATION, source="agent")
        assert result.total_matching == 1
        assert result.episodes[0].content == "DB observation"

    def test_recall_empty_store(self, store):
        result = store.recall()
        assert result.total_matching == 0
        assert result.episodes == []

    def test_recall_returns_newest_first(self, store):
        store.record("Old", EpisodeType.OBSERVATION, timestamp="2026-03-01T00:00:00Z")
        store.record("New", EpisodeType.OBSERVATION, timestamp="2026-03-30T00:00:00Z")
        result = store.recall()
        assert result.episodes[0].content == "New"
        assert result.episodes[1].content == "Old"

    def test_recall_query_params_captured(self, store):
        result = store.recall(episode_type=EpisodeType.DECISION, keyword="test", limit=5)
        assert result.query_params["type"] == "decision"
        assert result.query_params["keyword"] == "test"
        assert result.query_params["limit"] == 5


# -- Episodes Since Wrap --


class TestEpisodesSinceWrap:
    def test_no_wraps_returns_all(self, store):
        store.record("One", EpisodeType.OBSERVATION)
        store.record("Two", EpisodeType.OBSERVATION)
        eps = store.episodes_since_wrap()
        assert len(eps) == 2

    def test_returns_only_after_last_wrap(self, store):
        store.record("Before wrap", EpisodeType.OBSERVATION)
        store.wrap_completed(episodes_compressed=1, continuity_chars=100)

        store.record("After wrap", EpisodeType.OBSERVATION)
        eps = store.episodes_since_wrap()
        assert len(eps) == 1
        assert eps[0].content == "After wrap"

    def test_returns_chronological_order(self, store):
        store.record("First", EpisodeType.OBSERVATION, timestamp="2026-03-30T01:00:00Z")
        store.record("Second", EpisodeType.OBSERVATION, timestamp="2026-03-30T02:00:00Z")
        eps = store.episodes_since_wrap()
        assert eps[0].content == "First"
        assert eps[1].content == "Second"


# -- Wrap Lifecycle --


class TestWrapLifecycle:
    def test_wrap_started_sets_flag(self, store):
        store.wrap_started()
        status = store.status()
        assert status.wrap_in_progress is True

    def test_wrap_completed_clears_flag(self, store):
        store.wrap_started()
        store.wrap_completed(episodes_compressed=0, continuity_chars=0)
        status = store.status()
        assert status.wrap_in_progress is False

    def test_wrap_records_metrics(self, store):
        store.wrap_completed(
            episodes_compressed=5,
            continuity_chars=3000,
            graduations_validated=2,
            graduations_demoted=1,
            citation_reuse_max=1,
            patterns_extracted=8,
        )
        row = store._conn.execute(
            "SELECT * FROM wraps ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row["episodes_compressed"] == 5
        assert row["continuity_chars"] == 3000
        assert row["graduations_validated"] == 2
        assert row["graduations_demoted"] == 1
        assert row["patterns_extracted"] == 8

    def test_multiple_wraps_tracked(self, store):
        store.wrap_completed(episodes_compressed=3, continuity_chars=1000)
        store.wrap_completed(episodes_compressed=5, continuity_chars=2000)
        status = store.status()
        assert status.total_wraps == 2


# -- Pruning --


class TestPruning:
    def test_prune_disabled_by_default(self, store):
        store.record("Old", EpisodeType.OBSERVATION, timestamp="2020-01-01T00:00:00Z")
        pruned = store.prune()
        assert pruned == 0

    def test_prune_with_retention_days(self, tmp_db):
        s = Store(tmp_db, retention_days=30)
        s.record("Old", EpisodeType.OBSERVATION, timestamp="2020-01-01T00:00:00Z")
        s.record("Recent", EpisodeType.OBSERVATION)
        pruned = s.prune()
        assert pruned == 1
        assert s.recall().total_matching == 1
        s.close()

    def test_prune_override_retention(self, store):
        store.record("Old", EpisodeType.OBSERVATION, timestamp="2020-01-01T00:00:00Z")
        store.record("Recent", EpisodeType.OBSERVATION)
        pruned = store.prune(older_than_days=1)
        assert pruned == 1

    def test_prune_creates_tombstones(self, tmp_db):
        s = Store(tmp_db, retention_days=1, keep_tombstones=True)
        ep = s.record("Prunable", EpisodeType.OBSERVATION, timestamp="2020-01-01T00:00:00Z")
        s.prune()

        row = s._conn.execute(
            "SELECT * FROM tombstones WHERE id = ?", (ep.id,)
        ).fetchone()
        assert row is not None
        assert row["type"] == "observation"
        assert row["content_hash"]  # SHA256 present
        s.close()

    def test_prune_without_tombstones(self, tmp_db):
        s = Store(tmp_db, retention_days=1, keep_tombstones=False)
        ep = s.record("Prunable", EpisodeType.OBSERVATION, timestamp="2020-01-01T00:00:00Z")
        s.prune()

        row = s._conn.execute(
            "SELECT * FROM tombstones WHERE id = ?", (ep.id,)
        ).fetchone()
        assert row is None
        s.close()

    def test_prune_nothing_to_prune(self, tmp_db):
        s = Store(tmp_db, retention_days=30)
        s.record("Recent", EpisodeType.OBSERVATION)
        pruned = s.prune()
        assert pruned == 0
        s.close()


# -- Continuity I/O --


class TestContinuityIO:
    def test_continuity_path_derived(self, store):
        assert store.continuity_path.name == "test_memory.continuity.md"

    def test_meta_path_derived(self, store):
        assert store.meta_path.name == "test_memory.continuity.meta.json"

    def test_load_continuity_none_if_missing(self, store):
        assert store.load_continuity() is None

    def test_save_and_load_continuity(self, store):
        text = "# Test — Memory (v1)\n\n## State\nTesting.\n"
        store.save_continuity(text)
        loaded = store.load_continuity()
        assert loaded == text

    def test_save_continuity_atomic(self, store):
        store.save_continuity("Version 1")
        store.save_continuity("Version 2")
        assert store.load_continuity() == "Version 2"
        # No .tmp files should remain
        tmp_files = list(store.continuity_path.parent.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_save_and_load_meta(self, store):
        meta = {"sessions_produced": 3, "citations_seen": True, "format_version": 1}
        store.save_meta(meta)
        loaded = store.load_meta()
        assert loaded["sessions_produced"] == 3
        assert loaded["citations_seen"] is True

    def test_load_meta_defaults(self, store):
        meta = store.load_meta()
        assert meta["sessions_produced"] == 0
        assert meta["citations_seen"] is False
        assert meta["format_version"] == 1


# -- Status --


class TestStatus:
    def test_empty_store_status(self, store):
        status = store.status()
        assert status.total_episodes == 0
        assert status.episodes_since_wrap == 0
        assert status.total_wraps == 0
        assert status.last_wrap_at is None
        assert status.wrap_in_progress is False
        assert status.tombstone_count == 0
        assert status.continuity_chars is None

    def test_status_after_records(self, store):
        store.record("One", EpisodeType.OBSERVATION)
        store.record("Two", EpisodeType.DECISION)
        status = store.status()
        assert status.total_episodes == 2
        assert status.episodes_since_wrap == 2
        assert status.episodes_by_type["observation"] == 1
        assert status.episodes_by_type["decision"] == 1

    def test_status_with_continuity(self, store):
        store.save_continuity("Some continuity text here.")
        status = store.status()
        assert status.continuity_chars == len("Some continuity text here.")

    def test_status_wrap_in_progress(self, store):
        store.wrap_started()
        status = store.status()
        assert status.wrap_in_progress is True

    def test_status_after_wrap(self, store):
        store.record("Ep", EpisodeType.OBSERVATION)
        store.wrap_completed(episodes_compressed=1, continuity_chars=500)
        status = store.status()
        assert status.total_wraps == 1
        assert status.last_wrap_at is not None


# -- Delete --


class TestDelete:
    def test_delete_existing(self, store):
        ep = store.record("Delete me", EpisodeType.OBSERVATION)
        assert store.delete(ep.id) is True
        assert store.get(ep.id) is None

    def test_delete_nonexistent(self, store):
        assert store.delete("00000000") is False

    def test_delete_creates_tombstone(self, store):
        ep = store.record("Tombstone test", EpisodeType.DECISION)
        store.delete(ep.id)
        row = store._conn.execute(
            "SELECT * FROM tombstones WHERE id = ?", (ep.id,)
        ).fetchone()
        assert row is not None
        assert row["type"] == "decision"

    def test_delete_without_tombstone(self, tmp_db):
        s = Store(tmp_db, keep_tombstones=False)
        ep = s.record("No tombstone", EpisodeType.OBSERVATION)
        s.delete(ep.id)
        row = s._conn.execute(
            "SELECT * FROM tombstones WHERE id = ?", (ep.id,)
        ).fetchone()
        assert row is None
        s.close()

    def test_delete_reduces_count(self, store):
        store.record("One", EpisodeType.OBSERVATION)
        ep2 = store.record("Two", EpisodeType.OBSERVATION)
        store.delete(ep2.id)
        assert store.status().total_episodes == 1


# -- ID Collision Handling --


class TestIDCollision:
    def test_duplicate_content_same_timestamp_gets_different_id(self, store):
        """Same content + same timestamp should get different IDs via nonce retry."""
        ts = "2026-03-31T12:00:00.000000Z"
        ep1 = store.record("Identical content", EpisodeType.OBSERVATION, timestamp=ts)
        ep2 = store.record("Identical content", EpisodeType.OBSERVATION, timestamp=ts)
        assert ep1.id != ep2.id
        assert store.get(ep1.id) is not None
        assert store.get(ep2.id) is not None


# -- Prune Edge Cases --


class TestPruneEdgeCases:
    def test_prune_zero_days(self, store):
        """prune(older_than_days=0) should prune everything."""
        store.record("Recent", EpisodeType.OBSERVATION)
        pruned = store.prune(older_than_days=0)
        assert pruned == 1
        assert store.status().total_episodes == 0

    def test_prune_negative_days_raises(self, store):
        with pytest.raises(ValueError, match="must be >= 0"):
            store.prune(older_than_days=-1)


# -- Wrap Returns WrapResult --


class TestWrapResult:
    def test_wrap_completed_returns_result(self, store):
        result = store.wrap_completed(
            episodes_compressed=3,
            continuity_chars=2000,
            graduations_validated=2,
            graduations_demoted=1,
        )
        assert isinstance(result, WrapResult)
        assert result.saved is True
        assert result.chars == 2000
        assert result.episodes_compressed == 3
        assert result.graduations_validated == 2
        assert result.graduations_demoted == 1


# -- LIKE Wildcard Escaping --


class TestKeywordEscaping:
    def test_percent_in_keyword(self, store):
        store.record("100% complete", EpisodeType.OBSERVATION)
        store.record("Not matching", EpisodeType.OBSERVATION)
        result = store.recall(keyword="100%")
        assert result.total_matching == 1
        assert "100%" in result.episodes[0].content

    def test_underscore_in_keyword(self, store):
        store.record("file_name.py", EpisodeType.OBSERVATION)
        store.record("filename.py", EpisodeType.OBSERVATION)
        result = store.recall(keyword="file_name")
        assert result.total_matching == 1

    def test_case_insensitive_keyword(self, store):
        store.record("SQLite is the right choice", EpisodeType.OBSERVATION)
        store.record("Redis is fast", EpisodeType.OBSERVATION)
        # Search lowercase should find uppercase content
        result = store.recall(keyword="sqlite")
        assert result.total_matching == 1
        assert "SQLite" in result.episodes[0].content
        # Search uppercase should also work
        result2 = store.recall(keyword="SQLITE")
        assert result2.total_matching == 1
