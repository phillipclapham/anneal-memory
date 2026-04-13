"""Tests for the SQLite episodic store."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from dataclasses import asdict, is_dataclass
from datetime import date

from anneal_memory.store import (
    AnnealMemoryError,
    Store,
    StoreDatabaseError,
    StoreError,
)
from anneal_memory.types import EpisodeType, StoreStatus, WrapRecord, WrapResult


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
        with pytest.warns(DeprecationWarning, match="legacy call form"):
            store.wrap_started()
        status = store.status()
        assert status.wrap_in_progress is True

    def test_wrap_completed_clears_flag(self, store):
        with pytest.warns(DeprecationWarning, match="legacy call form"):
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

    def test_wrap_completed_auto_prunes_when_retention_set(self, tmp_db):
        """wrap_completed auto-prunes old episodes when retention_days is set."""
        s = Store(tmp_db, retention_days=30)
        s.record("Old episode", EpisodeType.OBSERVATION, timestamp="2020-01-01T00:00:00Z")
        s.record("Recent episode", EpisodeType.OBSERVATION)
        assert s.status().total_episodes == 2

        s.wrap_completed(episodes_compressed=2, continuity_chars=100)

        # Old episode pruned, recent kept
        assert s.status().total_episodes == 1
        s.close()

    def test_wrap_completed_no_prune_without_retention(self, store):
        """wrap_completed does NOT prune when retention_days is not set."""
        store.record("Old episode", EpisodeType.OBSERVATION, timestamp="2020-01-01T00:00:00Z")
        store.record("Recent episode", EpisodeType.OBSERVATION)

        store.wrap_completed(episodes_compressed=2, continuity_chars=100)

        # Both episodes still present (no auto-prune)
        assert store.status().total_episodes == 2


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


class TestStoreError:
    """Tests for the StoreError exception class (10.5c.3 + fix pass).

    Locks in the library-level exception hierarchy:

    - ``AnnealMemoryError(Exception)`` is the library boundary class.
    - ``StoreError(AnnealMemoryError)`` carries ``.operation`` and
      ``.path`` context so transports can surface clean error messages.
    - ``StoreError`` round-trips cleanly through pickle and copy —
      critical for downstream consumers that marshal exceptions
      (pytest-xdist, ProcessPoolExecutor, logging frameworks, RPC).
    - ``save_continuity`` and ``save_meta`` raise ``StoreError`` on
      OSError and preserve the atomic-write invariant (no stale
      ``.tmp``, original file untouched) on ANY exception class —
      not just OSError.

    Fault injection targets ``Path.replace`` (not ``builtins.open``).
    This is the right surface: the tmp file IS successfully created,
    the rename stage fails, and the cleanup path runs with a real
    tmp file to delete. Patching ``builtins.open`` crashes before the
    tmp file exists, making cleanup assertions vacuously true.
    """

    # -- hierarchy --

    def test_store_error_is_anneal_memory_error_subclass(self):
        """StoreError subclasses the library base — catching
        AnnealMemoryError at a transport boundary catches all
        anneal-memory failures regardless of subsystem."""
        err = StoreError("boom", operation="save_continuity", path="/tmp/x")
        assert isinstance(err, AnnealMemoryError)
        assert isinstance(err, StoreError)
        assert isinstance(err, Exception)
        # Explicit non-OSError check — 10.5c.3 dropped the OSError
        # subclass. Library-level domain errors don't pretend to be
        # kernel I/O errors; the original OSError lives in __cause__.
        assert not isinstance(err, OSError)

    def test_anneal_memory_error_catches_store_error(self):
        """``except AnnealMemoryError`` is the documented library
        boundary catch. Regression-locks the hierarchy."""
        try:
            raise StoreError("io failed", operation="save_continuity")
        except AnnealMemoryError as caught:
            assert isinstance(caught, StoreError)
            assert caught.operation == "save_continuity"
        else:
            pytest.fail("AnnealMemoryError handler did not catch StoreError")

    def test_store_error_carries_context(self):
        err = StoreError(
            "disk full",
            operation="save_meta",
            path="/var/run/anneal/meta.json",
        )
        assert err.operation == "save_meta"
        assert err.path == "/var/run/anneal/meta.json"
        assert "disk full" in str(err)

    def test_store_error_repr_is_informative(self):
        err = StoreError(
            "disk full",
            operation="save_continuity",
            path="/tmp/x",
        )
        r = repr(err)
        assert "StoreError" in r
        assert "disk full" in r
        assert "save_continuity" in r
        assert "/tmp/x" in r

    def test_store_error_exported_from_package(self):
        from anneal_memory import AnnealMemoryError as ExportedBase
        from anneal_memory import StoreError as ExportedStoreError
        from anneal_memory.store import AnnealMemoryError as InternalBase
        from anneal_memory.store import StoreError as InternalStoreError
        assert ExportedBase is InternalBase
        assert ExportedStoreError is InternalStoreError

    # -- pickle / copy round-trips --

    def test_store_error_pickle_roundtrip(self):
        """pytest-xdist, ProcessPoolExecutor, and RPC transports all
        pickle exceptions across process boundaries. StoreError MUST
        survive the round-trip with operation/path intact."""
        import pickle
        err = StoreError(
            "injected disk failure",
            operation="save_continuity",
            path="/tmp/c.md",
        )
        restored = pickle.loads(pickle.dumps(err))
        assert isinstance(restored, StoreError)
        assert str(restored) == "injected disk failure"
        assert restored.operation == "save_continuity"
        assert restored.path == "/tmp/c.md"

    def test_store_error_copy_roundtrip(self):
        """copy.copy + copy.deepcopy use the same __reduce__ path."""
        import copy
        err = StoreError(
            "permission denied",
            operation="save_meta",
            path="/etc/anneal/meta.json",
        )
        shallow = copy.copy(err)
        deep = copy.deepcopy(err)
        for other in (shallow, deep):
            assert isinstance(other, StoreError)
            assert str(other) == "permission denied"
            assert other.operation == "save_meta"
            assert other.path == "/etc/anneal/meta.json"

    def test_store_error_pickle_with_none_path(self):
        """Path is optional on the constructor; pickle must round-trip
        the None case."""
        import pickle
        err = StoreError("no path given", operation="save_continuity")
        restored = pickle.loads(pickle.dumps(err))
        assert restored.path is None
        assert restored.operation == "save_continuity"

    def test_store_error_pickle_reflects_args_mutation(self):
        """``__reduce__`` uses ``str(self)`` which reads ``args[0]``.
        If a caller mutates ``err.args`` after construction (matching
        stdlib Exception semantics), the pickled instance reflects
        the mutation. Locks in the stdlib-compatible behavior so
        a future refactor that caches the message in ``__init__``
        (instead of re-reading args) fails loudly."""
        import pickle
        err = StoreError("original", operation="save_continuity")
        err.args = ("mutated",)
        restored = pickle.loads(pickle.dumps(err))
        assert str(restored) == "mutated"
        assert restored.operation == "save_continuity"

    # -- save_continuity failure paths --

    def test_save_continuity_raises_store_error_on_replace_failure(
        self, store, monkeypatch
    ):
        """Patch ``Path.replace`` so the tmp file IS created, the
        atomic rename then fails, and the cleanup path runs on a
        real tmp file. This is the right fault-injection surface."""
        real_replace = Path.replace

        def exploding_replace(self, target):
            raise OSError("injected rename failure")

        monkeypatch.setattr(Path, "replace", exploding_replace)

        with pytest.raises(StoreError) as exc_info:
            store.save_continuity(
                "# T — Memory (v1)\n\n## State\nx\n"
                "\n## Patterns\np\n\n## Decisions\nd\n\n## Context\nc\n"
            )

        err = exc_info.value
        assert err.operation == "save_continuity"
        assert err.path == str(store.continuity_path)
        assert isinstance(err.__cause__, OSError)
        assert "injected rename failure" in str(err.__cause__)
        # Tmp cleanup ran — no stale sidecar on disk.
        assert not list(store.continuity_path.parent.glob("*.md.tmp"))
        # Main continuity file UNTOUCHED — the atomic-write
        # invariant is that the old file survives a failed new write.
        # First-write case: file never existed, still doesn't exist.
        assert not store.continuity_path.exists()

    def test_save_continuity_preserves_previous_file_on_failure(
        self, store, monkeypatch
    ):
        """Stronger atomic-write invariant: when a previous continuity
        file exists and a new write fails, the OLD file survives
        intact (not truncated, not partially overwritten)."""
        good_text = (
            "# T — Memory (v1)\n\n## State\nold state\n"
            "\n## Patterns\nold\n\n## Decisions\nold\n\n## Context\nold\n"
        )
        store.save_continuity(good_text)
        assert store.load_continuity() == good_text

        def exploding_replace(self, target):
            raise OSError("injected rename failure")

        monkeypatch.setattr(Path, "replace", exploding_replace)

        with pytest.raises(StoreError):
            store.save_continuity(
                "# T — Memory (v1)\n\n## State\nnew state\n"
                "\n## Patterns\nnew\n\n## Decisions\nnew\n\n## Context\nnew\n"
            )

        # Previous content survived verbatim.
        assert store.load_continuity() == good_text
        assert not list(store.continuity_path.parent.glob("*.md.tmp"))

    def test_save_continuity_cleans_tmp_on_non_oserror(
        self, store, monkeypatch
    ):
        """The atomic-write invariant MUST hold regardless of exception
        class. Inject a non-OSError failure and verify (a) the
        exception propagates bare (not wrapped as StoreError — we
        only wrap OSError), (b) the tmp file is still cleaned up."""

        # Real file gets created; write() then raises a non-OSError.
        # Use a sentinel subclass so we can catch it specifically
        # without accidentally masking real exceptions.
        class InjectedError(Exception):
            pass

        real_open = open

        def exploding_open(path, *args, **kwargs):
            f = real_open(path, *args, **kwargs)
            orig_write = f.write

            def fail_write(data):
                orig_write(data)  # partial write lands on disk
                raise InjectedError("not an OSError")

            f.write = fail_write
            return f

        monkeypatch.setattr("builtins.open", exploding_open)

        with pytest.raises(InjectedError):
            store.save_continuity(
                "# T — Memory (v1)\n\n## State\nx\n"
                "\n## Patterns\np\n\n## Decisions\nd\n\n## Context\nc\n"
            )

        # Critical: tmp cleanup ran even though exception was not an
        # OSError. Regression guard for I1 fix.
        assert not list(store.continuity_path.parent.glob("*.md.tmp"))
        # Stronger atomic-write invariant: on a first-write failure,
        # the main continuity file was never created. The replace
        # stage never ran because write raised before reaching it.
        assert not store.continuity_path.exists()

    def test_save_continuity_cleans_tmp_on_fsync_failure(
        self, store, monkeypatch
    ):
        """fsync() raises on flaky drives / network filesystems. Verify
        the cleanup path covers this scenario specifically.

        Intermediate assertion: the tmp file MUST be created before
        fsync fires (otherwise the 'no stale tmp' assertion at the
        bottom is vacuously true because there was never a tmp file
        to clean up). We capture the tmp file's existence at the
        moment fsync is called, then let the fault raise.
        """
        tmp_existed_at_fsync: list[bool] = []

        def exploding_fsync(fd):
            # At this point ``with open(...) as f`` has already
            # created the tmp file on disk. Snapshot existence
            # before raising so the assertion downstream is real.
            # 10.5c.5 L3 Fix #12 uses uuid-suffixed tmp filenames
            # (e.g. ``mystore.continuity.abc12345.md.tmp``), so we
            # glob for any matching tmp file rather than checking a
            # fixed path.
            tmps = list(
                store.continuity_path.parent.glob("*.md.tmp")
            )
            tmp_existed_at_fsync.append(len(tmps) > 0)
            raise OSError("injected fsync failure")

        monkeypatch.setattr(os, "fsync", exploding_fsync)

        with pytest.raises(StoreError) as exc_info:
            store.save_continuity(
                "# T — Memory (v1)\n\n## State\nx\n"
                "\n## Patterns\np\n\n## Decisions\nd\n\n## Context\nc\n"
            )
        assert exc_info.value.operation == "save_continuity"
        assert "injected fsync failure" in str(exc_info.value.__cause__)
        # Non-vacuous: the tmp file DID exist when fsync fired, and
        # it was cleaned up afterwards. If fsync had not been reached
        # for some reason, tmp_existed_at_fsync would be empty.
        assert tmp_existed_at_fsync == [True], (
            "fsync fault did not fire after tmp file was created — "
            "the cleanup assertion below would be vacuously true"
        )
        assert not list(store.continuity_path.parent.glob("*.md.tmp"))

    def test_save_continuity_tolerates_unlink_failure_in_cleanup(
        self, store, monkeypatch
    ):
        """The cleanup helper MUST swallow its own OSError so the
        primary exception is never masked. If unlink fails (e.g.
        permission denied), the primary StoreError still propagates
        and is what the user sees — not a confusing
        'secondary exception from cleanup' chain.

        Also verifies the secondary unlink failure does NOT leak into
        the raised exception's ``__context__`` chain. Python chains
        context exceptions by default; ``_safe_unlink``'s bare
        ``except OSError: pass`` should suppress context propagation.
        Catches a future refactor where someone adds logging in
        ``_safe_unlink`` that itself raises.
        """
        def exploding_replace(self, target):
            raise OSError("injected primary failure")

        def exploding_unlink(self, *, missing_ok=False):
            raise OSError("injected secondary unlink failure")

        monkeypatch.setattr(Path, "replace", exploding_replace)
        monkeypatch.setattr(Path, "unlink", exploding_unlink)

        with pytest.raises(StoreError) as exc_info:
            store.save_continuity(
                "# T — Memory (v1)\n\n## State\nx\n"
                "\n## Patterns\np\n\n## Decisions\nd\n\n## Context\nc\n"
            )
        # Primary error preserved — secondary unlink failure swallowed.
        assert "injected primary failure" in str(exc_info.value.__cause__)
        # Secondary unlink failure must NOT leak into the raised
        # exception's context chain. The __context__ should be the
        # original OSError (via 'raise ... from exc'), not the
        # secondary unlink OSError from the cleanup path.
        ctx = exc_info.value.__context__
        if ctx is not None:
            assert "injected secondary unlink failure" not in str(ctx), (
                "secondary unlink failure leaked into __context__ "
                "chain — _safe_unlink's suppression broke"
            )

    # -- save_meta failure paths --

    def test_save_meta_raises_store_error_on_replace_failure(
        self, store, monkeypatch
    ):
        def exploding_replace(self, target):
            raise OSError("injected meta rename failure")

        monkeypatch.setattr(Path, "replace", exploding_replace)

        with pytest.raises(StoreError) as exc_info:
            store.save_meta({"sessions_produced": 1})

        err = exc_info.value
        assert err.operation == "save_meta"
        assert err.path == str(store.meta_path)
        assert "injected meta rename failure" in str(err.__cause__)
        assert not list(store.meta_path.parent.glob("*.json.tmp"))

    def test_save_meta_cleans_tmp_on_type_error(
        self, store, monkeypatch
    ):
        """json.dump raises TypeError on non-JSON-serializable values
        (e.g. a set or a dataclass). This is not an OSError — it
        propagates bare — but the tmp cleanup MUST still run."""
        # Pass a set as a meta value — sets are not JSON-serializable.
        with pytest.raises(TypeError):
            store.save_meta({"sessions_produced": 1, "bad": {1, 2, 3}})

        assert not list(store.meta_path.parent.glob("*.json.tmp"))

    # -- happy path --

    def test_successful_save_continuity(self, store):
        path = store.save_continuity(
            "# T — Memory (v1)\n\n## State\nx\n"
            "\n## Patterns\np\n\n## Decisions\nd\n\n## Context\nc\n"
        )
        assert Path(path).exists()

    def test_successful_save_meta(self, store):
        path = store.save_meta({"sessions_produced": 3, "citations_seen": True})
        assert Path(path).exists()


class TestValidatedSaveContinuity:
    """Tests for the library-level validated_save_continuity function."""

    def test_validated_save_runs_full_pipeline(self, store):
        """validated_save_continuity should validate, save, form associations, and record wrap.

        Also serves as the integration test that graduation is not
        silently bypassed. Prior to Apr 10 2026 this test used a
        hardcoded ``2026-04-09`` date that drifted out of sync with
        the pipeline's internal wall-clock ``today``, silently turning
        any future 2x citation into a no-op (Diogenes Finding #3). Now
        uses ``date.today().isoformat()`` and cites a real episode so
        graduation actually runs.
        """
        from anneal_memory import validated_save_continuity

        today = date.today().isoformat()

        # Record some episodes first
        ep1 = store.record("Database is slow under load", EpisodeType.OBSERVATION)
        ep2 = store.record("Chose caching to improve latency", EpisodeType.DECISION)

        # Mark wrap as in progress

        # Build continuity with a real 2x citation so graduation fires
        text = (
            f"# Test — Memory (v1)\n\n"
            f"## State\nWorking on performance.\n\n"
            f"## Patterns\n"
            f"thought: database slow under load triggers caching"
            f" | 2x ({today})"
            f" [evidence: {ep1.id[:8]} \"database slow under load\"]\n\n"
            f"## Decisions\n"
            f"[decided(rationale: \"speed\", on: \"{today}\")] Use caching\n\n"
            f"## Context\nOptimizing database layer.\n"
        )

        result = validated_save_continuity(store, text)

        # Should have saved
        assert store.load_continuity() is not None
        assert result["path"] is not None
        assert result["episodes_compressed"] == 2

        # Graduation immune system must have actually fired — this is
        # the assertion Diogenes wanted: the integration path can no
        # longer silently bypass graduation validation.
        assert result["graduations_validated"] >= 1

        # Wrap should be recorded
        status = store.status()
        assert status.total_wraps >= 1
        assert not status.wrap_in_progress

        # Metadata should be updated
        meta = store.load_meta()
        assert meta["sessions_produced"] >= 1

    def test_validated_save_rejects_missing_sections(self, store):
        """Should raise ValueError for malformed continuity."""
        from anneal_memory import validated_save_continuity

        store.record("Something", EpisodeType.OBSERVATION)
        with pytest.raises(ValueError, match="4 sections"):
            validated_save_continuity(store, "Just some text without sections")

    def test_validated_save_rejects_empty_text(self, store):
        """Should raise ValueError for empty text."""
        from anneal_memory import validated_save_continuity

        with pytest.raises(ValueError, match="empty"):
            validated_save_continuity(store, "")

    def test_validated_save_with_affective_state(self, store):
        """Should accept and pass through affective state.

        Focus is affective state passthrough; graduation integration
        coverage lives in TestValidatedSaveContinuityReturnContract and
        test_validated_save_runs_full_pipeline above.
        """
        from anneal_memory import validated_save_continuity, AffectiveState

        today = date.today().isoformat()
        store.record("Interesting finding", EpisodeType.OBSERVATION)

        text = (
            "# Test — Memory (v1)\n\n"
            "## State\nExploring.\n\n"
            f"## Patterns\nthought: interesting | 1x ({today})\n\n"
            "## Decisions\nNone yet.\n\n"
            "## Context\nFirst session.\n"
        )

        affect = AffectiveState(tag="curious", intensity=0.8)
        result = validated_save_continuity(store, text, affective_state=affect)
        assert result["episodes_compressed"] == 1


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
        # Legacy no-arg form — this test only checks the
        # wrap_in_progress flag on status(), which reads
        # wrap_started_at directly and doesn't call
        # load_wrap_snapshot, so the partial state is tolerated.
        with pytest.warns(DeprecationWarning, match="legacy call form"):
            store.wrap_started()
        status = store.status()
        assert status.wrap_in_progress is True

    def test_status_after_wrap(self, store):
        store.record("Ep", EpisodeType.OBSERVATION)
        store.wrap_completed(episodes_compressed=1, continuity_chars=500)
        status = store.status()
        assert status.total_wraps == 1
        assert status.last_wrap_at is not None

    # -- Audit health visibility (Diogenes ARCH finding, Apr 13 2026) --

    def test_status_audit_enabled_exposes_health(self, store):
        """Default fixture has audit=True; status() must surface health."""
        store.record("trigger audit init", EpisodeType.OBSERVATION)
        status = store.status()
        assert status.audit_enabled is True
        assert status.audit_log_path is not None
        assert status.audit_log_path.endswith(".audit.jsonl")
        # At minimum the record() call above wrote one audit entry.
        assert status.audit_entry_count is not None
        assert status.audit_entry_count >= 1
        # Default retention is unlimited.
        assert status.audit_retention_days is None

    def test_status_audit_disabled_reports_cleanly(self, tmp_db):
        """audit=False: enabled flag false, all other audit fields None."""
        s = Store(tmp_db, project_name="NoAudit", audit=False)
        try:
            status = s.status()
            assert status.audit_enabled is False
            assert status.audit_log_path is None
            assert status.audit_entry_count is None
            assert status.audit_retention_days is None
        finally:
            s.close()

    def test_status_audit_retention_days_surfaced(self, tmp_db):
        """Retention window is part of the health snapshot."""
        s = Store(tmp_db, project_name="Retain", audit_retention_days=30)
        try:
            s.record("ep", EpisodeType.OBSERVATION)
            status = s.status()
            assert status.audit_enabled is True
            assert status.audit_retention_days == 30
        finally:
            s.close()

    def test_status_audit_count_reflects_activity(self, store):
        """Entry count should monotonically rise with write activity."""
        before = store.status().audit_entry_count or 0
        store.record("one", EpisodeType.OBSERVATION)
        store.record("two", EpisodeType.OBSERVATION)
        after = store.status().audit_entry_count
        assert after is not None
        assert after >= before + 2


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


# -- Wrap history (get_wrap_history + WrapRecord) --


def _valid_continuity_text(today: str, evidence: str = "") -> str:
    """Build a minimal valid 4-section continuity document.

    Uses dynamic today's date so graduation doesn't silently skip
    same-session citations. Optional evidence tag for graduation tests.
    """
    evidence_tag = f" [evidence: {evidence}]" if evidence else ""
    return (
        "# Test — Memory (v1)\n\n"
        "## State\nWorking.\n\n"
        f"## Patterns\nthought: pattern | 1x ({today}){evidence_tag}\n\n"
        "## Decisions\nNone.\n\n"
        "## Context\nFirst pass.\n"
    )


class TestGetWrapHistory:
    """Tests for Store.get_wrap_history() and the WrapRecord dataclass."""

    def test_empty_store_returns_empty_list(self, store):
        assert store.get_wrap_history() == []

    def test_returns_wrap_record_instances(self, store):
        """After a wrap, get_wrap_history should return list[WrapRecord]."""
        from anneal_memory import validated_save_continuity

        store.record("First observation", EpisodeType.OBSERVATION)
        validated_save_continuity(
            store, _valid_continuity_text(date.today().isoformat())
        )

        history = store.get_wrap_history()
        assert len(history) == 1
        assert isinstance(history[0], WrapRecord)
        assert is_dataclass(history[0])

    def test_wrap_record_fields_populated(self, store):
        """WrapRecord should surface all wrap metrics from the store."""
        from anneal_memory import validated_save_continuity

        store.record("Episode one", EpisodeType.OBSERVATION)
        store.record("Episode two", EpisodeType.DECISION)
        validated_save_continuity(
            store, _valid_continuity_text(date.today().isoformat())
        )

        record = store.get_wrap_history()[0]
        assert record.id == 1
        assert record.wrapped_at  # non-empty ISO timestamp
        assert record.episodes_compressed == 2
        # continuity_chars is non-None by type; also must be positive
        # because we just wrote a real continuity file
        assert record.continuity_chars > 0
        # Counter fields default to 0 when no graduation/association activity
        assert record.graduations_validated >= 0
        assert record.graduations_demoted >= 0
        assert record.citation_reuse_max >= 0
        assert record.patterns_extracted >= 0
        assert record.associations_formed >= 0
        assert record.associations_strengthened >= 0
        assert record.associations_decayed >= 0

    def test_multiple_wraps_ordered_by_id(self, store):
        """Wraps should be returned in chronological (id ASC) order."""
        from anneal_memory import validated_save_continuity

        today = date.today().isoformat()
        for i in range(3):
            store.record(f"Episode {i}", EpisodeType.OBSERVATION)
            validated_save_continuity(store, _valid_continuity_text(today))

        history = store.get_wrap_history()
        assert len(history) == 3
        ids = [w.id for w in history]
        assert ids == sorted(ids)
        assert ids[0] == 1

    def test_wrap_record_asdict_is_json_serializable(self, store):
        """asdict(WrapRecord) must round-trip through JSON.

        Used by cmd_export and cmd_history JSON output paths.
        """
        from anneal_memory import validated_save_continuity

        store.record("Observation", EpisodeType.OBSERVATION)
        validated_save_continuity(
            store, _valid_continuity_text(date.today().isoformat())
        )

        record = store.get_wrap_history()[0]
        d = asdict(record)
        assert isinstance(d, dict)
        # Round-trip through JSON should not raise
        serialized = json.dumps(d)
        restored = json.loads(serialized)
        assert restored["id"] == record.id
        assert restored["episodes_compressed"] == record.episodes_compressed

    def test_wrap_record_is_frozen(self):
        """WrapRecord should be frozen (immutable) like other value types."""
        record = WrapRecord(
            id=1,
            wrapped_at="2026-01-01T00:00:00Z",
            episodes_compressed=5,
            continuity_chars=1000,
            graduations_validated=2,
            graduations_demoted=1,
            citation_reuse_max=3,
            patterns_extracted=4,
            associations_formed=5,
            associations_strengthened=6,
            associations_decayed=7,
        )
        with pytest.raises((AttributeError, Exception)):
            record.id = 99  # type: ignore[misc]


class TestValidatedSaveContinuityReturnContract:
    """Tests for the widened validated_save_continuity return dict.

    The library function is the canonical save pipeline — its return
    contract must expose everything MCP and CLI transports need to
    format their output. Previously the dict was missing bare_demoted,
    gaming_suspects, citation_reuse_max, and sections.
    """

    def test_return_dict_has_all_widened_keys(self, store):
        from anneal_memory import validated_save_continuity

        store.record("Observation A", EpisodeType.OBSERVATION)

        result = validated_save_continuity(
            store, _valid_continuity_text(date.today().isoformat())
        )

        # Core pipeline results
        assert "path" in result
        assert "chars" in result  # top-level convenience key (10.5c.1 review fix)
        assert "episodes_compressed" in result
        assert "wrap_result" in result
        assert "skipped_prepare" in result
        # Graduation breakdown (the Diogenes Finding #1 fix)
        assert "graduations_validated" in result
        assert "graduations_demoted" in result
        assert "demoted" in result
        assert "bare_demoted" in result
        assert "citation_reuse_max" in result
        assert "gaming_suspects" in result
        # Association metrics
        assert "associations_formed" in result
        assert "associations_strengthened" in result
        assert "associations_decayed" in result
        # Section measurement (used by CLI text output)
        assert "sections" in result
        assert isinstance(result["sections"], dict)

    def test_chars_top_level_matches_wrap_result(self, store):
        """Top-level chars must match wrap_result["chars"] and the actual save.

        The chars key was added in the 10.5c.1 review fix to remove a
        footgun where transports had to reach into the nested
        wrap_result for a single field while every other metric was
        already flattened. As of 10.5c.3 ``wrap_result`` is a plain
        dict (``asdict(WrapResult)``) so the whole return value is
        JSON-serializable top-to-bottom — access is via subscript,
        not attribute.
        """
        from anneal_memory import validated_save_continuity

        store.record("Observation", EpisodeType.OBSERVATION)
        result = validated_save_continuity(
            store, _valid_continuity_text(date.today().isoformat())
        )

        assert result["chars"] == result["wrap_result"]["chars"]
        # And both should equal the saved file size
        from pathlib import Path
        assert result["chars"] == len(
            Path(result["path"]).read_text(encoding="utf-8")
        )

    def test_wrap_result_section_sizes_is_populated(self, store):
        """wrap_result["section_sizes"] must contain the real section data.

        Prior to the 10.5c.1 cleanup, ``wrap_completed()`` passed
        ``section_sizes={}`` unconditionally, so library users who
        accessed ``result["wrap_result"]["section_sizes"]`` got an
        empty dict — the type definition advertised a field that was
        structurally always empty. The fix wires ``section_sizes``
        through from ``validated_save_continuity`` to
        ``wrap_completed``. As of 10.5c.3 ``wrap_result`` is a plain
        dict via ``asdict(WrapResult)``; access is subscript, not
        attribute.
        """
        from anneal_memory import validated_save_continuity

        store.record("Observation", EpisodeType.OBSERVATION)
        result = validated_save_continuity(
            store, _valid_continuity_text(date.today().isoformat())
        )

        wrap_result = result["wrap_result"]
        section_sizes = wrap_result["section_sizes"]
        # Populated, not the empty dict of the pre-fix era
        assert section_sizes
        assert isinstance(section_sizes, dict)
        # Must contain the 4 canonical section names (plus _header)
        assert "State" in section_sizes
        assert "Patterns" in section_sizes
        assert "Decisions" in section_sizes
        assert "Context" in section_sizes
        # Must match the top-level sections key (same measurement,
        # two paths to it)
        assert section_sizes == result["sections"]

    def test_today_parameter_pins_graduation_date(self, store):
        """Passing an explicit ``today`` pins graduation to that date.

        This removes wall-clock dependency from tests and enables
        reproducible experiment runs. Without the pin, graduation
        uses ``date.today()`` internally and a test written with a
        specific date in the continuity text would silently skip
        citation validation as time drifts — the exact Diogenes
        Finding #3 class of bug.
        """
        from anneal_memory import validated_save_continuity

        # Use a pinned date that is NOT today's wall clock date.
        # With the today parameter, graduation must still fire
        # because the citation date matches the pinned value.
        pinned_today = "2026-06-15"

        ep1 = store.record("database slow under load", EpisodeType.OBSERVATION)

        text = (
            f"# Test — Memory (v1)\n\n"
            f"## State\nPinned-date test.\n\n"
            f"## Patterns\n"
            f"thought: slow database impacts throughput"
            f" | 2x ({pinned_today})"
            f" [evidence: {ep1.id[:8]} \"database slow under load\"]\n\n"
            f"## Decisions\n"
            f"[decided(rationale: \"speed\", on: \"{pinned_today}\")] "
            f"Use caching\n\n"
            f"## Context\nPinned-date determinism test.\n"
        )

        result = validated_save_continuity(store, text, today=pinned_today)

        # Graduation fired even though the citation date isn't today's
        # wall-clock date, because the pipeline used the pinned value.
        assert result["graduations_validated"] >= 1

    def test_today_parameter_defaults_to_wall_clock(self, store):
        """Omitting ``today`` falls back to ``date.today().isoformat()``."""
        from anneal_memory import validated_save_continuity

        ep1 = store.record("fresh observation today", EpisodeType.OBSERVATION)
        today = date.today().isoformat()

        text = (
            f"# Test — Memory (v1)\n\n"
            f"## State\nDefault date test.\n\n"
            f"## Patterns\n"
            f"thought: fresh observation drives today's decision"
            f" | 2x ({today})"
            f" [evidence: {ep1.id[:8]} \"fresh observation today\"]\n\n"
            f"## Decisions\nNone.\n\n"
            f"## Context\nDefault wall-clock path.\n"
        )

        # No today= parameter → falls back to wall clock
        result = validated_save_continuity(store, text)
        assert result["graduations_validated"] >= 1

    def test_graduations_demoted_includes_bare_demoted(self, store):
        """graduations_demoted must equal demoted + bare_demoted.

        This is the exact divergence Diogenes caught: the library used
        grad_result.demoted only, while MCP and CLI used
        demoted + bare_demoted. After Session 10.5c.1 the library is
        canonical — all three produce the same wrap metric.
        """
        from anneal_memory import validated_save_continuity

        # Set citations_seen=True so bare graduations get demoted
        meta = store.load_meta()
        meta["citations_seen"] = True
        store.save_meta(meta)

        today = date.today().isoformat()

        # Record one episode so there's at least something to compress
        store.record("Triggering observation", EpisodeType.OBSERVATION)

        # Build a continuity with a bare 2x graduation (no evidence tag)
        # — should be demoted due to citations_seen=True
        text = (
            "# Test — Memory (v1)\n\n"
            "## State\nWorking.\n\n"
            "## Patterns\n"
            f"thought: unsupported claim | 2x ({today})\n\n"
            "## Decisions\nNone.\n\n"
            "## Context\nFirst pass.\n"
        )

        result = validated_save_continuity(store, text)

        # graduations_demoted is the combined total
        assert result["graduations_demoted"] == (
            result["demoted"] + result["bare_demoted"]
        )
        # At least one bare demotion should have fired
        assert result["bare_demoted"] >= 1

    def test_wrap_metric_matches_return_dict(self, store):
        """The wraps table row must match the returned graduations_demoted.

        If the library under-reports to wrap_completed() but over-reports
        in the return dict (or vice versa), downstream history/diff/stats
        CLI output diverges from what the caller saw at save time.
        """
        from anneal_memory import validated_save_continuity

        # citations_seen=True so bare grads get demoted
        meta = store.load_meta()
        meta["citations_seen"] = True
        store.save_meta(meta)

        store.record("Observation", EpisodeType.OBSERVATION)
        today = date.today().isoformat()
        text = (
            "# Test — Memory (v1)\n\n"
            "## State\nWorking.\n\n"
            "## Patterns\n"
            f"thought: bare claim | 2x ({today})\n\n"
            "## Decisions\nNone.\n\n"
            "## Context\nFirst pass.\n"
        )
        result = validated_save_continuity(store, text)

        history = store.get_wrap_history()
        assert len(history) == 1
        assert history[0].graduations_demoted == result["graduations_demoted"]


# -- 10.5c.6 SQLite error wrapping (Store._db_boundary) --


class TestDbBoundaryErrorWrapping:
    """Every public Store method that touches SQLite must surface
    failures as ``StoreDatabaseError`` via :meth:`Store._db_boundary`
    instead of leaking bare ``sqlite3.Error`` subclasses.

    Pattern: monkeypatch ``store._conn.execute`` (or a specific helper)
    to raise ``sqlite3.OperationalError``, call the public method, and
    assert the raised exception is a StoreDatabaseError with the right
    ``operation`` field, that it's still catchable as ``StoreError`` and
    ``AnnealMemoryError``, and that ``__cause__`` preserves the original
    sqlite3 exception.
    """

    # -- helpers -------------------------------------------------

    @staticmethod
    def _flaky_execute(
        store,
        monkeypatch,
        *,
        message: str = "injected",
        exc_cls=None,
    ) -> None:
        """Swap ``store._conn`` with a proxy that raises on execute().

        ``sqlite3.Connection.execute`` is a C-level method and
        ``monkeypatch.setattr(store._conn, "execute", ...)`` fails with
        ``AttributeError: ... is read-only`` — the same constraint that
        drove the ``FlakyCommitProxy`` pattern in
        ``tests/test_continuity.py::test_batch_commit_failure_rolls_back``.

        The proxy delegates every attribute lookup except ``execute``
        (which raises a configurable ``sqlite3.Error`` subclass).
        Restored via ``monkeypatch.setattr`` so cleanup is automatic
        at test teardown; the fixture's ``store.close()`` still runs
        on the real connection.
        """
        import sqlite3

        if exc_cls is None:
            exc_cls = sqlite3.OperationalError

        class _FlakyExecuteProxy:
            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            def execute(self, *args, **kwargs):
                raise exc_cls(message)

        monkeypatch.setattr(store, "_conn", _FlakyExecuteProxy(store._conn))

    def _assert_wrapped(
        self,
        exc: StoreDatabaseError,
        expected_operation: str,
    ) -> None:
        """Common assertion bundle for a wrapped error."""
        import sqlite3

        assert isinstance(exc, StoreDatabaseError)
        assert isinstance(exc, StoreError)
        assert isinstance(exc, AnnealMemoryError)
        assert exc.operation == expected_operation
        assert exc.path is not None
        assert isinstance(exc.__cause__, sqlite3.Error)

    # -- record --------------------------------------------------

    def test_record_wraps_sqlite_error(self, store, monkeypatch):
        self._flaky_execute(store, monkeypatch, message="record blew up")
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.record("boom", EpisodeType.OBSERVATION)
        self._assert_wrapped(exc_info.value, "record")
        assert "record blew up" in str(exc_info.value.__cause__)

    # -- get -----------------------------------------------------

    def test_get_wraps_sqlite_error(self, store, monkeypatch):
        self._flaky_execute(store, monkeypatch)
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.get("deadbeef")
        self._assert_wrapped(exc_info.value, "get")

    # -- delete --------------------------------------------------

    def test_delete_wraps_sqlite_error(self, store, monkeypatch):
        store.record("to be deleted", EpisodeType.OBSERVATION)
        self._flaky_execute(store, monkeypatch)
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.delete("deadbeef")
        self._assert_wrapped(exc_info.value, "delete")

    # -- recall --------------------------------------------------

    def test_recall_wraps_sqlite_error(self, store, monkeypatch):
        store.record("ep", EpisodeType.OBSERVATION)
        self._flaky_execute(store, monkeypatch)
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.recall(limit=5)
        self._assert_wrapped(exc_info.value, "recall")

    # -- episodes_since_wrap ------------------------------------

    def test_episodes_since_wrap_wraps_sqlite_error(self, store, monkeypatch):
        self._flaky_execute(store, monkeypatch)
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.episodes_since_wrap()
        self._assert_wrapped(exc_info.value, "episodes_since_wrap")

    # -- status --------------------------------------------------

    def test_status_wraps_sqlite_error(self, store, monkeypatch):
        self._flaky_execute(store, monkeypatch)
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.status()
        self._assert_wrapped(exc_info.value, "status")

    # -- wrap lifecycle ------------------------------------------

    def test_wrap_started_wraps_sqlite_error(self, store, monkeypatch):
        self._flaky_execute(store, monkeypatch)
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.wrap_started(token="a" * 32, episode_ids=[])
        self._assert_wrapped(exc_info.value, "wrap_started")

    def test_wrap_cancelled_wraps_sqlite_error(self, store, monkeypatch):
        self._flaky_execute(store, monkeypatch)
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.wrap_cancelled()
        self._assert_wrapped(exc_info.value, "wrap_cancelled")

    def test_wrap_completed_wraps_sqlite_error(self, store, monkeypatch):
        self._flaky_execute(store, monkeypatch)
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.wrap_completed(episodes_compressed=0, continuity_chars=0)
        self._assert_wrapped(exc_info.value, "wrap_completed")

    def test_get_wrap_started_at_wraps_sqlite_error(self, store, monkeypatch):
        self._flaky_execute(store, monkeypatch)
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.get_wrap_started_at()
        self._assert_wrapped(exc_info.value, "get_wrap_started_at")

    def test_load_wrap_snapshot_wraps_sqlite_error(self, store, monkeypatch):
        self._flaky_execute(store, monkeypatch)
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.load_wrap_snapshot()
        self._assert_wrapped(exc_info.value, "load_wrap_snapshot")

    def test_get_wrap_history_wraps_sqlite_error(self, store, monkeypatch):
        # "no such table" is still swallowed (returns []); other
        # OperationalErrors must surface as StoreDatabaseError.
        self._flaky_execute(store, monkeypatch, message="database is locked")
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.get_wrap_history()
        self._assert_wrapped(exc_info.value, "get_wrap_history")

    def test_get_wrap_history_still_swallows_no_such_table(
        self, store, monkeypatch
    ):
        """Regression guard — 10.5c.6 must NOT change the legacy
        ``no such table: wraps`` swallow behavior. That path stays
        silent."""
        self._flaky_execute(
            store, monkeypatch, message="no such table: wraps"
        )
        result = store.get_wrap_history()
        assert result == []

    def test_get_wrap_history_does_not_swallow_other_missing_table(
        self, store, monkeypatch
    ):
        """10.5c.6 L1 H1 — the ``no such table`` swallow is now
        tightened to ``no such table: wraps`` specifically. A corrupt
        DB missing a DIFFERENT table (e.g. ``episodes``) must NOT
        silently return [] — that's a real store integrity failure
        that should surface as StoreDatabaseError.
        """
        self._flaky_execute(
            store, monkeypatch, message="no such table: episodes"
        )
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.get_wrap_history()
        self._assert_wrapped(exc_info.value, "get_wrap_history")

    # -- associations --------------------------------------------

    def test_get_associations_wraps_sqlite_error(self, store, monkeypatch):
        self._flaky_execute(store, monkeypatch)
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.get_associations(["deadbeef"])
        self._assert_wrapped(exc_info.value, "get_associations")

    def test_get_association_context_wraps_sqlite_error(
        self, store, monkeypatch
    ):
        self._flaky_execute(store, monkeypatch)
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.get_association_context(["deadbeef"])
        self._assert_wrapped(exc_info.value, "get_association_context")

    def test_record_associations_wraps_sqlite_error(
        self, store, monkeypatch
    ):
        # Need an existing episode to form a real pair shape. Record
        # before flaky-execute so the setup INSERT actually runs.
        ep_a = store.record("source a", EpisodeType.OBSERVATION)
        ep_b = store.record("source b", EpisodeType.OBSERVATION)
        self._flaky_execute(store, monkeypatch)
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.record_associations(
                direct_pairs={(ep_a.id, ep_b.id)},
            )
        self._assert_wrapped(exc_info.value, "record_associations")

    def test_decay_associations_wraps_sqlite_error(
        self, store, monkeypatch
    ):
        self._flaky_execute(store, monkeypatch)
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.decay_associations(strengthened_pairs=set())
        self._assert_wrapped(exc_info.value, "decay_associations")

    def test_association_stats_wraps_sqlite_error(self, store, monkeypatch):
        self._flaky_execute(store, monkeypatch)
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.association_stats()
        self._assert_wrapped(exc_info.value, "association_stats")

    # -- prune ---------------------------------------------------

    def test_prune_wraps_sqlite_error(self, store, monkeypatch):
        self._flaky_execute(store, monkeypatch)
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.prune(older_than_days=0)
        self._assert_wrapped(exc_info.value, "prune")

    # -- integrity of the hierarchy -----------------------------

    def test_store_error_is_still_the_catch_point(self, store, monkeypatch):
        """The whole point of making StoreDatabaseError a subclass of
        StoreError is that every existing transport handler continues
        to work unchanged. Regression guard."""
        self._flaky_execute(store, monkeypatch)
        with pytest.raises(StoreError):
            store.status()

    def test_anneal_memory_error_is_still_the_catch_point(
        self, store, monkeypatch
    ):
        self._flaky_execute(store, monkeypatch)
        with pytest.raises(AnnealMemoryError):
            store.status()

    def test_cause_chain_preserves_original_sqlite_subclass(
        self, store, monkeypatch
    ):
        """``__cause__`` must still be the exact sqlite3 subclass so
        callers with errno-level needs can drill down. The record()
        method retries on IntegrityError (ID collision) up to 3 times;
        after the final retry exhausts, the error propagates out of
        the retry loop and into _db_boundary."""
        import sqlite3

        self._flaky_execute(
            store,
            monkeypatch,
            message="UNIQUE constraint failed",
            exc_cls=sqlite3.IntegrityError,
        )
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.record("anything", EpisodeType.OBSERVATION)
        assert isinstance(exc_info.value.__cause__, sqlite3.IntegrityError)
        assert exc_info.value.operation == "record"

    def test_interface_error_propagates_bare_not_wrapped(
        self, store, monkeypatch
    ):
        """10.5c.6 L2 #2 — ``_db_boundary`` catches
        ``sqlite3.DatabaseError``, not the broader ``sqlite3.Error``,
        so ``sqlite3.InterfaceError`` (API misuse, a programming bug
        in the library or caller) propagates bare instead of being
        mis-wrapped as a 'retryable DB error.' This is load-bearing
        for retry-loop correctness in downstream consumers that
        treat StoreDatabaseError as transient.
        """
        import sqlite3

        self._flaky_execute(
            store,
            monkeypatch,
            message="bindings must be a sequence",
            exc_cls=sqlite3.InterfaceError,
        )
        with pytest.raises(sqlite3.InterfaceError):
            store.status()
        # And NOT StoreDatabaseError — verify the class specifically.
        self._flaky_execute(
            store,
            monkeypatch,
            message="bindings must be a sequence",
            exc_cls=sqlite3.InterfaceError,
        )
        try:
            store.status()
        except sqlite3.InterfaceError as err:
            assert not isinstance(err, StoreDatabaseError)
            assert not isinstance(err, StoreError)
            assert not isinstance(err, AnnealMemoryError)

    def test_wrap_completed_cas_path_wraps_sqlite_error(
        self, store, monkeypatch
    ):
        """10.5c.6 L1 L1 — the wrap_completed _db_boundary must
        cover the CAS UPDATE branch (``if wrap_token is not None``),
        not just the insert/update path. Seed a real wrap_token via
        ``wrap_started`` so the CAS branch executes, then flaky-
        execute to force the CAS UPDATE itself to fail.
        """
        store.wrap_started(token="a" * 32, episode_ids=[])
        self._flaky_execute(store, monkeypatch)
        with pytest.raises(StoreDatabaseError) as exc_info:
            store.wrap_completed(
                episodes_compressed=0,
                continuity_chars=0,
                wrap_token="a" * 32,
            )
        self._assert_wrapped(exc_info.value, "wrap_completed")

    def test_close_is_idempotent_after_success(self, tmp_db):
        """10.5c.6 L3 complement F4 — after a successful close,
        ``self._closed`` is True so subsequent close() calls
        early-return (no spurious ProgrammingError wrapped as
        StoreDatabaseError). Regression guard for the idempotency
        invariant."""
        s = Store(tmp_db, project_name="Idempotent")
        s.close()
        # Second and third calls must succeed silently.
        s.close()
        s.close()
        assert s._closed is True

    def test_post_close_operations_raise_store_error_not_attributeerror(
        self, tmp_db
    ):
        """10.5c.6 L3 codex H1 — after close(), calling any public
        method must surface as ``StoreError`` with the caller's
        operation name, NOT a bare ``AttributeError`` from
        ``self._conn.execute(...)``. Regression guard for the
        hierarchy-consistency contract."""
        s = Store(tmp_db, project_name="PostClose")
        s.close()

        # Every method that goes through _db_boundary should now
        # raise StoreError (the base class, not StoreDatabaseError —
        # closed-store state is a library-state error, not a DB
        # runtime error). Verify the catchpoint works for a
        # representative sample covering read + write + wrap ops.
        with pytest.raises(StoreError) as exc_info:
            s.status()
        assert "closed store" in str(exc_info.value)
        assert exc_info.value.operation == "status"

        with pytest.raises(StoreError) as exc_info:
            s.record("after close", EpisodeType.OBSERVATION)
        assert exc_info.value.operation == "record"

        with pytest.raises(StoreError) as exc_info:
            s.get("deadbeef")
        assert exc_info.value.operation == "get"

        with pytest.raises(StoreError) as exc_info:
            s.recall(limit=5)
        assert exc_info.value.operation == "recall"

        with pytest.raises(StoreError) as exc_info:
            s.wrap_cancelled()
        assert exc_info.value.operation == "wrap_cancelled"

        # Critically: none of these should be AttributeError.
        # A sanity check that the error type is right.
        try:
            s.prune(older_than_days=0)
        except AttributeError:  # pragma: no cover — regression guard
            pytest.fail(
                "prune() after close() raised AttributeError — "
                "this means the boundary's _closed pre-check is "
                "not firing and the hierarchy contract is broken."
            )
        except StoreError as exc:
            assert exc.operation == "prune"

    def test_close_wraps_sqlite_error(self, tmp_db, monkeypatch):
        """10.5c.6 L1 L4 — ``close()`` is now covered by
        _db_boundary. A failure during close surfaces as
        StoreDatabaseError with ``operation="close"`` instead of a
        bare sqlite3 error leaking out of a context manager exit."""
        s = Store(tmp_db, project_name="CloseFail")
        self._flaky_execute(s, monkeypatch, message="close failed")
        # The proxy's execute is swapped, but close() calls
        # self._conn.close(), not execute — so we need a proxy for
        # close too. Extend the proxy inline.
        import sqlite3

        real_conn = s._conn._real  # type: ignore[attr-defined]

        class CloseFailProxy:
            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            def close(self):
                raise sqlite3.OperationalError("close failed")

        s._conn = CloseFailProxy(real_conn)  # type: ignore[assignment]
        try:
            with pytest.raises(StoreDatabaseError) as exc_info:
                s.close()
            self._assert_wrapped(exc_info.value, "close")
        finally:
            # Clean up the real connection so the test doesn't leak.
            real_conn.close()

    def test_store_database_error_pickle_roundtrip(self):
        """The subclass must round-trip cleanly through pickle with
        its identity, context fields, and cause_type_name preserved.
        Tests the 10.5c.6-refactored single generic reconstructor.
        """
        import copy
        import pickle

        err = StoreDatabaseError(
            "SQLite record failed on /tmp/test.db: injected",
            operation="record",
            path="/tmp/test.db",
            cause_type_name="OperationalError",
        )

        for round_trip in (pickle.loads(pickle.dumps(err)), copy.copy(err), copy.deepcopy(err)):
            assert isinstance(round_trip, StoreDatabaseError)
            assert isinstance(round_trip, StoreError)  # hierarchy preserved
            assert isinstance(round_trip, AnnealMemoryError)
            assert round_trip.operation == "record"
            assert round_trip.path == "/tmp/test.db"
            assert round_trip.cause_type_name == "OperationalError"
            assert "injected" in str(round_trip)

    def test_store_error_parent_class_pickle_roundtrip(self):
        """The parent class must ALSO round-trip cleanly through the
        single generic reconstructor — regression guard that the
        10.5c.6 refactor didn't break StoreError pickle.
        """
        import pickle

        err = StoreError(
            "Failed to write continuity to /tmp/test.db: disk full",
            operation="save_continuity",
            path="/tmp/test.db",
        )
        round_trip = pickle.loads(pickle.dumps(err))
        assert type(round_trip) is StoreError
        assert not isinstance(round_trip, StoreDatabaseError)
        assert round_trip.operation == "save_continuity"
        assert round_trip.path == "/tmp/test.db"
        assert round_trip.cause_type_name is None

    def test_cause_type_name_populated_by_boundary(self, store, monkeypatch):
        """10.5c.6 L2 #5 + L3 F7 — the _db_boundary must populate
        cause_type_name with the sqlite3 exception class name at
        raise time. Cross-process consumers rely on this field for
        retry decisions after pickle drops __cause__.
        """
        import sqlite3

        self._flaky_execute(
            store, monkeypatch, exc_cls=sqlite3.OperationalError
        )
        try:
            store.status()
            assert False, "expected StoreDatabaseError"
        except StoreDatabaseError as err:
            assert err.cause_type_name == "OperationalError"

        self._flaky_execute(
            store, monkeypatch, exc_cls=sqlite3.IntegrityError
        )
        try:
            store.status()
            assert False, "expected StoreDatabaseError"
        except StoreDatabaseError as err:
            assert err.cause_type_name == "IntegrityError"

    def test_is_retryable_helper_works_post_pickle(self):
        """10.5c.6 L3 F7 — the is_retryable() helper shown in the
        docs must work BOTH in-process (with live __cause__) AND
        after pickle (where __cause__ is dropped) via the
        cause_type_name field. Regression guard for the docs snippet.
        """
        import pickle

        # Simulate what _db_boundary produces.
        err = StoreDatabaseError(
            "SQLite record failed on /tmp: database is locked",
            operation="record",
            path="/tmp",
            cause_type_name="OperationalError",
        )

        def is_retryable(e):
            if e.cause_type_name != "OperationalError":
                return False
            msg = str(e).lower()
            return "locked" in msg or "busy" in msg

        # In-process: should be retryable.
        assert is_retryable(err) is True

        # After pickle: must STILL be retryable (the whole point).
        round_trip = pickle.loads(pickle.dumps(err))
        assert is_retryable(round_trip) is True

        # Non-retryable cases also survive pickle correctly.
        integrity = StoreDatabaseError(
            "SQLite record failed on /tmp: UNIQUE constraint failed",
            operation="record",
            path="/tmp",
            cause_type_name="IntegrityError",
        )
        assert is_retryable(integrity) is False
        assert is_retryable(pickle.loads(pickle.dumps(integrity))) is False

        disk_full = StoreDatabaseError(
            "SQLite record failed on /tmp: disk I/O error",
            operation="record",
            path="/tmp",
            cause_type_name="OperationalError",
        )
        # disk I/O error is OperationalError but NOT locked/busy
        assert is_retryable(disk_full) is False
        assert is_retryable(pickle.loads(pickle.dumps(disk_full))) is False

    def test_store_operation_literal_has_no_drift(self):
        """10.5c.6 L2 #3 + codex L3.5 — the StoreOperation Literal
        is a soft contract without mypy-in-CI. Codex already caught
        one drift (prepare_continuity_write / prepare_meta_write
        raise sites missing from the Literal). This structural test
        catches future drift in BOTH directions:

        (a) New raise sites with an ``operation=`` string that's not
            in the Literal (the codex-class failure mode).
        (b) Literal values that no raise site uses (dead enum members
            that decay to noise).

        Enforcement is grep-based on the store.py source. A future
        refactor that adds a new public method touching SQLite
        without adding the operation value to the Literal fails THIS
        test at CI time, before the code ships. Turns the
        discipline-based contract into a structural invariant.
        """
        import re
        from typing import get_args

        from anneal_memory.store import StoreOperation

        source = Path(
            "anneal_memory/store.py"
        ).read_text(encoding="utf-8")

        literal_values = set(get_args(StoreOperation))

        # Collect every operation="..." or _db_boundary("...") string
        # literal that appears in the source. We accept both single
        # and double quotes and allow optional whitespace.
        used = set()
        for match in re.finditer(
            r'_db_boundary\(\s*["\']([^"\']+)["\']',
            source,
        ):
            used.add(match.group(1))
        for match in re.finditer(
            r'operation\s*=\s*["\']([^"\']+)["\']',
            source,
        ):
            used.add(match.group(1))

        # Drift direction (a): every used operation must be in the
        # Literal. This catches the codex-class failure mode.
        missing_from_literal = used - literal_values
        assert not missing_from_literal, (
            f"Raise sites use operation values not in StoreOperation "
            f"Literal: {sorted(missing_from_literal)}. Add them to "
            f"the Literal in store.py (the alias is a soft contract "
            f"but drift is still a bug — pickled errors from these "
            f"raise sites would have operation fields outside the "
            f"type, breaking exhaustive-match consumers)."
        )

        # Drift direction (b): every Literal value must appear as a
        # raise site. Dead enum members indicate the Literal has
        # drifted ahead of the code. Allow a small opt-out set for
        # values that are reserved but not-yet-used (currently none).
        reserved_for_future = set()  # Empty for now
        unused_literal_values = (
            literal_values - used - reserved_for_future
        )
        assert not unused_literal_values, (
            f"StoreOperation Literal has dead entries with no raise "
            f"site: {sorted(unused_literal_values)}. Either add a "
            f"raise site that uses them or remove them from the "
            f"Literal. If intentionally reserved for an upcoming "
            f"session, add to ``reserved_for_future`` in this test "
            f"with a comment explaining the timing."
        )

    # -- schema_init --------------------------------------------

    def test_schema_init_wraps_executescript_error(
        self, tmp_path, monkeypatch
    ):
        """10.5c.6 L3 contrarian F3 — the _FlakyExecuteProxy covers
        ``execute`` but ``_init_schema`` uses ``executescript``. The
        _db_boundary catches ``sqlite3.DatabaseError`` regardless of
        which connection method raised it, but a dedicated test
        proves this (and catches future refactors that move
        executescript out of the boundary).
        """
        import sqlite3

        real_connect = sqlite3.connect

        class FailingScriptConn:
            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            def executescript(self, *args, **kwargs):
                raise sqlite3.DatabaseError(
                    "malformed database schema"
                )

        def wrapping_connect(*args, **kwargs):
            real_conn = real_connect(*args, **kwargs)
            return FailingScriptConn(real_conn)

        monkeypatch.setattr(
            "anneal_memory.store.sqlite3.connect", wrapping_connect
        )

        with pytest.raises(StoreDatabaseError) as exc_info:
            Store(str(tmp_path / "bad_schema.db"), project_name="BadSchema")

        self._assert_wrapped(exc_info.value, "schema_init")
        assert "malformed" in str(exc_info.value.__cause__)

    def test_schema_init_wraps_sqlite_connect_error(
        self, tmp_path, monkeypatch
    ):
        """If ``sqlite3.connect`` itself fails during Store(), the
        error surfaces as StoreDatabaseError with
        ``operation="schema_init"`` rather than leaking a bare
        sqlite3 error from the constructor. Regression guard for the
        constructor wrapping."""
        import sqlite3

        real_connect = sqlite3.connect

        def failing_connect(*args, **kwargs):
            raise sqlite3.OperationalError("unable to open database file")

        monkeypatch.setattr("anneal_memory.store.sqlite3.connect", failing_connect)

        with pytest.raises(StoreDatabaseError) as exc_info:
            Store(str(tmp_path / "wont_exist.db"), project_name="Fail")

        self._assert_wrapped(exc_info.value, "schema_init")
        assert "unable to open" in str(exc_info.value.__cause__)

    # -- real database-locked integration smoke test -------------

    def test_real_database_locked_surfaces_as_store_database_error(
        self, tmp_path
    ):
        """Integration smoke test with an actual locked DB (no
        monkeypatching). Two Store instances share a path; Store A
        takes an EXCLUSIVE write lock; Store B attempts a write and
        must surface StoreDatabaseError (not bare sqlite3 error)."""
        import sqlite3

        db_path = str(tmp_path / "locked.db")
        store_a = Store(db_path, project_name="Locker")
        store_b = Store(db_path, project_name="Locker")

        try:
            # Drop Store B's busy_timeout so it fails fast instead of
            # hanging the test on the default 5s retry window.
            store_b._conn.execute("PRAGMA busy_timeout=50")

            # Start an EXCLUSIVE transaction on store_a
            store_a._conn.execute("BEGIN EXCLUSIVE")
            try:
                with pytest.raises(StoreDatabaseError) as exc_info:
                    store_b.record("blocked", EpisodeType.OBSERVATION)
                self._assert_wrapped(exc_info.value, "record")
                assert "lock" in str(exc_info.value.__cause__).lower()
            finally:
                store_a._conn.execute("ROLLBACK")
        finally:
            store_a.close()
            store_b.close()
