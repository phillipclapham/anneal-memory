"""Tests for the hash-chained JSONL audit trail."""

import gzip
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from anneal_memory.audit import GENESIS_HASH, AuditTrail, AuditVerifyResult


class TestAuditBasics:
    """Basic audit trail operations."""

    def test_log_creates_file(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)
        trail.log("record", {"episode_id": "abc123"})

        audit_path = tmp_path / "test.audit.jsonl"
        assert audit_path.exists()

    def test_log_returns_entry(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)
        entry = trail.log("record", {"episode_id": "abc123"})

        assert entry["v"] == 1
        assert entry["seq"] == 0
        assert entry["event"] == "record"
        assert entry["prev_hash"] == GENESIS_HASH
        assert entry["data"]["episode_id"] == "abc123"
        assert "ts" in entry

    def test_sequential_seq_numbers(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)

        e0 = trail.log("record", {"id": "1"})
        e1 = trail.log("record", {"id": "2"})
        e2 = trail.log("record", {"id": "3"})

        assert e0["seq"] == 0
        assert e1["seq"] == 1
        assert e2["seq"] == 2

    def test_hash_chain_links(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)

        e0 = trail.log("record", {"id": "1"})
        e1 = trail.log("record", {"id": "2"})

        assert e0["prev_hash"] == GENESIS_HASH
        assert e1["prev_hash"] != GENESIS_HASH
        assert e1["prev_hash"].startswith("sha256:")

    def test_deterministic_serialization(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)
        trail.log("record", {"z_key": "last", "a_key": "first"})

        audit_path = tmp_path / "test.audit.jsonl"
        line = audit_path.read_text(encoding="utf-8").strip()

        # Keys should be sorted in the JSON
        parsed = json.loads(line)
        keys = list(parsed["data"].keys())
        assert keys == sorted(keys)

        # No spaces in separators
        assert ": " not in line
        assert ", " not in line

    def test_log_without_data(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)
        entry = trail.log("wrap_started")

        assert "data" not in entry

    def test_all_event_types(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)

        events = ["record", "delete", "prune", "wrap_started",
                  "wrap_completed", "continuity_saved"]
        for event in events:
            entry = trail.log(event, {"test": True})
            assert entry["event"] == event


class TestHashChainVerification:
    """Hash chain integrity verification."""

    def test_verify_valid_chain(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)

        for i in range(10):
            trail.log("record", {"id": str(i)})

        result = AuditTrail.verify(db)
        assert result.valid is True
        assert result.total_entries == 10
        assert result.files_verified == 1
        assert result.chain_break_at is None

    def test_verify_empty(self, tmp_path):
        db = tmp_path / "test.db"
        result = AuditTrail.verify(db)
        assert result.valid is True
        assert result.total_entries == 0

    def test_verify_detects_tampering(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)

        for i in range(5):
            trail.log("record", {"id": str(i)})

        # Tamper with entry 2
        audit_path = tmp_path / "test.audit.jsonl"
        lines = audit_path.read_text(encoding="utf-8").strip().split("\n")
        entry2 = json.loads(lines[2])
        entry2["data"]["id"] = "TAMPERED"
        lines[2] = json.dumps(entry2, sort_keys=True, separators=(",", ":"))
        audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = AuditTrail.verify(db)
        assert result.valid is False
        assert result.chain_break_at is not None

    def test_verify_detects_deletion(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)

        for i in range(5):
            trail.log("record", {"id": str(i)})

        # Delete entry 2
        audit_path = tmp_path / "test.audit.jsonl"
        lines = audit_path.read_text(encoding="utf-8").strip().split("\n")
        del lines[2]
        audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = AuditTrail.verify(db)
        assert result.valid is False

    def test_verify_detects_insertion(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)

        for i in range(5):
            trail.log("record", {"id": str(i)})

        # Insert a fake entry between 2 and 3
        audit_path = tmp_path / "test.audit.jsonl"
        lines = audit_path.read_text(encoding="utf-8").strip().split("\n")
        fake = json.dumps({"v": 1, "seq": 99, "ts": "2026-01-01T00:00:00Z",
                          "event": "record", "prev_hash": "sha256:fake",
                          "data": {"id": "injected"}},
                         sort_keys=True, separators=(",", ":"))
        lines.insert(3, fake)
        audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = AuditTrail.verify(db)
        assert result.valid is False


class TestCrashRecovery:
    """Recovery from crashes and restarts."""

    def test_recover_from_existing_file(self, tmp_path):
        db = tmp_path / "test.db"

        # First writer
        trail1 = AuditTrail(db)
        for i in range(5):
            trail1.log("record", {"id": str(i)})

        # New writer (simulates restart)
        trail2 = AuditTrail(db)
        trail2.log("record", {"id": "5"})

        # Chain should be unbroken
        result = AuditTrail.verify(db)
        assert result.valid is True
        assert result.total_entries == 6

    def test_recover_seq_continuity(self, tmp_path):
        db = tmp_path / "test.db"

        trail1 = AuditTrail(db)
        for i in range(3):
            trail1.log("record", {"id": str(i)})

        trail2 = AuditTrail(db)
        entry = trail2.log("record", {"id": "3"})

        assert entry["seq"] == 3

    def test_partial_write_recovery(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)

        for i in range(3):
            trail.log("record", {"id": str(i)})

        # Simulate crash: append partial JSON
        audit_path = tmp_path / "test.audit.jsonl"
        with open(audit_path, "a") as f:
            f.write('{"v":1,"seq":3,"ts":"2026-')  # Incomplete

        # Verify should skip the partial line
        result = AuditTrail.verify(db)
        assert result.valid is True
        assert result.total_entries == 3


class TestWeeklyRotation:
    """Weekly rotation with gzip compression."""

    def test_rotation_creates_gzip(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)

        # Write some entries
        trail.log("record", {"id": "1"})
        trail.log("record", {"id": "2"})

        # Force rotation by changing the last_week
        trail._last_week = "2026-W01"  # Pretend we're in a past week

        # Next log triggers rotation
        trail.log("record", {"id": "3"})

        # Should have a gzipped file
        gz_files = list(tmp_path.glob("*.audit.2026-W01.jsonl.gz"))
        assert len(gz_files) == 1

        # Active file should have the new entry
        active = tmp_path / "test.audit.jsonl"
        lines = active.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1  # Just entry 3

    def test_rotation_updates_manifest(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)

        trail.log("record", {"id": "1"})
        trail._last_week = "2026-W01"
        trail.log("record", {"id": "2"})

        manifest_path = tmp_path / "test.audit.manifest.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert len(manifest["files"]) == 1
        assert manifest["files"][0]["period"] == "2026-W01"
        assert manifest["files"][0]["entries"] == 1

    def test_chain_survives_rotation(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)

        trail.log("record", {"id": "1"})
        trail.log("record", {"id": "2"})
        trail._last_week = "2026-W01"
        trail.log("record", {"id": "3"})
        trail.log("record", {"id": "4"})

        result = AuditTrail.verify(db)
        assert result.valid is True
        assert result.total_entries == 4
        assert result.files_verified == 2  # gzipped + active

    def test_seq_resets_after_rotation(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)

        trail.log("record", {"id": "1"})
        trail.log("record", {"id": "2"})
        trail._last_week = "2026-W01"
        entry = trail.log("record", {"id": "3"})

        assert entry["seq"] == 0  # Reset for new file

    def test_gzip_content_readable(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)

        trail.log("record", {"id": "1", "content": "test episode"})
        trail._last_week = "2026-W01"
        trail.log("record", {"id": "2"})

        gz_files = list(tmp_path.glob("*.jsonl.gz"))
        assert len(gz_files) == 1

        # Verify gzip is readable
        with gzip.open(gz_files[0], "rt", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["data"]["content"] == "test episode"


class TestRetentionCleanup:
    """Automatic cleanup of old rotated files."""

    def test_cleanup_removes_old_files(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db, retention_days=7)

        # Create a fake old rotated file + manifest
        old_gz = tmp_path / "test.audit.2025-W01.jsonl.gz"
        with gzip.open(old_gz, "wt", encoding="utf-8") as f:
            f.write('{"v":1,"seq":0,"ts":"2025-01-06T00:00:00Z","event":"record","prev_hash":"sha256:GENESIS"}\n')

        manifest = {
            "version": 1,
            "db_path": "test.db",
            "active_file": "test.audit.jsonl",
            "active_last_hash": GENESIS_HASH,
            "active_last_seq": 0,
            "files": [{
                "filename": "test.audit.2025-W01.jsonl.gz",
                "period": "2025-W01",
                "entries": 1,
                "first_ts": "2025-01-06T00:00:00Z",
                "last_ts": "2025-01-06T00:00:00Z",
                "last_hash": "sha256:test",
                "sha256_file": "sha256:test",
            }],
        }
        manifest_path = tmp_path / "test.audit.manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        # Trigger cleanup via rotation
        trail.log("record", {"id": "1"})
        trail._last_week = "2026-W13"
        trail.log("record", {"id": "2"})

        # Old file should be gone
        assert not old_gz.exists()

        # Manifest should be updated
        updated = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_periods = [f["period"] for f in updated["files"]]
        assert "2025-W01" not in old_periods

    def test_no_cleanup_when_retention_none(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db, retention_days=None)

        trail.log("record", {"id": "1"})
        trail._last_week = "2026-W01"
        trail.log("record", {"id": "2"})

        # No files should be deleted (rotation happens but no cleanup)
        gz_files = list(tmp_path.glob("*.jsonl.gz"))
        assert len(gz_files) == 1


class TestOnEventCallback:
    """Cloud/SIEM integration callback."""

    def test_callback_receives_entry(self, tmp_path):
        db = tmp_path / "test.db"
        received = []
        trail = AuditTrail(db, on_event=lambda e: received.append(e))

        trail.log("record", {"id": "1"})

        assert len(received) == 1
        assert received[0]["event"] == "record"

    def test_callback_failure_doesnt_break_trail(self, tmp_path):
        db = tmp_path / "test.db"

        def bad_callback(entry):
            raise RuntimeError("Cloud is down!")

        trail = AuditTrail(db, on_event=bad_callback)

        # Should NOT raise despite callback failure
        entry = trail.log("record", {"id": "1"})
        assert entry["seq"] == 0

        # File should still be written
        audit_path = tmp_path / "test.audit.jsonl"
        assert audit_path.exists()

    def test_callback_called_after_write(self, tmp_path):
        db = tmp_path / "test.db"
        audit_path = tmp_path / "test.audit.jsonl"

        def check_file_exists(entry):
            # At callback time, file should already have the entry
            assert audit_path.exists()
            content = audit_path.read_text(encoding="utf-8")
            assert entry["event"] in content

        trail = AuditTrail(db, on_event=check_file_exists)
        trail.log("record", {"id": "1"})


class TestActorIdentity:
    """Actor identity field in audit entries (EU AI Act Article 12(2))."""

    def test_default_actor(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)
        entry = trail.log("record", {"id": "1"})
        assert entry["actor"] == "agent"

    def test_custom_actor(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)
        entry = trail.log("record", {"id": "1"}, actor="research-agent-1")
        assert entry["actor"] == "research-agent-1"

    def test_actor_persisted_in_jsonl(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)
        trail.log("record", {"id": "1"}, actor="my-agent")

        audit_path = tmp_path / "test.audit.jsonl"
        entry = json.loads(audit_path.read_text(encoding="utf-8").strip())
        assert entry["actor"] == "my-agent"


class TestOrphanAdoption:
    """Recovery from crash during rotation (orphaned sealed files)."""

    def test_adopt_orphaned_gz(self, tmp_path):
        db = tmp_path / "test.db"

        # Create a valid orphaned .gz file (simulates crash after rename
        # but before manifest update)
        orphan_name = "test.audit.2026-W13.jsonl.gz"
        entry_json = json.dumps({
            "v": 1, "seq": 0, "ts": "2026-03-24T12:00:00.000000Z",
            "event": "record", "actor": "agent",
            "prev_hash": GENESIS_HASH, "data": {"id": "orphan"}
        }, sort_keys=True, separators=(",", ":"))
        with gzip.open(tmp_path / orphan_name, "wt", encoding="utf-8") as f:
            f.write(entry_json + "\n")

        # New trail should adopt the orphan on initialize
        trail = AuditTrail(db)
        trail.log("record", {"id": "new"})

        # Manifest should now include the orphaned file
        manifest_path = tmp_path / "test.audit.manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        filenames = [f["filename"] for f in manifest["files"]]
        assert orphan_name in filenames

    def test_no_double_adopt(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db)

        # Normal rotation creates a known .gz file
        trail.log("record", {"id": "1"})
        trail._last_week = "2026-W01"
        trail.log("record", {"id": "2"})

        # Re-initialize should not adopt the known file again
        trail2 = AuditTrail(db)
        trail2.log("record", {"id": "3"})

        manifest = json.loads(
            (tmp_path / "test.audit.manifest.json").read_text(encoding="utf-8")
        )
        # Should have exactly 1 sealed file, not duplicated
        periods = [f["period"] for f in manifest["files"]]
        assert periods.count("2026-W01") == 1


class TestLargeEntryRecovery:
    """Recovery from large entries (>8KB) and corrupt-then-valid sequences."""

    def test_large_entry_recovery(self, tmp_path):
        """Entries >8KB must not break crash recovery."""
        db = tmp_path / "test.db"
        trail = AuditTrail(db)

        # Record a large entry (10KB+ content)
        large_content = "x" * 12000
        trail.log("record", {"content": large_content})
        trail.log("record", {"id": "2"})

        # New writer should recover correctly
        trail2 = AuditTrail(db)
        entry = trail2.log("record", {"id": "3"})
        assert entry["seq"] == 2

        result = AuditTrail.verify(db)
        assert result.valid is True
        assert result.total_entries == 3

    def test_recovery_skips_corrupt_finds_valid(self, tmp_path):
        """If last line is corrupt, recovery should find previous valid entry."""
        db = tmp_path / "test.db"
        trail = AuditTrail(db)

        trail.log("record", {"id": "1"})
        trail.log("record", {"id": "2"})
        trail.log("record", {"id": "3"})

        # Append corrupt line
        audit_path = tmp_path / "test.audit.jsonl"
        with open(audit_path, "a") as f:
            f.write('{"v":1,"seq":3,"CORRUPT\n')

        # New writer should recover from entry 3 (seq=2), continue at seq=3
        trail2 = AuditTrail(db)
        entry = trail2.log("record", {"id": "4"})
        assert entry["seq"] == 3

        result = AuditTrail.verify(db)
        assert result.valid is True
        assert result.total_entries == 4  # 3 valid + 1 new (corrupt skipped)


class TestChainAnchorAfterCleanup:
    """Verification works correctly after retention cleanup removes old files."""

    def test_verify_after_cleanup(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db, retention_days=7)

        # Write entries, rotate with old week to trigger cleanup
        trail.log("record", {"id": "1"})
        trail.log("record", {"id": "2"})
        trail._last_week = "2025-W01"  # Very old

        # This rotation + new log should trigger cleanup of the old file
        trail.log("record", {"id": "3"})

        # Force another rotation with current week
        # The 2025-W01 file should get cleaned up
        result = AuditTrail.verify(db)
        assert result.valid is True

    def test_verify_fails_on_missing_sealed_file(self, tmp_path):
        db = tmp_path / "test.db"
        trail = AuditTrail(db, retention_days=None)

        trail.log("record", {"id": "1"})
        trail._last_week = "2026-W01"
        trail.log("record", {"id": "2"})

        # Manually delete the sealed gz file (simulating external tampering)
        gz_files = list(tmp_path.glob("*.jsonl.gz"))
        assert len(gz_files) == 1
        gz_files[0].unlink()

        result = AuditTrail.verify(db)
        assert result.valid is False
        assert "Missing sealed files" in result.error


class TestWrapCancelled:
    """wrap_cancelled audit events."""

    def test_wrap_cancelled_logged(self, tmp_path):
        from anneal_memory.store import Store

        db = tmp_path / "test.db"
        store = Store(db)
        store.wrap_started()
        store.wrap_cancelled()

        audit_path = tmp_path / "test.audit.jsonl"
        lines = audit_path.read_text(encoding="utf-8").strip().split("\n")
        events = [json.loads(l)["event"] for l in lines]
        assert "wrap_started" in events
        assert "wrap_cancelled" in events
        store.close()


class TestStoreIntegration:
    """Audit trail integration with Store."""

    def test_store_creates_audit_by_default(self, tmp_path):
        from anneal_memory.store import Store

        db = tmp_path / "test.db"
        store = Store(db)
        assert store._audit is not None
        store.close()

    def test_store_no_audit_flag(self, tmp_path):
        from anneal_memory.store import Store

        db = tmp_path / "test.db"
        store = Store(db, audit=False)
        assert store._audit is None
        store.close()

    def test_record_writes_audit(self, tmp_path):
        from anneal_memory.store import Store

        db = tmp_path / "test.db"
        store = Store(db)
        store.record("Test episode", "observation")

        audit_path = tmp_path / "test.audit.jsonl"
        assert audit_path.exists()
        line = audit_path.read_text(encoding="utf-8").strip()
        entry = json.loads(line)
        assert entry["event"] == "record"
        assert entry["data"]["content_hash"]  # Hash, not raw content
        assert "content" not in entry["data"]  # No raw content in audit
        assert entry["data"]["type"] == "observation"
        assert entry["actor"] == "agent"  # source forwarded as actor
        store.close()

    def test_delete_writes_audit(self, tmp_path):
        from anneal_memory.store import Store

        db = tmp_path / "test.db"
        store = Store(db)
        ep = store.record("Delete me", "observation")
        store.delete(ep.id)

        audit_path = tmp_path / "test.audit.jsonl"
        lines = audit_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        delete_entry = json.loads(lines[1])
        assert delete_entry["event"] == "delete"
        assert delete_entry["data"]["episode_id"] == ep.id
        assert "content_hash" in delete_entry["data"]
        store.close()

    def test_wrap_lifecycle_writes_audit(self, tmp_path):
        from anneal_memory.store import Store

        db = tmp_path / "test.db"
        store = Store(db)
        store.record("Episode 1", "observation")
        store.wrap_started()
        store.save_continuity("## State\nTest\n## Patterns\n\n## Decisions\n\n## Context\n")
        store.wrap_completed(episodes_compressed=1, continuity_chars=50)

        audit_path = tmp_path / "test.audit.jsonl"
        lines = audit_path.read_text(encoding="utf-8").strip().split("\n")

        events = [json.loads(l)["event"] for l in lines]
        assert "record" in events
        assert "wrap_started" in events
        assert "continuity_saved" in events
        assert "wrap_completed" in events
        store.close()

    def test_prune_writes_audit(self, tmp_path):
        from anneal_memory.store import Store

        db = tmp_path / "test.db"
        store = Store(db)

        # Record with old timestamp
        store.record("Old episode", "observation",
                     timestamp="2020-01-01T00:00:00.000000Z")
        store.prune(older_than_days=1)

        audit_path = tmp_path / "test.audit.jsonl"
        lines = audit_path.read_text(encoding="utf-8").strip().split("\n")

        events = [json.loads(l)["event"] for l in lines]
        assert "prune" in events
        prune_entry = json.loads(lines[-1])
        assert prune_entry["data"]["count"] == 1
        store.close()

    def test_full_chain_valid_through_store(self, tmp_path):
        from anneal_memory.store import Store

        db = tmp_path / "test.db"
        store = Store(db)

        store.record("Episode 1", "observation")
        store.record("Episode 2", "decision")
        store.wrap_started()
        store.save_continuity("## State\nTest\n## Patterns\n\n## Decisions\n\n## Context\n")
        store.wrap_completed(episodes_compressed=2, continuity_chars=50)
        store.record("Episode 3", "outcome")

        result = AuditTrail.verify(db)
        assert result.valid is True
        assert result.total_entries == 6  # 2 records + wrap_started + continuity + wrap_completed + 1 record
        store.close()

    def test_no_audit_means_no_files(self, tmp_path):
        from anneal_memory.store import Store

        db = tmp_path / "test.db"
        store = Store(db, audit=False)
        store.record("Episode 1", "observation")
        store.wrap_started()
        store.save_continuity("## State\nTest\n## Patterns\n\n## Decisions\n\n## Context\n")
        store.wrap_completed(episodes_compressed=1, continuity_chars=50)

        audit_path = tmp_path / "test.audit.jsonl"
        assert not audit_path.exists()
        store.close()
