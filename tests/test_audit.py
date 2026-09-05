"""Tests for the hash-chained JSONL audit trail."""

import gzip
import json
import os
import tempfile
import uuid
import warnings
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


class TestMultiRotationIntegration:
    """Multi-rotation → crash → recovery integration tests."""

    def test_multi_rotation_verify_and_recovery(self, tmp_path):
        """3+ organic rotations, verify after each, new writer, verify again.

        Exercises the full rotation lifecycle end-to-end: multiple week
        boundaries, chain continuity across rotated files, and recovery
        from a fresh AuditTrail instance reading the existing state.
        """
        db = tmp_path / "test.db"
        trail = AuditTrail(db)

        # Week 1: write 3 entries
        trail.log("record", {"id": "w1-1"})
        trail.log("record", {"id": "w1-2"})
        trail.log("record", {"id": "w1-3"})

        # Rotate to week 2
        trail._last_week = "2026-W10"
        trail.log("record", {"id": "w2-1"})
        trail.log("record", {"id": "w2-2"})

        result = AuditTrail.verify(db)
        assert result.valid is True
        assert result.total_entries == 5
        assert result.files_verified == 2  # W10.gz + active

        # Rotate to week 3
        trail._last_week = "2026-W11"
        trail.log("record", {"id": "w3-1"})

        result = AuditTrail.verify(db)
        assert result.valid is True
        assert result.total_entries == 6
        assert result.files_verified == 3  # W10.gz + W11.gz + active

        # Rotate to week 4
        trail._last_week = "2026-W12"
        trail.log("record", {"id": "w4-1"})
        trail.log("record", {"id": "w4-2"})

        result = AuditTrail.verify(db)
        assert result.valid is True
        assert result.total_entries == 8
        assert result.files_verified == 4  # W10 + W11 + W12 + active

        # Simulate crash: create a brand new AuditTrail instance
        # This tests recovery from manifest + active file state
        trail2 = AuditTrail(db)
        trail2.log("record", {"id": "recovery-1"})

        # Full chain should still verify end-to-end
        result = AuditTrail.verify(db)
        assert result.valid is True
        assert result.total_entries == 9
        assert result.files_verified == 4  # 3 sealed + active

        # Verify all .gz files exist
        gz_files = sorted(tmp_path.glob("*.jsonl.gz"))
        assert len(gz_files) == 3
        prefix = "test.audit."
        periods = {
            f.name.removeprefix(prefix).removesuffix(".jsonl.gz")
            for f in gz_files
        }
        assert periods == {"2026-W10", "2026-W11", "2026-W12"}

    def test_rotation_atomic_gz_no_tmp_residue(self, tmp_path):
        """Rotation should not leave .tmp files after successful completion."""
        db = tmp_path / "test.db"
        trail = AuditTrail(db)

        trail.log("record", {"id": "1"})
        trail._last_week = "2026-W01"
        trail.log("record", {"id": "2"})

        # No .tmp files should remain after successful rotation
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

        # .gz should exist and be valid
        gz_files = list(tmp_path.glob("*.jsonl.gz"))
        assert len(gz_files) == 1

        result = AuditTrail.verify(db)
        assert result.valid is True


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
        store.wrap_started(token=uuid.uuid4().hex, episode_ids=[])
        store.wrap_cancelled()

        audit_path = tmp_path / "test.audit.jsonl"
        lines = audit_path.read_text(encoding="utf-8").strip().split("\n")
        events = [json.loads(l)["event"] for l in lines]
        assert "wrap_started" in events
        assert "wrap_cancelled" in events
        store.close()


class TestDiogenesBugFixes:
    """Regression tests for bugs found by Diogenes code review (sweeps 4-7)."""

    def test_double_orphan_prefers_gz_and_removes_jsonl(self, tmp_path):
        """MEDIUM: If both .gz and .jsonl exist for same period (crash between
        gzip-complete and sealed_path.unlink()), prefer .gz and remove .jsonl.
        Without fix: both adopted into manifest → verify() false chain break."""
        db = tmp_path / "test.db"
        stem = "test"

        # Create the same content in both .gz and .jsonl for same period
        entry_json = json.dumps({
            "v": 1, "seq": 0, "ts": "2026-03-24T12:00:00.000000Z",
            "event": "record", "actor": "agent",
            "prev_hash": GENESIS_HASH, "data": {"id": "1"}
        }, sort_keys=True, separators=(",", ":"))

        # .gz file (gzip completed)
        gz_path = tmp_path / f"{stem}.audit.2026-W13.jsonl.gz"
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            f.write(entry_json + "\n")

        # .jsonl file (not yet deleted — crash scenario)
        jsonl_path = tmp_path / f"{stem}.audit.2026-W13.jsonl"
        jsonl_path.write_text(entry_json + "\n", encoding="utf-8")

        # Initialize trail — should adopt .gz, remove .jsonl
        trail = AuditTrail(db)
        trail.log("record", {"id": "new"})

        # .jsonl duplicate should be gone
        assert not jsonl_path.exists()
        assert gz_path.exists()

        # Manifest should have exactly one entry for this period
        manifest = json.loads(
            (tmp_path / f"{stem}.audit.manifest.json").read_text(encoding="utf-8")
        )
        periods = [f["period"] for f in manifest["files"]]
        assert periods.count("2026-W13") == 1
        assert manifest["files"][0]["filename"].endswith(".gz")

        # Chain should verify cleanly
        result = AuditTrail.verify(db)
        assert result.valid is True

    def test_init_failure_allows_retry(self, tmp_path):
        """MEDIUM: _initialized must not be set before init completes.
        If orphan adoption raises, next log() should retry init, not
        write with seq=0 + GENESIS_HASH."""
        db = tmp_path / "test.db"
        trail = AuditTrail(db)

        # Write some entries so there's state to recover
        trail.log("record", {"id": "1"})
        trail.log("record", {"id": "2"})

        # Create a new trail and monkeypatch adoption to fail once
        trail2 = AuditTrail(db)
        assert trail2._initialized is False

        call_count = 0
        original_adopt = trail2._adopt_orphaned_files

        def failing_adopt():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("Simulated disk full during orphan adoption")
            return original_adopt()

        trail2._adopt_orphaned_files = failing_adopt

        # First log() attempt: init fails, should propagate the error
        with pytest.raises(OSError, match="disk full"):
            trail2.log("record", {"id": "3"})

        # _initialized should still be False after failure
        assert trail2._initialized is False

        # Second log() attempt: init retries and succeeds
        entry = trail2.log("record", {"id": "3"})
        assert trail2._initialized is True
        assert entry["seq"] == 2  # Continues from where trail1 left off

        # Chain should be valid
        result = AuditTrail.verify(db)
        assert result.valid is True

    def test_jsonl_orphan_period_not_mangled(self, tmp_path):
        """LOW: Uncompressed .jsonl orphan should have clean period field,
        not ' 2026-W14.jsonl'."""
        db = tmp_path / "test.db"
        stem = "test"

        # Create uncompressed orphan (crash before gzip)
        entry_json = json.dumps({
            "v": 1, "seq": 0, "ts": "2026-03-31T12:00:00.000000Z",
            "event": "record", "actor": "agent",
            "prev_hash": GENESIS_HASH, "data": {"id": "1"}
        }, sort_keys=True, separators=(",", ":"))

        jsonl_path = tmp_path / f"{stem}.audit.2026-W14.jsonl"
        jsonl_path.write_text(entry_json + "\n", encoding="utf-8")

        trail = AuditTrail(db)
        trail.log("record", {"id": "new"})

        manifest = json.loads(
            (tmp_path / f"{stem}.audit.manifest.json").read_text(encoding="utf-8")
        )
        orphan_entry = [f for f in manifest["files"] if "2026-W14" in f["filename"]]
        assert len(orphan_entry) == 1
        assert orphan_entry[0]["period"] == "2026-W14"  # Not "2026-W14.jsonl"

    def test_seq_consistent_after_rotation_crash_recovery(self, tmp_path):
        """LOW: Seq should be 0 after rotation whether via normal path or
        crash recovery. Manifest must store active_last_seq=0 after rotation."""
        db = tmp_path / "test.db"
        trail = AuditTrail(db)

        # Write entries and rotate
        trail.log("record", {"id": "1"})
        trail.log("record", {"id": "2"})
        trail._last_week = "2026-W01"
        trail.log("record", {"id": "3"})  # Triggers rotation, seq resets to 0

        # Verify manifest has seq=0 (not the pre-rotation value)
        manifest = json.loads(
            (tmp_path / "test.audit.manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["active_last_seq"] == 0

        # Simulate crash: delete the active file (as if it was never written)
        active = tmp_path / "test.audit.jsonl"
        active.unlink()

        # New trail recovers from manifest — should start at seq 0
        trail2 = AuditTrail(db)
        entry = trail2.log("record", {"id": "4"})
        assert entry["seq"] == 0  # Matches normal rotation behavior

        # Chain should still verify
        result = AuditTrail.verify(db)
        assert result.valid is True

    def test_cleanup_preserves_files_with_empty_last_ts(self, tmp_path):
        """LOW: Files with empty last_ts should not be deleted by cleanup.
        Empty string < any date string in Python → was always deleting."""
        db = tmp_path / "test.db"
        trail = AuditTrail(db, retention_days=7)

        # Create a sealed file with empty last_ts (simulates orphan adoption
        # of file with no valid entries)
        empty_gz = tmp_path / "test.audit.2026-W13.jsonl.gz"
        with gzip.open(empty_gz, "wt", encoding="utf-8") as f:
            f.write("")  # Empty content

        manifest = trail._load_manifest()
        manifest["files"].append({
            "filename": "test.audit.2026-W13.jsonl.gz",
            "period": "2026-W13",
            "entries": 0,
            "first_ts": "",
            "last_ts": "",  # Empty — the bug trigger
            "last_hash": "",
            "sha256_file": "",
        })
        trail._save_manifest(manifest)

        # Run cleanup — should NOT delete file with empty last_ts
        removed = trail._cleanup()
        assert removed == 0
        assert empty_gz.exists()

    def test_multi_period_orphans_adopted_in_order(self, tmp_path):
        """Orphans from multiple periods must be adopted in chronological
        order so active_last_hash reflects the most recent file's chain."""
        db = tmp_path / "test.db"
        stem = "test"

        # Create two orphans: W13 and W14, each with one chained entry
        entry_w13 = json.dumps({
            "v": 1, "seq": 0, "ts": "2026-03-24T12:00:00.000000Z",
            "event": "record", "actor": "agent",
            "prev_hash": GENESIS_HASH, "data": {"id": "w13"}
        }, sort_keys=True, separators=(",", ":"))
        # Compute hash of W13 entry for W14's prev_hash
        w13_hash = "sha256:" + __import__("hashlib").sha256(
            entry_w13.encode("utf-8")
        ).hexdigest()

        entry_w14 = json.dumps({
            "v": 1, "seq": 0, "ts": "2026-03-31T12:00:00.000000Z",
            "event": "record", "actor": "agent",
            "prev_hash": w13_hash, "data": {"id": "w14"}
        }, sort_keys=True, separators=(",", ":"))

        with gzip.open(tmp_path / f"{stem}.audit.2026-W13.jsonl.gz", "wt", encoding="utf-8") as f:
            f.write(entry_w13 + "\n")
        with gzip.open(tmp_path / f"{stem}.audit.2026-W14.jsonl.gz", "wt", encoding="utf-8") as f:
            f.write(entry_w14 + "\n")

        # Initialize — should adopt both in order
        trail = AuditTrail(db)
        trail.log("record", {"id": "new"})

        manifest = json.loads(
            (tmp_path / f"{stem}.audit.manifest.json").read_text(encoding="utf-8")
        )
        periods = [f["period"] for f in manifest["files"]]
        assert "2026-W13" in periods
        assert "2026-W14" in periods

        # Chain should verify end-to-end
        result = AuditTrail.verify(db)
        assert result.valid is True
        assert result.total_entries == 3  # W13(1) + W14(1) + new(1)


class TestDiogenesSweep8Fixes:
    """Regression tests for Diogenes Sweep 8 bugs (Apr 2026)."""

    def test_orphan_adoption_chronological_order_with_mixed_types(self, tmp_path):
        """LOW: When mixed .gz and .jsonl orphans span non-adjacent periods,
        orphan adoption must sort by period before appending to manifest.
        Without fix: two-pass glob inserts all .gz periods before all .jsonl
        periods → manifest breaks chronological order → verify() chain break.

        Scenario: W13 exists as .jsonl (crash before gzip), W14 as .gz (normal).
        Without sort: W14.gz adopted first (glob *.gz runs first), then W13.jsonl.
        With sort: W13 first, W14 second → correct chain order."""
        import hashlib as _hl

        db = tmp_path / "test.db"
        stem = "test"

        # W13 as .jsonl (uncompressed orphan — crash before gzip)
        entry_w13 = json.dumps({
            "v": 1, "seq": 0, "ts": "2026-03-24T12:00:00.000000Z",
            "event": "record", "actor": "agent",
            "prev_hash": GENESIS_HASH, "data": {"id": "w13"}
        }, sort_keys=True, separators=(",", ":"))
        w13_hash = "sha256:" + _hl.sha256(entry_w13.encode("utf-8")).hexdigest()

        # W14 as .gz (normal sealed file)
        entry_w14 = json.dumps({
            "v": 1, "seq": 0, "ts": "2026-03-31T12:00:00.000000Z",
            "event": "record", "actor": "agent",
            "prev_hash": w13_hash, "data": {"id": "w14"}
        }, sort_keys=True, separators=(",", ":"))

        # Write .jsonl for W13 (no gzip)
        jsonl_w13 = tmp_path / f"{stem}.audit.2026-W13.jsonl"
        jsonl_w13.write_text(entry_w13 + "\n", encoding="utf-8")

        # Write .gz for W14
        with gzip.open(tmp_path / f"{stem}.audit.2026-W14.jsonl.gz", "wt", encoding="utf-8") as f:
            f.write(entry_w14 + "\n")

        # Initialize — should adopt W13 first, W14 second (chronological)
        trail = AuditTrail(db)
        trail.log("record", {"id": "new"})

        manifest = json.loads(
            (tmp_path / f"{stem}.audit.manifest.json").read_text(encoding="utf-8")
        )
        periods = [f["period"] for f in manifest["files"]]
        assert periods == ["2026-W13", "2026-W14"], (
            f"Manifest periods should be chronological, got: {periods}"
        )

        # Chain should verify end-to-end
        result = AuditTrail.verify(db)
        assert result.valid is True
        assert result.total_entries == 3  # W13(1) + W14(1) + new(1)

    def test_stale_tmp_gz_files_cleaned_on_init(self, tmp_path):
        """LOW: Crash during gzip write leaves *.jsonl.gz.tmp files forever.
        These are not caught by orphan adoption (looks for .gz and .jsonl only)
        and not by _cleanup (only removes manifest-tracked files). Should be
        cleaned up during _adopt_orphaned_files on next init."""
        db = tmp_path / "test.db"
        stem = "test"

        # Create a stale .tmp file (simulates crash during gzip write)
        tmp_gz = tmp_path / f"{stem}.audit.2026-W12.jsonl.gz.tmp"
        tmp_gz.write_bytes(b"partial gzip data")

        # Also create a second one to verify all are cleaned
        tmp_gz2 = tmp_path / f"{stem}.audit.2026-W11.jsonl.gz.tmp"
        tmp_gz2.write_bytes(b"more partial data")

        assert tmp_gz.exists()
        assert tmp_gz2.exists()

        # Initialize trail — should clean up .tmp files
        trail = AuditTrail(db)
        trail.log("record", {"id": "1"})

        # .tmp files should be gone
        assert not tmp_gz.exists()
        assert not tmp_gz2.exists()

        # No .tmp files in manifest either
        manifest_path = tmp_path / f"{stem}.audit.manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for f in manifest.get("files", []):
                assert ".tmp" not in f["filename"]

    def test_stale_tmp_cleanup_does_not_affect_active_file(self, tmp_path):
        """Ensure .tmp cleanup only targets gzip temp files, not the active file
        or any other files."""
        db = tmp_path / "test.db"
        stem = "test"

        # Create stale .tmp
        tmp_gz = tmp_path / f"{stem}.audit.2026-W12.jsonl.gz.tmp"
        tmp_gz.write_bytes(b"partial")

        # Initialize and write some entries
        trail = AuditTrail(db)
        trail.log("record", {"id": "1"})
        trail.log("record", {"id": "2"})

        # Active file should still exist and be valid
        active = tmp_path / f"{stem}.audit.jsonl"
        assert active.exists()

        # Chain should verify
        result = AuditTrail.verify(db)
        assert result.valid is True
        assert result.total_entries == 2


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
        store.wrap_started(token=uuid.uuid4().hex, episode_ids=[])
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
        store.wrap_started(token=uuid.uuid4().hex, episode_ids=[])
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
        store.wrap_started(token=uuid.uuid4().hex, episode_ids=[])
        store.save_continuity("## State\nTest\n## Patterns\n\n## Decisions\n\n## Context\n")
        store.wrap_completed(episodes_compressed=1, continuity_chars=50)

        audit_path = tmp_path / "test.audit.jsonl"
        assert not audit_path.exists()
        store.close()


# ---------------------------------------------------------------------------
# AM-AUDIT-AFTER-COMMIT (2026-09-04)
#
# ⛔ WHY THIS BLOCK EXISTS, AND IT IS THE POINT OF IT. On 2026-09-03 a codex L3
# HIGH established the policy "an audit-sink failure must not propagate once the
# work is committed" and it was implemented as an inline try/except at ONE call
# site (``wrap_cancelled``). Measured 2026-09-04: SIXTEEN post-commit emit sites
# existed and the correction reached one of them. The class is the day's
# portfolio-wide one — a guard that cannot see its own subject: the guard was
# real, correct, and scoped to the site that happened to be reported.
#
# ⚠ AND THE FIRST COUNT WAS ITSELF WRONG, WHICH IS THE POINT. The opening
# census said EIGHT, because it scoped by the SYMPTOM — the literal text
# ``self._audit.log`` — and so could not see the seven association methods
# that reach their commit through a helper's ``commit=`` argument and emit via
# ``_audit_log``. The L1 pass found them by asking about POSITION rather than
# spelling. A census scoped by symptom missed the class, inside the census
# taken to fix a guard scoped by symptom.
#
# So the policy now has one home (``Store._audit_log_after_commit``) and two
# tests: a BEHAVIOURAL one that drives every affected public method with a
# failing sink, and a MECHANICAL one that fails if a ninth site is ever written
# bare. Neither is a proxy for the other — the behavioural test proves the
# policy holds for the methods that exist, the mechanical one proves no new
# method can quietly opt out.
# ---------------------------------------------------------------------------

SCHEMA_OK = [
    {"heading": "State", "role": "live-state"},
    {"heading": "Patterns", "role": "graduating"},
]


def _sink_that_fails(store, exc):
    """Replace the store's audit sink with one that raises ``exc``."""
    def boom(*args, **kwargs):
        raise exc
    store._audit.log = boom


def _committed_op_cases(tmp_path_factory=None):
    """(label, setup, act, assert_committed) for every post-commit emit site."""
    from anneal_memory.store import Store

    def mk(sub):
        return Store(sub / "memory.db")

    def c_record(sub):
        s = mk(sub)
        return s, lambda: s.record("hello", episode_type="observation"), \
            lambda: len(s.recall(limit=99).episodes) == 1

    def c_delete(sub):
        s = mk(sub)
        s.record("bye", episode_type="observation")
        ep = s.recall(limit=1).episodes[0].id
        return s, lambda: s.delete(ep), \
            lambda: len(s.recall(limit=99).episodes) == 0

    def c_wrap_started(sub):
        s = mk(sub)
        s.record("e", episode_type="observation")
        ids = [e.id for e in s.recall(limit=9).episodes]
        return s, lambda: s.wrap_started(episode_ids=ids, token="tok"), \
            lambda: s._get_metadata("wrap_token") == "tok"

    def c_wrap_completed(sub):
        s = mk(sub)
        s.record("e", episode_type="observation")
        ids = [e.id for e in s.recall(limit=9).episodes]
        s.wrap_started(episode_ids=ids, token="tok")
        return s, lambda: s.wrap_completed(
            episodes_compressed=1, continuity_chars=10, wrap_token="tok"
        ), lambda: len(s.get_wrap_history()) == 1

    def c_wrap_cancelled(sub):
        s = mk(sub)
        s.record("e", episode_type="observation")
        ids = [e.id for e in s.recall(limit=9).episodes]
        s.wrap_started(episode_ids=ids, token="tok")
        return s, lambda: s.wrap_cancelled(), \
            lambda: not s._get_metadata("wrap_started_at")

    def c_prune(sub):
        s = mk(sub)
        s.record("old", episode_type="observation")
        return s, lambda: s.prune(older_than_days=0), \
            lambda: len(s.recall(limit=99).episodes) == 0

    def c_save_continuity(sub):
        s = mk(sub)
        return s, lambda: s.save_continuity("# hi\n"), \
            lambda: s.continuity_path.exists()

    def c_set_section_schema(sub):
        s = mk(sub)
        return s, lambda: s.set_section_schema(SCHEMA_OK), \
            lambda: [x["heading"] for x in s.section_schema] == ["State", "Patterns"]

    def _assoc_seed(sub):
        """A store with two episodes, one episode edge and one pattern edge."""
        st = mk(sub)
        a = st.record("alpha", episode_type="observation")
        b = st.record("beta", episode_type="observation")
        st.record_associations({(a.id, b.id)})
        st.seed_pattern_co_graduation(["p_one", "p_two"])
        return st, a, b

    def c_record_associations(sub):
        st, a, b = _assoc_seed(sub)
        return st, lambda: st.record_associations({(b.id, a.id)}), \
            lambda: st.association_stats().total_links > 0

    def c_decay_associations(sub):
        st, a, b = _assoc_seed(sub)
        return st, lambda: st.decay_associations(), lambda: True

    def c_seed_pattern_co_graduation(sub):
        st, a, b = _assoc_seed(sub)
        return st, lambda: st.seed_pattern_co_graduation(["p_three", "p_four"]), \
            lambda: True

    def c_rename_pattern_association(sub):
        st, a, b = _assoc_seed(sub)
        return st, lambda: st.rename_pattern_association("p_one", "p_new"), \
            lambda: True

    def c_sever_pattern_concept(sub):
        st, a, b = _assoc_seed(sub)
        return st, lambda: st.sever_pattern_concept("p_two"), lambda: True

    return [
        ("record", c_record),
        ("delete", c_delete),
        ("wrap_started", c_wrap_started),
        ("wrap_completed", c_wrap_completed),
        ("wrap_cancelled", c_wrap_cancelled),
        ("prune", c_prune),
        ("save_continuity", c_save_continuity),
        ("set_section_schema", c_set_section_schema),
        # ⛔ THE SEVEN THE FIRST CENSUS COULD NOT SEE. These reach their
        # commit through a free-function helper's ``commit=`` argument rather
        # than a literal ``commit()`` in the method body, and emitted via
        # ``_audit_log`` — so a scan for the text ``self._audit.log`` missed
        # every one. Measured 2026-09-04 (L1): ``gc_pattern_associations``
        # and ``sever_pattern_concept`` DELETED edges and then raised a raw
        # OSError. Five of the seven are here; ``gc_pattern_associations``
        # and ``drain_co_surface_events`` emit only when their count is
        # non-zero and are covered by the mechanical scan instead — stated
        # rather than quietly omitted, because a case that cannot fire would
        # make this table look wider than it is.
        ("record_associations", c_record_associations),
        ("decay_associations", c_decay_associations),
        ("seed_pattern_co_graduation", c_seed_pattern_co_graduation),
        ("rename_pattern_association", c_rename_pattern_association),
        ("sever_pattern_concept", c_sever_pattern_concept),
    ]


CASES = _committed_op_cases()


class TestAuditAfterCommitPolicy:
    """A failing audit sink must never fail an operation that already landed."""

    @pytest.mark.parametrize("label,builder", CASES, ids=[c[0] for c in CASES])
    @pytest.mark.parametrize(
        "exc",
        [OSError(28, "No space left on device"), RuntimeError("rotation failed")],
        ids=["oserror", "non-oserror"],
    )
    def test_sink_failure_does_not_fail_committed_work(
        self, tmp_path, label, builder, exc
    ):
        # ⚠ The non-OSError case is not decoration: ``set_section_schema``
        # caught only OSError and propagated a RuntimeError from the sink,
        # failing a completed migration (measured 2026-09-04).
        sub = tmp_path / f"{label}-{type(exc).__name__}"
        sub.mkdir()
        store, act, committed = builder(sub)
        _sink_that_fails(store, exc)

        with pytest.warns(UserWarning, match="COMMITTED"):
            act()

        assert committed(), f"{label}: the work did not land"

    @pytest.mark.parametrize("label,builder", CASES, ids=[c[0] for c in CASES])
    def test_sink_failure_survives_warnings_as_errors(self, tmp_path, label, builder):
        # ⛔ THE CASE THE GUARD'S OWN TEST ONCE MASKED. ``pytest.warns`` installs
        # a capturing filter, so a test written only in the form above cannot
        # see that ``warnings.warn`` ITSELF raises under ``-W error`` /
        # PYTHONWARNINGS=error — which recreates the exact "committed, then
        # reported as failed" path the policy exists to eliminate. This test
        # takes the filter away on purpose.
        sub = tmp_path / f"{label}-werror"
        sub.mkdir()
        store, act, committed = builder(sub)
        _sink_that_fails(store, OSError(28, "No space left on device"))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            act()  # must not raise, not even the warning

        assert committed(), f"{label}: the work did not land"


class TestNoBareAuditEmitSites:
    """Mechanically: every ``self._audit.log`` call lives inside a policy helper.

    ⚠ WHAT THIS DOES AND DOES NOT PROVE. It proves no emit site bypasses the two
    helpers, which is the drift this exists to stop. It does NOT prove a site
    picked the RIGHT helper (``_audit_log`` is correct pre-commit, and
    ``_audit_log_after_commit`` post-commit) — that judgment is what the
    behavioural tests above cover for the methods that exist. Two tests, two
    subjects, on purpose.
    """

    ALLOWED_ENCLOSING_DEFS = {"_audit_log", "_audit_log_after_commit"}
    # ⚠ These names are matched UNQUALIFIED, across every module. A future
    # class defining its own ``_audit_log``, or a nested closure with that
    # name, would inherit the exemption. Named here rather than engineered
    # around: qualifying it needs a class-path walk, and the next test in
    # this class closes the reachable half of the gap by asserting the
    # exempt helper has no production callers at all.

    def _bare_sites(self, module_path):
        import ast

        source = module_path.read_text()
        tree = ast.parse(source)
        # Map every node to its enclosing function name.
        enclosing: dict[int, str] = {}

        class Walk(ast.NodeVisitor):
            def __init__(self):
                self.stack: list[str] = []

            def visit_FunctionDef(self, node):
                self.stack.append(node.name)
                for child in ast.iter_child_nodes(node):
                    self.visit(child)
                self.stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def generic_visit(self, node):
                if isinstance(node, ast.Call):
                    enclosing[id(node)] = self.stack[-1] if self.stack else "<module>"
                super().generic_visit(node)

        walker = Walk()
        walker.visit(tree)

        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            # match `<anything>._audit.log(...)`
            if (
                isinstance(fn, ast.Attribute)
                and fn.attr == "log"
                and isinstance(fn.value, ast.Attribute)
                and fn.value.attr == "_audit"
            ):
                where = enclosing.get(id(node), "<unknown>")
                if where not in self.ALLOWED_ENCLOSING_DEFS:
                    found.append((module_path.name, node.lineno, where))
        return found

    def test_no_audit_emit_outside_the_policy_helpers(self):
        import anneal_memory

        pkg = Path(anneal_memory.__file__).parent
        offenders = []
        # rglob, not glob: the package is flat today, so these are the same
        # set — but this is a structural-invariant test, and a subpackage
        # added later must not silently drop out of its scope.
        for module in sorted(pkg.rglob("*.py")):
            offenders.extend(self._bare_sites(module))

        assert not offenders, (
            "bare `._audit.log(...)` call site(s) outside the policy helpers — "
            "use Store._audit_log (pre-commit) or Store._audit_log_after_commit "
            "(post-commit): " + ", ".join(f"{m}:{ln} in {fn}" for m, ln, fn in offenders)
        )

    def test_the_pre_commit_helper_has_no_production_callers(self):
        """⛔ THE GUARD THAT WOULD HAVE CAUGHT THE SEVEN.

        ``_audit_log`` fires immediately outside a batch. That is correct for a
        PRE-commit site (a raise aborts the operation, which is what the caller
        should hear) and wrong for a post-commit one. Seven association methods
        used it at a post-commit position — they reach their commit through a
        free-function helper's ``commit=`` argument — and two of them DELETED
        edges and then raised a raw OSError.

        None of the three guards written that day could see them: the census
        scoped by the text ``self._audit.log``, the mechanical scan matches
        that same AST shape, and the behavioural table enumerated the methods
        the census produced. Three guards, one inherited scoping decision. The
        third guard in a row sharing a blind spot is not defence in depth.

        So: ``_audit_log`` now has ZERO production call sites, and this asserts
        it. A future site that genuinely needs pre-commit emission has to come
        back here and say so — which is exactly the judgment that went
        unstated last time. Cheaper and more exact than inferring each call's
        position relative to its commit.
        """
        import ast
        from pathlib import Path

        import anneal_memory

        pkg = Path(anneal_memory.__file__).parent
        callers = []
        for module in sorted(pkg.rglob("*.py")):
            tree = ast.parse(module.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_audit_log"
                ):
                    callers.append(f"{module.name}:{node.lineno}")

        assert not callers, (
            "`_audit_log` (the PRE-commit, batch-aware emit helper) has "
            "production call site(s): " + ", ".join(callers) + ". If the "
            "mutation is already committed or externalized at that point, use "
            "`_audit_log_after_commit` — a raise there reports a completed "
            "operation as failed. If the site is genuinely pre-commit, add it "
            "to this test's expected set WITH the reason, so the choice is on "
            "the record instead of inferred from the helper's name."
        )

    def test_the_scan_can_actually_see_a_bare_site(self, tmp_path):
        # Non-vacuity: the scan must FAIL on a module that contains one.
        probe = tmp_path / "probe.py"
        probe.write_text(
            "class X:\n"
            "    def some_method(self):\n"
            "        self._audit.log('evt', {})\n"
        )
        found = self._bare_sites(probe)
        assert found == [("probe.py", 3, "some_method")], found


class TestSwallowedWriteIsStillVisible:
    """The four channels a swallowed post-commit write reports on.

    ⚠ WHY THIS CLASS EXISTS SEPARATELY. ``TestAuditAfterCommitPolicy`` varies
    the sink outcome and the method, and holds CONSTANT the thing that reports
    it — every one of its assertions is ``pytest.warns``. So it cannot see a
    regression in any channel except the warning, and mutation proved it:
    deleting the ``note_write_failure`` call left all 96 tests green. The
    defect lives in the dimension the fixture held constant.
    """

    def _failing(self, tmp_path, sub):
        from anneal_memory.store import Store

        d = tmp_path / sub
        d.mkdir()
        store = Store(d / "memory.db")

        def boom(*args, **kwargs):
            raise OSError(28, "No space left on device")

        store._audit.log = boom
        return store

    def test_the_gap_rides_into_the_next_entry_that_lands(self, tmp_path):
        """⛔ Channel 1 — the only DURABLE one, and the reason it is needed.

        ``AuditTrail`` is write-first: chain state advances only after fsync,
        so a failed write leaves ``_prev_hash``/``_seq`` untouched and the next
        entry chains cleanly OVER the hole. Measured 2026-09-04: 8 mutations
        with 7 dropped writes produced ONE entry and ``verify()`` returned
        ``valid=True, chain_break_at=None`` — the gap was not merely
        undetectable-as-tampering, it was indistinguishable from the mutations
        never happening, under a verifier reporting a clean bill of health.
        """
        import json

        from anneal_memory.audit import AuditTrail
        from anneal_memory.store import Store

        d = tmp_path / "chained"
        d.mkdir()
        store = Store(d / "memory.db")
        real = store._audit.log

        def boom(*args, **kwargs):
            raise OSError(28, "No space left on device")

        store._audit.log = boom
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(3):
                store.record(f"dropped {i}", episode_type="observation")

        store._audit.log = real
        store.record("this one lands", episode_type="observation")

        entries = [
            json.loads(line)
            for line in store._audit._active_path.read_text().splitlines()
            if line.strip()
        ]
        landed = entries[-1]
        assert landed["dropped_before"] == 3, (
            "the three swallowed writes left no trace in the chain — "
            f"entry was {landed!r}"
        )
        # And it is a CHAINED fact, not a side note: the trail still verifies,
        # now while carrying the loss instead of hiding it.
        assert AuditTrail.verify(store._path).valid

        # The pending count resets, so the next entry does not double-report.
        store.record("and this one", episode_type="observation")
        tail = json.loads(store._audit._active_path.read_text().splitlines()[-1])
        assert "dropped_before" not in tail

    def test_status_reports_degraded_audit_health(self, tmp_path):
        """Channel 2 — the POLLABLE one.

        A warning must have been caught at the moment it fired; an agent that
        started later, or ran under ``-W error``, has no way to ask. ``status()``
        can be asked at any time.
        """
        store = self._failing(tmp_path, "status")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            store.record("one", episode_type="observation")
            store.record("two", episode_type="observation")

        status = store.status()
        assert status.audit_write_failures == 2
        assert status.audit_last_failure is not None
        assert "record" in status.audit_last_failure
        # The divergence that was previously computable-but-uncomputed.
        assert status.total_episodes == 2
        assert status.audit_entry_count == 0

    def test_the_logger_still_fires_when_warnings_are_errors(self, tmp_path, caplog):
        """Channel 3 — the one that survives ``-W error``.

        Measured 2026-09-04: under ``simplefilter("error")`` the warning path
        raises, the nested guard swallows it, and the caller sees ZERO signal.
        The logger is what remains.
        """
        import logging

        store = self._failing(tmp_path, "logged")
        with caplog.at_level(logging.WARNING, logger="anneal-memory"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                store.record("x", episode_type="observation")

        assert any(
            "audit write failed" in r.getMessage() for r in caplog.records
        ), f"no log record: {[r.getMessage() for r in caplog.records]}"

    def test_an_unconditional_commit_does_not_defer_its_audit(self, tmp_path):
        """⛔ AUDIT ORDERING MUST MATCH DURABILITY ORDERING, SITE BY SITE.

        ``prune`` / ``save_continuity`` / ``set_section_schema`` /
        ``wrap_started`` / ``wrap_cancelled`` commit or externalize
        UNCONDITIONALLY — they carry no ``_defer_commit`` guard. Routing them
        through a batch-AWARE helper queued their events while their work
        landed immediately, so a batch that then rolled back discarded the
        record of a mutation that had already happened, with no warning.
        Measured 2026-09-04 (L2/L1 agreeing independently); the pre-change
        bare emit wrote it. Hence ``batch_aware=False`` at those sites.
        """
        import json

        from anneal_memory.store import Store

        d = tmp_path / "inversion"
        d.mkdir()
        store = Store(d / "memory.db")

        with pytest.raises(RuntimeError):
            with store._batch():
                store.save_continuity("# externalized\n")
                raise RuntimeError("batch body fails AFTER the file landed")

        assert store.continuity_path.exists(), "the file did not externalize"
        events = [
            json.loads(line)["event"]
            for line in store._audit._active_path.read_text().splitlines()
            if line.strip()
        ]
        assert "continuity_saved" in events, (
            "the file was externalized but its audit event was queued behind a "
            f"batch that rolled back, and lost. events={events}"
        )


class TestDegradedAuditHealthReachesEveryTransport:
    """⛔ A FIELD ON A DATACLASS IS NOT A SURFACE — AND A GREP FOR ITS NAME IS
    NOT A TRANSPORT TEST.

    Both halves of the history are the lesson.

    2026-09-04 L4 found ``audit_write_failures`` reaching NONE of the three
    transports: the CLI ``--json`` payload builds its own ``audit`` sub-object,
    the CLI human output prints its own keys, and the MCP ``status`` handler
    composes its own line. Found by running the real CLI — not by any of the
    1812 tests then passing, all of which asked the Python API.

    ⛔ THE REGRESSION TEST WRITTEN FROM THAT LESSON DID NOT RUN THE CLI EITHER.
    It read cli.py off disk and asserted ``"status.audit_write_failures" in
    text``. MUTATION-PROVEN HOLLOW the same day: hardcoding
    ``"write_failures": 0`` in the --json payload with the token left alive in
    a COMMENT, plus ``if False:`` on the human branch, left all three tests
    PASSING and the full 1822-test suite green. A substring assertion is
    satisfied by a comment, a docstring, or a dead branch.

    ⚡ AND THE DEEPER HALF, which is why every test here crosses a PROCESS
    boundary instead of merely executing more code. The field was also
    structurally always zero on the CLI: it was a plain instance attribute,
    and a CLI invocation is a one-shot process that opens a Store, runs one
    subcommand and exits — so the surface an operator polls could never report
    non-zero however correctly cli.py read it. A test that degrades and
    asserts inside ONE Store cannot see that. So each test below loses the
    audit write in one Store, CLOSES it, and asks a different reader.
    """

    def _store_that_lost_audit_writes(self, db_path, count=2):
        """Commit ``count`` episodes whose audit writes are refused, then close.

        Returns with nothing live: the only record that anything was lost is
        whatever survived to disk. That is the property under test.
        """
        from anneal_memory.store import Store

        store = Store(db_path)

        def boom(*args, **kwargs):
            raise OSError(28, "No space left on device")

        store._audit.log = boom
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(count):
                store.record(f"lost-{i}", episode_type="observation")
        store.close()
        return db_path

    def _run_cli(self, argv):
        """Dispatch through the REAL parser and the REAL command function.

        In-process on purpose. The venv installs anneal_memory as a COPY in
        site-packages rather than an editable link, so a naive
        ``subprocess([sys.executable, "-m", "anneal_memory", ...])`` grades
        whatever was last installed instead of the tree under test — which is
        the very defect class this file exists to catch, one level out. Going
        through ``build_parser()`` keeps the argv path real; importing the
        module here keeps the SOURCE real.
        """
        from anneal_memory.cli import build_parser

        args = build_parser().parse_args(argv)
        args.func(args)

    def test_cli_json_status_reports_a_write_lost_by_an_EARLIER_process(
        self, tmp_path, capsys
    ):
        """The transport the README points operators at, across the boundary."""
        db = self._store_that_lost_audit_writes(tmp_path / "memory.db")

        self._run_cli(["--db", str(db), "status", "--json"])
        payload = json.loads(capsys.readouterr().out)

        audit = payload["audit"]
        assert audit["write_failures"] == 2, (
            "the CLI --json status reports a clean audit trail for a store "
            "that lost two audit writes. This is what an operator polls; "
            f"got {audit!r}"
        )
        assert audit["last_failure"] is not None
        assert "record" in audit["last_failure"]
        # The pair is the whole point: what the trail HAS beside what it LOST.
        # Reporting entry_count alone is how a hole reads as health.
        assert audit["entry_count"] == 0

    def test_cli_human_status_reports_a_write_lost_by_an_EARLIER_process(
        self, tmp_path, capsys
    ):
        db = self._store_that_lost_audit_writes(tmp_path / "memory.db")

        self._run_cli(["--db", str(db), "status"])
        out = capsys.readouterr().out

        assert "2 audit write(s) FAILED" in out, (
            "the human `status` output — the other surface an operator "
            f"actually reads — shows no degradation. got:\n{out}"
        )
        assert "record" in out

    def test_mcp_status_reports_a_write_lost_by_an_EARLIER_process(self, tmp_path):
        from anneal_memory.server import Server
        from anneal_memory.store import Store

        db = self._store_that_lost_audit_writes(tmp_path / "memory.db")

        reopened = Store(db)
        try:
            result = Server(reopened)._tool_status({})
            text = result["content"][0]["text"]
        finally:
            reopened.close()

        assert "2 write(s) FAILED" in text, (
            "an MCP agent asking status cannot see that the trail is "
            f"incomplete. got:\n{text}"
        )

    def test_the_count_is_monotonic_across_processes_and_never_resets(
        self, tmp_path
    ):
        """A lost entry is lost forever, so the number must not heal.

        ``verify()`` walks a clean chain over the hole and returns valid=True
        indefinitely, which is exactly why this counter may not reset: it is
        the only durable statement that the trail is incomplete.
        """
        from anneal_memory.store import Store

        db = self._store_that_lost_audit_writes(tmp_path / "memory.db", count=2)

        # A healthy session afterwards must not launder the earlier loss.
        store = Store(db)
        store.record("this one is fine", episode_type="observation")
        assert store.status().audit_write_failures == 2
        store.close()

        # And a further loss accumulates rather than replacing.
        self._store_that_lost_audit_writes(db, count=1)
        store = Store(db)
        try:
            assert store.status().audit_write_failures == 3
        finally:
            store.close()

    def test_a_read_only_handle_also_sees_the_loss(self, tmp_path):
        """A reader that reports audit health must report the write side too."""
        from anneal_memory.store import Store

        db = self._store_that_lost_audit_writes(tmp_path / "memory.db")

        reader = Store(db, read_only=True)
        try:
            assert reader.status().audit_write_failures == 2
        finally:
            reader.close()

    def test_every_transport_that_reports_audit_health_reports_the_write_side(self):
        """A SECONDARY NET, AND LABELLED AS ONE — it is not the guard.

        Any module reporting ``audit_entry_count`` (what the trail HAS) must
        also report ``audit_write_failures`` (what it LOST). Reporting only the
        first is how a trail missing entries reads as healthy.

        ⚠ This is a source scan and therefore CANNOT see whether the reporting
        works — that is measured by the four behavioural tests above, and the
        2026-09-04 mutation showed a scan like this one passing over a
        hardcoded zero. What it CAN do that they cannot is notice a NEW module
        that starts reporting audit health and forgets the write side. Keep it
        for that reach; never read a pass here as coverage.
        """
        import anneal_memory

        pkg = Path(anneal_memory.__file__).parent
        offenders = []
        for module in sorted(pkg.rglob("*.py")):
            text = module.read_text(encoding="utf-8")
            if "audit_entry_count" not in text:
                continue
            if module.name in {"types.py", "store.py"}:
                continue  # the definition and the producer, not reporters
            if "audit_write_failures" not in text:
                offenders.append(module.name)
        assert not offenders, (
            "module(s) report what the audit trail HAS without reporting what "
            "it LOST: " + ", ".join(offenders) + ". A swallowed write is "
            "invisible to verify(); this is the only surface it is visible on."
        )


class TestCodexL3TwentySixOhNineOhFour:
    """Seven defects codex found in code I had shipped AND mutation-tested.

    ⛔ WHY THESE EXIST AS A BLOCK: every one of them lived in the audit-health
    persistence or the schema guard I wrote on 2026-09-04, both of which I had
    already pinned with mutation-checked tests. The mutations passed because I
    mutated the path I HAD IN MIND. codex drove the paths I had not.
    """

    def _boom(self, *args, **kwargs):
        raise OSError(28, "No space left on device")

    # -- #1: the standalone commit must not publish a caller's transaction --

    def test_a_health_write_never_commits_someone_elses_open_batch(self, tmp_path):
        """``save_continuity`` is NOT batch-aware, and that is the whole bug.

        My earlier test drove ``record()`` inside a batch — which DEFERS its
        audit write, so the failure handler never ran and the commit never
        happened. ``save_continuity`` logs immediately, so its audit failure
        committed the batch's uncommitted DML. Measured: two episodes survived
        a batch that rolled back.
        """
        from anneal_memory.store import Store

        db = tmp_path / "memory.db"
        store = Store(db)
        store.record("seed", episode_type="observation")
        store.close()

        store = Store(db)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(RuntimeError):
                    with store._batch():
                        store.record("inside the batch", episode_type="observation")
                        store._audit.log = self._boom
                        store.save_continuity("# not batch aware\n")
                        raise RuntimeError("force rollback")
        finally:
            store.close()

        reopened = Store(db)
        try:
            assert reopened.status().total_episodes == 1, (
                "the audit-health commit published DML from a batch that then "
                "rolled back"
            )
        finally:
            reopened.close()

    def test_a_health_write_never_destroys_a_batch_that_SUCCEEDS(self, tmp_path):
        """The discriminator the rolled-back version could not provide.

        ⚠ Mutation-driven. Reverting the ``in_transaction`` guard left the
        rolled-back test PASSING, because the mutant's failure mode —
        ``BEGIN IMMEDIATE`` raising inside an open transaction, then the
        handler's own ``rollback()`` destroying the caller's work — produces
        the SAME observable as a batch that was going to roll back anyway.
        Both end with the DML gone.

        So drive a batch that COMMITS. Correct code leaves the delta pending
        and the batch intact; the mutant rolls the caller's transaction back
        underneath it and the episode vanishes from a batch that reported
        success. A test whose scenario cannot separate the two outcomes is not
        a test of the guard.
        """
        from anneal_memory.store import Store

        db = tmp_path / "memory.db"
        store = Store(db)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with store._batch():
                    store.record("must survive", episode_type="observation")
                    store._audit.log = self._boom
                    store.save_continuity("# not batch aware\n")
                # batch exits normally — its DML must be committed
            assert store.status().total_episodes == 1, (
                "a batch that completed successfully lost its DML — the "
                "audit-health write rolled back the caller's transaction"
            )
        finally:
            store.close()

        reopened = Store(db)
        try:
            assert reopened.status().total_episodes == 1
        finally:
            reopened.close()

    # -- #4b: the deferred delta must actually reach disk (2026-09-05) --

    def test_a_delta_deferred_inside_a_batch_survives_the_process(self, tmp_path):
        """The property the test above drives the path of and never asserts.

        ⛔ THE DEFECT THIS PINS, found by Diogenes 2026-09-05 and reproduced
        before the fix: ``_persist_audit_health`` correctly refuses to commit
        inside a caller's transaction and leaves the delta pending — and until
        2026-09-05 the ONLY thing that ever flushed it was a LATER audit
        failure landing outside a transaction. Its own comment named three
        flush points; ``grep`` returned one. So a store really lost an audit
        write, reported it correctly in-process, and reported ZERO after
        reopen — while README.md, types.py and CHANGELOG.md all state the
        field is durable and lifetime-scoped, on the batched path
        ``validated_save_continuity`` actually uses.

        ⚡ WHY THE SIBLING ABOVE CANNOT CATCH IT: it drives this exact batched
        scenario and then asserts ``total_episodes == 1``. It grades whether
        the CALLER'S DML survived — the guard's other half — and never asks
        whether the count landed. The batch case was exercised and the batch
        case's own property was not.

        THE ONE EXTRA BEAT that separates this from the sibling: the sink
        HEALS before the batch exits (a transient ENOSPC; the disk is freed).
        Without it the deferred ``record()`` audit replays, fails again
        OUTSIDE the transaction, and THAT handler flushes both deltas — the
        loss needs the last audit failure of the process to be one the guard
        deferred.

        MUTATION-CHECKED: removing the ``_persist_audit_health()`` at
        ``_batch()`` exit returns after-reopen to 0 and this test to red.
        """
        from anneal_memory.store import Store

        db = tmp_path / "memory.db"
        store = Store(db)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with store._batch():
                    store.record("must survive", episode_type="observation")
                    healthy = store._audit.log
                    store._audit.log = self._boom
                    store.save_continuity("# not batch aware\n")
                    store._audit.log = healthy  # the sink heals mid-batch
            assert store._audit_failures_unpersisted == 0, (
                "the batch exited with a degraded-audit delta still pending — "
                "nothing after this point is guaranteed to run"
            )
        finally:
            store.close()

        reopened = Store(db)
        try:
            status = reopened.status()
            assert status.audit_write_failures == 1, (
                "a lost audit write did not survive the process that saw it. "
                "audit_write_failures is documented as durable and "
                "lifetime-scoped (README.md, types.py, CHANGELOG.md) and the "
                "canonical wrap pipeline is batched"
            )
            assert status.audit_last_failure is not None
            assert "save_continuity" in status.audit_last_failure
            assert status.total_episodes == 1, (
                "the flush published or destroyed the batch's own DML"
            )
        finally:
            reopened.close()

    def test_close_is_the_last_flush_point_when_the_batch_never_reaches_its(
        self, tmp_path
    ):
        """A batch that RAISES skips its own flush; ``close()`` is all that is left.

        The flush at ``_batch()`` exit sits after the deferred-audit replay,
        which a propagating exception never reaches. That is correct — the
        rollback path must not linger — but it means the batch-exit flush is
        not a total guarantee, and the delta is real either way: an audit
        write was attempted and lost, whether or not the caller's DML
        survived. ``close()`` is the process's last chance to write it down.

        MUTATION-CHECKED: removing the ``_persist_audit_health()`` in
        ``close()`` returns after-reopen to 0 and this test to red, while the
        test above stays green — the two flush points are pinned separately.
        """
        from anneal_memory.store import Store

        db = tmp_path / "memory.db"
        store = Store(db)
        store.record("seed", episode_type="observation")
        store.close()

        store = Store(db)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(RuntimeError):
                    with store._batch():
                        store.record("inside", episode_type="observation")
                        healthy = store._audit.log
                        store._audit.log = self._boom
                        store.save_continuity("# not batch aware\n")
                        store._audit.log = healthy
                        raise RuntimeError("force rollback")
                assert store._audit_failures_unpersisted == 1, (
                    "scenario no longer reaches close() with a pending delta "
                    "— it is not testing the close() flush point any more"
                )
        finally:
            store.close()

        reopened = Store(db)
        try:
            assert reopened.status().audit_write_failures == 1, (
                "the delta died with the process — close() did not flush"
            )
        finally:
            reopened.close()

    # -- #4c: a post-commit BaseException must not reach the caller (09-05) --

    def test_a_keyboardinterrupt_in_the_replay_cannot_look_like_a_failed_batch(
        self, tmp_path
    ):
        """codex L3 HIGH, 2026-09-05 — the data-loss path the guard walked past.

        ⛔ ``_audit_log_after_commit`` swallows with ``except Exception``. A
        ``KeyboardInterrupt`` raised while replaying a deferred audit therefore
        escaped ``_batch()`` AFTER its outer commit had already landed. The
        caller cannot tell that from a batch that failed to commit — and
        ``validated_save_continuity`` does not try: it sets ``db_committed =
        True`` only after the ``with`` block returns, and its ``except
        BaseException`` then unlinks BOTH staged sidecars. The comment on that
        cleanup says removing them "would destroy committed state permanently".
        It was written for ``Exception``.

        A Ctrl+C during a CLI wrap reaches this window, and the loss is the
        new continuity text while the wrap row and the episodes' wrap
        assignments stay durable.

        MUTATION-CHECKED, and the failure MODE is the point: narrowing the
        post-commit ``except BaseException`` in ``_batch()`` back to ``except
        Exception`` does not turn this test red — the interrupt escapes the
        ``with`` block and ABORTS the pytest run at that line. A caller has no
        more defence against it than the test runner does, which is the whole
        argument for containing it at the source.
        """
        from anneal_memory.store import Store

        db = tmp_path / "memory.db"
        store = Store(db)

        def interrupt(*a, **k):
            raise KeyboardInterrupt("user hit Ctrl+C during the audit replay")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with store._batch():
                    store.record("committed work", episode_type="observation")
                    store._audit.log = interrupt
            # Reaching here at all is the property: the batch exited normally,
            # so a caller's ``db_committed = True`` runs and its cleanup does
            # not fire.
            assert store.status().total_episodes == 1
        finally:
            store._audit.log = None
            store.close()

        reopened = Store(db)
        try:
            assert reopened.status().total_episodes == 1, (
                "the committed episode did not survive — a post-commit "
                "interrupt was allowed to look like a failed batch"
            )
        finally:
            reopened.close()

    def test_an_interrupt_mid_health_write_does_not_hold_the_writer_lock(
        self, tmp_path
    ):
        """The second half of the same HIGH, in ``_persist_audit_health``.

        Its rollback sat under ``except Exception``, so an interrupt landing
        between ``BEGIN IMMEDIATE`` and ``commit()`` left the transaction OPEN
        — holding SQLite's writer lock against every other process, which is
        the exact failure that rollback exists to prevent.

        MUTATION-CHECKED: narrowing that handler back to ``except Exception``
        makes the interrupt escape ``_persist_audit_health()`` itself and abort
        the run at that call — with the transaction still open behind it. The
        sibling test above stays unaffected, so the two are pinned separately.
        """
        from anneal_memory.store import Store

        db = tmp_path / "memory.db"
        store = Store(db)
        try:
            store._audit_failures_unpersisted = 1
            store._audit_last_failure = "synthetic"

            real_conn = store._conn

            class _InterruptOnCommit:
                # sqlite3.Connection.commit is read-only, so proxy instead of
                # patching. Everything else delegates to the real connection,
                # including ``in_transaction`` — which is the thing under test.
                def __init__(self, conn):
                    self._conn = conn

                def __getattr__(self, name):
                    return getattr(self._conn, name)

                def commit(self, *a, **k):
                    raise KeyboardInterrupt("Ctrl+C between BEGIN and COMMIT")

            store._conn = _InterruptOnCommit(real_conn)
            store._persist_audit_health()  # must not raise
            store._conn = real_conn

            assert not real_conn.in_transaction, (
                "an interrupt left the health transaction open — the writer "
                "lock is held against every other process"
            )
            assert store._audit_failures_unpersisted == 1, (
                "the delta was cleared despite the write never committing"
            )
        finally:
            store.close()

    # -- #4d: nor may the last-failure POINTER go backwards (codex, 09-05) --

    def test_a_stale_pending_failure_cannot_overwrite_a_newer_persisted_one(
        self, tmp_path
    ):
        """codex L3 MED, 2026-09-05 — opened by that morning's own HIGH fix.

        Writer A defers a failure inside a transaction; writer B persists a
        newer one; A then closes and its flush wrote A's OLDER string over B's
        via ``INSERT OR REPLACE``. The count is additive and unaffected — this
        is the forensic pointer naming the wrong final loss.

        ⚡ The window is NEW as of 2026-09-05. Before the ``_batch()``-exit and
        ``close()`` flush points were added that morning, the only flush ran
        inside the failure handler microseconds after the string was assigned,
        so a flush could never carry a stale value. A fix reintroducing a
        neighbour of the class it closed is why L3 runs after the fix and not
        before it.

        MUTATION-CHECKED: restoring the unconditional ``INSERT OR REPLACE``
        makes the final assertion fail with A's older record in the row.
        """
        from anneal_memory.store import Store

        db = tmp_path / "memory.db"
        seed = Store(db)
        seed.record("seed", episode_type="observation")
        seed.close()

        a = Store(db)
        try:
            # A holds an OLD pending failure it could not commit (the
            # in_transaction guard) — synthesised at a stamp that is
            # unambiguously earlier than B's.
            a._audit_failures_unpersisted = 1
            a._audit_last_failure = (
                "record: OSError(28, 'No space left on device') "
                "[dropped before audit seq 1] at 2026-09-05T10:00:00Z"
            )

            b = Store(db)
            try:
                b._audit_failures_unpersisted = 1
                b._audit_last_failure = (
                    "save_continuity: OSError(28, 'No space left on device') "
                    "[dropped before audit seq 4] at 2026-09-05T11:00:00Z"
                )
                b._persist_audit_health()
            finally:
                b.close()

            a._persist_audit_health()  # A's close-time flush, with a stale value
        finally:
            a.close()

        reopened = Store(db)
        try:
            status = reopened.status()
            assert status.audit_write_failures == 2, (
                "the additive count lost an update"
            )
            assert status.audit_last_failure is not None
            assert "11:00:00Z" in status.audit_last_failure, (
                "a stale pending failure overwrote a newer persisted one — "
                f"the row reads {status.audit_last_failure!r}"
            )
        finally:
            reopened.close()

    # -- #4e: the 3.10 fallback must recognise SQLITE_LOCKED (codex, 09-05) --

    def test_the_textual_fallback_recognises_every_measured_lock_message(self):
        """The branch three Diogenes nights named as never exercised here.

        ``_is_write_lock_contention`` classifies by primary result code when
        ``sqlite_errorcode`` exists — Python 3.11+. ``requires-python`` is
        >=3.10, and on 3.10 the TEXT fallback is the whole classifier. It read
        ``"database is locked" or "database is busy"``.

        ⛔ MEASURED on live connections, with the codes captured alongside, not
        reasoned from the docs:
          SQLITE_BUSY (5)                 -> 'database is locked'
          SQLITE_LOCKED_SHAREDCACHE (262) -> 'database table is locked: sqlite_master'
        The second matched NEITHER clause, so real contention was classified as
        not-contention: the CLI rethrew a low-level ``StoreDatabaseError`` and
        MCP missed its contention response.

        ⚠ The negative cases are the point of the ``"database"`` conjunct. The
        measured message carries an OBJECT NAME, so a bare ``"locked" in text``
        would also fire on an unrelated error that happens to name a table
        called ``locked_items``.
        """
        import sqlite3
        from anneal_memory.store import StoreDatabaseError, _is_write_lock_contention

        def _NoCode(msg):
            # The function reads ``exc.__cause__``, never the wrapper's own
            # message (which embeds the store PATH). A hand-built
            # OperationalError carries no ``sqlite_errorcode`` — the C layer
            # sets it — so this exercises the TEXT branch on any interpreter,
            # which is the 3.10 path the repo supports and never runs here.
            cause = sqlite3.OperationalError(msg)
            err = StoreDatabaseError("wrapped", operation="record")
            err.__cause__ = cause
            return err

        contention = [
            "database is locked",                          # SQLITE_BUSY, measured
            "database table is locked: sqlite_master",      # SQLITE_LOCKED_SHAREDCACHE, measured
            "database schema is locked",                    # SQLITE_LOCKED, documented
            "database is busy",
        ]
        for msg in contention:
            exc = _NoCode(msg)
            assert not isinstance(
                getattr(exc.__cause__, "sqlite_errorcode", None), int
            ), "this test must exercise the TEXT branch, not the code branch"
            assert _is_write_lock_contention(exc), (
                f"real lock contention classified as not-contention: {msg!r}"
            )

        not_contention = [
            "no such table: locked_items",   # names a table, is not a lock
            "attempt to write a readonly database",
            "unable to open database file",
            "cannot start a transaction within a transaction",
        ]
        for msg in not_contention:
            assert not _is_write_lock_contention(_NoCode(msg)), (
                f"non-contention classified as contention: {msg!r}"
            )

    # -- #4f: DIRECTION, not just frequency, for every conditional write --

    @pytest.mark.parametrize(
        "stored,expect_kept",
        [
            ("record: X [dropped before audit seq 1] at 2099-01-01T00:00:00Z", True),
            ("record: X [dropped before audit seq 1] at 2020-01-01T00:00:00Z", False),
            ("record: OSError(28) written before the stamp format existed", False),
            (None, False),
        ],
        ids=["stored-newer", "stored-older", "stored-legacy-unstamped", "stored-absent"],
    )
    def test_the_last_failure_write_fires_in_the_right_DIRECTION(
        self, tmp_path, stored, expect_kept
    ):
        """⚖ The lens this test exists for, named by ``0905+1 fanin`` 2026-09-05:

        **A conditional write has TWO questions — how often it fires, and WHICH
        WAY it fires — and reviewing only the first is the checkable-proxy
        class.** That is not hypothetical here. The `format_version` stamp
        shipped the same day with the predicate ``metadata.value IS NOT
        excluded.value``, which reads as "only when it changed" and actually
        says "whenever they differ, in EITHER direction". It was reviewed for
        frequency, ratified, and wrote a version marker BACKWARDS.

        So this pins the OTHER conditional write introduced that day — the
        ``audit_last_failure`` guard — as an explicit direction matrix rather
        than a "does it skip the redundant write" assertion. Every row was
        MEASURED before being written down.

        The one fail-open branch (a candidate with no parseable stamp
        overwrites a stamped stored value) is structurally unreachable: a write
        requires a pending delta, a delta requires a failure, and every failure
        restamps ``_audit_last_failure`` with a fresh ``at <ISO-Z>``. Recorded
        rather than guarded, because unreachability is not a property to lean
        on silently — see ``_RESERVED_AUDIT_KWARGS``, which makes the same
        argument in the other direction.
        """
        import sqlite3

        from anneal_memory.store import Store

        db = tmp_path / "m.db"
        seed = Store(db)
        seed.record("seed", episode_type="observation")
        seed.close()

        if stored is not None:
            conn = sqlite3.connect(db)
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) "
                "VALUES ('audit_last_failure', ?)",
                (stored,),
            )
            conn.commit()
            conn.close()

        writer = Store(db)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            writer._audit.log = self._boom
            writer.record("provokes a real, freshly stamped failure",
                          episode_type="observation")
        writer.close()

        reopened = Store(db)
        try:
            final = reopened.status().audit_last_failure
        finally:
            reopened.close()

        if expect_kept:
            assert final == stored, (
                "a stale pending failure overwrote a NEWER persisted one"
            )
        else:
            assert final != stored and "No space left on device" in (final or ""), (
                "the newer failure did not replace an older/unorderable record"
            )

    # -- #4g: SystemExit is not KeyboardInterrupt (complement L3, 09-05) --

    def test_systemexit_propagates_but_ctrl_c_is_still_swallowed(self, tmp_path):
        """⚖ complement L3 MED — conflating the two was the morning's mistake.

        Widening ``_persist_audit_health``'s handler to ``BaseException`` was
        correct for ``KeyboardInterrupt``: swallowing Ctrl-C for a few
        statements protects a committed wrap whose staged sidecars would
        otherwise be unlinked. **``SystemExit`` is a different fail-open with
        no equivalent justification.** A long-lived MCP server whose SIGTERM
        handler calls ``sys.exit(0)`` would run on past the point something
        explicitly told it to stop, and an orchestrator waiting out a grace
        period would SIGKILL it instead.

        ⚠ THE WRAP STAYS PROTECTED, which is why the re-raise lives here and
        NOT in ``_batch()``: on the batched path it is caught by ``_batch()``'s
        own post-commit ``except BaseException``. What changes is ``close()``,
        where nothing is staged and the caller genuinely is exiting.

        ⛔ Both cases must also leave NO OPEN TRANSACTION — the rollback is the
        reason the handler was widened in the first place, and an early
        ``raise`` that skipped it would hold SQLite's writer lock against every
        other process.
        """
        from anneal_memory.store import Store

        def _store_with_a_pending_delta(db):
            store = Store(db)
            store.record("seed", episode_type="observation")
            store._audit_failures_unpersisted = 1
            store._audit_last_failure = "synthetic at 2026-09-05T10:00:00Z"
            return store

        class _Raises:
            """Proxy: sqlite3.Connection.commit is read-only, so wrap it."""

            def __init__(self, conn, exc):
                self._conn = conn
                self._exc = exc

            def __getattr__(self, name):
                return getattr(self._conn, name)

            def commit(self, *a, **k):
                raise self._exc

        # (1) KeyboardInterrupt — swallowed, as the wrap-protection argument requires.
        store = _store_with_a_pending_delta(tmp_path / "a.db")
        try:
            real = store._conn
            store._conn = _Raises(real, KeyboardInterrupt("ctrl-c"))
            store._persist_audit_health()  # must NOT raise
            store._conn = real
            assert not real.in_transaction, "Ctrl-C left the writer lock held"
        finally:
            store.close()

        # (2) SystemExit — propagates, and still rolls back first.
        store = _store_with_a_pending_delta(tmp_path / "b.db")
        try:
            real = store._conn
            store._conn = _Raises(real, SystemExit(0))
            with pytest.raises(SystemExit):
                store._persist_audit_health()
            store._conn = real
            assert not real.in_transaction, (
                "the SystemExit re-raise skipped the rollback and left the "
                "writer lock held against every other process"
            )
        finally:
            store.close()

    # -- #5: the count must never go backwards between writers --

    def test_two_writers_cannot_make_the_lifetime_count_decrease(self, tmp_path):
        """codex's exact interleaving: both seed 5, one persists 6 then 7, the
        other persists its stale 6. A whole-value write loses an update; an
        UPSERT that adds a delta cannot."""
        from anneal_memory.store import Store

        db = tmp_path / "memory.db"
        seed = Store(db)
        seed.record("seed", episode_type="observation")
        seed.close()

        a, b = Store(db), Store(db)
        a._audit.log = self._boom
        b._audit.log = self._boom
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a.record("a1", episode_type="observation")
                a.record("a2", episode_type="observation")
                b.record("b1", episode_type="observation")
        finally:
            a.close()
            b.close()

        reopened = Store(db)
        try:
            assert reopened.status().audit_write_failures == 3, (
                "three writes were lost; a stale-seeded writer overwrote the "
                "count instead of adding to it"
            )
        finally:
            reopened.close()

    # -- #6: a long-lived handle must not report a constructor-time cache --

    def test_a_long_lived_reader_sees_a_loss_from_another_process(self, tmp_path):
        from anneal_memory.store import Store

        db = tmp_path / "memory.db"
        seed = Store(db)
        seed.record("seed", episode_type="observation")
        seed.close()

        reader = Store(db, read_only=True)
        try:
            assert reader.status().audit_write_failures == 0

            writer = Store(db)
            writer._audit.log = self._boom
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                writer.record("lost", episode_type="observation")
            writer.close()

            assert reader.status().audit_write_failures == 1, (
                "a long-lived handle reports the count it cached at "
                "construction, so an MCP server or reader is permanently stale"
            )
        finally:
            reader.close()

    # -- #8: a failed health transaction must not keep the writer lock --

    def test_a_failed_health_write_rolls_back_and_keeps_the_delta(self, tmp_path):
        import sqlite3

        from anneal_memory.store import Store

        db = tmp_path / "memory.db"
        seed = Store(db)
        seed.record("seed", episode_type="observation")
        seed.close()

        # Abort the SECOND statement, so the failure lands mid-transaction.
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TRIGGER boom_on_last_failure BEFORE INSERT ON metadata "
            "WHEN NEW.key = 'audit_last_failure' "
            "BEGIN SELECT RAISE(ABORT, 'simulated mid-transaction failure'); END;"
        )
        conn.commit()
        conn.close()

        store = Store(db)
        store._audit.log = self._boom
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                store.record("triggers a failed health write", episode_type="observation")

            assert store._conn.in_transaction is False, (
                "a failed health transaction was left open, holding SQLite's "
                "writer lock against every other process until close"
            )
            assert store._audit_failures_unpersisted == 1, (
                "the delta was cleared despite the commit failing, so the "
                "count cannot be recovered by a later flush"
            )

            other = sqlite3.connect(db, timeout=1.0)
            try:
                other.execute("BEGIN IMMEDIATE")
                other.rollback()
            finally:
                other.close()
        finally:
            store.close()


class TestTheGapLocationSurvivesTheProcess:
    """spore-745, the half that could be closed without an outbox.

    The hash-chained ``dropped_before`` marker pins WHERE a gap sits, and it
    becomes durable only once a later write lands IN THE SAME PROCESS. A close
    or crash before that loses it and ``verify()`` walks cleanly over the hole.
    The spore asked for a transactional outbox.

    ⛔ AN OUTBOX WAS NOT BUILT, AND THE REASONING IS THE POINT. What the
    chained marker uniquely provides is TAMPER-EVIDENCE — it is inside the
    hash chain. An outbox staged in the SQLite metadata table is not chained
    either, so it does not deliver that property during the window it exists;
    it delivers DURABILITY OF A LOCATION. That is obtainable far more cheaply,
    because ``note_write_failure`` already knows the seq and was discarding it.

    ⚡ ALSO FALSIFIED BEFORE BUILDING: "advance ``_seq`` on a dropped write so
    verify sees a numeric gap" looks cheaper still and buys NOTHING —
    ``_initialize`` recovers ``_seq`` from the last entry ON DISK
    (``last_entry["seq"] + 1``), so a reopen erases the gap. Identical
    durability to the marker it was meant to replace.

    So the location now rides the durable ``audit_last_failure`` record, and
    the chained marker keeps its own separate job. Both are reported; neither
    pretends to be the other.
    """

    def test_the_location_outlives_the_process_that_saw_it(self, tmp_path):
        from anneal_memory.store import Store

        db = tmp_path / "memory.db"
        store = Store(db)
        store.record("landed-0", episode_type="observation")
        store.record("landed-1", episode_type="observation")

        real = store._audit.log

        def boom(*args, **kwargs):
            raise OSError(28, "No space left on device")

        store._audit.log = boom
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            store.record("this one loses its audit write", episode_type="observation")
        store._audit.log = real
        store.close()

        reopened = Store(db)
        try:
            last = reopened.status().audit_last_failure
        finally:
            reopened.close()

        assert last is not None, "the failure record did not survive the process"
        assert "dropped before audit seq 2" in last, (
            "the durable record says a write was lost but not WHERE, so an "
            f"operator cannot locate the gap in the chain. got: {last!r}"
        )

    def test_the_wording_matches_the_chained_marker_rather_than_contradicting_it(
        self, tmp_path
    ):
        """"dropped before seq N", never "missing seq N".

        ``_seq`` advances only on a SUCCESSFUL append, so the next entry that
        lands REUSES the number the dropped one was assigned. Phrasing it as
        "missing seq N" would point an operator at an entry that exists — and
        would contradict the chained marker sitting on that very entry.
        """
        from anneal_memory.store import Store

        db = tmp_path / "memory.db"
        store = Store(db)
        try:
            store.record("landed-0", episode_type="observation")
            store.record("landed-1", episode_type="observation")
            real = store._audit.log

            def boom(*args, **kwargs):
                raise OSError(28, "No space left on device")

            store._audit.log = boom
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                store.record("lost", episode_type="observation")
            store._audit.log = real
            store.record("landed-after", episode_type="observation")

            last = store.status().audit_last_failure
        finally:
            store.close()

        assert "missing audit seq" not in last

        # The entry the chained marker rode in on must be the same seq the
        # durable record names — the two records agree by construction.
        entries = [
            json.loads(line)
            for line in (db.parent / f"{db.stem}.audit.jsonl").read_text().splitlines()
            if line.strip()
        ]
        carrier = [e for e in entries if e.get("dropped_before")]
        assert carrier, "no chained marker was written at all"
        assert f"dropped before audit seq {carrier[0]['seq']}" in last


class TestAFailedRotationDoesNotLeaveAFalseTamperingVerdict:
    """"Active file missing" is not proof the rotation succeeded (spore-746).

    Rotation renames the active file FIRST, then gzips, then updates the
    manifest. If a later step raises — disk full during the gzip is the
    measured case — the sealed file exists, the manifest does not know about
    it, and the active file is gone. The next call arrived at the
    "active missing" branch and simply advanced ``_last_week``, recording a
    rotation that never completed; the following append then started a fresh
    file chaining to the orphan's hash.

    ⛔ THE COST IS A FALSE TAMPERING VERDICT, NOT A LOST FILE. Measured
    2026-09-04, same process, no crash: ``verify()`` returned
    ``valid=False, "Hash mismatch at seq 3: expected sha256:GENESIS..."`` on a
    store where nothing had been tampered with and no data was lost.

    ⚠ AND THE RECOVERY COULD NOT BE REACHED BY THE COMMAND AN OPERATOR RUNS.
    ``_adopt_orphaned_files`` already existed and is idempotent, but only ran
    from ``_initialize``. ``AuditTrail.verify`` is a CLASSMETHOD and never
    constructs a trail, so ``anneal-memory verify`` — precisely what someone
    runs when they suspect tampering — could not trigger it. The store read as
    tampered until some unrelated operation happened to open it.
    """

    def _trail_whose_rotation_failed(self, tmp_path):
        import gzip as gzip_mod

        import anneal_memory.audit as audit_mod
        from anneal_memory.audit import AuditTrail

        db = tmp_path / "m.db"
        trail = AuditTrail(db)
        for i in range(3):
            trail.log("record", {"i": i})

        # Make the next append cross a week boundary, then break the gzip so
        # rotation dies AFTER the rename and BEFORE the manifest update.
        trail._last_week = "2026-W01"
        real_open = gzip_mod.open

        def boom(*args, **kwargs):
            raise OSError(28, "No space left on device")

        audit_mod.gzip.open = boom
        try:
            with pytest.raises(OSError):
                trail.log("record", {"i": "during the failed rotation"})
        finally:
            audit_mod.gzip.open = real_open
        return db, trail

    def test_the_orphan_is_adopted_by_the_process_that_created_it(self, tmp_path):
        from anneal_memory.audit import AuditTrail

        db, trail = self._trail_whose_rotation_failed(tmp_path)

        # The sealed orphan exists and the manifest does not know it yet.
        assert (tmp_path / "m.audit.2026-W01.jsonl").exists()
        assert trail._load_manifest()["files"] == []

        # The next append in the SAME process must repair rather than paper over.
        trail.log("record", {"i": "after"})

        assert [f["filename"] for f in trail._load_manifest()["files"]] == [
            "m.audit.2026-W01.jsonl"
        ], "the orphaned sealed file was never adopted; the manifest still omits it"

        result = AuditTrail.verify(db)
        assert result.valid is True, (
            "verify reports a broken chain after a failed rotation that lost "
            f"no data — a tampering-shaped verdict from a disk error. {result.error}"
        )

    def test_a_reopen_still_recovers_too(self, tmp_path):
        """The pre-existing path must keep working — the fix adds, not replaces."""
        from anneal_memory.audit import AuditTrail

        db, _ = self._trail_whose_rotation_failed(tmp_path)
        AuditTrail(db).log("record", {"i": "reopened"})
        assert AuditTrail.verify(db).valid is True


class TestAnInvalidTrailStillReportsWhatItCouldNotRead:
    """A tampering verdict must not also claim the file was fully readable.

    ``AuditTrail.verify`` counts malformed lines it skips, and the SUCCESS
    return has always carried that count out. The chain-break return —
    the one INSIDE the counting loop — omitted ``skipped_lines``, so the
    dataclass default of 0 overwrote a number already incremented. An operator
    investigating "possible tampering" was told zero lines were unreadable
    while unreadable lines sat in the very file they were being asked to
    distrust.

    ⚠ Unreadable lines are a COMPETING EXPLANATION for a chain break, not a
    footnote to it: a truncated write and a malicious edit both produce a
    break, and the skipped count is part of telling them apart.

    ⛔ Behavioural, not a scan over the return sites. Three of the four
    early returns in ``verify`` legitimately omit the field — they run before
    ``skipped`` exists — so a structural "every construction must pass it"
    assertion would be WRONG, and a scan tuned to exempt them would encode
    today's line numbers. Drive the real function instead.
    """

    def _trail_with(self, tmp_path, malformed: bool, break_chain: bool):
        from anneal_memory.store import Store

        db = tmp_path / "memory.db"
        store = Store(db)
        for i in range(3):
            store.record(f"ep-{i}", episode_type="observation")
        store.close()

        audit = db.parent / f"{db.stem}.audit.jsonl"
        lines = audit.read_text(encoding="utf-8").splitlines()
        if malformed:
            lines.insert(1, "{ this line is not json")
        if break_chain:
            last = json.loads(lines[-1])
            last["prev_hash"] = "0" * 64
            lines[-1] = json.dumps(last)
        audit.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return db

    def test_a_chain_break_carries_the_skipped_count_out(self, tmp_path):
        from anneal_memory.audit import AuditTrail

        db = self._trail_with(tmp_path, malformed=True, break_chain=True)
        result = AuditTrail.verify(db)

        assert result.valid is False
        assert result.skipped_lines == 1, (
            "verify reported a chain break and skipped_lines=0 while a "
            "malformed line was present. The count is incremented and then "
            "discarded by the invalid return, so an operator investigating a "
            "tampering verdict is told the file was fully readable."
        )

    def test_the_valid_path_still_carries_it(self, tmp_path):
        """The half that already worked — pinned so a fix cannot trade one for the other."""
        from anneal_memory.audit import AuditTrail

        db = self._trail_with(tmp_path, malformed=True, break_chain=False)
        result = AuditTrail.verify(db)

        assert result.valid is True
        assert result.skipped_lines == 1

    def test_the_cli_tells_the_operator_on_BOTH_paths(self, tmp_path, capsys):
        """The dataclass is not the surface — this is the operator's actual view."""
        from anneal_memory.cli import build_parser

        broken = self._trail_with(tmp_path / "a", malformed=True, break_chain=True)
        args = build_parser().parse_args(["--db", str(broken), "verify"])
        with pytest.raises(SystemExit):
            args.func(args)
        err = capsys.readouterr().err
        assert "malformed" in err, (
            "the CLI's INVALID branch prints the chain break and says nothing "
            f"about unreadable lines. got:\n{err}"
        )

        ok = self._trail_with(tmp_path / "b", malformed=True, break_chain=False)
        args = build_parser().parse_args(["--db", str(ok), "verify"])
        args.func(args)
        assert "malformed" in capsys.readouterr().out


class TestThePersistCommitDoesNotLeakOtherWork:
    """The durable counter writes with a standalone ``commit()``. Prove it is safe.

    ``_persist_audit_health`` runs inside the post-commit audit-failure handler
    and issues its own ``INSERT OR REPLACE`` + ``commit()``. That is only
    correct if there is genuinely no in-flight transaction at that point — and
    "there isn't one" was an ARGUMENT in a docstring, which is the weakest kind
    of claim in this repo. A stray commit here would publish another method's
    uncommitted DML, and the wrap state machine's whole invariant is that its
    metadata writes share ONE commit.

    So: reproduce both hazards rather than reason about them.
    """

    def _boom(self, *args, **kwargs):
        raise OSError(28, "No space left on device")

    def test_a_failure_inside_a_rolled_back_batch_publishes_nothing(self, tmp_path):
        from anneal_memory.store import Store

        store = Store(tmp_path / "memory.db")
        store._audit.log = self._boom
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(RuntimeError):
                    with store._batch():
                        store.record("in a batch that fails", episode_type="observation")
                        raise RuntimeError("force rollback")

            assert store.status().total_episodes == 0, (
                "the audit-health commit published DML from a batch that rolled "
                "back — a standalone commit in the post-commit handler is not "
                "safe after all"
            )
        finally:
            store.close()

        reopened = Store(tmp_path / "memory.db")
        try:
            assert reopened.status().total_episodes == 0
        finally:
            reopened.close()

    def test_a_failure_during_an_open_wrap_leaves_the_wrap_intact(self, tmp_path):
        """The wrap state machine must not notice this write at all."""
        import uuid

        from anneal_memory.store import Store

        db = tmp_path / "memory.db"
        store = Store(db)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                seed = store.record("seed", episode_type="observation")
                store.wrap_started(
                    token=uuid.uuid4().hex,
                    episode_ids=[seed["id"] if isinstance(seed, dict) else str(seed)],
                )
                # Break the sink only once the wrap is open.
                store._audit.log = self._boom
                store.record("during the wrap", episode_type="observation")
                status = store.status()

            assert status.wrap_in_progress is True
            assert status.audit_write_failures == 1
        finally:
            store.close()

        reopened = Store(db)
        try:
            after = reopened.status()
            assert after.wrap_in_progress is True, (
                "an audit-write failure during a wrap destroyed the in-progress "
                "wrap state — this is Alex's lockout class from the other side"
            )
            assert after.audit_write_failures == 1
        finally:
            reopened.close()


class TestL3ResidualsClosed:
    """Findings from the 2026-09-04 L3 mesh, each verified against disk first.

    ⚠ THE PASS ITSELF WAS NOT COVERAGE AND IS RECORDED AS SUCH: codex timed out
    and produced nothing, and glm was cut off part-way through the target. What
    is closed here is what two partial seats reached, not a clean bill.
    """

    def test_queued_audit_kwargs_may_not_shadow_the_flush_call_site(self, tmp_path):
        """complement MED — a collision would raise from the post-commit path.

        The ``_batch`` flush calls ``_audit_log_after_commit`` with
        ``method=``/``committed=``/``stacklevel=``/``batch_aware=`` explicit and
        then splats the QUEUED kwargs. A queued key with one of those names
        makes Python raise ``TypeError: got multiple values`` **at the call
        site, before the callee runs** — so it fires after ``commit_succeeded``
        and propagates a raw TypeError out of a fully committed batch.

        ⚠ THE FIRST FIX FOR THIS WAS UNREACHABLE, and the test is what proved
        it. A guard placed INSIDE ``_audit_log_after_commit`` can never fire:
        Python binds every one of those names to the PARAMETER, so they never
        arrive in that method's ``**kwargs``. The assertion's subject was the
        callee's kwargs; the claim's subject is the caller's splat. Hence a
        free function checked at ENQUEUE, which is the only path that can carry
        a bad key to the flush.
        """
        from anneal_memory.store import (
            _RESERVED_AUDIT_KWARGS,
            _reject_reserved_audit_kwargs,
        )

        # ⛔ DERIVED, NOT TYPED — and the test derives it INDEPENDENTLY from
        # the same ground truth. The hand-written version listed four names
        # and the method has six collision-capable parameters: ``event`` and
        # ``payload`` are supplied POSITIONALLY at the flush splat, which
        # collides identically. Measured 2026-09-04: {"event": "x"} passed the
        # guard and raised out of a fully committed batch. Asserting against
        # a re-typed literal is what let that sit — the old test derived its
        # cases FROM the set under test, so it could only ever confirm the
        # names already there.
        import inspect

        from anneal_memory.store import Store

        collision_capable = {
            name
            for name, param in inspect.signature(
                Store._audit_log_after_commit
            ).parameters.items()
            if name != "self" and param.kind is not inspect.Parameter.VAR_KEYWORD
        }
        assert _RESERVED_AUDIT_KWARGS == collision_capable, (
            "the guarded set has drifted from the callee's signature: "
            f"guarded={sorted(_RESERVED_AUDIT_KWARGS)} "
            f"collision-capable={sorted(collision_capable)}. A parameter added "
            "to _audit_log_after_commit widens the hole silently unless the "
            "set is derived."
        )
        assert {"event", "payload"} <= _RESERVED_AUDIT_KWARGS

        # The pass-through actually in use survives untouched.
        assert _reject_reserved_audit_kwargs({"actor": "src"}) == {"actor": "src"}

        for name in sorted(collision_capable):
            with pytest.raises(TypeError, match="may not use"):
                _reject_reserved_audit_kwargs({name: "collision"})

    def test_every_reserved_name_really_does_collide_at_the_flush_splat(
        self, tmp_path
    ):
        """The set must be justified by BEHAVIOUR, not by its own definition.

        Deriving the guarded set from the signature keeps it in sync, but on
        its own it is circular: it would happily guard a name that cannot
        actually collide, and the argument for refusing these keys is that
        Python raises at the call site before the callee's try/except can see
        it. So reproduce the splat and watch it raise, once per name.

        Nothing executes inside the method — argument binding fails first,
        which is the entire point of the finding.
        """
        from anneal_memory.store import _RESERVED_AUDIT_KWARGS, Store

        store = Store(tmp_path / "memory.db")
        try:
            # The flush passes event/payload positionally and the rest by
            # keyword; mirror that shape, then splat one candidate over it.
            by_keyword = {
                name: None
                for name in _RESERVED_AUDIT_KWARGS - {"event", "payload"}
            }
            for name in sorted(_RESERVED_AUDIT_KWARGS):
                with pytest.raises(TypeError, match="multiple values"):
                    store._audit_log_after_commit(
                        "evt", None, **by_keyword, **{name: "collision"}
                    )
        finally:
            store.close()

    def test_both_enqueue_paths_run_the_reserved_kwarg_check(self):
        """The guard must sit on EVERY path that can feed the flush splat.

        Two methods append to ``_deferred_audits``. A check on one of them is
        the same one-of-N shape this whole change set is about.
        """
        import ast
        from pathlib import Path

        import anneal_memory.store as store_mod

        tree = ast.parse(Path(store_mod.__file__).read_text(encoding="utf-8"))
        unguarded = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr == "append"):
                continue
            if not (
                isinstance(fn.value, ast.Attribute)
                and fn.value.attr == "_deferred_audits"
            ):
                continue
            src = ast.unparse(node)
            if "_reject_reserved_audit_kwargs" not in src:
                unguarded.append(node.lineno)
        assert not unguarded, (
            "append(s) to _deferred_audits that do not sanitise their kwargs, "
            f"at line(s) {unguarded}. Every enqueue path feeds the flush splat."
        )

    def test_a_stray_frozen_schema_is_named_in_the_cancel_audit(self, tmp_path):
        """glm LOW — confirmed on disk before believing it.

        ``had_any`` counts FOUR lifecycle keys; the audit payload named three.
        A store carrying only a stray ``wrap_section_schema`` — a crash between
        wrap_started's schema INSERT and its token INSERT — fired the event with
        an EMPTY payload plus ``partial_state``: a forensic record saying
        something was cleared without saying what, on the very recovery case the
        marker exists to flag.
        """
        import json

        from anneal_memory.store import Store

        store = Store(tmp_path / "memory.db")
        # Exactly the partial state: schema set, nothing else.
        store._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("wrap_section_schema", json.dumps([{"heading": "State",
                                                 "role": "live-state"}])),
        )
        store._conn.commit()

        receipt = store.wrap_cancelled()
        assert receipt.partial_state is True

        events = [
            json.loads(line)
            for line in store._audit._active_path.read_text().splitlines()
            if line.strip()
        ]
        cancelled = [e for e in events if e["event"] == "wrap_cancelled"]
        assert cancelled, "no wrap_cancelled audit event was written at all"
        data = cancelled[-1].get("data", {})
        assert data.get("wrap_section_schema_cleared") is True, (
            "the cancel audit event does not name the one key that was "
            f"actually cleared. payload={data!r}"
        )

    def test_the_drop_count_survives_a_rotation_and_is_carried_once(self, tmp_path):
        """Accounting across the two events that could corrupt it.

        ``log()`` calls ``_rotate_if_needed()`` BEFORE building the entry, and
        the chain is write-first, so a drop pending across a rotation boundary
        could plausibly be lost with the old file or double-counted into both.
        Measured 2026-09-04: it rides into the first entry of the NEW file,
        exactly once, and the chain still verifies across both files.
        """
        import json

        from anneal_memory.audit import AuditTrail
        from anneal_memory.store import Store

        store = Store(tmp_path / "memory.db")
        store.record("first", episode_type="observation")
        real = store._audit.log

        def boom(*args, **kwargs):
            raise OSError(28, "No space left on device")

        store._audit.log = boom
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(3):
                store.record(f"drop{i}", episode_type="observation")
        store._audit.log = real
        assert store._audit._dropped_since_last == 3

        store._audit._last_week = "1970-W01"  # force the weekly rotation
        store.record("after rotation", episode_type="observation")

        assert store._audit._dropped_since_last == 0
        last = json.loads(store._audit._active_path.read_text().splitlines()[-1])
        assert last["dropped_before"] == 3
        result = AuditTrail.verify(store._path)
        assert result.valid and result.files_verified == 2, result

    def test_repeated_failures_accumulate_and_are_not_reset_early(self, tmp_path):
        """The count clears only after fsync, never on a failed attempt.

        Clearing on the attempt would lose the very fact the mechanism exists
        to preserve — the failure that carried it.
        """
        import json

        from anneal_memory.store import Store

        store = Store(tmp_path / "memory.db")
        store.record("first", episode_type="observation")
        real = store._audit.log

        def boom(*args, **kwargs):
            raise OSError(28, "No space left on device")

        store._audit.log = boom
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(3):
                store.record(f"d{i}", episode_type="observation")
        assert store._audit._dropped_since_last == 3

        store._audit.log = real
        store.record("lands", episode_type="observation")
        last = json.loads(store._audit._active_path.read_text().splitlines()[-1])
        assert last["dropped_before"] == 3, "carried the wrong count"

        # And exactly once: the entry after it must be clean.
        store.record("next", episode_type="observation")
        tail = json.loads(store._audit._active_path.read_text().splitlines()[-1])
        assert "dropped_before" not in tail


class TestFailureLandsAtDifferentPointsInTheWrite:
    """⛔ THE DIMENSION EVERY OTHER TEST IN THIS FILE HOLDS CONSTANT.

    codex named it at L3 on 2026-09-04, and it was aimed at these tests:
    *"The current regression tests replace ``AuditTrail.log`` wholesale with a
    function that raises before writing, so they cannot detect the post-write
    ambiguity or incomplete-rotation paths."* Correct. Every fixture above
    swaps in a ``boom`` that raises BEFORE any bytes reach the file, so they
    vary the sink outcome and the method while holding constant WHERE IN THE
    WRITE the failure happens — and that is exactly where the defect lived.

    Same discriminator that killed two mutants earlier in this file, one layer
    deeper. Ask what the fixture varies, and whether the defect lives in the
    dimension it holds fixed.
    """

    def _store_with_pending_drops(self, tmp_path, n=2):
        from anneal_memory.store import Store

        store = Store(tmp_path / "memory.db")
        store.record("first", episode_type="observation")
        real = store._audit.log

        def boom(*args, **kwargs):
            raise OSError(28, "No space left on device")

        store._audit.log = boom
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(n):
                store.record(f"dropped{i}", episode_type="observation")
        store._audit.log = real
        assert store._audit._dropped_since_last == n
        return store

    def test_a_post_write_fsync_failure_does_not_corrupt_the_chain(self, tmp_path):
        """codex HIGH — reproduced before the fix, pinned after it.

        write + flush succeed, fsync raises EIO. The complete line is already
        on disk while ``_seq``/``_prev_hash``/the drop counter are unchanged, so
        the retry re-emits the SAME seq and prev_hash.

        MEASURED BEFORE THE FIX: seqs ``[0, 1, 1]``, dropped_before
        ``[None, 2, 3]`` — the pending drops counted twice AND the entry that
        actually landed counted as dropped — with ``verify()`` returning
        ``valid=False``. A durability hiccup reported as TAMPERING, by the
        record whose whole job is telling those apart.
        """
        import json

        from anneal_memory.audit import AuditTrail

        store = self._store_with_pending_drops(tmp_path, n=2)

        real_fsync = os.fsync

        def fsync_eio(fd):
            raise OSError(5, "EIO")

        os.fsync = fsync_eio
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                store.record("the ambiguous one", episode_type="observation")
        finally:
            os.fsync = real_fsync

        store.record("the retry", episode_type="observation")

        entries = [
            json.loads(line)
            for line in store._audit._active_path.read_text().splitlines()
            if line.strip()
        ]
        seqs = [e["seq"] for e in entries]
        assert len(seqs) == len(set(seqs)), f"duplicate seq on disk: {seqs}"
        assert AuditTrail.verify(store._path).valid, (
            "a post-write fsync failure broke the hash chain — a durability "
            "problem is now indistinguishable from tampering"
        )
        # The rolled-back entry counts as dropped exactly once: 2 + itself.
        assert entries[-1]["dropped_before"] == 3, entries[-1]

    def test_a_partial_write_is_rolled_back_not_left_to_concatenate(self, tmp_path):
        """The same seam, entered mid-line instead of at fsync.

        An ENOSPC part-way through the line leaves a truncated fragment that
        the next append concatenates with, producing one unparseable line and
        a permanent chain break. The pre-append size is restored instead.
        """
        import json

        from anneal_memory.audit import AuditTrail

        store = self._store_with_pending_drops(tmp_path, n=1)
        active = store._audit._active_path
        size_before = active.stat().st_size

        real_open = __builtins__["open"] if isinstance(__builtins__, dict) else open

        class HalfWriter:
            def __init__(self, fh):
                self._fh = fh

            def write(self, data):
                self._fh.write(data[: max(1, len(data) // 2)])
                raise OSError(28, "No space left on device")

            def __getattr__(self, name):
                return getattr(self._fh, name)

        import anneal_memory.audit as audit_mod

        def patched_open(path, mode="r", *args, **kwargs):
            fh = real_open(path, mode, *args, **kwargs)
            if "a" in mode and str(path) == str(active):
                return _Ctx(HalfWriter(fh), fh)
            return fh

        class _Ctx:
            def __init__(self, wrapper, fh):
                self._w, self._fh = wrapper, fh

            def __enter__(self):
                return self._w

            def __exit__(self, *exc):
                self._fh.close()
                return False

        audit_mod.open = patched_open  # type: ignore[assignment]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                store.record("the half-written one", episode_type="observation")
        finally:
            del audit_mod.open

        assert active.stat().st_size == size_before, (
            "a partial write was left on disk; the next append will "
            "concatenate with it into unparseable JSON"
        )
        store.record("the next one", episode_type="observation")
        for line in active.read_text().splitlines():
            if line.strip():
                json.loads(line)  # must all parse
        assert AuditTrail.verify(store._path).valid
