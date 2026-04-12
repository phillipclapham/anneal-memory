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
        with pytest.warns(DeprecationWarning, match="legacy call form"):
            store.wrap_started()
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
        with pytest.warns(DeprecationWarning, match="legacy call form"):
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
        with pytest.warns(DeprecationWarning, match="legacy call form"):
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
        with pytest.warns(DeprecationWarning, match="legacy call form"):
            store.wrap_started()
        store.save_continuity("## State\nTest\n## Patterns\n\n## Decisions\n\n## Context\n")
        store.wrap_completed(episodes_compressed=1, continuity_chars=50)

        audit_path = tmp_path / "test.audit.jsonl"
        assert not audit_path.exists()
        store.close()
