"""Hash-chained JSONL audit trail for anneal-memory.

Tamper-evident audit infrastructure for the episodic store. Each entry
includes the SHA-256 hash of the previous entry's JSON, creating an
unbroken chain. Any modification breaks the chain at that point.

The audit trail is a VIEW of the episodic store — it mirrors mutations
(record, delete, prune, wrap, continuity save) to an append-only JSONL
sidecar alongside the SQLite database. Content is referenced by hash,
not duplicated — the SQLite store is the source of truth.

The local hash chain provides integrity verification against accidental
corruption and unauthorized modification by parties without filesystem
access. For external compliance attestation (regulatory audits, third-party
verification), use the ``on_event`` callback to stream entries to an
external witness service. The local chain is defense-in-depth, not the
sole compliance control.

Weekly rotation with gzip compression keeps the active file small.
A manifest index enables cross-file chain verification and efficient
time-range queries without decompressing sealed files.

Zero dependencies beyond Python stdlib.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("anneal-memory")

# Chain anchors
GENESIS_HASH = "sha256:GENESIS"

# Schema version for JSONL entries
_ENTRY_VERSION = 1


@dataclass
class AuditVerifyResult:
    """Result of verifying a hash chain."""

    valid: bool
    total_entries: int
    files_verified: int
    chain_break_at: int | None = None  # seq number where chain broke
    chain_break_file: str | None = None  # file where break occurred
    skipped_lines: int = 0  # malformed JSON lines skipped during verification
    error: str | None = None


class AuditTrail:
    """Hash-chained JSONL audit trail.

    Appends tamper-evident entries to a JSONL sidecar file alongside
    the SQLite episodic store. Each entry includes the SHA-256 hash
    of the previous entry, creating a cryptographic chain.

    **Single-writer requirement:** Only one AuditTrail instance should
    write to a given db_path at a time. Concurrent writers will corrupt
    the hash chain (interleaved entries with incompatible prev_hash values).
    The MCP server's single-threaded model enforces this naturally.

    **Timestamp note:** Timestamps are self-reported via the local system
    clock (``datetime.now(timezone.utc)``). This provides audit logging but
    not externally attested time. True timestamp attestation (RFC 3161 TSA
    or similar) is out of scope for the local audit trail but planned for
    the cloud witness tier.

    Args:
        db_path: Path to the SQLite database (audit files derive from this).
        retention_days: Auto-cleanup threshold for rotated files. None = keep forever.
        on_event: Optional callback receiving each entry dict after write.
    """

    def __init__(
        self,
        db_path: str | Path,
        retention_days: int | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._retention_days = retention_days
        self._on_event = on_event

        # State — initialized lazily on first log()
        self._initialized = False
        self._seq: int = 0
        self._prev_hash: str = GENESIS_HASH
        self._last_week: str = ""

    # -- Public API --

    def log(
        self,
        event: str,
        data: dict[str, Any] | None = None,
        actor: str = "agent",
    ) -> dict[str, Any]:
        """Append a hash-chained entry to the audit trail.

        Args:
            event: Event type (record, delete, prune, wrap_started,
                   wrap_completed, wrap_cancelled, continuity_saved).
            data: Event-specific payload.
            actor: Identity of the actor triggering this event.
                   EU AI Act Article 12(2) requires actor identity on
                   all audit entries. Default "agent" for single-agent;
                   multi-agent passes agent name/ID.

        Returns:
            The complete entry dict that was written.
        """
        if not self._initialized:
            self._initialize()

        # Check for weekly rotation before writing
        self._rotate_if_needed()

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        entry = {
            "v": _ENTRY_VERSION,
            "seq": self._seq,
            "ts": ts,
            "event": event,
            "actor": actor,
            "prev_hash": self._prev_hash,
        }
        if data is not None:
            entry["data"] = data

        # Deterministic serialization — sorted keys, compact separators
        json_line = json.dumps(entry, sort_keys=True, separators=(",", ":"))

        # Write-first: fsync BEFORE updating internal state
        active = self._active_path
        active.parent.mkdir(parents=True, exist_ok=True)
        with open(active, "a", encoding="utf-8") as f:
            f.write(json_line + "\n")
            f.flush()
            os.fsync(f.fileno())

        # Update chain state
        self._prev_hash = self._compute_hash(json_line)
        self._seq += 1

        # Fire callback after successful write
        if self._on_event is not None:
            try:
                self._on_event(entry)
            except Exception:
                logger.warning("on_event callback failed for seq %d", entry["seq"], exc_info=True)

        return entry

    @classmethod
    def verify(cls, db_path: str | Path) -> AuditVerifyResult:
        """Verify hash chain integrity across all audit files.

        Walks sealed files (from manifest) then the active file,
        checking that each entry's prev_hash matches the computed
        hash of the previous entry.

        Args:
            db_path: Path to the SQLite database (audit files derive from this).

        Returns:
            AuditVerifyResult with chain validity and diagnostics.
        """
        db_path = Path(db_path)
        stem = db_path.stem
        audit_dir = db_path.parent

        manifest_path = audit_dir / f"{stem}.audit.manifest.json"
        active_path = audit_dir / f"{stem}.audit.jsonl"

        # Load manifest (once) for file list + chain anchor
        files_to_verify: list[Path] = []
        chain_anchor = GENESIS_HASH
        missing_files: list[str] = []

        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                # Chain anchor from retention cleanup — trust point for
                # chains that no longer start from GENESIS
                anchor = manifest.get("chain_anchor", "")
                if anchor:
                    chain_anchor = anchor
                for f in manifest.get("files", []):
                    fpath = audit_dir / f["filename"]
                    if fpath.exists():
                        files_to_verify.append(fpath)
                    else:
                        missing_files.append(f["filename"])
            except (json.JSONDecodeError, KeyError) as e:
                return AuditVerifyResult(
                    valid=False, total_entries=0, files_verified=0,
                    error=f"Corrupt manifest: {e}",
                )

        if active_path.exists():
            files_to_verify.append(active_path)

        if not files_to_verify:
            return AuditVerifyResult(valid=True, total_entries=0, files_verified=0)

        if missing_files:
            return AuditVerifyResult(
                valid=False, total_entries=0, files_verified=0,
                error=f"Missing sealed files referenced in manifest: {missing_files}",
            )

        # Walk all files, verify chain
        total_entries = 0
        skipped = 0
        expected_hash = chain_anchor
        files_verified = 0

        for fpath in files_to_verify:
            for line in _iter_lines(fpath):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                actual_prev = entry.get("prev_hash", "")
                if actual_prev != expected_hash:
                    return AuditVerifyResult(
                        valid=False,
                        total_entries=total_entries,
                        files_verified=files_verified,
                        chain_break_at=entry.get("seq", total_entries),
                        chain_break_file=fpath.name,
                        error=f"Hash mismatch at seq {entry.get('seq')}: "
                              f"expected {expected_hash[:20]}..., "
                              f"got {actual_prev[:20]}...",
                    )

                # Compute hash of this entry for next check
                # Re-serialize deterministically to match what was written
                canonical = json.dumps(entry, sort_keys=True, separators=(",", ":"))
                expected_hash = cls._compute_hash(canonical)
                total_entries += 1

            files_verified += 1

        return AuditVerifyResult(
            valid=True,
            total_entries=total_entries,
            files_verified=files_verified,
            skipped_lines=skipped,
        )

    # -- Internal --

    @property
    def _active_path(self) -> Path:
        """Path to the active (current) JSONL file."""
        return self._db_path.parent / f"{self._db_path.stem}.audit.jsonl"

    @property
    def _manifest_path(self) -> Path:
        """Path to the manifest index."""
        return self._db_path.parent / f"{self._db_path.stem}.audit.manifest.json"

    def _initialize(self) -> None:
        """Lazy init: recover seq and prev_hash from existing audit file."""
        self._initialized = True

        # Adopt orphaned sealed files — crash between rename and manifest
        # update during rotation leaves .gz files the manifest doesn't know about.
        self._adopt_orphaned_files()

        active = self._active_path
        if not active.exists():
            # Fresh start — check manifest for chain continuity from sealed files
            if self._manifest_path.exists():
                try:
                    manifest = json.loads(
                        self._manifest_path.read_text(encoding="utf-8")
                    )
                    self._prev_hash = manifest.get(
                        "active_last_hash", GENESIS_HASH
                    )
                    self._seq = manifest.get("active_last_seq", 0)
                except (json.JSONDecodeError, KeyError):
                    pass
            self._last_week = _iso_week_now()
            return

        # Recover from existing active file — find last valid JSON entry
        last_line = _read_last_valid_entry(active)
        if last_line:
            last_entry = json.loads(last_line)  # Guaranteed valid by helper
            self._seq = last_entry.get("seq", 0) + 1
            # Recompute hash from exact bytes on disk
            canonical = json.dumps(
                last_entry, sort_keys=True, separators=(",", ":")
            )
            self._prev_hash = self._compute_hash(canonical)
            # Recover week from last entry timestamp
            ts = last_entry.get("ts", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    self._last_week = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
                except ValueError:
                    self._last_week = _iso_week_now()
            else:
                self._last_week = _iso_week_now()
        else:
            self._last_week = _iso_week_now()

    def _adopt_orphaned_files(self) -> None:
        """Adopt sealed files that the manifest doesn't know about.

        This handles crash recovery: if the process dies between
        active.rename() and _save_manifest() during rotation, the sealed
        file exists on disk but the manifest has no record of it.
        Scans for both compressed (.gz) and uncompressed (.jsonl) orphans
        — crash can happen before or after gzip compression.
        """
        stem = self._db_path.stem
        audit_dir = self._db_path.parent
        active_name = f"{stem}.audit.jsonl"

        manifest = self._load_manifest()
        known_files = {f["filename"] for f in manifest.get("files", [])}

        orphans = []
        # Scan for both .gz and uncompressed .jsonl orphans
        for pattern in [f"{stem}.audit.*.jsonl.gz", f"{stem}.audit.*.jsonl"]:
            for path in sorted(audit_dir.glob(pattern)):
                if path.name == active_name:
                    continue  # Skip the active file
                if path.name not in known_files:
                    orphans.append(path)

        if not orphans:
            return

        for gz_path in orphans:
            # Read the orphaned file to get metadata
            entry_count = 0
            first_ts = ""
            last_ts = ""
            last_hash = ""

            for line in _iter_lines(gz_path):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    e = json.loads(stripped)
                    ts = e.get("ts", "")
                    if not first_ts:
                        first_ts = ts
                    last_ts = ts
                    entry_count += 1
                    # Compute hash of this entry for chain continuity
                    canonical = json.dumps(e, sort_keys=True, separators=(",", ":"))
                    last_hash = self._compute_hash(canonical)
                except json.JSONDecodeError:
                    pass

            # Extract period from filename (e.g., "memory.audit.2026-W14.jsonl.gz")
            name_parts = gz_path.name.replace(f"{stem}.audit.", "").replace(".jsonl.gz", "")

            manifest["files"].append({
                "filename": gz_path.name,
                "period": name_parts,
                "entries": entry_count,
                "first_ts": first_ts,
                "last_ts": last_ts,
                "last_hash": last_hash,
                "sha256_file": "",  # Not computed during adoption
            })

            if last_hash:
                manifest["active_last_hash"] = last_hash

            logger.info("Adopted orphaned audit file: %s (%d entries)", gz_path.name, entry_count)

        self._save_manifest(manifest)

    def _rotate_if_needed(self) -> None:
        """Rotate the active file if the ISO week has changed."""
        current_week = _iso_week_now()
        if not self._last_week:
            self._last_week = current_week
            return

        if current_week == self._last_week:
            return

        active = self._active_path
        if not active.exists() or active.stat().st_size == 0:
            self._last_week = current_week
            return

        # Seal the active file with the old week label
        sealed_name = f"{self._db_path.stem}.audit.{self._last_week}.jsonl"
        sealed_path = active.parent / sealed_name
        sealed_gz_path = sealed_path.with_suffix(".jsonl.gz")

        # Rename → compress → update manifest
        active.rename(sealed_path)

        # Gzip compress
        file_hash = hashlib.sha256()
        entry_count = 0
        first_ts = ""
        last_ts = ""

        with open(sealed_path, "rb") as f_in, gzip.open(sealed_gz_path, "wb") as f_out:
            for line in f_in:
                file_hash.update(line)
                f_out.write(line)
                stripped = line.decode("utf-8").strip()
                if stripped:
                    entry_count += 1
                    try:
                        e = json.loads(stripped)
                        ts = e.get("ts", "")
                        if not first_ts:
                            first_ts = ts
                        last_ts = ts
                    except json.JSONDecodeError:
                        pass

        # Remove uncompressed sealed file
        sealed_path.unlink()

        # Update manifest
        manifest = self._load_manifest()
        manifest["files"].append({
            "filename": sealed_gz_path.name,
            "period": self._last_week,
            "entries": entry_count,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "last_hash": self._prev_hash,
            "sha256_file": f"sha256:{file_hash.hexdigest()}",
        })
        manifest["active_last_hash"] = self._prev_hash
        manifest["active_last_seq"] = self._seq
        self._save_manifest(manifest)

        # Reset seq for new file, chain continues via prev_hash
        self._seq = 0
        self._last_week = current_week

        # Auto-cleanup old files
        if self._retention_days is not None:
            self._cleanup(manifest)

    def _cleanup(self, manifest: dict[str, Any] | None = None) -> int:
        """Remove rotated files older than retention_days."""
        if self._retention_days is None:
            return 0

        if manifest is None:
            manifest = self._load_manifest()

        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)
        cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        removed = 0
        removed_last_hash = ""
        remaining_files = []
        for f in manifest.get("files", []):
            if f.get("last_ts", "") < cutoff_str:
                fpath = self._db_path.parent / f["filename"]
                try:
                    fpath.unlink(missing_ok=True)
                    removed_last_hash = f.get("last_hash", "")
                    removed += 1
                except OSError:
                    remaining_files.append(f)
            else:
                remaining_files.append(f)

        if removed > 0:
            # Record chain anchor — the last_hash of the most recently
            # removed file becomes the trust anchor for verification.
            # Without this, verify() can't validate chains that start
            # after cleanup has removed earlier files.
            if removed_last_hash:
                manifest["chain_anchor"] = removed_last_hash
            manifest["files"] = remaining_files
            self._save_manifest(manifest)

        return removed

    def _load_manifest(self) -> dict[str, Any]:
        """Load or create the manifest index."""
        if self._manifest_path.exists():
            try:
                return json.loads(
                    self._manifest_path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                pass

        return {
            "version": 1,
            "db_path": self._db_path.name,
            "active_file": self._active_path.name,
            "active_last_hash": GENESIS_HASH,
            "active_last_seq": 0,
            "files": [],
        }

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        """Save manifest with atomic write."""
        path = self._manifest_path
        tmp_path = path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(path)
        except Exception:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise

    @staticmethod
    def _compute_hash(json_line: str) -> str:
        """Compute SHA-256 hash of a JSON line (the exact bytes written)."""
        return "sha256:" + hashlib.sha256(
            json_line.strip().encode("utf-8")
        ).hexdigest()


# -- Module-level helpers --


def _iso_week_now() -> str:
    """Current ISO week as YYYY-WNN string."""
    now = datetime.now(timezone.utc)
    iso = now.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _read_last_valid_entry(path: Path) -> str:
    """Read the last valid JSON line from an audit file.

    Reads line-by-line (not chunk-based) so entries of any size are
    handled correctly. Walks backward from the end to find the last
    line that parses as valid JSON — skips partial writes from crashes.

    Memory-safe: only keeps the last valid line in memory at a time.
    """
    last_valid = ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    json.loads(stripped)
                    last_valid = stripped
                except json.JSONDecodeError:
                    pass  # Partial write — skip
    except (OSError, UnicodeDecodeError):
        pass
    return last_valid


def _iter_lines(path: Path):
    """Iterate lines from an audit file (handles .gz transparently)."""
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            yield from f
    else:
        with open(path, "r", encoding="utf-8") as f:
            yield from f
