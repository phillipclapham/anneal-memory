"""SQLite episodic store for anneal-memory.

The episodic store is the substrate layer — fast, indexed, append-heavy.
It stores typed episodes and serves as citation evidence for graduation.
The continuity file (markdown) is the human-readable layer.

Zero dependencies beyond Python stdlib.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .associations import (
    ASSOCIATIONS_SCHEMA,
    association_stats as _association_stats,
    canonical_pair,
    decay_associations as _decay_associations,
    get_association_context as _get_association_context,
    get_associations as _get_associations,
    migrate_add_affective_columns as _migrate_affective,
    record_associations as _record_associations,
)
from .audit import AuditTrail
from .types import (
    AffectiveState,
    AssociationPair,
    AssociationStats,
    Episode,
    EpisodeType,
    RecallResult,
    StoreStatus,
    Tombstone,
    WrapResult,
)

# Schema version — increment on breaking changes
_SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'agent',
    session_id TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_type ON episodes(type);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_source ON episodes(source);

CREATE TABLE IF NOT EXISTS tombstones (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    type TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    pruned_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS wraps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wrapped_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    episodes_compressed INTEGER,
    graduations_validated INTEGER DEFAULT 0,
    graduations_demoted INTEGER DEFAULT 0,
    citation_reuse_max INTEGER DEFAULT 0,
    continuity_chars INTEGER,
    patterns_extracted INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Appended separately so existing DBs get the new table via CREATE IF NOT EXISTS
_ASSOCIATIONS_SCHEMA_SQL = ASSOCIATIONS_SCHEMA

_DEFAULT_METADATA = {
    "format_version": str(_SCHEMA_VERSION),
    "project_name": "Agent",
    "wrap_started_at": "",
}


def _now_utc() -> str:
    """Current time as ISO 8601 UTC string with microsecond precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _episode_id(content: str, timestamp: str, nonce: int = 0) -> str:
    """Generate 8-char hex ID from content + timestamp + optional nonce.

    Uses 8 hex chars (32 bits). Birthday collision expected around ~65k episodes.
    The nonce parameter handles same-content-same-timestamp edge cases.
    Microsecond-precision timestamps make natural collisions extremely unlikely.
    """
    raw = f"{content}{timestamp}{nonce}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:8]


def _content_hash(content: str) -> str:
    """SHA256 hash of content for tombstones."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class Store:
    """SQLite-backed episodic store.

    Thread safety: The underlying SQLite connection uses the default
    ``check_same_thread=True``. All access must happen from the thread
    that created the Store instance. For async or multi-threaded MCP
    servers, create a Store per thread or add external synchronization.

    Use as a context manager (``with Store(...) as s:``) to ensure the
    connection is closed properly.

    Args:
        path: Path to the SQLite database file. Created if it doesn't exist.
        retention_days: Optional. Prune episodes older than N days. None = keep all.
        keep_tombstones: When pruning, preserve audit trail (ID, timestamp, hash). Default True.
        project_name: Name for the continuity file header. Default "Agent".
        audit: Enable hash-chained JSONL audit trail. Default True.
        audit_retention_days: Auto-cleanup for rotated audit files. None = keep forever.
        on_audit_event: Optional callback receiving each audit entry dict after write.
    """

    def __init__(
        self,
        path: str | Path,
        retention_days: int | None = None,
        keep_tombstones: bool = True,
        project_name: str | None = None,
        audit: bool = True,
        audit_retention_days: int | None = None,
        on_audit_event: Callable | None = None,
    ) -> None:
        self._path = Path(path)
        self._retention_days = retention_days
        self._keep_tombstones = keep_tombstones
        self._project_name = project_name or "Agent"

        # Audit trail — hash-chained JSONL sidecar
        self._audit: AuditTrail | None = None
        if audit:
            self._audit = AuditTrail(
                self._path,
                retention_days=audit_retention_days,
                on_event=on_audit_event,
            )

        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._path))
        try:
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._init_schema()
        except Exception:
            self._conn.close()
            raise

    def _init_schema(self) -> None:
        """Initialize database schema and default metadata."""
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.executescript(_ASSOCIATIONS_SCHEMA_SQL)
        # Migrate existing associations tables to include affective columns
        # (safe no-op if columns already exist or table was just created)
        _migrate_affective(self._conn)
        # Migrate wraps table to include association metric columns
        self._migrate_wraps_association_columns()

        # Insert default metadata (ignore if already exists)
        defaults = {**_DEFAULT_METADATA, "project_name": self._project_name}
        for key, value in defaults.items():
            self._conn.execute(
                "INSERT OR IGNORE INTO metadata (key, value) VALUES (?, ?)",
                (key, value),
            )
        self._conn.commit()

    def _migrate_wraps_association_columns(self) -> None:
        """Add association metric columns to existing wraps tables.

        Safe to call on tables that already have the columns (checks first).
        Added post-Session 9.5 to track per-wrap association activity.
        """
        cursor = self._conn.execute("PRAGMA table_info(wraps)")
        existing_cols = {row[1] for row in cursor.fetchall()}

        if "associations_formed" not in existing_cols:
            self._conn.execute(
                "ALTER TABLE wraps ADD COLUMN associations_formed INTEGER NOT NULL DEFAULT 0"
            )
        if "associations_strengthened" not in existing_cols:
            self._conn.execute(
                "ALTER TABLE wraps ADD COLUMN associations_strengthened INTEGER NOT NULL DEFAULT 0"
            )
        if "associations_decayed" not in existing_cols:
            self._conn.execute(
                "ALTER TABLE wraps ADD COLUMN associations_decayed INTEGER NOT NULL DEFAULT 0"
            )
        self._conn.commit()

    # -- Core API --

    def record(
        self,
        content: str,
        episode_type: EpisodeType | str,
        source: str = "agent",
        metadata: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> Episode:
        """Record a new episode.

        Args:
            content: The episode content.
            episode_type: Episode type (EpisodeType enum or string).
            source: Agent/source attribution.
            metadata: Optional JSON-serializable metadata.
            timestamp: Optional ISO 8601 UTC timestamp. Defaults to now.

        Returns:
            The recorded Episode.
        """
        if not content or not content.strip():
            raise ValueError("Episode content cannot be empty")

        if isinstance(episode_type, str):
            episode_type = EpisodeType(episode_type)

        ts = timestamp or _now_utc()
        session_id = self._current_session_id()
        meta_json = json.dumps(metadata) if metadata is not None else None

        # Retry with incrementing nonce on ID collision (birthday or duplicate content)
        max_retries = 3
        for nonce in range(max_retries):
            ep_id = _episode_id(content, ts, nonce)
            try:
                self._conn.execute(
                    """INSERT INTO episodes (id, timestamp, type, content, source, session_id, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (ep_id, ts, episode_type.value, content, source, session_id, meta_json),
                )
                self._conn.commit()
                break
            except sqlite3.IntegrityError:
                if nonce == max_retries - 1:
                    raise
                continue

        episode = Episode(
            id=ep_id,
            timestamp=ts,
            type=episode_type,
            content=content,
            source=source,
            session_id=session_id,
            metadata=metadata,
        )

        if self._audit is not None:
            self._audit.log("record", {
                "episode_id": ep_id,
                "type": episode_type.value,
                "content_hash": _content_hash(content),
                "source": source,
            }, actor=source)

        return episode

    def get(self, episode_id: str) -> Episode | None:
        """Get a single episode by ID.

        Args:
            episode_id: The 8-char hex episode ID.

        Returns:
            The Episode, or None if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_episode(row)

    def delete(self, episode_id: str) -> bool:
        """Delete a single episode by ID.

        For corrections, mistakes, or privacy (right to erasure).
        Optionally creates a tombstone if keep_tombstones is enabled.

        Args:
            episode_id: The 8-char hex episode ID.

        Returns:
            True if the episode was found and deleted, False if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        if not row:
            return False

        if self._keep_tombstones:
            self._conn.execute(
                """INSERT OR IGNORE INTO tombstones
                   (id, timestamp, type, content_hash)
                   VALUES (?, ?, ?, ?)""",
                (row["id"], row["timestamp"], row["type"], _content_hash(row["content"])),
            )

        self._conn.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
        self._conn.commit()

        if self._audit is not None:
            self._audit.log("delete", {
                "episode_id": row["id"],
                "type": row["type"],
                "content_hash": _content_hash(row["content"]),
            })

        return True

    def recall(
        self,
        since: str | None = None,
        until: str | None = None,
        episode_type: EpisodeType | str | None = None,
        source: str | None = None,
        keyword: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> RecallResult:
        """Query episodes with filters.

        Args:
            since: ISO 8601 timestamp — episodes after this time.
            until: ISO 8601 timestamp — episodes before this time.
            episode_type: Filter by episode type.
            source: Filter by source/agent.
            keyword: Search content (LIKE %keyword%). Wildcards % and _ are escaped.
            limit: Max episodes to return.
            offset: Skip first N matching episodes.

        Returns:
            RecallResult with matching episodes and total count.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until:
            conditions.append("timestamp <= ?")
            params.append(until)
        if episode_type is not None:
            if isinstance(episode_type, str):
                episode_type = EpisodeType(episode_type)
            conditions.append("type = ?")
            params.append(episode_type.value)
        if source:
            conditions.append("source = ?")
            params.append(source)
        if keyword:
            # Escape LIKE wildcards so % and _ are treated as literals
            # Case-insensitive via LOWER() — agents need to find episodes
            # regardless of casing for citation during graduation
            escaped = keyword.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            conditions.append("LOWER(content) LIKE ? ESCAPE '\\'")
            params.append(f"%{escaped}%")

        where = " AND ".join(conditions) if conditions else "1=1"

        # Get total count
        count_row = self._conn.execute(
            f"SELECT COUNT(*) FROM episodes WHERE {where}", params
        ).fetchone()
        total = count_row[0]

        # Get episodes
        rows = self._conn.execute(
            f"""SELECT * FROM episodes WHERE {where}
                ORDER BY timestamp DESC LIMIT ? OFFSET ?""",
            [*params, limit, offset],
        ).fetchall()

        episodes = [self._row_to_episode(row) for row in rows]

        return RecallResult(
            episodes=episodes,
            total_matching=total,
            query_params={
                k: v
                for k, v in {
                    "since": since,
                    "until": until,
                    "type": episode_type.value if isinstance(episode_type, EpisodeType) else episode_type,
                    "source": source,
                    "keyword": keyword,
                    "limit": limit,
                    "offset": offset,
                }.items()
                if v is not None
            },
        )

    def episodes_since_wrap(self) -> list[Episode]:
        """Get all episodes since the last completed wrap.

        This is the compression window — what the LLM needs to see
        when producing the next continuity file.

        Uses session IDs (not timestamps) for precision — avoids
        same-second boundary issues.
        """
        last_wrap = self._conn.execute(
            "SELECT id FROM wraps ORDER BY id DESC LIMIT 1"
        ).fetchone()

        if last_wrap:
            # Episodes with session_id > last wrap's ID are post-wrap.
            # Also include any with NULL session_id (shouldn't happen, but safe).
            rows = self._conn.execute(
                """SELECT * FROM episodes
                   WHERE CAST(session_id AS INTEGER) > ? OR session_id IS NULL
                   ORDER BY timestamp ASC""",
                (last_wrap["id"],),
            ).fetchall()
        else:
            # No wraps yet — all episodes are in the compression window
            rows = self._conn.execute(
                "SELECT * FROM episodes ORDER BY timestamp ASC"
            ).fetchall()

        return [self._row_to_episode(row) for row in rows]

    def status(self) -> StoreStatus:
        """Get store status snapshot."""
        total = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        since_wrap = self._count_episodes_since_wrap()
        total_wraps = self._conn.execute("SELECT COUNT(*) FROM wraps").fetchone()[0]
        tombstone_count = self._conn.execute(
            "SELECT COUNT(*) FROM tombstones"
        ).fetchone()[0]

        last_wrap_row = self._conn.execute(
            "SELECT wrapped_at FROM wraps ORDER BY id DESC LIMIT 1"
        ).fetchone()
        last_wrap_at = last_wrap_row["wrapped_at"] if last_wrap_row else None

        wrap_started = self._get_metadata("wrap_started_at")
        wrap_in_progress = bool(wrap_started)

        # Episodes by type
        type_rows = self._conn.execute(
            "SELECT type, COUNT(*) as count FROM episodes GROUP BY type"
        ).fetchall()
        episodes_by_type = {row["type"]: row["count"] for row in type_rows}

        # Continuity file size
        continuity_path = self.continuity_path
        continuity_chars = None
        if continuity_path.exists():
            continuity_chars = len(continuity_path.read_text(encoding="utf-8"))

        # Association network metrics
        assoc_stats = _association_stats(self._conn, total)

        return StoreStatus(
            total_episodes=total,
            episodes_since_wrap=since_wrap,
            total_wraps=total_wraps,
            last_wrap_at=last_wrap_at,
            wrap_in_progress=wrap_in_progress,
            tombstone_count=tombstone_count,
            continuity_chars=continuity_chars,
            episodes_by_type=episodes_by_type,
            association_stats=assoc_stats,
        )

    # -- Wrap lifecycle --

    def wrap_started(self) -> None:
        """Mark that a wrap has been initiated (prepare_wrap called)."""
        self._set_metadata("wrap_started_at", _now_utc())

        if self._audit is not None:
            self._audit.log("wrap_started")

    def wrap_cancelled(self) -> None:
        """Clear wrap-in-progress flag without recording a completed wrap.

        Use when a wrap is abandoned (no episodes, LLM failure, validation
        failure with fallback). Prevents stale-wrap detection from false-firing.
        """
        self._set_metadata("wrap_started_at", "")

        if self._audit is not None:
            self._audit.log("wrap_cancelled")

    def wrap_completed(
        self,
        episodes_compressed: int,
        continuity_chars: int,
        graduations_validated: int = 0,
        graduations_demoted: int = 0,
        citation_reuse_max: int = 0,
        patterns_extracted: int = 0,
        associations_formed: int = 0,
        associations_strengthened: int = 0,
        associations_decayed: int = 0,
    ) -> WrapResult:
        """Record a completed wrap and clear the in-progress flag.

        Returns:
            WrapResult with the wrap metrics.
        """
        self._conn.execute(
            """INSERT INTO wraps
               (wrapped_at, episodes_compressed, continuity_chars, graduations_validated,
                graduations_demoted, citation_reuse_max, patterns_extracted,
                associations_formed, associations_strengthened, associations_decayed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                _now_utc(),
                episodes_compressed,
                continuity_chars,
                graduations_validated,
                graduations_demoted,
                citation_reuse_max,
                patterns_extracted,
                associations_formed,
                associations_strengthened,
                associations_decayed,
            ),
        )
        # Update session_id for episodes in this wrap cycle
        last_wrap = self._conn.execute(
            "SELECT id FROM wraps ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if last_wrap:
            session_id = str(last_wrap["id"])
            self._conn.execute(
                "UPDATE episodes SET session_id = ? WHERE session_id IS NULL",
                (session_id,),
            )
        self._set_metadata("wrap_started_at", "")
        self._conn.commit()

        if self._audit is not None:
            self._audit.log("wrap_completed", {
                "episodes_compressed": episodes_compressed,
                "continuity_chars": continuity_chars,
                "graduations_validated": graduations_validated,
                "graduations_demoted": graduations_demoted,
                "patterns_extracted": patterns_extracted,
                "associations_formed": associations_formed,
                "associations_strengthened": associations_strengthened,
                "associations_decayed": associations_decayed,
            })

        # Auto-prune old episodes if retention_days is configured
        pruned = 0
        if self._retention_days is not None:
            pruned = self.prune()

        return WrapResult(
            saved=True,
            chars=continuity_chars,
            section_sizes={},
            graduations_validated=graduations_validated,
            graduations_demoted=graduations_demoted,
            citation_reuse_max=citation_reuse_max,
            patterns_extracted=patterns_extracted,
            episodes_compressed=episodes_compressed,
            pruned_count=pruned,
            associations_formed=associations_formed,
            associations_strengthened=associations_strengthened,
            associations_decayed=associations_decayed,
        )

    # -- Associations (Hebbian) --

    def record_associations(
        self,
        direct_pairs: set[tuple[str, str]],
        session_pairs: set[tuple[str, str]] | None = None,
        affective_state: "AffectiveState | None" = None,
    ) -> tuple[int, int]:
        """Record or strengthen Hebbian association links from co-citation.

        Called during wrap validation after graduation citations are checked.
        Direct pairs get stronger links than session pairs.

        Args:
            direct_pairs: Episode ID pairs co-cited on the same pattern line.
            session_pairs: Episode ID pairs cited in the same wrap, different patterns.
            affective_state: Optional agent functional state during consolidation.
                Modulates association strength and is stored on the link.

        Returns:
            Tuple of (links_formed, links_strengthened).
        """
        ts = _now_utc()
        formed, strengthened = _record_associations(
            self._conn,
            direct_pairs,
            session_pairs or set(),
            ts,
            affective_state=affective_state,
        )

        if self._audit is not None and (formed or strengthened):
            audit_data: dict = {
                "formed": formed,
                "strengthened": strengthened,
                "direct_pairs": len(direct_pairs),
                "session_pairs": len(session_pairs) if session_pairs else 0,
            }
            if affective_state is not None:
                audit_data["affective_tag"] = affective_state.tag
                audit_data["affective_intensity"] = affective_state.intensity
            self._audit.log("associations_updated", audit_data)

        return formed, strengthened

    def decay_associations(
        self,
        strengthened_pairs: set[tuple[str, str]] | None = None,
        decay_factor: float = 0.9,
        cleanup_threshold: float = 0.1,
    ) -> int:
        """Decay associations not reinforced this wrap.

        Args:
            strengthened_pairs: Canonical pairs strengthened this wrap (skipped).
            decay_factor: Multiplier for unreinforced links.
            cleanup_threshold: Delete links weaker than this.

        Returns:
            Number of associations decayed (including deleted).
        """
        # Canonicalize the pairs (filter out self-pairs)
        canonical = set()
        if strengthened_pairs:
            for pair in strengthened_pairs:
                cp = canonical_pair(*pair)
                if cp is not None:
                    canonical.add(cp)

        decayed = _decay_associations(
            self._conn, canonical, decay_factor, cleanup_threshold
        )

        if self._audit is not None and decayed:
            self._audit.log("associations_decayed", {
                "decayed": decayed,
                "decay_factor": decay_factor,
                "cleanup_threshold": cleanup_threshold,
            })

        return decayed

    def get_associations(
        self,
        episode_ids: list[str],
        min_strength: float = 0.0,
        limit: int = 50,
    ) -> list[AssociationPair]:
        """Get Hebbian associations for the given episodes.

        Args:
            episode_ids: Episode IDs to query.
            min_strength: Minimum strength to include.
            limit: Maximum results.

        Returns:
            List of AssociationPair ordered by strength descending.
        """
        return _get_associations(self._conn, episode_ids, min_strength, limit)

    def get_association_context(
        self,
        episode_ids: list[str],
        min_strength: float = 0.5,
        limit: int = 20,
    ) -> str:
        """Format association context for a wrap package.

        Returns human-readable text describing which episodes have been
        thought about together before, or empty string if none.
        """
        return _get_association_context(
            self._conn, episode_ids, min_strength, limit
        )

    def association_stats(self) -> AssociationStats:
        """Get Hebbian association network health metrics."""
        total = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        return _association_stats(self._conn, total)

    # -- Pruning --

    def prune(self, older_than_days: int | None = None) -> int:
        """Prune old episodes, optionally creating tombstones.

        Args:
            older_than_days: Override retention_days for this call.
                            Uses instance retention_days if None.

        Returns:
            Number of episodes pruned.
        """
        days = older_than_days if older_than_days is not None else self._retention_days
        if days is None:
            return 0
        if days < 0:
            raise ValueError(f"older_than_days must be >= 0, got {days}")

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )

        # Find episodes to prune
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE timestamp < ?", (cutoff,)
        ).fetchall()

        if not rows:
            return 0

        pruned = 0
        for row in rows:
            if self._keep_tombstones:
                self._conn.execute(
                    """INSERT OR IGNORE INTO tombstones
                       (id, timestamp, type, content_hash)
                       VALUES (?, ?, ?, ?)""",
                    (
                        row["id"],
                        row["timestamp"],
                        row["type"],
                        _content_hash(row["content"]),
                    ),
                )
            self._conn.execute("DELETE FROM episodes WHERE id = ?", (row["id"],))
            pruned += 1

        self._conn.commit()

        if self._audit is not None and pruned > 0:
            self._audit.log("prune", {
                "count": pruned,
                "older_than_days": days,
            })

        return pruned

    # -- Continuity file I/O --

    @property
    def continuity_path(self) -> Path:
        """Path to the continuity sidecar file.

        Pattern: ./memory.db -> ./memory.continuity.md
        """
        return self._path.parent / f"{self._path.stem}.continuity.md"

    @property
    def meta_path(self) -> Path:
        """Path to the continuity metadata sidecar.

        Pattern: ./memory.db -> ./memory.continuity.meta.json
        """
        return self._path.parent / f"{self._path.stem}.continuity.meta.json"

    def load_continuity(self) -> str | None:
        """Load the current continuity file.

        Returns:
            The continuity text, or None if no continuity file exists.
        """
        if not self.continuity_path.exists():
            return None
        return self.continuity_path.read_text(encoding="utf-8")

    def save_continuity(self, text: str) -> str:
        """Save continuity text to the sidecar file. Atomic write.

        Returns:
            The path where the file was saved.
        """
        path = self.continuity_path
        tmp_path = path.with_suffix(".md.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(path)
        except Exception:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise

        if self._audit is not None:
            self._audit.log("continuity_saved", {
                "chars": len(text),
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            })

        return str(path)

    def load_meta(self) -> dict:
        """Load continuity metadata from the JSON sidecar."""
        defaults = {"sessions_produced": 0, "citations_seen": False, "format_version": 1}
        if not self.meta_path.exists():
            return defaults
        try:
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
            return {**defaults, **data}
        except (json.JSONDecodeError, OSError):
            return defaults

    def save_meta(self, meta: dict) -> str:
        """Save continuity metadata. Atomic write."""
        path = self.meta_path
        tmp_path = path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
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
        return str(path)

    # -- Properties --

    @property
    def path(self) -> Path:
        """Path to the SQLite database."""
        return self._path

    @property
    def project_name(self) -> str:
        """Project name for continuity file headers."""
        return self._project_name

    # -- Internal helpers --

    def _count_episodes_since_wrap(self) -> int:
        """Count episodes since last wrap without loading them into memory."""
        last_wrap = self._conn.execute(
            "SELECT id FROM wraps ORDER BY id DESC LIMIT 1"
        ).fetchone()

        if last_wrap:
            row = self._conn.execute(
                """SELECT COUNT(*) FROM episodes
                   WHERE CAST(session_id AS INTEGER) > ? OR session_id IS NULL""",
                (last_wrap["id"],),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()

        return row[0]

    def _current_session_id(self) -> str | None:
        """Get current session ID (last wrap ID + 1, or None if no wraps).

        SESSION ID LIFECYCLE (cross-reference — these 4 methods form a state machine):
        1. _current_session_id(): Returns str(last_wrap_id + 1) or None before first wrap.
           Called during record() to assign session_id at episode creation time.
        2. wrap_completed(): UPDATEs episodes with session_id IS NULL to str(wrap_id).
           Only matters for the first wrap cycle (pre-first-wrap episodes have NULL).
        3. episodes_since_wrap(): Uses CAST(session_id AS INTEGER) > last_wrap_id.
           Finds episodes belonging to the current (unwrapped) session.
        4. _count_episodes_since_wrap(): Same query as #3 but SELECT COUNT(*).
           Must stay in sync with #3.
        """
        last_wrap = self._conn.execute(
            "SELECT id FROM wraps ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if last_wrap:
            return str(last_wrap["id"] + 1)
        return None

    def _get_metadata(self, key: str) -> str:
        """Get a metadata value."""
        row = self._conn.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else ""

    def _set_metadata(self, key: str, value: str) -> None:
        """Set a metadata value."""
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    @staticmethod
    def _row_to_episode(row: sqlite3.Row) -> Episode:
        """Convert a database row to an Episode."""
        meta = None
        if row["metadata"]:
            try:
                meta = json.loads(row["metadata"])
            except json.JSONDecodeError:
                import sys
                print(
                    f"[anneal-memory] WARNING: corrupt metadata JSON for episode {row['id']}",
                    file=sys.stderr,
                )
        return Episode(
            id=row["id"],
            timestamp=row["timestamp"],
            type=EpisodeType(row["type"]),
            content=row["content"],
            source=row["source"],
            session_id=row["session_id"],
            metadata=meta,
        )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
