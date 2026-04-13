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
import re
import sqlite3
import uuid
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal

# Shared wrap-token shape constant. Mirrors the JSON schema pattern in
# ``integrity.py`` for the ``save_continuity`` MCP tool's
# ``wrap_token`` field. Lives at ``store.py`` module scope so both
# the CLI and the MCP server import from the store layer (not across
# the cli ↔ server boundary). Moved from ``server.py`` in 10.5c.5
# L1-review fix — importing from a transport module for a shape
# constant was an architectural wart (CLI had to load the entire MCP
# server module to reach one compiled regex). Both transports now
# import from here.
_WRAP_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")

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
    WrapRecord,
    WrapResult,
    WrapSnapshot,
)

class AnnealMemoryError(Exception):
    """Base class for all anneal-memory library errors.

    Users integrating anneal-memory should catch this at their
    boundary to handle "something in the memory system broke" as a
    single category, regardless of which subsystem raised it
    (store I/O, future SQLite wrapping, future audit integrity,
    etc.). All domain-specific errors in the library inherit from
    this class.

    anneal-memory does **not** subclass :class:`OSError` for its
    domain errors. File-write failures are wrapped in
    :class:`StoreError` and the original ``OSError`` is preserved via
    ``__cause__`` chaining (``raise StoreError(...) from exc``).
    Library-level domain errors belong in a library-level hierarchy;
    users who need the underlying errno dig via ``err.__cause__``.
    This mirrors the convention in ``sqlalchemy.exc``, ``httpx``, and
    other mature library-level Python packages — the boundary is
    "the library failed," not "a file operation failed."

    Subclasses:

    - :class:`StoreError` — store I/O or integrity failure

    .. note::
        Currently colocated in ``store.py`` because ``StoreError`` is
        the only subclass and is raised only from store methods. When
        the hierarchy grows beyond store-family errors (e.g. future
        ``PartialCommitError`` from the 10.5c.5 two-phase-commit work
        would straddle store+continuity), the base class and its
        subclasses will move to a dedicated ``exceptions.py`` module.
        Import paths via ``anneal_memory.AnnealMemoryError`` will
        remain stable across the relocation.
    """


# ``operation`` values are typed as a Literal so callers catching
# ``StoreError`` can safely switch on them with autocomplete and
# type-checker coverage.
#
# **Enforcement is compile-time only** — this is a soft contract
# until mypy-in-CI lands (see ``projects/anneal_memory/next.md``
# Session 10.5d+). Without a static type gate, a new raise site can
# drift from this alias and only fail at runtime when a user tries
# exhaustive narrowing. The convention until then: new raise sites
# MUST add their identifier to the Literal before raising, and
# reviewers MUST reject diffs that raise with a value not in the
# alias. When 10.5c.6 adds SQLite-origin operations, if the growth
# pressure makes this brittle, promote the alias to an ``Enum`` (or
# enforce statically via mypy once 10.5d ships — either works).
StoreOperation = Literal[
    "save_continuity",
    "save_meta",
    "load_wrap_snapshot",
    "wrap_completed",
]


class StoreError(AnnealMemoryError):
    """Raised when a store I/O or integrity operation fails.

    Carries operation context (``operation``, ``path``) so transports
    can surface meaningful errors to agents and users without parsing
    ``errno``/``strerror`` strings. The underlying exception is
    always attached as ``__cause__`` (``raise StoreError(...) from exc``),
    so users who need errno or the original traceback can dig one
    level deeper.

    Raised by:

    - :meth:`Store.save_continuity` — atomic continuity file write
    - :meth:`Store.save_meta` — atomic metadata sidecar write

    These are the file-write paths where a transport boundary is
    expected to translate I/O failures into protocol-level errors.

    **Not currently wrapped** (tracked as follow-up work — see
    ``projects/anneal_memory/next.md``):

    - ``sqlite3.OperationalError`` and related DB errors from the
      episode/wrap/associations path. These continue to propagate
      bare as of 10.5c.3. Wrapping them into
      ``StoreDatabaseError(StoreError)`` is filed as a scheduled
      follow-up; doing so is a non-breaking expansion of this
      hierarchy because both new and old errors will be catchable
      as ``StoreError`` or ``AnnealMemoryError``.

    **Pickle / copy safety.** ``StoreError`` uses a keyword-only
    constructor for the context fields but implements ``__reduce__``
    so the instance round-trips cleanly through ``pickle``,
    ``copy.copy``, and ``copy.deepcopy``. This matters for downstream
    consumers that marshal exceptions across process boundaries
    (``concurrent.futures.ProcessPoolExecutor``, ``pytest-xdist``,
    logging frameworks that pickle exception context, RPC
    transports).

    .. note::
        ``__cause__`` is **not** preserved across pickle/copy round
        trips. This is a standard Python limitation — the default
        exception pickling model does not serialize the cause chain,
        and reconstructing the original exception on the receiving
        side would require pickling arbitrary exception types. The
        message on the wrapped ``StoreError`` already embeds the
        underlying failure text (e.g. ``"Failed to write continuity
        file to /path: [Errno 28] No space left on device"``), so
        the human-readable cause survives; only the ``__cause__``
        attribute chain does not. In-process callers see the full
        chain via ``err.__cause__`` as normal; cross-process callers
        should read the message for the cause and treat the chain
        as locally-scoped.
    """

    def __init__(
        self,
        message: str,
        *,
        operation: StoreOperation,
        path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.operation: StoreOperation = operation
        self.path = path

    def __reduce__(self) -> tuple:
        """Support pickle / copy round-trips.

        Default ``Exception.__reduce__`` reconstructs via
        ``type(self)(*self.args)`` which would lose the keyword-only
        ``operation`` and ``path`` fields (and crash with
        ``TypeError: missing 1 required keyword-only argument``).
        We override to pass them through.
        """
        return (
            _reconstruct_store_error,
            (str(self), self.operation, self.path),
        )

    def __repr__(self) -> str:
        return (
            f"StoreError({str(self)!r}, "
            f"operation={self.operation!r}, path={self.path!r})"
        )


def _reconstruct_store_error(
    message: str,
    operation: StoreOperation,
    path: str | None,
) -> StoreError:
    """Module-level reconstructor for :class:`StoreError` under pickle.

    Lives at module scope (not as a ``@classmethod``) because pickle
    requires the reconstructor to be importable by qualified name.
    """
    return StoreError(message, operation=operation, path=path)


def _safe_unlink(path: Path) -> None:
    """Delete ``path`` if it exists, swallowing any OSError.

    Used by atomic-write cleanup paths. The cleanup MUST NOT raise:
    we're already in an exception-handler or a ``finally`` block
    where the real error is the reason we're cleaning up. A second
    failure from the unlink itself (permission denied, file already
    gone) must not mask the primary exception. Any unlink failure
    leaves a stale ``.tmp`` sidecar on disk, which is recoverable on
    next successful write via ``.replace()``.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _fsync_dir(path: Path) -> None:
    """Best-effort directory fsync for crash-safe atomic writes.

    On POSIX filesystems (ext4, xfs, btrfs, APFS), ``fsync(file)``
    guarantees the file's data blocks are on disk but does NOT
    guarantee the containing directory's entry for the file is
    durable. A crash between the file fsync and a subsequent
    ``rename(2)`` can leave the directory without an entry for the
    tmp file — the rename then fails with ``ENOENT`` even though the
    data was durable. Syncing the parent directory closes this gap.

    Windows does NOT support ``os.open`` on a directory for fsync;
    on that platform this function is a no-op. macOS ``fsync`` is
    weaker than Linux (true durability needs ``F_FULLFSYNC`` via
    ``fcntl``, which Python's stdlib doesn't expose) but the
    directory entry sync still helps reduce the residual crash
    window.

    Swallows all OSError — this is a best-effort durability
    reinforcement, NOT a hard guarantee. Callers must not rely on
    this function either succeeding or raising; it's one layer of
    defense among several.
    """
    if os.name != "posix":
        return
    try:
        dir_fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        try:
            os.close(dir_fd)
        except OSError:
            pass


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
    # Wrap lifecycle state. ``wrap_started_at`` is the legacy
    # in-progress flag (ISO 8601 UTC timestamp when prepare_wrap ran,
    # empty when no wrap in progress). 10.5c.4 added two companions
    # that form the session-handshake-token snapshot: ``wrap_token``
    # is a uuid4().hex minted at prepare_wrap time, and
    # ``wrap_episode_ids`` is a JSON-encoded list of 8-char episode
    # IDs frozen at prepare time — the exact set the agent was shown
    # for compression. ``validated_save_continuity`` reads both to
    # (a) optionally verify the caller passed the right token and
    # (b) filter ``episodes_since_wrap()`` down to the frozen set
    # regardless of what's been recorded since. All three are
    # cleared together in ``wrap_completed`` and ``wrap_cancelled``
    # inside a single SQL transaction so a mid-clear crash cannot
    # leave partial state.
    #
    # **State machine invariant:** the three keys are canonically
    # either all zero-length (idle — no wrap in progress) or all
    # populated (canonical wrap in progress). Any partial state
    # (e.g. ``wrap_started_at`` set but ``wrap_token`` empty) is a
    # store-integrity failure and is surfaced as a ``StoreError``
    # from ``load_wrap_snapshot``. This is the 10.5c.4 fix-pass
    # tightening: the pre-fix-pass code tolerated the partial state
    # as a "legacy skipped_prepare" path, which silently bypassed
    # the snapshot filter. The canonical pipeline has exactly one
    # valid state machine.
    "wrap_started_at": "",
    "wrap_token": "",
    "wrap_episode_ids": "",
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

    **Wrap-lifecycle invariants (10.5c.4):** The wrap state machine is
    tracked across three metadata keys — ``wrap_started_at``,
    ``wrap_token``, ``wrap_episode_ids`` — that MUST stay in sync:

    - **All zero-length** (idle / no wrap in progress) — canonical
      state between wraps.
    - **All populated** (canonical wrap in progress) — a snapshot
      exists for the current wrap.
    - **Any partial state** (e.g. timestamp set but token empty) is
      a store integrity failure and :meth:`load_wrap_snapshot` will
      raise :class:`StoreError` on it. The canonical pipeline has
      exactly one valid shape.

    **Rule for contributors:** all metadata writes inside
    :meth:`wrap_started`, :meth:`wrap_cancelled`, and
    :meth:`wrap_completed` MUST share a single
    ``self._conn.commit()`` call. Write the ``INSERT OR REPLACE INTO
    metadata`` SQL inline — never introduce or reuse a per-key
    helper inside those methods, because a per-key helper would
    commit after every write and a crash mid-sequence would leave
    the state machine in a partial (integrity-failure) state.
    ``_set_metadata`` was intentionally removed in 10.5c.4 for
    exactly this reason.

    Args:
        path: Path to the SQLite database file. Created if it doesn't exist.
        retention_days: Optional. Prune episodes older than N days. None = keep all.
        keep_tombstones: When pruning or deleting, preserve a tombstone row
            (episode ID, original timestamp, episode type, SHA-256 content hash)
            as an existence proof for the audit trail. Original text is erased.
            Default True.
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

        # Two-phase commit / batch mode (10.5c.5). When ``_defer_commit``
        # is True, write-path methods accumulate DML without calling
        # ``self._conn.commit()`` and queue audit events into
        # ``_deferred_audits`` instead of firing them immediately. The
        # :meth:`_batch` context manager manages the flag and flushes
        # both on successful exit. Audit events are only replayed AFTER
        # the outer DB commit succeeds, preserving the "audit after
        # externalize" invariant the rest of the codebase relies on.
        self._defer_commit: bool = False
        # Deferred audit queue: each entry is
        # ``(event, payload, kwargs)`` — the kwargs slot carries
        # pass-through keyword arguments for ``AuditTrail.log``
        # (e.g. ``actor="source"`` for record events). 10.5c.5 L3
        # Fix #17 widened the tuple from 2 to 3 elements.
        self._deferred_audits: list[
            tuple[str, dict[str, Any] | None, dict[str, Any]]
        ] = []

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
            # synchronous=FULL makes commit() fsync the WAL so a
            # successful COMMIT is durable at the block-device layer.
            # Under the default synchronous=NORMAL (which WAL
            # promotes from NORMAL's weaker semantic to "fsync on
            # checkpoint but not on every commit") a post-commit
            # power loss can revert the most recent committed
            # transactions. 10.5c.5's two-phase commit ordering
            # assumes "DB commit returns ⇒ data is durable" — FULL
            # is required for that invariant to actually hold. The
            # cost is ~1 extra fsync per commit, paid only at wrap
            # boundaries in the canonical pipeline, which is
            # negligible against the fsync already happening for
            # the continuity/meta tmp writes. L2 L3.
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._init_schema()
            # 10.5c.5 L3 Fix #13: detect orphan tmp sidecars from a
            # prior crashed pipeline before any new wrap runs. If a
            # previous ``validated_save_continuity`` crashed between
            # DB commit and file rename, the tmp files hold committed
            # content that needs manual recovery. We emit a warning
            # (not a hard raise) so programmatic callers aren't
            # broken and interactive users see the issue. Detection
            # must run AFTER _init_schema so the metadata table
            # exists — Fix #21 cross-references active wrap_token
            # from that table to avoid flagging live in-flight tmps.
            self._warn_orphan_tmp_files()
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

        # Retry with incrementing nonce on ID collision (birthday or duplicate content).
        # 10.5c.5 L3 Fix #17: batch-aware commit. If this method is
        # called inside a ``_batch()`` context, skip the commit so
        # the outer pipeline owns the single commit boundary. The
        # canonical 10.5c.5 pipeline does NOT call record() inside
        # a batch, but ``_batch()`` is documented as a general
        # primitive so extending it to new pipelines should not
        # silently break atomicity. L3 contrarian F1.
        max_retries = 3
        for nonce in range(max_retries):
            ep_id = _episode_id(content, ts, nonce)
            try:
                self._conn.execute(
                    """INSERT INTO episodes (id, timestamp, type, content, source, session_id, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (ep_id, ts, episode_type.value, content, source, session_id, meta_json),
                )
                if not self._defer_commit:
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

        # Audit goes through the batch-aware helper so the event is
        # queued (not fired) inside a batch, and flushed only after
        # the outer commit succeeds. Same ordering invariant as the
        # existing batched write methods.
        self._audit_log("record", {
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

        For corrections, mistakes, or privacy (right to erasure). The
        original content is fully erased from the episodes table. If
        ``keep_tombstones`` is enabled (default), a tombstone row is
        inserted recording the episode ID, original timestamp, episode
        type, and SHA-256 content hash — the text itself is gone but the
        audit trail retains enough metadata to prove the episode existed.

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
        # 10.5c.5 L4 Fix: batch-aware commit for consistency with
        # record() and the other write-path methods. No current
        # caller invokes delete() inside a _batch(), but making it
        # batch-safe removes a forward-looking misuse hazard and
        # matches the rest of the batch-aware method set.
        if not self._defer_commit:
            self._conn.commit()

        self._audit_log("delete", {
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

        # Audit health snapshot (Diogenes ARCH finding, Apr 13 2026).
        # Cheap visibility — lazy init inside stats() if needed, no chain
        # walk. Defaults below keep the fields backward-compatible when
        # audit is disabled.
        audit_enabled = self._audit is not None
        audit_log_path: str | None = None
        audit_entry_count: int | None = None
        audit_retention_days: int | None = None
        if self._audit is not None:
            try:
                audit_stats = self._audit.stats()
            except OSError:
                # Audit layer is enabled but its files are unreadable
                # (permissions, disk issue, etc.). Surface enabled=True
                # with None fields so the operator sees the state without
                # status() itself raising.
                pass
            else:
                audit_log_path = audit_stats["log_path"]
                audit_entry_count = audit_stats["entry_count"]
                audit_retention_days = audit_stats["retention_days"]

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
            audit_enabled=audit_enabled,
            audit_log_path=audit_log_path,
            audit_entry_count=audit_entry_count,
            audit_retention_days=audit_retention_days,
        )

    # -- Wrap lifecycle --

    def wrap_started(
        self,
        *,
        token: str = "",
        episode_ids: list[str] | None = None,
    ) -> None:
        """Mark that a wrap has been initiated (prepare_wrap called).

        **The canonical call form is ``wrap_started(token=<uuid>,
        episode_ids=<list>)``.** The no-arg form ``wrap_started()`` is
        legacy — it writes only ``wrap_started_at`` and leaves
        ``wrap_token`` / ``wrap_episode_ids`` empty, creating a state
        that :meth:`load_wrap_snapshot` now rejects as a store
        integrity failure (see the 10.5c.4 fix-pass review for the
        rationale). The no-arg form is preserved for tests and
        low-level tooling that only need the ``wrap_in_progress``
        flag semantics (e.g. ``store.status()`` reads), but emits a
        :class:`DeprecationWarning` to surface the contract drift.

        Writes the ``wrap_started_at`` flag and, when the caller
        provides a session-handshake token and frozen episode-ID
        snapshot, persists them alongside so
        :meth:`validated_save_continuity` can verify the token and
        filter its re-fetched episode set to exactly the frozen list.

        The three metadata writes happen inside a single SQL
        transaction (one ``INSERT OR REPLACE`` per key, one commit at
        the end) so a crash mid-write cannot leave the store with a
        timestamp but no token, or a token but no episode list. See
        the 10.5c.4 design note on ``_DEFAULT_METADATA`` for the
        invariant.

        Args:
            token: Session-handshake token (uuid4().hex from
                :func:`prepare_wrap`). Empty string triggers the
                legacy flag-only behavior and emits a
                :class:`DeprecationWarning`. The canonical pipeline
                always passes a real token.
            episode_ids: Frozen list of 8-char episode IDs the agent
                was shown for compression. Required when ``token`` is
                non-empty; ignored when ``token`` is empty. Stored
                JSON-encoded in the ``wrap_episode_ids`` metadata key.
        """
        if not token and episode_ids is None:
            # Legacy no-arg form. Emits a DeprecationWarning so any
            # future caller reaches for the canonical (token,
            # episode_ids) form. The state produced by this path is
            # ``wrap_started_at`` set + ``wrap_token`` empty, which
            # ``load_wrap_snapshot`` now raises StoreError on — so
            # callers on this path MUST NOT subsequently call
            # ``validated_save_continuity`` without first clearing
            # with ``wrap_cancelled``.
            warnings.warn(
                "Store.wrap_started() with no arguments is a legacy "
                "call form. The canonical pipeline is "
                "wrap_started(token=uuid.uuid4().hex, "
                "episode_ids=<list>) — see prepare_wrap() in "
                "anneal_memory.continuity for the reference caller. "
                "Calling validated_save_continuity after the no-arg "
                "form will raise StoreError from load_wrap_snapshot "
                "because the metadata is in a partial "
                "wrap-in-progress state; call wrap_cancelled() to "
                "clear it before proceeding.",
                DeprecationWarning,
                stacklevel=2,
            )
        # Validate the caller-supplied token/episode_ids combo at the
        # write boundary. The defensive asymmetry the reviewer caught:
        # load_wrap_snapshot raises StoreError for a "token present,
        # ids empty" state, but without this guard the write side
        # would happily store that exact malformed pair. Surface the
        # bad-caller early so debugging points at the write site, not
        # at the next load.
        if token and episode_ids is None:
            raise ValueError(
                "wrap_started: token was provided but episode_ids is "
                "None. The canonical pipeline must pass both together "
                "— pass episode_ids=[] explicitly if you really mean "
                "an empty snapshot."
            )
        if episode_ids is not None and not token:
            raise ValueError(
                "wrap_started: episode_ids was provided but token is "
                "empty. Snapshot episode IDs only make sense with a "
                "handshake token; pass token=uuid.uuid4().hex."
            )

        # Batch the three metadata writes into a single commit so a
        # crash mid-write cannot leave the store with a partial
        # snapshot (e.g. timestamp set but token blank, which would
        # look like legacy skipped_prepare state and silently bypass
        # the frozen-snapshot filter at save time — a state
        # ``load_wrap_snapshot`` now treats as an integrity failure
        # to catch any code path that slips past the write-side
        # validation above).
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("wrap_started_at", _now_utc()),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("wrap_token", token),
        )
        # Empty-string encoding means "no snapshot stored" (legacy
        # path where ``store.wrap_started()`` was called with no
        # args). An empty list from a validated ``token + episode_ids``
        # pair encodes as ``"[]"`` JSON so ``load_wrap_snapshot`` can
        # distinguish "caller explicitly passed an empty snapshot"
        # from "legacy no-snapshot path."
        if token:
            ids_json = json.dumps(list(episode_ids or []))
        else:
            ids_json = ""
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("wrap_episode_ids", ids_json),
        )
        self._conn.commit()

        if self._audit is not None:
            payload: dict[str, Any] = {}
            if token:
                payload["wrap_token"] = token
            if token and episode_ids:
                # Record both the count (for fast integrity scans) and
                # the full ID list (for forensic reconstruction). The
                # audit trail is the tamper-evident record, so logging
                # the full list is load-bearing for chain-of-custody
                # — with this field the audit alone can answer "which
                # episodes were shown to the agent for wrap #N" without
                # joining against the wraps + episodes tables. Key
                # names mirror the metadata keys for vocabulary
                # consistency (Layer 2 NIT-2).
                payload["wrap_episode_count"] = len(episode_ids)
                payload["wrap_episode_ids"] = list(episode_ids)
            self._audit.log("wrap_started", payload if payload else None)

    def wrap_cancelled(self) -> None:
        """Clear wrap-in-progress flag without recording a completed wrap.

        Use when a wrap is abandoned (no episodes, LLM failure, validation
        failure with fallback). Prevents stale-wrap detection from false-firing.

        Clears all three wrap-in-progress metadata keys
        (``wrap_started_at``, ``wrap_token``, ``wrap_episode_ids``) in a
        single SQL transaction, matching the batched-write invariant
        :meth:`wrap_started` establishes.
        """
        # Capture all three wrap-lifecycle keys before we clear them,
        # so the audit entry records the full chain-of-custody
        # context for the abandoned wrap. Layer 1 M3 flagged that
        # cancelled wraps were previously logging only the token,
        # forcing auditors to cross-join against the wrap_started
        # event to see which episodes were abandoned. Logging the
        # ids inline on the cancel event removes that join.
        #
        # We read ``wrap_started_at`` too so partial-state cancels
        # (the recovery scenario wrap_cancelled most exists to
        # handle) still emit a forensic trail. Pre-10.5c.5 the
        # partial-state cancel emitted NO audit event at all —
        # silent_error_swallowing inside the very tool that exists
        # to recover from silent error states. 10.5c.5 L1 MEDIUM.
        cancelled_started_at = self._get_metadata("wrap_started_at")
        cancelled_token = self._get_metadata("wrap_token")
        cancelled_ids_raw = self._get_metadata("wrap_episode_ids")

        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("wrap_started_at", ""),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("wrap_token", ""),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("wrap_episode_ids", ""),
        )
        self._conn.commit()

        if self._audit is not None:
            # Three cases:
            # 1. Healthy cancel — all three keys were set. Log full
            #    chain-of-custody with token + episode list.
            # 2. Partial state — at least one of the three keys was
            #    set but not all three. Log what we DID find plus a
            #    ``partial_state: true`` marker so auditors can
            #    distinguish "operator cleaned up a broken store"
            #    from "operator abandoned a healthy wrap."
            # 3. Clean store — none of the three keys was set. No
            #    audit event (there was nothing to cancel).
            any_state = bool(
                cancelled_started_at or cancelled_token or cancelled_ids_raw
            )
            if any_state:
                payload: dict[str, Any] = {}
                if cancelled_token:
                    payload["wrap_token"] = cancelled_token
                if cancelled_started_at:
                    payload["wrap_started_at"] = cancelled_started_at
                if cancelled_ids_raw:
                    # Audit is best-effort: a corrupt JSON here should
                    # not prevent the cancellation from being recorded.
                    try:
                        decoded = json.loads(cancelled_ids_raw)
                        if isinstance(decoded, list):
                            payload["wrap_episode_ids"] = decoded
                            payload["wrap_episode_count"] = len(decoded)
                    except json.JSONDecodeError:
                        pass
                # Healthy cancels have all three keys set. Anything
                # else is partial state — tag it so operators
                # reviewing the audit trail know this was recovery,
                # not routine abandonment.
                healthy = bool(
                    cancelled_started_at
                    and cancelled_token
                    and cancelled_ids_raw
                )
                if not healthy:
                    payload["partial_state"] = True
                self._audit.log("wrap_cancelled", payload)

    def get_wrap_started_at(self) -> str | None:
        """Return the wrap-in-progress timestamp, or None if no wrap is pending.

        Thin accessor for operator tooling (``wrap-status`` CLI
        subcommand) that needs to display *when* a wrap was started
        alongside the snapshot contents. Kept separate from
        :meth:`load_wrap_snapshot` so operator display code doesn't pay
        the partial-state integrity checks that TOCTOU callers need,
        and so the ``WrapSnapshot`` TypedDict stays minimal for its
        critical-path use.

        Returns:
            ISO 8601 UTC timestamp string (as persisted by
            :meth:`wrap_started`) if a wrap is in progress, else
            ``None``.
        """
        started = self._get_metadata("wrap_started_at")
        return started if started else None

    def load_wrap_snapshot(self) -> WrapSnapshot | None:
        """Return the frozen wrap-in-progress snapshot, or None if no wrap is pending.

        Used by :func:`validated_save_continuity` to filter the
        re-fetched episode set down to exactly what the agent was
        shown at prepare time, and to verify any caller-provided
        token. Returns ``None`` in two cases:

        1. **No wrap in progress** — ``wrap_started_at`` metadata is
           empty. This is the legacy idle state.
        2. **Legacy prepare path with no token** — ``wrap_started_at``
           is set but ``wrap_token`` is empty. This shouldn't happen
           under the canonical 10.5c.4 pipeline but stays a tolerated
           state for the ``skipped_prepare`` path where callers
           invoke ``validated_save_continuity`` without having called
           ``prepare_wrap`` first.

        Returns:
            :class:`WrapSnapshot` with ``token`` and ``episode_ids`` if
            a snapshot is stored, else ``None``.

        Raises:
            StoreError: If the ``wrap_episode_ids`` metadata key is
                populated but its JSON fails to decode — treated as a
                store-integrity failure, not silently recovered.
                ``operation`` is ``"load_wrap_snapshot"``, ``path`` is
                the store's DB file.
        """
        # Codex Layer 3 MEDIUM: batch the three metadata reads into
        # a single query so concurrent writers can't interleave
        # between them. Prior pattern called _get_metadata three
        # times, which is itself TOCTOU-prone inside the TOCTOU
        # fix. One SELECT reads all three keys atomically.
        rows = self._conn.execute(
            "SELECT key, value FROM metadata WHERE key IN (?, ?, ?)",
            ("wrap_started_at", "wrap_token", "wrap_episode_ids"),
        ).fetchall()
        meta = {row["key"]: row["value"] for row in rows}

        wrap_started = meta.get("wrap_started_at", "")
        if not wrap_started:
            return None

        token = meta.get("wrap_token", "")
        if not token:
            # wrap_started_at is set but wrap_token is empty — this is
            # a malformed state after 10.5c.4. The canonical path
            # ``prepare_wrap → store.wrap_started(token=..., episode_ids=...)``
            # writes all three keys atomically; the only way to reach
            # "timestamp present, token empty" is (a) a caller invoking
            # the no-arg ``store.wrap_started()`` form directly (bypass
            # of the canonical pipeline), (b) a mid-upgrade v0.1.x
            # database with a pre-existing stale flag, or (c) a manual
            # metadata edit. Layer 1 review caught this as a silent
            # bypass of the snapshot filter — surfacing it as a
            # StoreError is the right move: the canonical pipeline has
            # one valid state machine, and partial states are integrity
            # failures, not "try to recover silently."
            raise StoreError(
                "wrap_started_at is set but wrap_token is empty — "
                "store metadata is in a partial wrap-in-progress state. "
                "This happens if a caller bypasses the canonical "
                "prepare_wrap pipeline (e.g. by calling "
                "store.wrap_started() with no arguments) or if a v0.1.x "
                "database is mid-upgrade with a stale legacy flag. "
                "Call store.wrap_cancelled() to clear the stale flag, "
                "then re-run prepare_wrap for a clean snapshot.",
                operation="load_wrap_snapshot",
                path=str(self.path),
            )

        ids_raw = meta.get("wrap_episode_ids", "")
        if not ids_raw:
            # Token without episode IDs is a malformed state — the
            # batched write in wrap_started() should have set both or
            # neither. Surface as store integrity rather than guess.
            raise StoreError(
                "wrap_token is set but wrap_episode_ids is empty — "
                "store metadata is inconsistent. This indicates a "
                "wrap-lifecycle state machine bug or manual metadata "
                "edit. Either clear wrap-in-progress state with "
                "`wrap_cancelled()` or restore the episode ID list.",
                operation="load_wrap_snapshot",
                path=str(self.path),
            )

        try:
            episode_ids = json.loads(ids_raw)
        except json.JSONDecodeError as exc:
            raise StoreError(
                f"wrap_episode_ids metadata is not valid JSON: {exc}. "
                "Either clear wrap-in-progress state with "
                "`wrap_cancelled()` or restore the snapshot.",
                operation="load_wrap_snapshot",
                path=str(self.path),
            ) from exc

        if not isinstance(episode_ids, list) or not all(
            isinstance(x, str) for x in episode_ids
        ):
            raise StoreError(
                "wrap_episode_ids metadata decoded to an unexpected "
                f"shape ({type(episode_ids).__name__}); expected a "
                "list of 8-char episode ID strings.",
                operation="load_wrap_snapshot",
                path=str(self.path),
            )

        return WrapSnapshot(token=token, episode_ids=episode_ids)

    # SQLite default SQLITE_MAX_VARIABLE_NUMBER is 999 (raised to 32766
    # in SQLite 3.32.0 from 2020, but many platform builds still ship
    # the older limit). The 10.5c.4 ``wrap_completed`` builds a
    # dynamic IN clause from the frozen episode IDs with no chunking,
    # so we guard against the limit explicitly rather than letting a
    # bare ``sqlite3.OperationalError: too many SQL variables`` fall
    # out through the transport. The limit is documented as a
    # compile-time constant so there's no runtime query for it; we
    # use 998 (leave one placeholder for the ``session_id`` param).
    # Chunking is 10.5c.5+ work if anyone ever actually hits this.
    _MAX_SQL_VARS_IN_CLAUSE = 998

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
        section_sizes: dict[str, int] | None = None,
        *,
        episode_ids: list[str] | None = None,
        wrap_token: str | None = None,
    ) -> WrapResult:
        """Record a completed wrap and clear the in-progress flag.

        Args:
            episode_ids: The exact set of episode IDs that participated
                in this wrap. When provided, only those episodes get
                their ``session_id`` stamped with the new wrap's ID —
                any episode with ``session_id IS NULL`` that is NOT in
                the list stays null, so it naturally falls into the
                next wrap's compression window. This is the fix for
                the 10.5c.4 TOCTOU window: episodes recorded between
                ``prepare_wrap`` and ``validated_save_continuity`` are
                preserved for the next wrap instead of being silently
                absorbed into this one. When ``None`` (the legacy
                ``skipped_prepare`` path), the UPDATE falls back to
                the pre-10.5c.4 behavior of stamping every episode
                with a NULL ``session_id``.
            wrap_token: The session-handshake token from the prepare
                event this completion is committing. Passed through
                to the audit payload as the chain-of-custody link
                between ``wrap_started`` and ``wrap_completed``
                events. When ``None`` (legacy or skipped_prepare
                path), no token field is added to the audit entry.
                Passed as an explicit parameter rather than re-read
                from metadata because the 10.5c.4 canonical caller
                already has the token in hand from ``load_wrap_snapshot``
                — removes a read and removes a within-method
                read-before-clear sequence that Layer 1 flagged.

        Returns:
            WrapResult with the wrap metrics.

        Raises:
            StoreError: If ``episode_ids`` exceeds the SQLite variable
                limit for an IN clause (998 by default). This is a
                hard guard, not a silent truncation — chunking is
                10.5c.5+ work.
        """
        if (
            episode_ids is not None
            and len(episode_ids) > self._MAX_SQL_VARS_IN_CLAUSE
        ):
            raise StoreError(
                f"wrap_completed: episode_ids has {len(episode_ids)} "
                f"entries, which exceeds the SQLite IN-clause limit "
                f"of {self._MAX_SQL_VARS_IN_CLAUSE} (platform-dependent "
                f"default). Chunking large wraps is scheduled for "
                f"10.5c.5+; as a workaround, compress in two passes "
                f"with fewer episodes per wrap.",
                operation="wrap_completed",
                path=str(self.path),
            )

        # Replay-race closure: if the caller passed a wrap_token (i.e.
        # they came through the canonical pipeline with a verified
        # snapshot at save time), re-verify the token inside the write
        # transaction using a compare-and-swap UPDATE. The CAS UPDATE
        # is the first DML in this method, which opens SQLite's
        # deferred transaction and acquires the reserved write lock.
        # If the UPDATE affects zero rows, the token has changed
        # (another process ran prepare_wrap / wrap_cancelled /
        # wrap_completed between validated_save_continuity's earlier
        # token check and now) — raise and rollback so no partial
        # wraps row gets inserted.
        #
        # Codex Layer 3 HIGH finding: without this CAS, two concurrent
        # save_continuity calls using the same valid token could both
        # pass their earlier verification at continuity.py:700 and
        # both proceed to insert separate wraps rows. In the current
        # single-process model that race is theoretical, but the
        # whole point of the 10.5c.4 fix is "TOCTOU structurally
        # impossible" — leaving a second race open contradicts the
        # thesis. Closing it costs ~10 lines and one test.
        if wrap_token is not None:
            cas_cursor = self._conn.execute(
                "UPDATE metadata SET value = '' "
                "WHERE key = 'wrap_token' AND value = ?",
                (wrap_token,),
            )
            if cas_cursor.rowcount != 1:
                # Roll back the implicit transaction the CAS opened
                # (legacy path) OR let the outer ``_batch.__exit__``
                # handle rollback (batched path). In batched mode the
                # enclosing transaction may contain uncommitted DML
                # from ``record_associations`` + ``decay_associations``
                # that must be discarded together with the failed CAS
                # — that's exactly what ``_batch.__exit__`` does on
                # exception. Double-rolling back would be harmless but
                # semantically muddled; 10.5c.5 L2 review flagged the
                # ownership confusion. Fix: rollback here only for the
                # legacy (non-batched) caller, which is still a valid
                # path for library users managing their own wrap
                # lifecycle. The caller sees a ValueError that's
                # distinguishable from the earlier mismatch at
                # continuity.py (different message) so an operator
                # knows a concurrent process interfered versus a
                # client-side stale-token bug.
                if not self._defer_commit:
                    self._conn.rollback()
                raise ValueError(
                    f"wrap_completed: wrap_token has been cleared or "
                    f"replaced during save. The persisted token no "
                    f"longer matches '{wrap_token[:8]}…' — a concurrent "
                    f"process probably ran prepare_wrap, wrap_cancelled, "
                    f"or wrap_completed between this session's token "
                    f"verification and the save commit. Re-run "
                    f"prepare_wrap to start a fresh wrap."
                )

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
        # Update session_id for episodes in this wrap cycle.
        #
        # Pre-10.5c.4 behavior: stamp every NULL-session_id episode
        # with the new wrap's ID. This still runs on the legacy
        # ``skipped_prepare`` path when ``episode_ids`` is None.
        #
        # 10.5c.4 behavior: when the caller provides the frozen
        # snapshot ID list, restrict the UPDATE to exactly those IDs.
        # Any NULL-session_id episodes recorded AFTER prepare_wrap
        # (the TOCTOU window) are NOT in the list, so they stay NULL
        # and land in the next wrap's ``episodes_since_wrap()`` result.
        # Data loss is impossible — new episodes are preserved, just
        # carried over to the wrap they semantically belong to.
        last_wrap = self._conn.execute(
            "SELECT id FROM wraps ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if last_wrap:
            session_id = str(last_wrap["id"])
            if episode_ids is None:
                # Legacy / skipped_prepare path.
                self._conn.execute(
                    "UPDATE episodes SET session_id = ? WHERE session_id IS NULL",
                    (session_id,),
                )
            elif episode_ids:
                # Frozen snapshot path. Build a dynamic IN clause —
                # SQLite supports up to 999 parameters by default, which
                # covers the realistic per-session episode count by a
                # comfortable margin. If a wrap ever exceeds that, the
                # caller should chunk the ID list and call us per chunk
                # (we don't do it here because the batching concerns
                # belong upstream where chunking the compression is also
                # a relevant choice).
                placeholders = ",".join("?" for _ in episode_ids)
                self._conn.execute(
                    f"UPDATE episodes SET session_id = ? "
                    f"WHERE session_id IS NULL AND id IN ({placeholders})",
                    (session_id, *episode_ids),
                )
            # If episode_ids is an empty list, skip the UPDATE entirely.
            # An empty snapshot means "this wrap compressed nothing" —
            # the prepare_wrap empty path already calls wrap_cancelled
            # instead of wrap_completed, so the only way to reach here
            # with an empty list is a library user deliberately
            # recording a no-op wrap. Do nothing rather than
            # accidentally stamping all NULL episodes.

        # Clear all three wrap-in-progress metadata keys in the same
        # SQL transaction as the wraps INSERT + episodes UPDATE so a
        # mid-clear crash cannot leave partial state (e.g. wraps row
        # committed but wrap_started_at still set, which would look
        # like a stuck wrap on the next session's prepare_wrap call).
        # Using bare ``execute`` instead of ``_set_metadata`` so the
        # three writes share the single ``commit()`` below.
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("wrap_started_at", ""),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("wrap_token", ""),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("wrap_episode_ids", ""),
        )
        # 10.5c.5: defer the commit when we're inside a ``_batch()``
        # context — the outer two-phase-commit pipeline wraps the
        # whole sequence (associations + meta tmp write + wrap_completed)
        # in a single SQLite transaction so a mid-pipeline crash
        # cannot leave the wraps table ahead of the episodes.session_id
        # UPDATE or the metadata-key clears.
        if not self._defer_commit:
            self._conn.commit()

        audit_payload: dict[str, Any] = {
            "episodes_compressed": episodes_compressed,
            "continuity_chars": continuity_chars,
            "graduations_validated": graduations_validated,
            "graduations_demoted": graduations_demoted,
            "patterns_extracted": patterns_extracted,
            "associations_formed": associations_formed,
            "associations_strengthened": associations_strengthened,
            "associations_decayed": associations_decayed,
        }
        # Chain-of-custody link: the token ties this completion
        # event to the earlier wrap_started audit event that carried
        # the same token + the full wrap_episode_ids. An auditor
        # reconstructing a specific wrap walks wrap_started →
        # (optional wrap_cancelled) → wrap_completed events by
        # matching ``wrap_token``. The token is passed in by the
        # caller rather than re-read from metadata to avoid a
        # within-method read-before-clear sequence.
        if wrap_token:
            audit_payload["wrap_token"] = wrap_token
        # Route through the batch-aware audit helper so the event is
        # queued (and flushed only after the outer commit succeeds)
        # when we're inside ``_batch()``, and fired immediately
        # otherwise.
        self._audit_log("wrap_completed", audit_payload)

        # Auto-prune old episodes if retention_days is configured.
        # Skip inside a batch — auto-prune is a separate DML burst
        # with its own commit semantics that don't belong inside the
        # wrap two-phase commit. The pipeline caller can invoke prune
        # after the batch exits if desired.
        pruned = 0
        if self._retention_days is not None and not self._defer_commit:
            pruned = self.prune()

        return WrapResult(
            saved=True,
            chars=continuity_chars,
            section_sizes=dict(section_sizes) if section_sizes else {},
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

    def get_wrap_history(self) -> list[WrapRecord]:
        """Return the full wrap history in chronological order.

        Public read API for the wraps table. Replaces ad-hoc ``_conn``
        access in CLI monitoring subcommands (``history``, ``diff``,
        ``stats``, ``export``).

        Returns an empty list only when the wraps table does not exist
        yet (unmigrated legacy DB). Any other ``sqlite3.OperationalError``
        (locked database, corruption, connection gone) propagates so
        callers can surface the real failure instead of silently
        reporting "no wraps" during an outage.

        Returns:
            List of WrapRecord, oldest first.

        Raises:
            sqlite3.OperationalError: On any database failure other
                than a missing ``wraps`` table.
        """
        try:
            rows = self._conn.execute(
                """SELECT id, wrapped_at, episodes_compressed, continuity_chars,
                          graduations_validated, graduations_demoted,
                          citation_reuse_max, patterns_extracted,
                          associations_formed, associations_strengthened,
                          associations_decayed
                   FROM wraps ORDER BY id ASC"""
            ).fetchall()
        except sqlite3.OperationalError as exc:
            # Only swallow "no such table" — every other OperationalError
            # (database locked, corruption, disk full, connection gone)
            # indicates a real failure that monitoring subcommands MUST
            # surface rather than silently report an empty history.
            if "no such table" in str(exc).lower():
                return []
            raise

        # Coerce nullable columns (episodes_compressed, continuity_chars)
        # and legacy-default columns to 0 at construction, so the
        # WrapRecord dataclass can expose non-None int fields and callers
        # don't need to sprinkle `or 0` guards at every consumption site.
        return [
            WrapRecord(
                id=row["id"],
                wrapped_at=row["wrapped_at"],
                episodes_compressed=row["episodes_compressed"] or 0,
                continuity_chars=row["continuity_chars"] or 0,
                graduations_validated=row["graduations_validated"] or 0,
                graduations_demoted=row["graduations_demoted"] or 0,
                citation_reuse_max=row["citation_reuse_max"] or 0,
                patterns_extracted=row["patterns_extracted"] or 0,
                associations_formed=row["associations_formed"] or 0,
                associations_strengthened=row["associations_strengthened"] or 0,
                associations_decayed=row["associations_decayed"] or 0,
            )
            for row in rows
        ]

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
        # 10.5c.5: inside a ``_batch()`` the free-function helper must
        # NOT commit — the outer pipeline owns the single commit.
        formed, strengthened = _record_associations(
            self._conn,
            direct_pairs,
            session_pairs or set(),
            ts,
            affective_state=affective_state,
            commit=not self._defer_commit,
        )

        if formed or strengthened:
            audit_data: dict = {
                "formed": formed,
                "strengthened": strengthened,
                "direct_pairs": len(direct_pairs),
                "session_pairs": len(session_pairs) if session_pairs else 0,
            }
            if affective_state is not None:
                audit_data["affective_tag"] = affective_state.tag
                audit_data["affective_intensity"] = affective_state.intensity
            self._audit_log("associations_updated", audit_data)

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

        # 10.5c.5: inside a ``_batch()`` the free-function helper must
        # NOT commit — the outer pipeline owns the single commit.
        decayed = _decay_associations(
            self._conn,
            canonical,
            decay_factor,
            cleanup_threshold,
            commit=not self._defer_commit,
        )

        if decayed:
            self._audit_log("associations_decayed", {
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
        """Low-level continuity file write. **Bypasses the immune system.**

        This method writes the continuity text to its sidecar file with
        an atomic fsync-and-rename and logs a ``continuity_saved`` audit
        event. It does **not** validate structure, does **not** run
        graduation citation checking, does **not** form or decay
        Hebbian associations, does **not** update wrap metadata, and
        does **not** record a wrap in the wraps table.

        For agent-driven session wraps — the normal case — call
        :func:`anneal_memory.validated_save_continuity(store, text)`
        instead. That runs the full canonical pipeline (structure
        validation → graduation → associations → decay → metadata →
        wrap completion) which IS anneal-memory's immune system. Using
        this raw method for session wraps silently degrades the memory
        system to a flat file write.

        Legitimate uses of this method:

        - Internal orchestration by ``validated_save_continuity``
          (that is the one place it is safe to call directly — the
          surrounding pipeline has already validated and will
          continue after this write).
        - Test fixtures that seed a continuity file in a known state
          without exercising the pipeline.
        - Advanced library users explicitly managing their own
          wrap lifecycle and replicating the full pipeline by hand.

        If you are not one of those three cases, you want
        :func:`validated_save_continuity`.

        Returns:
            The path where the file was saved.

        Raises:
            StoreError: On file-I/O failure of the atomic write
                (most common case — disk full, permission denied,
                cross-device rename). The stale temp file is cleaned
                up before raising. Users catching ``StoreError`` get
                ``.operation == "save_continuity"`` and ``.path ==
                str(continuity_path)`` for clean error messages.
            Exception: Non-``OSError`` failures (e.g. an exotic
                ``UnicodeEncodeError`` from malformed agent text)
                propagate bare. The stale temp file is still cleaned
                up — the atomic-write invariant is preserved
                regardless of the exception class — but such
                failures represent bugs or data-shape problems
                rather than transport-level I/O errors and should
                not be masked behind a ``StoreError`` wrapper.
        """
        path = self.continuity_path
        # 10.5c.5: delegate the tmp-write + fsync to the shared
        # primitive, keeping the atomic-write invariant in ONE place.
        # Re-wrap any StoreError raised by the primitive with the
        # public-API operation name so callers see
        # ``operation="save_continuity"`` regardless of which internal
        # helper actually raised (preserves the pre-10.5c.5 contract).
        try:
            tmp_path = self._prepare_continuity_write(text)
        except StoreError as exc:
            raise StoreError(
                str(exc).replace("tmp sidecar", "file"),
                operation="save_continuity",
                path=str(path),
            ) from exc.__cause__
        succeeded = False
        try:
            tmp_path.replace(path)
            succeeded = True
        except OSError as exc:
            raise StoreError(
                f"Failed to rename continuity tmp to {path}: {exc}",
                operation="save_continuity",
                path=str(path),
            ) from exc
        finally:
            if not succeeded:
                _safe_unlink(tmp_path)

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
        """Save continuity metadata. Atomic write.

        Raises:
            StoreError: On file-I/O failure of the atomic write.
                Stale temp file is cleaned up before raising.
                ``.operation == "save_meta"``, ``.path == str(meta_path)``.
            Exception: Non-``OSError`` failures (``TypeError`` from a
                non-JSON-serializable ``meta`` value, etc.) propagate
                bare. Stale temp file is still cleaned up via the
                finally block — the atomic-write invariant holds
                regardless of exception class.
        """
        path = self.meta_path
        # 10.5c.5: delegate the tmp-write to the shared primitive.
        # Re-wrap primitive StoreErrors with the public-API operation
        # name so the pre-10.5c.5 contract holds.
        try:
            tmp_path = self._prepare_meta_write(meta)
        except StoreError as exc:
            raise StoreError(
                str(exc).replace("tmp sidecar", "sidecar"),
                operation="save_meta",
                path=str(path),
            ) from exc.__cause__
        succeeded = False
        try:
            tmp_path.replace(path)
            succeeded = True
        except OSError as exc:
            raise StoreError(
                f"Failed to rename meta tmp to {path}: {exc}",
                operation="save_meta",
                path=str(path),
            ) from exc
        finally:
            if not succeeded:
                _safe_unlink(tmp_path)
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

    # NOTE: ``_set_metadata`` was removed in 10.5c.4. The rationale
    # and forward-looking rule live in the Store class docstring
    # ("Wrap-lifecycle invariants"). Do not reintroduce a single-key
    # helper inside wrap_started/wrap_cancelled/wrap_completed.


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

    # -- Two-phase commit / batched write mode (10.5c.5) --

    def _audit_log(
        self,
        event: str,
        payload: dict[str, Any] | None,
        **kwargs: Any,
    ) -> None:
        """Audit log with batch-aware deferral.

        Fires the audit event immediately when the store is in normal
        mode. During a :meth:`_batch` context, queues the event into
        ``_deferred_audits`` so it can be flushed only AFTER the outer
        DB commit succeeds — preserving the "audit after externalize"
        invariant even across batched multi-step writes.

        Additional keyword arguments (e.g. ``actor="source"`` for
        ``record()`` events) are passed through to ``AuditTrail.log``
        when firing and stored alongside the queued tuple when
        deferring. The tuple shape is
        ``(event, payload, kwargs)`` — stable internal contract
        since 10.5c.5 L3 Fix #17.

        No-op if audit is disabled (``self._audit is None``).
        """
        if self._audit is None:
            return
        if self._defer_commit:
            self._deferred_audits.append((event, payload, kwargs))
        else:
            self._audit.log(event, payload, **kwargs)

    @contextmanager
    def _batch(self) -> Iterator[None]:
        """Suspend auto-commit + audit-firing for a multi-step pipeline.

        The 10.5c.5 two-phase commit primitive. Write-path methods
        check ``self._defer_commit`` and skip their own ``commit()``
        calls while the flag is True; audit events route through
        :meth:`_audit_log`, which queues them into
        ``_deferred_audits``. On successful ``__exit__`` this context
        manager does the single outer commit and then replays the
        queued audit events, preserving the "audit only fires for
        committed DML" invariant. On any exception it rolls back the
        accumulated DML and discards the queued audits.

        The caller is responsible for ordering file-side externalizations
        (atomic renames of ``continuity.md.tmp`` and ``meta.json.tmp``)
        AFTER the ``with`` block exits. The batch context only covers
        the SQLite side; file atomicity is managed by
        :meth:`_prepare_continuity_write` + :meth:`_prepare_meta_write`
        at the caller level.

        Nested use is not supported — nesting ``_batch()`` calls raises
        ``RuntimeError``. Two-phase commit has exactly one outer
        transaction by design.

        **Audit-flush exceptions are swallowed after successful commit.**
        Once the outer commit has returned, any exception from
        replaying ``_deferred_audits`` is trapped inside ``__exit__``
        and not re-raised. This preserves the load-bearing invariant
        that a successful ``_batch()`` exit means "DB committed +
        safe for the caller to proceed with file externalization."
        Propagating an audit-flush failure would trick the caller's
        outer ``except`` clause into cleaning up tmp files that
        represent committed state — a data-loss path. L2 review M2.

        **Python sqlite3 stdlib implicit-BEGIN quirk:** the store uses
        the default ``isolation_level=""`` (legacy transaction
        control). In theory this means ``BEGIN`` only fires before
        DML, not before SELECT, so a read-only batch could leave the
        connection with no open transaction at exit. In practice the
        canonical 10.5c.5 pipeline always issues DML before any point
        where rollback matters, and the
        ``test_batched_dml_invisible_to_other_connection_until_commit``
        regression gate empirically verifies that batched DML is in
        a single transaction under the Python version this ships on.
        If you ever add a read-only batch consumer, revisit this
        assumption explicitly.

        **Not reentrant. Not thread-safe. Not task-safe.** The
        ``_defer_commit`` + ``_deferred_audits`` state lives on
        ``self``. Do not share a Store instance across threads or
        asyncio tasks while a batch is in flight.

        **Batch-aware methods** (check ``_defer_commit`` and route
        audit events through :meth:`_audit_log`):

        - :meth:`record` (episode writes)
        - :meth:`delete` (episode deletions)
        - :meth:`record_associations` / :meth:`decay_associations`
        - :meth:`wrap_completed`

        All other write methods (``prune``, ``wrap_started``,
        ``wrap_cancelled``, ``save_continuity``, ``save_meta``)
        commit immediately and are NOT safe to call inside a batch —
        doing so would break the single-transaction invariant.
        Extend this list by adding the batch-aware guard to any new
        write method that becomes part of a batched pipeline. L3
        contrarian F1 flagged this as a forward-looking hazard;
        check this list whenever adding a new write path.

        **macOS durability caveat:** ``os.fsync`` on macOS (APFS)
        flushes to the drive cache but NOT to the physical medium.
        True durability requires ``fcntl(F_FULLFSYNC)``, which
        Python's stdlib does not expose. The two-phase commit
        ordering (DB commit → file rename) still holds under normal
        crashes, but a power loss in the wrong microsecond can lose
        the most recent committed wrap on macOS. L2 H1.
        """
        if self._defer_commit:
            raise RuntimeError(
                "Store._batch() does not support nesting. Two-phase "
                "commit has exactly one outer transaction; nested "
                "batches would break the commit-once invariant."
            )
        self._defer_commit = True
        self._deferred_audits = []
        commit_succeeded = False
        try:
            try:
                yield
            except BaseException:
                # Rollback the accumulated DML. SQLite's rollback is
                # safe to call even if an earlier method already
                # rolled back (legacy ``wrap_completed`` CAS path
                # outside a batch does this) — the second rollback
                # is a harmless no-op. The queued audit events are
                # discarded so no phantom audit entries are written
                # for DML that never committed.
                self._conn.rollback()
                raise
            # Single outer commit. If this raises (disk full, locked
            # DB, etc.) rollback and re-raise — the exception
            # propagates to the pipeline caller which has NOT yet
            # externalized files, so the outer ``except`` cleans up
            # tmp files correctly. The explicit rollback before
            # re-raise closes a gap L3 complement F2 flagged: a
            # failed commit leaves Python's sqlite3 wrapper in an
            # ambiguous transaction state, and subsequent store use
            # could see partial DML from the failed batch. Rolling
            # back explicitly restores clean state.
            try:
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            commit_succeeded = True
        finally:
            # Unconditional flag reset — even if SIGINT fires between
            # the commit and the audit flush, the store does NOT
            # stay poisoned in ``_defer_commit=True``. L2 review H2.
            deferred = self._deferred_audits
            self._deferred_audits = []
            self._defer_commit = False

        # Commit succeeded. Flush deferred audits with exception
        # swallowing — the DB is already committed and the load-
        # bearing invariant is "successful batch exit ⇒ caller can
        # externalize files." An audit-flush exception here must not
        # propagate; it would trick the caller's ``except`` into
        # unlinking tmp files that represent committed state (L2 M2
        # / L1 audit-flush-destroys-committed-state path). Audit is
        # best-effort at this point; we trade a missing audit event
        # for durability of the actual wrap data.
        if commit_succeeded and self._audit is not None:
            for event, payload, kwargs in deferred:
                try:
                    self._audit.log(event, payload, **kwargs)
                except Exception:
                    # Best-effort. No logger wired at this layer;
                    # silently drop the failing event and continue
                    # flushing the rest. A future pass can route
                    # this to a stderr warning or metric.
                    pass

    # -- 10.5c.5 L3 fix: unique tmp sidecar filename pattern.
    #
    # Two concurrent ``validated_save_continuity`` calls would race on
    # a fixed ``*.md.tmp`` / ``*.json.tmp`` path — Process B's write
    # could overwrite Process A's content before Process A renames,
    # silently externalizing the wrong continuity text. The CAS token
    # closes the DB race but happens AFTER Phase 1 tmp writes; the
    # filesystem race is a distinct failure mode all three L3
    # reviewers flagged independently (complement F1, gemini, contrarian F2).
    #
    # Fix: each prepare_*_write call generates a fresh ~12 char uuid
    # suffix and writes to a unique tmp path. The suffix pattern
    # (``continuity.{hex}.md.tmp`` / ``meta.{hex}.json.tmp``) stays
    # globbable for startup orphan detection and operator recovery
    # (L3 Fix #13).
    _TMP_CONTINUITY_GLOB = "*.md.tmp"
    _TMP_META_GLOB = "*.json.tmp"

    def _continuity_tmp_path(self, token_hex: str | None = None) -> Path:
        """Return a tmp sidecar path for continuity writes.

        Pattern: ``<continuity_stem>.<12hex>.md.tmp`` (sits beside
        the real continuity file). The hex suffix is either a
        provided token prefix (for pairing with meta tmp writes in
        the same wrap) or a fresh ``uuid.uuid4()`` hex slice for the
        legacy standalone caller path.

        **Pairing (10.5c.5 L3 Fix #19, codex HIGH).** When called
        from the canonical pipeline, both ``_prepare_continuity_write``
        and ``_prepare_meta_write`` pass the SAME wrap_token-derived
        suffix so the two tmp files share an identifying prefix:
        ``mystore.continuity.<token>.md.tmp`` +
        ``mystore.meta.<token>.json.tmp``. Operators recovering from
        multiple crashed wraps can pair files by token prefix,
        which is impossible when each call uses an independent
        random suffix.
        """
        suffix = token_hex or uuid.uuid4().hex[:12]
        final = self.continuity_path
        return final.with_name(f"{final.stem}.{suffix}.md.tmp")

    def _meta_tmp_path(self, token_hex: str | None = None) -> Path:
        """Return a tmp sidecar path for meta writes.

        Same pairing-via-token semantics as
        :meth:`_continuity_tmp_path`; callers in the canonical
        pipeline pass the wrap-token-derived suffix so continuity
        and meta tmp files share an identifying prefix. 10.5c.5
        L3 Fix #19.
        """
        suffix = token_hex or uuid.uuid4().hex[:12]
        final = self.meta_path
        return final.with_name(f"{final.stem}.{suffix}.json.tmp")

    def _find_orphan_tmp_files(self) -> list[Path]:
        """Return list of orphan tmp sidecars from prior crashed pipelines.

        A crashed ``validated_save_continuity`` between DB commit and
        file rename leaves tmp sidecars on disk holding committed
        content (L3 Fix #12's unique-uuid tmp paths make this
        detectable rather than racy). This helper globs the
        continuity / meta sidecar directories for any matching tmp
        files and returns them sorted. Operator recovery: rename each
        orphan tmp file to its final destination (``mv
        mystore.continuity.ab12cd34.md.tmp mystore.continuity.md``).

        **Active-wrap filtering (10.5c.5 L3 Fix #21, codex MEDIUM):**
        tmp files whose embedded token matches the currently-active
        ``wrap_token`` metadata are NOT treated as orphans — they
        belong to an in-flight pipeline that hasn't reached Phase 3
        renames yet. Filtering these out avoids a false-positive
        warning when a second Store instance opens the same DB
        during another writer's batch window. The post-commit
        pre-rename window (microseconds) still produces false
        positives because ``wrap_completed`` cleared the metadata
        token before the rename happened — that narrow case is
        accepted as documented residual risk.

        Returns:
            List of orphan tmp sidecar Paths, continuity-first then
            meta, sorted within each class for deterministic output.
            Empty list when the store is clean or when all detected
            tmp files belong to an active wrap.
        """
        # Read current active wrap token (if any) so we can filter
        # out in-flight tmps. 12-char prefix because that's what the
        # tmp filenames embed.
        active_token = self._get_metadata("wrap_token") or ""
        active_prefix = active_token[:12] if active_token else ""

        candidates: list[Path] = []
        cont_parent = self.continuity_path.parent
        cont_stem = self.continuity_path.stem  # e.g. "mystore.continuity"
        if cont_parent.exists():
            candidates.extend(
                sorted(
                    cont_parent.glob(f"{cont_stem}.*.md.tmp")
                )
            )
        meta_parent = self.meta_path.parent
        meta_stem = self.meta_path.stem  # e.g. "mystore.meta"
        if meta_parent.exists():
            candidates.extend(
                sorted(
                    meta_parent.glob(f"{meta_stem}.*.json.tmp")
                )
            )

        # Filter out tmp files whose embedded token matches the
        # currently-active wrap — those belong to an in-flight
        # pipeline, not a crashed one.
        if not active_prefix:
            return candidates
        return [
            c for c in candidates
            if self._token_from_orphan(c) != active_prefix
        ]

    def _warn_orphan_tmp_files(self) -> None:
        """Emit a ``warnings.warn`` for each orphan tmp sidecar found.

        Called from :meth:`__init__` before any pipeline activity
        runs. The warning text includes the operator-recovery
        instruction so an interactive user sees exactly what to do:
        ``mv <tmp_path> <final_path>``. Programmatic callers that
        filter warnings are unaffected; the detection is best-effort
        and non-breaking.

        Does NOT delete the files — they hold committed content and
        cleanup is the operator's explicit decision.
        """
        orphans = self._find_orphan_tmp_files()
        if not orphans:
            return

        # Group orphans by their embedded token prefix so paired
        # continuity + meta tmp files surface as a single warning.
        # The pipeline writes both tmp files with the SAME token
        # prefix (10.5c.5 L3 Fix #19), so operators recovering
        # multiple crashed wraps can match them: the `.md.tmp` with
        # token X pairs with the `.json.tmp` with token X. Without
        # this grouping, N crashed wraps would emit 2N unrelated
        # warnings with no pairing information. 10.5c.5 L3 Fix #20.
        groups: dict[str, dict[str, Path]] = {}
        for orphan in orphans:
            token = self._token_from_orphan(orphan)
            bucket = groups.setdefault(token, {})
            if orphan.name.endswith(".md.tmp"):
                bucket["cont"] = orphan
            elif orphan.name.endswith(".json.tmp"):
                bucket["meta"] = orphan

        for token, files in groups.items():
            cont_orphan = files.get("cont")
            meta_orphan = files.get("meta")
            pair_desc_parts = []
            recovery_parts = []
            if cont_orphan is not None:
                cont_final = self._final_path_for_orphan(cont_orphan)
                pair_desc_parts.append(f"continuity: {cont_orphan}")
                recovery_parts.append(f"mv '{cont_orphan}' '{cont_final}'")
            if meta_orphan is not None:
                meta_final = self._final_path_for_orphan(meta_orphan)
                pair_desc_parts.append(f"meta: {meta_orphan}")
                recovery_parts.append(f"mv '{meta_orphan}' '{meta_final}'")
            pair_desc = " + ".join(pair_desc_parts)
            recovery_cmd = " && ".join(recovery_parts)
            missing = ""
            if cont_orphan is None:
                missing = (
                    " (NOTE: no paired continuity tmp found — "
                    "only the meta sidecar is present; recovery "
                    "may be incomplete.)"
                )
            elif meta_orphan is None:
                missing = (
                    " (NOTE: no paired meta tmp found — only the "
                    "continuity file is present; recovery may be "
                    "incomplete — the meta sidecar will stay "
                    "stale after recovery.)"
                )
            # Warning text: point operators at the ``continuity_saved``
            # audit event, which is where the ``content_hash`` field
            # lives. (Earlier 10.5c.5 text pointed at ``wrap_completed``
            # which has no content_hash — codex L3 MEDIUM caught the
            # unfollowable instruction.)
            warnings.warn(
                f"anneal-memory: orphan tmp sidecar(s) detected for "
                f"wrap token prefix '{token}' — this indicates a "
                f"prior wrap pipeline crashed between DB commit and "
                f"file rename. Files: {pair_desc}.{missing} "
                f"These tmp files hold committed content and need "
                f"manual recovery. To finish the wrap, run: "
                f"{recovery_cmd}. If you're unsure whether these "
                f"represent committed state or stale debris, inspect "
                f"the audit trail for a ``continuity_saved`` event "
                f"(which carries the content_hash) or a "
                f"``wrap_completed`` event with wrap_token starting "
                f"'{token}'; a paired record in the audit trail "
                f"confirms the wrap was committed before the "
                f"crash and the recovery is safe.",
                UserWarning,
                stacklevel=3,
            )

    @staticmethod
    def _token_from_orphan(orphan: Path) -> str:
        """Extract the token prefix from an orphan tmp filename.

        Pattern: ``<stem>.<12hex>.md.tmp`` → ``<12hex>``.
        Returns ``"unknown"`` if the pattern doesn't match (legacy
        or malformed filenames).
        """
        parts = orphan.name.split(".")
        if len(parts) >= 4 and parts[-1] == "tmp":
            return parts[-3]  # the token slot
        return "unknown"

    @staticmethod
    def _final_path_for_orphan(orphan: Path) -> Path:
        """Derive the final destination path from an orphan tmp path.

        Pattern: ``<stem>.<12hex>.md.tmp`` → ``<stem>.md``.
        Pattern: ``<stem>.<12hex>.json.tmp`` → ``<stem>.json``.
        """
        # Example: mystore.continuity.abcd12345678.md.tmp
        # name parts split on '.': [mystore, continuity, abcd12345678, md, tmp]
        parts = orphan.name.split(".")
        if len(parts) >= 3 and parts[-1] == "tmp":
            # Drop the uuid (second-to-last-before-extension) and "tmp".
            # Real extension is parts[-2] (md or json).
            ext = parts[-2]
            stem_parts = parts[:-3]  # drop uuid, ext, tmp
            return orphan.with_name(".".join(stem_parts) + "." + ext)
        # Defensive fallback: strip only the ``.tmp`` suffix.
        return orphan.with_suffix("")

    def _prepare_continuity_write(
        self, text: str, token_hex: str | None = None
    ) -> Path:
        """Write continuity text to a tmp sidecar; do not rename.

        Primitive used by the 10.5c.5 two-phase commit pipeline. Writes
        a uuid-suffixed tmp sidecar (e.g.
        ``mystore.continuity.ab12cd34ef56.md.tmp``) with ``fsync`` so
        the content is durable on disk, but leaves the final
        ``continuity.md`` untouched. The caller is responsible for:

        1. Calling :meth:`_commit` (directly or via :meth:`_batch`) on
           any DB DML that should be atomic with this write.
        2. After the DB commit succeeds, renaming the returned tmp
           path to :attr:`continuity_path` (atomic on POSIX and
           Windows).
        3. On failure anywhere in the pipeline, cleaning up the tmp
           file via :func:`_safe_unlink`.

        Internal use only — outside the library pipeline, call
        :meth:`save_continuity` for the full write+rename+audit
        sequence.

        **Concurrent-writer safety (10.5c.5 L3 Fix #12):** each call
        generates a unique suffix so two concurrent
        ``validated_save_continuity`` invocations cannot race on the
        same tmp path. Orphan tmp files from crashed pipelines are
        discoverable via :meth:`_find_orphan_tmp_files` at store open.

        **Pairing via wrap token (10.5c.5 L3 Fix #19, codex HIGH):**
        the canonical pipeline passes a 12-char slice of the
        current wrap_token as ``token_hex`` so the returned tmp path
        is paired with the meta tmp path written for the same wrap.
        Operators recovering from multiple crashed wraps can pair
        continuity + meta tmps by their shared token prefix. When
        ``token_hex`` is None (legacy ``save_continuity`` standalone
        path), a fresh random uuid is used — standalone writes
        rename immediately and never leave orphans, so pairing is
        unnecessary.

        Args:
            text: Continuity markdown content.
            token_hex: Optional pairing prefix; should be a 12-char
                hex slice of the wrap_token to pair with the
                corresponding meta tmp write.

        Returns:
            Path to the tmp sidecar that was written.

        Raises:
            StoreError: On file-I/O failure. The tmp file is cleaned
                up before raising. ``.operation ==
                "prepare_continuity_write"``, ``.path`` points at the
                final continuity destination (not the tmp sidecar).
        """
        path = self.continuity_path
        tmp_path = self._continuity_tmp_path(token_hex)
        succeeded = False
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            # Sync the parent directory so the tmp file's directory
            # entry is durable before the caller proceeds to the DB
            # commit. Without this, a crash between file fsync and
            # the later rename can leave the tmp file non-existent
            # on recovery even though the data was durable. 10.5c.5
            # L2 H1. Best-effort; POSIX only.
            _fsync_dir(path.parent)
            succeeded = True
        except OSError as exc:
            raise StoreError(
                f"Failed to write continuity tmp sidecar to {tmp_path}: {exc}",
                operation="prepare_continuity_write",
                path=str(path),
            ) from exc
        finally:
            if not succeeded:
                _safe_unlink(tmp_path)
        return tmp_path

    def _prepare_meta_write(
        self, meta: dict, token_hex: str | None = None
    ) -> Path:
        """Write meta sidecar to a tmp file; do not rename.

        Primitive used by the 10.5c.5 two-phase commit pipeline. Same
        semantics as :meth:`_prepare_continuity_write` — passes
        ``token_hex`` through to :meth:`_meta_tmp_path` so the tmp
        file pairs with the corresponding continuity tmp via a
        shared prefix. Internal use only.

        Args:
            meta: Meta sidecar JSON-serializable dict.
            token_hex: Optional pairing prefix; pass a 12-char hex
                slice of the wrap_token to pair with the
                corresponding continuity tmp write.

        Returns:
            Path to the tmp sidecar that was written.

        Raises:
            StoreError: On file-I/O failure. The tmp file is cleaned
                up before raising. ``.operation ==
                "prepare_meta_write"``, ``.path`` points at the final
                meta destination.
        """
        path = self.meta_path
        tmp_path = self._meta_tmp_path(token_hex)
        succeeded = False
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            # Parent dir fsync — see _prepare_continuity_write for
            # rationale. 10.5c.5 L2 H1.
            _fsync_dir(path.parent)
            succeeded = True
        except OSError as exc:
            raise StoreError(
                f"Failed to write meta tmp sidecar to {tmp_path}: {exc}",
                operation="prepare_meta_write",
                path=str(path),
            ) from exc
        finally:
            if not succeeded:
                _safe_unlink(tmp_path)
        return tmp_path

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
