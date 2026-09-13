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
import re
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("anneal-memory")

# Chain anchors
GENESIS_HASH = "sha256:GENESIS"

# Schema version for JSONL entries
_ENTRY_VERSION = 1

# ⛔ ONE FILENAME LANGUAGE, SHARED BY EVERY WRITER AND EVERY READER.
# ``_sealed_filename`` is what rotation writes; ``_is_sealed_filename`` is
# what the manifest parser AND orphan adoption accept. Round 6 used a
# stem-agnostic regex whose first character class excluded ``.``, so a
# database named ``.vault.db`` rotated files the parser then refused, and
# orphan adoption globbed a wider language than the parser accepted — in
# both cases the writer put a name into the manifest that the next read
# rejected, and ``_load_manifest``'s fresh-manifest fallback then WIPED the
# manifest's history on the following rotation (measured 2026-09-13, round
# 7). Binding the check to the exact stem also means a manifest cannot
# reference another database's sealed files (codex round 7).
_SEALED_SUFFIX_PATTERN = r"\.audit\.\d{4}-W\d{2}\.jsonl(?:\.gz)?"

# ⛔ EVERY WAY A LINE OR MANIFEST CAN FAIL TO BE AN ENTRY, IN ONE PLACE.
# ``ValueError`` covers ``JSONDecodeError``, ``UnicodeDecodeError`` and the
# int-string limit (a >4300-digit integer); ``RecursionError`` is deep
# nesting and is NOT a ``ValueError``; ``TypeError`` is a wrong shape from
# the validators. Round 9 measured both of the first two blocking every
# ``log()`` through a readable file, because each site kept its own tuple.
_UNPARSEABLE_JSON: tuple[type[Exception], ...] = (ValueError, RecursionError, TypeError)
# Derived, never re-listed, so a new member of the base reaches every site.
_UNPARSEABLE_OR_IO: tuple[type[Exception], ...] = _UNPARSEABLE_JSON + (OSError,)
_CORRUPT_MANIFEST: tuple[type[Exception], ...] = _UNPARSEABLE_JSON + (KeyError, OSError)

# ``verify()`` re-checks an invalid pass after this long, and keeps polling
# while a rotation is visibly compressing, up to the cap. See ``verify()``.
_ROTATION_POLL_SECONDS = 0.1
_ROTATION_SETTLE_MAX_SECONDS = 5.0
# Appended to the verdicts a concurrent rotation or retention cleanup can
# produce on a healthy trail, so the operator knows a re-run may clear it.
_RERUN_HINT = (
    " — if a rotation or retention cleanup was in progress, re-run verify;"
    " if this persists, the trail is damaged"
)

# Recovery reads each candidate sealed file up to this many times before
# treating a read error as final. See ``AuditTrail._scan_sealed``.
_ADOPTION_READ_ATTEMPTS = 3
_ADOPTION_RETRY_SECONDS = 0.05


class _CorruptAuditFile(OSError):
    """A sealed file whose bytes are corrupt (truncated/invalid gzip) — as
    opposed to a transient read failure, which callers must retry rather
    than treat as permanent."""


def _sealed_filename(stem: str, week: str) -> str:
    """The uncompressed sealed-file name rotation writes (``.gz`` is appended)."""
    return f"{stem}.audit.{week}.jsonl"


def _is_sealed_filename(name: str, stem: str) -> bool:
    """True iff ``name`` is a sealed audit file of the database ``stem``."""
    return re.fullmatch(re.escape(stem) + _SEALED_SUFFIX_PATTERN, name) is not None


def _sealed_period(name: str, stem: str) -> str:
    """The ISO-week label of a sealed filename (``2026-W37``); sorts in time order."""
    return name[len(f"{stem}.audit."):].removesuffix(".gz").removesuffix(".jsonl")


# ⛔ QUARANTINE IS A FILE ON DISK, NOT A FLAG IN MEMORY (hybrid, ruled by Phill
# 2026-09-13). An invalid manifest is renamed to
# ``<stem>.audit.manifest.json.corrupt-<UTC stamp>`` and never overwritten.
# Every later process must see that, including one that finds no manifest at
# all — otherwise the rename would read as "absent" and the next writer would
# rebuild a fresh manifest automatically, which is exactly the history loss the
# hybrid exists to stop. ``anneal-memory audit-repair`` is the only way out; it
# renames markers to ``...corrupt-<stamp>.repaired``, which this no longer matches.
_QUARANTINE_SUFFIX_PATTERN = r"\.corrupt-\d{8}T\d{12}Z"


def _quarantine_markers(audit_dir: Path, stem: str) -> list[str]:
    """Unresolved quarantined manifests for ``stem``, oldest first."""
    pattern = re.escape(f"{stem}.audit.manifest.json") + _QUARANTINE_SUFFIX_PATTERN
    try:
        return sorted(p.name for p in audit_dir.iterdir() if re.fullmatch(pattern, p.name))
    except FileNotFoundError:
        return []


class _ManifestUnavailable(OSError):
    """The manifest cannot be used right now. Writers that need it skip their
    step; appending to the active file does not need it. A plain instance is
    transient (a read error) and callers retry; see ``_ManifestQuarantined``."""


class _ManifestQuarantined(_ManifestUnavailable):
    """The manifest was invalid and is quarantined; only audit-repair clears it."""


def _fsync_dir(path: Path) -> None:
    """Best-effort directory fsync after an atomic rename/replace.

    Same idiom as ``store._fsync_dir`` / ``spores._fsync_dir`` (duplicated
    rather than imported — this module is zero-dependency, including on
    its siblings). ``fsync(file)`` durability does not extend to the
    directory entry created by a subsequent rename; a crash in that gap
    can leave the rename's target missing on recovery even though the
    data was durable. macOS ``fsync`` is weaker than Linux (true
    durability needs ``F_FULLFSYNC``, which stdlib doesn't expose) but
    the directory sync still narrows the window. Windows can't fsync a
    directory handle; no-op there. Swallows ``OSError`` — best-effort,
    not a guarantee callers may rely on.
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
        os.close(dir_fd)


def _parse_manifest_bytes(raw: bytes, stem: str) -> dict[str, Any]:
    """Parse manifest bytes into a mapping, or raise trying.

    codex (L3, 2026-09-13) found two ways a manifest could parse
    "successfully" into something every caller's ``.get()``/subscript
    access then crashes on, past the 2026-09-09 fix that added
    ``UnicodeDecodeError`` to the callers' catch tuples:

    1. ``json.loads(bytes)`` decodes via ``surrogatepass``, which does
       NOT raise on byte sequences that are invalid strict UTF-8 but
       happen to be a valid lone-surrogate encoding — a corrupt manifest
       silently parses into a string field containing ``'\\ud800'``
       instead of raising. Decoding strictly first (``bytes.decode``,
       no ``surrogatepass``) makes an invalid manifest raise
       ``UnicodeDecodeError`` the way the callers already expect.
    2. A syntactically valid JSON document whose root isn't an object
       (``null``, a list, a bare number) parses fine and then blows up
       with ``AttributeError``/``TypeError`` at the first ``.get()`` —
       uncaught by any of the three callers' tuples, which only expect
       parse/decode failures.

    codex (L3, 2026-09-13, second pass) found the same class one level
    deeper: an object root with the WRONG FIELD TYPES parses fine and
    crashes past the root check — ``{"chain_anchor": 1}`` degrades
    ``verify()`` into ``expected_hash[:20]`` on an int (``TypeError``,
    uncaught); ``{"active_last_seq": "7"}`` degrades ``log()``'s
    ``self._seq += 1`` the same way; ``{"files": null}`` crashes
    ``_adopt_orphaned_files()``'s ``manifest.get("files", [])`` iteration.
    Validated here rather than at each of the (growing) call sites, same
    as the root-type check above.

    codex (L3, 2026-09-13, round 3) found two more: ``isinstance(x, int)``
    accepts a JSON boolean (``bool`` is an ``int`` subclass in Python), so
    ``{"active_last_seq": true}`` passed this check, then set ``_seq =
    True`` and wrote ``"seq": true`` into the chain, which ``verify()``
    also accepted as valid. And an empty ``"filename"`` resolves to the
    audit DIRECTORY itself, so ``verify()``'s ``open(fpath, "rb")``
    crashed with an uncaught ``IsADirectoryError`` instead of "Corrupt
    manifest" — closed by requiring a nonempty basename with no path
    separators, not full path-traversal hardening (this manifest is
    written only by this process; the threat model is corruption, not a
    hostile author).

    complement + glm (round 4, independently) found the empty-filename
    fix incomplete: ``"."`` and ``".."`` are nonempty and contain no
    path separator, so both passed the check above and resolve to a
    directory the SAME way ``""`` did (``audit_dir / "."`` is
    ``audit_dir`` itself). codex (round 6) found the general case one
    blacklist entry could never close: ANY basename that happens to
    exist and isn't a regular file (a subdirectory, a FIFO) passes a
    nonempty/no-separator/not-dot check the same way. Replaced the
    growing blacklist with a positive requirement: the filename must
    be a sealed file of THIS database's ``stem`` (``_is_sealed_filename``,
    the same predicate orphan adoption uses, matching what
    ``_sealed_filename`` generates). codex also found a duplicated
    filename entry passed every per-record check and made every reader
    walk that file twice; rejected as a set-uniqueness check below.

    complement (L3, round 3) found a third: this function validated
    ``"files"``'s type ONLY IF THE KEY WAS PRESENT, never requiring it to
    exist — so a manifest that's a valid object but omits ``"files"``
    entirely (a plausible older/migrated shape, given the ``"version"``
    field) passed clean and then crashed the two WRITER call sites
    (``_adopt_orphaned_files``, ``_rotate_if_needed``) at an unguarded
    ``manifest["files"].append(...)`` — the sweep had covered only the
    ``.get("files", [])`` reader sites. Normalized here instead: missing
    ``"files"`` degrades to ``[]``, matching ``_load_manifest()``'s own
    from-scratch default.
    """
    manifest = json.loads(raw.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError(f"manifest root is {type(manifest).__name__}, not an object")
    for key in ("chain_anchor", "active_last_hash"):
        if key in manifest and not isinstance(manifest[key], str):
            raise TypeError(f"manifest field {key!r} is not a string")
    if "active_last_seq" in manifest:
        seq = manifest["active_last_seq"]
        if not isinstance(seq, int) or isinstance(seq, bool):
            raise TypeError("manifest field 'active_last_seq' is not an int")
    files = manifest.get("files", [])
    if not isinstance(files, list) or not all(
        isinstance(f, dict)
        and isinstance(f.get("filename"), str)
        and _is_sealed_filename(f["filename"], stem)
        for f in files
    ):
        raise TypeError("manifest field 'files' is not a list of file records")
    # codex, round 6: a duplicated filename (two records, same sealed
    # file) passed every per-record check above and made every reader
    # walk that file twice — doubled totals in cmd_audit, a duplicated
    # hash-chain segment in verify().
    names = [f["filename"] for f in files]
    if len(names) != len(set(names)):
        raise TypeError("manifest field 'files' contains a duplicate filename")
    if "chain_anchor_recovered" in manifest and not isinstance(
        manifest["chain_anchor_recovered"], bool
    ):
        raise TypeError("manifest field 'chain_anchor_recovered' is not a boolean")
    manifest["files"] = files
    return manifest


def _require_entry_dict(entry: Any) -> dict[str, Any]:
    """An audit-entry JSONL line that parses to anything but an object
    (a bare list, string, or number) is exactly as corrupt as one that
    fails to parse at all — every one of this file's four entry-line
    readers immediately calls ``.get()`` on the result, uncaught by any
    of them (same class as ``_parse_manifest_bytes``'s root check,
    swept to every entry-line call site: complement/codex L3,
    2026-09-13, round 3).

    codex (L3, round 3) found the same class one level deeper: an
    object-root entry with the WRONG FIELD TYPES parses fine and crashes
    past the root check — ``{"prev_hash": 1}`` degrades ``verify()``
    into ``actual_prev[:20]`` on an int; a string ``seq`` degrades
    ``_initialize()``'s ``last_entry.get("seq", 0) + 1``; a numeric
    ``ts`` degrades the same method's ``ts.replace("Z", "+00:00")``.

    codex (L3, round 4) found two more, in ``cli.py``'s TEXT-rendering
    path (below the JSON return, so missed by every earlier test that
    only checked ``--json``): ``data`` non-dict crashes
    ``data.get('episode_id', ...)``; ``event`` non-str crashes the
    ``f"{event:<24}"`` format spec. Every field this file (and its
    callers) dereferences in a type-unsafe way is now covered here in
    one place — ``actor`` is only ever compared for equality or printed
    bare, safe for any type.
    """
    if not isinstance(entry, dict):
        raise TypeError(f"entry line root is {type(entry).__name__}, not an object")
    if "prev_hash" in entry and not isinstance(entry["prev_hash"], str):
        raise TypeError("entry field 'prev_hash' is not a string")
    if "seq" in entry:
        seq = entry["seq"]
        if not isinstance(seq, int) or isinstance(seq, bool):
            raise TypeError("entry field 'seq' is not an int")
    if "ts" in entry and not isinstance(entry["ts"], str):
        raise TypeError("entry field 'ts' is not a string")
    if "event" in entry and not isinstance(entry["event"], str):
        raise TypeError("entry field 'event' is not a string")
    if "data" in entry and not isinstance(entry["data"], dict):
        raise TypeError("entry field 'data' is not an object")
    return entry


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
    # False when the chain's starting anchor was RECOVERED by audit-repair
    # rather than recorded by retention cleanup. ``valid`` then speaks only for
    # the linked chain from that anchor on: whatever preceded it (entries
    # removed by retention, or a truncated front) cannot be verified. Ruled by
    # Phill 2026-09-13: reported alongside ``valid``, never folded into it.
    anchor_trusted: bool = True


@dataclass
class AuditRepairResult:
    """Result of :meth:`AuditTrail.repair_manifest`."""

    repaired: bool
    files: list[str] = field(default_factory=list)
    chain_anchor_recovered: bool = False
    # Sealed files left on disk that the rebuilt manifest does not list (a
    # duplicate copy of a week already covered). Nothing is deleted; verify()
    # reports them until the operator removes them.
    untracked: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class _SealedScan:
    """One full read of a candidate sealed file, for orphan adoption."""

    entries: int = 0
    first_ts: str = ""
    last_ts: str = ""
    last_hash: str = ""
    first_prev_hash: str | None = None  # prev_hash of the first valid entry
    digest: str = ""  # sha256 of the uncompressed bytes
    error: OSError | None = None  # the read error that ended the last attempt


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
        # A refusal is logged once, and a successful rotation re-arms it, so a
        # later refusal in a long-lived process is logged again (L1 re-pass of
        # round 10b, LOW).
        self._rotation_refusal_logged = False
        # Writes a caller swallowed since the last entry that landed. Rides
        # into the next successful entry as ``dropped_before`` so a gap becomes
        # a chained fact rather than an absence — see :meth:`note_write_failure`.
        # ⛔ PROCESS-LOCAL, AND THE WINDOW IS MUCH WIDER THAN "A CRASH" —
        # this comment has now been wrong twice in one day, each time in the
        # direction of overclaiming. The count reaches the durable chain only
        # when a LATER write lands ON THIS INSTANCE. Any ``Store`` close and
        # reopen resets it: ``__init__`` sets it to 0, ``_initialize()``
        # recovers ``seq``/``prev_hash`` from disk, and the next event chains
        # cleanly over the hole with no ``dropped_before`` marker.
        #
        # MEASURED 2026-09-04 (codex L3 HIGH): a committed mutation whose audit
        # write was dropped, followed by an ORDINARY ``close()`` + reopen — no
        # crash — leaves 3 episodes against 2 audit entries, no marker, and
        # ``verify()`` returning ``valid=True``. **Every CLI invocation opens
        # and closes a Store**, so across CLI commands this mechanism
        # effectively never fires. It closes the swallow-then-keep-going case
        # within one live process, and nothing wider.
        #
        # ⚠ CORRECTION, 2026-09-04 (same day, later): that measurement also
        # said ``status()`` reported ``audit_write_failures = 0`` after the
        # reopen. THAT HALF IS NO LONGER TRUE and the sentence above has been
        # amended rather than left to rot — ``Store._audit_write_failures`` is
        # now persisted in the SQLite metadata table and seeded at open, so a
        # reopened store reports the lifetime count. Stated narrowly, because
        # this comment has been wrong twice already in the overclaiming
        # direction: what became durable is the COUNT. This attribute — the
        # ride-along ``dropped_before`` MARKER — did NOT, and everything else
        # above stands. The two are different promises: the count says "N
        # writes were lost", the marker says "the gap is HERE in the chain",
        # and only the second has to survive into a hash-chained entry.
        #
        # Flagged independently by BOTH L3 seats, 2026-09-04. Making THE
        # MARKER durable is still open: it is a write issued from inside a
        # post-commit exception handler and needs its own failure discipline,
        # which is a design question, not a patch. Tracked for the next
        # anneal touch; the honest thing meanwhile is that this comment says
        # so rather than the docstring implying a guarantee it does not have.
        self._dropped_since_last: int = 0

    # -- Public API --

    def log(
        self,
        event: str,
        data: dict[str, Any] | None = None,
        actor: str = "agent",
    ) -> dict[str, Any]:
        """Append a hash-chained entry to the audit trail.

        Args:
            event: Event type. Must be a ``str`` — a non-str ``event`` or a
                   non-dict ``data`` raises ``TypeError`` before anything is
                   written (the readers' own entry validator, so a written
                   entry is always one recovery accepts). The VOCABULARY is
                   not enforced, but it is closed. Keep this list in
                   sync when adding a raise site — it is the only inventory
                   of event types that exists.

                   Episode/store lifecycle:
                     record, delete, prune
                   Wrap lifecycle:
                     wrap_started, wrap_cancelled, wrap_completed
                   Continuity:
                     continuity_saved, section_schema_set
                   Hebbian (episode-level) associations:
                     associations_updated, associations_decayed
                   Cortical (pattern-level) association graph:
                     pattern_associations_seeded, pattern_co_surface_drained,
                     pattern_associations_gc, pattern_association_renamed,
                     pattern_concept_severed
            data: Event-specific payload.
            actor: Identity of the actor triggering this event.
                   EU AI Act Article 12(2) requires actor identity on
                   all audit entries. Default "agent" for single-agent;
                   multi-agent passes agent name/ID.

        Returns:
            The complete entry dict that was written.
        """
        # Validate the caller's fields FIRST (codex, round 8): initialization
        # and rotation below rename, compress and save files, so a type
        # check that only ran on the finished entry let a refused call
        # mutate the trail before raising. The check on the full entry
        # further down stays as the writer/reader invariant.
        probe: dict[str, Any] = {"event": event}
        if data is not None:
            probe["data"] = data
        _require_entry_dict(probe)

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
        # ⛔ WRITER AND READER MUST AGREE ON WHAT A VALID ENTRY IS (codex,
        # round 6): every reader in this file now rejects an entry whose
        # ``event``/``data``/``seq``/``ts``/``prev_hash`` has the wrong
        # type, via ``_require_entry_dict``. Before this call, ``log()``
        # itself enforced nothing at runtime (``event: str`` was a type
        # hint, not a guard) — a caller passing e.g. a non-str ``event``
        # wrote a record that recovery then treats as NOT A VALID ENTRY,
        # silently resetting the chain to genesis and reusing ``seq``.
        # Calling the SAME validator here, before the write, means writer
        # and reader cannot disagree about what "valid" means by
        # construction, not by keeping two checks in sync by hand.
        _require_entry_dict(entry)
        # ⛔ A DROPPED ENTRY MUST BE A FACT IN THE CHAIN, NOT AN ABSENCE.
        # This class is write-first: chain state is updated only after fsync
        # returns, so a failed write leaves ``_prev_hash``/``_seq`` untouched
        # and the NEXT entry chains cleanly over the hole. MEASURED
        # 2026-09-04: 8 store mutations with 7 sink failures produced ONE
        # entry and ``verify()`` returned ``valid=True, chain_break_at=None``.
        # The gap was not merely undetectable-as-tampering — it was
        # indistinguishable from the mutations never happening, while the
        # verifier reported a clean bill of health over it.
        #
        # Callers that swallow a write failure (see
        # ``Store._audit_log_after_commit``, which must not fail an operation
        # that already committed) record it via :meth:`note_write_failure`;
        # the count then rides into the next entry that DOES land, so it is
        # hash-chained and tamper-evident like everything else here. Emitted
        # only when non-zero, so a healthy trail is byte-identical to before.
        if self._dropped_since_last:
            entry["dropped_before"] = self._dropped_since_last

        # Deterministic serialization — sorted keys, compact separators
        json_line = json.dumps(entry, sort_keys=True, separators=(",", ":"))

        # Write-first: fsync BEFORE updating internal state.
        #
        # ⛔ AND THE APPEND MUST BE ALL-OR-NOTHING, because "write-first" alone
        # leaves a THIRD state between written and not-written. If ``write``
        # and ``flush`` succeed and ``fsync`` (or close) then raises EIO, the
        # complete line is ALREADY VISIBLE on disk while ``_seq``,
        # ``_prev_hash`` and ``_dropped_since_last`` are all unchanged. The
        # caller — which by contract swallows and calls
        # :meth:`note_write_failure` — then retries, and the retry emits the
        # SAME ``seq`` and the SAME ``prev_hash``.
        #
        # MEASURED 2026-09-04 (codex L3 HIGH, reproduced by failing fsync after
        # a successful write): seqs on disk ``[0, 1, 1]``, ``dropped_before``
        # ``[None, 2, 3]`` — the two pending drops counted TWICE and the entry
        # that actually landed counted as dropped — and ``verify()`` returning
        # ``valid=False`` with a hash mismatch. **A durability hiccup read as
        # tampering**, on the record whose entire value is telling those apart.
        # A partial write (ENOSPC mid-line) concatenates with the retry into
        # malformed JSON and breaks the chain the same way.
        #
        # So: remember the pre-append size and roll the file back to it if the
        # append does not fully complete. Disk is then made to agree with the
        # in-memory state — nothing landed — which is what makes the caller's
        # retry sound. Safe because this class is single-writer by contract
        # (see the module docstring); truncating a file a peer was appending to
        # would not be.
        active = self._active_path
        active.parent.mkdir(parents=True, exist_ok=True)
        resume_at = active.stat().st_size if active.exists() else 0
        # ⛔ THE APPEND MUST START AT A LINE BOUNDARY, AND UNTIL 2026-09-07
        # NOTHING MADE IT. Every write here is ``json_line + "\n"``, so a
        # file NOT ending in a newline is ALWAYS an incomplete write — there
        # is no legitimate reading of that state. Opening in ``"a"`` and
        # writing anyway CONCATENATES this entry onto the fragment, and the
        # merged line parses as nothing: THIS ENTRY IS DESTROYED, not the
        # torn one.
        #
        # ⚠ AND IT IS DESTROYED SILENTLY, WHICH IS THE WHOLE COST. MEASURED
        # 2026-09-07 on the re-opening-process shape — the general case on
        # the surface the README points operators at, because
        # ``_dropped_since_last``'s comment in ``__init__`` records that
        # EVERY CLI INVOCATION OPENS AND CLOSES A STORE (anchored on that
        # comment, not on a line number: this file moves every time it is
        # touched, and it was touched three times today) — four events
        # logged, the file holding
        # ``['first', 'second', 'MALFORMED(233B)', 'fourth']`` and
        # ``verify()`` returning ``valid=True, skipped_lines=1``. The third
        # event is simply gone. From the writer's side the append SUCCEEDED,
        # so ``note_write_failure`` never fires, ``dropped_before`` is never
        # emitted and ``audit_write_failures`` stays 0 — none of the
        # loss-reporting machinery has anything to report.
        #
        # ⚖ THE REPAIR IS ADDITIVE, AND THAT IS A DELIBERATE REJECTION OF
        # THE FIX THIS WAS FILED WITH. The recorded candidate was a
        # recovery-time TRUNCATION — delete the torn bytes at open — which
        # is what made it "not a thing to land unreviewed". It is also not
        # necessary: the damage comes from the CONCATENATION, not from the
        # fragment existing. Terminating the fragment costs one byte,
        # DELETES NOTHING, and leaves the torn bytes on disk as a skipped
        # line an operator can still read. On a tamper-evident log, a repair
        # that never removes bytes is the strictly better primitive — and
        # deleting at open is an operation this module should not own at
        # all. MEASURED, same fixture: additive gives
        # ``['first', 'second', 'MALFORMED(60B)', 'third_real', 'fourth']``
        # — the entry survives AND the evidence survives.
        #
        # ⛔ IT LIVES HERE AND NOT IN ``_initialize`` FOR A REASON THAT IS
        # NOT STYLE: ``log()`` calls ``_initialize()`` only when
        # ``_initialized`` is False, so a LONG-LIVED process that tore its
        # own tail mid-run never re-initialises and an init-time repair
        # never fires for it. That is L2's loud ``[0,1,MALFORMED,3]`` shape.
        # The append is the operation that does the damage, so the guard
        # belongs on the append — where it covers both shapes. It also then
        # sits INSIDE the existing all-or-nothing rollback: ``resume_at`` is
        # taken above, so a failed append rolls the boundary byte back too.
        needs_boundary = False
        if resume_at:
            with open(active, "rb") as f_probe:
                f_probe.seek(-1, os.SEEK_END)
                needs_boundary = f_probe.read(1) != b"\n"
        # ``_compute_hash`` is a staticmethod, pure in ``json_line``, so
        # hoisting it OUT of the guarded region below is side-effect-free
        # and leaves that region holding only the file write and the three
        # stores — ``open`` / ``write`` / ``flush`` / ``fsync``, none of
        # which can reach ``self``. (It did NOT leave "nothing but the three
        # stores", as this line said until L1 read it on 2026-09-07; the
        # allow-list in the AST test is the authority on that set.)
        # It also means a hash failure now raises BEFORE anything is
        # written, instead of after — no line on disk, nothing to roll back.
        new_prev_hash = self._compute_hash(json_line)
        # Built out here so the guarded region below still holds nothing but
        # ``open`` / ``write`` / ``flush`` / ``fsync`` and the three stores.
        payload = ("\n" if needs_boundary else "") + json_line + "\n"
        # Snapshot for the rollback. See the handler for why rolling the
        # FILE back is only half of it.
        saved_chain_state = (
            self._prev_hash,
            self._seq,
            self._dropped_since_last,
        )
        try:
            with open(active, "a", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            # Update chain state
            self._prev_hash = new_prev_hash
            self._seq += 1
            # Cleared only now — after fsync — so a failure while writing THIS
            # entry keeps the pending count for the next attempt rather than
            # losing the very fact it exists to preserve.
            self._dropped_since_last = 0
        except BaseException:
            # ⛔ ROLLING THE FILE BACK IS ONLY HALF OF IT — the in-memory
            # chain state is restored too, at the BOTTOM of this handler.
            # The advance in the guarded block above is THREE SEPARATE
            # STORES, so an interrupt between any two of them leaves memory
            # ahead of disk; truncating without restoring then chains the
            # NEXT entry over a hole — the same false-tampering signature
            # this handler exists to prevent, reached from the other side.
            #
            # ⚖ AND THE TRUNCATE GOES FIRST, WHICH IS THE OPPOSITE OF THE
            # ORDER THIS HANDLER SHIPPED WITH FOR AN HOUR ON 2026-09-07.
            # The original argument was "restore first, it cannot raise."
            # That optimises the wrong thing.
            # ⭐ THE REASON THAT ACTUALLY GENERALISES (L2, 2026-09-07, after
            # it had argued the other side and withdrew): **THE TRUNCATE'S
            # EFFECT OUTLIVES THE PROCESS AND THE RESTORE'S DIES WITH IT.**
            # The file is the only durable state, so the durable operation
            # goes first — after it, every subsequent partial failure leaves
            # a file ``_initialize`` can re-derive from correctly. Restore
            # first inverts that: a signal mid-restore escapes with the
            # entry STILL on disk and memory half-restored, which is the
            # compounding case. Measured, original failure an ordinary
            # ``OSError`` (ENOSPC) plus ONE ``KeyboardInterrupt`` during the
            # restore — the realistic case, not two signals:
            #
            #   signal during restore of | restore first | truncate first
            #   -------------------------|---------------|---------------
            #   _prev_hash               | seqs [0,1,2,2]| seqs [0,1,2]
            #                            | valid=FALSE   | valid=True
            #   _seq                     | valid=FALSE   | valid=True
            #   _dropped_since_last      | valid=FALSE   | valid=True
            #
            # Found by codex at L3 against the restore-first version, which
            # had shipped with a comment claiming the residual needed a
            # SECOND terminal signal. It does not: one is enough when the
            # original failure is ordinary I/O. Pinned by
            # ``test_the_rollback_truncates_before_it_restores``.
            #
            # MEASURED 2026-09-07 — four interrupt points x three
            # structural variants, graded by the tests named at the bottom
            # of this comment, each variant a full copy of the tree, each
            # arm run as interrupt -> ``note_write_failure()`` -> retry ->
            # read the seqs off disk -> ``verify()``:
            #
            #   interrupt before    | advance   | advance in | advance in
            #                       | OUTSIDE   | try, NO    | try, WITH
            #                       | try (was) | restore    | restore
            #   --------------------|-----------|------------|-----------
            #   os.fsync            |   pass    |   pass     |   pass
            #   _prev_hash store    |   FAIL    |   pass     |   pass
            #   _seq store          |   FAIL    |   FAIL     |   pass
            #   _dropped_since_last |   pass    |   FAIL     |   pass
            #
            # ⚠ THE TWO FAILING COLUMNS FAIL DIFFERENTLY, which is why one
            # assertion would not have found both: OUTSIDE-the-``try`` fails
            # on DUPLICATE SEQS on disk (``[0, 1, 2, 2]``) because nothing
            # rolls back, while inside-without-restore fails on
            # ``verify(): valid=False`` because the file rolled back and
            # memory did not.
            #
            # ⛔ AND THAT SECOND COLUMN'S TWO ARMS DO NOT LOOK ALIKE ON
            # DISK — an earlier version of this comment gave one seq list
            # for both, which is wrong and wrong in the reassuring
            # direction:
            #     ``_seq`` arm ............ seqs ``[0, 1, 2]``, mismatch at 2
            #     ``_dropped`` arm ........ seqs ``[0, 1, 3]``, mismatch at 3
            # The ``_seq`` arm is the ALARMING one: the seqs are
            # CONTIGUOUS, there is no gap to notice, and ``verify()`` still
            # reports tampering. An operator handed only the ``[0, 1, 3]``
            # signature goes looking for a hole in the numbering and finds
            # none. (Caught by L1, 2026-09-07, against a block labelled
            # "transcribed from the run".)
            #
            # ⛔ SO DO NOT "SIMPLIFY" THIS BY DROPPING THE RESTORE. Moving
            # the advance inside the ``try`` without it does not take the
            # broken windows from two to zero — it takes them from two to
            # two, and moves which ones. That middle column is the fix
            # exactly as it was first prescribed on 09-07; it was measured
            # rather than adopted. Only the third column is green.
            #
            # ⚠ WHAT IS STILL NOT CLOSED, AND THE SIGNAL COUNT HERE WAS
            # WRONG FOR AN HOUR — this paragraph first said truncate-first
            # "needs TWO terminal signals", tagged MEASURED. It is false in
            # the ordinary-entry regime, and the tag was unearned: the table
            # above and the ordering test both interrupt only at the three
            # RESTORE stores, which is the one place truncate-first wins.
            # Neither ever interrupted the truncate. Caught by glm-5.3 at
            # L3 (2026-09-07) and then measured — ordinary ``OSError``
            # entry plus ONE ``KeyboardInterrupt``:
            #
            #   signal lands at        | result
            #   -----------------------|--------------------------------
            #   any of the 3 restores  | seqs [0,1,2]   valid=True
            #   the truncate's open()  | seqs [0,1,2,2] valid=False
            #   the truncate's fsync() | seqs [0,1,2]   valid=True
            #                          | (``truncate()`` already took
            #                          |  effect by then)
            #
            # ⚖ SO, STATED HONESTLY: in the ordinary-entry regime BOTH
            # orderings need exactly ONE terminal signal. Truncate-first
            # does not raise the count — it NARROWS WHERE the signal has to
            # land, from "anywhere in the handler" to "inside the truncate,
            # before it takes effect". That is still strictly better and is
            # why the order stands, but it is a smaller claim than the one
            # this comment made. TWO signals are needed only when the
            # exception that ENTERED the handler was itself terminal.
            # The residual is pinned by an xfail arm on
            # ``test_the_rollback_truncates_before_it_restores``; when the
            # structural close lands, that arm starts passing and says so.
            # Collapsing the three attributes into ONE would make each
            # half a single store and close the first of those. ⚖ SCOPE,
            # MEASURED 2026-09-07 rather than estimated: the three are
            # PRIVATE TO THIS MODULE — ``store.py``'s only mention of any
            # of them is a comment, so there is no cross-module contract to
            # renegotiate. Count the sites with
            #     grep -c 'self\._prev_hash\|self\._seq\b\|self\._dropped_since_last' anneal_memory/audit.py
            # It is cheaper than it looks and still NOT done here: it is a
            # separate change with its own review, and it closes only the
            # second-signal-during-the-handler window, not anything a first
            # signal can reach.
            #
            # Graded by ``TestTheAppendIsAllOrNothingForTerminalExceptions
            # Too`` in ``tests/test_audit.py``, whose two mutants are the
            # re-derivation recipe for the table above.
            # ⛔ ``BaseException``, NOT ``Exception`` (codex L3 HIGH,
            # 2026-09-06). The all-or-nothing property this block exists to
            # provide did not hold for terminal exceptions: a
            # ``KeyboardInterrupt`` landing after the line was written and
            # fsynced but before the chain state advanced left the entry ON
            # DISK with ``_seq``/``_prev_hash`` unchanged and NO rollback,
            # because it walked past this handler. ⚠ AND WIDENING THIS
            # HANDLER WAS NOT SUFFICIENT, WHICH THIS COMMENT ASSERTED FOR A
            # DAY (2026-09-06 -> 09-07): the advance it names sat OUTSIDE
            # the ``try`` — thirty-three lines past the end of the guarded
            # body, one whole handler in between — so no widening could
            # reach it. The advance was moved inside on 09-07, and that is
            # what makes this handler's promise true rather than stated.
            # ⚠ IT BECAME REACHABLE THE SAME DAY. Until the store's own
            # per-event catch was widened to ``BaseException``, an interrupt
            # here abandoned the whole replay, so no retry followed and the
            # inconsistency died with the run. Once the store started
            # RECORDING the drop and CONTINUING, the next append reused the
            # same ``seq`` and ``prev_hash``. REPRODUCED 2026-09-06: seqs on
            # disk ``[0, 1, 1]`` and ``verify()`` reporting a hash mismatch —
            # the identical signature as the 2026-09-04 fsync-EIO HIGH this
            # rollback was written for, reached through the other exception
            # branch. A durability hiccup read as tampering, on the record
            # whose entire value is telling those apart.
            # Best-effort rollback. If THIS fails too we do not mask the
            # original failure with a rollback failure — the exception below
            # still surfaces it.
            # ⛔ A FAILED TRUNCATE IS NOT A NEUTRAL UNKNOWN, AND THIS LINE
            # SAID "THE AMBIGUITY STANDS" UNTIL 2026-09-07. The entry stays
            # on disk while the restore rewinds memory to "nothing landed",
            # so the caller's retry reuses the seq. MEASURED (L2,
            # 2026-09-07) with NO terminal signal at all — an ``fsync``
            # reporting EIO after the data landed, then the rollback's
            # ``open`` failing EROFS on the same sick disk: seqs
            # ``[0, 1, 2, 2]``, ``verify(): valid=False``. A FALSE TAMPERING
            # VERDICT, in the dangerous direction, reachable with ordinary
            # exceptions only.
            #
            # ⚖ SO THE RESTORE IS CONDITIONAL — AND WHEN THE TRUNCATE FAILS
            # THIS HANDLER STOPS GUESSING WHAT DISK LOOKS LIKE AND ASKS IT.
            # The fix first recorded here was "leave memory ADVANCED, the
            # entry is still on disk". ⛔ THAT IS WRONG, AND WRONG IN THE
            # DANGEROUS DIRECTION, BECAUSE IT ASSUMES THE LINE ON DISK IS
            # COMPLETE. It is not, whenever the failure was an ENOSPC in the
            # middle of ``write`` — advancing then sets ``_prev_hash`` to the
            # hash of the line we MEANT to write while disk holds a fragment
            # that hashes to nothing, and every later entry chains from a
            # line that does not exist.
            #
            # ⭐ INVALIDATING THE CACHE IS CORRECT IN EVERY BRANCH BECAUSE IT
            # ASSERTS NOTHING (L2's answer, 2026-09-07, verified here):
            # ``log()`` re-runs ``_initialize()`` when ``_initialized`` is
            # False, and ``_initialize`` re-derives ``seq``/``prev_hash``
            # from the FILE. Complete line on disk -> it chains from that
            # line. Fragment on disk -> ``_read_last_valid_entry`` skips it
            # and chains from the last good entry, and the boundary guard at
            # the top of this method keeps the retry from merging into it.
            # ``truncate()`` took effect but its ``fsync`` raised -> it
            # re-derives from the truncated file, which is what the restore
            # would have produced anyway. No branch needs to be identified,
            # which is why this needs no condition beyond "did the truncate
            # succeed".
            #
            # ⛔ ORDER WAS FORCED AND IS NOW PAID: re-deriving from the file
            # is only correct if recovery is correct, and until the boundary
            # guard landed above, recovery handed the next append straight
            # into the torn-tail defect. These were filed as two HIGHs on
            # 2026-09-07; they are ONE change, and the cheap-looking half is
            # downstream of the other.
            # ⛔ INVALIDATE FIRST, RE-VALIDATE ONLY ON A COMPLETE RESTORE.
            # This closes the residual that stood as a strict-xfail from
            # 2026-09-07 morning: a terminal signal landing inside the
            # rollback's ``open`` walks past the ``except Exception`` below,
            # so NEITHER branch ran and memory stayed behind disk with
            # ``_initialized`` still true — the retry then reused the seq.
            # Clearing it up here means every exceptional or terminal exit
            # from this handler leaves DISK as the authority, which is the
            # property the conditional restore already relies on.
            # ⚖ THE SHAPE IS CODEX'S (L3 round 2, 2026-09-07) AND IT IS
            # CHEAPER THAN BOTH RECORDED ALTERNATIVES. This was HELD on the
            # argument that closing the window meant deleting the restore
            # entirely, retiring two mutation-graded gates and orphaning
            # ``_dropped_since_last``. **That cost was real for that fix and
            # is not a property of the problem.** MEASURED with this shape:
            # the residual closes (seqs ``[0,1,2,3]``, valid=True) AND the
            # reversed-restore mutant is still killed AND the drop count is
            # still restored on both paths. Nothing was retired.
            self._initialized = False
            truncated = True
            try:
                with open(active, "r+b") as f_trunc:
                    f_trunc.truncate(resume_at)
                    f_trunc.flush()
                    os.fsync(f_trunc.fileno())
            except Exception:
                # ⛔ NOT A BARE ``pass``. This module logs the far more
                # benign ``on_event`` and orphan-adoption failures, and this
                # branch means the file could not be rolled back. Logged,
                # not raised: raising here would mask the original failure
                # with the rollback's.
                # ⛔ THE SAFE STATE IS ESTABLISHED BEFORE THE LOG CALL, NOT
                # AFTER. Logging handlers are application callbacks and can
                # raise; with the invalidation below the warning, a handler
                # that raised skipped it entirely and the next append reused
                # the seq — a false tampering verdict caused by a logging
                # config. codex (L3, 2026-09-07).
                truncated = False
                self._initialized = False
                # ⛔ A LOGGING HANDLER MUST NOT REPLACE THE DISK FAILURE.
                # Handlers are application callbacks; one that raises here
                # means the caller receives ITS exception instead of the
                # original ``OSError`` and never reaches the bare ``raise``
                # below — the disk fault is swapped for a logging fault on
                # the way out. codex (L3 round 2, 2026-09-07).
                # ⚠ "MAY still be on disk", not "is": the truncate can fail
                # AFTER ``truncate()`` took effect (its ``fsync`` raising),
                # and the original ``open`` can fail before anything was
                # written at all. This message said "is" and was wrong in
                # both. What is certain is the part the operator needs:
                # state comes from the file now.
                try:
                    logger.warning(
                        "audit rollback failed for seq %d; the aborted entry "
                        "may still be on disk, so the chain state has NOT "
                        "been rewound — the next append re-derives "
                        "seq/prev_hash from the file rather than reusing "
                        "this seq",
                        entry["seq"],
                        exc_info=True,
                    )
                except Exception:
                    pass
            # ⛔ THE ORDER OF THESE THREE IS LOAD-BEARING — ``_prev_hash``
            # BEFORE ``_seq``. It was accidental until 2026-09-07, when L2
            # asked and the measurement answered. A signal landing BETWEEN
            # them (so: one signal mid-advance to get here with memory
            # partly advanced, one mid-restore) leaves:
            #   this order  — ``_prev_hash`` back at hash(E1), ``_seq``
            #                 still advanced. The next entry chains
            #                 CORRECTLY and merely skips a seq number, and
            #                 ``verify()`` checks linkage, not seq
            #                 monotonicity. MEASURED: valid=True.
            #   reversed    — ``_prev_hash`` still pointing at the entry
            #                 just truncated away. The next entry chains
            #                 from a line that is no longer on disk.
            #                 MEASURED: valid=False, "Hash mismatch at
            #                 seq 2" — and the seqs are CONTIGUOUS, so
            #                 there is no gap to notice.
            # Pinned by ``test_the_restore_puts_prev_hash_before_seq``.
            if truncated:
                (
                    self._prev_hash,
                    self._seq,
                    self._dropped_since_last,
                ) = saved_chain_state
                self._initialized = True
            else:
                # Disk is the authority now — see the block above.
                # (``_initialized`` was already cleared in the except above,
                # before the log call that can raise. This branch owns only
                # the drop count.)
                # ⚠ THE DROP COUNT IS RESTORED IN BOTH BRANCHES, AND IN THIS
                # ONE IT MAY OVER-REPORT. If the line landed COMPLETE it
                # already carries ``dropped_before``, so the next entry
                # carries the same count a second time. That is deliberate:
                # this counter's whole job is to say "writes were lost
                # here", and double-counting a loss is the safe direction
                # where under-counting is the failure the counter exists to
                # prevent. ``_initialize`` does not touch this attribute, so
                # it survives the re-derivation.
                self._dropped_since_last = saved_chain_state[2]
            raise

        # Fire callback after successful write
        if self._on_event is not None:
            try:
                self._on_event(entry)
            except Exception:
                logger.warning("on_event callback failed for seq %d", entry["seq"], exc_info=True)

        return entry

    def note_write_failure(self) -> int | None:
        """Record that a caller swallowed a failed audit write.

        The count rides into the next entry that lands, as
        ``dropped_before``. See the comment in :meth:`log` — without this a
        swallowed write is invisible to :meth:`verify`, which walks a
        continuous chain over the hole and reports ``valid=True``.

        Deliberately cannot raise: it is called from inside an exception
        handler whose whole contract is that nothing after a commit
        propagates.

        ⚠ The CHAINED record is durable only once a later entry lands — see
        the ``_dropped_since_last`` comment in ``__init__``. A close or crash
        before that loses the marker, and ``verify()`` reports ``valid=True``
        over the gap.

        ▶ WHICH IS WHY THIS RETURNS THE LOCATION. The seq the missing entry
        would have carried is known right here and is otherwise thrown away.
        Handing it back lets ``Store`` fold it into the DURABLE
        ``audit_last_failure`` record, so the *where* survives the process even
        when the chained marker does not. That is not a substitute for the
        marker — it is not hash-chained, so it proves nothing against an
        attacker — but it is the difference between an operator knowing "this
        store lost 2 writes" and knowing "the gaps are at seq 3 and seq 7".

        Returns:
            The seq the dropped entry would have had, or ``None`` if even that
            could not be determined. Deliberately cannot raise.
        """
        try:
            # ⛔ ``None`` WHEN THE CACHE HAS BEEN INVALIDATED. A failed
            # rollback deliberately leaves ``_seq`` stale and defers the
            # truth to the next ``_initialize()``, so reading it here
            # persists a location that is knowingly wrong — the durable
            # ``audit_last_failure`` record would point an operator at an
            # entry that exists. Flagged by BOTH L3 seats, 2026-09-07.
            # The docstring already promises ``None`` means "could not be
            # determined", which is exactly the case; re-deriving instead
            # would mean doing disk I/O in a method contracted never to
            # raise, on the disk that just failed.
            missing_seq = self._seq if self._initialized else None
        except Exception:  # pragma: no cover — defensive, see docstring
            missing_seq = None
        try:
            self._dropped_since_last += 1
        except Exception:  # pragma: no cover — defensive, see docstring
            pass
        return missing_seq

    def stats(self) -> dict[str, Any]:
        """Return a cheap health snapshot of the audit trail.

        Lazy-initializes the trail if it hasn't been touched yet so
        ``entry_count`` reflects the true count on disk (including any
        entries recovered from a prior active file). Does NOT walk the
        full hash chain — for integrity verification, call :meth:`verify`.

        Returns:
            Dict with keys ``log_path`` (str), ``entry_count`` (int),
            ``retention_days`` (int | None). Callers that also care
            about the enabled/disabled distinction should check for
            ``None`` at the ``Store._audit`` level before calling this.
        """
        if not self._initialized:
            self._initialize()
        return {
            "log_path": str(self._active_path),
            "entry_count": self._seq,
            "retention_days": self._retention_days,
        }

    @classmethod
    def verify(cls, db_path: str | Path) -> AuditVerifyResult:
        """Verify hash chain integrity across all audit files.

        Walks sealed files (from manifest) then the active file,
        checking that each entry's prev_hash matches the computed
        hash of the previous entry.

        ⛔ AN INVALID PASS IS RE-CHECKED BEFORE IT IS RETURNED (complement,
        round 10, reproduced). Nothing serialises this classmethod against a
        writer in another process, and a healthy rotation passes through
        states that read as broken, and one that read as valid over a week
        the pass never saw (L2, round 10, reproduced). So an invalid pass is
        repeated after ``_ROTATION_POLL_SECONDS``, and again for as long as
        the directory shows a rotation in flight (see ``_verify_once``), up
        to ``_ROTATION_SETTLE_MAX_SECONDS`` after the first pass. A result is
        returned invalid only after two consecutive failing passes with no
        rotation visible, or at the cap. A trail that stays broken fails
        every pass, so this delays that verdict; it does not change it. It
        is not a lock: a retention cleanup in another process leaves no
        marker, so a cleanup slower than one poll interval can still be
        reported invalid, with a hint to re-run.

        Args:
            db_path: Path to the SQLite database (audit files derive from this).

        Returns:
            AuditVerifyResult with chain validity and diagnostics.
        """
        db_path = Path(db_path)
        result, compressing = cls._verify_once(db_path)
        if result.valid:
            return result
        deadline = time.monotonic() + _ROTATION_SETTLE_MAX_SECONDS
        was_compressing = compressing
        while True:
            time.sleep(_ROTATION_POLL_SECONDS)
            result, compressing = cls._verify_once(db_path)
            if result.valid:
                return result
            if not (compressing or was_compressing) or time.monotonic() >= deadline:
                return result
            was_compressing = compressing

    @classmethod
    def _verify_once(cls, db_path: Path) -> tuple[AuditVerifyResult, bool]:
        """One pass over the audit files as they are now, and whether a
        rotation was in flight at the listing (``_rotation_in_flight``).

        ⛔ THE DIRECTORY IS LISTED ONCE, FIRST, AND THE MANIFEST, ACTIVE AND
        UNMANIFESTED CHECKS BELOW ARE MEMBERSHIP TESTS ON THAT LISTING (codex
        #5, round 10, reproduced as a traceback). A manifested file's own
        check stays ``is_file()``, which reports a permission fault on that
        file as missing. ``Path.exists()`` raised
        ``PermissionError`` on a directory without search permission, and
        ``Path.is_dir()`` swallows the same error into ``False``. An absent
        directory, or a parent path that is not a directory, is an empty
        trail; any other listing failure is invalid.
        """
        audit_dir = db_path.parent
        # ⛔ THE MANIFEST IS STATTED BEFORE THE LISTING (codex, L3 of round 10b,
        # reproduced). Statted after it, a first rotation landing in between
        # left a stable signature on a manifest the listing never showed: the
        # pass skipped the sealed week and called an empty active file valid.
        manifest_signature = _stat_signature(audit_dir / f"{db_path.stem}.audit.manifest.json")
        try:
            names = {p.name for p in audit_dir.iterdir()}
        except (FileNotFoundError, NotADirectoryError):
            return AuditVerifyResult(valid=True, total_entries=0, files_verified=0), False
        except OSError as e:
            return AuditVerifyResult(
                valid=False, total_entries=0, files_verified=0,
                error=f"Cannot list audit directory: {e}",
            ), False
        return (
            cls._verify_listed(db_path, names, manifest_signature),
            _rotation_in_flight(db_path, names),
        )

    @classmethod
    def _verify_listed(
        cls,
        db_path: Path,
        names: set[str],
        manifest_signature: tuple[int, int, int] | None,
    ) -> AuditVerifyResult:
        """The pass itself, against one directory listing (``names``) and the
        manifest's signature taken before that listing."""
        stem = db_path.stem
        audit_dir = db_path.parent
        manifest_path = audit_dir / f"{stem}.audit.manifest.json"
        active_path = audit_dir / f"{stem}.audit.jsonl"

        # Load manifest (once) for file list + chain anchor
        files_to_verify: list[Path] = []
        chain_anchor = GENESIS_HASH
        missing_files: list[str] = []
        anchor_trusted = True

        markers = _quarantine_markers(audit_dir, stem)
        if markers:
            return AuditVerifyResult(
                valid=False, total_entries=0, files_verified=0,
                error=(
                    f"Manifest quarantined ({markers[-1]}): sealed history is "
                    "not covered until `anneal-memory audit-repair` rebuilds it"
                ),
            )

        # ⛔ THE MANIFEST'S SIGNATURE, TAKEN BEFORE THE LISTING, IS CHECKED AGAIN
        # BEFORE THE PASS IS CALLED VALID (L2, round 10, reproduced). A whole
        # rotation can land during the pass: it then walks the old file list,
        # reads an empty new active file, and returned valid=True without the
        # week it never walked. Rotation and retention replace the manifest,
        # so a changed signature means this pass was not a snapshot.
        if manifest_path.name in names:
            try:
                manifest = _parse_manifest_bytes(manifest_path.read_bytes(), stem)
                # Chain anchor from retention cleanup — trust point for
                # chains that no longer start from GENESIS
                anchor = manifest.get("chain_anchor", "")
                if anchor:
                    chain_anchor = anchor
                anchor_trusted = manifest.get("chain_anchor_recovered") is not True
                for f in manifest.get("files", []):
                    fpath = audit_dir / f["filename"]
                    # is_file(), not exists() (codex, round 6): a filename
                    # that resolves to a subdirectory (or a FIFO/device)
                    # passes exists() and then crashes open() with
                    # IsADirectoryError/OSError further down — "." and ".."
                    # were the two guaranteed-to-exist cases closed at the
                    # manifest-validation boundary; this closes the general
                    # one (any non-regular-file target) at the point it's
                    # actually used.
                    if fpath.is_file():
                        files_to_verify.append(fpath)
                    else:
                        missing_files.append(f["filename"])
            except _CORRUPT_MANIFEST as e:
                # OSError added round 6 (complement): the twin of cmd_audit's
                # round-4 fix — a real read failure (permission, I/O) on
                # this manifest still tracebacked out of the CLASSMETHOD
                # every "is this trail intact" check (`verify()`,
                # `--verify-audit`) depends on, instead of reporting it.
                return AuditVerifyResult(
                    valid=False, total_entries=0, files_verified=0,
                    error=f"Corrupt manifest: {e}",
                )

        # ⛔ A SEALED FILE THE MANIFEST DOES NOT COVER IS A GAP, NOT NOISE
        # (codex, round 9). Skipping a corrupt orphan left it on disk while
        # this method walked only the manifest, so it returned valid=True
        # over missing history, measured. The file on disk is the record.
        known = {p.name for p in files_to_verify} | set(missing_files)
        unmanifested = sorted(
            n for n in names
            if _is_sealed_filename(n, stem) and n not in known
            # A week's leftover .jsonl beside its manifested .gz is pending
            # cleanup only if it holds the same bytes: rotation leaves it
            # between saving the manifest and unlinking, and adoption sets an
            # identical copy aside. A different copy is reported (L1, round
            # 10: a clock regression can put another segment under that name).
            and not (
                n.endswith(".jsonl")
                and n + ".gz" in known
                and _same_uncompressed_bytes(audit_dir / n, audit_dir / (n + ".gz"))
            )
        )
        if unmanifested:
            return AuditVerifyResult(
                valid=False, total_entries=0, files_verified=0,
                error=(
                    "Unmanifested sealed audit file(s) on disk, not "
                    f"covered by the manifest: {unmanifested}{_RERUN_HINT}"
                ),
            )

        if active_path.name in names:
            files_to_verify.append(active_path)

        # ⛔ MISSING FILES ARE CHECKED BEFORE THE EMPTY-TRAIL VERDICT (complement,
        # codex and glm, L3 re-pass of round 10b, reproduced here and on main):
        # with the sealed and active files both deleted, the empty-trail return
        # came first and reported total loss as a valid, empty trail.
        if missing_files:
            return AuditVerifyResult(
                valid=False, total_entries=0, files_verified=0,
                error=(
                    "Missing sealed files referenced in manifest: "
                    f"{missing_files}{_RERUN_HINT}"
                ),
            )

        if not files_to_verify:
            # ⛔ THE EMPTY-TRAIL VALID RETURN RE-CHECKS THE MANIFEST TOO (codex,
            # L3 re-pass of round 10b, reproduced with a simulated empty
            # listing): a listing that saw nothing while a first rotation
            # landed returned valid=True with 0 entries. Every valid=True
            # return goes through the signature check.
            if _stat_signature(manifest_path) != manifest_signature:
                return AuditVerifyResult(
                    valid=False, total_entries=0, files_verified=0,
                    error=f"The manifest changed during verification{_RERUN_HINT}",
                )
            # ⛔ AND A FRESH LISTING MUST SHOW NO AUDIT FILE THE PASS DID NOT SEE
            # (codex, same re-pass, reproduced with a simulated listing): an
            # enumeration that missed a crashed first rotation's sealed file, with
            # no manifest yet, called that history an empty valid trail.
            try:
                fresh = {p.name for p in audit_dir.iterdir()}
            except OSError as e:
                return AuditVerifyResult(
                    valid=False, total_entries=0, files_verified=0,
                    error=f"Cannot list audit directory: {e}",
                )
            appeared = sorted(
                n for n in fresh - names
                if n in (active_path.name, manifest_path.name) or _is_sealed_filename(n, stem)
            )
            if appeared:
                return AuditVerifyResult(
                    valid=False, total_entries=0, files_verified=0,
                    error=f"Audit files appeared during verification: {appeared}{_RERUN_HINT}",
                )
            return AuditVerifyResult(valid=True, total_entries=0, files_verified=0)

        # Walk all files, verify chain
        total_entries = 0
        skipped = 0
        expected_hash = chain_anchor
        files_verified = 0

        for fpath in files_to_verify:
            # ⛔ SEQ MONOTONICITY, WITHIN EACH FILE ONLY (rotation MAY
            # restart ``_seq`` at 0 for a new file on the sealing path —
            # see ``_rotate_if_needed`` — but its early-return orphan-
            # adoption path does not touch ``_seq`` at all, so seq is only
            # ever comparable within a single file, never across the
            # boundary). ``prev_hash`` linkage alone cannot see a duplicated
            # entry whose chain is otherwise continuous: a retry that chains
            # correctly off an entry still on disk reuses that entry's
            # ``seq``, and the hash check above has nothing to say about it.
            # Strictly increasing (not ``== last_seq + 1``) so a legitimate
            # gap from a skipped torn line does not itself become a false
            # tampering verdict.
            last_seq: int | None = None
            # A file can vanish or fail to read after the is_file() check
            # above — a concurrent rotation/cleanup, a truncated gzip
            # stream (complement + glm + codex, round 7). That is an
            # unreadable trail, reported as a result, never a traceback.
            read_error: list[OSError] = []
            for line in _guarded_lines(fpath, read_error):
                line = line.strip()
                if not line:
                    continue

                try:
                    # Decode strictly FIRST, then parse the text — not
                    # ``json.loads(line)`` on the raw bytes, which decodes
                    # via ``surrogatepass`` and does not raise for a byte
                    # sequence that is invalid strict UTF-8 but happens to
                    # be a valid lone-surrogate encoding (complement L3,
                    # 2026-09-13 — the exact class ``_parse_manifest_bytes``
                    # above was written to close, present again a few lines
                    # away in the same method: the old order left this
                    # decode AFTER json.loads and OUTSIDE this try, so that
                    # shape raised ``UnicodeDecodeError`` uncaught).
                    line = line.decode("utf-8")
                    entry = _require_entry_dict(json.loads(line))
                except _UNPARSEABLE_JSON:
                    # A torn multibyte tail (diogenes, 2026-09-08) is the
                    # same "not a complete entry" shape as malformed JSON —
                    # ``_iter_lines`` now yields raw bytes so this is the
                    # single place both get counted, never a traceback.
                    skipped += 1
                    continue

                actual_prev = entry.get("prev_hash", "")
                if actual_prev != expected_hash:
                    return AuditVerifyResult(
                        valid=False,
                        total_entries=total_entries,
                        files_verified=files_verified,
                        # ⛔ CARRY THE SKIPPED COUNT OUT. This return sits
                        # INSIDE the counting loop and used to omit
                        # ``skipped_lines``, so the dataclass default of 0
                        # overwrote a count already incremented above — an
                        # operator investigating a TAMPERING verdict was told
                        # "0 malformed lines" while unreadable lines sat in the
                        # file they were being asked to trust. The three other
                        # early returns in this method legitimately omit it:
                        # they run BEFORE ``skipped`` exists. This was the only
                        # one that discarded a real number (measured
                        # 2026-09-04 by walking every construction site).
                        skipped_lines=skipped,
                        chain_break_at=entry.get("seq", total_entries),
                        chain_break_file=fpath.name,
                        anchor_trusted=anchor_trusted,
                        error=f"Hash mismatch at seq {entry.get('seq')}: "
                              f"expected {expected_hash[:20]}..., "
                              f"got {actual_prev[:20]}...",
                    )

                actual_seq = entry.get("seq")
                if (
                    isinstance(actual_seq, int)
                    and last_seq is not None
                    and actual_seq <= last_seq
                ):
                    return AuditVerifyResult(
                        valid=False,
                        total_entries=total_entries,
                        files_verified=files_verified,
                        skipped_lines=skipped,
                        chain_break_at=actual_seq,
                        chain_break_file=fpath.name,
                        anchor_trusted=anchor_trusted,
                        error=f"Duplicated or non-increasing seq {actual_seq} "
                              f"after seq {last_seq}: prev_hash linked "
                              "cleanly but the entry did not advance the "
                              "sequence",
                    )
                if isinstance(actual_seq, int):
                    last_seq = actual_seq

                # Compute hash from the line on disk, not a re-serialization.
                # _compute_hash normalizes whitespace (see its docstring).
                # Re-serializing via json.dumps would be byte-identical in
                # CPython today but fragile for cross-language verifiers.
                expected_hash = cls._compute_hash(line)
                total_entries += 1

            if read_error:
                # A file that vanished mid-pass is what a concurrent rotation
                # or retention cleanup looks like, so it carries the hint.
                vanished = isinstance(read_error[0], FileNotFoundError)
                return AuditVerifyResult(
                    valid=False,
                    total_entries=total_entries,
                    files_verified=files_verified,
                    skipped_lines=skipped,
                    chain_break_file=fpath.name,
                    anchor_trusted=anchor_trusted,
                    error=(
                        f"Unreadable audit file {fpath.name}: {read_error[0]}"
                        f"{_RERUN_HINT if vanished else ''}"
                    ),
                )
            files_verified += 1

        if _stat_signature(manifest_path) != manifest_signature:
            return AuditVerifyResult(
                valid=False,
                total_entries=total_entries,
                files_verified=files_verified,
                skipped_lines=skipped,
                error=f"The manifest changed during verification{_RERUN_HINT}",
            )

        return AuditVerifyResult(
            valid=True,
            total_entries=total_entries,
            files_verified=files_verified,
            skipped_lines=skipped,
            anchor_trusted=anchor_trusted,
        )

    @classmethod
    def repair_manifest(cls, db_path: str | Path) -> AuditRepairResult:
        """Rebuild a quarantined (or missing) manifest from the sealed files on disk.

        The ONLY way out of quarantine (hybrid, ruled by Phill 2026-09-13).
        Operator-run: do not run it while another process is writing the trail.

        Refuses, writing nothing, when a sealed week is unreadable or empty, or
        when consecutive sealed files do not hash-chain — it never guesses
        across a gap. ``sha256_file`` is left empty rather than recomputed,
        because a checksum of the bytes now on disk would bless whatever
        changed. If the first sealed file does not start at genesis, its
        starting hash is recorded as ``chain_anchor`` together with
        ``chain_anchor_recovered: true``, and :meth:`verify` then reports
        ``anchor_trusted=False``.
        """
        db_path = Path(db_path)
        stem = db_path.stem
        audit_dir = db_path.parent
        trail = cls(db_path)
        markers = _quarantine_markers(audit_dir, stem)

        if not markers and trail._manifest_path.exists():
            try:
                trail._load_manifest()
            except _ManifestQuarantined:
                markers = _quarantine_markers(audit_dir, stem)  # quarantined just now
            except _ManifestUnavailable as e:
                return AuditRepairResult(repaired=False, error=str(e))
            else:
                return AuditRepairResult(
                    repaired=False, error="The manifest is valid; there is nothing to repair."
                )

        try:
            by_period: dict[str, list[Path]] = {}
            for p in audit_dir.iterdir():
                if _is_sealed_filename(p.name, stem):
                    by_period.setdefault(_sealed_period(p.name, stem), []).append(p)

            records: list[dict[str, Any]] = []
            untracked: list[str] = []
            for period in sorted(by_period):
                chosen: tuple[Path, dict[str, Any]] | None = None
                for path in sorted(by_period[period], key=lambda q: not q.name.endswith(".gz")):
                    info = _sealed_record(path)
                    if chosen is None and info is not None and info["entries"] > 0:
                        chosen = (path, info)
                    elif chosen is not None:
                        untracked.append(path.name)
                if chosen is None:
                    names = sorted(q.name for q in by_period[period])
                    return AuditRepairResult(
                        repaired=False,
                        error=(
                            f"No readable entry in the sealed file(s) for {period} "
                            f"({names}); nothing was written."
                        ),
                    )
                path, info = chosen
                records.append({"path": path, "period": period, **info})
        except OSError as e:
            return AuditRepairResult(
                repaired=False, error=f"Could not read the sealed files: {e}; nothing was written."
            )

        for prev, cur in zip(records, records[1:]):
            if cur["first_prev_hash"] != prev["last_hash"]:
                return AuditRepairResult(
                    repaired=False,
                    error=(
                        f"{cur['path'].name} does not chain from {prev['path'].name}; "
                        "nothing was written."
                    ),
                )

        manifest: dict[str, Any] = {
            "version": 1,
            "db_path": db_path.name,
            "active_file": trail._active_path.name,
            "active_last_hash": records[-1]["last_hash"] if records else GENESIS_HASH,
            "active_last_seq": 0,
            "files": [
                {
                    "filename": r["path"].name,
                    "period": r["period"],
                    "entries": r["entries"],
                    "first_ts": r["first_ts"],
                    "last_ts": r["last_ts"],
                    "last_hash": r["last_hash"],
                    "sha256_file": "",  # never recomputed: see docstring
                }
                for r in records
            ],
        }
        recovered = bool(records) and records[0]["first_prev_hash"] != GENESIS_HASH
        if recovered:
            manifest["chain_anchor"] = records[0]["first_prev_hash"]
            manifest["chain_anchor_recovered"] = True
        trail._save_manifest(manifest)

        # Markers are released only AFTER the rebuilt manifest is durable: a
        # crash in between leaves the trail quarantined, and repair re-runs.
        for marker in markers:
            src = audit_dir / marker
            try:
                src.rename(src.with_name(f"{marker}.repaired"))
            except OSError as e:
                return AuditRepairResult(
                    repaired=False,
                    files=[r["path"].name for r in records],
                    chain_anchor_recovered=recovered,
                    untracked=untracked,
                    error=f"Manifest rebuilt, but quarantine marker {marker} could not be released: {e}",
                )
        _fsync_dir(audit_dir)

        return AuditRepairResult(
            repaired=True,
            files=[r["path"].name for r in records],
            chain_anchor_recovered=recovered,
            untracked=untracked,
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
        """Lazy init: recover seq and prev_hash from existing audit file.

        Sets _initialized only after all recovery steps complete. If any
        step raises (disk full, permission error during orphan adoption),
        the next log() call retries init instead of writing with broken state.
        """
        # Adopt orphaned sealed files — crash between rename and manifest
        # update during rotation leaves .gz files the manifest doesn't know about.
        self._adopt_orphaned_files()

        active = self._active_path
        # ⛔ ``or st_size == 0`` IS LOAD-BEARING, AND THIS FILE ALREADY KNEW
        # IT — ``_rotate_if_needed`` uses exactly this predicate. The two
        # disagreed, and the rollback below can CREATE the state they
        # disagree about: an append that fails as the first write into a
        # freshly rotated file truncates back to ``resume_at = 0`` and
        # leaves a ZERO-BYTE active file. A bare ``exists()`` then reads
        # that as "an active file with entries", skips the manifest
        # continuity branch, and keeps ``_prev_hash = GENESIS`` while the
        # sealed files ended somewhere else entirely.
        # MEASURED 2026-09-07 (L2): the next process writes seq 0 chained
        # from GENESIS and ``verify()`` returns
        # ``Hash mismatch at seq 0: expected sha256:6101402b..., got
        # sha256:GENESIS...`` — a false tampering verdict produced by the
        # rollback SUCCEEDING. A zero-byte file holds no entries, so
        # continuity must come from the manifest; there is no reading under
        # which the bare ``exists()`` was right.
        if not active.exists() or active.stat().st_size == 0:
            # Fresh start — anchor on the sealed files via the manifest.
            self._seed_from_manifest()
            self._last_week = _iso_week_now()
            self._initialized = True
            return

        # Recover from existing active file — find last valid JSON entry
        last_line = _read_last_valid_entry(active)
        if last_line:
            last_entry = json.loads(last_line)  # Guaranteed valid by helper
            self._seq = last_entry.get("seq", 0) + 1
            # Hash the line from disk, not a re-serialization
            self._prev_hash = self._compute_hash(last_line)
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
            # ⛔ A NONEMPTY ACTIVE FILE WITH NO VALID ENTRY IS THE SAME CASE
            # AS A ZERO-BYTE ONE, AND THIS BRANCH DID NOT KNOW IT. It fell
            # through keeping the constructor defaults — ``_seq = 0``,
            # ``_prev_hash = GENESIS`` — so the next append started a fresh
            # chain from genesis after sealed files that ended somewhere
            # else. MEASURED 2026-09-07 (codex, L3): rotate, tear the first
            # append into the new file, reopen ->
            # ``Hash mismatch at seq 0: expected sha256:a0db9699..., got
            # sha256:GENESIS...``. A false tampering verdict, identical in
            # shape to the zero-byte one fixed hours earlier the same day.
            # ⚡ THE ZERO-BYTE FIX WAS SCOPED BY SYMPTOM. It asked "is the
            # file empty?" when the question is "does the active file give
            # me a chain anchor?" — and a file holding only a torn fragment
            # answers no just as completely. Both branches now call ONE
            # helper, so they cannot drift apart again; two pieces of code
            # computing one thing, disagreeing exactly where the rollback
            # puts you, is the defect this file has now shipped twice.
            self._seed_from_manifest()
            self._last_week = _iso_week_now()

        self._initialized = True

    def _seed_from_manifest(self) -> None:
        """Anchor the chain on the sealed files when the active file has none.

        Called from BOTH no-usable-entry branches of :meth:`_initialize` —
        a missing/zero-byte active file, and a nonempty one holding no line
        that parses.

        ⛔ IT RESETS TO GENESIS FIRST, AND OMITTING THAT WAS A DEFECT THIS
        HELPER INTRODUCED BY BEING EXTRACTED. The inlined version it came
        from only ever ran on a FRESH instance, where ``_prev_hash`` was
        already ``GENESIS_HASH``, so "leave the cached values alone when
        there is no manifest" was correct by accident of its caller. The
        2026-09-07 rollback change made ``_initialize`` re-runnable on a
        DIRTY instance — and then "leave the cached values alone" retains
        the hash of an entry that has just been truncated away.
        MEASURED (codex L3 round 2, 2026-09-07): one entry, no manifest,
        the file emptied by a rollback whose fsync failed, re-init, next
        append -> ``Hash mismatch at seq 1: expected sha256:GENESIS...,
        got sha256:04eb388b...``. **A false tampering verdict, produced by
        the fix that was written to prevent false tampering verdicts.**

        ⛔ AND A MANIFEST THAT CANNOT BE READ IS NOT A MANIFEST THAT IS
        ABSENT. This caught ``OSError`` and returned — failing OPEN to
        genesis while sealed history ended somewhere else, so a TRANSIENT
        read error silently restarts the chain and ``verify()`` cries
        tampering once the disk recovers. ⚠ The inlined original caught
        ``(json.JSONDecodeError, KeyError)``; **the ``OSError`` was added by
        the same commit that made read errors PROPAGATE out of
        ``_read_last_valid_entry`` twenty lines away.** Two opposite
        decisions about the same class, in one commit.
        ▶ Absent is ``FileNotFoundError`` and nothing else. A transient read
        error propagates, leaving ``_initialized`` False so the next ``log()``
        retries rather than writing from a guessed anchor.

        ⛔ HYBRID (Phill, 2026-09-13): an INVALID manifest no longer degrades
        to genesis. :meth:`_load_manifest` quarantines it, and this seeds from
        the newest sealed file's last entry instead — the value rotation would
        have recorded — so appending continues on the true chain. If there is
        no readable sealed tail, it refuses (raises); genesis would be a guess.
        """
        # Reset FIRST: this can run on an instance whose cached chain state
        # is stale, and genesis is the only defensible starting anchor.
        self._prev_hash = GENESIS_HASH
        self._seq = 0
        try:
            manifest = self._load_manifest()  # absent -> fresh (genesis)
        except _ManifestQuarantined:
            self._seed_from_sealed_tail()
            return
        self._prev_hash = manifest.get("active_last_hash", GENESIS_HASH)
        self._seq = manifest.get("active_last_seq", 0)

    def _seed_from_sealed_tail(self) -> None:
        """Seed from the newest sealed file's last valid entry, or refuse.

        Used only while the manifest is quarantined. ``seq`` restarts at 0, as
        it does after a rotation. Refuses rather than falling back to an older
        week, which would silently skip a segment.
        """
        stem = self._db_path.stem
        audit_dir = self._db_path.parent
        sealed = [p for p in audit_dir.iterdir() if _is_sealed_filename(p.name, stem)]
        if not sealed:
            raise _ManifestQuarantined(
                "the audit manifest is quarantined and there is no sealed file to "
                "continue the chain from; run `anneal-memory audit-repair`"
            )
        newest = max(_sealed_period(p.name, stem) for p in sealed)
        candidates = sorted(
            (p for p in sealed if _sealed_period(p.name, stem) == newest),
            key=lambda p: not p.name.endswith(".gz"),
        )
        for path in candidates:
            last = _last_valid_sealed_line(path)
            if last is not None:
                self._prev_hash = self._compute_hash(last)
                self._seq = 0
                return
        raise _ManifestQuarantined(
            f"the audit manifest is quarantined and the newest sealed week ({newest}) "
            "has no readable entry to continue the chain from; run `anneal-memory audit-repair`"
        )

    def _adopt_orphaned_files(self) -> None:
        """Adopt sealed files that the manifest doesn't know about.

        This handles crash recovery: if the process dies between
        active.rename() and _save_manifest() during rotation, the sealed
        file exists on disk but the manifest has no record of it.
        Scans for both compressed (.gz) and uncompressed (.jsonl) orphans
        — crash can happen before or after gzip compression.

        ⛔ RECOVERY NEVER DELETES AN AUDIT FILE (round 10, adopted with the
        fan-in desk). Rounds 7, 8 and 9 each lost history in a recovery
        path that deleted a copy on a precondition the next review showed
        was not enough: a ``.gz`` that decompresses to EOF is not a ``.gz``
        holding the same entries (codex #2), a check made on one read does
        not cover a second read (codex #3), and a crash between saving the
        manifest and deleting left a counterpart that the next open adopted
        as a second segment (codex #1). A copy that is not adopted is
        renamed aside to ``<name>.dup-<UTC stamp>``, a stale gzip temp file
        to ``<name>.stale-<UTC stamp>``. Neither name is in the sealed-file
        language, so neither is adopted again or reported by ``verify()``,
        and the bytes stay on disk for an operator.

        Per week:
        - the manifest already names a copy → another copy holding the same
          bytes is an unfinished cleanup and is set aside; a different one
          stays on its name, where ``verify()`` reports it;
        - both copies read and hold the same bytes → the ``.gz`` is the copy,
          and the ``.jsonl`` is set aside;
        - both read and differ → the ``.jsonl`` is, because rotation writes
          the ``.gz`` from it, and the ``.gz`` stays on its name, where
          ``verify()`` reports it;
        - only one reads → that one is, and the unreadable copy stays on its
          name. (codex, L3 of round 10b, reproduced: setting a differing or
          unread copy aside hid entries only it held behind a valid verify.)
        - neither reads → nothing is set aside or adopted, ``verify()``
          reports the week, and writes continue.

        ⛔ AND A COPY IS ADOPTED ONLY WHERE IT CHAINS (L1 + L2, round 10,
        reproduced). An orphan skipped while unreadable let writes continue
        from the sealed tip; once readable it was appended after them and
        ``verify()`` reported a hash mismatch for good — a tampering verdict
        built from a transient read error. So an orphan's first entry must
        link to the sealed chain's tip (the last manifested file, else
        ``chain_anchor``, else genesis), each adopted week moves the tip, and
        an active file holding a valid entry must link to the last week adopted; weeks
        after that one are left. A week that does not chain stays on its
        name, unadopted, and ``verify()`` reports it: loud, and not shaped
        like tampering.
        Every decision and every manifest field for a file come from one
        read of it (``_scan_sealed``).
        """
        stem = self._db_path.stem
        audit_dir = self._db_path.parent
        prefix = f"{stem}.audit."
        # ⛔ QUARANTINE RETURNS BEFORE ANYTHING IS LISTED OR SET ASIDE (hybrid,
        # 2026-09-13). Adopting into a fresh manifest is the automatic rebuild
        # the hybrid forbids. Orphans stay on disk; verify() reports the
        # quarantine.
        try:
            manifest = self._load_manifest()
        except _ManifestQuarantined as e:
            logger.warning("Not adopting orphaned audit files: %s", e)
            return
        try:
            names = sorted(p.name for p in audit_dir.iterdir())
        except FileNotFoundError:
            return  # no directory yet, so nothing to adopt
        except OSError as e:
            # Writable but not listable. Recovery is skipped rather than
            # failing every write (L1, round 10, reproduced at mode 0o300);
            # verify() reports the directory itself.
            logger.warning("Cannot list audit directory for recovery: %s", e)
            return

        # A crash while compressing leaves ``<sealed>.jsonl.gz.tmp`` beside
        # the ``.jsonl`` it was being written from. Nothing adopts it.
        for name in names:
            if name.endswith(".jsonl.gz.tmp") and _is_sealed_filename(
                name.removesuffix(".tmp"), stem
            ):
                _set_aside(audit_dir / name, "stale")

        files = manifest["files"]
        known_files = {f["filename"] for f in files}
        listed_by_week = {_week_of(n, prefix): audit_dir / n for n in known_files}
        tip = files[-1].get("last_hash", "") if files else (
            manifest.get("chain_anchor") or GENESIS_HASH
        )

        # Grouped by week and walked in sorted order, so manifest entries are
        # chronological whatever mix of .gz and .jsonl orphans there is.
        orphans_by_week: dict[str, list[Path]] = {}
        for name in names:
            # Adopt only names the manifest parser will accept back
            # (round 7): a stray ``<stem>.audit.<x>.jsonl`` written into the
            # manifest made the next read reject it whole.
            if name not in known_files and _is_sealed_filename(name, stem):
                orphans_by_week.setdefault(_week_of(name, prefix), []).append(
                    audit_dir / name
                )

        chain: list[tuple[str, Path, _SealedScan, list[Path]]] = []
        all_scans: dict[Path, _SealedScan] = {}
        for week, paths in sorted(orphans_by_week.items()):
            if week in listed_by_week:
                listed = self._scan_sealed(listed_by_week[week])
                for path in paths:
                    copy = self._scan_sealed(path)
                    if (
                        listed.error is None
                        and copy.error is None
                        and copy.digest == listed.digest
                    ):
                        _set_aside(path, "dup")
                    else:
                        logger.warning(
                            "Leaving %s on its name: it is not a readable, "
                            "byte-identical copy of the manifested %s "
                            "(verify() reports it)",
                            path.name, listed_by_week[week].name,
                        )
                continue

            scans = {path: self._scan_sealed(path) for path in paths}
            all_scans.update(scans)
            readable = [path for path in paths if scans[path].error is None]
            if not readable:
                # ⛔ LEFT ON DISK, UNADOPTED, AND NOT RAISED (complement, round
                # 10). ``verify()`` reports it as unmanifested, so the gap is
                # loud; raising made every ``log()`` fail for as long as the
                # file stayed unreadable (``chmod 000``, reproduced 3 of 3).
                for path in paths:
                    logger.warning(
                        "Not adopting unreadable orphaned audit file %s (left "
                        "on disk; verify() reports it): %s",
                        path.name, scans[path].error,
                    )
                continue

            keep = readable[0]
            if len(readable) == 2:
                gz = next(p for p in readable if p.name.endswith(".gz"))
                plain = next(p for p in readable if not p.name.endswith(".gz"))
                if scans[gz].digest == scans[plain].digest:
                    keep = gz
                else:
                    keep = plain
                    logger.warning(
                        "Audit copies %s and %s hold different bytes; adopting "
                        "the uncompressed copy",
                        plain.name, gz.name,
                    )
            scan = scans[keep]
            if scan.first_prev_hash != tip:
                logger.warning(
                    "Not adopting orphaned audit file %s: its first entry does "
                    "not continue the sealed chain (verify() reports it)",
                    keep.name,
                )
                continue
            chain.append((week, keep, scan, paths))
            tip = scan.last_hash

        if chain and self._active_path.name in names:
            try:
                active_prev = _first_prev_hash(self._active_path)
            except OSError:
                active_prev = ""  # unreadable: no week can be shown to lead into it
            # An active file with no valid entry (None) has nothing to contradict,
            # and _initialize seeds it from the manifest tip anyway, so every week
            # that chains is adopted and there is one chain. An unreadable one ("")
            # cannot be inspected: no week can be shown to lead into it, so none is
            # adopted (L1 re-pass of round 10b, MED; the docstring said "non-empty").
            if active_prev is not None:
                links = [
                    i for i, (_, _, linked, _) in enumerate(chain)
                    if linked.last_hash == active_prev
                ]
                kept = chain[: links[-1] + 1] if links else []
                for _, path, _, _ in chain[len(kept):]:
                    logger.warning(
                        "Not adopting orphaned audit file %s: the active file "
                        "does not continue from it (verify() reports it)",
                        path.name,
                    )
                chain = kept

        for week, keep, scan, paths in chain:
            for path in paths:
                if path == keep:
                    continue
                other = all_scans[path]
                if other.error is None and other.digest == scan.digest:
                    _set_aside(path, "dup")
                else:
                    logger.warning(
                        "Leaving %s on its name: it is not a readable, "
                        "byte-identical copy of the adopted %s (verify() "
                        "reports it)",
                        path.name, keep.name,
                    )
            manifest["files"].append({
                "filename": keep.name,
                "period": week,
                "entries": scan.entries,
                "first_ts": scan.first_ts,
                "last_ts": scan.last_ts,
                "last_hash": scan.last_hash,
                "sha256_file": "",  # Not computed during adoption
            })
            if scan.last_hash:
                manifest["active_last_hash"] = scan.last_hash
            logger.info(
                "Adopted orphaned audit file: %s (%d entries)", keep.name, scan.entries
            )

        if chain:
            self._save_manifest(manifest)

    def _scan_sealed(self, path: Path) -> _SealedScan:
        """Read ``path`` once to the end: its entry metadata, and a digest of
        its uncompressed bytes so two copies of one week can be compared.

        ⛔ A READ ERROR IS RETRIED HERE, INSIDE THE CALL, AND THEN RETURNED —
        NEVER RAISED (complement, round 10, reproduced). Round 9 raised any
        error that was not corrupt gzip so the next ``log()`` would retry;
        a file that stays unreadable then made every ``log()`` raise. The
        bound is an attempt count, not a guess from the errno. It lives in
        the call rather than across ``log()`` calls because the
        ``_dropped_since_last`` comment in ``__init__`` records that every
        CLI invocation opens and closes a store, so a count kept per
        instance would spend a short-lived command's only event on the
        retry. Corrupt bytes are not retried.
        """
        scan = _SealedScan()
        for attempt in range(_ADOPTION_READ_ATTEMPTS):
            if attempt:
                time.sleep(_ADOPTION_RETRY_SECONDS)
            scan = _SealedScan()
            digest = hashlib.sha256()
            errors: list[OSError] = []
            for line in _guarded_lines(path, errors):
                digest.update(line)
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    text = stripped.decode("utf-8")
                    e = _require_entry_dict(json.loads(text))
                except _UNPARSEABLE_JSON:
                    continue  # Torn or malformed — skip, same shape either way
                if scan.entries == 0:
                    scan.first_prev_hash = e.get("prev_hash", "")
                ts = e.get("ts", "")
                if not scan.first_ts:
                    scan.first_ts = ts
                scan.last_ts = ts
                scan.entries += 1
                # Hash the line from disk, not a re-serialization
                scan.last_hash = self._compute_hash(text)
            scan.digest = digest.hexdigest()
            if not errors:
                return scan
            scan.error = errors[0]
            if isinstance(errors[0], _CorruptAuditFile):
                return scan
        return scan

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
            # ⛔ "ACTIVE MISSING" IS NOT PROOF THE ROTATION SUCCEEDED.
            # Rotation renames the active file before it compresses it and
            # before it records the week in the manifest (the numbered order
            # below). If a later step raises — disk full during the gzip is the
            # measured case — the sealed file exists, the manifest does not know
            # about it, and the active file is GONE. Arriving
            # here and simply advancing ``_last_week`` records that rotation as
            # done, and the next append starts a fresh active file chaining to
            # the orphan's hash.
            #
            # MEASURED 2026-09-04, same process, no crash: the following
            # ``verify()`` returned valid=False with "Hash mismatch at seq 3:
            # expected sha256:GENESIS..." — a TAMPERING-SHAPED VERDICT caused
            # by a failed disk write. ``verify`` is a classmethod and never
            # constructs a trail, so ``anneal-memory verify`` cannot trigger
            # the recovery that would fix it; the store reads as tampered
            # until some other operation happens to open it.
            #
            # Adoption already exists and is idempotent — it just only ran at
            # ``_initialize``. Run it here too, so the process that broke the
            # rotation is the one that repairs it rather than leaving a false
            # alarm for whoever looks next.
            try:
                self._adopt_orphaned_files()
            except Exception:
                # Adoption is best-effort recovery; failing it must not stop
                # the caller. The orphan stays on disk and the next open
                # retries, which is exactly the pre-existing behaviour.
                logger.warning(
                    "could not adopt orphaned sealed audit file(s) while "
                    "rotating; the trail may verify as broken until the store "
                    "is reopened", exc_info=True,
                )
            self._last_week = current_week
            return

        # ⛔ LOAD THE MANIFEST BEFORE THE RENAME (hybrid, 2026-09-13). If it is
        # quarantined or unreadable, do not rotate: keep appending to the active
        # file, leave ``_last_week`` alone so the next log() retries, and never
        # reach the save below with a manifest that stands in for one we could
        # not read.
        try:
            manifest = self._load_manifest()
        except _ManifestUnavailable as e:
            if not self._rotation_refusal_logged:
                logger.warning("Not rotating the audit trail: %s", e)
                self._rotation_refusal_logged = True
            return

        # Seal the active file with the old week label
        sealed_name = _sealed_filename(self._db_path.stem, self._last_week)
        sealed_path = active.parent / sealed_name
        sealed_gz_path = sealed_path.with_suffix(".jsonl.gz")
        tmp_gz_path = Path(str(sealed_gz_path) + ".tmp")

        # ⛔ NEVER ROTATE ONTO A WEEK ALREADY ON DISK (L2, round 10,
        # reproduced). The label comes from the clock, and a clock stepped
        # back across a week boundary (NTP, a restored VM snapshot) sealed the
        # same week twice: the replace overwrote the first copy's bytes, and
        # the duplicate manifest record then made every read reject the
        # manifest. A leftover temp of that week may be the last trace of it.
        # Refused: appending continues in the active file, and ``_last_week``
        # moves to the current week, so the next boundary seals under a label
        # that is not on disk. Leaving it made one refusal permanent for the
        # process: every later call re-derived the same sealed name, and
        # neither rotation nor retention ran again (codex + complement, L3 of
        # round 10b, reproduced).
        if sealed_path.exists() or sealed_gz_path.exists() or tmp_gz_path.exists():
            if not self._rotation_refusal_logged:
                logger.warning(
                    "Not rotating the audit trail: sealed week %s is already on "
                    "disk; appending to the active file instead",
                    self._last_week,
                )
                self._rotation_refusal_logged = True
            self._last_week = current_week
            return

        # ⛔ THIS ORDER IS WHAT LETS verify() IN ANOTHER PROCESS TELL A ROTATION
        # IN FLIGHT FROM A BROKEN TRAIL, AND WHAT KEEPS A POWER LOSS FROM
        # LEAVING AN EMPTY .gz AS THE ONLY COPY (round 10: the fan-in desk's
        # stall control and L2, reproduced). ``verify()`` re-checks an invalid
        # pass only while ``_rotation_in_flight`` sees a rotation, so no step
        # may leave a shape it cannot recognise:
        #   1. create the gzip temp before the rename; the temp is the marker
        #   2. rename the active file to the sealed .jsonl and compress it
        #   3. fsync the temp's data, then replace it with the .gz; a .gz the
        #      manifest does not name, beside its .jsonl, is also in flight
        #   4. save the manifest before the unlink; an identical leftover
        #      .jsonl of a manifested week is not reported
        #   5. unlink the .jsonl
        # The previous order renamed first and unlinked before saving: a 300ms
        # stall after the rename, after the replace, or between the unlink and
        # the save gave a false invalid in about 0.1s. Pinned per step by
        # ``test_verify_inside_a_stalled_rotation_step_settles_to_valid``.
        file_hash = hashlib.sha256()
        entry_count = 0
        first_ts = ""
        last_ts = ""

        try:
            with open(tmp_gz_path, "wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb") as f_out:
                    active.rename(sealed_path)
                    _fsync_dir(sealed_path.parent)
                    with open(sealed_path, "rb") as f_in:
                        for line in f_in:
                            file_hash.update(line)
                            f_out.write(line)
                            if line.strip():
                                entry_count += 1
                                try:
                                    # Decode strictly, then parse — both guarded
                                    # (same class complement L3 found at the
                                    # ``verify()`` entry loop, 2026-09-13: the old
                                    # unguarded ``line.decode("utf-8")`` outside
                                    # this try raised ``UnicodeDecodeError``
                                    # uncaught for a torn tail inside the sealed
                                    # file the rotation is writing).
                                    e = _require_entry_dict(
                                        json.loads(line.decode("utf-8").strip())
                                    )
                                    ts = e.get("ts", "")
                                    if not first_ts:
                                        first_ts = ts
                                    last_ts = ts
                                except _UNPARSEABLE_JSON:
                                    pass
                # ⛔ FSYNC THROUGH THE HANDLE THAT WROTE THE TEMP (complement, L3 of
                # round 10b; reasoned from documents, not run, since nothing here
                # runs Windows). There os.fsync is _commit, which calls
                # FlushFileBuffers, and that needs a handle with GENERIC_WRITE: the
                # read-only reopen this replaced would have failed every rotation.
                # Closing the GzipFile writes the trailer and leaves ``raw`` open.
                raw.flush()
                os.fsync(raw.fileno())
        except BaseException:
            # The temp was created before the rename. If the rename never
            # happened it holds no audit data, and left behind it would read
            # as a rotation in flight until the next open.
            if not sealed_path.exists():
                try:
                    tmp_gz_path.unlink()
                except OSError:
                    pass
            raise

        tmp_gz_path.replace(sealed_gz_path)
        _fsync_dir(sealed_gz_path.parent)

        # Update the manifest loaded before the rename (hybrid); the .jsonl is
        # unlinked only after the save below (round 10 order, step 4 then 5).
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
        # Reset seq for new file BEFORE saving manifest, so crash
        # recovery restores the correct starting seq (0), not the
        # pre-rotation value.
        self._seq = 0
        manifest["active_last_seq"] = self._seq
        self._save_manifest(manifest)

        # Only now does the manifest name the .gz; until this line the .jsonl
        # was the copy recovery would fall back to.
        sealed_path.unlink()

        self._last_week = current_week
        self._rotation_refusal_logged = False

        # Auto-cleanup old files
        if self._retention_days is not None:
            self._cleanup(manifest)

    def _cleanup(self, manifest: dict[str, Any] | None = None) -> int:
        """Remove rotated files older than retention_days."""
        if self._retention_days is None:
            return 0

        if manifest is None:
            try:
                manifest = self._load_manifest()
            except _ManifestUnavailable as e:
                logger.warning("Skipping audit retention cleanup: %s", e)
                return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)
        cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        removed = 0
        removed_last_hash = ""
        remaining_files = []
        for f in manifest.get("files", []):
            last_ts = f.get("last_ts", "")
            if not last_ts:
                # No timestamp — preserve file (can't determine age).
                # Empty last_ts occurs when orphan adoption processes a
                # file with zero valid JSON entries.
                remaining_files.append(f)
                continue
            if last_ts < cutoff_str:
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
        """Load the manifest, or a fresh one ONLY when none exists.

        ⛔ THE ONE GATE (hybrid, ruled by Phill 2026-09-13). This used to return
        a fresh manifest for ANY failure, and the next writer saved it over the
        original: every validator stricter than a writer became history loss
        (rounds 6 and 7), and a new process overwrote a corrupt manifest on open.

        - Quarantined (a marker is on disk) -> raises ``_ManifestQuarantined``.
        - Invalid -> renamed to a quarantine marker (never overwritten), then
          raises ``_ManifestQuarantined``.
        - Unreadable right now (permission, I/O) -> raises ``_ManifestUnavailable``
          with nothing renamed, so a flaky read can neither quarantine nor
          overwrite.
        - Absent -> a fresh manifest, as before.
        """
        stem = self._db_path.stem
        markers = _quarantine_markers(self._db_path.parent, stem)
        if markers:
            raise _ManifestQuarantined(
                f"the audit manifest is quarantined as {markers[-1]}; "
                "run `anneal-memory audit-repair`"
            )
        try:
            raw = self._manifest_path.read_bytes()
        except FileNotFoundError:
            return self._fresh_manifest()
        except OSError as e:
            raise _ManifestUnavailable(f"the audit manifest cannot be read right now: {e}") from e
        try:
            return _parse_manifest_bytes(raw, stem)
        except _UNPARSEABLE_JSON as e:
            marker = self._quarantine_manifest()
            raise _ManifestQuarantined(
                f"the audit manifest is invalid ({e}) and was quarantined as {marker}; "
                "run `anneal-memory audit-repair`"
            ) from e

    def _quarantine_manifest(self) -> str:
        """Rename the invalid manifest to a quarantine marker; never overwrite."""
        path = self._manifest_path
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = path.with_name(f"{path.name}.corrupt-{stamp}")
        if target.exists():
            raise _ManifestUnavailable(
                f"quarantine target {target.name} already exists; not overwriting it"
            )
        try:
            path.rename(target)
        except OSError as e:
            raise _ManifestUnavailable(f"could not quarantine the invalid audit manifest: {e}") from e
        _fsync_dir(path.parent)
        logger.warning(
            "Quarantined an invalid audit manifest as %s. Appending continues; "
            "rotation, orphan adoption and retention are paused until "
            "`anneal-memory audit-repair` rebuilds it.", target.name,
        )
        return target.name

    def _fresh_manifest(self) -> dict[str, Any]:
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
            _fsync_dir(path.parent)
        except Exception:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise

    @staticmethod
    def _compute_hash(json_line: str) -> str:
        """Compute SHA-256 hash of a JSON line.

        IMPORTANT: The .strip() call is LOAD-BEARING. Four call sites
        feed this method with inconsistent whitespace (some pre-stripped,
        some with trailing newlines from file iterators). The strip()
        normalizes all inputs to the same canonical form — the JSON
        content without surrounding whitespace. Removing it will cause
        silent hash chain verification failures.
        """
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
    handled correctly. Scans FORWARD to the end keeping the last line
    that parses as valid JSON — skips partial writes from crashes.

    ⚠ THIS SAID "WALKS BACKWARD FROM THE END" UNTIL 2026-09-07 AND THE
    CODE HAS NEVER DONE THAT — it is a plain ``for line in f``. The two
    are not equivalent for cost (backward would stop at the first valid
    line; this reads the whole active file every open) and a reader
    sizing the recovery path would have been misled in the cheap
    direction. Behaviour is identical, which is why it survived.

    Memory-safe: only keeps the last valid line in memory at a time.
    """
    last_valid = ""
    # ⛔ READ ERRORS PROPAGATE — THEY USED TO BE SWALLOWED, AND THAT MADE A
    # FAILED SCAN INDISTINGUISHABLE FROM AN EMPTY ONE. ``except (OSError,
    # UnicodeDecodeError): pass`` returned whatever had been found before the
    # error, and ``_initialize`` then marked itself initialised on a partial
    # answer. codex (L3, 2026-09-07): a complete seq N on disk, a read that
    # hits EIO after seq N-1, and the next append re-emits seq N — a false
    # tampering verdict built out of a suppressed read error.
    # ⚠ AND THE TWO FAILURES ARE CORRELATED, NOT INDEPENDENT: the caller that
    # most needs this is a rollback that ALREADY failed on this disk.
    # ⚖ Raising is the DOCUMENTED contract, not a new one — ``_initialize``'s
    # own docstring says a step that raises leaves the trail uninitialised so
    # the next ``log()`` retries "instead of writing with broken state".
    # Swallowing here was the deviation from it.
    # ⛔ BUT PROPAGATION WAS SCOPED BY EXCEPTION TYPE, NOT BY WHAT FAILED
    # (diogenes, 2026-09-08): opening in text mode means the file's own
    # line-splitting has to decode first, so a torn multibyte character
    # raised ``UnicodeDecodeError`` out of the ``for line in f`` as soon as
    # the decoder reached the tear — discarding every valid line already
    # scanned before it, not merely failing fast on none of them — a TORN
    # LINE, not a failed read, treated as the latter. Reading raw bytes and
    # decoding per line puts the tear back where the rest of this loop
    # already handles it: skipped, like a partial ``json.loads``.
    with open(path, "rb") as f:
        for raw in f:
            try:
                stripped = raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue  # Torn line — skip, same as a partial write
            if not stripped:
                continue
            try:
                _require_entry_dict(json.loads(stripped))
                last_valid = stripped
            except _UNPARSEABLE_JSON:
                pass  # Partial write, or valid JSON that isn't an entry — skip
    return last_valid


def _iter_lines(path: Path):
    """Iterate raw (undecoded) lines from an audit file, .gz transparent.

    ⛔ YIELDS BYTES, NOT STR (diogenes, 2026-09-08). This used to open in
    text mode, which decodes while splitting lines — so a torn multibyte
    character anywhere in the file raised ``UnicodeDecodeError`` straight
    out of the generator, past every caller's ``except json.JSONDecodeError``,
    including out of ``verify()`` and the CLI. Yielding raw bytes lets each
    call site decode strictly ITSELF before parsing, so a torn line raises
    ``UnicodeDecodeError`` at a point the caller's own try/except covers.
    ⚠ CORRECTED 2026-09-13 (complement L3): this docstring used to say
    decoding was deferred "to ``json.loads``," which raises
    ``UnicodeDecodeError`` the same way it raises ``JSONDecodeError`` —
    false. ``json.loads(bytes)`` decodes via ``surrogatepass`` and does
    NOT raise for a byte sequence that is invalid strict UTF-8 but happens
    to be a valid lone-surrogate encoding; every call site now decodes
    with ``bytes.decode("utf-8")`` (strict) BEFORE calling ``json.loads``
    on the resulting text, inside the same try.
    """
    if path.name.endswith(".gz"):
        # A truncated or corrupt gzip stream raises ``EOFError`` or
        # ``zlib.error``, neither an ``OSError`` (codex, round 7: both
        # escaped ``verify()`` and ``cmd_audit``'s OSError handler).
        # Normalized here, once. ⚠ That only helps a consumer that HAS an
        # OSError path — read through ``_guarded_lines`` where a raise
        # must not escape (round 8 found adoption had none).
        try:
            with gzip.open(path, "rb") as f:
                yield from f
        except (EOFError, zlib.error, gzip.BadGzipFile) as e:
            raise _CorruptAuditFile(f"corrupt compressed stream: {e!r}") from e
    else:
        with open(path, "rb") as f:
            yield from f


def _rotation_in_flight(db_path: Path, names: set[str]) -> bool:
    """Whether a directory listing shows a rotation between its first and last
    step, in the order ``_rotate_if_needed`` documents: a sealed gzip temp
    exists, or a week's ``.gz`` sits beside its ``.jsonl`` while the manifest
    does not name the ``.gz``. A pair with either file named is not in flight:
    it is cleanup ``verify()`` ignores or a different copy it reports, and
    counting it would make every verify of that trail wait out the cap.
    """
    stem = db_path.stem
    if any(
        n.endswith(".jsonl.gz.tmp") and _is_sealed_filename(n.removesuffix(".tmp"), stem)
        for n in names
    ):
        return True
    pairs = {
        n for n in names
        if n.endswith(".gz") and _is_sealed_filename(n, stem) and n.removesuffix(".gz") in names
    }
    if not pairs:
        return False
    manifest_path = db_path.parent / f"{stem}.audit.manifest.json"
    try:
        manifest = _parse_manifest_bytes(manifest_path.read_bytes(), stem)
    except FileNotFoundError:
        return True
    except _CORRUPT_MANIFEST:
        return False
    named = {f["filename"] for f in manifest["files"]}
    # A .gz beside a manifested .jsonl of its week is a differing copy that
    # adoption leaves on its name (codex, L3 of round 10b), not a rotation:
    # rotation names neither file of a week until it names the .gz.
    return any(p not in named and p.removesuffix(".gz") not in named for p in pairs)


def _stat_signature(path: Path) -> tuple[int, int, int] | None:
    """``(inode, size, mtime_ns)`` of ``path``; None if it does not exist, and
    ``(-1, -1, -1)`` if it cannot be stat'ed for another reason."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    except OSError:
        return (-1, -1, -1)
    return (st.st_ino, st.st_size, st.st_mtime_ns)


def _same_uncompressed_bytes(a: Path, b: Path) -> bool:
    """True iff both files read to the end and yield identical bytes."""
    digests = []
    for path in (a, b):
        errors: list[OSError] = []
        digest = hashlib.sha256()
        for line in _guarded_lines(path, errors):
            digest.update(line)
        if errors:
            return False
        digests.append(digest.digest())
    return digests[0] == digests[1]


def _last_valid_sealed_line(path: Path) -> str | None:
    """The last line of ``path`` that is a valid entry, or None if the file is
    corrupt or holds none. A transient read failure is re-raised."""
    errors: list[OSError] = []
    last: str | None = None
    for raw in _guarded_lines(path, errors):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            text = stripped.decode("utf-8")
            _require_entry_dict(json.loads(text))
        except _UNPARSEABLE_JSON:
            continue
        last = text
    if errors:
        if not isinstance(errors[0], _CorruptAuditFile):
            raise errors[0]
        return None
    return last


def _sealed_record(path: Path) -> dict[str, Any] | None:
    """What a manifest record needs, computed from the file's own lines, or
    None if the file is corrupt. A transient read failure is re-raised."""
    errors: list[OSError] = []
    entries = 0
    first_ts = last_ts = last_hash = ""
    first_prev_hash: str | None = None
    for raw in _guarded_lines(path, errors):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            text = stripped.decode("utf-8")
            entry = _require_entry_dict(json.loads(text))
        except _UNPARSEABLE_JSON:
            continue
        if first_prev_hash is None:
            first_prev_hash = entry.get("prev_hash", "")
        ts = entry.get("ts", "")
        first_ts = first_ts or ts
        last_ts = ts
        entries += 1
        last_hash = AuditTrail._compute_hash(text)
    if errors:
        if not isinstance(errors[0], _CorruptAuditFile):
            raise errors[0]
        return None
    return {
        "entries": entries,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "first_prev_hash": first_prev_hash or "",
        "last_hash": last_hash,
    }


def _first_prev_hash(path: Path) -> str | None:
    """``prev_hash`` of the first valid entry in ``path`` ("" if that entry has
    none), or None if the file holds no valid entry. A read error raises."""
    errors: list[OSError] = []
    for line in _guarded_lines(path, errors):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            entry = _require_entry_dict(json.loads(stripped.decode("utf-8")))
        except _UNPARSEABLE_JSON:
            continue
        prev = entry.get("prev_hash", "")
        return prev if isinstance(prev, str) else ""
    if errors:
        raise errors[0]
    return None


def _week_of(name: str, prefix: str) -> str:
    """The ISO week label in a sealed filename ``<prefix><week>.jsonl[.gz]``."""
    return name.removeprefix(prefix).removesuffix(".gz").removesuffix(".jsonl")


def _set_aside(path: Path, reason: str) -> None:
    """Rename ``path`` to ``<name>.<reason>-<UTC stamp>`` in the same
    directory — the only way recovery takes a file off its name.

    Never replaces an existing file and never deletes. A failure is logged,
    not raised, and writes continue: the file stays under its own name and
    the next open tries again. ⚠ A sealed name left in place is still
    reported by ``verify()``; a ``.jsonl.gz.tmp`` left in place is not
    reported, and ``verify()`` reads it as a rotation compressing, so an
    invalid verdict on that trail waits out ``_ROTATION_SETTLE_MAX_SECONDS``.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = path.with_name(f"{path.name}.{reason}-{stamp}")
    suffix = 0
    try:
        while os.path.lexists(target):
            suffix += 1
            target = path.with_name(f"{path.name}.{reason}-{stamp}-{suffix}")
        os.rename(path, target)
    except OSError:
        logger.warning(
            "Could not set aside audit file %s; it stays under its own name",
            path.name, exc_info=True,
        )
        return
    _fsync_dir(path.parent)
    logger.warning("Set aside audit file %s as %s", path.name, target.name)


def _guarded_lines(path: Path, errors: list[OSError]):
    """``_iter_lines``, but a read failure stops iteration and is appended
    to ``errors`` instead of raising — for callers that must turn an
    unreadable file into a result rather than a traceback."""
    try:
        yield from _iter_lines(path)
    except OSError as e:
        errors.append(e)
