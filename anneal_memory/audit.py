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

import errno
import gzip
import hashlib
import json
import logging
import os
import stat
import secrets
import sys
import traceback
import re
import threading
import time
import zlib
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

try:  # POSIX advisory locking; absent on Windows (see AuditTrail._manifest_lock).
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

# ``flock`` raising one of these means "advisory locking is unavailable on this
# filesystem" (some NFS configs), not a fault: the manifest lock then degrades
# to no lock, as ``store.continuity_lock`` does. Duplicated rather than imported
# from ``store`` because this module imports none of its siblings.
_LOCK_UNAVAILABLE_ERRNOS = frozenset(
    e for e in (
        getattr(errno, "ENOLCK", None),
        getattr(errno, "EOPNOTSUPP", None),
        getattr(errno, "ENOTSUP", None),
    ) if e is not None
)

# ``flock(LOCK_NB)`` raising one of these means "another holder has it".
_LOCK_HELD_ERRNOS = frozenset({errno.EWOULDBLOCK, errno.EAGAIN})

# How long an append waits for another holder of the append lock before the
# append is refused and counted as a dropped audit write (see
# AuditTrail._append_lock). A healthy holder keeps it for one append, or for a
# week's rotation; this bounds a stopped or hung one.
_APPEND_LOCK_TIMEOUT_SECONDS = 30.0
# How long ``stats()`` waits on a peer's append lock when only a staged first
# entry makes the trail look unknown (a first append stages it for an instant).
_STATS_STAGED_WAIT_SECONDS = 2.0


def _week_of_ts(ts: str) -> str:
    """``YYYY-WNN`` of an entry timestamp, or ``""`` when it does not parse."""
    try:
        iso = datetime.fromisoformat(ts.replace("Z", "+00:00")).isocalendar()
    except (ValueError, AttributeError):
        return ""
    return f"{iso[0]}-W{iso[1]:02d}"


def _unlock_and_close(fd: int) -> None:
    """Release a ``flock`` explicitly, then close. A close alone leaves the lock
    held while a forked child still has a copy of the descriptor."""
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        os.close(fd)


# The module's logger, used as logging registered it: never re-classed or
# wrapped, so an application's logger class, levels, filters and handlers on it
# behave as the application set them up. From "anneal-memory", its parent,
# handlers apply by propagation and its level only while this logger's own is
# unset; logging never runs a parent's filters.
logger = logging.getLogger("anneal-memory.audit")


def _log(level: int, msg: str, *args: object, exc_info: bool = False,
         stacklevel: int = 1) -> str:
    """Emit one diagnostic through ``logger``. Returns where it went: "logged"
    (the logger call returned), "stderr" (it raised and the fallback below was
    written) or "lost" (both raised).

    Every diagnostic here sits on a degrade, refusal or recovery path, and an
    application's logging code that raised replaced those outcomes (L3 10-03,
    run). So the whole emission (record creation, filters, handlers) is one
    guarded call: an ``Exception`` from any of it is swallowed, the message goes
    to stderr instead, and handlers after the raising one get nothing for that
    record. ``BaseException`` subclasses that are not ``Exception``
    (KeyboardInterrupt, SystemExit, asyncio's CancelledError) still propagate."""
    caller_exc = sys.exc_info() if exc_info else None  # before our own except replaces it
    if caller_exc is not None and caller_exc[0] is None:
        caller_exc = None
    try:
        logger.log(level, msg, *args, exc_info=caller_exc, stacklevel=stacklevel + 1)
        return "logged"
    except Exception:
        try:
            text = msg % args if args else msg
        except Exception:
            text = str(msg)
        try:
            if caller_exc is not None:
                text += "\n" + "".join(traceback.format_exception(*caller_exc))
            # One write, so a failure cannot leave half a line that the caller's
            # retry then duplicates (L3 r3 10-03, complement).
            sys.stderr.write(f"[anneal-memory] {logging.getLevelName(level)}: {text}\n")
            return "stderr"
        except Exception:
            return "lost"


# Lock paths whose runtime ``flock`` degrade has been reported in this process
# (AuditTrail._open_and_flock).
_lock_degrade_warned: set[str] = set()
_ENOLCK_RETRIES = 3
_ENOLCK_RETRY_SECONDS = 0.05


def _emit_warning(message: str, *, stderr: bool = False) -> None:
    """A diagnostic to the logger (see ``_log``) and, when ``stderr``, a copy on
    standard error for an application that keeps only that channel. When the
    logger raised, ``_log``'s own fallback is that copy, so this module does not
    print a second one (an application handler that writes to stderr before
    raising is not counted)."""
    went = _log(logging.WARNING, message, stacklevel=2)
    if stderr and went != "stderr":
        try:
            sys.stderr.write(f"[anneal-memory] WARNING: {message}\n")
        except Exception:
            pass

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


def _markers_in(names: set[str] | list[str], stem: str) -> list[str]:
    """The quarantine markers for ``stem`` among ``names``, oldest first."""
    pattern = re.escape(f"{stem}.audit.manifest.json") + _QUARANTINE_SUFFIX_PATTERN
    return sorted(n for n in names if re.fullmatch(pattern, n))


def _quarantine_markers(audit_dir: Path, stem: str) -> list[str]:
    """Unresolved quarantined manifests for ``stem``, oldest first. A listing
    error other than a missing directory is raised."""
    try:
        return _markers_in([p.name for p in audit_dir.iterdir()], stem)
    except FileNotFoundError:
        return []


class _AuditLockError(OSError):
    """The manifest lock could not be taken for a reason other than "this
    filesystem has no advisory locks" (that case degrades to no lock). Callers
    that would quarantine or repair refuse instead of proceeding unlocked."""


class _ManifestUnavailable(OSError):
    """The manifest cannot be used right now. Maintenance steps that need it
    (rotation, retention) skip; an append that needs it to record or check the
    active file's first entry is refused (Phill 2026-10-08, "A"). A plain
    instance is transient (a read error) and callers retry; see
    ``_ManifestQuarantined``."""


class _ManifestQuarantined(_ManifestUnavailable):
    """The manifest was invalid and is quarantined; only audit-repair clears it.

    ``markers`` names every marker the raiser saw, oldest first, when it knows
    them (``_load_manifest`` always does), so a caller never has to list the
    directory again, and never releases fewer markers than there are (glm,
    re-pass 598cd40ffcfcbc18, reproduced by injection).
    """

    def __init__(self, message: str, markers: list[str] | None = None) -> None:
        super().__init__(message)
        self.markers = list(markers or [])


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


# The fields of one record in the manifest's ``set_aside`` list.
_SET_ASIDE_KEYS = ("filename", "set_aside_as", "period", "cause", "at")
# The reason ``audit-repair`` sets an unreadable or corrupt sealed file aside
# under: ``<sealed name>.unreadable-<UTC stamp>`` (see ``_set_aside``).
_UNREADABLE_REASON = "unreadable"
# The reason a staged first entry that did not commit is set aside under:
# ``<active>.first.discarded-<UTC stamp>`` (KL-24 L3 r6: staged bytes are never
# deleted).
_DISCARDED_REASON = "discarded"
# ``certainty`` on a set-aside record that may not be a gap at all: a manifest
# rebuilt from quarantine over an active file with no entry cannot know whether
# it ever held one. A record without the field is a definite gap.
_POSSIBLE = "possible"


def _discarded_name_pattern(stem: str) -> "re.Pattern[str]":
    """The exact ASCII name of a set-aside staged entry of ``stem``'s trail:
    ``<stem>.audit.jsonl.first.discarded-<UTC stamp>[-n]``; group 1 is the stamp."""
    return re.compile(
        re.escape(f"{stem}.audit.jsonl.first.{_DISCARDED_REASON}-")
        + r"([0-9]{8}T[0-9]{12}Z)(?:-[1-9][0-9]*)?",
        re.ASCII,
    )


def _canonical_stamp(stamp: str) -> bool:
    """True when ``stamp`` is a UTC stamp ``_set_aside`` could have written: it
    parses and formats back to the same text."""
    try:
        return datetime.strptime(stamp, "%Y%m%dT%H%M%S%fZ").strftime(
            "%Y%m%dT%H%M%S%fZ"
        ) == stamp
    except ValueError:
        return False


def _discarded_name_ok(stem: str, name: str) -> bool:
    """``name`` is exactly a set-aside staged-entry name of ``stem``'s trail."""
    match = _discarded_name_pattern(stem).fullmatch(name)
    return match is not None and _canonical_stamp(match[1])


def _week_bounds(period: str) -> tuple[str, str] | None:
    """The first instant of ISO week ``period`` (``YYYY-Www``) and of the NEXT
    ISO week, in the stamp format of a set-aside name; None when ``period``
    does not parse."""
    match = re.fullmatch(r"([0-9]{4})-W([0-9]{2})", period, re.ASCII)
    if match is None:
        return None
    try:
        start = datetime.fromisocalendar(int(match[1]), int(match[2]), 1)
    except ValueError:
        return None
    fmt = "%Y%m%dT%H%M%S%fZ"
    return start.strftime(fmt), (start + timedelta(days=7)).strftime(fmt)


_NOTHING_TO_REPAIR = "The manifest is valid; there is nothing to repair."
# The manifest's ``active_begun`` record: the week of the active file's first
# entry, that entry's hash, and the ``prev_hash`` it chained from, saved once per
# active file by ``log()``. Cleared by the seal that moves the file into
# ``files`` and by an adoption of the orphan that starts from that ``prev_hash``
# (a rotation that crashed before its manifest save).
_ACTIVE_BEGUN_KEYS = ("period", "first_hash", "first_prev_hash")


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
    set_aside = manifest.get("set_aside", [])
    if not isinstance(set_aside, list) or not all(
        isinstance(r, dict) and all(isinstance(r.get(k), str) for k in _SET_ASIDE_KEYS)
        for r in set_aside
    ):
        raise TypeError("manifest field 'set_aside' is not a list of set-aside records")
    # ``certainty`` is written only on the rebuild's active-file record (KL-24
    # L3 r7, codex LOW): anything else would silently read as a definite gap.
    rated = [r for r in set_aside if "certainty" in r]
    # No limit on their number: each repair of a loss in the same week records
    # its own (measured: a cap made the second repair's manifest invalid).
    if not all(
        r["certainty"] == _POSSIBLE and r["set_aside_as"] == ""
        and r["filename"] == f"{stem}.audit.jsonl"
        for r in rated
    ):
        raise TypeError("manifest field 'set_aside' holds an invalid 'certainty'")
    # ``preserved_attempts``: names of kept staged entries, on an active-file
    # record only.
    if not all(
        "preserved_attempts" not in r or (
            r["set_aside_as"] == ""
            and r["filename"] == f"{stem}.audit.jsonl"
            and isinstance(r["preserved_attempts"], list)
            and all(
                isinstance(n, str) and _discarded_name_ok(stem, n)
                for n in r["preserved_attempts"]
            )
        )
        for r in set_aside
    ):
        raise TypeError("manifest field 'set_aside' holds an invalid 'preserved_attempts'")
    begun = manifest.get("active_begun")
    if begun is not None and not (
        isinstance(begun, dict)
        and all(isinstance(begun.get(k), str) for k in _ACTIVE_BEGUN_KEYS)
    ):
        raise TypeError("manifest field 'active_begun' is not an active-file record")
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
    # Sealed files audit-repair set aside as unreadable or corrupt, from the
    # manifest's ``set_aside`` record (Phill, 2026-10-03): each a dict with
    # ``filename``, ``set_aside_as``, ``period``, ``cause`` and ``at``. Their
    # entries are a known gap the chain continues past, so ``valid`` speaks
    # for the chain without them, reported beside it as ``anchor_trusted`` is.
    set_aside: list[dict[str, str]] = field(default_factory=list)


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
    # Unreadable or corrupt sealed files this repair set aside and recorded in
    # the manifest (the same dicts as ``AuditVerifyResult.set_aside``).
    set_aside: list[dict[str, str]] = field(default_factory=list)
    # What repair did with a staged first entry (``<active>.first``) it found,
    # before anything else: "finished: ..." or "set aside: ..." (KL-24 L3 r6).
    # None when there was none.
    staged_first_entry: str | None = None


@dataclass
class _SealedScan:
    """One full read of a candidate sealed file, for orphan adoption."""

    entries: int = 0
    first_ts: str = ""
    last_ts: str = ""
    last_hash: str = ""
    first_prev_hash: str | None = None  # prev_hash of the first valid entry
    first_hash: str = ""  # hash of the first valid entry's line
    digest: str = ""  # sha256 of the uncompressed bytes
    error: OSError | None = None  # the read error that ended the last attempt


class AuditTrail:
    """Hash-chained JSONL audit trail.

    Appends tamper-evident entries to a JSONL sidecar file alongside
    the SQLite episodic store. Each entry includes the SHA-256 hash
    of the previous entry, creating a cryptographic chain.

    **Concurrent writers:** several instances, in one process or several, may
    write to one db_path. Each append holds a cross-process lock
    (:meth:`_append_lock`) and first re-reads the chain's tip from disk
    (:meth:`_resync_with_disk`), so every entry chains from the one before
    it on disk. Where advisory locks do not exist (Windows, or a filesystem
    without ``flock``), appends are not serialized and only one instance may
    write at a time; sequential writers stay chained through the re-sync.

    **No fork support:** do not fork a process while an AuditTrail in it is
    in use; a child must open its own AuditTrail. A child forked while the
    manifest lock is held inherits its descriptor, and so the lock, along
    with the parent's lock state.

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
        # The cross-process manifest lock (spore-1030), taken ONCE per public
        # operation (``_operation_span``, ``repair_manifest``) and never nested:
        # the internals require it (``_require_lock``) instead of taking it.
        # ``_lock_owner`` is the thread holding it; ``_lock_failures`` holds,
        # per thread, the error an operation's acquisition raised, which each
        # internal then raises as if its own acquisition had failed.
        self._lock_mutex = threading.Lock()
        self._lock_fd: int | None = None
        self._lock_owner: int | None = None
        self._lock_failures: dict[int, _AuditLockError] = {}
        self._span_threads: set[int] = set()  # threads inside _operation_span
        # Why the last orphan adoption did not complete, for the refusal that
        # follows it (L2: a refusal that named no cause left no way out).
        self._adoption_skip_reason = ""
        # Unmanifested sealed files newer than the manifest's last week that
        # the last adoption could not read, as (name, cause).
        self._unreadable_newer: list[tuple[str, str]] = []
        # Whether the active file holds an entry this instance knows of: set by
        # recovery and by each append, cleared by every seed and by a seal. An
        # append that finds the file absent or empty while this is True is
        # writing past a file that vanished under it (L3 r1 10-03, codex HIGH +
        # complement, run).
        self._active_has_entry = False
        # Threads inside :meth:`log`'s append-lock span (KL-24): a nested
        # ``log()`` on one of them is refused rather than deadlocked.
        self._append_threads: set[int] = set()
        # Where the entry this instance's chain state was taken from sits in the
        # active file: ``(st_dev, st_ino, offset, length)``. None whenever the
        # tip is not in the active file (``_active_has_entry`` False: a fresh
        # file, a seed from the manifest, a seal). ``_resync_with_disk`` checks
        # those bytes still hash to ``_prev_hash`` before trusting anything
        # after them (L2 r1: a reused inode made a size check unsound).
        self._tip: tuple[int, int, int, int] | None = None
        # The ISO week of the tip entry's own timestamp (see _lost_active).
        self._tip_week = ""

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
                     continuity_saved, continuity_refused, section_schema_set
                   Consolidate policy:
                     consolidate_policy_set
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

        # ⛔ RE-ENTRY IS REFUSED HERE, BEFORE THE GATE BELOW (L3 r1 10-03, codex
        # HIGH): _operation_span's own refusal runs only when this call needs a
        # span, and a refused rotation advances _last_week before its warning,
        # so a handler's nested log() needed none and appended while the outer
        # operation held the manifest lock. It also runs before the append lock
        # is taken: a nested log() would open its own descriptor and block on
        # this thread's own lock forever (KL-24).
        me = threading.get_ident()
        if me in self._span_threads or me in self._append_threads:
            raise RuntimeError(
                "an audit operation is already in progress on this thread; "
                "the trail is not reentrant (e.g. from a logging handler)"
            )

        try:
            # Inside the ``try``: an interrupt between the add and a ``try`` would
            # leave this thread marked for good, and every later append refused
            # as reentrant (L3 r1, codex).
            self._append_threads.add(me)
            with self._append_lock():
                entry = self._log_locked(event, data, actor)
        finally:
            self._append_threads.discard(me)

        # Fire callback after successful write, outside every lock (the
        # ``_manifest_lock`` invariant: no user callback inside a locked span).
        if self._on_event is not None:
            try:
                self._on_event(entry)
            except Exception:
                _log(logging.WARNING, "on_event callback failed for seq %d", entry["seq"], exc_info=True)

        return entry

    def _log_locked(
        self,
        event: str,
        data: dict[str, Any] | None,
        actor: str,
    ) -> dict[str, Any]:
        """:meth:`log` from the re-sync on: the caller holds the append lock
        (:meth:`_append_lock`) and has validated ``event`` and ``data``."""
        # Quarantine first (KL-24 L3 r6, codex 5): run the other way round, the
        # staged-entry recovery read a quarantined manifest as absent and
        # discarded a staged entry its record still named.
        self._refuse_while_quarantined()
        self._finish_first_entry()
        # ⛔ RE-SYNC WITH DISK FIRST (KL-24, run 2026-10-07 on 8542f49: three
        # writer processes, ``verify()`` valid=False, ``status()`` counting no
        # audit failure). Another process may have appended, rotated or replaced the
        # active file since this instance last touched it, so the cached
        # ``seq``/``prev_hash`` may be a tip that is no longer the chain's.
        self._resync_with_disk()

        # ⛔ ONE MANIFEST-LOCK SPAN FOR EVERYTHING THIS CALL DOES TO THE MANIFEST
        # (Phill, 10-03, item 2; reproduced with a real second process first):
        # with adoption and the seed in separate spans, adoption saw a
        # quarantined manifest, an ``audit-repair`` in another process rebuilt
        # it in between, and the seed refused this write on adoption's stale
        # "did not complete" while naming the repair that had just succeeded.
        # Taken by EVERY append, and released before the append and before
        # ``on_event``, which must never run under it. Taken only when a call
        # initialized or rotated, an initialized same-week writer appended with
        # the lock file unopenable and never found out (KL-24 L3 r6, codex 3,
        # run): the span is the preflight, and costs no manifest read when
        # nothing else needs one.
        reinit = not self._initialized or _iso_week_now() != self._last_week
        with self._operation_span():
            self._refuse_without_manifest_lock()
            if reinit:
                if not self._initialized:
                    self._initialize()
                self._rotate_if_needed()
        if reinit:
            # The span may have just found and quarantined an invalid manifest;
            # the append that found it is refused too (ruling A, lane run: it
            # was accepted while only the next one was refused).
            self._refuse_while_quarantined()

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
        # retry sound. Safe because the caller holds the append lock, so no
        # peer appends between ``resume_at`` and the truncate; where locking is
        # unavailable (see :meth:`_append_lock`) the trail is single-writer by
        # contract, and truncating a file a peer was appending to would not be.
        active = self._active_path
        active.parent.mkdir(parents=True, exist_ok=True)
        resume_at = active.stat().st_size if active.exists() else 0
        if resume_at == 0 and self._active_has_entry:
            # ⛔ THE FILE THIS INSTANCE WAS APPENDING TO IS GONE. A file already
            # gone at ``_resync_with_disk`` was refused (or followed, when
            # another writer sealed its week) there, so this fires only when it
            # went between that re-sync and here: removed by something that
            # takes no append lock (a hand, an anneal from before the lock), or
            # with no lock held at all (see _append_lock). Appending would
            # chain from its lost tip into a new file and overwrite the
            # manifest's record of it; verify then reports a hash break that
            # audit-repair cannot see (L3 r1 10-03, run). Re-init on the next
            # call, so the refusal and the repair go through the manifest.
            self._initialized = False
            raise _ManifestUnavailable(
                f"the active audit file {active.name} this process was appending to "
                "is gone (deleted or emptied, or sealed by another process); not "
                "continuing the chain past it. If "
                "it can be restored, put it back and retry; otherwise run "
                "`anneal-memory audit-repair` to record the week as a gap"
            )
        had_entry = self._active_has_entry
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
            with _open_regular(active) as f_probe:
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
        saved_has_entry = self._active_has_entry  # L3 r2 10-03, codex MED
        saved_tip = self._tip
        # ⛔ WRITE-AHEAD: THE FIRST ENTRY'S RECORD IS DURABLE BEFORE THE ENTRY
        # (KL-24 L3 r4, codex HIGH). Saved after the write and best-effort, a
        # failed save left only this process's memory knowing the file had
        # entries: delete the file and, after one refusal, the next append
        # seeded from genesis and verify() read valid; a restart read a healthy
        # 0. Now a save that fails refuses this append, with nothing written.
        # The first entry of an active file is not appended: it is staged in a
        # temp file with any bytes already there, fsynced, its record saved, and
        # then renamed into place (KL-24 L3 r5, codex + complement: saved first
        # and appended after, a crash between the two left a record naming an
        # entry never written, and repair then recorded a false permanent gap).
        # The temp is what recovery finishes (:meth:`_finish_first_entry`); the
        # same order as LevelDB publishing a new MANIFEST through CURRENT.
        first_tmp: Path | None = None
        if not had_entry:
            first_tmp = self._stage_first_entry(
                active, payload, new_prev_hash, saved_chain_state[0]
            )
        try:
            if first_tmp is not None:
                os.replace(first_tmp, active)
                _fsync_dir(active.parent)
            else:
                with open(active, "a", encoding="utf-8") as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
            # Update chain state
            self._prev_hash = new_prev_hash
            self._seq += 1
            self._active_has_entry = True
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
            # A staged first entry still on disk under its own name was never
            # renamed in, so the active file was not touched and there is
            # nothing to truncate (KL-24 L3 r6, codex 8 + complement 2, run: the
            # truncate's open failed on an absent active file, the temp and its
            # record stayed, and the next append committed this failed entry).
            not_renamed = first_tmp is not None and os.path.lexists(first_tmp)
            try:
                if not not_renamed:
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
                _log(logging.WARNING,
                    "audit rollback failed for seq %d; the aborted entry "
                    "may still be on disk, so the chain state has NOT "
                    "been rewound — the next append re-derives "
                    "seq/prev_hash from the file rather than reusing "
                    "this seq",
                    entry["seq"],
                    exc_info=True,
                )
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
                self._active_has_entry = saved_has_entry
                self._tip = saved_tip
                self._initialized = True
                if not had_entry:
                    # The record names an entry that is not on disk. A staged
                    # temp that was not renamed in is set aside, never deleted.
                    if not_renamed and first_tmp is not None:
                        _set_aside(first_tmp, _DISCARDED_REASON)
                    self._withdraw_active_begun(new_prev_hash)
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

        # The new tip is where THIS entry was written, computed from the append
        # itself, not from a stat of the file afterwards: a stat would also take
        # in anything an unlocked writer appended since, and the next re-sync
        # would skip it (L2 r1, run). The stat is for the file's identity only.
        # A failed stat cannot say which file this is, so the next call
        # re-derives everything from disk instead.
        try:
            st_after = os.stat(active)
        except OSError:
            self._tip = None
            self._initialized = False
        else:
            self._tip = (
                st_after.st_dev,
                st_after.st_ino,
                resume_at + (1 if needs_boundary else 0),
                len(json_line.encode("utf-8")),
            )
            self._tip_week = _week_of_ts(ts)

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

        ``entry_count`` is read from disk on every call: the active file's last
        valid entry's ``seq`` + 1, or 0 when it holds none (``seq`` restarts in
        each week's file), so it includes what other writers appended. Does NOT
        walk the full hash chain — for integrity verification, call
        :meth:`verify`. A read error (other than no file) propagates as
        ``OSError``.

        ⛔ READ-ONLY, AND THAT IS THE FIX (L3 r1 on KL-24, codex + complement):
        a status read used to re-sync or initialise this instance's chain
        state. Done without the append lock it could adopt an entry a peer then
        rolled back (a false "is gone" at the next append); told not to touch a
        lost file it stayed stale for good; and initialising took the manifest
        lock, on which a writer holding the append lock could wait until other
        writers' waits ran out. It takes no lock and changes nothing now.
        An active file with no valid entry that the manifest says once held
        some raises ``OSError`` (see :meth:`_raise_if_active_lost`) rather than
        reading as 0 entries. So do a quarantined manifest, a staged first entry
        waiting to be resolved, and a directory that cannot be listed to rule
        those out, whatever the active file holds and whether or not a manifest
        is present (KL-24 L3 r6, codex 6, run: an active file with entries
        beside a quarantine marker read as a normal count, and a staged entry
        beside a manifest with no record as a healthy 0).

        Returns:
            Dict with keys ``log_path`` (str), ``entry_count`` (int),
            ``retention_days`` (int | None). Callers that also care
            about the enabled/disabled distinction should check for
            ``None`` at the ``Store._audit`` level before calling this.
        """
        audit_dir, stem = self._db_path.parent, self._db_path.stem
        try:
            names = [p.name for p in audit_dir.iterdir()]
        except FileNotFoundError:
            names = []
        except OSError as e:
            raise OSError(f"the audit directory cannot be listed: {e}") from e
        if self._first_entry_path().name in names:
            names = self._names_after_staged_window(audit_dir, stem, names)
        if _markers_in(names, stem):
            raise OSError("the audit manifest is quarantined; run `anneal-memory audit-repair`")
        if self._first_entry_path().name in names:
            raise OSError(
                "a staged first audit entry is waiting to be finished or set aside "
                "by the next append or `anneal-memory audit-repair`"
            )
        try:
            with _open_regular(self._active_path) as f:
                last_line = _last_valid_entry_in(f, 0)[0]
        except FileNotFoundError:
            last_line = ""
        if not last_line:
            self._raise_if_active_lost(names)
        entry_count = json.loads(last_line)["seq"] + 1 if last_line else 0
        return {
            "log_path": str(self._active_path),
            "entry_count": entry_count,
            "retention_days": self._retention_days,
        }

    def _names_after_staged_window(
        self, audit_dir: Path, stem: str, names: list[str]
    ) -> list[str]:
        """For :meth:`stats` when a staged first entry is on disk: a peer's first
        append of the week stages it for an instant, holding the append lock. Wait
        for that lock (bounded, ``_STATS_STAGED_WAIT_SECONDS``), then list again;
        the staged file still there with no holder is the crash case and stays
        unknown (KL-24 L3 r7, complement MED 1). Writes nothing but the lock file
        the append already uses. Without advisory locks, or on a lock error, the
        original names stand."""
        if fcntl is None:
            return names
        try:
            fd = self._open_and_flock(
                audit_dir / f"{stem}.audit-append.lock",
                "audit append lock",
                "a staged first entry cannot be told from a crashed one",
                timeout=_STATS_STAGED_WAIT_SECONDS,
            )
        except OSError:
            return names
        if fd is not None:
            _unlock_and_close(fd)
        try:
            return [p.name for p in audit_dir.iterdir()]
        except OSError as e:
            raise OSError(f"the audit directory cannot be listed: {e}") from e

    def _raise_if_active_lost(self, names: list[str]) -> None:
        """For :meth:`stats` when the active file holds no valid entry: raise
        ``OSError`` if the manifest (``vanished_active_week``) or this instance
        says it once did, so ``Store.status()`` reports the counts as
        unknown instead of a healthy 0 (KL-24 L3 r2, codex MED). Read-only: no
        lock, no quarantine, nothing written. The same reading holds for the
        instant inside a peer's rotation between the rename and the manifest
        save, which reads as unknown, never as healthy."""
        manifest_path = self._db_path.parent / f"{self._db_path.stem}.audit.manifest.json"
        try:
            manifest = _parse_manifest_bytes(
                _read_regular_bytes(manifest_path), self._db_path.stem
            )
        except FileNotFoundError:
            manifest = {}
            stem = self._db_path.stem
            # Sealed weeks with no manifest: the trail is not empty, and nothing
            # here can say what it holds (L3 r4 complement LOW). ``names`` is
            # :meth:`stats`'s one listing, which has already refused a marker or
            # a staged entry. Names are matched exactly, never globbed (a stem
            # may hold glob characters: r5 complement LOW).
            sealed_re = re.compile(re.escape(f"{stem}.audit.") + r"\d{4}-W\d{2}\.jsonl(?:\.gz)?")
            if any(sealed_re.fullmatch(n) for n in names):
                raise OSError("the audit manifest is missing beside sealed audit files")
        except _CORRUPT_MANIFEST + (AttributeError,) as e:
            # Every way a manifest fails to parse reads as unknown, never as a
            # crash of the status call (L3 r3 ``[]``: TypeError; r4: RecursionError).
            raise OSError(f"the audit manifest cannot be read: {e}") from e
        try:
            begun = vanished_active_week(manifest)
        except (TypeError, KeyError, AttributeError) as e:
            raise OSError(f"the audit manifest cannot be read: {e}") from e
        if begun is not None or self._active_has_entry:
            # The manifest's record is best-effort; this instance's own
            # knowledge that the file held an entry counts too (L3 r3, codex
            # MED: with no record the lost trail read as a healthy 0).
            raise OSError(
                "the active audit file held entries and is gone or empty; run "
                "`anneal-memory audit-repair`"
            )

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
            # No listing, no manifest read: nothing establishes the anchor, so it
            # is not reported trusted (glm MED, review 9dfdcc21bc482a70).
            return AuditVerifyResult(
                valid=False, total_entries=0, files_verified=0,
                anchor_trusted=False,
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
        manifest's signature taken before that listing. Every result carries
        the manifest's ``set_aside`` record when the manifest was read."""
        set_aside: list[dict[str, str]] = []
        result = cls._verify_pass(db_path, names, manifest_signature, set_aside)
        return replace(result, set_aside=set_aside) if set_aside else result

    @classmethod
    def _verify_pass(
        cls,
        db_path: Path,
        names: set[str],
        manifest_signature: tuple[int, int, int] | None,
        set_aside: list[dict[str, str]],
    ) -> AuditVerifyResult:
        """:meth:`_verify_listed`'s body; fills ``set_aside`` from the manifest."""
        stem = db_path.stem
        audit_dir = db_path.parent
        manifest_path = audit_dir / f"{stem}.audit.manifest.json"
        active_path = audit_dir / f"{stem}.audit.jsonl"

        # Load manifest (once) for file list + chain anchor
        files_to_verify: list[Path] = []
        chain_anchor = GENESIS_HASH
        missing_files: list[str] = []
        anchor_trusted = True

        # From the listing this pass already took, never a second one: round
        # 10's list-once rule, which a second listing broke by raising out of
        # verify() (complement + codex, L3 of the hybrid).
        markers = _markers_in(names, stem)
        if markers:
            return AuditVerifyResult(
                valid=False, total_entries=0, files_verified=0,
                # A quarantined manifest is not read, so its anchor is unknown
                # (glm MED, review 9dfdcc21bc482a70, reproduced: True was reported).
                anchor_trusted=False,
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
        begun: dict[str, str] | None = None
        if manifest_path.name in names:
            try:
                manifest = _parse_manifest_bytes(_read_regular_bytes(manifest_path), stem)
                # Chain anchor from retention cleanup — trust point for
                # chains that no longer start from GENESIS
                anchor = manifest.get("chain_anchor", "")
                if anchor:
                    chain_anchor = anchor
                anchor_trusted = manifest.get("chain_anchor_recovered") is not True
                set_aside.extend(dict(r) for r in manifest.get("set_aside", []))
                begun = vanished_active_week(manifest)
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
                    anchor_trusted=False,
                    error=f"Corrupt manifest: {e}",
                )

        # ⛔ A SEALED FILE THE MANIFEST DOES NOT COVER IS A GAP, NOT NOISE
        # (codex, round 9). Skipping a corrupt orphan left it on disk while
        # this method walked only the manifest, so it returned valid=True
        # over missing history, measured. The file on disk is the record.
        known = {p.name for p in files_to_verify} | set(missing_files)
        unmanifested = _unmanifested_sealed_names(names, stem, known, audit_dir)
        if unmanifested:
            return AuditVerifyResult(
                valid=False, total_entries=0, files_verified=0,
                anchor_trusted=anchor_trusted,
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
                anchor_trusted=anchor_trusted,
                error=(
                    "Missing sealed files referenced in manifest: "
                    f"{missing_files}{_RERUN_HINT}"
                ),
            )

        # ⛔ AN ACTIVE FILE THE MANIFEST SAYS HELD ENTRIES, NOW WITH NONE, IS A
        # LOSS NOBODY HAS ACKNOWLEDGED (1003+12 [run]: three entries deleted read
        # valid=True with 0 entries). Invalid until ``audit-repair`` records it as
        # a gap, as an unmanifested sealed week is.
        if begun is not None:
            try:
                has_entry = (
                    active_path.name in names and _first_prev_hash(active_path) is not None
                )
            except OSError:
                has_entry = True  # unreadable: the walk below reports it
            if not has_entry:
                return AuditVerifyResult(
                    valid=False, total_entries=0, files_verified=0,
                    anchor_trusted=anchor_trusted,
                    error=(
                        f"The active audit file {active_path.name} held entries in week "
                        f"{begun['period']} and now holds none; run `anneal-memory "
                        f"audit-repair` to record them as a gap{_RERUN_HINT}"
                    ),
                )

        if not files_to_verify:
            # ⛔ THE EMPTY-TRAIL VALID RETURN RE-CHECKS THE MANIFEST TOO (codex,
            # L3 re-pass of round 10b, reproduced with a simulated empty
            # listing): a listing that saw nothing while a first rotation
            # landed returned valid=True with 0 entries. Every valid=True
            # return goes through the signature check.
            if not _signatures_match(_stat_signature(manifest_path), manifest_signature):
                return AuditVerifyResult(
                    valid=False, total_entries=0, files_verified=0,
                    anchor_trusted=anchor_trusted,
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
                    anchor_trusted=anchor_trusted,
                    error=f"Cannot list audit directory: {e}",
                )
            appeared = sorted(
                n for n in fresh - names
                if n in (active_path.name, manifest_path.name) or _is_sealed_filename(n, stem)
            )
            if appeared:
                return AuditVerifyResult(
                    valid=False, total_entries=0, files_verified=0,
                    anchor_trusted=anchor_trusted,
                    error=f"Audit files appeared during verification: {appeared}{_RERUN_HINT}",
                )
            return AuditVerifyResult(
                valid=True, total_entries=0, files_verified=0, anchor_trusted=anchor_trusted
            )

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

        if not _signatures_match(_stat_signature(manifest_path), manifest_signature):
            return AuditVerifyResult(
                valid=False,
                total_entries=total_entries,
                files_verified=files_verified,
                skipped_lines=skipped,
                anchor_trusted=anchor_trusted,
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
    def repair_manifest(
        cls, db_path: str | Path, *, set_aside_unreadable: bool = False
    ) -> AuditRepairResult:
        """Rebuild a quarantined (or missing) manifest from the sealed files on disk.

        The ONLY way out of quarantine (hybrid, ruled by Phill 2026-09-13).
        Operator-run: do not run it while another process is writing the trail.

        Refuses, writing nothing, when the directory cannot be listed, when a
        sealed week is unreadable, empty or breaks its own chain, or when
        consecutive sealed files do not hash-chain — it never guesses
        across a gap. ``sha256_file`` is left empty rather than recomputed,
        because a checksum of the bytes now on disk would bless whatever
        changed. If the first sealed file, or with none left the active
        file's first entry, does not start at genesis, its starting hash is recorded as ``chain_anchor`` together with
        ``chain_anchor_recovered: true``, and :meth:`verify` then reports
        ``anchor_trusted=False``.

        Holds the append lock and then the manifest lock (``_append_lock``,
        ``_manifest_lock``, the documented order) from its first listing to its
        return, so no append runs while it decides, and a writer in another
        process cannot quarantine the rebuilt manifest while repair runs, or
        after it from bytes it read before (spore-1030). Where a lock cannot be
        taken for any reason other than the platform having none, repair
        refuses.

        A staged first entry (``<active>.first``, left by an append that
        stopped between saving its record and renaming it in) is resolved
        first, by the rule the next append would apply: a valid manifest whose
        record names it, with the active file still holding no entry, finishes
        the rename; otherwise it is set aside, never deleted (KL-24 L3 r6,
        complement 1 + codex 1, run: repair recorded a permanent gap over it and
        the next append deleted it). What was done is ``staged_first_entry``.
        """
        trail = cls(Path(db_path))
        try:
            with trail._append_lock(), trail._manifest_lock():
                return cls._repair_locked(trail, set_aside_unreadable)
        except _AuditLockError as e:
            return AuditRepairResult(repaired=False, error=f"{e}; nothing was written.")

    @classmethod
    def _repair_locked(
        cls, trail: "AuditTrail", set_aside_unreadable: bool = False
    ) -> AuditRepairResult:
        """:meth:`repair_manifest`'s body; the caller holds the manifest lock."""
        db_path = trail._db_path
        stem = db_path.stem
        audit_dir = db_path.parent
        # Every refusal ends with this. Once repair has quarantined the manifest
        # itself, "nothing was written" is false (codex, re-pass 598cd40ffcfcbc18).
        nothing = "nothing was written."
        # A listing error is a refusal, not a traceback (complement + codex, L3
        # of the hybrid, reproduced at mode 0o300).
        try:
            markers = _quarantine_markers(audit_dir, stem)
        except OSError as e:
            return AuditRepairResult(
                repaired=False,
                error=f"Cannot list the audit directory: {e}; nothing was written.",
            )

        current: dict[str, Any] | None = None
        if not markers and trail._manifest_path.exists():
            try:
                current = trail._load_manifest()
            except _ManifestQuarantined as e:
                # The markers _load_manifest saw or just created. Listing again
                # here could fail after the rename and report "nothing was
                # written" (codex, re-pass a927e791ce5df4eb).
                if not e.markers:
                    return AuditRepairResult(repaired=False, error=str(e))
                markers = e.markers
                nothing = (
                    # Another process may have quarantined it; this names what is
                    # on disk, not who wrote it (complement LOW, c7c73130c1022f53).
                    f"the manifest is quarantined as {', '.join(markers)}; "
                    "nothing else was written."
                )
            except _ManifestUnavailable as e:
                return AuditRepairResult(repaired=False, error=str(e))

        # Before any vanished-file or possible-gap decision: a staged first
        # entry is finished or set aside under the manifest repair holds (None
        # when it is quarantined or absent: then it is set aside).
        try:
            staged = trail._resolve_staged_entry(current)
        except OSError as e:
            return AuditRepairResult(
                repaired=False,
                error=f"Could not resolve the staged first audit entry: {e}; {nothing}",
            )
        if current is not None:
            result = cls._set_aside_unreadable_locked(trail, current, set_aside_unreadable)
        else:
            result = cls._rebuild_locked(trail, markers, nothing)
        if staged is None:
            return result
        if not result.repaired and result.error == _NOTHING_TO_REPAIR:
            return AuditRepairResult(repaired=True, staged_first_entry=staged)
        return replace(
            result,
            staged_first_entry=staged,
            error=(f"{result.error} (Before that, the staged first entry was {staged}.)"
                   if result.error else None),
        )

    @classmethod
    def _rebuild_locked(
        cls, trail: "AuditTrail", markers: list[str], nothing: str
    ) -> AuditRepairResult:
        """:meth:`_repair_locked`'s rebuild of a quarantined or missing manifest
        from the sealed files on disk; the caller holds both locks."""
        db_path = trail._db_path
        stem = db_path.stem
        audit_dir = db_path.parent
        try:
            by_period: dict[str, list[Path]] = {}
            for p in audit_dir.iterdir():
                if _is_sealed_filename(p.name, stem):
                    by_period.setdefault(_sealed_period(p.name, stem), []).append(p)

            records: list[dict[str, Any]] = []
            untracked: list[str] = []
            for period in sorted(by_period):
                # Every copy is scanned before one is chosen, so a copy that
                # was not chosen is listed as untracked however it read (codex,
                # L3 of the hybrid: an unreadable .gz scanned first was dropped
                # from ``untracked``). A copy that breaks its own chain is never
                # chosen (codex, same review: repair released the marker over a
                # week verify() rejected at once).
                candidates = sorted(by_period[period], key=lambda q: not q.name.endswith(".gz"))
                scanned = [(p, _sealed_record(p)) for p in candidates]
                usable = [
                    (p, i) for p, i in scanned
                    if i is not None and i["entries"] > 0 and i["chain_break_seq"] is None
                ]
                if not usable:
                    broken = [(p, i) for p, i in scanned if i is not None and i["chain_break_seq"] is not None]
                    names = sorted(q.name for q in by_period[period])
                    return AuditRepairResult(
                        repaired=False,
                        error=(
                            f"{broken[0][0].name} does not hash-chain internally at seq "
                            f"{broken[0][1]['chain_break_seq']}; {nothing}"
                            if broken else
                            f"No readable entry in the sealed file(s) for {period} "
                            f"({names}); {nothing}"
                        ),
                    )
                path, info = usable[0]
                untracked.extend(p.name for p, _ in scanned if p != path)
                records.append({"path": path, "period": period, **info})
        except OSError as e:
            return AuditRepairResult(
                repaired=False, error=f"Could not read the sealed files: {e}; {nothing}"
            )

        for prev, cur in zip(records, records[1:]):
            if cur["first_prev_hash"] != prev["last_hash"]:
                return AuditRepairResult(
                    repaired=False,
                    error=(
                        f"{cur['path'].name} does not chain from {prev['path'].name}; "
                        f"{nothing}"
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
        if records:
            anchor = records[0]["first_prev_hash"]
        else:
            # No sealed file survives (glm, L3 of the hybrid, reproduced: the
            # rebuilt manifest anchored at genesis and verify() reported a hash
            # mismatch at seq 0). The active file's first entry is then the
            # only record of where the chain starts.
            try:
                anchor = _first_prev_hash(trail._active_path) or GENESIS_HASH
            except FileNotFoundError:
                anchor = GENESIS_HASH
            except OSError as e:
                return AuditRepairResult(
                    repaired=False,
                    error=f"Could not read the active audit file: {e}; {nothing}",
                )
        recovered = anchor != GENESIS_HASH
        if recovered:
            manifest["chain_anchor"] = anchor
            manifest["chain_anchor_recovered"] = True
        # The quarantined manifest's set_aside record is not readable here, but
        # every file repair set aside is still on disk under its set-aside name,
        # so the gap stays recorded through a rebuild.
        try:
            carried = _set_aside_records_on_disk(audit_dir, stem)
        except OSError as e:
            return AuditRepairResult(
                repaired=False, error=f"Cannot list the audit directory: {e}; {nothing}"
            )
        # ⛔ A REBUILD CANNOT VOUCH FOR AN EMPTY ACTIVE FILE (Phill 2026-10-08,
        # "A"; run first: the active file deleted while the manifest was
        # quarantined, and after this rebuild verify read VALID with that week's
        # entries gone and no gap). The record that would say the file once held
        # entries was in the quarantined manifest. With no usable entry in the
        # active file the rebuild records an UNKNOWN gap: loud, and possibly a
        # false one when the file really never held an entry.
        try:
            active_holds_entry = _first_valid_line(trail._active_path) is not None
        except FileNotFoundError:
            active_holds_entry = False
        except OSError as e:
            return AuditRepairResult(
                repaired=False,
                error=f"Could not read the active audit file: {e}; {nothing}",
            )
        possible_gap: list[dict[str, Any]] = []
        if not active_holds_entry:
            period = _iso_week_now()
            try:
                preserved = trail._discarded_staged_names(period)
            except OSError as e:
                return AuditRepairResult(
                    repaired=False,
                    error=f"Cannot list the audit directory: {e}; {nothing}",
                )
            possible_gap = [{
                "filename": trail._active_path.name,
                "set_aside_as": "",
                # Explicit, so no reader infers it from the cause text (KL-24 L3
                # r6, codex 10, run: the CLI and verify printed it as a definite
                # GAP, "went missing with its entries").
                "certainty": _POSSIBLE,
                "period": period,
                "cause": (
                    "the manifest was rebuilt from quarantine and the active file holds "
                    "no entry; whether it held entries before cannot be known, so this "
                    "is recorded as a possible gap"
                ),
                "at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
                **({"preserved_attempts": preserved} if preserved else {}),
            }]
            carried = list(carried or []) + possible_gap
        if carried:
            manifest["set_aside"] = carried
        try:
            trail._save_manifest(manifest)
            saved_signature = _stat_signature(trail._manifest_path)
        except OSError as e:
            # codex, L3 of the hybrid: a failed save escaped as a traceback.
            # The markers are untouched, so the trail stays quarantined.
            return AuditRepairResult(
                repaired=False,
                error=f"Could not save the rebuilt manifest: {e}; the quarantine markers are kept.",
            )

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

        # ``markers`` is a snapshot from before the rebuild. A reader that parsed
        # the old invalid bytes can quarantine the manifest just saved, and repair
        # then returned repaired=True over a trail verify() rejected (codex HIGH,
        # re-pass c7c73130c1022f53, reproduced by injection). A marker only ever
        # comes from renaming the manifest, so the rebuilt manifest's signature,
        # unchanged since the save, rules one out without another listing (a
        # listing here is what test_repair_does_not_relist_after_quarantining
        # forbids).
        if saved_signature is None or not _signatures_match(
            _stat_signature(trail._manifest_path), saved_signature
        ):
            return AuditRepairResult(
                repaired=False,
                files=[r["path"].name for r in records],
                chain_anchor_recovered=recovered,
                untracked=untracked,
                error=(
                    "The rebuilt manifest was changed or quarantined again while repair "
                    "ran; run `anneal-memory audit-repair` again."
                ),
            )

        return AuditRepairResult(
            repaired=True,
            files=[r["path"].name for r in records],
            chain_anchor_recovered=recovered,
            untracked=untracked,
            # The possible gap is reported to whoever ran the repair, not only
            # recorded where a later verify finds it.
            set_aside=possible_gap,
        )

    @classmethod
    def _set_aside_unreadable_locked(
        cls, trail: "AuditTrail", manifest: dict[str, Any],
        set_aside_unreadable: bool = False,
    ) -> AuditRepairResult:
        """With a valid manifest: set aside every unmanifested sealed week none
        of whose copies can be read, and record each in the manifest's
        ``set_aside`` list (Phill, 2026-10-03). The caller holds the lock.

        ⛔ CORRUPT ONLY, UNLESS ASKED (Phill, 2026-10-03: "corrupt-only by
        default plus the flag"). A set-aside is one-way once a write follows,
        so a week that failed only to READ (permissions, I/O) is not set aside
        unless ``set_aside_unreadable``: repair refuses, writing nothing, and
        names the error, because fixing access lets writes resume with no gap.

        ⛔ RECORDED BEFORE IT IS MOVED, AND NEVER DELETED. A file renamed with
        no record would be a gap nothing reports. So the manifest naming the
        file's new name is saved first and the rename follows; a crash in
        between leaves the file on its sealed name, which ``verify()`` reports
        as unmanifested and the next repair moves (dropping the stale record
        first). The new name, ``<name>.unreadable-<UTC stamp>``, is outside
        the sealed-file language, so adoption and ``verify()``'s unmanifested
        check no longer see it, and writes continue from the manifest's tip.
        """
        db_path = trail._db_path
        stem = db_path.stem
        audit_dir = db_path.parent
        prefix = f"{stem}.audit."
        try:
            names = sorted(p.name for p in audit_dir.iterdir())
        except OSError as e:
            return AuditRepairResult(
                repaired=False,
                error=f"Cannot list the audit directory: {e}; nothing was written.",
            )
        manifested = {_week_of(f["filename"], prefix) for f in manifest["files"]}
        by_week: dict[str, list[Path]] = {}
        for name in names:
            if _is_sealed_filename(name, stem) and _week_of(name, prefix) not in manifested:
                by_week.setdefault(_week_of(name, prefix), []).append(audit_dir / name)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        new: list[dict[str, str]] = []
        not_corrupt: list[str] = []
        # (week, first entry hash, the hash it chained from) of each readable
        # unmanifested week with an entry: repair's view of what adoption sees
        readable_weeks: list[tuple[str, str, str | None]] = []
        for week, paths in sorted(by_week.items()):
            scans = {path: trail._scan_sealed(path) for path in paths}
            if any(scan.error is None for scan in scans.values()):
                readable_weeks.extend(
                    (week, scan.first_hash, scan.first_prev_hash)
                    for scan in scans.values() if scan.error is None and scan.entries > 0
                )
                continue  # a readable copy is adoption's to take, not repair's
            if not set_aside_unreadable and not all(
                isinstance(scan.error, _CorruptAuditFile) for scan in scans.values()
            ):
                not_corrupt.extend(
                    f"{path.name} ({scans[path].error})" for path in paths
                    if not isinstance(scans[path].error, _CorruptAuditFile)
                )
                continue
            for path in paths:
                new.append({
                    "filename": path.name,
                    "set_aside_as": f"{path.name}.{_UNREADABLE_REASON}-{stamp}",
                    "period": week,
                    "cause": str(scans[path].error),
                    "at": stamp,
                })
        if not_corrupt:
            return AuditRepairResult(
                repaired=False,
                error=(
                    "Sealed audit file(s) newer than the manifest cannot be read, and "
                    f"are not known to be corrupt: {'; '.join(not_corrupt)}. Fix the "
                    "access (e.g. chmod) or the disk and retry: writes then resume with "
                    "no gap. To set them aside anyway, run `anneal-memory audit-repair "
                    "--set-aside-unreadable`. Nothing was written."
                ),
            )
        if new and trail._active_path.name in names:
            try:
                active_prev = _first_prev_hash(trail._active_path)
            except OSError as e:
                return AuditRepairResult(
                    repaired=False,
                    error=f"Cannot inspect the active audit file: {e}; nothing was written.",
                )
            if active_prev is not None:  # None: no valid entry, the refusal's own case
                # The refusal this repair answers only fires with no usable
                # active file. With entries in it, the chain may already run
                # through the week, and moving it would leave a gap in the
                # middle of the chain (L3 r1 10-03, complement).
                return AuditRepairResult(
                    repaired=False,
                    error=(
                        "The active audit file holds entries, so the chain may run "
                        f"through {', '.join(r['filename'] for r in new)}; not setting "
                        "it aside. verify reports it. Nothing was written."
                    ),
                )
        vanished: dict[str, Any] | None = None
        begun = vanished_active_week(manifest)
        if (
            begun is not None
            and len({w for w, _, _ in readable_weeks}) == 1
            and any(
                h == begun["first_hash"]
                and prev == manifest.get("active_last_hash", GENESIS_HASH)
                for _, h, prev in readable_weeks
            )
        ):
            # A readable sealed week whose first entry IS the recorded active
            # file's first entry is that file, renamed by a rotation that crashed
            # before its manifest save; the next open adopts it, entries and all,
            # so recording a gap would be false and permanent (L3 r1 10-03,
            # complement, run). Matched by first entry, not by the hash it chained
            # from: an unrelated file can share that (L3 r2 10-03, codex HIGH, run).
            # Only the shape a single writer leaves: one readable week, continuing
            # the manifest's ``active_last_hash``. With a second readable week
            # adoption may take that one and reject this, and a suppressed gap
            # then refused every write with no way out (L3 r3 10-03, codex MED,
            # run); the gap is recorded instead. This does NOT follow adoption's
            # own copy choice or tip (L3 r4 10-03: differing .jsonl/.gz copies of
            # the week, a manifest rebuilt with no sealed files, a chain of orphan
            # weeks); those need outside damage, and the CHANGELOG names them.
            begun = None
        stale_cleared = False
        if begun is not None:
            # A record of a file that IS sealed and manifested (an older release
            # sealed it and left the field set): matched by the sealed file's
            # first entry, never by period alone (KL-24 L3 r4, codex MED: repair
            # recorded a false, permanent gap for entries still in the chain).
            for f in manifest.get("files", []):
                if f.get("period") != begun["period"]:
                    continue
                path = audit_dir / str(f.get("filename", ""))
                # Absent, not a regular file, and unreadable are three answers
                # (KL-24 L3 r6, codex 9, run: ``is_file()`` read a directory at
                # the sealed name as absent, and repair recorded a permanent gap
                # instead of refusing). Only absent is a mismatch.
                try:
                    st = os.lstat(path)
                    if not stat.S_ISREG(st.st_mode):
                        raise OSError(errno.EINVAL, "not a regular file", str(path))
                    first = _first_valid_line(path)
                except FileNotFoundError:
                    first = None
                except OSError as e:
                    # Unreadable is not a mismatch (L3 r5, codex MED): reading
                    # it as one recorded a false permanent gap.
                    return AuditRepairResult(
                        repaired=False,
                        error=(f"Cannot read the sealed week {path.name} to check it "
                               f"against the manifest's active-file record: {e}; "
                               "nothing was written."),
                    )
                if first is not None and AuditTrail._compute_hash(first) == begun["first_hash"]:
                    manifest["active_begun"] = None
                    begun = None
                    stale_cleared = True
                    break
        if begun is not None and any(r["period"] == begun["period"] for r in new):
            # The same, unreadable: setting it aside records the gap, and a second
            # record would count it twice. Only the week the record names: an
            # unrelated corrupt week must not clear it (L3 r1 10-03, codex + glm,
            # run). A deletion on top of such a crash reads as one gap, not two.
            manifest["active_begun"] = None
        elif begun is not None:
            try:
                active_entry = (
                    trail._active_path.name in names
                    and _first_prev_hash(trail._active_path) is not None
                )
            except OSError as e:
                return AuditRepairResult(
                    repaired=False,
                    error=f"Cannot inspect the active audit file: {e}; nothing was written.",
                )
            if not active_entry:
                # ``set_aside_as`` "" marks it: there is no file to move. A vanished
                # active file is a definite loss whatever else is on disk; set-aside
                # staged entries are listed as pointers only (KL-24 L3 r7, r8: every
                # rule that let a directory listing change the certainty drew a new
                # finding, so none remains).
                try:
                    preserved = trail._discarded_staged_names(begun["period"])
                except OSError as e:
                    return AuditRepairResult(
                        repaired=False,
                        error=f"Cannot list the audit directory: {e}; nothing was written.",
                    )
                vanished = {
                    "filename": trail._active_path.name,
                    "set_aside_as": "",
                    "period": begun["period"],
                    "cause": (
                        "the active file was deleted or emptied after its first entry "
                        f"(hash {begun['first_hash']}) was recorded"
                    ),
                    "at": stamp,
                }
                if preserved:
                    vanished["preserved_attempts"] = preserved
        if not new and vanished is None and not stale_cleared:
            return AuditRepairResult(repaired=False, error=_NOTHING_TO_REPAIR)
        taken = [r["set_aside_as"] for r in new if os.path.lexists(audit_dir / r["set_aside_as"])]
        if taken:
            # os.rename replaces an existing file on POSIX; recovery never does.
            return AuditRepairResult(
                repaired=False,
                error=f"A set-aside name is already taken ({taken}); nothing was written.",
            )
        # ⛔ A RECORD IS THE ONLY EVIDENCE OF A GAP, SO REPAIR NEVER DROPS ONE
        # BECAUSE ITS FILE IS MISSING (L3 r2 10-03, codex HIGH + complement: a
        # moved or lost set-aside file then erased the gap from verify). Only
        # the record of a file being moved again is replaced; adoption drops
        # the record of a week it proves into the chain.
        moving = {r["filename"] for r in new}
        kept = [
            r for r in manifest.get("set_aside", [])
            if not (r["filename"] in moving and r["set_aside_as"] not in names)
        ]
        manifest["set_aside"] = kept + new + ([vanished] if vanished else [])
        if vanished:
            manifest["active_begun"] = None
        try:
            trail._save_manifest(manifest)
        except OSError as e:
            return AuditRepairResult(
                repaired=False, error=f"Could not save the manifest: {e}; nothing was moved."
            )
        moved: list[dict[str, str]] = []
        for record in new:
            try:
                os.rename(audit_dir / record["filename"], audit_dir / record["set_aside_as"])
            except OSError as e:
                _fsync_dir(audit_dir)
                return AuditRepairResult(
                    repaired=False,
                    set_aside=moved,  # only what was actually moved (L3 r1, glm)
                    error=(
                        f"Recorded {record['filename']} as set aside, but could not move it: "
                        f"{e}; run `anneal-memory audit-repair` again."
                    ),
                )
            moved.append(record)
            _emit_warning(
                f"Set aside unreadable audit file {record['filename']} as "
                f"{record['set_aside_as']} ({record['cause']})"
            )
        _fsync_dir(audit_dir)
        if vanished:
            _emit_warning(
                f"Recorded the missing active audit file {vanished['filename']} "
                f"({vanished['period']}) as a "
                + "gap"
            )
        return AuditRepairResult(
            repaired=True, set_aside=new + ([vanished] if vanished else [])
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

    @contextmanager
    def _manifest_lock(self) -> Iterator[bool]:
        """Hold the cross-process lock that serializes every change to the
        manifest PATH: a save, a quarantine rename, and a whole repair.

        ⛔ INVARIANT: never call a user callback (``on_event``) or
        ``repair_manifest`` inside a locked span; a second instance in the
        same thread would block on this lock and deadlock (L2).

        ⛔ WHY (spore-1030, reproduced with two real processes): a writer that
        parsed the old invalid manifest renamed whatever stood at the manifest
        path by the time it got there, which could be the manifest repair had
        just rebuilt. Repair returned ``repaired=True`` and ``verify()`` then
        reported the trail quarantined. Under this lock a quarantine re-reads
        the manifest and renames only bytes it parsed as invalid while holding
        it, and a repair holds it from its first listing to its return.

        The lock file is ``<stem>.audit-manifest.lock`` in the directory the
        audit files live in (``flock`` keys on the file's inode, so two
        spellings of that directory take one lock). Its name is deliberately
        outside ``<stem>.audit.*``, the set the CLI compares before and after a
        read to detect a change in the trail. A symlink, FIFO or directory at
        that path is refused (``_AuditLockError``), never followed or degraded.

        Yields ``True`` when held. Yields ``False``, with no lock, where advisory
        locking does not exist: no ``fcntl`` (Windows, silently, as the README
        documents), or ``flock`` raising an errno in ``_LOCK_UNAVAILABLE_ERRNOS``
        (a warning on stderr, once per lock path). That is the behaviour before the lock
        existed. Forking while a trail is in use is unsupported (see the class
        docstring). Any other failure to open or lock raises
        ``_AuditLockError``. Who takes it: each public operation, ONCE, for
        everything it does to the manifest (10-03): ``log()`` through
        ``_operation_span`` on every append (KL-24 L3 r6: the preflight; it was
        only when initializing or rotating), the first-entry record save and its
        withdrawal in their own spans, and ``repair_manifest`` (inside the append
        lock). ``stats()`` takes no lock. Saves, quarantines, adoption, rotation and
        retention require it (``_require_lock``) and never take it, so it is
        held from an operation's first load to its last save, before any
        irreversible step, and never nested (a nested take raises
        ``RuntimeError``). Blocking, with no timeout: a holder waits on no other lock, but a repair
        or a rotation holds it while it reads or compresses sealed files, and a
        stopped holder blocks other processes' rotations and quarantines until
        it exits. Two ``AuditTrail``
        instances for one database in ONE thread must not nest it: the inner
        one opens its own descriptor and blocks on the outer (an earlier draft
        of this fix's own test did exactly that and hung).
        """
        if self._lock_owner == threading.get_ident():
            raise RuntimeError(
                "the audit manifest lock is already held by this operation; an "
                "internal must require it (_require_lock), not take it again"
            )
        with self._lock_mutex:
            fd = self._open_and_flock() if fcntl is not None else None
            self._lock_fd = fd
            self._lock_owner = threading.get_ident()
            try:
                yield fd is not None
            finally:
                self._lock_owner = None
                self._lock_fd = None
                if fd is not None:
                    _unlock_and_close(fd)

    @contextmanager
    def _operation_span(self) -> Iterator[None]:
        """Hold the manifest lock across one public operation's work on the
        manifest. If it cannot be taken, run the body anyway with the error
        recorded for this thread: every internal that requires the lock then
        raises that ``_AuditLockError`` and refuses exactly as it did when it
        took the lock itself (adoption skipped, rotation and retention not
        run, a quarantine and a save refused)."""
        me = threading.get_ident()
        # ⛔ REFUSE NESTING WHETHER OR NOT THE LOCK WAS TAKEN (L2 10-03, probe
        # run): _manifest_lock's own guard sees only an owner, so after a failed
        # acquisition a nested span (a logging handler calling back into this
        # trail) got through and its exit popped the outer span's recorded
        # error, turning the outer refusal into a RuntimeError.
        if me in self._span_threads:
            raise RuntimeError(
                "an audit operation is already in progress on this thread; "
                "the trail is not reentrant (e.g. from a logging handler)"
            )
        self._span_threads.add(me)
        try:
            with ExitStack() as stack:
                try:
                    stack.enter_context(self._manifest_lock())
                except _AuditLockError as e:
                    self._lock_failures[me] = e
                try:
                    yield
                finally:
                    self._lock_failures.pop(me, None)
        finally:
            self._span_threads.discard(me)

    def _require_lock(self) -> None:
        """For an internal that changes the manifest path: the caller's
        operation must hold the manifest lock. Raises the operation's recorded
        ``_AuditLockError`` when its acquisition failed, and ``RuntimeError``
        when no operation took it (a programming error, loud on purpose)."""
        me = threading.get_ident()
        if self._lock_owner == me:
            return
        failure = self._lock_failures.get(me)
        if failure is not None:
            raise _AuditLockError(str(failure)) from failure
        raise RuntimeError(
            "the audit manifest lock is not held: take it with an operation span "
            "(log(), repair_manifest()) or _manifest_lock() before this call"
        )

    @contextmanager
    def _append_lock(self) -> Iterator[None]:
        """Hold the cross-process lock that serializes appends to this trail
        (KL-24): ``log()`` holds it from its re-sync to its last manifest save.

        ⛔ WHY (KL-24, run 2026-10-07): three processes, each with its own
        ``Store(audit=True)`` on one database, recorded 600 episodes; every
        episode landed, ``verify()`` returned valid=False with a hash mismatch,
        and ``status()`` reported 0 audit write failures. Each instance chained
        from its own cached tip, so its entries skipped every peer entry
        written since its previous one.

        The lock file is ``<stem>.audit-append.lock``, beside the manifest lock
        and like it outside ``<stem>.audit.*``. ⛔ LOCK ORDER: this lock is
        taken FIRST and the manifest lock, when an operation needs it, inside
        it. Nothing that holds the manifest lock takes this one;
        ``repair_manifest`` takes this one first and then the manifest lock
        (KL-24 L3 r6, so no append races its staged-entry and gap decisions),
        so the two never wait on each other in opposite orders. Not reentrant: ``log()`` refuses a nested
        call on the same thread before taking it, and two instances for one
        database must not nest it in one thread (the inner one waits on the
        outer, as with the manifest lock).

        What a failure does, which is NOT what the manifest lock's does:
        - advisory locks unavailable: no ``fcntl`` (Windows, silently, as the
          README's Windows section says), or ``flock`` raising an errno in
          ``_LOCK_UNAVAILABLE_ERRNOS`` (warned on stderr once per lock path).
          Not held, and appends are as unserialized as they were before it
          existed;
        - the lock file cannot be opened or is not a regular file, or another
          holder keeps it past ``_APPEND_LOCK_TIMEOUT_SECONDS`` (a stopped or
          hung process): ``_AuditLockError``, so the append is REFUSED. The
          store counts it as a dropped audit write (``audit_write_failures``,
          ``dropped_before`` on the next entry), so a wedged lock is loud and
          bounded rather than an unserialized append or a hang.
        Released with ``LOCK_UN`` before the close, so a child forked while it
        was held does not keep it (L2 r1, run: another writer waited out the
        child's whole lifetime).
        """
        fd = None
        try:
            if fcntl is not None:
                try:
                    fd = self._open_and_flock(
                        self._db_path.parent / f"{self._db_path.stem}.audit-append.lock",
                        "audit append lock",
                        "concurrent audit writers are not serialized and can break the hash chain",
                        timeout=_APPEND_LOCK_TIMEOUT_SECONDS,
                    )
                except BaseException:
                    # Without the lock this call never re-synced, so the cached
                    # tip says nothing about where the dropped entry belonged:
                    # cleared, ``note_write_failure`` reports the location as
                    # unknown rather than a seq another writer already used
                    # (L3 r1, codex), and the next call re-derives from disk.
                    # ``BaseException`` (L3 r2, codex MED): an interrupt while
                    # polling for the lock is the same case. Only acquisition
                    # is inside this handler; the protected body is not.
                    self._initialized = False
                    raise
            yield
        finally:
            if fd is not None:
                _unlock_and_close(fd)

    def _resync_with_disk(self) -> None:
        """Bring the cached chain tip up to date with the active file.

        :meth:`log` calls it under :meth:`_append_lock`; nothing else does.

        With a tip in the active file (``self._tip``): if that file is still
        the same one AND the tip's bytes still hash to ``_prev_hash``, the
        chain continues from the last valid entry after the tip (another
        writer's), or from the tip itself when nothing valid follows. Any
        other answer (no file, another inode, shorter than the tip, different
        bytes there) means the cache says nothing reliable: ``_initialized``
        is cleared and the caller re-derives everything through
        :meth:`_initialize`, which also refuses a deleted active file.
        Without a tip (the chain's tip is in the manifest or a sealed file):
        any valid entry now in the active file was written by someone else,
        so the same full re-derivation runs.

        ⛔ A LOST ACTIVE FILE IS REFUSED HERE, NOT ONLY BY THE MANIFEST (L1 r1,
        run): when the file holding the tip is gone, emptied below the tip, or
        replaced, this raises ``_ManifestUnavailable`` once, whether a peer
        sealed it or it was lost (see :meth:`_lost_active`). The manifest's
        ``active_begun`` record refuses a loss too, but that record is
        best-effort, and without it the re-derivation continued the chain over
        the lost entries with nothing counted.

        ⛔ THE BYTES ARE CHECKED, NOT A (dev, inode, size) SIGNATURE (L2 r1,
        run with a simulated inode): a rotation frees the inode, the new
        active file can get the same number back (ext4, xfs), and a size that
        happens to be larger sent the read to an offset in a different file.
        """
        if not self._initialized:
            return  # _initialize reads the file itself
        active = self._active_path
        try:
            f_cm = _open_regular(active)
        except FileNotFoundError:
            if self._tip is not None:
                self._lost_active("deleted")
            return
        with f_cm as f:
            if self._tip is None:
                if _last_valid_entry_in(f, 0)[0]:
                    self._initialized = False
                return
            dev, ino, at, length = self._tip
            st = os.fstat(f.fileno())
            if (st.st_dev, st.st_ino) != (dev, ino):
                self._lost_active("replaced")
                return
            if st.st_size < at + length:
                self._lost_active("emptied or truncated")
                return
            f.seek(at)
            try:
                tip_line = f.read(length).decode("utf-8").strip()
            except UnicodeDecodeError:
                tip_line = ""
            if not tip_line or self._compute_hash(tip_line) != self._prev_hash:
                self._initialized = False
                return
            last_line, last_at, last_len = _last_valid_entry_in(f, at + length)
        if last_line:
            last_entry = json.loads(last_line)  # Guaranteed valid by helper
            # ⛔ ``_seq``, THEN ``_prev_hash``, THEN ``_tip`` (L1 r1, run). An
            # interrupt after ``_seq`` alone leaves the old tip still hashing to
            # the old ``_prev_hash``, so the next call adopts the same entry
            # again, and ``note_write_failure`` meanwhile reports the right next
            # seq. After ``_prev_hash`` too, the old tip no longer hashes to it,
            # so the next call re-derives from disk. The reverse order left
            # ``_seq`` stale, and a dropped write was located at a seq already
            # on disk.
            self._seq = last_entry["seq"] + 1
            self._prev_hash = self._compute_hash(last_line)
            self._tip_week = _week_of_ts(last_entry["ts"]) or self._tip_week
            self._tip = (dev, ino, last_at, last_len)

    def _lost_active(self, what: str) -> None:
        """The active file holding this instance's tip was ``what``: refuse
        once, and re-derive from disk on the next call (see
        :meth:`_resync_with_disk`).

        ⛔ EVERY LOST FILE IS REFUSED, A PEER'S ROTATION INCLUDED (KL-24 L3 r2,
        ruled by Phill 2026-10-08, option (b)). This used to return quietly when
        a sealed file, its ``.gz`` or a ``.gz.tmp`` for the tip's week existed,
        reading a filename as proof that another writer sealed this file. A
        stale same-week orphan or temp then hid a deleted active file (codex
        HIGH), and retention removing a just-sealed week produced a false "is
        gone" (complement MED). Deleted, not repaired: a peer's rotation now
        costs this instance one refused, counted append, and a real loss is
        never followed silently."""
        self._initialized = False
        raise _ManifestUnavailable(
            f"the active audit file {self._active_path.name} this process was "
            f"appending to is gone ({what}; another process may have sealed its "
            f"week {self._tip_week or self._last_week or '?'}); not continuing the "
            "chain past it. The next write re-reads the trail from disk. If the "
            "file was lost rather than sealed, put it back and retry, or run "
            "`anneal-memory audit-repair` to record the week as a gap"
        )

    def _open_and_flock(
        self,
        lock_path: Path | None = None,
        label: str = "audit manifest lock",
        consequence: str = "audit manifest changes are not serialized across processes",
        timeout: float | None = None,
    ) -> int | None:
        """Open the lock file and take ``LOCK_EX`` on it; ``None`` when advisory
        locks are unavailable here (warned on stderr once per lock path). See
        :meth:`_manifest_lock` (the default lock) and :meth:`_append_lock`."""
        assert fcntl is not None
        if lock_path is None:
            lock_path = self._db_path.parent / f"{self._db_path.stem}.audit-manifest.lock"
        # O_RDWR first: on Linux NFS an exclusive lock needs a descriptor open for
        # writing (L3: complement + codex). O_RDONLY is the fallback for a lock
        # file this user may not write (another user's, or 0444 under a umask),
        # where local filesystems lock it anyway (L1 + L2, reproduced: O_RDWR
        # alone refused it and rotation failed mid-way). O_NOFOLLOW + O_NONBLOCK
        # + the regular-file check: a symlink, FIFO or directory planted at the
        # path is refused, not followed, waited on, or silently degraded to no
        # lock (L2, reproduced: a FIFO made ``flock`` report ENOTSUP).
        flags = os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
        try:
            try:
                fd = os.open(lock_path, os.O_RDWR | flags, 0o644)
            except PermissionError:
                fd = os.open(lock_path, os.O_RDONLY | flags, 0o644)
        except OSError as e:
            raise _AuditLockError(
                f"cannot open the {label} {lock_path}: {e}"
            ) from e
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise _AuditLockError(
                    f"the {label} {lock_path} is not a regular file"
                )
            # ENOLCK is also what a lock table out of records returns, which is
            # transient; Linux NFS without lock support returns it for good. A
            # few short retries separate the two before degrading (L3: codex).
            # With a ``timeout`` the wait is polled (LOCK_NB) and bounded; a
            # holder past it raises ``_AuditLockError`` (see _append_lock).
            op = fcntl.LOCK_EX if timeout is None else fcntl.LOCK_EX | fcntl.LOCK_NB
            deadline = None if timeout is None else time.monotonic() + timeout
            enolck_retries = 0
            delay = 0.002
            while True:
                try:
                    fcntl.flock(fd, op)
                    return fd
                except OSError as e:
                    if deadline is not None and e.errno in _LOCK_HELD_ERRNOS:
                        if time.monotonic() >= deadline:
                            raise _AuditLockError(
                                f"timed out after {timeout:g}s waiting for the {label} "
                                f"{lock_path.name}: another process holds it"
                            ) from e
                        time.sleep(delay)
                        delay = min(delay * 2, 0.05)
                        continue
                    if e.errno != errno.ENOLCK or enolck_retries == _ENOLCK_RETRIES:
                        raise
                    enolck_retries += 1
                    time.sleep(_ENOLCK_RETRY_SECONDS)
        except BaseException as e:
            # Any exception, an interrupt included, closes the descriptor.
            os.close(fd)
            if isinstance(e, _AuditLockError) or not isinstance(e, OSError):
                raise
            if e.errno not in _LOCK_UNAVAILABLE_ERRNOS:
                raise _AuditLockError(
                    f"cannot lock the {label} {lock_path.name}: {e}"
                ) from e
        # ⛔ STDERR, NOT ONLY THE LOGGER (L2 MED "silent degrade", ruled 10-03:
        # degrade with a stderr warning). Reproduced: an application that sends
        # its logging to a file showed nothing, and a second store in the same
        # process was silent because the warning fired once per process. Now
        # once per lock path, on stderr and to the logger.
        # Recorded BEFORE emitting, then emitted to the logger AND stderr (an app
        # may keep only one of them), each on its own so that neither a raising
        # log handler nor a broken stderr can turn the degrade into a failure.
        if str(lock_path) not in _lock_degrade_warned:
            _lock_degrade_warned.add(str(lock_path))
            message = (
                f"Advisory locks are unavailable for {lock_path}; the {label} "
                f"is NOT held, so {consequence}."
            )
            _emit_warning(message, stderr=True)
        return None

    def _initialize(self) -> None:
        """Lazy init: recover seq and prev_hash from existing audit file.

        Sets _initialized only after all recovery steps complete. If any
        step raises (disk full, permission error during orphan adoption),
        the next log() call retries init instead of writing with broken state.
        """
        # Adopt orphaned sealed files — crash between rename and manifest
        # update during rotation leaves .gz files the manifest doesn't know about.
        adopted = self._adopt_orphaned_files()

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
            self._seed_from_manifest(adopted=adopted)
            self._last_week = _iso_week_now()
            self._initialized = True
            return

        # Recover from existing active file — find last valid JSON entry
        with _open_regular(active) as f_scan:
            st_scan = os.fstat(f_scan.fileno())
            last_line, last_at, last_len = _last_valid_entry_in(f_scan, 0)
        if last_line:
            last_entry = json.loads(last_line)  # Guaranteed valid by helper
            self._seq = last_entry.get("seq", 0) + 1
            self._active_has_entry = True
            self._tip = (st_scan.st_dev, st_scan.st_ino, last_at, last_len)
            self._tip_week = _week_of_ts(last_entry.get("ts", ""))
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
            self._backfill_active_begun(active)
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
            self._seed_from_manifest(adopted=adopted)
            self._last_week = _iso_week_now()

        self._initialized = True

    def _refuse_seed_after_incomplete_adoption(self) -> None:
        """⛔ NO CHAIN ANCHOR FROM A MANIFEST THAT MAY BE MISSING A SEALED WEEK (L3
        r1 + r2: codex). With no usable active file the chain continues from the
        manifest's last week. If orphan adoption did not finish its scan (the
        lock could not be taken, the manifest could not be read, or the
        directory could not be listed) and sealed files may exist, a crashed
        rotation's orphan may be newer than that week: appending from the
        manifest would continue the chain past it, and adoption could never
        reconnect it. Raising fails this write, naming the cause, and
        ``_initialize`` retries on the next ``log()``; a cause that does not
        clear (a symlink or directory at the lock path) refuses every write
        until it is removed, which the lock error's text says how to do. A store with no sealed file has nothing to continue
        past, so it is not refused; a directory that cannot be listed may hold
        one, so it is."""
        stem = self._db_path.stem
        try:
            sealed = any(_is_sealed_filename(p.name, stem) for p in self._db_path.parent.iterdir())
        except FileNotFoundError:
            return
        except OSError as e:
            raise _ManifestUnavailable(
                f"orphan adoption did not complete ({self._adoption_skip_reason or 'cause not recorded'}) "
                f"and the audit directory cannot be listed: {e}"
            ) from e
        if sealed:
            raise _ManifestUnavailable(
                f"orphan adoption did not complete ({self._adoption_skip_reason or 'cause not recorded'}); "
                "not starting the chain from a manifest that may be missing a sealed week"
            )

    def _seed_from_manifest(self, *, adopted: bool) -> None:
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
        have recorded. If there is no readable sealed tail, it refuses (raises);
        genesis would be a guess. Since 2026-10-08 ("A") :meth:`log` refuses
        every append while a quarantine is unresolved, so a log() does not
        continue past this seed until ``audit-repair`` has run.

        ``adopted`` is False when orphan adoption did not finish its scan; then
        a chain anchor read from the manifest is refused (see
        :meth:`_refuse_seed_after_incomplete_adoption`). The quarantine branch
        is not, since it anchors on the newest sealed file on disk.
        """
        # Reset FIRST: this can run on an instance whose cached chain state
        # is stale, and genesis is the only defensible starting anchor.
        self._prev_hash = GENESIS_HASH
        self._seq = 0
        self._active_has_entry = False
        self._tip = None
        try:
            manifest = self._load_manifest()  # absent -> fresh (genesis)
        except _ManifestQuarantined:
            self._seed_from_sealed_tail()
            return
        if not adopted:
            self._refuse_seed_after_incomplete_adoption()
        self._refuse_past_unreadable_newer()
        self._refuse_vanished_active(manifest)
        self._prev_hash = manifest.get("active_last_hash", GENESIS_HASH)
        self._seq = manifest.get("active_last_seq", 0)

    def _backfill_active_begun(self, active: Path) -> None:
        """Record ``active_begun`` for an active file begun before it existed (an
        upgraded store), from that file's first valid entry, so the protection
        starts at the upgrade rather than at the next rotation. Runs inside
        ``_initialize``'s span; saves only when the record is missing. A failure
        is logged and never fails the open."""
        try:
            if self._lock_failures.get(threading.get_ident()) is not None:
                return
            manifest = self._load_manifest()
            first = _first_valid_line(active)
            if first is None:
                return
            first_hash = self._compute_hash(first)
            recorded = manifest.get("active_begun")
            if recorded and recorded["first_hash"] == first_hash:
                return
            if recorded and vanished_active_week(manifest) is not None:
                # A record of a different active file whose week is not sealed:
                # that file is gone and this one replaced it. Overwriting would
                # erase the only record of the loss, so it stays, unresolved.
                _log(logging.WARNING,
                    "the audit manifest records an earlier active file (week %s) that "
                    "is gone; its entries are not in the chain", recorded["period"],
                )
                return
            # No record, or a stale one a release that does not clear it left
            # behind when it sealed that week (L3 r1 10-03, codex).
            entry = json.loads(first)
            prev = entry.get("prev_hash", "")
            manifest["active_begun"] = {
                # ``_last_week``, not the first entry's week: it is the week the
                # seal names the file by (L3 r1 10-03: complement and glm asked
                # for the first entry's; a seal's period is the match that
                # matters, and repair's period match needs the same label).
                "period": self._last_week,
                "first_hash": first_hash,
                "first_prev_hash": prev if isinstance(prev, str) else "",
            }
            self._save_manifest(manifest)
        except Exception:
            _log(logging.WARNING,
                "could not record the existing active audit file in the manifest; "
                "a deletion of it will not be detected", exc_info=True,
            )

    def _refuse_without_manifest_lock(self) -> None:
        """Inside an operation span: refuse the append when the span could not
        take the manifest lock (Phill 2026-10-08, "A": an append that cannot
        record or check the manifest leaves a window in which a deleted active
        file goes undetected, so it fails closed; the store counts the drop)."""
        failure = self._lock_failures.get(threading.get_ident())
        if failure is not None:
            raise _ManifestUnavailable(
                f"audit append refused: the audit manifest lock cannot be taken "
                f"({failure}); fix the lock file, then retry. Until then audit "
                "writes are counted as dropped (status: audit_write_failures)"
            )

    def _refuse_while_quarantined(self) -> None:
        """Refuse every append while a quarantined manifest is unresolved (Phill
        2026-10-08, "A", superseding the 09-13 hybrid; run first: with the
        manifest quarantined and the active file deleted, appends seeded from
        the sealed tail, and after audit-repair verify read VALID with the
        deleted week's entries gone and no gap). A directory that cannot be
        listed cannot rule a quarantine out, so it refuses too (KL-24 L3 r6,
        codex 2, run: with a marker on disk and the directory at mode 0300 an
        initialized writer appended; ruling A supersedes round 10b's "writes in
        an unlistable directory must not block")."""
        try:
            markers = _quarantine_markers(self._db_path.parent, self._db_path.stem)
        except OSError as e:
            raise _ManifestUnavailable(
                f"audit append refused: the audit directory cannot be listed to "
                f"rule out a quarantined manifest ({e}); fix its permissions, then "
                "retry. Until then audit writes are counted as dropped (status: "
                "audit_write_failures)"
            ) from e
        if markers:
            raise _ManifestQuarantined(
                f"audit append refused: the audit manifest is quarantined "
                f"({markers[-1]}); run `anneal-memory audit-repair`, then writes "
                "resume. Until then audit writes are counted as dropped (status: "
                "audit_write_failures)"
            )

    def _first_entry_path(self) -> Path:
        return self._active_path.with_name(self._active_path.name + ".first")

    def _stage_first_entry(
        self, active: Path, payload: str, first_hash: str, first_prev_hash: str
    ) -> Path:
        """Write the active file's current bytes (none, or a torn fragment kept
        as evidence) plus ``payload`` to the staging temp, fsynced, then save
        its record (:meth:`_record_active_begun`). Any failure sets the temp
        aside (never deletes it: KL-24 L3 r6, ruled by the desk), withdraws a
        record it may have saved, and raises, so nothing is committed.

        The temp is created exclusively (``O_CREAT | O_EXCL``): whatever is
        already at its name, a symlink included, is never followed, truncated or
        replaced, and the append is refused instead. ``O_NOFOLLOW`` is not
        needed for that and is absent on Windows (KL-24 L3 r6, codex 4, run:
        every first-entry append raised ``AttributeError`` without it). The
        write loops until every byte is down: ``os.write`` may write fewer bytes
        than asked without raising (codex 7, run: half an entry was renamed in
        and reported as written)."""
        tmp = self._first_entry_path()
        try:
            existing = _read_regular_bytes(active)
        except FileNotFoundError:
            existing = b""
        fd = os.open(
            tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644
        )
        try:
            data = memoryview(existing + payload.encode("utf-8"))
            while data:
                written = os.write(fd, data)
                if written <= 0:
                    raise OSError(errno.EIO, "no progress writing the staged audit entry")
                data = data[written:]
            os.fsync(fd)
        except BaseException:
            os.close(fd)
            _set_aside(tmp, _DISCARDED_REASON)
            raise
        os.close(fd)
        try:
            self._record_active_begun(first_hash, first_prev_hash)
        except BaseException:
            _set_aside(tmp, _DISCARDED_REASON)
            # The save may have landed before the failure that raised.
            self._withdraw_active_begun(first_hash)
            raise
        return tmp

    def _staged_entry_commits(self, manifest: dict[str, Any] | None) -> bool:
        """Whether the staged first entry is the one ``manifest`` records as the
        active file's first entry, with the active file holding no entry yet:
        the rule both recovery (:meth:`_finish_first_entry`) and
        ``audit-repair`` apply. ``None`` (no readable manifest) never commits."""
        begun = manifest.get("active_begun") if isinstance(manifest, dict) else None
        if not isinstance(begun, dict):
            return False
        # Never a symlink or anything but a regular file: it would be renamed
        # into the active file's place as it is.
        if not stat.S_ISREG(os.lstat(self._first_entry_path()).st_mode):
            return False
        with _open_regular(self._first_entry_path()) as f:
            staged = _last_valid_entry_in(f, 0)[0]
        try:
            active_holds_entry = _first_valid_line(self._active_path) is not None
        except FileNotFoundError:
            active_holds_entry = False
        return bool(
            staged and self._compute_hash(staged) == begun.get("first_hash")
            and not active_holds_entry
        )

    def _discarded_staged_names(self, period: str) -> list[str]:
        """Names of the set-aside staged entries beside the active file
        (``<active>.first.discarded-<stamp>[-n]``) stamped within ISO week
        ``period``, sorted: regular files only, stamped at or after the week's
        first instant and before the next week's (the manifest's record of the
        first entry carries a week, not a time). A period that does not parse
        yields none. These are POINTERS for a person to inspect: they never
        change a record's certainty. Never renamed or deleted. A listing error
        propagates; a file that vanishes before its ``lstat`` is skipped."""
        bounds = _week_bounds(period)
        if bounds is None:
            return []
        floor, ceiling = bounds
        pattern = _discarded_name_pattern(self._db_path.stem)
        names = []
        for p in self._active_path.parent.iterdir():
            match = pattern.fullmatch(p.name)
            if not (match and floor <= match[1] < ceiling and _canonical_stamp(match[1])):
                continue
            try:
                is_file = stat.S_ISREG(os.lstat(p).st_mode)
            except FileNotFoundError:
                continue
            if is_file:
                names.append(p.name)
        return sorted(names)

    def _resolve_staged_entry(self, manifest: dict[str, Any] | None) -> str | None:
        """Finish or set aside a staged first entry under ``manifest`` (see
        :meth:`_staged_entry_commits`); ``None`` when there is none. Returns what
        was done, for a report. Never deletes it: a staged entry that does not
        commit is renamed to ``<name>.discarded-<UTC stamp>`` and kept."""
        tmp = self._first_entry_path()
        if not os.path.lexists(tmp):
            return None
        if self._staged_entry_commits(manifest):
            os.replace(tmp, self._active_path)
            _fsync_dir(self._active_path.parent)
            return f"finished: {tmp.name} renamed into place as {self._active_path.name}"
        kept = _set_aside(tmp, _DISCARDED_REASON)
        if kept is None:
            raise OSError(f"the staged first entry {tmp.name} did not commit and could not be set aside")
        return f"set aside: {tmp.name} did not commit and is kept as {kept}"

    def _finish_first_entry(self) -> None:
        """Recovery for a staged first entry left by a process that stopped
        between saving its record and the rename (see :meth:`_log_locked`). Under
        the append lock, after the quarantine check. The record names the staged
        entry: the rename is finished. No record names it: the append never
        committed and the temp is set aside (KL-24 L3 r6: never deleted). The
        manifest cannot be read, or is absent beside a quarantine marker (its
        record is in the marker) or a directory that cannot be listed: refused,
        nothing decided (codex 5, glm 1, run: the staged entry was deleted and
        the append refused only afterwards)."""
        tmp = self._first_entry_path()
        if not os.path.lexists(tmp):
            return
        self._initialized = False  # whatever happens here, re-derive from disk
        manifest_path = self._db_path.parent / f"{self._db_path.stem}.audit.manifest.json"
        undecided = (
            f"a staged first audit entry ({tmp.name}) is waiting and the manifest "
            "cannot be read to decide it"
        )
        manifest: dict[str, Any] | None
        try:
            manifest = _parse_manifest_bytes(
                _read_regular_bytes(manifest_path), self._db_path.stem
            )
        except FileNotFoundError:
            try:
                markers = _quarantine_markers(self._db_path.parent, self._db_path.stem)
            except OSError as e:
                raise _ManifestUnavailable(f"{undecided}: {e}") from e
            if markers:
                raise _ManifestQuarantined(
                    f"{undecided}: it is quarantined as {markers[-1]}; run "
                    "`anneal-memory audit-repair`",
                    markers,
                )
            manifest = None  # no manifest, no record
        except _CORRUPT_MANIFEST + (AttributeError,) as e:
            raise _ManifestUnavailable(f"{undecided}: {e}") from e
        try:
            self._resolve_staged_entry(manifest)
        except _ManifestUnavailable:
            raise
        except OSError as e:
            # A refusal that counts as a dropped write, not a bare OSError (KL-24
            # L3 r7, complement LOW 2).
            raise _ManifestUnavailable(f"{undecided}: {e}") from e

    def _record_active_begun(self, first_hash: str, first_prev_hash: str) -> None:
        """Save ``active_begun`` for the active file this call is about to
        start, BEFORE its first entry is written (see :meth:`_log_locked`), so a
        restart can tell a deleted active file from an empty one (see
        ``_refuse_vanished_active``). Once per active file: only the append into
        a file that holds no valid entry (absent, empty, or only a torn fragment:
        L3 r1 10-03, codex + glm, run). The append holds the append lock but not
        the manifest lock, so the save takes the span here.

        Raises ``_ManifestUnavailable`` when it cannot be saved, refusing the
        append (KL-24 L3 r4: best-effort, a failure left the week unprotected),
        including when the manifest lock cannot be taken or the manifest is
        quarantined (Phill 2026-10-08, "A": fail closed, superseding the 10-03
        degrade and the 09-13 hybrid). Where advisory locks do not exist at all
        the span holds no lock and the save still runs."""
        try:
            with self._operation_span():
                self._refuse_without_manifest_lock()
                manifest = self._load_manifest()
                manifest["active_begun"] = {
                    "period": self._last_week, "first_hash": first_hash,
                    "first_prev_hash": first_prev_hash,
                }
                self._save_manifest(manifest)
        except _ManifestUnavailable:
            raise
        except Exception as e:
            raise _ManifestUnavailable(
                "cannot record the active audit file's first entry in the "
                f"manifest, so this append is refused: {e}"
            ) from e

    def _withdraw_active_begun(self, first_hash: str) -> None:
        """Best-effort: clear ``active_begun`` when it still names ``first_hash``,
        an entry whose write was rolled back. Left in place, a restart would read
        the empty file as a deleted one and refuse until ``audit-repair`` (a loud
        false loss, never a silent one)."""
        try:
            with self._operation_span():
                if self._lock_failures.get(threading.get_ident()) is not None:
                    return
                manifest = self._load_manifest()
                begun = manifest.get("active_begun")
                if isinstance(begun, dict) and begun.get("first_hash") == first_hash:
                    manifest["active_begun"] = None
                    self._save_manifest(manifest)
        except Exception:
            _log(logging.WARNING,
                "could not withdraw the manifest's record of a rolled-back first "
                "audit entry; the next open may report the empty file as lost "
                "until audit-repair", exc_info=True,
            )

    def _refuse_vanished_active(self, manifest: dict[str, Any]) -> None:
        """Raise when the manifest records that the active file held entries
        and the seed is running anyway, so there is no usable entry in it now.

        ⛔ THE ENTRIES OF A DELETED ACTIVE FILE WERE INVISIBLE AFTER A RESTART
        (codex L3 r2 10-03 on the rotation re-seed; [run] 10-03 on main before
        this: three entries deleted, reopen, one append, ``verify()`` valid with
        no gap). Within a week nothing on disk but the active file said it had
        entries. ``active_begun`` is that record, and it survives a restart.
        Not refused once the record is cleared: by a seal or adoption of that
        file, or by ``audit-repair`` (a gap, or a sealed week whose first entry is
        the recorded one)."""
        begun = vanished_active_week(manifest)
        if begun is None:
            return
        raise _ManifestUnavailable(
            f"the active audit file {self._active_path.name} held entries in week "
            f"{begun['period']} (first entry hash {begun['first_hash']}) and now holds "
            "none: it was deleted or emptied. Appending would continue the chain past "
            "them with nothing reporting the loss. If the file can be restored, put it "
            "back and retry; otherwise run `anneal-memory audit-repair` to record the "
            "week as a gap, and writes resume"
        )

    def _refuse_past_unreadable_newer(self) -> None:
        """Raise when the last adoption found sealed weeks newer than the
        manifest that cannot be read: with no usable active file the chain
        would continue past them (Phill, 2026-10-03). Called from every path
        that continues the chain after an adoption: the seed and the rotation
        whose active file is missing."""
        if not self._unreadable_newer:
            return
        bad = "; ".join(f"{name} ({cause})" for name, cause in self._unreadable_newer)
        raise _ManifestUnavailable(
            f"sealed audit file(s) newer than the manifest cannot be read: {bad}. "
            "With no usable active file the chain would continue past them. A "
            "permission or I/O error: fix it (e.g. chmod) and retry, and writes "
            "resume with no gap. A corrupt file: `anneal-memory audit-repair` sets "
            "it aside (kept on disk, recorded in the manifest, reported by verify); "
            "for a read error that cannot be fixed, `anneal-memory audit-repair "
            "--set-aside-unreadable`. A set-aside week can be renamed back only "
            "before the next write; after it, the chain has continued without it"
        )

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

    def _adopt_orphaned_files(self) -> bool:
        """Run :meth:`_adopt_locked` inside the caller's operation span
        (spore-1030; one span per operation, 10-03): its load, renames and save,
        and the seed that follows in the same ``log()``, see one manifest. Returns True only when the scan completed; a lock that
        cannot be taken skips recovery and returns False, as
        :meth:`_adopt_locked` does on each early return."""
        self._adoption_skip_reason = ""
        self._unreadable_newer = []
        try:
            self._require_lock()
        except _AuditLockError as exc:
            self._adoption_skip_reason = str(exc)
            _emit_warning(f"Not adopting orphaned audit files: {exc}")
            return False
        return self._adopt_locked()

    def _adopt_locked(self) -> bool:
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
        - neither reads → nothing is set aside or adopted, and ``verify()``
          reports the week. Where the week is newer than the manifest's last,
          it is noted for :meth:`_seed_from_manifest`, which refuses a write
          with no usable active file until ``audit-repair`` sets the week aside
          (Phill, 2026-10-03, superseding round 10's seat ruling that writes
          continue); a write the active file anchors still continues.

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

        Returns True when the manifest loaded and the directory was listed, so
        every name in it was considered, whether or not anything was adopted.
        An unreadable or corrupt orphan does NOT make it False; a newer one is
        reported through ``_unreadable_newer`` instead (above). Returns
        False when it returned before listing: the manifest could not be loaded
        or the directory could not be listed. A missing directory has nothing
        to adopt and returns True.
        """
        stem = self._db_path.stem
        audit_dir = self._db_path.parent
        prefix = f"{stem}.audit."
        # ⛔ QUARANTINE RETURNS BEFORE ANYTHING IS LISTED OR SET ASIDE (hybrid,
        # 2026-09-13). Adopting into a fresh manifest is the automatic rebuild
        # the hybrid forbids. Orphans stay on disk; verify() reports the
        # quarantine. A manifest unavailable right now skips recovery the same
        # way instead of failing the write: the next open retries.
        try:
            manifest = self._load_manifest()
        except _ManifestUnavailable as exc:
            self._adoption_skip_reason = str(exc)
            _emit_warning(f"Not adopting orphaned audit files: {exc}")
            return False
        try:
            names = sorted(p.name for p in audit_dir.iterdir())
        except FileNotFoundError:
            return True  # no directory yet, so nothing to adopt
        except OSError as e:
            # Writable but not listable. Recovery is skipped rather than
            # failing every write (L1, round 10, reproduced at mode 0o300);
            # verify() reports the directory itself.
            self._adoption_skip_reason = f"cannot list the audit directory: {e}"
            _emit_warning(f"Cannot list audit directory for recovery: {e}")
            return False

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
        last_week = _week_of(files[-1]["filename"], prefix) if files else ""

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
                        _log(logging.WARNING,
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
                # Newer than the manifest's last week: with an empty active
                # file the chain would continue from the manifest past it, so
                # _seed_from_manifest refuses until audit-repair sets it aside
                # (Phill, 2026-10-03, superseding round 10's seat ruling).
                if week > last_week:
                    self._unreadable_newer.extend(
                        (path.name, str(scans[path].error)) for path in paths
                    )
                # ⛔ LEFT ON DISK, UNADOPTED, AND NOT RAISED HERE (complement,
                # round 10). ``verify()`` reports it as unmanifested. Raising
                # here made every ``log()`` fail with no way out (``chmod 000``,
                # reproduced 3 of 3); the refusal above is narrower (no usable
                # active file, a newer week) and ``audit-repair`` ends it.
                for path in paths:
                    _emit_warning(
                        f"Not adopting unreadable orphaned audit file {path.name} "
                        f"(left on disk; verify() reports it): {scans[path].error}"
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
                    _log(logging.WARNING,
                        "Audit copies %s and %s hold different bytes; adopting "
                        "the uncompressed copy",
                        plain.name, gz.name,
                    )
            scan = scans[keep]
            if scan.first_prev_hash != tip:
                _log(logging.WARNING,
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
                    _log(logging.WARNING,
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
                    _log(logging.WARNING,
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
            begun = manifest.get("active_begun")
            if begun and scan.first_hash == begun["first_hash"]:
                # The active file the record describes, sealed by a rotation
                # that crashed before its manifest save.
                manifest["active_begun"] = None
            _log(logging.INFO,
                "Adopted orphaned audit file: %s (%d entries)", keep.name, scan.entries
            )

        if chain:
            # A week renamed back and adopted is no longer set aside: its record
            # would print a stale line forever (L3 r1 10-03, codex + glm).
            adopted_names = {keep.name for _, keep, _, _ in chain}
            if manifest.get("set_aside"):
                manifest["set_aside"] = [
                    r for r in manifest["set_aside"] if r["filename"] not in adopted_names
                ]
            self._save_manifest(manifest)
        return True

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
                    scan.first_hash = self._compute_hash(text)
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
            self._unreadable_newer = []
            adopted = False
            try:
                adopted = self._adopt_orphaned_files()
            except Exception:
                # The orphan stays on disk; the re-seed below refuses rather
                # than writing from a guess (adopted stays False).
                _log(logging.WARNING,
                    "could not adopt orphaned sealed audit file(s) while "
                    "rotating; the trail may verify as broken until the store "
                    "is reopened", exc_info=True,
                )
            # ⛔ RE-SEED FROM THE MANIFEST, NEVER FROM THE CACHED HASH (L1 10-03,
            # then L3 r1: codex HIGH, glm, complement). The cached hash is the
            # tip of a week this branch has just found missing from the active
            # name: it was adopted (the manifest now ends there), set aside by
            # another instance's repair (the manifest does not), or not scanned
            # (adoption skipped). The seed the open uses handles all three, with
            # both refusals: an unreadable newer week, an incomplete adoption.
            # On a refusal _initialized stays False, so the next call re-inits.
            self._initialized = False
            self._seed_from_manifest(adopted=adopted)
            self._initialized = True
            self._last_week = current_week
            return

        # ⛔ ONE LOCK FROM THE LOAD TO THE SAVE, HELD BEFORE THE RENAME (spore-1030,
        # L1 reproduced): loading unlocked let a manifest read before a repair
        # overwrite the rebuilt one afterwards. The caller's operation span
        # holds it; one that could not take it refuses the rotation like an
        # unreadable manifest does.
        try:
            self._require_lock()
        except _AuditLockError as exc:
            if not self._rotation_refusal_logged:
                self._rotation_refusal_logged = True
                _log(logging.WARNING, "Not rotating the audit trail: %s", exc)
            return
        self._rotate_locked(active, current_week)

    def _rotate_locked(self, active: Path, current_week: str) -> None:
        """:meth:`_rotate_if_needed` from the manifest load on; the caller holds
        the manifest lock."""
        # ⛔ LOAD THE MANIFEST BEFORE THE RENAME (hybrid, 2026-09-13). If it is
        # quarantined or unreadable, do not rotate: keep appending to the active
        # file, leave ``_last_week`` alone so the next log() retries, and never
        # reach the save below with a manifest that stands in for one we could
        # not read.
        try:
            manifest = self._load_manifest()
        except _ManifestUnavailable as exc:
            if not self._rotation_refusal_logged:
                self._rotation_refusal_logged = True
                _log(logging.WARNING, "Not rotating the audit trail: %s", exc)
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
            refused_week = self._last_week
            self._last_week = current_week
            if not self._rotation_refusal_logged:
                self._rotation_refusal_logged = True
                _log(logging.WARNING,
                    "Not rotating the audit trail: sealed week %s is already on "
                    "disk; appending to the active file instead",
                    refused_week,
                )
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
                    # From here this instance's tip is in the sealed file, not
                    # the active one, whatever happens next (L3 r3, codex MED).
                    self._tip = None
                    _fsync_dir(sealed_path.parent)
                    with _open_regular(sealed_path) as f_in:
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
            if sealed_path.exists():
                # This instance renamed the file holding its own tip: re-derive
                # at the next append (which adopts the orphan) instead of
                # refusing it as lost. Its own act, not a filename read as a
                # peer's seal (see _lost_active).
                self._initialized = False
            # The temp was created before the rename. If the rename never
            # happened it holds no audit data, and left behind it would read
            # as a rotation in flight until the next open.
            if not sealed_path.exists():
                try:
                    tmp_gz_path.unlink()
                except OSError:
                    pass
            raise

        try:
            self._seal_after_rename(
                manifest, tmp_gz_path, sealed_gz_path, sealed_path, current_week,
                entry_count, first_ts, last_ts, file_hash,
            )
        except BaseException:
            # The rename happened; a failure in any later step leaves this
            # instance's cached chain state unproven (L3 r3, codex MED: a failed
            # replace refused the next committed append as a lost file).
            self._initialized = False
            raise

    def _seal_after_rename(
        self, manifest: dict[str, Any], tmp_gz_path: Path, sealed_gz_path: Path,
        sealed_path: Path, current_week: str, entry_count: int, first_ts: str,
        last_ts: str, file_hash: Any,
    ) -> None:
        """Steps 3-5 of :meth:`_rotate_locked`, after the rename."""
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
        manifest["active_begun"] = None  # the file it describes is sealed above
        # Reset seq for new file BEFORE saving manifest, so crash
        # recovery restores the correct starting seq (0), not the
        # pre-rotation value.
        self._seq = 0
        self._active_has_entry = False
        self._tip = None
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
        """Remove rotated files older than retention_days, inside the caller's
        operation span from the load to the save (spore-1030). A ``manifest``
        passed in must have been loaded in that same span (rotation)."""
        if self._retention_days is None:
            return 0
        try:
            self._require_lock()
        except _AuditLockError as exc:
            _log(logging.WARNING, "Skipping audit retention cleanup: %s", exc)
            return 0
        return self._cleanup_locked(manifest)

    def _cleanup_locked(self, manifest: dict[str, Any] | None) -> int:
        retention_days = self._retention_days
        if retention_days is None:
            return 0

        if manifest is None:
            try:
                manifest = self._load_manifest()
            except _ManifestUnavailable as e:
                _log(logging.WARNING, "Skipping audit retention cleanup: %s", e)
                return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
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
        - Invalid -> under the caller's operation span, the markers are listed
          and the bytes read again (see the comment below for what that does
          and does not cover). Still invalid: renamed to a quarantine marker
          (never overwritten), then raises ``_ManifestQuarantined``. Valid by
          then: returned. A marker that appeared meanwhile: raises
          ``_ManifestQuarantined`` with nothing renamed. The span's lock could
          not be taken: ``_ManifestUnavailable`` with nothing renamed. No span
          at all: ``RuntimeError`` (a programming error).
        - Unreadable right now (permission, I/O) -> raises ``_ManifestUnavailable``
          with nothing renamed, so a flaky read can neither quarantine nor
          overwrite.
        - Absent -> a fresh manifest, as before.
        """
        stem = self._db_path.stem
        # ⛔ A DIRECTORY THAT CANNOT BE LISTED CANNOT RULE OUT A MARKER (rebase
        # onto round 10b). Raising here failed every log() in a writable but
        # unlistable directory, the case round 10 had just fixed. A present
        # manifest is read as usual. An absent one is refused below: returning
        # a fresh manifest there is the rebuild the hybrid forbids whenever an
        # unseen marker exists.
        list_error: OSError | None = None
        try:
            markers = _quarantine_markers(self._db_path.parent, stem)
        except OSError as e:
            markers, list_error = [], e
        if markers:
            raise _ManifestQuarantined(
                f"the audit manifest is quarantined as {markers[-1]}; "
                "run `anneal-memory audit-repair`",
                markers,
            )
        try:
            raw = _read_regular_bytes(self._manifest_path)
        except FileNotFoundError:
            if list_error is not None:
                raise _ManifestUnavailable(
                    "the audit manifest is absent and the directory cannot be "
                    f"listed to rule out a quarantine: {list_error}"
                ) from list_error
            return self._fresh_manifest()
        except OSError as e:
            raise _ManifestUnavailable(f"the audit manifest cannot be read right now: {e}") from e
        try:
            return _parse_manifest_bytes(raw, stem)
        except _UNPARSEABLE_JSON:
            pass
        # ⛔ QUARANTINE ONLY UNDER THE CALLER'S OPERATION SPAN (spore-1030,
        # reproduced with two processes: a rename of bytes read unlocked could
        # quarantine a manifest a repair had just rebuilt). While the lock is
        # really held, no writer of this version changes the path between the
        # read above and the rename. The markers are listed and the bytes read
        # again here for writers that do NOT hold it: one whose lock degraded
        # to none (ENOLCK, ruled) or an older anneal-memory writer. That
        # narrows their window; it does not close it, and mixed-version writers
        # stay unsupported. The marker re-list is tested; the re-parse that
        # returns a manifest rebuilt in between is not (L1 10-03, mutation run).
        try:
            self._require_lock()
            try:
                markers = _quarantine_markers(self._db_path.parent, stem)
            except OSError as e:
                raise _ManifestUnavailable(
                    "the audit manifest is invalid and the directory cannot be "
                    f"listed to rule out a quarantine: {e}"
                ) from e
            if markers:
                raise _ManifestQuarantined(
                    f"the audit manifest is quarantined as {markers[-1]}; "
                    "run `anneal-memory audit-repair`",
                    markers,
                )
            try:
                raw = _read_regular_bytes(self._manifest_path)
            except FileNotFoundError as e:
                # Gone with no marker while we held the lock: not ours to
                # rebuild here. Nothing is renamed; the next caller decides.
                raise _ManifestUnavailable(
                    "the invalid audit manifest disappeared before it could be "
                    "quarantined"
                ) from e
            except OSError as e:
                raise _ManifestUnavailable(
                    f"the audit manifest cannot be read right now: {e}"
                ) from e
            try:
                return _parse_manifest_bytes(raw, stem)
            except _UNPARSEABLE_JSON as e:
                marker = self._quarantine_manifest()
                raise _ManifestQuarantined(
                    f"the audit manifest is invalid ({e}) and was quarantined as "
                    f"{marker}; run `anneal-memory audit-repair`",
                    [marker],
                ) from e
        except _AuditLockError as e:
            raise _ManifestUnavailable(
                f"the audit manifest is invalid and was not quarantined: {e}"
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
        _log(logging.WARNING,
            "Quarantined an invalid audit manifest as %s. Audit appends are refused "
            "(counted as dropped writes), and rotation, orphan adoption and retention "
            "are paused, until `anneal-memory audit-repair` rebuilds it.", target.name,
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
        """Save manifest with atomic write, inside the caller's operation span
        (spore-1030). Raises ``_AuditLockError`` (an ``OSError``) when that
        operation could not take the lock.

        ⛔ A TEMP NAME PER SAVE (L2, run: two writers with no lock clobbered one
        fixed ``.json.tmp``, saves failed and a reader saw torn bytes). Where the
        lock degrades to none, each save still writes its own file and replaces
        the manifest atomically, and a failure unlinks only that file."""
        path = self._manifest_path
        self._require_lock()
        # O_EXCL on a random name, mode 0o666 under the umask: the manifest
        # keeps the mode a plain ``open()`` gave it (mkstemp would make it 0600).
        # The basename is fixed-length and independent of the stem, so a stem
        # the manifest name fits never pushes the temp name past NAME_MAX.
        tmp_path = path.with_name(f".anneal-manifest-{secrets.token_hex(8)}.tmp")
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(path)
            _fsync_dir(path.parent)
        except BaseException:
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
    The scan itself is :func:`_last_valid_entry_in`, which the append's
    re-sync also uses, so the two can never disagree about what is valid.
    """
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
    with _open_regular(path) as f:
        return _last_valid_entry_in(f, 0)[0]


def _last_valid_entry_in(f: Any, start: int) -> tuple[str, int, int]:
    """The last valid entry in an open binary audit file from ``start`` to its
    end, as ``(line, offset, length)``: the stripped line the chain hashes, the
    offset its bytes start at, and their length without the newline.
    ``("", -1, 0)`` when there is none. Torn lines, undecodable bytes and
    JSON that is not an entry are skipped; read errors propagate (the reasons
    are in :func:`_read_last_valid_entry`)."""
    best: tuple[str, int, int] = ("", -1, 0)
    if start:
        f.seek(start)
    pos = start
    # Iterated, not looped on ``readline()``: the read-failure tests inject
    # their error at iteration.
    for raw in f:
        here, pos = pos, pos + len(raw)
        try:
            stripped = raw.decode("utf-8").strip()
        except UnicodeDecodeError:
            continue  # Torn line — skip, same as a partial write
        if not stripped:
            continue
        try:
            _require_entry_dict(json.loads(stripped))
        except _UNPARSEABLE_JSON:
            continue  # Partial write, or valid JSON that isn't an entry — skip
        best = (stripped, here, len(raw) - (1 if raw.endswith(b"\n") else 0))
    return best


_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)


def _open_regular(path: Path):
    """Open an audit file for binary reading, and refuse anything that is not a
    regular file — the ONE way this module reads an audit file.

    ⛔ Checking the name first and opening second was the defect twice: an active
    file that was a FIFO was opened by name and blocked ``verify()`` and
    ``anneal-memory audit`` forever (complement HIGH, re-pass f0b24290a7232c06,
    reproduced), and a regular file swapped for a FIFO between an ``lstat`` and
    the open did the same (codex MED, same re-pass, reproduced). So the open is
    non-blocking (a FIFO with no writer returns at once instead of waiting), the
    check is ``fstat`` on the DESCRIPTOR that will be read (nothing can be
    swapped in between), and anything but a regular file raises ``OSError`` —
    which every reader already turns into an unreadable, untrusted result.

    On a regular file the descriptor is switched back to blocking before it is
    read: O_NONBLOCK is not meaningful for regular files on POSIX, and clearing
    it means no reader depends on that. Where the platform has no O_NONBLOCK
    (Windows) there are no FIFOs to open, and the fstat check still applies.

    ``O_BINARY`` (Windows only; ``getattr`` gives 0 elsewhere): without it the
    CRT opens the descriptor in text mode, which rewrites CR LF to LF and stops a
    read at 0x1A — bytes a sealed ``.gz`` holds — and ``os.fdopen(fd, "rb")``
    does not undo that (Diogenes MEDIUM 2026-09-15, reasoned from the Python
    ``os`` and Microsoft ``_read`` documentation, never run on Windows). It is
    read at call time so a test can supply it on a platform that lacks it.
    """
    fd = os.open(path, os.O_RDONLY | _O_NONBLOCK | getattr(os, "O_BINARY", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "not a regular file", str(path))
        if _O_NONBLOCK:
            os.set_blocking(fd, True)
    except BaseException:
        os.close(fd)
        raise
    # Outside the try: ``os.fdopen`` owns the descriptor (closefd=True) and closes
    # it itself if the reader cannot be built, so closing it again here raised
    # EBADF over the real error (glm MED, re-pass 191bdcdd254b37be, reproduced).
    return os.fdopen(fd, "rb")


def _read_regular_bytes(path: Path) -> bytes:
    """All bytes of an audit file, read through :func:`_open_regular`."""
    with _open_regular(path) as f:
        return f.read()


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
        with _open_regular(path) as raw:
            try:
                with gzip.GzipFile(fileobj=raw, mode="rb") as f:
                    yield from f
            except (EOFError, zlib.error, gzip.BadGzipFile) as e:
                raise _CorruptAuditFile(f"corrupt compressed stream: {e!r}") from e
    else:
        with _open_regular(path) as f:
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
        manifest = _parse_manifest_bytes(_read_regular_bytes(manifest_path), stem)
    except FileNotFoundError:
        return True
    except _CORRUPT_MANIFEST:
        return False
    named = {f["filename"] for f in manifest["files"]}
    # A .gz beside a manifested .jsonl of its week is a differing copy that
    # adoption leaves on its name (codex, L3 of round 10b), not a rotation:
    # rotation names neither file of a week until it names the .gz.
    return any(p not in named and p.removesuffix(".gz") not in named for p in pairs)


_STAT_ERROR = (-1, -1, -1)


def _signatures_match(
    a: tuple[int, int, int] | None, b: tuple[int, int, int] | None
) -> bool:
    """True iff two :func:`_stat_signature` results show the same file state.
    A stat error on either side never matches: two failed stats compared equal
    and accepted a manifest that may have changed in between (codex MED +
    complement MED, re-pass 745129a900596363)."""
    return a == b and a != _STAT_ERROR


def _unmanifested_sealed_names(
    names: set[str], stem: str, known: set[str], audit_dir: Path
) -> list[str]:
    """Sealed filenames among ``names`` that ``known`` does not cover — the one
    predicate ``verify()`` and ``anneal-memory audit`` share.

    A week's leftover .jsonl beside its manifested .gz is pending cleanup only
    if it holds the same bytes: rotation leaves it between saving the manifest
    and unlinking, and adoption sets an identical copy aside. A different copy
    is reported (L1, round 10: a clock regression can put another segment
    under that name).
    """
    out = []
    for n in sorted(names):
        if not _is_sealed_filename(n, stem) or n in known:
            continue
        kind = _regular_or_gone(audit_dir / n)
        if kind is None:
            # Gone since the listing: nothing on disk is left uncovered (codex LOW,
            # re-pass adde8c3bcfc957e5 — rotation's unlink of its leftover .jsonl).
            continue
        if (
            kind
            and n.endswith(".jsonl")
            and n + ".gz" in known
            and _regular_or_gone(audit_dir / (n + ".gz"))
            and _same_uncompressed_bytes(audit_dir / n, audit_dir / (n + ".gz"))
        ):
            continue
        out.append(n)
    return out


def _regular_or_gone(path: Path) -> bool | None:
    """True for a regular file, False for anything else that exists (a FIFO or
    device is never opened: comparing bytes through one blocked ``verify()`` and
    ``anneal-memory audit`` indefinitely, codex MED, re-pass adde8c3bcfc957e5,
    reproduced), None when it is gone. ``lstat``, so a symlink is not followed."""
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except FileNotFoundError:
        return None
    except OSError:
        return False


def _stat_signature(path: Path) -> tuple[int, int, int] | None:
    """``(inode, size, mtime_ns)`` of ``path``; None if it does not exist, and
    ``(-1, -1, -1)`` if it cannot be stat'ed for another reason."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    except OSError:
        return _STAT_ERROR
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
    None if the file is corrupt. A transient read failure is re-raised.

    ``chain_break_seq`` is the seq of the first entry whose ``prev_hash`` does
    not match the entry before it, or whose seq does not advance; None when the
    file chains end to end (codex, L3 of the hybrid: without this check repair
    accepted a week with an internal break).
    """
    errors: list[OSError] = []
    entries = 0
    first_ts = last_ts = last_hash = ""
    first_prev_hash: str | None = None
    last_seq: int | None = None
    chain_break_seq: int | None = None
    for raw in _guarded_lines(path, errors):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            text = stripped.decode("utf-8")
            entry = _require_entry_dict(json.loads(text))
        except _UNPARSEABLE_JSON:
            continue
        seq = entry.get("seq")
        if first_prev_hash is None:
            first_prev_hash = entry.get("prev_hash", "")
        elif chain_break_seq is None and (
            entry.get("prev_hash", "") != last_hash
            or (isinstance(seq, int) and last_seq is not None and seq <= last_seq)
        ):
            chain_break_seq = seq if isinstance(seq, int) else -1
        if isinstance(seq, int):
            last_seq = seq
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
        "chain_break_seq": chain_break_seq,
    }


def _first_valid_line(path: Path) -> str | None:
    """The first valid entry line in ``path``, stripped, or None if it holds
    none. A read error raises."""
    errors: list[OSError] = []
    for line in _guarded_lines(path, errors):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            text = stripped.decode("utf-8")
            _require_entry_dict(json.loads(text))
        except _UNPARSEABLE_JSON:
            continue
        return text
    if errors:
        raise errors[0]
    return None


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


def vanished_active_week(manifest: dict[str, Any]) -> dict[str, str] | None:
    """The manifest's ``active_begun`` record. A seal and an adoption clear it;
    ``audit-repair`` clears it in the same save that records the gap, or when a
    readable sealed week's first entry is the recorded file's first entry, so a
    later deletion in the same week is seen again. The caller establishes that
    the active file holds no usable entry; this says only that it once did.

    ⛔ A ``files`` record with the same PERIOD no longer clears it (KL-24 L3 r3,
    codex HIGH, reasoned trace): after a clock rollback a new active file began
    in a week already sealed, and deleting it read as sealed, so the chain went
    on with its entries silently missing. A seal by an older release that kept
    the field set now reads as a vanished file until ``audit-repair`` matches
    its first entry."""
    begun = manifest.get("active_begun")
    if not begun:
        return None
    return dict(begun)


def set_aside_report_lines(
    records: list[dict[str, str]], db_path: str | Path
) -> list[str]:
    """One line per ``AuditVerifyResult.set_aside`` record, for every surface
    that prints a verify verdict (the CLI and ``server.py --verify-audit``), so
    they cannot drift apart. A record whose set-aside file is on disk is a GAP.
    A record marked ``certainty: possible`` (a manifest rebuilt from quarantine
    over an active file with no entry) is a POSSIBLE GAP: whether that file
    ever held entries cannot be known (KL-24 L3 r6, codex 10). A vanished active
    file is a definite GAP; any ``preserved_attempts`` are printed as files to
    inspect and never change that.
    One whose file is missing is reported as such, not as a gap: the week was
    renamed back (adopted if before any write; after one it stays unmanifested
    and must be renamed to its set-aside name again), or a repair stopped
    between saving the record and the rename (re-run it)."""
    audit_dir = Path(db_path).expanduser().parent
    lines = []
    for record in records:
        if record.get("certainty") == _POSSIBLE:
            kept = record.get("preserved_attempts")
            lines.append(
                f"POSSIBLE GAP: the active audit file {record['filename']} "
                f"({record['period']}) held no entry when audit-repair recorded it "
                f"at {record['at']}; whether it held entries before cannot "
                f"be known: {record['cause']}"
                + (f"; set-aside staged entries to inspect: {', '.join(kept)}" if kept else "")
            )
        elif record["set_aside_as"] == "":
            kept = record.get("preserved_attempts")
            lines.append(
                f"GAP: the active audit file {record['filename']} ({record['period']}) "
                f"went missing with its entries; audit-repair recorded it at "
                f"{record['at']}: {record['cause']}; its entries are not in the "
                "verified chain"
                + (f"; set-aside staged entries (files to inspect): {', '.join(kept)}" if kept else "")
            )
        elif os.path.lexists(audit_dir / record["set_aside_as"]):
            lines.append(
                f"GAP: {record['filename']} ({record['period']}) was set aside by "
                f"audit-repair as {record['set_aside_as']} at {record['at']}: "
                f"{record['cause']}; its entries are not in the verified chain"
            )
        else:
            lines.append(
                f"SET-ASIDE RECORD WITHOUT ITS FILE: {record['filename']} "
                f"({record['period']}) was recorded as set aside as "
                f"{record['set_aside_as']}, which is not on disk. If the week was "
                "renamed back after a write, rename it to that name again; if a "
                "repair stopped before its rename, run `anneal-memory audit-repair`"
            )
    return lines


def _set_aside_records_on_disk(audit_dir: Path, stem: str) -> list[dict[str, str]]:
    """Set-aside records rebuilt from the names ``audit-repair`` gave the files
    it set aside (``<sealed name>.unreadable-<stamp>``), for a manifest rebuilt
    from quarantine. The cause was in the quarantined manifest and is not
    recovered. A listing error is raised."""
    pattern = re.compile(
        rf"(?P<orig>.+)\.{_UNREADABLE_REASON}-(?P<stamp>\d{{8}}T\d{{12}}Z)"
    )
    records = []
    for name in sorted(p.name for p in audit_dir.iterdir()):
        m = pattern.fullmatch(name)
        if m and _is_sealed_filename(m["orig"], stem):
            records.append({
                "filename": m["orig"],
                "set_aside_as": name,
                "period": _sealed_period(m["orig"], stem),
                "cause": "not recovered: it was recorded in the quarantined manifest",
                "at": m["stamp"],
            })
    return records


def _set_aside(path: Path, reason: str) -> str | None:
    """Rename ``path`` to ``<name>.<reason>-<UTC stamp>`` in the same
    directory — the only way recovery takes a file off its name. Returns the
    new name, or None when it could not be moved.

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
        _log(logging.WARNING,
            "Could not set aside audit file %s; it stays under its own name",
            path.name, exc_info=True,
        )
        return None
    _fsync_dir(path.parent)
    _log(logging.WARNING, "Set aside audit file %s as %s", path.name, target.name)
    return target.name


def _guarded_lines(path: Path, errors: list[OSError]):
    """``_iter_lines``, but a read failure stops iteration and is appended
    to ``errors`` instead of raising — for callers that must turn an
    unreadable file into a result rather than a traceback."""
    try:
        yield from _iter_lines(path)
    except OSError as e:
        errors.append(e)
