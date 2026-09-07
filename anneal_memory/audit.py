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
            event: Event type. Not enforced (``event`` is a bare ``str``),
                   but the emitted vocabulary is closed. Keep this list in
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
        # Snapshot for the rollback. See the handler for why rolling the
        # FILE back is only half of it.
        saved_chain_state = (
            self._prev_hash,
            self._seq,
            self._dropped_since_last,
        )
        try:
            with open(active, "a", encoding="utf-8") as f:
                f.write(json_line + "\n")
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
            # ⛔ BUT "THE AMBIGUITY STANDS" UNDERSTATES IT, AND THIS LINE
            # SAID EXACTLY THAT UNTIL 2026-09-07. A failed truncate does not
            # leave a neutral unknown: the entry stays on disk while the
            # restore below unconditionally rewinds memory to "nothing
            # landed", so the caller's retry reuses the seq. MEASURED (L2,
            # 2026-09-07) with NO terminal signal at all — an ``fsync``
            # reporting EIO after the data landed, then the rollback's
            # ``open`` failing EROFS on the same sick disk: seqs
            # ``[0, 1, 2, 2]``, ``verify(): valid=False``. A FALSE TAMPERING
            # VERDICT, in the dangerous direction, reachable with ordinary
            # exceptions only.
            # ▶ NOT CLOSED HERE, and the candidate fix is recorded so the
            # next reader does not have to re-derive it: make the restore
            # CONDITIONAL on the truncate having succeeded. If the entry is
            # still on disk, leaving memory ADVANCED is what makes the two
            # agree. It is a rollback-semantics change on a file whose
            # independent review was cut off twice on 2026-09-07, and the
            # outcome is unchanged from before this handler existed, so it
            # is filed rather than smuggled in. See ``next_steps.md``.
            try:
                with open(active, "r+b") as f_trunc:
                    f_trunc.truncate(resume_at)
                    f_trunc.flush()
                    os.fsync(f_trunc.fileno())
            except Exception:
                # ⛔ NOT A BARE ``pass``. This module logs the far more
                # benign ``on_event`` and orphan-adoption failures, and this
                # is the branch that ends in ``verify(): valid=False`` — an
                # operator hitting it got a red integrity verdict with zero
                # breadcrumbs. Logged, not raised: raising here would mask
                # the original failure with the rollback's.
                logger.warning(
                    "audit rollback failed for seq %d; the aborted entry is "
                    "still on disk and the chain state has been rewound, so "
                    "a retry will reuse this seq and verify() will report a "
                    "hash mismatch",
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
            (
                self._prev_hash,
                self._seq,
                self._dropped_since_last,
            ) = saved_chain_state
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
            missing_seq = self._seq
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
                        error=f"Hash mismatch at seq {entry.get('seq')}: "
                              f"expected {expected_hash[:20]}..., "
                              f"got {actual_prev[:20]}...",
                    )

                # Compute hash from the line on disk, not a re-serialization.
                # _compute_hash normalizes whitespace (see its docstring).
                # Re-serializing via json.dumps would be byte-identical in
                # CPython today but fragile for cross-language verifiers.
                expected_hash = cls._compute_hash(line)
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
            self._last_week = _iso_week_now()

        self._initialized = True

    def _adopt_orphaned_files(self) -> None:
        """Adopt sealed files that the manifest doesn't know about.

        This handles crash recovery: if the process dies between
        active.rename() and _save_manifest() during rotation, the sealed
        file exists on disk but the manifest has no record of it.
        Scans for both compressed (.gz) and uncompressed (.jsonl) orphans
        — crash can happen before or after gzip compression.

        If both .gz and .jsonl exist for the same period (crash between
        gzip-complete and sealed_path.unlink()), prefers .gz and removes
        the .jsonl duplicate to prevent false verify() failures.
        """
        stem = self._db_path.stem
        audit_dir = self._db_path.parent
        active_name = f"{stem}.audit.jsonl"
        prefix = f"{stem}.audit."

        # Clean up stale .tmp files from crashed gzip writes.
        # Crash during rotation leaves *.jsonl.gz.tmp files that no other
        # code path catches (orphan adoption looks for .gz and .jsonl only,
        # _cleanup only removes manifest-tracked files). Pure disk waste.
        for tmp_path in audit_dir.glob(f"{prefix}*.jsonl.gz.tmp"):
            try:
                tmp_path.unlink()
                logger.info("Cleaned up stale gzip temp file: %s", tmp_path.name)
            except OSError:
                logger.warning("Failed to clean up stale temp file: %s", tmp_path.name)

        manifest = self._load_manifest()
        known_files = {f["filename"] for f in manifest.get("files", [])}

        # Collect orphans grouped by period to detect duplicates
        orphans_by_period: dict[str, list[Path]] = {}
        for pattern in [f"{prefix}*.jsonl.gz", f"{prefix}*.jsonl"]:
            for path in sorted(audit_dir.glob(pattern)):
                if path.name == active_name:
                    continue  # Skip the active file
                if path.name not in known_files:
                    # Extract period from filename
                    period = path.name.removeprefix(prefix)
                    period = period.removesuffix(".jsonl.gz").removesuffix(".jsonl")
                    orphans_by_period.setdefault(period, []).append(path)

        if not orphans_by_period:
            return

        # Deduplicate: if both .gz and .jsonl exist for same period,
        # prefer .gz (gzip completed) and remove the .jsonl duplicate.
        # Sort by period to ensure manifest entries are chronological —
        # without sorting, two-pass glob inserts all .gz periods before
        # all .jsonl periods, breaking chronological order in the manifest
        # when mixed orphan types span non-adjacent periods.
        orphans: list[Path] = []
        for period, paths in sorted(orphans_by_period.items()):
            if len(paths) > 1:
                gz_paths = [p for p in paths if p.name.endswith(".gz")]
                jsonl_paths = [p for p in paths if not p.name.endswith(".gz")]
                if gz_paths:
                    orphans.append(gz_paths[0])
                    for dup in jsonl_paths:
                        try:
                            dup.unlink()
                            logger.info("Removed duplicate orphan: %s (preferring .gz)", dup.name)
                        except OSError:
                            logger.warning("Failed to remove duplicate orphan: %s", dup.name)
                else:
                    orphans.append(paths[0])
            else:
                orphans.append(paths[0])

        for orphan_path in orphans:
            # Read the orphaned file to get metadata
            entry_count = 0
            first_ts = ""
            last_ts = ""
            last_hash = ""

            for line in _iter_lines(orphan_path):
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
                    # Hash the line from disk, not a re-serialization
                    last_hash = self._compute_hash(stripped)
                except json.JSONDecodeError:
                    pass

            # Extract period from filename (e.g., "memory.audit.2026-W14.jsonl.gz")
            period = orphan_path.name.removeprefix(prefix)
            period = period.removesuffix(".jsonl.gz").removesuffix(".jsonl")

            manifest["files"].append({
                "filename": orphan_path.name,
                "period": period,
                "entries": entry_count,
                "first_ts": first_ts,
                "last_ts": last_ts,
                "last_hash": last_hash,
                "sha256_file": "",  # Not computed during adoption
            })

            if last_hash:
                manifest["active_last_hash"] = last_hash

            logger.info("Adopted orphaned audit file: %s (%d entries)", orphan_path.name, entry_count)

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
            # ⛔ "ACTIVE MISSING" IS NOT PROOF THE ROTATION SUCCEEDED.
            # Rotation renames the active file FIRST, then gzips, then updates
            # the manifest. If either later step raises — disk full during the
            # gzip is the measured case — the sealed file exists, the manifest
            # does not know about it, and the active file is GONE. Arriving
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

        # Seal the active file with the old week label
        sealed_name = f"{self._db_path.stem}.audit.{self._last_week}.jsonl"
        sealed_path = active.parent / sealed_name
        sealed_gz_path = sealed_path.with_suffix(".jsonl.gz")

        # Rename → compress (atomic) → update manifest
        active.rename(sealed_path)

        # Gzip compress to temp file, then atomic rename.
        # Crash during gzip write → partial .tmp + complete .jsonl on disk.
        # Orphan adoption handles the .jsonl; .tmp is harmless dead weight.
        # Without atomic write, crash → partial .gz + complete .jsonl, and
        # dedup logic prefers .gz → deletes the good .jsonl copy.
        tmp_gz_path = Path(str(sealed_gz_path) + ".tmp")
        file_hash = hashlib.sha256()
        entry_count = 0
        first_ts = ""
        last_ts = ""

        with open(sealed_path, "rb") as f_in, gzip.open(tmp_gz_path, "wb") as f_out:
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

        # Atomic rename — .gz is either complete or doesn't exist
        tmp_gz_path.replace(sealed_gz_path)

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
        # Reset seq for new file BEFORE saving manifest, so crash
        # recovery restores the correct starting seq (0), not the
        # pre-rotation value.
        self._seq = 0
        manifest["active_last_seq"] = self._seq
        self._save_manifest(manifest)

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
