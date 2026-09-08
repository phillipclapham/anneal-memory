"""Tests for the hash-chained JSONL audit trail."""

import gzip
import json
import logging
import os
import tempfile
import uuid
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import anneal_memory.audit as audit_module
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


class TestTheAppendIsAllOrNothingForTerminalExceptionsToo:
    """codex L3 HIGH, 2026-09-06 — the rollback's own class had a hole."""

    def test_an_interrupt_mid_append_does_not_duplicate_a_sequence_number(
        self, tmp_path, monkeypatch
    ):
        """The write-first rollback was ``except Exception``, so it skipped this.

        ``log()`` is write-first: chain state advances only after fsync
        returns, and the rollback exists because that leaves a THIRD state —
        the line already visible on disk while ``_seq``/``_prev_hash`` are
        unchanged. A caller that swallows and retries then emits the SAME seq
        and the SAME prev_hash.

        The 2026-09-04 HIGH fixed that for ``Exception`` (a failing fsync).
        A ``KeyboardInterrupt`` in the same window walked straight past the
        handler and no rollback ran.

        ⚠ IT BECAME REACHABLE ON 2026-09-06, hours before this test was
        written. Until the store's per-event catch was widened to
        ``BaseException``, an interrupt here abandoned the whole replay — no
        retry followed, so the inconsistency died with the run. Once the store
        began RECORDING the drop and CONTINUING, the next append reused the
        stale sequence. REPRODUCED: seqs on disk ``[0, 1, 1]`` and ``verify()``
        reporting a hash mismatch — the identical signature as the fsync-EIO
        HIGH, reached through the other exception branch. **A durability
        hiccup read as tampering**, on the record whose entire value is
        telling those two apart.
        """
        import json
        import os

        import anneal_memory.audit as audit_module
        from anneal_memory.audit import AuditTrail

        db = tmp_path / "chain.db"
        trail = AuditTrail(db)
        trail.log("record", {"i": 0})

        real_fsync = os.fsync
        fired = {"yet": False}

        def fsync_then_interrupt(fd):
            # The line is already durable when this raises — the exact window
            # between a completed append and the chain-state advance.
            real_fsync(fd)
            if not fired["yet"]:
                fired["yet"] = True
                raise KeyboardInterrupt("Ctrl+C after the line landed")

        monkeypatch.setattr(audit_module.os, "fsync", fsync_then_interrupt)
        with pytest.raises(KeyboardInterrupt):
            trail.log("record", {"i": 1})
        monkeypatch.setattr(audit_module.os, "fsync", real_fsync)

        # What the store now does on that path: record the drop and continue.
        trail.note_write_failure()
        trail.log("record", {"i": 2})

        entries = [
            json.loads(line)
            for line in (tmp_path / "chain.audit.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        seqs = [e["seq"] for e in entries]
        assert len(seqs) == len(set(seqs)), (
            f"duplicate sequence numbers on disk ({seqs}) — the interrupted "
            "append was left in place while the chain state stayed behind, so "
            "the retry reused its seq and prev_hash"
        )
        assert AuditTrail.verify(str(db)).valid, (
            "the chain no longer verifies: an interrupted append was read back "
            "as tampering, which is the one thing this record exists to rule "
            "out"
        )

    # -- 2026-09-07: the window above is not the whole window --

    @pytest.mark.parametrize(
        "store_attr",
        ["_prev_hash", "_seq", "_dropped_since_last"],
    )
    def test_an_interrupt_between_the_chain_state_stores_rolls_back_too(
        self, tmp_path, monkeypatch, store_attr
    ):
        """The advance is three stores; a signal can land between any two.

        The sibling test above injects at ``os.fsync`` — INSIDE the ``with``,
        inside the guarded block. The truncate covers that TODAY; it did not
        always, and the sibling's own docstring says so 70-odd lines up: for
        a ``KeyboardInterrupt`` at ``fsync`` — exactly what it injects — the
        handler was ``except Exception`` and no rollback ran until the
        2026-09-06 widening. (An earlier draft of this line said "has always
        covered that", contradicting its own sibling; caught by L1.)
        It is a real arm and it is narrower than the window the handler's own
        comment describes: *"after the line was written and fsynced but before
        the chain state advanced"*. On 2026-09-06 the tail of that window sat
        OUTSIDE the ``try`` entirely, so widening the handler could not reach
        it (diogenes, 2026-09-07).

        Each parameter interrupts immediately BEFORE one of the three stores,
        so the arms bracket the whole advance:

        · ``_prev_hash``          — nothing stored yet
        · ``_seq``                — ``_prev_hash`` stored, ``_seq`` not
        · ``_dropped_since_last`` — ``_prev_hash`` and ``_seq`` both stored

        ⛔ MUTATION RECIPE — BOTH MUTANTS BUILT AND RUN 2026-09-07, ARM
        SETS TRANSCRIBED FROM THE RUN. Selected alone
        (``-k between_the_chain_state_stores``), three arms, 3 passed at
        HEAD:

        1. Move the three chain-state stores back OUT of the ``try`` in
           ``AuditTrail.log``, below the handler's ``raise`` (where they sat
           until 09-07) → ``_prev_hash`` and ``_seq`` FAIL on DUPLICATE SEQS
           (``[0, 1, 2, 2]``: nothing rolls back, so the retry reuses the
           seq); ``_dropped_since_last`` PASSES, because by then ``_seq``
           has already advanced and the retry gets a fresh number.
        2. Move them INSIDE the ``try`` but delete the chain-state restore
           at the top of the handler → ``_seq`` and ``_dropped_since_last``
           FAIL on ``verify(): valid=False`` — the file rolled back and
           memory did not, so the retry chains over a hole. ⚠ THE TWO ARMS
           LEAVE DIFFERENT DISKS: ``_seq`` gives seqs ``[0, 1, 2]``,
           mismatch at 2 — CONTIGUOUS, no gap to spot — and
           ``_dropped_since_last`` gives ``[0, 1, 3]``, mismatch at 3. An
           earlier draft gave ``[0, 1, 3]`` for both and hid the alarming
           one. ``_prev_hash`` PASSES, because nothing had been stored.

        ⚠ THE TWO MUTANTS OVERLAP AT ``_seq`` AND EACH LEAVES A DIFFERENT
        ARM GREEN — they are NOT disjoint, and an earlier draft of this
        docstring said they were, which is why the sets above are
        transcribed from the run rather than reasoned from the diff. What
        the arms actually establish: no single arm grades both containment
        layers, and neither layer can be dropped on the grounds that the
        suite still passes without it.
        """
        import json

        from anneal_memory.audit import AuditTrail

        shadow = "__shadow" + store_attr
        armed = {"attr": None}

        def _getter(self):
            return self.__dict__[shadow]

        def _setter(self, value):
            if armed["attr"] == store_attr:
                armed["attr"] = None
                raise KeyboardInterrupt(f"Ctrl+C just before {store_attr}")
            self.__dict__[shadow] = value

        monkeypatch.setattr(
            AuditTrail, store_attr, property(_getter, _setter), raising=False
        )

        db = tmp_path / "chain.db"
        trail = AuditTrail(db)
        trail.log("record", {"i": 0})
        trail.log("record", {"i": 1})

        armed["attr"] = store_attr
        with pytest.raises(KeyboardInterrupt):
            trail.log("record", {"i": 2})

        # Caller contract: swallow, record the drop, retry.
        trail.note_write_failure()
        trail.log("record", {"i": 3})

        entries = [
            json.loads(line)
            for line in (tmp_path / "chain.audit.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        seqs = [e["seq"] for e in entries]
        assert len(seqs) == len(set(seqs)), (
            f"duplicate sequence numbers on disk ({seqs}) after an interrupt "
            f"before the {store_attr} store"
        )
        assert AuditTrail.verify(str(db)).valid, (
            f"an interrupt before the {store_attr} store left disk and the "
            f"in-memory chain state disagreeing, so the retry chained over a "
            f"hole and verify() reports tampering — seqs on disk {seqs}"
        )


    @pytest.mark.parametrize(
        "store_attr",
        ["_prev_hash", "_seq", "_dropped_since_last"],
    )
    def test_the_rollback_truncates_before_it_restores(
        self, tmp_path, monkeypatch, store_attr
    ):
        """ORDER, not presence: the fallible step must run first.

        The handler does two things — truncate the file back, and restore
        the in-memory chain state. It shipped restore-first for an hour on
        2026-09-07 on the argument that the restore cannot raise. codex (L3)
        pointed out that this optimises the wrong thing: the TRUNCATE is the
        step that must happen, and a terminal signal delivered inside the
        handler kills everything after where it lands.

        The realistic case needs only ONE terminal signal, because the
        original failure need not be terminal at all — here an ordinary
        ``OSError`` (ENOSPC) after a successful write+flush puts us in the
        handler, and a single ``KeyboardInterrupt`` during the restore then
        skips the truncate. Restore-first leaves the entry on disk with the
        chain state never advanced, so the caller's retry reuses the seq:
        ``[0, 1, 2, 2]`` and ``verify(): valid=False`` — a durability
        hiccup read as tampering, which is the one verdict this record
        exists to make impossible.

        ⛔ MUTATION-CHECKED 2026-09-07: move the chain-state restore back
        ABOVE the truncate's ``try`` in ``AuditTrail.log``'s handler and all
        three arms fail with ``valid=False``. Selected alone
        (``-k truncates_before_it_restores``).
        """
        import json

        import anneal_memory.audit as audit_module
        from anneal_memory.audit import AuditTrail

        shadow = "__order" + store_attr
        armed = {"attr": None, "fired": 0}

        def _getter(self):
            return self.__dict__[shadow]

        def _setter(self, value):
            if armed["attr"] == store_attr:
                armed["attr"] = None
                armed["fired"] += 1
                raise KeyboardInterrupt("Ctrl+C inside the rollback handler")
            self.__dict__[shadow] = value

        monkeypatch.setattr(
            AuditTrail, store_attr, property(_getter, _setter), raising=False
        )

        db = tmp_path / "order.db"
        trail = AuditTrail(db)
        trail.log("record", {"i": 0})
        trail.log("record", {"i": 1})

        real_fsync = audit_module.os.fsync
        fsyncs = {"n": 0}

        def fsync_then_enospc(fd):
            # The line is durable; the failure is ORDINARY, not terminal.
            #
            # ⛔ FIRE ONCE, ON THE ENTRY'S OWN FSYNC ONLY. Until 2026-09-07
            # this raised on EVERY fsync, so the rollback's own
            # ``os.fsync(f_trunc.fileno())`` raised too and the scenario
            # actually run was TWO I/O failures — not the one this
            # docstring describes and grades. It passed anyway because
            # ``truncate()`` takes effect before its fsync, so the file was
            # rolled back regardless and the restore-order property was
            # still the only thing left to observe.
            #
            # ⚠ IT STOPPED BEING HARMLESS THE MOMENT THE RESTORE BECAME
            # CONDITIONAL. A handler that asks "did the truncate block
            # succeed?" reads the injected second failure as "it did not",
            # takes the invalidation branch, and never reaches the restore
            # this test exists to order — the arm then fires inside
            # ``_initialize`` during the RETRY, outside the
            # ``pytest.raises`` below, and escapes as a bare
            # ``KeyboardInterrupt`` that aborts the whole run.
            #
            # ⚖ THE GENERAL RULE, and this file already knew it one layer
            # over: scope a fault injection to the CALL SITE it names.
            # ``next_steps.md`` records the identical defect being caught in
            # a hand probe the same day ("my probe raised ENOSPC on *every*
            # fsync including the truncate's") — in the shipped test it went
            # unchecked.
            real_fsync(fd)
            fsyncs["n"] += 1
            if fsyncs["n"] == 1:
                raise OSError(28, "No space left on device")

        monkeypatch.setattr(audit_module.os, "fsync", fsync_then_enospc)
        armed["attr"] = store_attr
        # ⛔ ``KeyboardInterrupt``, NOT ``BaseException``. The injected
        # ``OSError`` also satisfies a bare ``BaseException``, so the loose
        # form passes on a tree where the restore stopped running and the
        # interrupt therefore never fired — it would grade nothing and say
        # nothing. Named by codex (L3, 2026-09-07) as a false-green risk;
        # ⚠ MEASURED the same hour, the setter DOES fire on all three arms
        # today (`fired=1`, ``KeyboardInterrupt`` escaping, seqs [0,1,2]),
        # so the gate is under-asserted rather than broken. The counter
        # below is what makes the difference detectable.
        with pytest.raises(KeyboardInterrupt):
            trail.log("record", {"i": 2})
        assert armed["fired"] == 1, (
            "the interrupt never fired at the "
            f"{store_attr} store, so this arm graded nothing"
        )
        monkeypatch.setattr(audit_module.os, "fsync", real_fsync)

        trail.note_write_failure()
        trail.log("record", {"i": 3})

        entries = [
            json.loads(line)
            for line in (tmp_path / "order.audit.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        seqs = [e["seq"] for e in entries]
        assert len(seqs) == len(set(seqs)), (
            f"duplicate sequence numbers on disk ({seqs}): one terminal "
            f"signal during the restore of {store_attr} skipped the "
            f"truncate, so the aborted entry stayed on disk"
        )
        assert AuditTrail.verify(str(db)).valid, (
            f"a single terminal signal during the restore of {store_attr} "
            f"produced a false tampering verdict — seqs on disk {seqs}"
        )

    def test_the_restore_puts_prev_hash_before_seq(self):
        """The restore's element order is load-bearing, so it is asserted.

        Two terminal signals — one mid-advance (leaving memory partly
        advanced), one mid-restore between two of the three stores:

          ``_prev_hash`` first ... ``_prev_hash`` is back at hash(E1) while
            ``_seq`` stays advanced. The next entry chains CORRECTLY and
            merely skips a seq number; ``verify()`` checks linkage, not seq
            monotonicity. MEASURED 2026-09-07: valid=True.
          ``_seq`` first .......... ``_prev_hash`` is still pointing at the
            entry the truncate just removed, so the next entry chains from
            a line that is no longer on disk. MEASURED: valid=False, "Hash
            mismatch at seq 2" — with CONTIGUOUS seqs, so nothing looks
            wrong until verify() runs.

        ⚠ THIS ORDER WAS ACCIDENTAL UNTIL IT WAS MEASURED. It is written
        the way the snapshot tuple happens to be written, and nothing said
        it mattered. Asserted structurally rather than commented, because a
        future edit reordering a three-element tuple for tidiness would not
        think to re-derive any of the above.

        ⛔ MUTATION-CHECKED 2026-09-07: swap the first two names in the
        restore tuple and this test fails.
        """
        import ast
        import inspect

        import anneal_memory.audit as audit_module

        tree = ast.parse(inspect.getsource(audit_module))
        cls = next(
            n for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "AuditTrail"
        )
        log = next(
            n for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "log"
        )
        # ⛔ ANCHORED TO THE HANDLER, AND TO THE SNAPSHOT IT RESTORES FROM.
        # This search was ``ast.walk(log)`` for any single-target tuple
        # assign whose elements are Attributes — it required neither
        # ``self`` as the base, nor placement inside the rollback handler,
        # nor ``saved_chain_state`` as the RHS. So a dead or unrelated
        # correctly-ordered tuple anywhere in ``log()`` satisfied it while
        # the REAL restore was reversed, or expressed as sequential
        # ``Assign``s, and this gate stayed green. codex (L3, 2026-09-07).
        handlers = [
            h for n in ast.walk(log) if isinstance(n, ast.Try)
            for h in n.handlers
            if isinstance(h.type, ast.Name) and h.type.id == "BaseException"
        ]
        assert len(handlers) == 1, (
            f"expected exactly one BaseException handler in log(), found "
            f"{len(handlers)}"
        )
        handler = handlers[0]

        def _is_self_attr(e):
            return (
                isinstance(e, ast.Attribute)
                and isinstance(e.value, ast.Name)
                and e.value.id == "self"
            )

        restores = [
            n for n in ast.walk(handler)
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Tuple)
            and n.targets[0].elts
            and all(_is_self_attr(e) for e in n.targets[0].elts)
            and isinstance(n.value, ast.Name)
            and n.value.id == "saved_chain_state"
        ]
        assert len(restores) == 1, (
            f"expected exactly one tuple restore from ``saved_chain_state`` "
            f"inside the rollback handler, found {len(restores)} — if the "
            f"rollback grew a second one, or the restore stopped reading the "
            f"snapshot, this test no longer knows which order it is grading"
        )
        names = [e.attr for e in restores[0].targets[0].elts]
        assert names[0] == "_prev_hash", (
            f"the rollback restores {names} — ``_prev_hash`` must come "
            f"FIRST. A terminal signal between it and ``_seq`` otherwise "
            f"leaves _prev_hash pointing at the entry the truncate removed, "
            f"and the next append chains from a line that is not on disk: "
            f"verify() reports a hash mismatch over contiguous seqs."
        )
        assert names == ["_prev_hash", "_seq", "_dropped_since_last"], (
            f"the rollback restores {names}; the measured-safe order is "
            f"['_prev_hash', '_seq', '_dropped_since_last']"
        )

    def test_a_signal_inside_the_truncate_is_still_an_open_window(
        self, tmp_path, monkeypatch
    ):
        """Pins the residual the ordering table does NOT cover.

        ``test_the_rollback_truncates_before_it_restores`` interrupts at the
        three restore stores — the one place truncate-first wins. It never
        interrupts the truncate. That gap is exactly where the handler's
        comment briefly claimed "TWO terminal signals" and was wrong: in the
        ordinary-entry regime BOTH orderings need only one, and truncate-
        first merely narrows WHERE it has to land.

        So this arm lands it there: ENOSPC after a successful write+flush
        puts us in the handler, then a single ``KeyboardInterrupt`` at the
        truncate's ``open`` — before ``truncate(resume_at)`` takes effect.
        The aborted line stays on disk, the retry reuses its seq, and
        ``verify()`` reports tampering. Measured: seqs ``[0, 1, 2, 2]``,
        ``valid=False``.

        ⚠ A signal at the truncate's *fsync* is SAFE — ``truncate()`` has
        already taken effect by then — which is why the window is "before it
        takes effect" and not "anywhere in the truncate".
        """
        import builtins
        import json

        import anneal_memory.audit as audit_module
        from anneal_memory.audit import AuditTrail

        db = tmp_path / "resid.db"
        trail = AuditTrail(db)
        trail.log("record", {"i": 0})
        trail.log("record", {"i": 1})

        real_fsync = audit_module.os.fsync
        real_open = builtins.open
        phase = {"at": "append"}

        def fsync_then_enospc(fd):
            if phase["at"] == "append":
                real_fsync(fd)
                phase["at"] = "handler"
                raise OSError(28, "No space left on device")
            return real_fsync(fd)

        def open_that_dies_in_the_handler(*args, **kwargs):
            mode = args[1] if len(args) > 1 else kwargs.get("mode", "")
            if phase["at"] == "handler" and "b" in str(mode):
                phase["fired"] = phase.get("fired", 0) + 1
                raise KeyboardInterrupt("Ctrl+C before the truncate landed")
            return real_open(*args, **kwargs)

        monkeypatch.setattr(audit_module.os, "fsync", fsync_then_enospc)
        monkeypatch.setattr(builtins, "open", open_that_dies_in_the_handler)
        with pytest.raises(KeyboardInterrupt):
            trail.log("record", {"i": 2})
        monkeypatch.setattr(builtins, "open", real_open)
        monkeypatch.setattr(audit_module.os, "fsync", real_fsync)

        trail.note_write_failure()
        trail.log("record", {"i": 3})

        entries = [
            json.loads(line)
            for line in (tmp_path / "resid.audit.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        seqs = [e["seq"] for e in entries]
        r = AuditTrail.verify(str(db))

        # ⛔ THE PRECONDITIONS ARE HARD FAILURES, NOT PART OF THE xfail.
        # This gate used ``@pytest.mark.xfail(strict=True)``, which treats
        # ANY failure anywhere in the test as the expected one — so if the
        # residual were CLOSED but the fixture, the retry or the parsing
        # developed a different bug, CI would still print XFAIL and the
        # promised XPASS notification would never arrive. **A gate whose
        # green covers every possible red is not reporting on its subject.**
        # Named by codex (L3, 2026-09-07).
        assert phase.get("fired") == 1, (
            "the interrupt never landed inside the truncate, so this test "
            "did not reach the window it exists to describe"
        )

        # ✅ CLOSED 2026-09-07 (same day it was opened). Kept as a POSITIVE
        # assertion rather than deleted: the window is narrow, reachable only
        # by an ordinary I/O failure plus a terminal signal in a specific
        # place, and nothing else in the suite would notice it reopening.
        #
        # ⚡ THIS GATE ANNOUNCED ITS OWN CLOSURE. It had been rewritten hours
        # earlier so that anything OTHER than the exact known-bad signature
        # fails loudly — and on its first real occasion it printed
        # "the known-open residual did NOT reproduce. THIS IS THE
        # NOTIFICATION" with seqs [0,1,2,3] and valid=True. A blanket
        # ``xfail(strict=True)`` would have swallowed the good news as an
        # expected failure.
        assert len(seqs) == len(set(seqs)), (
            f"the closed residual REOPENED — duplicate seqs on disk: {seqs}"
        )
        assert r.valid, (
            f"the closed residual REOPENED — one ordinary I/O failure plus "
            f"one terminal signal inside the truncate produced a false "
            f"tampering verdict again: seqs {seqs}, {r.error}"
        )
        return

    def test_the_guarded_region_cannot_be_widened_into_something_self_touching(
        self,
    ):
        """Structural: the rollback is only sound because the region is tiny.

        The handler restores a snapshot of ``_prev_hash`` / ``_seq`` /
        ``_dropped_since_last``. That is correct ONLY while nothing inside
        the guarded ``try`` can legitimately change them — otherwise the
        restore would UNDO a real change instead of an aborted one. Today
        that holds because the region contains three stores and five calls
        (``open``, ``write``, ``flush``, ``fsync``, ``fileno``), none of
        which can reach ``self``.

        That is a property of the region's SHAPE, so it is asserted rather
        than commented: a future edit that moves a store out, or that calls
        a method on ``self`` in there (a rotation, a re-seed, anything
        re-entering ``log()``), silently invalidates the rollback and is
        exactly the kind of change nobody re-derives this argument for.

        ⛔ MUTATION-CHECKED 2026-09-07, FOUR mutants, each verified present
        on disk by re-parsing before the test was run, selected alone:

          move ``self._seq += 1`` below the handler ... the OUTSIDE-the-try
            assertion fires
          ``self._initialize()`` inside the ``try`` ... the allow-list
            assertion fires
          ``finally: self._rotate_if_needed()`` on the try ... the
            no-else/finally assertion fires
          ``(self._seq, self._prev_hash) = (...)`` outside the try ... the
            OUTSIDE-the-try assertion fires, via the tuple-unpack path

        ⚠ THE LAST TWO WERE HOLES IN THIS TEST UNTIL 2026-09-07, both found
        by L1 and both demonstrated with a mutant that left it GREEN. They
        are listed here as arms rather than as history because a gate's
        recipe should name every hole that was ever in it — the next edit
        that reintroduces one will reintroduce it the same way.
        """
        import ast
        import inspect

        import anneal_memory.audit as audit_module

        src = inspect.getsource(audit_module)
        tree = ast.parse(src)
        cls = next(
            n for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "AuditTrail"
        )
        log = next(
            n for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "log"
        )
        guarded = [
            n for n in ast.walk(log)
            if isinstance(n, ast.Try)
            and any(
                isinstance(h.type, ast.Name) and h.type.id == "BaseException"
                for h in n.handlers
            )
        ]
        assert len(guarded) == 1, (
            f"expected exactly one BaseException-guarded try in log(), "
            f"found {len(guarded)}"
        )
        try_node = guarded[0]
        lo = try_node.body[0].lineno
        hi = try_node.body[-1].end_lineno

        # ⛔ NO ``else:`` / ``finally:`` ON THIS TRY. Both run outside the
        # body the allow-list below walks, so either one is a hole straight
        # through this test — measured 2026-09-07: adding
        # ``finally: self._rotate_if_needed()``, the exact hazard this
        # docstring names, left the invariant GREEN. Refusing the clauses
        # outright beats walking them, because it cannot be half-done.
        assert not try_node.orelse and not try_node.finalbody, (
            "the guarded try grew an else/finally clause. Code there runs "
            "outside the region this test checks, so it can reach `self` "
            "unseen. Put it before the try or after the whole statement."
        )

        trio = {"_prev_hash", "_seq", "_dropped_since_last"}

        # ⛔ ENUMERATE THE NODE TYPES THE PROPERTY CAN BE EXPRESSED IN, NOT
        # THE SCENARIOS. Mutation-checking the arms that exist cannot find a
        # MISSING arm: every mutant anyone wrote for this gate happened to
        # be an ``Assign`` or an ``AugAssign``, so the gate scored perfectly
        # while being blind to most of Python's store forms. codex (L3,
        # 2026-09-07) enumerated what it could not see: ``AnnAssign``,
        # ``For``/``AsyncFor``, ``With``/``AsyncWith``, comprehension and
        # walrus targets, ``Delete``, ``Starred``, nesting deeper than one
        # level, ``self.__dict__[...]`` subscript writes, and
        # ``setattr``/``object.__setattr__``. **Each is a store to the trio
        # that would have left this gate GREEN.**
        def _leaves(tgt):
            """Every leaf target, through arbitrary nesting."""
            if isinstance(tgt, (ast.Tuple, ast.List)):
                out = []
                for e in tgt.elts:
                    out.extend(_leaves(e))
                return out
            if isinstance(tgt, ast.Starred):
                return _leaves(tgt.value)
            return [tgt]

        def trio_targets(node):
            """Every trio attribute this statement stores to.

            ⚠ TUPLE-UNPACK IS NOT OPTIONAL TO HANDLE: the handler's own
            restore is written that way, so it is the local idiom a future
            edit copies. Matching only ``ast.Attribute`` left
            ``(self._seq, self._prev_hash) = (...)`` invisible anywhere in
            ``log()`` — measured green 2026-09-07.
            """
            if isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
                raw = _leaves(node.target)
            elif isinstance(node, ast.Assign):
                raw = []
                for tgt in node.targets:
                    raw.extend(_leaves(tgt))
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                raw = _leaves(node.target)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                raw = []
                for item in node.items:
                    if item.optional_vars is not None:
                        raw.extend(_leaves(item.optional_vars))
            elif isinstance(node, ast.Delete):
                raw = []
                for tgt in node.targets:
                    raw.extend(_leaves(tgt))
            else:
                return []
            return [
                t.attr for t in raw
                if isinstance(t, ast.Attribute)
                and t.attr in trio
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
            ]

        def walk_this_scope(root):
            """``ast.walk``, but never descending into a NESTED scope.

            ⛔ ``ast.walk`` descends ``FunctionDef``/``Lambda``/``ClassDef``,
            so a trio store inside a nested closure — which does NOT run when
            ``log()`` runs — counted toward ``stored_inside`` and could
            satisfy the ``== trio`` assertion on its own. A gate satisfied by
            code that never executes is worse than no gate.
            """
            todo = [root]
            while todo:
                n = todo.pop()
                for child in ast.iter_child_nodes(n):
                    if isinstance(
                        child,
                        (ast.FunctionDef, ast.AsyncFunctionDef,
                         ast.Lambda, ast.ClassDef),
                    ):
                        continue
                    todo.append(child)
                    yield child

        # ⛔ NO INDIRECT MUTATION. These reach the trio without ever
        # producing an ``ast.Attribute`` target, so every name-based check
        # above is blind to them by construction.
        for node in walk_this_scope(log):
            if isinstance(node, ast.Call):
                fname = ast.unparse(node.func)
                assert fname not in {"setattr", "object.__setattr__"}, (
                    f"{fname}() inside log() can store to the trio without "
                    f"an Attribute target, making every check in this test "
                    f"blind to it. Assign directly."
                )
            if isinstance(node, ast.Attribute) and node.attr == "__dict__":
                raise AssertionError(
                    "log() reaches self.__dict__ — a subscript write there "
                    "mutates the trio invisibly to this gate. Assign to the "
                    "attribute directly."
                )

        stored_inside = set()
        for node in walk_this_scope(log):
            attrs = trio_targets(node)
            if not attrs:
                continue
            # the handler's restore is allowed to store — that is its job
            if any(
                h.lineno <= node.lineno <= h.end_lineno
                for h in try_node.handlers
            ):
                continue
            for attr in attrs:
                assert lo <= node.lineno <= hi, (
                    f"self.{attr} is assigned at line {node.lineno}, "
                    f"OUTSIDE the guarded try ({lo}-{hi}). An interrupt "
                    f"there advances the chain state with no rollback — the "
                    f"2026-09-06 defect, reintroduced."
                )
                stored_inside.add(attr)
        assert stored_inside == trio, (
            f"the guarded region advances {sorted(stored_inside)} but the "
            f"rollback restores {sorted(trio)} — the two sets must match or "
            f"the restore is either incomplete or clobbering"
        )

        allowed = {"open", "f.write", "f.flush", "os.fsync", "f.fileno"}
        for node in ast.walk(try_node):
            if not isinstance(node, ast.Call):
                continue
            if not (lo <= node.lineno <= hi):
                continue
            name = ast.unparse(node.func)
            assert name in allowed, (
                f"a new call {name!r} appeared inside the guarded try at "
                f"line {node.lineno}. If it can reach `self`, the handler's "
                f"snapshot restore may now UNDO a legitimate change rather "
                f"than an aborted one. Widen this allow-list only after "
                f"establishing that it cannot touch "
                f"_prev_hash / _seq / _dropped_since_last."
            )


class TestAZeroByteActiveFileIsNotAnActiveFile:
    """L2 (L3 domain lens), 2026-09-07 — the rollback can create this state."""

    def test_a_rolled_back_first_append_does_not_restart_the_chain(
        self, tmp_path
    ):
        """A zero-byte active file must fall through to the manifest.

        ``log()``'s rollback truncates back to the pre-append size. When the
        failing append is the FIRST write into a freshly rotated file that
        size is 0, so a successful rollback leaves a zero-byte active file
        on disk. ``_initialize`` tested ``active.exists()`` alone, read that
        as "an active file with entries", skipped the manifest continuity
        branch, and kept ``_prev_hash = GENESIS`` — while the sealed files
        ended somewhere else entirely. The next process then wrote seq 0
        chained from GENESIS and ``verify()`` cried tampering.

        ⚖ The same file already had the right predicate in
        ``_rotate_if_needed`` (``not exists() or st_size == 0``). Two places
        computing one thing, disagreeing exactly where the rollback puts
        you. **This is a false tampering verdict produced by the rollback
        SUCCEEDING**, which is why it is graded here rather than filed.

        ⛔ MUTATION-CHECKED 2026-09-07, verified on disk: drop the
        ``or active.stat().st_size == 0`` clause from ``_initialize`` and
        this test fails with ``Hash mismatch at seq 0 ... got
        sha256:GENESIS...``.
        """
        from anneal_memory.audit import AuditTrail

        db = tmp_path / "zero.db"
        trail = AuditTrail(db)
        for i in range(3):
            trail.log("before", {"i": i})

        # force the weekly rotation, then land one entry in the new file
        trail._last_week = "1999-W01"
        trail.log("after_rotation", {})
        active = trail._active_path
        assert active.stat().st_size > 0

        # what a rolled-back first-append-into-a-fresh-file leaves behind
        active.write_text("")
        assert active.exists() and active.stat().st_size == 0

        AuditTrail(db).log("next_process", {})

        result = AuditTrail.verify(str(db))
        assert result.valid, (
            f"a zero-byte active file restarted the chain from GENESIS "
            f"instead of continuing from the manifest: {result.error}"
        )


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

    def test_a_torn_tail_inside_a_sealed_gz_file_is_skipped_not_raised(
        self, tmp_path
    ):
        """The gzip branch of ``_iter_lines`` carries the identical
        conflation the plain-text branch had, and diogenes (2026-09-08)
        flagged it as NEVER EXERCISED: the HIGH measured only the
        plain-text branch, "the ``gzip.open(..., 'rt')`` branch carries the
        identical conflation by inspection and I did not plant an
        undecodable byte in a sealed ``.gz``". An unexercised branch of the
        same defect is how the class survives its own fix.

        A torn multibyte tail inside a SEALED (already-rotated) file must
        not raise ``UnicodeDecodeError`` out of ``verify()`` any more than
        one in the active file does.
        """
        db = tmp_path / "sealed.db"
        trail = AuditTrail(db)
        trail.log("record", {"id": "1"})
        trail._last_week = "1999-W01"
        trail.log("record", {"id": "2"})       # forces rotation, seals W01

        gz_files = list(tmp_path.glob("*.audit.1999-W01.jsonl.gz"))
        assert len(gz_files) == 1
        gz_path = gz_files[0]

        content = gzip.decompress(gz_path.read_bytes())
        torn = '{"v":1,"seq":9,"ts":"2026-09-08T00:00:00.0000⛔'.encode("utf-8")[:-1]
        gz_path.write_bytes(gzip.compress(content + b"\n" + torn))

        result = AuditTrail.verify(db)  # must NOT raise
        assert result.skipped_lines >= 1, (
            "a torn multibyte tail inside a sealed .gz file must be "
            "counted as a skipped line, not silent or fatal"
        )


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
        Exception`` does not turn THIS test red — the interrupt escapes the
        ``with`` block and ABORTS the pytest run at that line.

        ⛔ AMENDED 2026-09-06 (Diogenes MED, tests/test_audit.py). That abort
        is a fact about THIS coordinate, not about the property, and the
        paragraph above originally read as though a red test were therefore
        unavailable. It is not: the interrupt is catchable one frame out, at
        the ``validated_save_continuity`` CALL rather than at the batch. What
        this test grades is that the batch exited and the caller's DML
        survived — it never reads the continuity file, which is what the HIGH
        is about. The regression gate for THAT is
        ``test_an_interrupt_in_the_replay_cannot_destroy_a_committed_wrap``
        below; it asserts on the file and it does go red.
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

    def test_an_interrupt_in_the_replay_cannot_destroy_a_committed_wrap(
        self, tmp_path
    ):
        """Diogenes MED, 2026-09-06 — the same guard, graded on the ARTIFACT.

        The sibling test above drives ``store._batch()`` directly and asserts
        ``total_episodes == 1``. That grades whether the batch exited and
        whether the caller's DML survived. It never reads the continuity
        file — and a destroyed continuity file is the entire loss the HIGH
        describes. Removing the guard does not turn it red, so the property
        had no regression gate at all: a green check that cannot fail for the
        reason it exists (``a_gate_nothing_can_pass_is_not_a_gate``, this
        repo, 2026-09-05).

        This one drives the CANONICAL pipeline — ``prepare_wrap`` ->
        ``validated_save_continuity`` — with ``_audit.log`` raising
        ``KeyboardInterrupt`` on its first call of the second wrap. That call
        lands inside ``_replay_deferred_audits``, the post-commit window the
        guard exists to contain (verified by stack trace at
        ``store.py`` ``_replay_deferred_audits`` -> ``_audit_log_after_commit``
        -> ``self._audit.log``, not assumed from the call order).

        ⛔ MUTATION RECIPE — THE SINGLE-SITE FORM THIS DOCSTRING CARRIED
        UNTIL 2026-09-07 WAS FALSE, AND IT IS THE RECIPE, NOT THE GATE, THAT
        WAS BROKEN. It read: narrow ``_batch()``'s post-commit
        ``except BaseException`` to ``except Exception`` and this test goes
        red. Run verbatim it returns ``1 passed``. The claim was true when
        written (``7de2007``) and two commits LATER IN THE SAME WINDOW —
        ``926be6a`` widening ``_audit_log_after_commit``'s catch, ``200382d``
        adding the per-event catch in ``_replay_deferred_audits`` — each
        independently contains the interrupt before it can reach ``_batch``.

        ▶ WHY NO SINGLE-SITE MUTATION CAN WORK, which is the part worth
        keeping: the three handlers are a NESTED CONTAINMENT CHAIN on one
        path, so only the innermost ever fires and defeating any one of them
        just hands the interrupt to the next. The containment chain, from
        outermost to innermost:

            validated_save_continuity   (continuity.py)
            _batch                      (store.py, its POST-COMMIT handler)
            _replay_deferred_audits     (store.py, the per-event handler)
            _audit_log_after_commit     (store.py, innermost)

        ⛔ **DERIVE THE COORDINATES; DO NOT READ THEM FROM HERE.** This
        docstring carried literal line numbers until 2026-09-07 and **every
        one of them was stale** — it named ``store.py`` 5212 / 5798 / 5858
        while the handlers had moved to 5242 / 5827 / 5887. Following it
        verbatim narrows a COMMENT and an ASSIGNMENT, changes nothing, and
        returns the reassuring green this docstring warns about. Caught by
        codex (L3, 2026-09-07). ⚡ **A stored coordinate is an answer, and
        answers rot; this file moves on almost every touch.**
        ⚠ **AND A BARE ``grep -n "except BaseException" store.py`` IS NOT
        ENOUGH EITHER** — ``_batch`` has THREE of them and only the
        post-commit one (the LAST in the function) is on this path. Use:

            python3 - <<'EOF'
            import ast, pathlib
            src = pathlib.Path("anneal_memory/store.py").read_text()
            want = {"_batch", "_replay_deferred_audits",
                    "_audit_log_after_commit"}
            for n in ast.walk(ast.parse(src)):
                if isinstance(n, ast.FunctionDef) and n.name in want:
                    for h in ast.walk(n):
                        if (isinstance(h, ast.ExceptHandler) and h.type
                                and "BaseException" in ast.unparse(h.type)):
                            print(n.name, h.lineno)
            EOF

        ⛔ THE WORKING RECIPE, RUN 2026-09-07, ALL FOUR ARMS, THIS TEST
        SELECTED ALONE (``-k cannot_destroy_a_committed_wrap``). Narrow
        ``except BaseException`` -> ``except Exception`` at:

            the _batch post-commit handler ALONE ...... 1 passed
            innermost + per-event ..................... 1 passed
            ALL THREE ................................ 1 FAILED
            control (no mutation) .................... 1 passed

        The red arm fails on the assertion below: the interrupt escapes the
        batch, ``validated_save_continuity`` never reaches
        ``db_committed = True``, its own handler unlinks BOTH staged
        sidecars, and the continuity file is left holding the PREVIOUS
        session's text while the wrap row and the episodes' wrap assignments
        stay durable.

        ⚠ VERIFY THAT YOUR MUTATION APPLIED, BY READING IT BACK OFF DISK.
        Two of the three sites are ``except BaseException:`` and one is
        ``except BaseException as exc:``; a mutator that string-matches the
        first form silently no-ops on the third site and reports success,
        and the resulting ``1 passed`` is indistinguishable from containment
        working. That is not hypothetical — it happened while this docstring
        was being repaired, and it produced a confident, wrong contradiction
        of a correct finding.

        ⚠ RE-RUN WITH THIS TEST SELECTED ALONE, by node id or
        ``-k cannot_destroy_a_committed_wrap``. Under the mutant the SIBLING
        test above aborts the whole pytest session before this one is
        reached, so a mutation run that selects both (``-k interrupt``)
        reports an abort and looks like this test cannot go red either.

        ▶ NOT SPLIT INTO PER-LAYER ARMS, and the reason is structural rather
        than effort: because the layers are nested on ONE path, no injection
        point exists that only one of them can contain. An arm that graded a
        single layer would have to assert that layer's own distinctive side
        effect (the drop is RECORDED, the replay CONTINUES, the sidecars are
        NOT unlinked) rather than that the wrap survived — a different test
        with a different subject. Worth building; it is not this test.
        """
        from anneal_memory import prepare_wrap, validated_save_continuity
        from anneal_memory.store import EpisodeType, Store

        def continuity_text(marker: str) -> str:
            return (
                "# Interrupt — Memory (v1)\n\n"
                f"## State\n{marker}\n\n"
                "## Patterns\nNone yet.\n\n"
                "## Decisions\nNone.\n\n"
                f"## Context\n{marker}\n"
            )

        db = tmp_path / "memory.db"
        store = Store(str(db), project_name="Interrupt")
        try:
            # Wrap 1 is not scaffolding — it is what makes the mutant
            # distinguishable. Without a prior wrap the regression leaves NO
            # continuity file, and a weaker assertion ("the file exists")
            # would report the loss as a pass. With one, the destroyed wrap
            # shows up as the OLD text sitting where the NEW text belongs,
            # which is the loss the CHANGELOG actually describes.
            store.record("first observation", EpisodeType.OBSERVATION)
            prepare_wrap(store)
            validated_save_continuity(store, continuity_text("OLDTEXT"))
            assert "OLDTEXT" in store.continuity_path.read_text(
                encoding="utf-8"
            ), "wrap 1 did not land — the baseline for wrap 2 is not set up"

            store.record("second observation", EpisodeType.OBSERVATION)
            prepare_wrap(store)

            real_log = store._audit.log
            calls = {"n": 0}

            def interrupt_once(*args, **kwargs):
                # ONE-SHOT on purpose. The first audit emit of this wrap is
                # the deferred ``wrap_completed`` replay; later emits must run
                # normally or the arms stop discriminating. Phase 4's
                # post-rename ``continuity_saved`` site swallows only
                # ``Exception``, so a second interrupt would escape THERE and
                # the failure would no longer name the handler under test.
                calls["n"] += 1
                if calls["n"] == 1:
                    raise KeyboardInterrupt("Ctrl+C during the audit replay")
                return real_log(*args, **kwargs)

            escaped: BaseException | None = None
            store._audit.log = interrupt_once
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    validated_save_continuity(store, continuity_text("NEWTEXT"))
            except BaseException as exc:  # noqa: BLE001
                # DELIBERATE, and it is the whole reason this test can go red
                # where its sibling cannot. An uncaught ``KeyboardInterrupt``
                # ABORTS the pytest run rather than failing a test; catching
                # it one frame out converts the regression into the
                # assertions below. Nothing is swallowed — ``escaped`` is
                # asserted on.
                escaped = exc
            finally:
                store._audit.log = real_log

            assert calls["n"] >= 1, (
                "the interrupt never fired — the post-commit replay no longer "
                "emits an audit event, so this test is not exercising the "
                "guard it was written for"
            )
            assert store.continuity_path.exists(), (
                "the continuity file is gone entirely after an interrupted "
                "wrap"
            )
            body = store.continuity_path.read_text(encoding="utf-8")
            assert "NEWTEXT" in body and "OLDTEXT" not in body, (
                "a post-commit interrupt DESTROYED a committed wrap: the "
                "continuity file still holds the previous session's text "
                "while the wrap row and the episodes' wrap assignments are "
                "durable. This is the data-loss path the post-commit "
                "``except BaseException`` in ``_batch()`` exists to close."
            )
            assert escaped is None, (
                f"{type(escaped).__name__} escaped the pipeline after the "
                "batch committed; the caller cannot tell that from a batch "
                "that failed, which is what triggers the sidecar unlink"
            )
        finally:
            store.close()

        reopened = Store(str(db))
        try:
            assert reopened.status().total_episodes == 2, (
                "the committed episodes did not survive — a post-commit "
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
            # ⛔ codex L3 MED, 2026-09-06. The row above it was already here
            # and passed for the WRONG REASON: it happens to omit the word
            # "database", so the old two-substring test never fired on it. Put
            # "database" INTO the identifier and the same predicate classified
            # a schema error as write-lock contention, and the operator was
            # told another process was writing right now. The list looked like
            # it covered this class and covered one spelling of it.
            "no such table: database_locked_items",
            "no such column: database_is_locked",
            "table locked_database has no column named x",
        ]
        for msg in not_contention:
            assert not _is_write_lock_contention(_NoCode(msg)), (
                f"non-contention classified as contention: {msg!r}"
            )

    # -- #4h2: a terminal exception must not silently eat the replay tail --

    def test_an_interrupt_in_the_replay_does_not_silently_eat_the_tail(
        self, tmp_path
    ):
        """codex L3 HIGH, 2026-09-06 — the swallow HAD become silence.

        ``_audit_log_after_commit``'s per-event catch was ``except
        Exception``, so a ``KeyboardInterrupt`` from the sink walked past it
        and out of ``_replay_deferred_audits``. ``_batch()``'s post-commit
        handler caught it, warned once, and ABANDONED THE REST OF THE QUEUE.

        MEASURED BEFORE THE FIX, with a four-episode batch: four episodes
        committed, ONE audit emit attempted, ``audit_write_failures`` 0,
        nothing pending, and no audit file at all. Every one of the four
        "swallow must not become silence" channels stayed quiet because none
        of them ran, and ``verify()`` would have walked the hole clean.

        Two properties, and the second is the one the four channels exist for:
        the tail is still written (a real Ctrl-C is delivered ONCE, so
        containing it per event costs the one interrupted emit), and the drop
        is RECORDED in the hash chain rather than only as a warning.

        ⚠ "RECORDED IN THE CHAIN" CARRIES ``spore-774``'s BOUND AND IS NOT THE
        SAME AS "DURABLE". The ``dropped_before`` marker rides into the NEXT
        entry that lands, so it survives the process only once a later write
        actually happens — a crash before that still loses it. What this test
        pins is that the marker is EMITTED at all, which it was not before:
        the pre-fix measurement had zero entries and zero counters. The field
        is best-effort by design; do not restate it as durable.
        """
        import json

        from anneal_memory.store import Store

        db = tmp_path / "tail.db"
        store = Store(db)
        real_log = store._audit.log
        attempts = {"n": 0}

        def interrupt_the_first_emit(*args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise KeyboardInterrupt("Ctrl+C on the first replayed event")
            return real_log(*args, **kwargs)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with store._batch():
                    for i in range(4):
                        store.record(f"episode {i}", episode_type="observation")
                    store._audit.log = interrupt_the_first_emit
            store._audit.log = real_log

            assert store.status().total_episodes == 4, (
                "the committed episodes did not survive"
            )
            assert attempts["n"] > 1, (
                "the replay stopped at the interrupted event — every audit "
                f"event after it was dropped (attempted {attempts['n']} of 4)"
            )
            assert store.status().audit_write_failures == 1, (
                "the dropped audit event left the failure counter at zero, so "
                "a caller polling for degraded audit health sees a clean trail "
                "over a real hole"
            )
        finally:
            store._audit.log = real_log
            store.close()

        entries = [
            json.loads(line)
            for line in (tmp_path / "tail.audit.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        assert entries, "the tail was never written"
        assert any(e.get("dropped_before") for e in entries), (
            "no dropped_before marker rode into the chain — the ONLY channel "
            "that outlives the process is the hash-chained one, and without it "
            "verify() walks the gap and reports a clean trail"
        )

    # -- #4h3: WHERE a terminal exception is suppressed is a policy (09-06) --

    def test_the_terminal_exception_policy_is_split_by_call_site(self, tmp_path):
        """codex L3 MED x2, 2026-09-06 — round 1's fix over-applied.

        Widening ``_audit_log_after_commit``'s catch to ``BaseException`` was
        right for RECORDING and wrong for POLICY: it swallowed an explicit
        termination request on EVERY post-commit call. An unbatched
        ``record()`` whose sink raised ``SystemExit`` committed, recorded, and
        returned — a SIGTERM handler written as ``sys.exit()`` eaten, and the
        server running on until something killed it. That is the fail-open
        ``_persist_audit_health`` refuses one level down, acquired by the same
        edit that fixed the replay tail.

        Three behaviours, and they must hold TOGETHER — each one alone is
        satisfied by a wrong fix:

        A. unbatched + ``SystemExit`` → PROPAGATES, after recording. A
           termination request is not the audit layer's to eat.
        B. unbatched + ``KeyboardInterrupt`` → still swallowed. That trade
           dates to 2026-09-05 and reversing it reintroduces the data-loss
           defect that was mutation-tested and refused.
        C. batched + ``SystemExit`` → suppressed AND the tail still replayed.
           The batched path is the one place a raise makes the caller read a
           committed wrap as failed and unlink its staged sidecars, so
           suppression lives at that call site, visibly, rather than in the
           shared handler on everyone's behalf.
        """
        from anneal_memory.store import Store

        def sink_raising(exc):
            def raise_it(*args, **kwargs):
                raise exc
            return raise_it

        # A -- the termination request must get out.
        store = Store(tmp_path / "a.db")
        store._audit.log = sink_raising(SystemExit(3))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(SystemExit):
                    store.record("unbatched", episode_type="observation")
            assert store.status().total_episodes == 1, (
                "the episode was rolled back; the exit must not undo a "
                "committed write"
            )
            assert store.status().audit_write_failures == 1, (
                "the drop was not recorded before the exit propagated"
            )
        finally:
            store._audit.log = None
            store.close()

        # B -- Ctrl-C stays swallowed.
        store = Store(tmp_path / "b.db")
        store._audit.log = sink_raising(KeyboardInterrupt("ctrl-c"))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                store.record("unbatched", episode_type="observation")
        finally:
            store._audit.log = None
            store.close()

        # C -- batched: suppressed, and the tail is still replayed.
        store = Store(tmp_path / "c.db")
        real_log = store._audit.log
        attempts = {"n": 0}

        def exit_on_the_first_emit(*args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise SystemExit(0)
            return real_log(*args, **kwargs)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with store._batch():
                    for i in range(4):
                        store.record(f"episode {i}", episode_type="observation")
                    store._audit.log = exit_on_the_first_emit
            store._audit.log = real_log
            assert attempts["n"] == 4, (
                "a SystemExit on the first replayed event abandoned the tail "
                f"(attempted {attempts['n']} of 4)"
            )
            assert store.status().total_episodes == 4
            assert store.status().audit_write_failures == 1
        finally:
            store._audit.log = real_log
            store.close()

    # -- #4i: the ordering key must be a TIME, not a 20-char string (09-06) --

    def test_a_corrupt_stamp_cannot_pin_the_last_failure_forever(self):
        """codex L3 LOW, 2026-09-06 — the shape check accepted impossible dates.

        ``_failure_stamp`` tested for "20 characters, ends in Z, hyphen at
        index 4" and returned the STRING for a lexicographic compare. So
        ``9999-99-99T99:99:99Z`` passed, and it string-compares GREATER than
        every real timestamp — one corrupt stored value pinned
        ``audit_last_failure`` permanently, which is the exact opposite of the
        function's documented "fails toward REPLACING an unreadable stored
        value".
        """
        from anneal_memory.store import Store

        for impossible in (
            "9999-99-99T99:99:99Z",
            "2026-13-45T99:99:99Z",
            "2026-02-30T00:00:00Z",
        ):
            assert Store._failure_stamp(f"record: X at {impossible}") is None, (
                f"{impossible!r} passed the stamp check; it compares greater "
                "than every real timestamp and pins the field forever"
            )

    def test_two_failures_in_one_second_are_ordered(self):
        """codex L3 MED, 2026-09-06 — second resolution made the guard a tie.

        The stale-writer guard exists so a writer flushing an OLD failure
        cannot overwrite a newer one. At second resolution any two failures
        inside the same second compared EQUAL, ``theirs > mine`` was False,
        and the older one won by falling through to the replace branch —
        exactly the overwrite the guard was written to prevent.

        ⚠ THE LEGACY FORM MUST STILL ORDER CORRECTLY AGAINST THE NEW ONE, and
        this is why the comparison is on parsed datetimes rather than text: in
        a string compare ``Z`` sorts after ``.``, so a stored
        ``...:00:00Z`` would compare GREATER than a newer
        ``...:00:00.500000Z`` and win the second it was supposed to lose.
        """
        from anneal_memory.store import Store

        early = Store._failure_stamp("record: A at 2026-09-06T10:00:00.100000Z")
        late = Store._failure_stamp("record: B at 2026-09-06T10:00:00.900000Z")
        assert early is not None and late is not None
        assert early < late, (
            "two failures in the same second did not order — the stale-writer "
            "guard cannot fire inside one second"
        )

        legacy = Store._failure_stamp("record: OLD at 2026-09-06T10:00:00Z")
        assert legacy is not None, (
            "a stamp written before 2026-09-06 no longer parses; stores in the "
            "wild hold this form"
        )
        assert legacy < late, (
            "a legacy second-resolution stamp compared NEWER than a sub-second "
            "one from later in the same second"
        )

    def test_status_reports_the_genuinely_newer_failure_not_just_the_local_one(
        self, tmp_path
    ):
        """codex L3 MED, 2026-09-06 — the stale-writer class, on the READ side.

        ``_current_audit_last_failure`` returned this process's own unflushed
        failure whenever it had one, on the written premise that it "is
        strictly newer than anything already committed". False as soon as a
        second writer exists: A's flush fails and its failure stays pending, B
        persists a LATER one, and A's ``status()`` kept reporting its own older
        string without ever reading the row — the forensic pointer naming the
        wrong final loss.

        Both directions are asserted, because returning the stored value
        unconditionally would pass the first half and be just as wrong.
        """
        from anneal_memory.store import Store, _AUDIT_LAST_FAILURE_KEY

        db = tmp_path / "status.db"
        store = Store(db)
        try:
            store.record("seed", episode_type="observation")
            store._audit_failures_unpersisted = 1

            def persist_from_another_writer(value):
                other = Store(db)
                try:
                    other._conn.execute(
                        "INSERT OR REPLACE INTO metadata (key, value) "
                        "VALUES (?, ?)",
                        (_AUDIT_LAST_FAILURE_KEY, value),
                    )
                    other._conn.commit()
                finally:
                    other.close()

            persist_from_another_writer(
                "record: THEIRS at 2026-09-06T10:00:05.000000Z"
            )

            store._audit_last_failure = (
                "record: MINE at 2026-09-06T10:00:00.100000Z"
            )
            assert "THEIRS" in (store.status().audit_last_failure or ""), (
                "status reported this process's older pending failure while a "
                "newer one was already persisted by another writer"
            )

            store._audit_last_failure = (
                "record: MINE at 2026-09-06T10:00:09.000000Z"
            )
            assert "MINE" in (store.status().audit_last_failure or ""), (
                "status ignored a genuinely newer local failure in favour of "
                "the stored one"
            )
        finally:
            store.close()

    def test_mixed_precision_in_one_second_does_not_clobber_the_stored_value(
        self, tmp_path
    ):
        """codex L3 MED, 2026-09-06 — a gap opened by the precision fix itself.

        Writing microseconds fixed same-second ties between two NEW writers and
        created a new hazard for a MIXED-VERSION fleet: a legacy stamp parses
        as ``.000000``, which is not the same as being older — it is UNKNOWN
        within its second. So a new writer overwrote a genuinely NEWER failure
        persisted by an old one: A fails at ``10:00:00.100000`` and stays
        pending, B fails LATER and persists ``10:00:00Z``, A flushes, reads B
        as ``.000000`` and clobbers it. The stale-writer race this guard exists
        to prevent, reopened by the fix meant to close it.

        ⚠ This is the ONE case that does not fail toward replacing. The general
        direction exists so an UNREADABLE stored value cannot pin the field; a
        READABLE value of coarser precision cannot be shown to be older, so it
        is kept. The unambiguous cases are asserted alongside so the exception
        cannot quietly widen into "never replace".
        """
        from anneal_memory.store import Store, _AUDIT_LAST_FAILURE_KEY

        def stored_is_newer(stored, candidate):
            store = Store(tmp_path / f"p{abs(hash((stored, candidate)))}.db")
            try:
                store.record("seed", episode_type="observation")
                store._conn.execute(
                    "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                    (_AUDIT_LAST_FAILURE_KEY, f"record: X at {stored}"),
                )
                store._conn.commit()
                return store._stored_failure_is_newer(f"record: Y at {candidate}")
            finally:
                store.close()

        assert stored_is_newer(
            "2026-09-06T10:00:00Z", "2026-09-06T10:00:00.100000Z"
        ), (
            "a sub-second writer overwrote a legacy stamp from the same second "
            "— coarser precision was read as older, which it is not"
        )

        # The unambiguous cases must be unaffected.
        assert stored_is_newer(
            "2026-09-06T10:00:00.900000Z", "2026-09-06T10:00:00.100000Z"
        )
        assert not stored_is_newer(
            "2026-09-06T10:00:00.100000Z", "2026-09-06T10:00:00.900000Z"
        )
        assert not stored_is_newer(
            "2026-09-06T09:00:00Z", "2026-09-06T10:00:00.100000Z"
        ), "a legacy stamp from an EARLIER second must still lose"
        assert stored_is_newer(
            "2026-09-06T11:00:00Z", "2026-09-06T10:00:00.100000Z"
        ), "a legacy stamp from a LATER second must still win"

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

    # -- #4h: and close() is the ONE site that re-raise changes (09-06) --

    def test_a_systemexit_in_the_close_flush_still_closes_the_connection(
        self, tmp_path
    ):
        """Diogenes MED, 2026-09-06 — the call site #4g's rationale singles out.

        ``_persist_audit_health`` re-raises ``SystemExit``. That re-raise's own
        comment says sites 1 and 2 are unaffected (both sit under
        ``_batch()``'s post-commit ``except BaseException``) and that
        ``close()`` is the one path that changes. ``close()`` called it BEFORE
        ``self._conn.close()``, under a comment promising the flush "cannot
        turn a clean close into a raising one" — true when written, false
        twelve commits later in the same window.

        MEASURED BEFORE THE FIX: the exit skipped the sqlite close entirely.
        ``self._closed`` stayed False and the handle stayed USABLE, so a
        caller that catches the exit — a CLI wrapper, a test, an embedding
        app — inherits a live connection holding its locks while the store
        reports itself open.

        BOTH halves are asserted, because either one alone can be satisfied by
        the wrong fix: swallowing the exit would close the handle and lose the
        termination request; leaving it as it was propagates the request and
        leaks the handle.
        """
        from anneal_memory.store import Store

        store = Store(tmp_path / "exit_close.db")

        def exit_during_the_flush() -> None:
            # Stands in for a SIGTERM handler's ``sys.exit(0)`` landing inside
            # the flush window. Patched at the method rather than driven
            # through a real signal so the test is deterministic; #4g above
            # already pins that the real handler re-raises this exception.
            raise SystemExit(0)

        store._persist_audit_health = exit_during_the_flush  # type: ignore[method-assign]

        with pytest.raises(SystemExit):
            store.close()

        assert store._closed, (
            "close() left _closed False after a SystemExit in the flush — the "
            "store believes it is still open"
        )
        with pytest.raises(Exception):
            store._conn.execute("select 1")

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

    def test_a_torn_multibyte_tail_reports_skipped_not_a_traceback(
        self, tmp_path
    ):
        """The HIGH one layer out from the recovery fix (diogenes,
        2026-09-08): ``_iter_lines`` opened in text mode, so a torn
        multibyte tail raised ``UnicodeDecodeError`` straight out of
        ``verify()``'s scan loop — the one surface ``AuditTrail.verify``
        (a classmethod that never constructs a trail) cannot recover
        itself, and the one an operator runs FIRST to ask "was my log
        tampered with". It answered with a bare traceback.

        ``AuditVerifyResult`` already carries ``skipped_lines`` for exactly
        this condition; a decodable torn tail already reports it via
        ``json.JSONDecodeError``. This pins the undecodable case getting
        the same treatment instead of escaping the generator entirely.
        """
        from anneal_memory.audit import AuditTrail

        db = tmp_path / "torn_verify.db"
        store_trail = AuditTrail(db)
        for i in range(3):
            store_trail.log(f"ep-{i}", {"i": i})

        active = db.parent / "torn_verify.audit.jsonl"
        with open(active, "ab") as f:
            f.write(b"\n")
            # Cut one byte inside a multibyte character (⛔, U+26D4).
            f.write('{"v":1,"seq":9,"ts":"2026-09-08T00:00:00.0000⛔'.encode("utf-8")[:-1])

        result = AuditTrail.verify(db)  # must NOT raise

        assert result.valid is True
        assert result.skipped_lines == 1, (
            "a torn multibyte tail must be counted the same way a torn "
            "JSON line already is, not silently dropped or fatal"
        )

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


def _audit_lines(active):
    """(events, seqs) off disk, malformed lines named rather than skipped."""
    events, seqs = [], []
    for line in active.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            e = json.loads(s)
            events.append(e.get("event"))
            seqs.append(e.get("seq"))
        except json.JSONDecodeError:
            events.append(f"MALFORMED({len(s)}B)")
            seqs.append(f"MALFORMED({len(s)}B)")
    return events, seqs


class TestAnAppendNeverMergesIntoATornTail:
    """L2 HIGH, filed 2026-09-07 and closed 2026-09-07 by the next seat.

    ``_read_last_valid_entry`` recovers ``seq``/``prev_hash`` from the last
    line that PARSES and leaves the unparseable tail in place. The next
    ``open(active, "a")`` then writes straight onto those bytes, and the
    merged line parses as nothing — **so the entry that is destroyed is the
    NEW one, not the torn one**, and the writer sees a successful append.

    ⛔ WHY IT WAS FILED RATHER THAN FIXED, AND WHY THAT PREMISE DID NOT
    SURVIVE RE-TESTING. The recorded reason was that the fix "is a
    recovery-time truncation — it DELETES bytes at open — which is not a
    thing to land unreviewed". That is an argument against ONE FIX SHAPE,
    and it was never an argument against the fix: the damage comes from the
    CONCATENATION, not from the fragment existing. Terminating the fragment
    costs one byte and deletes nothing.
    """

    def test_the_entry_after_a_torn_tail_survives_a_reopen(self, tmp_path):
        """The re-opening-process shape — the general case on the CLI.

        ``audit.py`` records that EVERY CLI INVOCATION OPENS AND CLOSES A
        STORE, so this is the shape an operator actually meets.

        ⛔ MUTATION-CHECKED 2026-09-07, mutant re-read off disk before the
        run: drop the ``needs_boundary`` prefix from ``payload`` in
        ``log()`` and this fails with the third event MISSING and
        ``verify()`` still reporting ``valid=True`` — which is the whole
        finding: the loss is silent.
        """
        db = tmp_path / "torn.db"
        AuditTrail(db).log("first")
        active = db.parent / "torn.audit.jsonl"
        AuditTrail(db).log("second")

        # A torn write: a partial line with NO terminating newline.
        with open(active, "a", encoding="utf-8") as f:
            f.write('{"v":1,"seq":2,"ts":"2026-09-07T00:00:00.0000')

        AuditTrail(db).log("third")          # the entry at risk
        AuditTrail(db).log("fourth")

        events, _ = _audit_lines(active)
        assert "third" in events, (
            f"the entry written after a torn tail was merged into it and "
            f"destroyed — events on disk: {events}"
        )
        assert events[:2] == ["first", "second"] and events[-1] == "fourth"

        r = AuditTrail.verify(db)
        assert r.valid, f"chain broken at {r.chain_break_at}: {r.error}"
        # ⚠ PAIRED POSITIVE — without it this test passes on a tree that
        # "fixed" the merge by DELETING the fragment, which is the fix this
        # change deliberately rejected. The evidence must still be there.
        assert r.skipped_lines == 1, (
            f"the torn bytes were removed rather than terminated "
            f"(skipped_lines={r.skipped_lines}). On a tamper-evident log a "
            f"repair that deletes is the wrong primitive — terminate the "
            f"fragment and leave it readable."
        )

    def test_the_guard_is_on_the_append_not_on_init(self, tmp_path):
        """PLACEMENT, not presence — the arm an init-time fix would fail.

        ``log()`` calls ``_initialize()`` only when ``_initialized`` is
        False, so a LONG-LIVED process that tore its own tail mid-run never
        re-initialises. A repair living in ``_initialize`` never fires for
        it and this shape stays broken; a repair on the APPEND covers both.

        ⛔ This is the arm that distinguishes the two candidate homes, and
        nothing else in this file does. ⚠ Mutation-checked BOTH ways: with
        the boundary prefix dropped it fails; with the repair moved into
        ``_initialize`` it also fails, while the sibling test above passes.
        """
        db = tmp_path / "live.db"
        trail = AuditTrail(db)
        trail.log("first")
        trail.log("second")
        active = db.parent / "live.audit.jsonl"

        assert trail._initialized is True, (
            "fixture precondition: this arm only grades placement while the "
            "trail is already initialised, so _initialize() cannot run again"
        )

        with open(active, "a", encoding="utf-8") as f:
            f.write('{"v":1,"seq":2,"ts":"2026-09-07T00:00:00.0000')

        trail.log("third")                   # SAME instance — no re-init

        events, _ = _audit_lines(active)
        assert "third" in events, (
            f"a long-lived process merged its next entry into its own torn "
            f"tail — an init-time repair cannot reach this. events: {events}"
        )


class TestAFailedRollbackDoesNotRewindMemoryPastDisk:
    """L2 HIGH, filed 2026-09-07 and closed 2026-09-07 by the next seat.

    The truncate is best-effort; the restore was unconditional. So a failed
    truncate left the entry ON DISK while memory was rewound to "nothing
    landed", and the caller's contracted retry reused the seq.

    ⛔ THE FIX RECORDED AT THE SITE WAS WRONG, AND WRONG IN THE DANGEROUS
    DIRECTION. It read: "leave memory ADVANCED — the entry is still on
    disk." That assumes the on-disk line is COMPLETE, which it is not when
    the failure was an ENOSPC mid-``write``. MEASURED 2026-09-07 with that
    fix in place: seqs ``[0, 1, MALFORMED, 3]``, ``verify(): valid=False`` —
    the exact false-tampering verdict the handler exists to prevent.
    Invalidating the cache instead asserts nothing about disk and is
    therefore correct in every branch.
    """

    @staticmethod
    def _sick_disk(monkeypatch, active, mode):
        """Ordinary exceptions only — NO terminal signal anywhere.

        ``mode='complete'``: fsync reports EIO after a full line landed.
        ``mode='partial'``:  write dies mid-line (ENOSPC).
        Both then fail the rollback's ``open`` with EROFS, same sick disk.

        ⛔ EACH INJECTION FIRES **ONCE** AND RECORDS THAT IT FIRED, AND THE
        FIRST VERSION OF THIS HELPER DID NEITHER — written by the same seat,
        in the same file, hours after it FOUND AND FIXED exactly this defect
        in ``test_the_rollback_truncates_before_it_restores`` and wrote a
        rule about it. Knowing the class does not immunise you against
        producing it.
        ⚡ AND CODEX (L3, 2026-09-07) NAMED THE TRIGGER THAT MAKES IT LIVE,
        WHICH IS ALREADY ON THIS REPO'S OWN OPEN LIST: *if ``log()`` gains an
        earlier directory fsync inside the guarded region, that call consumes
        the fault before any line is written* — the arm then passes without
        ever grading the complete-line ambiguity. `next_steps.md` item 2 is
        precisely "add the ``_fsync_dir`` idiom to this module". **The latent
        defect was scheduled to be activated by this repo's next filed task.**
        ▶ So the counters are not bookkeeping: ``fired`` is asserted by the
        caller, and an injection that stops reaching its intended call site
        fails LOUDLY instead of quietly passing.
        """
        import builtins

        real_open, real_fsync = builtins.open, os.fsync
        armed = {"on": True, "write": 0, "fsync": 0, "rollback_open": 0}

        class _Sick:
            def __init__(self, f):
                self._f = f

            def write(self, s):
                if armed["on"] and mode == "partial" and not armed["write"]:
                    armed["write"] += 1
                    self._f.write(s[:60])
                    self._f.flush()
                    raise OSError(28, "No space left on device")
                return self._f.write(s)

            def __getattr__(self, n):
                return getattr(self._f, n)

        class _SickCtx:
            def __init__(self, f):
                self._f = f

            def __enter__(self):
                return _Sick(self._f.__enter__())

            def __exit__(self, *a):
                return self._f.__exit__(*a)

        def sick_open(path, mode_="r", *a, **kw):
            if armed["on"] and str(path) == str(active):
                if mode_ == "a":
                    return _SickCtx(real_open(path, mode_, *a, **kw))
                if mode_ == "r+b":
                    armed["rollback_open"] += 1
                    raise OSError(30, "Read-only file system")

            return real_open(path, mode_, *a, **kw)

        def sick_fsync(fd):
            # FIRE ONCE — see the class note in this helper's docstring.
            if armed["on"] and mode == "complete" and not armed["fsync"]:
                armed["fsync"] += 1
                raise OSError(5, "Input/output error")
            return real_fsync(fd)

        monkeypatch.setattr(builtins, "open", sick_open)
        monkeypatch.setattr(os, "fsync", sick_fsync)
        return armed

    @pytest.mark.parametrize("mode", ["complete", "partial"])
    def test_the_retry_after_a_failed_rollback_does_not_reuse_the_seq(
        self, tmp_path, monkeypatch, caplog, mode
    ):
        """⛔ MUTATION-CHECKED 2026-09-07, each mutant re-read off disk:

          restore unconditionally (delete the ``if truncated``) .......
            ``complete`` fails on ``[0, 1, 2, 2]`` / valid=False
          leave memory ADVANCED instead of invalidating (the fix that was
          recorded at the site) ......................................
            ``partial`` fails on ``[0, 1, MALFORMED, 3]`` / valid=False

        ⚠ THE TWO ARMS FAIL UNDER DIFFERENT MUTANTS AND THAT IS THE POINT.
        One mutant alone leaves the other arm green, so a single-arm test
        would have graded whichever half its author happened to write.
        """
        db = tmp_path / "sick.db"
        trail = AuditTrail(db)
        trail.log("first")
        trail.log("second")
        active = db.parent / "sick.audit.jsonl"

        armed = self._sick_disk(monkeypatch, active, mode)
        with caplog.at_level(logging.WARNING, logger="anneal-memory"):
            with pytest.raises(OSError):
                trail.log("third")
        armed["on"] = False

        # ⛔ THE INJECTIONS MUST HAVE REACHED THE SITES THEY NAME. Without
        # this the arm can pass having graded nothing — the failure mode
        # codex found in this file's other gates, asserted rather than
        # assumed.
        assert armed["rollback_open"] == 1, (
            "the rollback's open() was never reached, so the failed-truncate "
            "branch this test exists to grade never ran"
        )
        if mode == "complete":
            assert armed["fsync"] == 1, "the entry's fsync fault never fired"
        else:
            assert armed["write"] == 1, "the mid-line write fault never fired"

        assert any("audit rollback failed" in r.message for r in caplog.records), (
            "the rollback failed and left no breadcrumb — this is the branch "
            "an operator reaches with a red verdict and nothing to read"
        )

        trail.note_write_failure()           # caller contract: swallow, retry
        trail.log("retry")

        events, seqs = _audit_lines(active)
        real = [s for s in seqs if isinstance(s, int)]
        assert len(real) == len(set(real)), (
            f"the retry reused a seq already on disk: {seqs}. Memory was "
            f"rewound past what the failed truncate left behind."
        )
        assert "retry" in events, f"the retry entry never landed: {events}"

        r = AuditTrail.verify(db)
        assert r.valid, (
            f"a durability failure was read as TAMPERING — seqs {seqs}, "
            f"break at {r.chain_break_at}: {r.error}"
        )


class TestRecoveryHasAnAnchorOrRefusesToInitialise:
    """codex L3, 2026-09-07 — three ways ``_initialize`` claimed to have
    recovered when it had not. All three end in the same place: a false
    tampering verdict from ``verify()``, which is the one verdict this
    record exists to make impossible.
    """

    def test_a_torn_only_active_file_still_anchors_on_the_manifest(
        self, tmp_path
    ):
        """The zero-byte fix was scoped by symptom; this is the class.

        A nonempty active file holding no line that PARSES gives no chain
        anchor, exactly as a zero-byte one gives none — but the branch for
        it fell through on the constructor defaults and started a fresh
        chain from genesis after sealed files that ended elsewhere.

        ⛔ MUTATION-CHECKED 2026-09-07, mutant re-read off disk: drop the
        ``_seed_from_manifest()`` call from the no-valid-entry branch and
        this fails with ``Hash mismatch at seq 0: expected sha256:...,
        got sha256:GENESIS...``.

        ⛔ AND THIS CHANGE RETIRED THE SIBLING GATE'S MUTATION CLAIM, WHICH
        IS A COST AND IS RECORDED AS ONE. An earlier draft of this docstring
        cited "the zero-byte sibling stays green under that mutant" as a
        PAIRED POSITIVE proving correct scoping. **That was worthless as a
        control**: codex (L3, 2026-09-07) showed the sibling now stays green
        under its OWN mutant too, so it discriminates nothing. MEASURED —
        delete ``or active.stat().st_size == 0`` from ``_initialize`` and the
        WHOLE audit suite still passes, 165 green.
        ⚡ Because ``_seed_from_manifest()`` here catches the empty file as
        well, so the zero-byte clause stopped being load-bearing FOR
        CORRECTNESS the moment this branch was added. **A gate written and
        mutation-verified that morning became decoration by lunchtime,
        without being edited, while staying green** — the same shape as an
        answer invalidated by something that never touched it.
        ▶ The clause is KEPT because it is still load-bearing for a
        different property — it avoids a read that can now raise — and that
        property gets its own gate below rather than an obsolete claim.
        """
        db = tmp_path / "rot.db"
        trail = AuditTrail(db)
        for i in range(3):
            trail.log("before", {"i": i})
        trail._last_week = "1999-W01"          # force the weekly rotation
        trail.log("after_rotation", {})

        active = db.parent / "rot.audit.jsonl"
        # what a failed rollback on a freshly rotated file leaves behind
        active.write_text('{"v":1,"seq":4,"ts":"2026-09-07T00:00:00.0000')
        assert active.stat().st_size > 0, "fixture: must be NONEMPTY"

        fresh = AuditTrail(db)
        fresh._initialize()
        assert fresh._prev_hash != GENESIS_HASH, (
            "recovery restarted the chain from genesis over a torn-only "
            "active file, ignoring the sealed files the manifest names"
        )

        fresh.log("next", {})
        r = AuditTrail.verify(db)
        assert r.valid, f"false tampering verdict: {r.error}"

    def test_the_zero_byte_fast_path_avoids_a_read_that_can_now_raise(
        self, tmp_path, monkeypatch
    ):
        """What the ``st_size == 0`` clause still protects, now that
        ``_seed_from_manifest()`` covers its correctness case.

        Read errors propagate from the scan as of 2026-09-07. So on a sick
        disk an empty active file MUST NOT be routed through
        ``_read_last_valid_entry`` — there is nothing in it to read, and
        attempting the read turns a recoverable state into a raise.

        ⛔ MUTATION-CHECKED 2026-09-07, mutant re-read off disk: delete
        ``or active.stat().st_size == 0`` from ``_initialize`` and this
        fails with the injected OSError escaping. **This is the claim the
        sibling gate can no longer make** — it is stated here because the
        clause is now a durability guard rather than a correctness one.
        """
        import builtins

        db = tmp_path / "fast.db"
        trail = AuditTrail(db)
        for i in range(2):
            trail.log("before", {"i": i})
        trail._last_week = "1999-W01"
        trail.log("after_rotation", {})

        active = db.parent / "fast.audit.jsonl"
        active.write_text("")                      # the rolled-back state
        assert active.stat().st_size == 0

        real_open = builtins.open

        def sick_open(path, mode="r", *a, **kw):
            if str(path) == str(active) and "r" in mode and "+" not in mode:
                raise OSError(5, "Input/output error")
            return real_open(path, mode, *a, **kw)

        monkeypatch.setattr(builtins, "open", sick_open)
        fresh = AuditTrail(db)
        fresh._initialize()                        # must NOT raise
        monkeypatch.undo()

        assert fresh._initialized is True
        assert fresh._prev_hash != GENESIS_HASH, (
            "the fast path ran but did not anchor on the manifest"
        )

    @pytest.mark.parametrize(
        "exc",
        [OSError(5, "Input/output error")],
        ids=["OSError"],
    )
    def test_a_read_failure_during_recovery_is_not_an_empty_file(
        self, tmp_path, monkeypatch, exc
    ):
        """A failed scan must not be mistaken for a completed empty one.

        ``_read_last_valid_entry`` suppressed ``OSError`` and returned
        whatever it had found so far; ``_initialize`` then set
        ``_initialized = True`` on that partial answer. The failure is
        CORRELATED with the caller that most needs recovery — a rollback
        that already failed on this disk.

        ⛔ MUTATION-CHECKED 2026-09-07: restore the
        ``except (OSError, UnicodeDecodeError): pass`` around the scan and
        this fails — ``_initialize`` returns having "recovered" from a read
        that died, with ``_initialized`` True and ``_seq`` short.

        ⚠ **THE UnicodeDecodeError ARM RETIRED 2026-09-08 (diogenes), AND
        THIS IS THE COST STATED IN THE FINDING, NOT A QUIET DROP.** The
        scan now opens ``rb`` and decodes per line, so ``for raw in f``
        itself can never raise ``UnicodeDecodeError`` — only the explicit
        ``.decode()`` inside the loop can, and that is now CAUGHT AND
        SKIPPED, not propagated (a torn line is not a failed read). The
        injected-iterator shape this test used for that arm modelled a
        text-mode ``for line in f`` raising mid-iteration, which is no
        longer the code path; re-pointing it at bytes would test a
        scenario the source can't produce. The real behaviour — a torn
        multibyte tail on disk is skipped, not raised — is pinned by
        ``test_a_torn_multibyte_tail_is_skipped_not_raised`` below, using a
        real torn byte instead of an injected exception. OSError still
        must propagate, so that arm stays and is the only one left.
        """
        db = tmp_path / "eio.db"
        trail = AuditTrail(db)
        for i in range(3):
            trail.log("entry", {"i": i})
        active = db.parent / "eio.audit.jsonl"

        real_open = open
        state = {"armed": True}

        class _DyingReader:
            """Yields the first line, then the disk gives out."""

            def __init__(self, f):
                self._f = f
                self._n = 0

            def __iter__(self):
                return self

            def __next__(self):
                self._n += 1
                if self._n > 1:
                    raise exc
                return next(iter(self._f))

            def __getattr__(self, n):
                return getattr(self._f, n)

        class _Ctx:
            def __init__(self, f):
                self._f = f

            def __enter__(self):
                return _DyingReader(self._f.__enter__())

            def __exit__(self, *a):
                return self._f.__exit__(*a)

        def dying_open(path, mode="r", *a, **kw):
            if state["armed"] and str(path) == str(active) and mode == "rb":
                return _Ctx(real_open(path, mode, *a, **kw))
            return real_open(path, mode, *a, **kw)

        monkeypatch.setattr("builtins.open", dying_open)
        fresh = AuditTrail(db)
        with pytest.raises(type(exc)):
            fresh._initialize()
        assert state["armed"], "fixture never armed"
        assert fresh._initialized is False, (
            "the trail marked itself initialised after a scan that FAILED "
            "— the next append writes a chain derived from a partial read"
        )

    def test_a_torn_multibyte_tail_is_skipped_not_raised(self, tmp_path):
        """The class diogenes named HIGH (2026-09-08): propagation scoped
        by EXCEPTION TYPE instead of by what failed.

        A ``UnicodeDecodeError`` off a torn multibyte tail is not a failed
        read — it's the same "incomplete entry" shape ``json.JSONDecodeError``
        already gets skipped for four lines below in this same loop.
        Before the fix this raised out of ``_read_last_valid_entry``,
        ``_initialized`` stayed False, and ``_seed_from_manifest`` (added
        this same window for exactly this no-valid-entry case) never ran —
        every subsequent ``log()`` re-scanned and re-raised. Permanent,
        silent trail loss while the store kept writing.
        """
        db = tmp_path / "torn.db"
        trail = AuditTrail(db)
        for i in range(3):
            trail.log("before", {"i": i})
        trail._last_week = "1999-W01"          # force a weekly rotation
        trail.log("after_rotation", {})

        active = db.parent / "torn.audit.jsonl"
        # A JSON line cut one byte inside a multibyte character (⛔,
        # U+26D4) — the diogenes probe's arm B, the untested half of the
        # already-tested arm A (a torn ASCII tail).
        active.write_bytes(
            '{"v":1,"seq":4,"ts":"2026-09-07T00:00:00.0000⛔'.encode("utf-8")[:-1]
        )
        assert active.stat().st_size > 0, "fixture: must be NONEMPTY"

        fresh = AuditTrail(db)
        fresh._initialize()  # must NOT raise
        assert fresh._initialized is True
        assert fresh._prev_hash != GENESIS_HASH, (
            "a torn multibyte tail left the trail unanchored — recovery "
            "should fall back to the manifest, not stay uninitialised"
        )

        fresh.log("next", {})
        r = AuditTrail.verify(db)
        assert r.valid, f"false tampering verdict: {r.error}"

    def test_a_raising_log_handler_does_not_skip_the_invalidation(
        self, tmp_path, monkeypatch
    ):
        """Logging is an application callback; it can raise.

        The rollback's ``logger.warning`` ran BEFORE ``_initialized`` was
        cleared, so a handler that raised took the safe state with it and
        the next append reused the seq.

        ⛔ MUTATION-CHECKED 2026-09-07: move ``self._initialized = False``
        back below the ``logger.warning`` call and this fails on duplicate
        seqs with ``verify(): valid=False``.
        """
        import builtins

        db = tmp_path / "loud.db"
        trail = AuditTrail(db)
        trail.log("first")
        trail.log("second")
        active = db.parent / "loud.audit.jsonl"

        real_open, real_fsync = builtins.open, os.fsync
        armed = {"on": True, "fsync": 0, "rollback_open": 0}
        exploded = {"n": 0}

        def sick_open(path, mode="r", *a, **kw):
            if armed["on"] and str(path) == str(active) and mode == "r+b":
                armed["rollback_open"] += 1
                raise OSError(30, "Read-only file system")
            return real_open(path, mode, *a, **kw)

        def sick_fsync(fd):
            # FIRE ONCE, on the entry's own fsync.
            if armed["on"] and not armed["fsync"]:
                armed["fsync"] += 1
                raise OSError(5, "Input/output error")
            return real_fsync(fd)

        def exploding_warning(msg, *a, **kw):
            # ⛔ SCOPED TO THE ROLLBACK WARNING. Exploding on EVERY
            # ``logger.warning`` grades more than this test claims and would
            # pass if some unrelated warning happened to fire first —
            # the same unscoped-injection class codex found across this
            # file's gates on 2026-09-07.
            if isinstance(msg, str) and "audit rollback failed" in msg:
                exploded["n"] += 1
                raise RuntimeError("a logging handler blew up")
            return None

        monkeypatch.setattr(builtins, "open", sick_open)
        monkeypatch.setattr(os, "fsync", sick_fsync)
        monkeypatch.setattr(
            audit_module.logger, "warning", exploding_warning
        )

        with pytest.raises(BaseException):
            trail.log("third")
        armed["on"] = False
        monkeypatch.undo()

        assert exploded["n"] == 1, (
            "the rollback warning never fired, so the raising-handler "
            "scenario this test names was never actually exercised"
        )
        assert armed["rollback_open"] == 1 and armed["fsync"] == 1

        trail.note_write_failure()
        trail.log("retry")

        _, seqs = _audit_lines(active)
        real = [s for s in seqs if isinstance(s, int)]
        assert len(real) == len(set(real)), (
            f"a raising log handler skipped the invalidation and the retry "
            f"reused a seq: {seqs}"
        )
        assert AuditTrail.verify(db).valid

    def test_note_write_failure_reports_no_location_when_invalidated(
        self, tmp_path, monkeypatch
    ):
        """Flagged by BOTH L3 seats — a knowingly-stale seq made durable.

        After a failed rollback ``_seq`` is deliberately stale until the
        next ``_initialize()``. ``note_write_failure`` handed it back, and
        ``Store`` folds it into the durable ``audit_last_failure`` record,
        so an operator is pointed at an entry that exists.

        ⛔ MUTATION-CHECKED 2026-09-07: drop the ``if self._initialized``
        guard and this fails — a seq is returned where the location is not
        known. The docstring already promises ``None`` means exactly that.
        """
        import builtins

        db = tmp_path / "loc.db"
        trail = AuditTrail(db)
        trail.log("first")
        trail.log("second")

        assert trail.note_write_failure() is not None, (
            "paired positive: a HEALTHY trail must still report the "
            "location, or this test passes on a method that always "
            "returns None"
        )
        trail._dropped_since_last = 0

        # ⛔ REACH THE STATE THROUGH A REAL FAILED ROLLBACK, NOT BY FLIPPING
        # THE FLAG. An earlier draft set ``_initialized = False`` by hand
        # after two healthy writes — where ``_seq`` is 2, disk ends at seq 1,
        # and 2 is therefore the CORRECT location. It asserted the guard
        # fires without ever creating the staleness the guard exists for,
        # so it could not distinguish "returns None when invalidated" from
        # "returns None usefully". Named by codex (L3, 2026-09-07).
        active = db.parent / "loc.audit.jsonl"
        real_open, real_fsync = builtins.open, os.fsync
        armed = {"on": True, "fsync": 0, "rollback_open": 0}

        def sick_open(path, mode="r", *a, **kw):
            if armed["on"] and str(path) == str(active) and mode == "r+b":
                armed["rollback_open"] += 1
                raise OSError(30, "Read-only file system")
            return real_open(path, mode, *a, **kw)

        def sick_fsync(fd):
            if armed["on"] and not armed["fsync"]:
                armed["fsync"] += 1
                raise OSError(5, "Input/output error")
            return real_fsync(fd)

        monkeypatch.setattr(builtins, "open", sick_open)
        monkeypatch.setattr(os, "fsync", sick_fsync)
        with pytest.raises(OSError):
            trail.log("third")
        monkeypatch.undo()
        armed["on"] = False

        assert armed["fsync"] == 1 and armed["rollback_open"] == 1
        assert trail._initialized is False, (
            "fixture precondition: the failed rollback must have invalidated"
        )
        # the entry IS on disk at seq 2 — the stale value would name it
        _, seqs = _audit_lines(active)
        assert 2 in seqs, f"fixture: seq 2 should be on disk, got {seqs}"

        assert trail.note_write_failure() is None, (
            f"a seq known to be stale was handed back to be persisted as "
            f"the durable location of the gap — and seq 2 is ON DISK "
            f"({seqs}), so it names an entry that exists"
        )
