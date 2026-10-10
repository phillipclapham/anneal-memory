"""Tests for the prospective-intention layer (spores.py).

Covers germination computation, the full plant→grow→resolve lifecycle, the
library-ization invariants (raises not sys.exit; returns dicts; injectable
clock for determinism), the type-specific terminal-kind enforcement, and the
load-bearing store invariant: a corrupt store NEVER silently re-inits.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from datetime import date, datetime, timedelta, timezone

import pytest

from anneal_memory import SporeError, SporeStore, germination_tier, spore_version
from anneal_memory import spores as _spores
from anneal_memory.spores import SPORE_SCHEMA_VERSION


# Module-level so the "spawn" start method can pickle it by qualified name.
def _concurrent_add_worker(path_str: str, barrier, idx: int, count: int) -> None:
    """Plant ``count`` spores against a shared store, aligned to a barrier so all
    writers hit the load→mutate→save window together (maximizes contention)."""
    store = SporeStore(path_str)
    try:
        barrier.wait(timeout=30)
    except Exception:  # pragma: no cover - a sibling died before the barrier
        return
    for j in range(count):
        store.add(type="task", text=f"w{idx}-{j}", domain="concurrency", today=date(2026, 1, 1))


# Module-level for "spawn": holds the store's lock while it rewrites the text, so a
# caller that read before it must meet the change under the same lock.
def _hold_lock_and_edit(path_str: str, held, release) -> None:
    store = SporeStore(path_str)
    with store._transaction() as data:
        data["spores"][0]["text"] = "edited by the other writer"
        held.set()
        release.wait(timeout=30)


# A guarded write (expected_version / expect_disposition) is refused without a file
# lock; those cases are tested in TestExpectedVersion with fcntl patched out.
_NEEDS_LOCK = pytest.mark.skipif(_spores.fcntl is None, reason="a guarded write needs a file lock")


@pytest.fixture
def store(tmp_path):
    return SporeStore(tmp_path / "spores.json")


# Reference clock used across tests so date math is explicit.
T0 = date(2026, 1, 1)


# -- germination -----------------------------------------------------------


class TestGerminationTier:
    def _spore(self, **kw):
        base = {"tier": "warm", "seen": T0.isoformat(), "next": None}
        base.update(kw)
        return base

    def test_parked_overrides_everything(self):
        s = self._spore(tier="parked", seen=T0.isoformat(), next="2020-01-01")
        assert germination_tier(s, T0) == "parked"

    def test_growing_under_3_days(self):
        s = self._spore(seen=date(2026, 1, 1).isoformat())
        assert germination_tier(s, date(2026, 1, 3)) == "growing"  # 2 days

    def test_resting_lower_boundary_3_days(self):
        s = self._spore(seen=date(2026, 1, 1).isoformat())
        assert germination_tier(s, date(2026, 1, 4)) == "resting"  # 3 days

    def test_resting_upper_boundary_7_days(self):
        s = self._spore(seen=date(2026, 1, 1).isoformat())
        assert germination_tier(s, date(2026, 1, 8)) == "resting"  # 7 days

    def test_dormant_over_7_days(self):
        s = self._spore(seen=date(2026, 1, 1).isoformat())
        assert germination_tier(s, date(2026, 1, 9)) == "dormant"  # 8 days

    def test_next_in_past_forces_dormant_despite_fresh_seen(self):
        s = self._spore(seen=date(2026, 1, 1).isoformat(), next="2026-01-02")
        # seen is "today" (growing) but next: has elapsed → dormant
        assert germination_tier(s, date(2026, 1, 5)) == "dormant"

    def test_next_equal_today_is_dormant(self):
        # the day a next: alarm arrives, it surfaces (day-of was once inert)
        s = self._spore(seen=date(2026, 1, 5).isoformat(), next="2026-01-05")
        assert germination_tier(s, date(2026, 1, 5)) == "dormant"

    def test_no_seen_is_dormant(self):
        s = self._spore(seen=None)
        assert germination_tier(s, T0) == "dormant"

    def test_hand_edited_seen_with_trailing_note_still_parses(self):
        s = self._spore(seen="2026-01-01 (migrated)")
        assert germination_tier(s, date(2026, 1, 2)) == "growing"


# -- add -------------------------------------------------------------------


class TestAdd:
    def test_defaults(self, store):
        s = store.add(type="task", text="ship it", today=T0)
        assert s["id"] == "spore-001"
        assert s["type"] == "task"
        assert s["tier"] == "warm"
        assert s["salience"] == 0
        assert s["seen"] == T0.isoformat()
        assert s["created"] == T0.isoformat()
        assert s["next"] is None
        assert s["status"] == "open"
        assert s["resolution"] is None
        assert s["notes"] == []

    def test_all_fields(self, store):
        s = store.add(
            type="thought",
            text="what if",
            domain="architecture",
            tier="hot",
            salience=3,
            next="2026-06-03",
            pointer="projects/x",
            today=T0,
        )
        assert s["domain"] == "architecture"
        assert s["tier"] == "hot"
        assert s["salience"] == 3
        assert s["next"] == "2026-06-03"
        assert s["pointer"] == "projects/x"

    def test_ids_increment(self, store):
        a = store.add(type="task", text="a", today=T0)
        b = store.add(type="task", text="b", today=T0)
        assert (a["id"], b["id"]) == ("spore-001", "spore-002")

    def test_ids_never_reused_after_resolve(self, store):
        store.add(type="task", text="a", today=T0)
        store.add(type="task", text="b", today=T0)
        store.descend("spore-001", kind="done", today=T0)
        # next id counts resolved too → 003, not a reused 001
        c = store.add(type="task", text="c", today=T0)
        assert c["id"] == "spore-003"

    def test_persists_to_disk(self, store):
        store.add(type="question", text="why", today=T0)
        reloaded = SporeStore(store.path)
        got = reloaded.get("spore-001")
        assert got is not None
        assert got["text"] == "why"

    @pytest.mark.parametrize(
        "kw",
        [
            {"type": "nope", "text": "x"},
            {"type": "task", "text": "x", "tier": "nope"},
            {"type": "task", "text": "x", "salience": 4},
            {"type": "task", "text": "x", "salience": -1},
            {"type": "task", "text": "   "},
            {"type": "task", "text": ""},
            {"type": "task", "text": "x", "next": "2026/06/03"},
            {"type": "task", "text": "x", "next": "06-03-2026"},
            {"type": "task", "text": 123},                       # non-str text
            {"type": "task", "text": "x", "salience": True},     # bool is not a valid salience
            {"type": "task", "text": "x", "domain": {"k": "v"}}, # non-str domain
            {"type": "task", "text": "x", "pointer": 5},         # non-str pointer
        ],
    )
    def test_invalid_args_raise_valueerror(self, store, kw):
        with pytest.raises(ValueError):
            store.add(today=T0, **kw)


# -- get / list / surface --------------------------------------------------


class TestRead:
    def test_get_open_and_resolved_and_missing(self, store):
        store.add(type="task", text="a", today=T0)
        store.add(type="task", text="b", today=T0)
        store.descend("spore-002", kind="done", today=T0)
        assert store.get("spore-001")["text"] == "a"
        assert store.get("spore-002")["status"] == "resolved"
        assert store.get("spore-999") is None

    def test_list_open_excludes_resolved(self, store):
        store.add(type="task", text="a", today=T0)
        store.add(type="task", text="b", today=T0)
        store.descend("spore-001", kind="done", today=T0)
        ids = [s["id"] for s in store.list_open(today=T0)]
        assert ids == ["spore-002"]

    def test_list_filters(self, store):
        store.add(type="task", text="a", domain="health", tier="hot", today=T0)
        store.add(type="question", text="b", domain="career", tier="warm", today=T0)
        assert len(store.list_open(type="task", today=T0)) == 1
        assert len(store.list_open(tier="warm", today=T0)) == 1
        assert len(store.list_open(domain="career", today=T0)) == 1
        assert store.list_open(type="task", today=T0)[0]["text"] == "a"

    def test_list_germination_filter(self, store):
        store.add(type="task", text="fresh", today=date(2026, 1, 8))
        store.add(type="task", text="old", today=date(2026, 1, 1))
        # query 2026-01-09: fresh=1d growing, old=8d dormant
        growing = store.list_open(germination="growing", today=date(2026, 1, 9))
        assert [s["text"] for s in growing] == ["fresh"]

    def test_ranking_tier_then_salience_then_germination(self, store):
        store.add(type="task", text="warm-hi", tier="warm", salience=3, today=T0)
        store.add(type="task", text="hot-lo", tier="hot", salience=0, today=T0)
        store.add(type="task", text="hot-hi", tier="hot", salience=2, today=T0)
        ordered = [s["text"] for s in store.list_open(today=T0)]
        # hot before warm; within hot, higher salience first
        assert ordered == ["hot-hi", "hot-lo", "warm-hi"]

    def test_surface_top_of_mind_is_hot_or_growing(self, store):
        # cold + dormant → excluded; hot → included; growing-but-cold → included
        store.add(type="task", text="hot", tier="hot", today=T0)
        store.add(type="task", text="cold-old", tier="cold", today=date(2025, 1, 1))
        store.add(type="task", text="cold-fresh", tier="cold", today=T0)
        tom = [s["text"] for s in store.surface(top_of_mind=True, today=T0)]
        assert "hot" in tom
        assert "cold-fresh" in tom  # growing
        assert "cold-old" not in tom  # cold + dormant

    def test_surface_does_not_annotate_germination_onto_stored_shape(self, store):
        store.add(type="task", text="a", today=T0)
        s = store.surface(today=T0)[0]
        assert "germination" not in s  # computed, never stored


# -- touch -----------------------------------------------------------------


class TestTouch:
    def test_bumps_seen(self, store):
        store.add(type="task", text="a", today=date(2026, 1, 1))
        s = store.touch("spore-001", today=date(2026, 1, 5))
        assert s["seen"] == "2026-01-05"

    def test_clears_elapsed_next_returning_to_growing(self, store):
        store.add(type="task", text="a", next="2026-01-03", today=date(2026, 1, 1))
        store.touch("spore-001", today=date(2026, 1, 5))
        s = store.get("spore-001")
        assert s["next"] is None
        assert germination_tier(s, date(2026, 1, 5)) == "growing"

    def test_keeps_future_next(self, store):
        store.add(type="task", text="a", next="2026-02-01", today=date(2026, 1, 1))
        store.touch("spore-001", today=date(2026, 1, 5))
        assert store.get("spore-001")["next"] == "2026-02-01"

    def test_missing_raises(self, store):
        with pytest.raises(SporeError, match="not found"):
            store.touch("spore-999", today=T0)

    def test_resolved_raises_distinctly(self, store):
        store.add(type="task", text="a", today=T0)
        store.descend("spore-001", kind="done", today=T0)
        with pytest.raises(SporeError, match="already resolved"):
            store.touch("spore-001", today=T0)


# -- update ----------------------------------------------------------------


class TestUpdate:
    def test_sets_fields(self, store):
        store.add(type="task", text="a", today=T0)
        s = store.update(
            "spore-001", tier="hot", salience=2, text="b", domain="career", today=T0
        )
        assert (s["tier"], s["salience"], s["text"], s["domain"]) == (
            "hot",
            2,
            "b",
            "career",
        )

    def test_omitted_fields_unchanged(self, store):
        store.add(type="task", text="a", tier="warm", salience=1, today=T0)
        s = store.update("spore-001", tier="hot", today=T0)
        assert s["salience"] == 1  # untouched
        assert s["text"] == "a"

    def test_clear_next_and_pointer_and_domain(self, store):
        store.add(
            type="task", text="a", next="2026-06-03", pointer="p", domain="d", today=T0
        )
        s = store.update("spore-001", next="", pointer="", domain="", today=T0)
        assert s["next"] is None
        assert s["pointer"] is None
        assert s["domain"] == ""

    def test_add_note_appends_with_date_stamp(self, store):
        store.add(type="task", text="a", today=T0)
        s = store.update("spore-001", add_note="hit a wall", today=date(2026, 1, 9))
        assert s["notes"] == ["[2026-01-09] hit a wall"]

    def test_does_not_bump_seen(self, store):
        store.add(type="task", text="a", today=date(2026, 1, 1))
        s = store.update("spore-001", tier="hot", today=date(2026, 1, 9))
        assert s["seen"] == "2026-01-01"  # engagement is explicit via touch

    @pytest.mark.parametrize(
        "kw",
        [
            {"tier": "nope"},
            {"salience": 9},
            {"salience": True},          # bool is not a valid salience
            {"next": "2026/01/01"},
            {"text": "  "},
            {"text": 123},               # non-str text
            {"domain": {"k": "v"}},      # non-str domain
            {"pointer": 5},              # non-str pointer
        ],
    )
    def test_invalid_update_raises(self, store, kw):
        store.add(type="task", text="a", today=T0)
        with pytest.raises(ValueError):
            store.update("spore-001", today=T0, **kw)

    def test_add_note_repairs_corrupt_notes(self, store):
        store.add(type="task", text="a", today=T0)
        data = json.loads(store.path.read_text())
        data["spores"][0]["notes"] = None  # corrupt: notes not a list
        store.path.write_text(json.dumps(data))
        s = store.update("spore-001", add_note="hi", today=date(2026, 1, 9))
        assert s["notes"] == ["[2026-01-09] hi"]


# -- descend / ascend ------------------------------------------------------


class TestResolve:
    def test_descend_moves_to_resolved_with_record(self, store):
        store.add(type="task", text="a", today=T0)
        s = store.descend("spore-001", kind="done", today=T0)
        assert s["status"] == "resolved"
        res = s["resolution"]
        assert res["direction"] == "descend"
        assert res["kind"] == "done"
        assert res["ref"] is None
        assert res["on"] == T0.isoformat()
        assert res["at"].endswith("+00:00")  # UTC instant
        assert store.list_open(today=T0) == []

    def test_ascend_records_ref_and_record(self, store):
        store.add(type="question", text="a", today=T0)
        s = store.ascend("spore-001", kind="pattern", ref="my_pattern", today=T0)
        assert s["status"] == "resolved"
        assert s["resolution"]["direction"] == "ascend"
        assert s["resolution"]["kind"] == "pattern"
        assert s["resolution"]["ref"] == "my_pattern"

    @pytest.mark.parametrize(
        "type_,kind",
        [
            ("task", "answered"),   # answered is question-only
            ("question", "done"),   # done is task-only
            ("thought", "mooted"),  # mooted is question-only
        ],
    )
    def test_descend_kind_must_fit_type(self, store, type_, kind):
        store.add(type=type_, text="a", today=T0)
        with pytest.raises(ValueError, match="invalid for"):
            store.descend("spore-001", kind=kind, today=T0)

    def test_composted_is_valid_for_every_type(self, store):
        for t in ("task", "question", "thought"):
            st = SporeStore(store.path.with_name(f"{t}.json"))
            st.add(type=t, text="a", today=T0)
            st.descend("spore-001", kind="composted", today=T0)
            assert st.get("spore-001")["status"] == "resolved"

    @pytest.mark.parametrize(
        "type_,kind",
        [("task", "essay"), ("question", "project"), ("thought", "thread")],
    )
    def test_ascend_kind_must_fit_type(self, store, type_, kind):
        store.add(type=type_, text="a", today=T0)
        with pytest.raises(ValueError, match="invalid for"):
            store.ascend("spore-001", kind=kind, ref="r", today=T0)

    def test_ascend_requires_ref(self, store):
        store.add(type="question", text="a", today=T0)
        with pytest.raises(ValueError, match="requires a ref"):
            store.ascend("spore-001", kind="pattern", ref="", today=T0)

    def test_resolve_missing_raises(self, store):
        with pytest.raises(SporeError, match="not found"):
            store.descend("spore-999", kind="done", today=T0)

    def test_ambiguous_duplicate_id_refuses(self, store):
        # hand-craft store drift: two open spores share an id
        store.add(type="task", text="a", today=T0)
        data = json.loads(store.path.read_text())
        dup = dict(data["spores"][0])
        data["spores"].append(dup)
        store.path.write_text(json.dumps(data))
        with pytest.raises(SporeError, match="ambiguous"):
            store.descend("spore-001", kind="done", today=T0)

    def test_resolved_stranded_in_open_refuses(self, store):
        # drift: a resolved-status spore left in the open list must not re-resolve
        store.add(type="task", text="a", today=T0)
        data = json.loads(store.path.read_text())
        data["spores"][0]["status"] = "resolved"
        store.path.write_text(json.dumps(data))
        with pytest.raises(SporeError, match="store drift"):
            store.touch("spore-001", today=T0)

    def test_resolve_at_is_injectable(self, store):
        store.add(type="task", text="a", today=T0)
        fixed = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        s = store.descend("spore-001", kind="done", today=T0, now=fixed)
        assert s["resolution"]["at"] == "2026-01-01T12:00:00+00:00"

    def test_naive_now_rejected(self, store):
        store.add(type="task", text="a", today=T0)
        with pytest.raises(ValueError, match="timezone-aware"):
            store.descend("spore-001", kind="done", today=T0, now=datetime(2026, 1, 1, 12))

    def test_non_utc_now_normalized_to_utc(self, store):
        store.add(type="task", text="a", today=T0)
        est = timezone(timedelta(hours=-5))
        s = store.descend(
            "spore-001",
            kind="done",
            today=T0,
            now=datetime(2026, 1, 1, 7, 0, 0, tzinfo=est),
        )
        assert s["resolution"]["at"] == "2026-01-01T12:00:00+00:00"

    def test_cross_list_duplicate_id_refuses(self, store):
        # drift: the same id present in BOTH the open and resolved lists
        store.add(type="task", text="a", today=T0)
        data = json.loads(store.path.read_text())
        data["resolved"].append(dict(data["spores"][0], status="resolved"))
        store.path.write_text(json.dumps(data))
        with pytest.raises(SporeError, match="already exists in the resolved"):
            store.descend("spore-001", kind="done", today=T0)


# -- store io invariants ---------------------------------------------------


class TestStoreIO:
    def test_missing_file_is_empty_store(self, store):
        assert store.list_open(today=T0) == []
        assert store.get("spore-001") is None

    def test_corrupt_json_refuses_never_silently_reinits(self, store):
        store.path.write_text("{ this is not json")
        with pytest.raises(SporeError, match="not valid JSON"):
            store.list_open(today=T0)
        # the corrupt file is untouched — recoverable data not overwritten
        assert store.path.read_text() == "{ this is not json"

    def test_non_object_json_refuses(self, store):
        store.path.write_text("[1, 2, 3]")
        with pytest.raises(SporeError, match="must contain a JSON object"):
            store.list_open(today=T0)

    def test_non_list_containers_refuse(self, store):
        # corrupt: spores is an object, not a list — setdefault won't fix it
        store.path.write_text(json.dumps({"spores": {}, "resolved": []}))
        with pytest.raises(SporeError, match="must be lists"):
            store.list_open(today=T0)

    def test_newer_schema_version_refuses(self, store):
        store.path.write_text(
            json.dumps({"spores": [], "resolved": [], "schema_version": 999})
        )
        with pytest.raises(SporeError, match="newer spore schema"):
            store.list_open(today=T0)

    def test_non_dict_rows_refuse(self, store):
        store.path.write_text(json.dumps({"spores": ["x"], "resolved": []}))
        with pytest.raises(SporeError, match="non-object rows"):
            store.list_open(today=T0)

    def test_non_integer_schema_version_refuses(self, store):
        store.path.write_text(
            json.dumps({"spores": [], "resolved": [], "schema_version": "999"})
        )
        with pytest.raises(SporeError, match="non-integer schema_version"):
            store.list_open(today=T0)

    def test_forward_compat_defaults_missing_keys(self, store):
        store.path.write_text(json.dumps({"spores": []}))  # no resolved/schema_version
        # should not crash; resolved + schema_version defaulted on load
        store.add(type="task", text="a", today=T0)
        data = json.loads(store.path.read_text())
        assert data["schema_version"] == SPORE_SCHEMA_VERSION
        assert data["resolved"] == []

    def test_write_leaves_no_tmp_sidecar(self, store):
        store.add(type="task", text="a", today=T0)
        assert not store.path.with_name(store.path.name + ".tmp").exists()


# -- public surface --------------------------------------------------------


class TestExports:
    def test_top_level_imports(self):
        import anneal_memory as am

        assert hasattr(am, "SporeStore")
        assert hasattr(am, "SporeError")
        assert hasattr(am, "germination_tier")
        assert issubclass(am.SporeError, am.AnnealMemoryError)


# -- concurrency (the SP-CONCURRENCY fix: many writers serialize safely) ---


class TestConcurrency:
    """The prospective layer is inherently multi-writer (parallel sessions +
    overnight wrappers). diogenes repro'd the pre-fix bug: 8 parallel writers →
    5 crashed at ``os.replace`` and the store ended with 1 spore instead of 8.
    These exercise the real cross-process path with ``multiprocessing`` (threads
    would not test the cross-process advisory lock).
    """

    def _run_writers(self, path_str: str, n: int, count: int) -> None:
        ctx = mp.get_context("spawn")
        barrier = ctx.Barrier(n)
        procs = [
            ctx.Process(target=_concurrent_add_worker, args=(path_str, barrier, i, count))
            for i in range(n)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
        for p in procs:
            assert p.exitcode == 0, f"a writer crashed (exitcode={p.exitcode})"

    @pytest.mark.skipif(
        _spores.fcntl is None,
        reason="cross-process serialization is a POSIX-fcntl guarantee; the lock "
        "no-ops without fcntl, so this assertion would false-green / flaky-fail",
    )
    def test_parallel_writers_no_lost_updates_or_id_collisions(self, tmp_path):
        path = tmp_path / "spores.json"
        n, count = 8, 5
        self._run_writers(str(path), n, count)

        data = json.loads(path.read_text())
        spores = data["spores"]
        # No lost updates: every write landed.
        assert len(spores) == n * count
        # No id collisions: reload-inside-lock makes _next_id see committed state.
        ids = [s["id"] for s in spores]
        assert len(set(ids)) == n * count
        # No torn writes / leftover tmp sidecars from the concurrent replaces.
        assert not list(tmp_path.glob("spores.json.*.tmp"))
        # The store reloads cleanly through the public API.
        assert len(SporeStore(path).list_open()) == n * count

    def test_lockfile_is_a_sibling_and_does_not_corrupt_reads(self, tmp_path):
        path = tmp_path / "spores.json"
        store = SporeStore(path)
        store.add(type="task", text="solo", today=date(2026, 1, 1))
        # The advisory lock file sits beside the store and is not the store.
        lock = path.with_name(path.name + ".lock")
        if lock.exists():  # POSIX only; a no-op (no lock file) on non-POSIX
            assert lock.read_text() == ""
        # The lock file is never confused for a spore document.
        assert len(store.list_open()) == 1


# -- disposition (the opaque operator-I/O routing tag) --------------------------
# anneal stores it verbatim and NEVER interprets it — the Tray taxonomy
# (loop/seed/handoff/agenda) lives in the Levain/flow disposition-aware layer.


class TestDisposition:
    def test_add_omits_the_key_for_a_normal_loop(self, store):
        """A plain loop stays key-free (store-minimal, backward-identical) — the
        additive-no-schema-bump invariant."""
        s = store.add(type="task", text="loop", today=T0)
        assert "disposition" not in s
        assert "disposition" not in SporeStore(store.path).get("spore-001")

    def test_add_stores_a_truthy_disposition_verbatim(self, store):
        s = store.add(type="thought", text="dumped", disposition="seed", today=T0)
        assert s["disposition"] == "seed"
        assert SporeStore(store.path).get("spore-001")["disposition"] == "seed"

    def test_add_stores_an_arbitrary_value_uninterpreted(self, store):
        """Anneal is blind to the taxonomy: a value it has never heard of is stored
        exactly, not rejected — the disposition-aware layer owns validation."""
        s = store.add(type="task", text="x", disposition="not-a-real-disposition", today=T0)
        assert s["disposition"] == "not-a-real-disposition"

    def test_add_empty_string_stays_key_free(self, store):
        s = store.add(type="task", text="x", disposition="", today=T0)
        assert "disposition" not in s

    def test_add_rejects_a_non_string_disposition(self, store):
        with pytest.raises(ValueError, match="disposition must be a string"):
            store.add(type="task", text="x", disposition=3, today=T0)  # type: ignore[arg-type]

    def test_update_sets_a_disposition(self, store):
        store.add(type="thought", text="x", today=T0)
        s = store.update("spore-001", disposition="handoff")
        assert s["disposition"] == "handoff"
        assert SporeStore(store.path).get("spore-001")["disposition"] == "handoff"

    def test_update_clears_to_a_plain_loop(self, store):
        """Metabolize: None/'' pops the key (back to a key-free loop), mirroring
        how next/pointer/domain clear."""
        store.add(type="thought", text="x", disposition="seed", today=T0)
        s = store.update("spore-001", disposition=None)
        assert "disposition" not in s
        assert "disposition" not in SporeStore(store.path).get("spore-001")

    def test_update_empty_string_also_clears(self, store):
        store.add(type="thought", text="x", disposition="seed", today=T0)
        s = store.update("spore-001", disposition="")
        assert "disposition" not in s

    def test_update_omitted_leaves_disposition_untouched(self, store):
        """The _UNSET sentinel: not passing disposition must NOT clear an existing
        one (the omit-vs-clear distinction)."""
        store.add(type="thought", text="x", disposition="agenda", today=T0)
        s = store.update("spore-001", tier="hot")
        assert s["disposition"] == "agenda"

    def test_update_rejects_a_non_string_disposition(self, store):
        store.add(type="task", text="x", today=T0)
        with pytest.raises(ValueError, match="disposition must be a string"):
            store.update("spore-001", disposition=[])  # type: ignore[arg-type]

    # -- expect_disposition: the optimistic compare-and-set (TOCTOU guard) --------
    @_NEEDS_LOCK
    def test_update_expect_disposition_matches_applies(self, store):
        store.add(type="thought", text="x", disposition="seed", today=T0)
        s = store.update("spore-001", disposition="handoff", expect_disposition="seed")
        assert s["disposition"] == "handoff"

    @_NEEDS_LOCK
    def test_update_expect_disposition_none_matches_a_keyfree_loop(self, store):
        # a plain loop's raw disposition is ABSENT → expect None must match it
        store.add(type="thought", text="x", today=T0)
        s = store.update("spore-001", disposition="seed", expect_disposition=None)
        assert s["disposition"] == "seed"

    @_NEEDS_LOCK
    def test_update_expect_disposition_mismatch_raises_and_writes_nothing(self, store):
        store.add(type="thought", text="x", disposition="seed", today=T0)
        # a concurrent writer would have changed it; we expected "agenda" but it's "seed"
        with pytest.raises(SporeError, match="disposition changed since read"):
            store.update("spore-001", disposition="loop-ish", expect_disposition="agenda")
        # nothing written — the disposition is untouched
        assert SporeStore(store.path).get("spore-001")["disposition"] == "seed"

    @_NEEDS_LOCK
    def test_update_expect_disposition_none_mismatches_a_tagged_spore(self, store):
        # caller expected a key-free loop but it's actually a seed → reject
        store.add(type="thought", text="x", disposition="seed", today=T0)
        with pytest.raises(SporeError, match="disposition changed since read"):
            store.update("spore-001", disposition="agenda", expect_disposition=None)

    def test_update_without_expect_disposition_skips_the_cas(self, store):
        # omitted (the _UNSET default) → no CAS, unconditional set (back-compat)
        store.add(type="thought", text="x", disposition="seed", today=T0)
        s = store.update("spore-001", disposition="handoff")
        assert s["disposition"] == "handoff"

    # -- ascend expect_disposition CAS: the read-then-resolve TOCTOU guard (codex L3 HIGH, --
    # -- 2026-06-23). A host enforcing "a Keep note can't ascend" reads the disposition in a -
    # -- separate step; the CAS makes a concurrent flip between that read and the resolve fail.
    @_NEEDS_LOCK
    def test_ascend_expect_disposition_matches_resolves(self, store):
        store.add(type="thought", text="x", today=T0)  # a key-free loop → expect None
        s = store.ascend("spore-001", kind="pattern", ref="r", expect_disposition=None, today=T0)
        assert s["status"] == "resolved" and s["resolution"]["direction"] == "ascend"

    @_NEEDS_LOCK
    def test_ascend_expect_disposition_mismatch_raises_and_resolves_nothing(self, store):
        # THE TOCTOU: a host read the spore as a loop (None) and guard-passed, but a concurrent
        # writer flipped it to a note. The CAS must refuse — nothing resolved, the note stays open.
        store.add(type="thought", text="x", disposition="note", today=T0)
        with pytest.raises(SporeError, match="disposition changed since read"):
            store.ascend("spore-001", kind="pattern", ref="r", expect_disposition=None, today=T0)
        fresh = SporeStore(store.path)
        assert "spore-001" in [s["id"] for s in fresh.list_open()]  # NOT resolved
        assert fresh.get("spore-001")["disposition"] == "note"      # untouched

    @_NEEDS_LOCK
    def test_ascend_expect_disposition_none_mismatches_a_tagged_spore(self, store):
        # caller expected a key-free loop but it's a seed → reject (symmetry with update's CAS)
        store.add(type="thought", text="x", disposition="seed", today=T0)
        with pytest.raises(SporeError, match="disposition changed since read"):
            store.ascend("spore-001", kind="pattern", ref="r", expect_disposition=None, today=T0)

    def test_ascend_without_expect_disposition_skips_the_cas(self, store):
        # omitted (the _UNSET default) → no CAS, resolves unconditionally (back-compat)
        store.add(type="thought", text="x", disposition="seed", today=T0)
        s = store.ascend("spore-001", kind="pattern", ref="r", today=T0)
        assert s["status"] == "resolved"

    # -- descend expect_disposition CAS: the snapshot-then-resolve TOCTOU guard (codex L3 HIGH --
    # -- on spore-170, spore-173). A control surface picks the resolve KIND ("remove" for a Keep --
    # -- note / "compost" for a loop) off a RENDERED snapshot; the CAS makes a concurrent re-route -
    # -- between that snapshot and this resolve fail closed (parity with ascend's CAS).
    @_NEEDS_LOCK
    def test_descend_expect_disposition_matches_resolves(self, store):
        store.add(type="thought", text="x", today=T0)  # a key-free loop → expect None
        s = store.descend("spore-001", kind="composted", expect_disposition=None, today=T0)
        assert s["status"] == "resolved" and s["resolution"]["direction"] == "descend"

    @_NEEDS_LOCK
    def test_descend_expect_disposition_tag_matches_resolves(self, store):
        store.add(type="thought", text="x", disposition="note", today=T0)  # a Keep note → "remove"
        s = store.descend("spore-001", kind="composted", expect_disposition="note", today=T0)
        assert s["status"] == "resolved"

    @_NEEDS_LOCK
    def test_descend_expect_disposition_mismatch_raises_and_resolves_nothing(self, store):
        # THE TOCTOU: a surface rendered this as a Keep note and chose "remove" (expect "note"),
        # but a concurrent writer promoted it to a live loop. The CAS must refuse — nothing
        # composted, the (now-live) loop stays open. Without it, a live loop is silently composted.
        store.add(type="thought", text="x", today=T0)  # actually a key-free loop now
        with pytest.raises(SporeError, match="disposition changed since read"):
            store.descend("spore-001", kind="composted", expect_disposition="note", today=T0)
        fresh = SporeStore(store.path)
        assert "spore-001" in [s["id"] for s in fresh.list_open()]  # NOT resolved
        assert fresh.get("spore-001").get("disposition") is None      # still a key-free loop

    @_NEEDS_LOCK
    def test_descend_expect_disposition_none_mismatches_a_tagged_spore(self, store):
        # caller expected a key-free loop but it's a note → reject (symmetry with ascend's CAS)
        store.add(type="thought", text="x", disposition="note", today=T0)
        with pytest.raises(SporeError, match="disposition changed since read"):
            store.descend("spore-001", kind="composted", expect_disposition=None, today=T0)

    @_NEEDS_LOCK
    def test_descend_cas_runs_before_kind_validation(self, store):
        # a stale snapshot must fail on the CAS (re-read the spore) before a kind error — the
        # operator's fix is "re-read", not "your kind is wrong for a type you no longer see".
        store.add(type="thought", text="x", disposition="seed", today=T0)
        with pytest.raises(SporeError, match="disposition changed since read"):
            store.descend("spore-001", kind="not-a-real-kind", expect_disposition=None, today=T0)

    def test_descend_without_expect_disposition_skips_the_cas(self, store):
        # omitted (the _UNSET default) → no CAS, resolves unconditionally (back-compat)
        store.add(type="thought", text="x", disposition="seed", today=T0)
        s = store.descend("spore-001", kind="composted", today=T0)
        assert s["status"] == "resolved"

    def test_disposition_round_trips_through_the_lifecycle_verbs(self, store):
        """The 3a de-risk, locked as a test: touch/descend leave an unknown
        disposition untouched (anneal never strips a field it doesn't model)."""
        store.add(type="task", text="x", disposition="seed", today=T0)
        store.touch("spore-001", today=T0)
        assert SporeStore(store.path).get("spore-001")["disposition"] == "seed"
        store.descend("spore-001", kind="done")
        # resolved rows keep the field too (the row moves whole into `resolved`)
        assert SporeStore(store.path).get("spore-001")["disposition"] == "seed"


# -- update(type=): the retype primitive (the "forming-vs-committed" lock is the
# caller's policy, NOT anneal's — anneal mechanically allows it). -----------------


class TestRetype:
    def test_update_retypes_a_spore(self, store):
        store.add(type="thought", text="x", today=T0)
        s = store.update("spore-001", type="task")
        assert s["type"] == "task"
        assert SporeStore(store.path).get("spore-001")["type"] == "task"

    def test_update_rejects_a_bad_type(self, store):
        store.add(type="thought", text="x", today=T0)
        with pytest.raises(ValueError, match="type must be one of"):
            store.update("spore-001", type="bogus")  # type: ignore[arg-type]

    def test_retype_changes_the_valid_terminal_kinds(self, store):
        # the point of retype: a thought retyped to a task can now descend 'done'
        # (which a thought cannot) — the forming->ready path.
        store.add(type="thought", text="x", today=T0)
        store.update("spore-001", type="task")
        store.descend("spore-001", kind="done")  # task-only kind; would raise on a thought
        assert SporeStore(store.path).get("spore-001")["status"] == "resolved"

    def test_update_omitting_type_leaves_it_unchanged(self, store):
        store.add(type="question", text="x", today=T0)
        store.update("spore-001", tier="hot")
        assert SporeStore(store.path).get("spore-001")["type"] == "question"


# -- expected_version: the whole-spore compare-and-set (cockpit slice P) --------


class TestExpectedVersion:
    def test_version_ignores_seen_and_tracks_every_other_field(self, store):
        s = store.add(type="task", text="x", today=T0)
        v = spore_version(s)
        touched = store.touch("spore-001", today=T0 + timedelta(days=2))
        assert spore_version(touched) == v
        noted = store.update("spore-001", add_note="n", today=T0)
        assert spore_version(noted) != v

    def test_a_touch_that_clears_an_elapsed_next_changes_the_version(self, store):
        s = store.add(type="task", text="x", next=(T0 + timedelta(days=1)).isoformat(), today=T0)
        cleared = store.touch("spore-001", today=T0 + timedelta(days=2))
        assert cleared["next"] is None
        assert spore_version(cleared) != spore_version(s)

    @_NEEDS_LOCK
    @pytest.mark.parametrize("op", ["update", "touch", "descend", "ascend"])
    def test_a_current_version_applies(self, store, op):
        s = store.add(type="question", text="x", today=T0)
        v = spore_version(s)
        later = T0 + timedelta(days=1)
        call, landed = {
            "update": (lambda: store.update("spore-001", text="y", expected_version=v),
                       lambda sp: sp["text"] == "y"),
            "touch": (lambda: store.touch("spore-001", today=later, expected_version=v),
                      lambda sp: sp["seen"] == later.isoformat()),
            "descend": (lambda: store.descend("spore-001", kind="answered", today=T0, expected_version=v),
                        lambda sp: sp["status"] == "resolved"),
            "ascend": (lambda: store.ascend("spore-001", kind="pattern", ref="r", today=T0, expected_version=v),
                       lambda sp: sp["status"] == "resolved"),
        }[op]
        call()
        assert landed(SporeStore(store.path).get("spore-001"))

    @_NEEDS_LOCK
    @pytest.mark.parametrize("op", ["update", "touch", "descend", "ascend"])
    def test_a_stale_version_is_refused_and_nothing_is_written(self, store, op):
        s = store.add(type="question", text="x", today=T0)
        stale = spore_version(s)
        store.update("spore-001", text="changed elsewhere")
        before = store.path.read_bytes()
        call = {
            "update": lambda: store.update("spore-001", text="y", expected_version=stale),
            "touch": lambda: store.touch("spore-001", today=T0 + timedelta(days=1), expected_version=stale),
            "descend": lambda: store.descend("spore-001", kind="answered", today=T0, expected_version=stale),
            "ascend": lambda: store.ascend("spore-001", kind="pattern", ref="r", today=T0, expected_version=stale),
        }[op]
        with pytest.raises(SporeError, match="changed since read"):
            call()
        assert store.path.read_bytes() == before

    @_NEEDS_LOCK
    def test_version_of_lets_a_caller_bind_its_own_hash(self, store):
        store.add(type="task", text="x", today=T0)

        def text_only(sp):
            return sp["text"]

        store.update("spore-001", tier="hot", expected_version="x", version_of=text_only)
        with pytest.raises(SporeError, match="changed since read"):
            store.update("spore-001", tier="cold", expected_version="other", version_of=text_only)

    @pytest.mark.parametrize("bad", ["", 7, b"v"])
    def test_a_malformed_expected_version_is_a_value_error(self, store, bad):
        store.add(type="task", text="x", today=T0)
        with pytest.raises(ValueError, match="expected_version"):
            store.update("spore-001", text="y", expected_version=bad)  # type: ignore[arg-type]

    def test_version_of_without_expected_version_is_a_value_error(self, store):
        store.add(type="task", text="x", today=T0)
        with pytest.raises(ValueError, match="without expected_version"):
            store.update("spore-001", text="y", version_of=lambda sp: "v")
        assert SporeStore(store.path).get("spore-001")["text"] == "x"

    @_NEEDS_LOCK
    def test_a_version_of_returning_a_non_str_is_a_type_error(self, store):
        store.add(type="task", text="x", today=T0)
        with pytest.raises(TypeError, match="must return a str"):
            store.update("spore-001", text="y", expected_version="1", version_of=lambda sp: 1)  # type: ignore[arg-type,return-value]

    @_NEEDS_LOCK
    def test_version_of_cannot_change_what_is_saved(self, store):
        store.add(type="task", text="x", today=T0)

        def meddling(sp):
            sp.pop("seen")
            sp["text"] = "meddled"
            return "v"

        store.update("spore-001", tier="hot", expected_version="v", version_of=meddling)
        saved = SporeStore(store.path).get("spore-001")
        assert saved["text"] == "x" and saved["seen"] == T0.isoformat() and saved["tier"] == "hot"

    @pytest.mark.parametrize("guard", ["expected_version", "expect_disposition"])
    @pytest.mark.parametrize("op", ["update", "descend", "ascend", "touch"])
    def test_without_a_file_lock_a_guarded_write_is_refused(self, store, monkeypatch, op, guard):
        if op == "touch" and guard == "expect_disposition":
            pytest.skip("touch takes no expect_disposition")
        s = store.add(type="question", text="x", today=T0)
        monkeypatch.setattr(_spores, "fcntl", None)
        before = store.path.read_bytes()
        kw = {"expected_version": spore_version(s)} if guard == "expected_version" else {"expect_disposition": None}
        call = {
            "update": lambda: store.update("spore-001", text="y", **kw),
            "touch": lambda: store.touch("spore-001", today=T0, **kw),
            "descend": lambda: store.descend("spore-001", kind="answered", today=T0, **kw),
            "ascend": lambda: store.ascend("spore-001", kind="pattern", ref="r", today=T0, **kw),
        }[op]
        with pytest.raises(SporeError, match="needs a file lock"):
            call()
        assert store.path.read_bytes() == before
        store.update("spore-001", text="y")  # an unguarded write is unchanged

    @_NEEDS_LOCK
    def test_a_falsy_version_of_is_still_the_one_called(self, store):
        store.add(type="task", text="x", today=T0)

        class Falsy:
            def __bool__(self):
                return False

            def __call__(self, sp):
                return "mine"

        with pytest.raises(SporeError, match="changed since read"):
            store.update("spore-001", text="y", expected_version=spore_version(store.get("spore-001")),
                         version_of=Falsy())
        store.update("spore-001", text="y", expected_version="mine", version_of=Falsy())

    def test_a_non_callable_version_of_is_a_value_error(self, store):
        store.add(type="task", text="x", today=T0)
        with pytest.raises(ValueError, match="must be callable"):
            store.update("spore-001", text="y", expected_version="v", version_of=0)  # type: ignore[arg-type]

    def test_an_explicit_default_version_of_without_expected_version_is_a_value_error(self, store):
        store.add(type="task", text="x", today=T0)
        with pytest.raises(ValueError, match="without expected_version"):
            store.update("spore-001", text="y", version_of=spore_version)

    def test_a_lone_surrogate_in_the_store_still_versions(self, store):
        store.add(type="task", text="x", today=T0)
        raw = store.path.read_text(encoding="utf-8").replace('"text": "x"', '"text": "\\ud800"')
        store.path.write_text(raw, encoding="utf-8")
        sp = SporeStore(store.path).get("spore-001")
        assert sp["text"] == "\ud800"
        assert len(spore_version(sp)) == 64

    @pytest.mark.skipif(_spores.fcntl is None, reason="the cross-process lock is POSIX fcntl")
    @pytest.mark.parametrize("with_version", [True, False])
    def test_a_write_landing_under_the_lock_after_the_read_is_refused(self, tmp_path, with_version):
        """A second process takes the lock and rewrites the text AFTER this caller
        read the spore. The caller's update waits on the lock, then must see that
        write. The control (no expected_version) shows the window is real: there
        the late update overwrites the other writer's text."""
        import time

        path = tmp_path / "spores.json"
        store = SporeStore(path)
        read = store.add(type="task", text="original", today=T0)
        ctx = mp.get_context("spawn")
        held, release = ctx.Event(), ctx.Event()
        proc = ctx.Process(target=_hold_lock_and_edit, args=(str(path), held, release))
        proc.start()
        try:
            assert held.wait(timeout=30)
            t0 = time.monotonic()
            timer = __import__("threading").Timer(0.5, release.set)
            timer.start()
            kwargs = {"expected_version": spore_version(read)} if with_version else {}
            if with_version:
                with pytest.raises(SporeError, match="changed since read"):
                    store.update("spore-001", text="late write", **kwargs)
            else:
                store.update("spore-001", text="late write")
            waited = time.monotonic() - t0
        finally:
            release.set()
            proc.join(timeout=30)
        assert proc.exitcode == 0
        assert waited >= 0.45  # the update really queued behind the held lock
        expected_text = "edited by the other writer" if with_version else "late write"
        assert SporeStore(path).get("spore-001")["text"] == expected_text


# -- origin_key and the stored-text normaliser (cockpit slice P, second half) --


class TestOriginKey:
    def test_every_new_spore_gets_a_distinct_key(self, store):
        a = store.add(type="task", text="a", today=T0)
        b = store.add(type="task", text="b", today=T0)
        assert a["origin_key"] and b["origin_key"] and a["origin_key"] != b["origin_key"]

    def test_the_same_key_planted_twice_is_one_spore_and_returns_the_first(self, store):
        first = store.add(type="task", text="a", origin_key="pend-1", today=T0)
        second = store.add(type="task", text="a retried", origin_key="pend-1", today=T0)
        assert second["id"] == first["id"]
        assert len(SporeStore(store.path).list_open()) == 1
        assert SporeStore(store.path).get(first["id"])["text"] == "a"

    def test_a_resolved_spore_still_holds_its_key(self, store):
        first = store.add(type="task", text="a", origin_key="pend-1", today=T0)
        store.descend(first["id"], kind="done", today=T0)
        again = store.add(type="task", text="a", origin_key="pend-1", today=T0)
        assert again["id"] == first["id"] and again["status"] == "resolved"
        assert store.list_open() == []

    def test_lookup(self, store):
        s = store.add(type="task", text="a", today=T0)
        assert store.get_by_origin_key(s["origin_key"])["id"] == s["id"]
        assert store.get_by_origin_key("nope") is None

    @_NEEDS_LOCK
    def test_writes_keep_the_key(self, store):
        s = store.add(type="question", text="a", today=T0)
        key = s["origin_key"]
        store.update(s["id"], text="b", tier="hot", domain="d", disposition="seed", add_note="n",
                     expected_version=spore_version(s))
        store.touch(s["id"], today=T0 + timedelta(days=1))
        store.ascend(s["id"], kind="pattern", ref="r", today=T0)
        assert SporeStore(store.path).get(s["id"])["origin_key"] == key

    def test_a_store_without_keys_is_backfilled_once_and_versions_do_not_move(self, store):
        store.add(type="task", text="a", today=T0)
        store.add(type="task", text="b", today=T0)
        store.descend("spore-002", kind="done", today=T0)
        data = json.loads(store.path.read_text())
        for item in data["spores"] + data["resolved"]:
            del item["origin_key"]
        store.path.write_text(json.dumps(data))
        before = spore_version(store.get("spore-001"))
        assert store.get_by_origin_key("anything") is None
        assert store.backfill_origin_keys() == 2
        assert store.backfill_origin_keys() == 0
        a, b = store.get("spore-001"), store.get("spore-002")
        assert a["origin_key"] and b["origin_key"] and a["origin_key"] != b["origin_key"]
        assert spore_version(a) == before
        assert store.get_by_origin_key(b["origin_key"])["id"] == "spore-002"

    def test_any_write_backfills(self, store):
        store.add(type="task", text="a", today=T0)
        data = json.loads(store.path.read_text())
        del data["spores"][0]["origin_key"]
        store.path.write_text(json.dumps(data))
        store.add(type="task", text="b", today=T0)
        assert store.get("spore-001")["origin_key"]

    @pytest.mark.parametrize("bad", ["", "   ", 5, " k", "k ", "k\x07", "k\u200b", None])
    def test_a_malformed_key_is_a_value_error(self, store, bad):
        if bad is not None:
            with pytest.raises(ValueError, match="origin_key"):
                store.add(type="task", text="a", origin_key=bad, today=T0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="origin_key"):
            store.get_by_origin_key(bad)  # type: ignore[arg-type]

    def test_a_keyless_legacy_spore_is_never_found_by_a_missing_key(self, store):
        store.add(type="task", text="a", today=T0)
        data = json.loads(store.path.read_text())
        del data["spores"][0]["origin_key"]
        store.path.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            store.get_by_origin_key(None)  # type: ignore[arg-type]
        assert store._find_by_origin_key(json.loads(store.path.read_text()), None) is None  # type: ignore[arg-type]

    def test_a_retried_create_does_not_rewrite_the_file(self, store, monkeypatch):
        store.add(type="task", text="a", origin_key="pend-1", today=T0)
        calls = []
        monkeypatch.setattr(store, "_save", lambda data: calls.append(1))
        again = store.add(type="task", text="a", origin_key="pend-1", today=T0)
        assert again["id"] == "spore-001" and calls == []

    def test_a_backfill_count_is_per_call(self, store, monkeypatch):
        store.add(type="task", text="a", today=T0)
        data = json.loads(store.path.read_text())
        del data["spores"][0]["origin_key"]
        store.path.write_text(json.dumps(data))
        assert not hasattr(store, "_last_backfill")
        assert store.backfill_origin_keys() == 1


class TestNormalizer:
    FIXTURE = "  A‮B⁦C‏  \r\nline two\x07\x00\t \r\ne‎́nd  \n\n"

    def test_what_the_normaliser_returns_is_what_add_and_update_store(self, store):
        from anneal_memory import normalize_spore_field

        want = normalize_spore_field(self.FIXTURE)
        assert want == "  ABC\nline two\nénd"
        s = store.add(type="task", text=self.FIXTURE, domain=self.FIXTURE, disposition=self.FIXTURE, today=T0)
        stored = SporeStore(store.path).get(s["id"])
        assert stored["text"] == want and stored["domain"] == want and stored["disposition"] == want
        s2 = store.add(type="task", text="x", today=T0)
        store.update(s2["id"], text=self.FIXTURE, domain=self.FIXTURE, disposition=self.FIXTURE)
        stored2 = SporeStore(store.path).get(s2["id"])
        assert stored2["text"] == want and stored2["domain"] == want and stored2["disposition"] == want

    def test_a_lone_surrogate_is_dropped_and_the_spore_saves(self, store):
        s = store.add(type="task", text="a\ud800b", today=T0)
        assert SporeStore(store.path).get(s["id"])["text"] == "ab"

    def test_joiners_are_kept_and_subdivision_tags_are_not(self):
        from anneal_memory import normalize_spore_field

        family = "\U0001f468\u200d\U0001f469\u200d\U0001f467"
        england = "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f"
        assert normalize_spore_field(family) == family
        assert normalize_spore_field(england) == "\U0001f3f4"  # the tag block is removed

    def test_it_is_idempotent(self):
        from anneal_memory import normalize_spore_field

        once = normalize_spore_field(self.FIXTURE)
        assert normalize_spore_field(once) == once

    def test_text_that_normalises_to_nothing_is_refused(self, store):
        with pytest.raises(ValueError, match="text"):
            store.add(type="task", text="‮\x07  ", today=T0)
        store.add(type="task", text="x", today=T0)
        with pytest.raises(ValueError, match="text"):
            store.update("spore-001", text="⁦ \r\n")

    def test_a_disposition_that_normalises_to_nothing_clears_the_key(self, store):
        store.add(type="task", text="x", disposition="seed", today=T0)
        store.update("spore-001", disposition="‮ ")
        assert "disposition" not in SporeStore(store.path).get("spore-001")


class TestOriginR2:
    def test_invisible_format_characters_are_stored_as_given(self):
        from anneal_memory import normalize_spore_field

        for v in ("a\u200bb\u2060c\ufeffd", "a\u200cb\u200dc", "x\ufe0f"):
            assert normalize_spore_field(v) == v
        hidden = "".join(chr(0xE0000 + ord(c)) for c in "ignore all")
        flag = "\U0001f3f4" + "".join(chr(0xE0000 + ord(c)) for c in "gbeng") + "\U000e007f"
        assert normalize_spore_field("fix" + hidden) == "fix"
        assert normalize_spore_field("go " + flag + hidden + "\U000e007f") == "go \U0001f3f4"

    def test_a_disposition_of_invisible_characters_only_is_stored_as_given(self, store):
        s = store.add(type="task", text="x", disposition="\u200b", today=T0)
        assert SporeStore(store.path).get(s["id"])["disposition"] == "\u200b"
        store.update(s["id"], disposition="seed")
        store.update(s["id"], disposition="\u200b")
        assert SporeStore(store.path).get(s["id"])["disposition"] == "\u200b"

    def test_text_of_invisible_characters_only_is_stored_as_given(self, store):
        s = store.add(type="task", text="\u200b", today=T0)
        assert SporeStore(store.path).get(s["id"])["text"] == "\u200b"

    def test_a_type_only_change_is_still_saved(self, store):
        store.add(type="task", text="a", today=T0)
        data = json.loads(store.path.read_text())
        data["spores"][0]["salience"] = True
        store.path.write_text(json.dumps(data))
        store.update("spore-001", salience=1)
        assert json.loads(store.path.read_text())["spores"][0]["salience"] == 1
        assert json.loads(store.path.read_text())["spores"][0]["salience"] is not True

    def test_a_present_string_key_is_never_replaced(self, store):
        store.add(type="task", text="a", today=T0)
        data = json.loads(store.path.read_text())
        data["spores"][0]["origin_key"] = "hand\tedited"
        store.path.write_text(json.dumps(data))
        assert store.backfill_origin_keys() == 0
        assert store.get("spore-001")["origin_key"] == "hand\tedited"

    @pytest.mark.parametrize("bad", [5, "", None])
    def test_a_missing_or_non_string_key_is_replaced_by_the_backfill(self, store, bad):
        store.add(type="task", text="a", today=T0)
        data = json.loads(store.path.read_text())
        data["spores"][0]["origin_key"] = bad
        store.path.write_text(json.dumps(data))
        assert store.backfill_origin_keys() == 1
        key = store.get("spore-001")["origin_key"]
        assert isinstance(key, str) and len(key) == 32


class TestDuplicateOpenId:
    """Two open spores sharing an id (store drift): every mutator refuses rather
    than writing to one copy, whatever version the caller read (levain K2a r3)."""

    def _drifted(self, store):
        store.add(type="task", text="first", today=T0)
        data = json.loads(store.path.read_text())
        dup = dict(data["spores"][0])
        dup["text"] = "second copy"
        dup["origin_key"] = "f" * 32
        data["spores"].append(dup)
        store.path.write_text(json.dumps(data))
        return SporeStore(store.path), data

    def test_a_guarded_update_matching_one_copy_is_refused(self, store):
        s, data = self._drifted(store)
        before = store.path.read_text()
        for copy in data["spores"]:
            with pytest.raises(SporeError, match="ambiguous"):
                s.update("spore-001", text="edited", expected_version=spore_version(copy))
        assert store.path.read_text() == before

    def test_every_mutator_is_refused(self, store):
        s, _ = self._drifted(store)
        before = store.path.read_text()
        for call in (
            lambda: s.touch("spore-001"),
            lambda: s.update("spore-001", text="x"),
            lambda: s.descend("spore-001", kind="done", today=T0),
            lambda: s.ascend("spore-001", kind="pattern", ref="my_pattern", today=T0),
        ):
            with pytest.raises(SporeError, match="ambiguous"):
                call()
        assert store.path.read_text() == before
