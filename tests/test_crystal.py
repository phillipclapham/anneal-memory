"""Tests for the crystallized-pattern layer (crystal.py).

Covers activation-tier computation, the full crystallize→activate→retire
lifecycle (incl. monotonic upsert + un-retire on re-crystallize), the
library-ization invariants (raises not sys.exit; returns dicts; injectable
clock for determinism), ranking/filtering, the re-warm surface, and the
load-bearing store invariant shared with spores: a corrupt store NEVER silently
re-inits.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from datetime import date, datetime, timezone

import pytest

from anneal_memory import CrystalError, CrystalStore, activation_tier
from anneal_memory import crystal as _crystal
from anneal_memory.crystal import CRYSTAL_SCHEMA_VERSION


# Module-level so the "spawn" start method can pickle it by qualified name.
def _concurrent_crystallize_worker(path_str: str, barrier, idx: int, count: int) -> None:
    """Crystallize ``count`` distinct patterns against a shared store, aligned to a
    barrier so all writers hit the load→mutate→save window together."""
    store = CrystalStore(path_str)
    try:
        barrier.wait(timeout=30)
    except Exception:  # pragma: no cover - a sibling died before the barrier
        return
    for j in range(count):
        store.crystallize(
            name=f"w{idx}_{j}", level=2, explanation=f"pattern w{idx}-{j}",
            today=date(2026, 1, 1),
        )


@pytest.fixture
def store(tmp_path):
    return CrystalStore(tmp_path / "mem.crystal.json")


T0 = date(2026, 1, 1)


# -- activation tier -------------------------------------------------------


class TestActivationTier:
    def _crystal(self, **kw):
        base = {"last_activated_on": T0.isoformat()}
        base.update(kw)
        return base

    def test_hot_under_7_days(self):
        c = self._crystal(last_activated_on=date(2026, 1, 1).isoformat())
        assert activation_tier(c, date(2026, 1, 6)) == "hot"  # 5 days

    def test_warm_7_to_30(self):
        c = self._crystal(last_activated_on=date(2026, 1, 1).isoformat())
        assert activation_tier(c, date(2026, 1, 8)) == "warm"   # 7 days (boundary)
        assert activation_tier(c, date(2026, 1, 31)) == "warm"  # 30 days (boundary)

    def test_cold_30_to_90(self):
        c = self._crystal(last_activated_on=date(2026, 1, 1).isoformat())
        assert activation_tier(c, date(2026, 2, 1)) == "cold"   # 31 days
        assert activation_tier(c, date(2026, 4, 1)) == "cold"   # 90 days (boundary)

    def test_dormant_over_90(self):
        c = self._crystal(last_activated_on=date(2026, 1, 1).isoformat())
        assert activation_tier(c, date(2026, 4, 2)) == "dormant"  # 91 days

    def test_unparseable_is_dormant(self):
        assert activation_tier({"last_activated_on": None}, T0) == "dormant"
        assert activation_tier({"last_activated_on": "garbage"}, T0) == "dormant"
        assert activation_tier({}, T0) == "dormant"

    def test_future_activation_is_hot(self):
        # A future last_activated (clock skew) reads as age<0 → hot, never dormant.
        c = self._crystal(last_activated_on=date(2026, 1, 10).isoformat())
        assert activation_tier(c, date(2026, 1, 1)) == "hot"


# -- crystallize -----------------------------------------------------------


class TestCrystallize:
    def test_basic_fields(self, store):
        c = store.crystallize(
            name="play_first", level=2, explanation="grinding breaks productivity",
            evidence=["ep111111", "ep222222"], tags=["cognition", "voice"],
            source="flow", today=T0,
        )
        assert c["name"] == "play_first"
        assert c["level"] == 2
        assert c["evidence"] == ["ep111111", "ep222222"]
        assert c["tags"] == ["cognition", "voice"]
        assert c["permanence"] == "timeless"
        assert c["activation_mode"] == "just-in-time"
        assert c["status"] == "crystallized"
        assert c["crystallized_on"] == "2026-01-01"
        assert c["last_activated_on"] == "2026-01-01"
        assert c["source"] == "flow"
        assert c["retirement"] is None

    def test_routing_axes(self, store):
        c = store.crystallize(
            name="p", level=3, explanation="x",
            permanence="phase-specific", activation_mode="catastrophic", today=T0,
        )
        assert c["permanence"] == "phase-specific"
        assert c["activation_mode"] == "catastrophic"

    def test_upsert_monotonic_level_keeps_origin(self, store):
        store.crystallize(name="p", level=3, explanation="v1", today=T0)
        c = store.crystallize(name="p", level=2, explanation="v2", today=date(2026, 2, 1))
        assert c["level"] == 3  # monotonic — never lowered
        assert c["explanation"] == "v2"  # content refreshed
        assert c["crystallized_on"] == "2026-01-01"  # origin preserved
        assert len(store.active()) == 1  # upsert, not duplicate

    def test_upsert_raises_monotonic_via_lower_then_higher(self, store):
        store.crystallize(name="p", level=2, explanation="v1", today=T0)
        c = store.crystallize(name="p", level=3, explanation="v2", today=T0)
        assert c["level"] == 3  # can rise

    def test_evidence_dedup_order_preserving(self, store):
        c = store.crystallize(name="p", level=2, explanation="x",
                              evidence=["a", "b", "a", "c"], today=T0)
        assert c["evidence"] == ["a", "b", "c"]

    @pytest.mark.parametrize("bad", [
        {"name": "", "level": 3, "explanation": "x"},
        {"name": "  ", "level": 3, "explanation": "x"},
        {"name": "p", "level": 1, "explanation": "x"},   # 1x not Proven-tier
        {"name": "p", "level": 4, "explanation": "x"},
        {"name": "p", "level": True, "explanation": "x"},  # bool rejected
        {"name": "p", "level": 3, "explanation": ""},
        {"name": "p", "level": 3, "explanation": "x", "evidence": "not-a-list"},
        {"name": "p", "level": 3, "explanation": "x", "evidence": [""]},
        {"name": "p", "level": 3, "explanation": "x", "permanence": "forever"},
        {"name": "p", "level": 3, "explanation": "x", "activation_mode": "always"},
    ])
    def test_validation_raises_valueerror(self, store, bad):
        with pytest.raises(ValueError):
            store.crystallize(today=T0, **bad)

    def test_bare_string_evidence_rejected_not_charsplit(self, store):
        # A bare string would otherwise store each character — reject it.
        with pytest.raises(ValueError):
            store.crystallize(name="p", level=2, explanation="x", evidence="abc", today=T0)


# -- read / rank / filter --------------------------------------------------


class TestReadRankFilter:
    def _seed(self, store):
        store.crystallize(name="hot3", level=3, explanation="hot three", tags=["a"], today=date(2026, 6, 6))
        store.crystallize(name="hot2", level=2, explanation="hot two", tags=["a", "b"], today=date(2026, 6, 6))
        store.crystallize(name="dorm", level=3, explanation="dormant three", tags=["b"], today=date(2026, 1, 1))

    def test_get_returns_raw_record_no_computed_field(self, store):
        store.crystallize(name="p", level=2, explanation="x", today=T0)
        c = store.get("p")
        assert c is not None
        assert "activation" not in c  # computed, never stored

    def test_get_missing_is_none(self, store):
        assert store.get("nope") is None

    def test_rank_activation_then_level(self, store):
        self._seed(store)
        names = [c["name"] for c in store.list_crystal(today=date(2026, 6, 6))]
        # hot first (hot3 before hot2 by level), dormant last
        assert names == ["hot3", "hot2", "dorm"]

    def test_filter_by_tag(self, store):
        self._seed(store)
        names = {c["name"] for c in store.list_crystal(tag="b", today=date(2026, 6, 6))}
        assert names == {"hot2", "dorm"}

    def test_filter_by_activation(self, store):
        self._seed(store)
        hot = store.list_crystal(activation="hot", today=date(2026, 6, 6))
        assert {c["name"] for c in hot} == {"hot3", "hot2"}

    def test_filter_by_permanence_and_mode(self, store):
        store.crystallize(name="t", level=2, explanation="x", permanence="timeless", today=T0)
        store.crystallize(name="ph", level=2, explanation="x", permanence="phase-specific", today=T0)
        assert {c["name"] for c in store.list_crystal(permanence="timeless")} == {"t"}
        assert {c["name"] for c in store.list_crystal(permanence="phase-specific")} == {"ph"}

    def test_active_excludes_retired(self, store):
        self._seed(store)
        store.retire("dorm", kind="obsolete", today=date(2026, 6, 7))
        assert {c["name"] for c in store.active()} == {"hot3", "hot2"}

    def test_surface_rewarm_only_hot(self, store):
        self._seed(store)
        rw = store.surface_rewarm_candidates(today=date(2026, 6, 6))
        assert {c["name"] for c in rw} == {"hot3", "hot2"}  # dorm excluded


# -- touch / activate ------------------------------------------------------


class TestTouch:
    def test_touch_reheats(self, store):
        store.crystallize(name="p", level=2, explanation="x", today=date(2026, 1, 1))
        assert activation_tier(store.get("p"), date(2026, 6, 6)) == "dormant"
        store.touch("p", today=date(2026, 6, 6))
        assert activation_tier(store.get("p"), date(2026, 6, 6)) == "hot"

    def test_touch_missing_raises(self, store):
        with pytest.raises(CrystalError):
            store.touch("nope")

    def test_touch_retired_raises_with_retired_message(self, store):
        store.crystallize(name="p", level=2, explanation="x", today=T0)
        store.retire("p", kind="superseded", today=T0)
        with pytest.raises(CrystalError, match="already retired"):
            store.touch("p")


# -- update ----------------------------------------------------------------


class TestUpdate:
    def test_reroute_and_retag(self, store):
        store.crystallize(name="p", level=2, explanation="x", today=T0)
        c = store.update("p", permanence="phase-specific", activation_mode="catastrophic",
                         tags=["x", "y"], today=T0)
        assert c["permanence"] == "phase-specific"
        assert c["activation_mode"] == "catastrophic"
        assert c["tags"] == ["x", "y"]

    def test_update_level_is_explicit_not_monotonic(self, store):
        # update is the explicit-correction path: it CAN lower (unlike crystallize)
        store.crystallize(name="p", level=3, explanation="x", today=T0)
        c = store.update("p", level=2, today=T0)
        assert c["level"] == 2

    def test_update_level_still_validated(self, store):
        store.crystallize(name="p", level=3, explanation="x", today=T0)
        with pytest.raises(ValueError):
            store.update("p", level=1)

    def test_update_does_not_bump_activation(self, store):
        store.crystallize(name="p", level=2, explanation="x", today=date(2026, 1, 1))
        store.update("p", tags=["z"], today=date(2026, 6, 6))
        # last_activated unchanged → still dormant relative to 2026-06-06
        assert store.get("p")["last_activated_on"] == "2026-01-01"

    def test_clear_source_with_empty_string(self, store):
        store.crystallize(name="p", level=2, explanation="x", source="flow", today=T0)
        c = store.update("p", source="", today=T0)
        assert c["source"] is None

    def test_add_note_stamped(self, store):
        store.crystallize(name="p", level=2, explanation="x", today=T0)
        c = store.update("p", add_note="re-routed after review", today=date(2026, 2, 1))
        assert any("2026-02-01" in n and "re-routed" in n for n in c["notes"])


# -- retire / re-crystallize ----------------------------------------------


class TestRetireAndRevive:
    def test_retire_moves_to_retired_with_record(self, store):
        store.crystallize(name="p", level=2, explanation="x", today=T0)
        c = store.retire("p", kind="falsified", reason="contradiction scan pulled it",
                         today=date(2026, 6, 8),
                         now=datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc))
        assert c["status"] == "retired"
        assert c["retirement"]["kind"] == "falsified"
        assert c["retirement"]["reason"] == "contradiction scan pulled it"
        assert c["retirement"]["on"] == "2026-06-08"
        assert c["retirement"]["at"] == "2026-06-08T12:00:00+00:00"
        assert store.get("p")["status"] == "retired"
        assert len(store.active()) == 0

    def test_retire_invalid_kind_raises(self, store):
        store.crystallize(name="p", level=2, explanation="x", today=T0)
        with pytest.raises(ValueError):
            store.retire("p", kind="deleted")

    def test_retire_naive_now_raises(self, store):
        store.crystallize(name="p", level=2, explanation="x", today=T0)
        with pytest.raises(ValueError):
            store.retire("p", kind="obsolete", now=datetime(2026, 6, 8, 12, 0))

    def test_retire_missing_raises(self, store):
        with pytest.raises(CrystalError):
            store.retire("nope", kind="obsolete")

    def test_recrystallize_unretires_with_note_and_fresh_date(self, store):
        store.crystallize(name="p", level=3, explanation="v1", today=T0)
        store.retire("p", kind="superseded", today=date(2026, 6, 8))
        c = store.crystallize(name="p", level=2, explanation="re-proven", today=date(2026, 6, 9))
        assert c["status"] == "crystallized"
        assert c["level"] == 2  # fresh crystallization, not the old high-water
        assert c["crystallized_on"] == "2026-06-09"
        assert any("re-crystallized after retirement" in n for n in c["notes"])
        assert len(store.active()) == 1
        # the retired row is gone (revived, not duplicated)
        assert not any(r["name"] == "p" for r in store._load()["retired"])


# -- store invariants (shared with spores) ---------------------------------


class TestStoreInvariants:
    def test_corrupt_json_never_reinits(self, store):
        store.path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(CrystalError, match="not valid JSON"):
            store.crystallize(name="p", level=2, explanation="x")

    def test_non_object_root_rejected(self, store):
        store.path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(CrystalError, match="must contain a JSON object"):
            store._load()

    def test_non_list_sections_rejected(self, store):
        store.path.write_text(json.dumps({"crystal": {}, "retired": []}), encoding="utf-8")
        with pytest.raises(CrystalError, match="must be lists"):
            store._load()

    def test_newer_schema_version_refused(self, store):
        store.path.write_text(
            json.dumps({"crystal": [], "retired": [], "schema_version": CRYSTAL_SCHEMA_VERSION + 1}),
            encoding="utf-8",
        )
        with pytest.raises(CrystalError, match="newer crystal schema"):
            store._load()

    def test_missing_file_loads_empty(self, store):
        data = store._load()
        assert data == {"crystal": [], "retired": [], "schema_version": CRYSTAL_SCHEMA_VERSION}

    def test_atomic_write_no_tmp_leak(self, store, tmp_path):
        store.crystallize(name="p", level=2, explanation="x", today=T0)
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []


# -- concurrency (mirrors test_spores) -------------------------------------


class TestConcurrency:
    def test_parallel_crystallize_no_lost_updates(self, tmp_path):
        path = tmp_path / "mem.crystal.json"
        n_workers, per_worker = 4, 10
        ctx = mp.get_context("spawn")
        barrier = ctx.Barrier(n_workers)
        procs = [
            ctx.Process(target=_concurrent_crystallize_worker,
                        args=(str(path), barrier, i, per_worker))
            for i in range(n_workers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
            assert p.exitcode == 0
        store = CrystalStore(path)
        assert len(store.active()) == n_workers * per_worker  # no lost updates
