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
import sys
from datetime import date, datetime, timezone

import pytest

from anneal_memory import CrystalError, CrystalSnapshot, CrystalStore, activation_tier
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
        {"name": "p", "level": 0, "explanation": "x"},   # below the floor
        {"name": "p", "level": -3, "explanation": "x"},  # negative
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

    @pytest.mark.parametrize("level", [2, 3, 4, 5, 10, 27])
    def test_there_is_NO_UPPER_BOUND_on_level(self, store, level):
        """AM-LEVELCAP (2026-08-14). ``{"level": 4}`` used to sit in the bad-args list
        above, asserting a ceiling that made a well-earned pattern IMPOSSIBLE TO
        CRYSTALLIZE OUT — while graduation.py's own ``([23])x`` cap simultaneously made
        it invisible to validation. A pattern above 3x was trapped in the working set
        with no exit, which is the precise failure the module docstring says this store
        exists to prevent ("the section had no OUT path — a one-way ratchet, so the
        working set monotonically bloats").

        The ceiling also contradicted this store's own contract: ``crystallize`` raises
        level "monotonically (never lowers it — a pattern's earned high-water mark
        holds)", and a high-water mark that saturates at 3 is not a high-water mark.

        Level is the STRENGTH axis (times lived experience re-earned it);
        ``last_activated_on`` is the RECENCY axis. Both are real and they are not
        interchangeable, so strength must be free to climb.
        """
        rec = store.crystallize(name=f"deep{level}", level=level,
                                explanation="x", evidence=["aabbccdd"], today=T0)
        assert rec["level"] == level

    def test_the_floor_is_still_enforced_at_2(self, store):
        """The half of the old scoping that was CORRECT and must survive: 1x is a
        developing pattern and belongs in the working set, not the deep store. Lifting
        the ceiling must not lower the floor."""
        with pytest.raises(ValueError, match="Proven-tier"):
            store.crystallize(name="dev", level=1, explanation="x", today=T0)

    def test_monotonic_raise_holds_above_3(self, store):
        """The high-water mark now actually works past the old ceiling: re-crystallizing
        raises the level and NEVER lowers it, at any magnitude."""
        store.crystallize(name="climber", level=3, explanation="x", today=T0)
        assert store.crystallize(name="climber", level=9, explanation="x", today=T0)["level"] == 9
        assert store.crystallize(name="climber", level=4, explanation="x", today=T0)["level"] == 9

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
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="CrystalStore._transaction's own docstring discloses that the "
        "fcntl advisory lock degrades to a no-op on non-POSIX platforms; "
        "cross-process write serialization is a documented POSIX-only "
        "guarantee, not a bug this test should catch on Windows",
    )
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


def test_load_present_but_unreadable_raises_crystal_error(tmp_path):
    """A present-but-unreadable crystal path (here: a directory at the file path) makes
    open() raise IsADirectoryError (an OSError, NOT FileNotFoundError) — normalized to
    CrystalError at the source so a crystal fault always surfaces as the crystal store's
    OWN error type, never a bare OSError a caller misclassifies. (codex L3 MED, spore-059.)"""
    p = tmp_path / "mem.crystal.json"
    p.mkdir()
    with pytest.raises(CrystalError):
        CrystalStore(p).active()


def test_expect_refuses_a_stale_write_and_writes_nothing(tmp_path):
    """The lost update codex found (2026-09-24, input 2db1236917b0cfa1): A reads, B
    writes, A writes from its stale read. Without ``expect`` A silently overwrote B's
    newer explanation and evidence; with it, A is refused and B's record stands."""
    from anneal_memory import CrystalConflictError

    p = tmp_path / "mem.crystal.json"
    a, b = CrystalStore(p), CrystalStore(p)
    a.crystallize(name="x", level=2, explanation="v1", today=date(2026, 1, 1))
    seen = a.get("x")["rev"]
    b.crystallize(name="x", level=3, explanation="B newer", evidence=["e1"],
                  today=date(2026, 1, 1))
    before = p.read_bytes()
    with pytest.raises(CrystalConflictError) as exc:
        a.crystallize(name="x", level=2, explanation="A stale", expect=seen,
                      today=date(2026, 1, 1))
    assert exc.value.current["explanation"] == "B newer"
    assert p.read_bytes() == before
    assert all("rev" not in r for r in json.loads(before)["crystal"])  # computed, never stored


def test_rev_survives_a_lone_surrogate_on_disk(tmp_path):
    """JSON can escape a lone surrogate, so a store can hold one. The rev digest
    once encoded it as UTF-8, which raised a bare UnicodeEncodeError on every read."""
    p = tmp_path / "mem.crystal.json"
    CrystalStore(p).crystallize(name="x", level=2, explanation="v1",
                                today=date(2026, 1, 1))
    doc = json.loads(p.read_text())
    doc["crystal"][0]["explanation"] = "a\ud800b"
    p.write_text(json.dumps(doc))  # default ensure_ascii writes the escape
    assert CrystalStore(p).get("x")["rev"]


def test_gen3_crystals_live_in_the_db_and_the_json_is_a_projection(tmp_path, monkeypatch):
    """Design B2 T3. Beside a generation-3 store, a crystal mutation is in the
    store's ``crystal_document`` row, so deleting the JSON loses nothing: reads come
    from the row and NEVER write (review r1 C1: no file, lock file or ``.diverged``
    copy, even in a copied snapshot directory), and the next write-capable Store
    open re-projects it byte-identical. A foreign edit to the JSON is kept as
    ``.diverged-`` (at that open and on a write, never overwriting an earlier copy)
    and never becomes the crystal state. A ``.db`` that cannot be probed raises,
    never falling to the file (r1 C2, C3)."""
    import hashlib
    import sqlite3

    from anneal_memory import Store
    from anneal_memory.store import connect

    def files():  # SQLite's own -wal/-shm aside: a mode=ro open of a WAL db may create them
        return {f.name: f.read_bytes() for f in tmp_path.iterdir()
                if not f.name.endswith(("-wal", "-shm"))}

    db = tmp_path / "memory.db"
    Store(db, project_name="t3").close()
    path = tmp_path / "memory.crystal.json"
    CrystalStore(path).crystallize(name="kept_in_db", level=2,
                                   explanation="lives in the row", today=T0)
    projected = path.read_bytes()
    conn = connect(db, read_only=True)
    try:
        doc_json, content_hash = conn.execute(
            "SELECT doc_json, content_hash FROM crystal_document WHERE id = 1").fetchone()
    finally:
        conn.close()
    assert doc_json.encode("utf-8") == projected
    assert hashlib.sha256(projected).hexdigest() == content_hash

    path.unlink()  # a copy of the .db alone is the whole crystal state
    # no fcntl on Windows, so no lock file there (CrystalStore._transaction)
    (tmp_path / "memory.crystal.json.lock").unlink(missing_ok=sys.platform == "win32")
    before = files()
    assert CrystalStore(path).get("kept_in_db")["explanation"] == "lives in the row"
    assert files() == before  # the read wrote nothing
    Store(db, project_name="t3").close()  # a write-capable open re-projects
    assert path.read_bytes() == projected

    foreign = json.loads(projected)
    foreign["crystal"][0]["explanation"] = "a hand edit"
    foreign_bytes = json.dumps(foreign).encode("utf-8")
    path.write_bytes(foreign_bytes)
    before = files()
    assert CrystalStore(path).get("kept_in_db")["explanation"] == "lives in the row"
    assert files() == before  # nor on a foreign file: no rewrite, no .diverged copy
    with pytest.warns(UserWarning, match="diverged"):
        Store(db, project_name="t3").close()
    assert path.read_bytes() == projected
    first = list(tmp_path.glob("memory.crystal.json.diverged-*"))
    assert [c.read_bytes() for c in first] == [foreign_bytes]

    path.write_bytes(foreign_bytes + b" ")  # a second edit, then a write over it
    with pytest.warns(UserWarning, match="diverged"):
        CrystalStore(path).crystallize(name="second", level=2,
                                       explanation="after the edit", today=T0)
    copies = sorted(tmp_path.glob("memory.crystal.json.diverged-*"))
    assert len(copies) == 2 and first[0].read_bytes() == foreign_bytes
    assert foreign_bytes + b" " in [c.read_bytes() for c in copies]
    state = CrystalStore(path)
    assert state.get("kept_in_db")["explanation"] == "lives in the row"
    assert sorted(c["name"] for c in state.active()) == ["kept_in_db", "second"]
    conn = connect(db, read_only=True)
    try:
        (doc_json,) = conn.execute("SELECT doc_json FROM crystal_document").fetchone()
    finally:
        conn.close()
    assert path.read_bytes() == doc_json.encode("utf-8")

    # Integration L3 I1: the store's one lease fence covers crystal changes: under
    # a live lease a change without its token is refused and writes nothing.
    from anneal_memory import LeaseFencedError

    leased = Store(db, project_name="t3")
    grant = leased.lease_acquire("A", ttl_seconds=60)
    with pytest.raises(LeaseFencedError) as fenced:
        CrystalStore(path).crystallize(name="stranger", level=2, explanation="no token",
                                       today=T0)
    assert fenced.value.reason == "lease_held" and CrystalStore(path).get("stranger") is None
    with pytest.raises(LeaseFencedError):
        CrystalStore(path).retire("second", kind="obsolete", lease_token="0" * 32)
    prior = CrystalStore(path).snapshot(["leased"])
    CrystalStore(path).crystallize(name="leased", level=2, explanation="the holder's",
                                   today=T0, lease_token=grant.token)
    with pytest.raises(LeaseFencedError):  # the rollback path is fenced too
        CrystalStore(path).restore(prior)
    CrystalStore(path).restore(prior, lease_token=grant.token)
    assert path.read_bytes() == doc_json.encode("utf-8")
    assert leased.lease_release(grant.token)
    leased.close()

    # L3 r1 D1: a file-mode writer (it probed generation 2) holds the crystal lock;
    # the migrating open's seed waits for it, so the writer's change is seeded, not
    # filed as .diverged at the next open.
    import threading

    mig = tmp_path / "mig"
    mig.mkdir()
    mig_db, mig_path = mig / "memory.db", mig / "memory.crystal.json"
    Store(mig_db, project_name="t3").close()
    conn = connect(mig_db)
    conn.execute("DROP TABLE crystal_document")
    conn.execute("UPDATE metadata SET value = '2' WHERE key = 'format_version'")
    conn.commit()
    conn.close()
    CrystalStore(mig_path).crystallize(name="legacy", level=2, explanation="gen 2", today=T0)
    if sys.platform == "win32":  # no fcntl: no lock for the seed to wait behind
        Store(mig_db, project_name="t3").close()
        assert [c["name"] for c in CrystalStore(mig_path).active()] == ["legacy"]
    else:
        writer = CrystalStore(mig_path)._transaction()
        data = writer.__enter__()
        opened = threading.Event()
        opener = threading.Thread(
            target=lambda: (Store(mig_db, project_name="t3").close(), opened.set()))
        opener.start()
        assert not opened.wait(1.0)  # the seed waits behind the writer's lock
        data["crystal"].append({**data["crystal"][0], "name": "mid_migration"})
        writer.__exit__(None, None, None)
        opener.join(30)
        assert opened.is_set()
        Store(mig_db, project_name="t3").close()
        assert sorted(c["name"] for c in CrystalStore(mig_path).active()) == [
            "legacy", "mid_migration"]
    assert not list(mig.glob("*.diverged-*"))
    # L3 r1 D2: once the table exists the store holds the document, whatever the
    # marker says; a marker >= 3 without the table raises.
    conn = connect(mig_db)
    conn.execute("UPDATE metadata SET value = 'garbled' WHERE key = 'format_version'")
    conn.commit()
    conn.close()
    CrystalStore(mig_path).crystallize(name="garbled", level=2, explanation="x", today=T0)
    conn = connect(mig_db, read_only=True)
    try:
        (mig_json,) = conn.execute("SELECT doc_json FROM crystal_document").fetchone()
    finally:
        conn.close()
    assert "garbled" in [c["name"] for c in json.loads(mig_json)["crystal"]]
    conn = connect(mig_db)
    conn.execute("DROP TABLE crystal_document")
    conn.execute("UPDATE metadata SET value = '3' WHERE key = 'format_version'")
    conn.commit()
    conn.close()
    with pytest.raises(CrystalError, match="no crystal_document table"):
        CrystalStore(mig_path).crystallize(name="lost", level=2, explanation="x", today=T0)

    # r1 C2: one probe error on a write raises; neither the file nor the row moves.
    real_connect = _crystal.connect

    def probe_fails(message):
        def fake(target, **kw):
            if kw.get("read_only"):
                raise sqlite3.OperationalError(message)
            return real_connect(target, **kw)
        return fake

    monkeypatch.setattr(_crystal, "connect", probe_fails("database is locked"))
    with pytest.raises(CrystalError, match="could not be probed"):
        CrystalStore(path).crystallize(name="third", level=2, explanation="x", today=T0)
    monkeypatch.setattr(_crystal, "connect", real_connect)
    assert path.read_bytes() == doc_json.encode("utf-8")
    assert sorted(c["name"] for c in CrystalStore(path).active()) == ["kept_in_db", "second"]
    # r1 C3: an unprobeable store with the file absent raises, never [].
    path.unlink()
    monkeypatch.setattr(_crystal, "connect",
                        probe_fails("attempt to write a readonly database"))
    with pytest.raises(CrystalError, match="could not be probed"):
        CrystalStore(path).active()


class TestSnapshotRestore:
    """snapshot()/restore() undo a crystallize for the named patterns only (the flow caller)."""

    def test_undoes_a_new_name_and_leaves_another_writers_row(self, store):
        store.crystallize(name="kept", level=2, explanation="already here", today=T0)
        snap = store.snapshot(["fresh"])
        store.crystallize(name="fresh", level=2, explanation="added by the step that fails", today=T0)
        # another writer crystallizes between the snapshot and the restore
        store.crystallize(name="other_writer", level=2, explanation="not ours", today=T0)
        store.restore(snap)
        assert store.get("fresh") is None
        assert store.get("kept") is not None and store.get("other_writer") is not None

    def test_restores_an_upserted_and_a_re_warmed_retired_pattern_in_place(self, store):
        store.crystallize(name="a", level=2, explanation="first a", today=T0)
        store.crystallize(name="b", level=2, explanation="first b", today=T0)
        store.crystallize(name="gone", level=2, explanation="was retired", today=T0)
        store.retire("gone", kind="obsolete", today=T0)
        before = json.loads(store.path.read_text())
        snap = store.snapshot(["a", "gone"])
        assert isinstance(snap, CrystalSnapshot) and snap.names == ("a", "gone")
        store.crystallize(name="a", level=3, explanation="sharpened a", today=T0)
        store.crystallize(name="gone", level=2, explanation="re-warmed", today=T0)
        store.restore(snap)
        after = json.loads(store.path.read_text())
        assert after["crystal"] == before["crystal"] and after["retired"] == before["retired"]
        with pytest.raises(TypeError):
            store.snapshot("a")

    def test_order_survives_a_reversed_name_list_and_a_foreign_store_is_refused(self, store, tmp_path):
        for n in ("d", "c", "b", "a"):
            store.crystallize(name=n, level=2, explanation=f"pattern {n}", today=T0)
            store.retire(n, kind="obsolete", today=T0)
        before = json.loads(store.path.read_text())["retired"]
        snap = store.snapshot([" a ", "b", "c", "d"])  # stripped, as crystallize() strips it
        assert snap.names == ("a", "b", "c", "d")
        for n in ("a", "b", "c", "d"):
            store.crystallize(name=n, level=2, explanation=f"re-warmed {n}", today=T0)
        store.restore(snap)
        assert json.loads(store.path.read_text())["retired"] == before
        with pytest.raises(ValueError):
            CrystalStore(tmp_path / "other.crystal.json").restore(snap)
        # L3 r1 D5: one file path read through two stores (db=) is two stores.
        from anneal_memory import Store

        for stem in ("a", "b"):
            Store(tmp_path / f"{stem}.db", project_name=stem).close()
        shared = tmp_path / "shared.crystal.json"
        in_a = CrystalStore(shared, db=tmp_path / "a.db")
        in_b = CrystalStore(shared, db=tmp_path / "b.db")
        in_a.crystallize(name="x", level=2, explanation="in A", today=T0)
        in_b.crystallize(name="x", level=2, explanation="in B", today=T0)
        with pytest.raises(ValueError):
            in_b.restore(in_a.snapshot(["x"]))
        assert in_b.get("x")["explanation"] == "in B"
