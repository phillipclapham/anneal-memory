"""SporeStore.apply (one effect, typed outcome, in-transaction postcondition),
SporeStore.delete and the deleted registry, and the full flush in _save.
Design: project_memory/k2b_spore_apply_design_1010.md (the consumer is levain's
K2b broker, K2B_DESIGN.md §4.1)."""

from __future__ import annotations

import json
import multiprocessing as mp
from datetime import date
from unittest import mock

import pytest

from anneal_memory import (
    ApplyRefused,
    PostconditionFailed,
    SporeApply,
    SporeError,
    SporeStore,
    spore_version,
)
from anneal_memory import spores as _spores

pytestmark = pytest.mark.skipif(_spores.fcntl is None, reason="apply needs a file lock")
DAY = date(2026, 10, 10)


@pytest.fixture
def store(tmp_path):
    return SporeStore(tmp_path / "spores.json")


def _raw(store: SporeStore) -> str:
    return store.path.read_text(encoding="utf-8") if store.path.exists() else ""


def _seed(store: SporeStore, key: str = "pend.1:seed", text: str = "plant me"):
    return store.apply(SporeApply(
        op="add", origin_key=key,
        args={"type": "task", "text": text, "today": DAY},
        postcondition={"type": "task", "text": text}))


# --- add ---------------------------------------------------------------------

def test_add_applies_once_then_reports_already(store):
    first = _seed(store)
    assert first.outcome == "applied"
    assert first.spore["origin_key"] == "pend.1:seed"
    assert first.version == spore_version(first.spore)
    again = _seed(store)
    assert again.outcome == "already"
    assert again.spore_id == first.spore_id
    assert len(store.list_open()) == 1


def test_add_whose_stored_value_differs_from_the_leaf_saves_nothing(store):
    # The created text is not the signed one: refused, not half-made.
    with pytest.raises(PostconditionFailed):
        store.apply(SporeApply(
            op="add", origin_key="pend.2:seed",
            args={"type": "task", "text": "trailing ", "today": DAY},
            postcondition={"type": "task", "text": "trailing!"}))
    assert _raw(store) == ""


# --- update ------------------------------------------------------------------

KEY = "pend.1:seed"


def test_update_applied_already_and_precondition_lost(store):
    seeded = _seed(store)
    v0 = seeded.version
    eff = SporeApply(op="update", origin_key=KEY, args={"text": "edited"},
                     expected_version=v0, postcondition={"text": "edited"})
    applied = store.apply(eff)
    assert applied.outcome == "applied" and applied.spore["text"] == "edited"
    # A retry after the save: the postcondition holds, nothing is written.
    before = _raw(store)
    assert store.apply(eff).outcome == "already"
    assert _raw(store) == before
    # A different write on the stale version: lost, with the version it moved to.
    lost = store.apply(SporeApply(op="update", origin_key=KEY, args={"text": "other"},
                                  expected_version=v0, postcondition={"text": "other"}))
    assert lost.outcome == "precondition_lost"
    assert lost.version == applied.version
    assert store.get(seeded.spore_id)["text"] == "edited"


def test_a_leaf_written_like_the_args_matches_the_stored_form(store):
    seeded = _seed(store)
    done = store.apply(SporeApply(
        op="update", origin_key=KEY, args={"text": "a\r\nb  ", "domain": None, "pointer": ""},
        expected_version=seeded.version,
        postcondition={"text": "a\r\nb  ", "domain": None, "pointer": ""}))
    assert done.outcome == "applied"
    assert (done.spore["text"], done.spore["domain"], done.spore["pointer"]) == ("a\nb", "", None)


def test_a_cleared_disposition_is_an_absent_key(store):
    made = store.apply(SporeApply(
        op="add", origin_key="pend.3:seed",
        args={"type": "task", "text": "routed", "disposition": "seed", "today": DAY},
        postcondition={"type": "task", "text": "routed", "disposition": "seed"}))
    for cleared in (None, ""):
        r = store.apply(SporeApply(
            op="update", origin_key="pend.3:seed", args={"disposition": cleared},
            expected_version=made.version, postcondition={"disposition": cleared}))
        assert r.outcome in ("applied", "already")
        assert "disposition" not in r.spore


def test_leaf_equality_is_canonical_json_so_true_is_not_one(store):
    seeded = _seed(store)
    with pytest.raises(PostconditionFailed, match="salience"):
        store.apply(SporeApply(op="update", origin_key=KEY, args={"salience": 1},
                               expected_version=seeded.version,
                               postcondition={"salience": True}))
    assert store.get(seeded.spore_id)["salience"] == 0


def test_a_leaf_that_disagrees_with_the_args_saves_nothing(store):
    # The signed value is the leaf; an adapter that passes a different value to
    # the mutator is refused inside the transaction (K2B §4.3's injected write).
    seeded = _seed(store)
    before = _raw(store)
    with pytest.raises(PostconditionFailed, match="text"):
        store.apply(SporeApply(op="update", origin_key=KEY, args={"text": "divergent"},
                               expected_version=seeded.version,
                               postcondition={"text": "signed"}))
    assert _raw(store) == before


def test_update_of_a_resolved_spore_is_a_lost_precondition(store):
    seeded = _seed(store)
    store.descend(seeded.spore_id, kind="done")
    resolved = store.get(seeded.spore_id)
    lost = store.apply(SporeApply(op="update", origin_key=KEY, args={"text": "x"},
                                  expected_version=spore_version(resolved),
                                  postcondition={"text": "x"}))
    assert lost.outcome == "precondition_lost"


def test_an_unknown_key_is_a_lost_precondition(store):
    _seed(store)
    lost = store.apply(SporeApply(op="update", origin_key="pend.404:seed", args={"text": "x"},
                                  expected_version="0" * 64, postcondition={"text": "x"}))
    assert (lost.outcome, lost.spore) == ("precondition_lost", None)


def test_a_spore_id_that_is_not_the_keys_spore_is_refused(store):
    seeded = _seed(store)
    store.add(type="task", text="another", today=DAY)
    with pytest.raises(ApplyRefused, match="spore_id"):
        store.apply(SporeApply(op="update", origin_key=KEY, spore_id="spore-002",
                               args={"text": "x"}, expected_version=seeded.version,
                               postcondition={"text": "x"}))


# --- ascend / descend ----------------------------------------------------------

def _ascend(version, kind="project", ref="levain@abc", **post):
    leaves = {"status": "resolved", ("resolution", "kind"): kind,
              ("resolution", "ref"): ref, **post}
    return SporeApply(op="ascend", origin_key=KEY, args={"kind": kind, "ref": ref, "today": DAY},
                      expected_version=version, postcondition=leaves)


def test_ascend_applies_then_retries_as_already(store):
    seeded = _seed(store)
    done = store.apply(_ascend(seeded.version))
    assert done.outcome == "applied"
    assert done.spore["resolution"]["direction"] == "ascend"
    assert store.apply(_ascend(seeded.version)).outcome == "already"


def test_a_spore_descended_is_not_already_ascended(store):
    seeded = _seed(store)
    store.apply(SporeApply(op="descend", origin_key=KEY, args={"kind": "done"},
                           expected_version=seeded.version,
                           postcondition={"status": "resolved", ("resolution", "kind"): "done"}))
    # Same kind name, other direction: the implied direction leaf fails, so it is lost.
    lost = store.apply(SporeApply(op="descend", origin_key=KEY, args={"kind": "done"},
                                  expected_version=seeded.version,
                                  postcondition={"status": "resolved",
                                                 ("resolution", "kind"): "dropped"}))
    assert lost.outcome == "precondition_lost"


def test_an_invalid_kind_is_refused_and_saves_nothing(store):
    seeded = _seed(store)
    before = _raw(store)
    with pytest.raises(ApplyRefused, match="kind"):
        store.apply(_ascend(seeded.version, kind="shipped"))
    assert _raw(store) == before


# --- delete and the registry -----------------------------------------------------

def test_apply_delete_then_already_and_the_id_and_key_are_never_reused(store):
    seeded = _seed(store)
    sid = seeded.spore_id
    eff = SporeApply(op="delete", origin_key=KEY, expected_version=seeded.version)
    assert store.apply(eff).outcome == "applied"
    assert store.get(sid) is None and store.get_by_origin_key(KEY) is None
    assert store.apply(eff).outcome == "already"
    row = json.loads(_raw(store))["deleted"][0]
    assert set(row) == {"id", "origin_key", "on", "at"} and row["id"] == sid
    nxt = store.add(type="task", text="next", today=DAY)
    assert nxt["id"] != sid
    # The create's retry after its undo: it happened, so already, with no spore.
    again = _seed(store)
    assert (again.outcome, again.spore) == ("already", None)
    with pytest.raises(SporeError, match="never reused"):
        store.add(type="task", text="plant me", origin_key=KEY, today=DAY)
    assert store.apply(SporeApply(op="update", origin_key=KEY, args={"text": "x"},
                                  expected_version=seeded.version,
                                  postcondition={"text": "x"})).outcome == "precondition_lost"


def test_an_id_reused_by_an_old_writer_is_never_taken_for_the_deleted_spore(store):
    # Codex L3 r1 critical: 0.9.42 does not count the registry, so it can reuse
    # the deleted id. Simulated by writing the reused row by hand.
    seeded = _seed(store)
    store.apply(SporeApply(op="delete", origin_key=KEY, expected_version=seeded.version))
    doc = json.loads(_raw(store))
    reused = dict(seeded.spore, origin_key="0" * 32)
    doc["spores"].append(reused)
    store.path.write_text(json.dumps(doc), encoding="utf-8")
    v = spore_version(reused)
    assert v == seeded.version  # the version does not see origin_key
    # The retried delete addresses the deleted key: already, the new row untouched.
    assert store.apply(SporeApply(op="delete", origin_key=KEY, expected_version=v)).outcome == "already"
    assert store.get(seeded.spore_id) is not None
    # The public delete by id meets the live row first.
    with pytest.raises(SporeError, match="origin_key"):
        store.delete(seeded.spore_id, expected_version=v, origin_key=KEY)
    assert store.delete(seeded.spore_id, expected_version=v) is True


def test_public_delete(store):
    seeded = _seed(store)
    with pytest.raises(SporeError, match="changed since read"):
        store.delete(seeded.spore_id, expected_version="0" * 64)
    assert store.delete(seeded.spore_id, expected_version=seeded.version) is True
    assert store.delete(seeded.spore_id, expected_version=seeded.version) is False
    with pytest.raises(SporeError, match="not found"):
        store.delete("spore-999", expected_version="0" * 64)


def test_a_resolved_spore_can_be_deleted(store):
    seeded = _seed(store)
    store.descend(seeded.spore_id, kind="done")
    v = spore_version(store.get(seeded.spore_id))
    assert store.delete(seeded.spore_id, expected_version=v) is True
    assert json.loads(_raw(store))["resolved"] == []


# --- malformed effects are refused before the lock ------------------------------------

@pytest.mark.parametrize("eff", [
    # vacuous: a leaf on a field the op does not write
    SporeApply(op="update", origin_key=KEY, args={"text": "x"}, expected_version="v",
               postcondition={"tier": "warm"}),
    SporeApply(op="update", origin_key=KEY, args={"text": "x"}, expected_version="v"),
    SporeApply(op="descend", origin_key=KEY, args={"kind": "done"}, expected_version="v",
               postcondition={"domain": "x"}),
    SporeApply(op="delete", origin_key=KEY, expected_version="v", postcondition={"text": "x"}),
    SporeApply(op="update", origin_key=KEY, args={"add_note": "n"},
               expected_version="v", postcondition={"add_note": "n"}),
    SporeApply(op="update", origin_key=KEY, args={"text": "x"}, postcondition={"text": "x"}),
    SporeApply(op="add", origin_key=KEY, args={"type": "task", "text": "x"},
               expected_version="v", postcondition={"type": "task", "text": "x"}),
    SporeApply(op="ascend", origin_key=KEY, args={"kind": "project"},
               expected_version="v", postcondition={"status": "resolved"}),
    SporeApply(op="update", origin_key=" padded", args={"text": "x"},
               expected_version="v", postcondition={"text": "x"}),
    SporeApply(op="update", origin_key=KEY, args={"text": "x"},
               expected_version="v", postcondition={(): "x"}),
])
def test_malformed_effects_are_refused_and_write_nothing(store, eff):
    with pytest.raises(ApplyRefused):
        store.apply(eff)
    assert _raw(store) == ""


def test_a_refusal_is_not_a_store_error_and_a_store_error_is_not_a_refusal(store):
    assert not issubclass(ApplyRefused, SporeError)
    assert not issubclass(PostconditionFailed, SporeError)
    store.path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SporeError) as caught:
        _seed(store)
    assert not isinstance(caught.value, (ApplyRefused, PostconditionFailed))


# --- two processes settling one unit ------------------------------------------------

def _settle(path_str: str, barrier, out) -> None:
    store = SporeStore(path_str)
    barrier.wait(timeout=30)
    out.put(store.apply(SporeApply(
        op="add", origin_key="pend.9:seed", args={"type": "task", "text": "once"},
        postcondition={"type": "task", "text": "once"})).outcome)


def test_two_processes_settling_one_create_land_it_once(tmp_path):
    ctx = mp.get_context("spawn")
    barrier, out = ctx.Barrier(4), ctx.Queue()
    path = str(tmp_path / "spores.json")
    procs = [ctx.Process(target=_settle, args=(path, barrier, out)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    outcomes = sorted(out.get(timeout=5) for _ in procs)
    assert outcomes == ["already"] * 3 + ["applied"]
    assert len(SporeStore(path).list_open()) == 1


# --- the full flush ----------------------------------------------------------------

@pytest.mark.skipif(not hasattr(_spores.fcntl, "F_FULLFSYNC"), reason="macOS only")
def test_save_full_flushes_the_file_and_the_directory(store):
    real = _spores.fcntl.fcntl
    calls: list[int] = []

    def spy(fd, cmd, *a):
        if cmd == _spores.fcntl.F_FULLFSYNC:
            calls.append(fd)
        return real(fd, cmd, *a)

    with mock.patch.object(_spores.fcntl, "fcntl", spy):
        store.add(type="task", text="durable", today=DAY)
    assert len(calls) == 2


@pytest.mark.skipif(not hasattr(_spores.fcntl, "F_FULLFSYNC"), reason="macOS only")
def test_the_episodic_store_full_flushes_on_macos(tmp_path):
    from anneal_memory import Store
    s = Store(tmp_path / "m.db")
    try:
        assert s._conn.execute("PRAGMA fullfsync").fetchone()[0] == 1
        assert s._conn.execute("PRAGMA checkpoint_fullfsync").fetchone()[0] == 1
    finally:
        s.close()


def test_a_mutator_that_writes_a_field_the_effect_does_not_name_saves_nothing(store):
    # K2B §4.3: a write made divergent inside apply must be refused by the
    # in-transaction check, whatever the mutator did.
    seeded = _seed(store)
    before = _raw(store)
    real = SporeStore._update_fields

    def also_tier(item, **kw):
        real(item, **kw)
        item["tier"] = "hot"

    with mock.patch.object(SporeStore, "_update_fields", staticmethod(also_tier)), \
            pytest.raises(PostconditionFailed, match="tier"):
        store.apply(SporeApply(op="update", origin_key=KEY, args={"text": "edited"},
                               expected_version=seeded.version, postcondition={"text": "edited"}))
    assert _raw(store) == before


# --- code L3 r1 (complement + codex) ----------------------------------------------

def test_a_delete_never_removes_another_spore_sharing_its_id(store):
    # codex HIGH: under id drift, removal by id took both rows.
    seeded = _seed(store)
    doc = json.loads(_raw(store))
    doc["resolved"].append(dict(seeded.spore, origin_key="1" * 32, status="resolved",
                                resolution={"direction": "descend", "kind": "done", "ref": None,
                                            "on": "2026-10-10", "at": "2026-10-10T00:00:00+00:00"}))
    store.path.write_text(json.dumps(doc), encoding="utf-8")
    before = _raw(store)
    with pytest.raises(SporeError, match="drift"):
        store.apply(SporeApply(op="delete", origin_key=KEY, expected_version=seeded.version))
    assert _raw(store) == before


def test_a_disposition_race_is_a_lost_precondition(store):
    made = store.apply(SporeApply(
        op="add", origin_key="pend.5:seed",
        args={"type": "task", "text": "t", "disposition": "seed"},
        postcondition={"type": "task", "text": "t", "disposition": "seed"}))
    lost = store.apply(SporeApply(
        op="update", origin_key="pend.5:seed", args={"text": "u", "expect_disposition": "agenda"},
        expected_version=made.version, postcondition={"text": "u"}))
    assert lost.outcome == "precondition_lost"
    ok = store.apply(SporeApply(
        op="update", origin_key="pend.5:seed", args={"text": "u", "expect_disposition": "seed"},
        expected_version=made.version, postcondition={"text": "u"}))
    assert ok.outcome == "applied"


def test_a_cleared_next_matches_its_stored_form(store):
    seeded = _seed(store)
    v = store.apply(SporeApply(op="update", origin_key=KEY, args={"next": "2026-11-01"},
                               expected_version=seeded.version,
                               postcondition={"next": "2026-11-01"})).version
    done = store.apply(SporeApply(op="update", origin_key=KEY, args={"next": ""},
                                  expected_version=v, postcondition={"next": ""}))
    assert done.outcome == "applied" and done.spore["next"] is None


def test_a_store_that_is_not_utf8_is_a_store_error(store):
    store.path.write_bytes(b'{"spores": ["\xff"]}')
    with pytest.raises(SporeError) as caught:
        _seed(store)
    assert not isinstance(caught.value, ApplyRefused)


def test_a_spore_id_is_checked_against_a_deleted_spore_too(store):
    seeded = _seed(store)
    store.apply(SporeApply(op="delete", origin_key=KEY, expected_version=seeded.version))
    with pytest.raises(ApplyRefused, match="spore_id"):
        store.apply(SporeApply(op="delete", origin_key=KEY, spore_id="spore-999",
                               expected_version=seeded.version))


@pytest.mark.parametrize("args", [{"kind": "done", "today": "2026-10-10"},
                                  {"kind": "done", "now": "2026-10-10T00:00:00Z"}])
def test_a_mistyped_clock_arg_is_refused(store, args):
    seeded = _seed(store)
    with pytest.raises(ApplyRefused):
        store.apply(SporeApply(op="descend", origin_key=KEY, args=args,
                               expected_version=seeded.version,
                               postcondition={"status": "resolved", ("resolution", "kind"): "done"}))
