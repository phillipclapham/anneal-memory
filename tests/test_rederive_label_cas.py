"""spore-1282: a compare-and-swap on the label -> root map between prepare_wrap
and the save.

The race was reproduced before the fix [run 2026-10-01, 1001+28]: bind label
``other`` to repo b, prepare_wrap, rebind ``other`` to repo c (which holds the
same tag), save. The save accepted, certifying in c a line composed against b.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import pytest

from anneal_memory import Store, prepare_wrap, validated_save_continuity
from anneal_memory.rederive import allow_store, check_state_for_save, revoke_store, trusted_roots
from anneal_memory.schema import PROJECT_SCHEMA
from anneal_memory.store import StoreError

pytestmark = [
    pytest.mark.skipif(shutil.which("git") is None, reason="needs git"),
    pytest.mark.skipif(os.name != "posix", reason="re-derive is POSIX-only by design"),
]

_GIT = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
_LINE = "- other tag [derive@other: git describe --tags --abbrev=0 @REF => v2.0.0]"


def _repo(path: Path, tag: str) -> Path:
    path.mkdir()
    run = lambda *a: subprocess.run([*_GIT, *a], cwd=path, check=True, capture_output=True)
    run("init", "-q")
    (path / "a.txt").write_text("1\n")
    run("add", ".")
    run("commit", "-qm", "i")
    run("tag", tag)
    return path


def _continuity(state: list[str]) -> str:
    return "\n".join([
        "# S — Memory (v1)", "",
        "## Plan", "- p", "",
        "## State", *state, "",
        "## Decisions", "- d", "",
        "## Open", "- o", "",
        "## Lessons", "- l", "",
        "## History", "- h", "",
    ])


def _save(store, text, token):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return validated_save_continuity(store, text, wrap_token=token)


@pytest.fixture
def three(tmp_path, monkeypatch):
    monkeypatch.setenv("ANNEAL_MEMORY_DERIVE_TRUST", str(tmp_path / "trust.json"))
    main = _repo(tmp_path / "main", "v1.0.0")
    b = _repo(tmp_path / "b", "v2.0.0")
    c = _repo(tmp_path / "c", "v2.0.0")
    store = Store(tmp_path / "store" / "m.db", project_name="S", section_schema=PROJECT_SCHEMA)
    allow_store(store.path, main, visibility="public")
    allow_store(store.path, b, label="other", visibility="public")
    store.record("e", "observation")
    yield store, b, c
    store.close()


def test_a_rebind_during_compose_refuses_the_save_and_the_wrap_recovers(three):
    store, b, c = three
    res = prepare_wrap(store)
    assert store.wrap_derive_roots() == trusted_roots(store.path)
    allow_store(store.path, c, label="other", visibility="public")  # the operator rebinds
    with pytest.raises(ValueError, match=r"roots changed after prepare_wrap \(label 'other'"):
        _save(store, _continuity([_LINE]), res["wrap_token"])
    # nothing written, and the wrap is still the one prepare opened
    assert store.load_continuity() is None
    assert store.load_wrap_snapshot()["token"] == res["wrap_token"]
    # the documented recovery: cancel, prepare against the new map, save
    store.wrap_cancelled()
    res2 = prepare_wrap(store)
    assert store.wrap_derive_roots()["other"] == str(c.resolve())
    _save(store, _continuity([_LINE]), res2["wrap_token"])
    assert "[derive@other:" in store.load_continuity()
    assert store.wrap_derive_roots() is None  # the completed wrap cleared it


def test_an_unchanged_map_saves(three):
    store, _, _ = three
    res = prepare_wrap(store)
    allow_store(store.path, three[1], label="other", visibility="public")  # same root again
    _save(store, _continuity([_LINE]), res["wrap_token"])


def test_a_label_bound_during_compose_refuses_the_save(three):
    store, _, c = three
    res = prepare_wrap(store)
    allow_store(store.path, c, label="late", visibility="public")
    with pytest.raises(ValueError, match=r"label 'late': unbound -> "):
        _save(store, _continuity(["- late [derive@late: git describe --tags --abbrev=0 @REF => v2.0.0]"]), res["wrap_token"])


def test_a_revoke_during_compose_is_left_to_the_existing_gates(three):
    # A removal certifies nothing, so the compare does not refuse it: a revoked
    # label reads UNBOUND and refuses on its own, and with every root revoked no
    # command runs (require_rederive is the caller's way to insist on a check;
    # tests/test_rederive_gaps.py::test_save_can_require_rederive).
    store, _, _ = three
    res = prepare_wrap(store)
    revoke_store(store.path, label="other")
    with pytest.raises(ValueError, match="UNBOUND"):
        _save(store, _continuity([_LINE]), res["wrap_token"])


def test_a_wrap_that_froze_no_map_refuses_only_when_a_root_is_bound(three):
    store, _, _ = three
    store.wrap_started(token="t" * 32, episode_ids=[])  # an older prepare, or a direct caller
    assert store.wrap_derive_roots() is None
    with pytest.raises(ValueError, match="recorded no re-derive root map"):
        _save(store, _continuity(["- j [judged: x]"]), "t" * 32)
    revoke_store(store.path)
    _save(store, _continuity(["- j [judged: x]"]), "t" * 32)


def test_cancel_clears_the_frozen_map(three):
    store, _, _ = three
    prepare_wrap(store)
    assert store._get_metadata("wrap_derive_roots")
    store.wrap_cancelled()
    assert store._get_metadata("wrap_derive_roots") == ""


def test_an_unreadable_frozen_map_fails_closed(three):
    store, _, _ = three
    res = prepare_wrap(store)
    for bad in ("{not json", json.dumps({"other": "/x"}), json.dumps([["other", "/x"], ["other", "/y"]])):
        store._conn.execute("UPDATE metadata SET value=? WHERE key='wrap_derive_roots'", (bad,))
        store._conn.commit()
        with pytest.raises(StoreError):
            store.wrap_derive_roots()
        with pytest.raises(StoreError):
            _save(store, _continuity([_LINE]), res["wrap_token"])


def test_a_store_without_derived_state_freezes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("ANNEAL_MEMORY_DERIVE_TRUST", str(tmp_path / "trust.json"))
    store = Store(tmp_path / "m.db", project_name="S")
    store.record("e", "observation")
    prepare_wrap(store)
    assert store._get_metadata("wrap_derive_roots") == ""
    assert store.wrap_derive_roots() is None
    store.close()


def test_direct_callers_of_the_gate_are_unchanged(three):
    store, _, c = three
    allow_store(store.path, c, label="other", visibility="public")
    # no frozen_roots argument: no comparison, as before spore-1282
    check_state_for_save(_continuity([_LINE]), PROJECT_SCHEMA, store.path)
