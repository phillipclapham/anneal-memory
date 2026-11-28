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
from anneal_memory.rederive import (
    allow_store,
    check_state_for_save,
    revoke_store,
    root_identities,
    trusted_roots,
)
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


def _replace(dst: Path, src: Path) -> None:
    """Replace ``dst`` with a copy of ``src`` at the same path. The copy is made
    while ``dst`` still exists, so its inode cannot be ``dst``'s reused one
    (L3 r2: rmtree-then-copytree reuses it on ext4/xfs/tmpfs)."""
    tmp = dst.parent / (dst.name + ".new")
    shutil.copytree(src, tmp)
    shutil.rmtree(dst)
    os.rename(tmp, dst)


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




def test_an_unchanged_map_saves(three):
    store, _, _ = three
    res = prepare_wrap(store)
    allow_store(store.path, three[1], label="other", visibility="public")  # same root again
    _save(store, _continuity([_LINE]), res["wrap_token"])














def test_direct_callers_of_the_gate_are_unchanged(three):
    store, _, c = three
    allow_store(store.path, c, label="other", visibility="public")
    # no frozen_roots argument: no comparison, as before spore-1282
    check_state_for_save(_continuity([_LINE]), PROJECT_SCHEMA, store.path)








def test_an_unrelated_label_bound_during_compose_does_not_refuse(three):
    # L1 W2: no line runs in the new root, so nothing is at risk.
    store, _, c = three
    res = prepare_wrap(store)
    allow_store(store.path, c, label="unrelated", visibility="public")
    _save(store, _continuity([_LINE]), res["wrap_token"])










def test_a_leftover_map_on_an_idle_store_is_not_a_partial_wrap(three):
    # complement L3 r1
    store, _, _ = three
    store._conn.execute("UPDATE metadata SET value='{}' WHERE key='wrap_derive_roots'")
    store._conn.commit()
    receipt = store.wrap_cancelled()
    assert not receipt.partial_state
    assert store._get_metadata("wrap_derive_roots") == ""


