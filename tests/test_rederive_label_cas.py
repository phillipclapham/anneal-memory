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
    assert store.wrap_derive_roots() == root_identities(trusted_roots(store.path))
    allow_store(store.path, c, label="other", visibility="public")  # the operator rebinds
    with pytest.raises(ValueError, match=r"roots changed after prepare_wrap \(label 'other'") as e:
        _save(store, _continuity([_LINE]), res["wrap_token"])
    # the recovery names this wrap's token, never a tokenless cancel (L2)
    assert f"--wrap-token {res['wrap_token']}" in str(e.value)
    # nothing written, and the wrap is still the one prepare opened
    assert store.load_continuity() is None
    assert store.load_wrap_snapshot()["token"] == res["wrap_token"]
    # the documented recovery: cancel, prepare against the new map, save
    store.wrap_cancelled()
    res2 = prepare_wrap(store)
    assert store.wrap_derive_roots()["other"].endswith(":" + str(c.resolve()))
    _save(store, _continuity([_LINE]), res2["wrap_token"])
    assert "[derive@other:" in store.load_continuity()
    assert store._get_metadata("wrap_derive_roots") == ""  # the completed wrap cleared it (L1 W1)


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
        _save(store, _continuity([_LINE]), "t" * 32)
    # judged-only text runs nothing, so nothing is at risk
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
    tok = res["wrap_token"]
    for bad in (
        "{not json",
        json.dumps([["other", "/x"]]),  # the pre-token shape
        json.dumps({"token": tok, "roots": {"other": "/x"}}),
        json.dumps({"token": tok, "roots": [["other", "/x"], ["other", "/y"]]}),
    ):
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


def test_a_directory_replaced_at_the_same_path_refuses_the_save(three):
    # L2 2026-10-01, reproduced: the path alone passed while the repository at
    # it was a different one.
    store, b, c = three
    res = prepare_wrap(store)
    before = root_identities({"other": str(b.resolve())})
    shutil.rmtree(b)
    shutil.copytree(c, b)
    if root_identities({"other": str(b.resolve())}) == before:
        pytest.skip("this filesystem reused the inode and reports no creation time")
    with pytest.raises(ValueError, match="was replaced by a different directory at the same path"):
        _save(store, _continuity([_LINE]), res["wrap_token"])


def test_a_dot_git_replaced_inside_the_same_root_refuses_the_save(three):
    store, b, c = three
    res = prepare_wrap(store)
    before = root_identities({"other": str(b.resolve())})
    shutil.rmtree(b / ".git")
    shutil.copytree(c / ".git", b / ".git")
    if root_identities({"other": str(b.resolve())}) == before:
        pytest.skip("this filesystem reused the inode and reports no creation time")
    with pytest.raises(ValueError, match="was replaced"):
        _save(store, _continuity([_LINE]), res["wrap_token"])


def test_a_full_revoke_during_compose_saves_but_says_nothing_was_checked(three):
    store, _, _ = three
    res = prepare_wrap(store)
    revoke_store(store.path)
    with pytest.warns(UserWarning, match="no State line was checked at this save"):
        validated_save_continuity(store, _continuity([_LINE]), wrap_token=res["wrap_token"])


def test_an_unrelated_label_bound_during_compose_does_not_refuse(three):
    # L1 W2: no line runs in the new root, so nothing is at risk.
    store, _, c = three
    res = prepare_wrap(store)
    allow_store(store.path, c, label="unrelated", visibility="public")
    _save(store, _continuity([_LINE]), res["wrap_token"])


def test_a_map_frozen_under_another_token_is_not_this_wraps(three):
    # L1 W3: an older binary that does not clear the key leaves it behind; the
    # next wrap must not read it as its own. It reads as "froze none", which
    # refuses while a derive line would run.
    store, _, _ = three
    res = prepare_wrap(store)
    leftover = store._get_metadata("wrap_derive_roots")
    store.wrap_cancelled(expect_token=res["wrap_token"])
    store.wrap_started(token="u" * 32, episode_ids=[])  # the older binary's prepare
    store._conn.execute("UPDATE metadata SET value=? WHERE key='wrap_derive_roots'", (leftover,))
    store._conn.commit()
    assert store.wrap_derive_roots() is None
    with pytest.raises(ValueError, match="recorded no re-derive root map"):
        _save(store, _continuity([_LINE]), "u" * 32)


def test_a_legacy_wrap_and_a_full_revoke_still_say_nothing_was_checked(three):
    # codex L3 r1: a wrap that froze no map took the silent path.
    store, _, _ = three
    store.wrap_started(token="t" * 32, episode_ids=[])
    revoke_store(store.path)
    with pytest.warns(UserWarning, match="no State line was checked at this save"):
        validated_save_continuity(store, _continuity([_LINE]), wrap_token="t" * 32)


def test_a_root_replaced_while_its_commands_run_refuses(three, monkeypatch):
    # codex L3 r1: the identity is taken again after the commands ran.
    import anneal_memory.rederive as rd

    store, b, c = three
    res = prepare_wrap(store)
    real = rd.rederive_text

    def swap_then_run(text, schema, roots, **kw):
        report = real(text, schema, roots, **kw)
        shutil.rmtree(b)
        shutil.copytree(c, b)
        return report

    monkeypatch.setattr(rd, "rederive_text", swap_then_run)
    before = root_identities({"other": str(b.resolve())})
    try:
        _save(store, _continuity([_LINE]), res["wrap_token"])
    except ValueError as e:
        assert "changed while its commands ran" in str(e)
    else:
        assert root_identities({"other": str(b.resolve())}) == before, "replaced root was not caught"
        pytest.skip("this filesystem reused the inode and reports no creation time")


def test_the_map_is_read_against_the_snapshot_token(three):
    # complement L3 r1: a map frozen under another token is not this snapshot's.
    store, _, _ = three
    res = prepare_wrap(store)
    assert store.wrap_derive_roots(expect_token=res["wrap_token"]) is not None
    assert store.wrap_derive_roots(expect_token="x" * 32) is None


def test_a_leftover_map_on_an_idle_store_is_not_a_partial_wrap(three):
    # complement L3 r1
    store, _, _ = three
    store._conn.execute("UPDATE metadata SET value='{}' WHERE key='wrap_derive_roots'")
    store._conn.commit()
    receipt = store.wrap_cancelled()
    assert not receipt.partial_state
    assert store._get_metadata("wrap_derive_roots") == ""


def test_a_root_replaced_while_prepares_flags_run_refuses_the_save(three, monkeypatch):
    # codex L3 r1: prepare took identities after its flags ran, so a root
    # replaced in between was frozen as the replacement.
    import anneal_memory.continuity as co

    store, b, c = three
    res = prepare_wrap(store)
    _save(store, _continuity([_LINE]), res["wrap_token"])  # a continuity to flag
    store.record("e2", "observation")
    real = co.rederive_text
    before = root_identities({"other": str(b.resolve())})

    def flag_then_swap(text, schema, roots, **kw):
        report = real(text, schema, roots, **kw)
        shutil.rmtree(b)
        shutil.copytree(c, b)
        return report

    monkeypatch.setattr(co, "rederive_text", flag_then_swap)
    res2 = prepare_wrap(store)
    monkeypatch.setattr(co, "rederive_text", real)
    if root_identities({"other": str(b.resolve())}) == before:
        pytest.skip("this filesystem reused the inode and reports no creation time")
    with pytest.raises(ValueError, match="was replaced by a different directory"):
        _save(store, _continuity([_LINE]), res2["wrap_token"])
