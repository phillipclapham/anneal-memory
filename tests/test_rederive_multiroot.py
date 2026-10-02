"""Multi-root re-derive (spore-1272, ruled by Phill 2026-10-01: option A plus one
visibility class per store), and the two spore-1233 L3 LOWs closed with it.

Every case here was first run by hand on a copy of a real project store
(cross-repo claims, an unbound label, the save gate); these pin what that run
showed.
"""

from __future__ import annotations

import json
import time
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import pytest

from anneal_memory import Store, prepare_wrap, validated_save_continuity
from anneal_memory.rederive import (
    DeriveRefused,
    _check_repo_shape,
    allow_store,
    check_state_for_save,
    drop_header,
    parse_annotation,
    rederive_text,
    revoke_store,
    trusted_roots,
)
from anneal_memory.schema import PROJECT_SCHEMA

pytestmark = [
    pytest.mark.skipif(shutil.which("git") is None, reason="needs git"),
    pytest.mark.skipif(os.name != "posix", reason="re-derive is POSIX-only by design"),
]

_GIT = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]


def _repo(path: Path, tag: str) -> Path:
    path.mkdir()
    run = lambda *a: subprocess.run([*_GIT, *a], cwd=path, check=True, capture_output=True)
    run("init", "-q")
    (path / "a.txt").write_text("1\n2\n")
    (path / "b.txt").write_text("x\n")
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


@pytest.fixture
def two(tmp_path, monkeypatch):
    trust = tmp_path / "trust.json"
    monkeypatch.setenv("ANNEAL_MEMORY_DERIVE_TRUST", str(trust))
    main = _repo(tmp_path / "main", "v1.0.0")
    other = _repo(tmp_path / "other", "v2.0.0")
    store = Store(tmp_path / "store" / "m.db", project_name="S", section_schema=PROJECT_SCHEMA)
    yield store, main, other, trust
    store.close()


def test_parse_label():
    a = parse_annotation("- x [derive@other: git describe --tags --abbrev=0 @REF => v2.0.0]")
    assert (a.kind, a.root, a.command, a.expected) == ("derive", "other", "git describe --tags --abbrev=0 @REF", "v2.0.0")
    assert parse_annotation("- x [derive: test -e a.txt]").root is None
    # a second opener is refused, never resolved to the rightmost (spore-1300)
    with pytest.raises(DeriveRefused, match="2 annotation openers"):
        parse_annotation("- see [derive@a: x] then [derive@b: test -e a.txt]")
    assert parse_annotation("- x [derive@: test -e a.txt]").root == ""  # refused later, never dropped


def test_binding_rules(two):
    store, main, other, trust = two
    with pytest.raises(ValueError, match="default root first"):
        allow_store(store.path, other, label="other", visibility="public")
    allow_store(store.path, main)
    with pytest.raises(ValueError, match="declares no visibility"):
        allow_store(store.path, other, label="other", visibility="public")
    allow_store(store.path, main, visibility="public")
    with pytest.raises(ValueError, match="needs --visibility"):
        allow_store(store.path, other, label="other")
    with pytest.raises(ValueError, match="one visibility class"):
        allow_store(store.path, other, label="other", visibility="private")
    with pytest.raises(ValueError, match="must be lowercase"):
        allow_store(store.path, other, label="Other_X", visibility="public")
    with pytest.raises(ValueError, match="nested"):
        allow_store(store.path, main / ".." / "main", label="same", visibility="public")
    allow_store(store.path, other, label="other", visibility="public")
    assert trusted_roots(store.path) == {None: str(main.resolve()), "other": str(other.resolve())}
    # re-binding the default to a different visibility would split the class
    with pytest.raises(ValueError, match="needs --visibility public"):
        allow_store(store.path, main, visibility="private")
    # an old reader takes the store entry's own "root": it must stay the default
    entry = json.loads(trust.read_text())["stores"][0]
    assert entry["root"] == str(main.resolve()) and "other" in entry["labels"]
    assert revoke_store(store.path, label="other") is True
    assert trusted_roots(store.path) == {None: str(main.resolve())}
    assert revoke_store(store.path, label="other") is False


def test_read_rules_drop_what_allow_would_refuse(two):
    store, main, other, trust = two
    allow_store(store.path, main, visibility="public")
    allow_store(store.path, other, label="other", visibility="public")
    data = json.loads(trust.read_text())
    data["stores"][0]["labels"]["other"]["visibility"] = "private"  # a hand edit
    trust.write_text(json.dumps(data))
    os.chmod(trust, 0o600)
    assert "other" not in trusted_roots(store.path)
    data["stores"][0]["labels"]["other"].update(visibility="public", root=str(main / "sub"))
    (main / "sub").mkdir()
    trust.write_text(json.dumps(data))
    assert "other" not in trusted_roots(store.path)  # nested in the default root


def test_lines_run_in_their_root_and_unbound_refuses_the_save(two):
    store, main, other, _ = two
    allow_store(store.path, main, visibility="public")
    allow_store(store.path, other, label="other", visibility="public")
    state = [
        "- main tag [derive: git describe --tags --abbrev=0 @REF => v1.0.0]",
        "- other tag [derive@other: git describe --tags --abbrev=0 @REF => v2.0.0]",
        "- other stale [derive@other: git describe --tags --abbrev=0 @REF => v1.0.0]",
        "- nobody bound this [derive@site: test -f index.html]",
        "- bad label [derive@Bad_X: test -f a.txt]",
    ]
    r = rederive_text(_continuity(state), PROJECT_SCHEMA, trusted_roots(store.path))
    assert [x.status for x in r.results] == ["ok", "ok", "stale", "unbound", "refused"]
    assert set(r.refs) == {None, "other"} and all(r.refs.values())
    # the multi-root header is the line drop_header removes, and nothing else
    assert r.text.startswith("> [anneal re-derive]") and "; other in " in r.text.split("\n")[0]
    assert drop_header(r.text).startswith("# S — Memory (v1)")
    with pytest.raises(ValueError, match="refused"):
        check_state_for_save(_continuity(state), PROJECT_SCHEMA, store.path)
    with pytest.raises(ValueError, match="UNBOUND"):
        check_state_for_save(_continuity(state[:4]), PROJECT_SCHEMA, store.path)

    # a real wrap: the composer sees the flags, the save keeps labels, never flags
    store.record("e", "observation")
    res = prepare_wrap(store)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        validated_save_continuity(store, _continuity(state[:3]), wrap_token=res["wrap_token"])
    saved = Path(store.continuity_path).read_text()
    assert saved.count("[derive@other:") == 2 and "  ✓" not in saved and "[anneal re-derive]" not in saved
    store.record("e2", "observation")
    pkg = prepare_wrap(store)
    assert any("STALE" in u for u in pkg["package"].get("unconfirmed_state", []))
    store.wrap_cancelled()


def test_exit_1_after_partial_output_is_an_error(two):
    store, main, _, _ = two
    state = [
        # ls-files lists the tracked name, then exits 1 for the untracked one
        "- both tracked [derive: git ls-files --error-unmatch a.txt nope.txt => a.txt]",
        # grep's exit 1 is a complete answer: no match
        "- no match [derive: grep -c zzz a.txt => 0]",
    ]
    r = rederive_text(_continuity(state), PROJECT_SCHEMA, str(main))
    assert [x.status for x in r.results] == ["error", "ok"]
    (main / "b.txt").unlink()  # tracked, gone from the working tree: wc prints a.txt, exits 1
    r = rederive_text(
        _continuity(["- lines [derive: wc -l a.txt b.txt => 2 a.txt 2 total]"]), PROJECT_SCHEMA, str(main)
    )
    assert r.results[0].status == "error"


def test_the_git_walk_is_bounded(two):
    _, main, _, _ = two
    d = main / ".git" / "objects" / "zz"
    d.mkdir()
    for i in range(600):
        (d / f"f{i}").touch()
    assert _check_repo_shape(str(main)) is None
    assert "load budget" in _check_repo_shape(str(main), timeout=0)


def test_header_regex_is_linear_on_a_crafted_first_line():
    import time

    text = "> [anneal re-derive] 1 STATE line(s) in /x" + "; a in /y" * 22 + " X\n\nbody"
    start = time.monotonic()
    assert drop_header(text) == text  # not a header: kept
    assert time.monotonic() - start < 1.0  # was exponential: 20 segments took 0.58s


def test_per_root_checks_share_the_load_budget(two, tmp_path, monkeypatch):
    from anneal_memory import rederive

    store, main, _, _ = two
    allow_store(store.path, main, visibility="public")
    names = [f"r{i}" for i in range(4)]
    for n in names:
        allow_store(store.path, _repo(tmp_path / n, "v0"), label=n, visibility="public")
    calls = []
    real = rederive._check_repo_shape
    monkeypatch.setattr(rederive, "_check_repo_shape", lambda *a, **k: calls.append(a) or real(*a, **k))
    state = [f"- {n} [derive@{n}: git rev-parse --verify -q @REF]" for n in names]
    r = rederive_text(_continuity(state), PROJECT_SCHEMA, trusted_roots(store.path), budget=0)
    assert calls == []  # a spent budget checks no further root, so nothing runs in one
    assert all(x.status in ("skipped", "error") for x in r.results)


def test_read_side_resolves_a_symlinked_label_root(two, tmp_path):
    store, main, other, trust = two
    allow_store(store.path, main, visibility="public")
    allow_store(store.path, other, label="other", visibility="public")
    (main / "sub").mkdir()
    link = tmp_path / "link"
    link.symlink_to(main / "sub")
    data = json.loads(trust.read_text())
    data["stores"][0]["labels"]["other"]["root"] = str(link)  # a hand edit
    trust.write_text(json.dumps(data))
    assert "other" not in trusted_roots(store.path)  # it is nested in the default root


def test_l3_parsers_and_trust_reading_fail_closed(two, tmp_path):
    from anneal_memory.rederive import strip_flag

    # strip_flag is linear on many flag-shaped fragments (was quadratic)
    line = "- x [judged: ok]" + "]  ⚠ FOO (" * 160000 + ")"  # quadratic took 3.8s here
    start = time.monotonic()
    strip_flag(line)
    assert time.monotonic() - start < 1.0
    store, main, other, trust = two
    allow_store(store.path, main, visibility="public")
    allow_store(store.path, other, label="other", visibility="public")
    (other / "inner").mkdir()
    data = json.loads(trust.read_text())
    L = data["stores"][0]["labels"]
    # nesting is dropped on both sides whatever the order (was order-dependent)
    L["inner"] = {"root": str((other / "inner").resolve()), "visibility": "public"}
    trust.write_text(json.dumps(data))
    assert set(trusted_roots(store.path)) == {None}
    # a malformed entry is unusable, never a crash; allow still works over it
    L.pop("inner")
    L["bad"] = {"root": "/x\u0000y", "visibility": "public"}
    L["scalar"] = "nope"
    trust.write_text(json.dumps(data))
    assert set(trusted_roots(store.path)) == {None, "other"}
    allow_store(store.path, other, label="other", visibility="public")


def test_the_cap_counts_unbound_lines_so_no_unchecked_root_runs(two, monkeypatch):
    from anneal_memory import rederive

    store, main, other, _ = two
    allow_store(store.path, main, visibility="public")
    allow_store(store.path, other, label="other", visibility="public")
    checked = []
    real = rederive._check_repo_shape
    monkeypatch.setattr(rederive, "_check_repo_shape", lambda r, **k: checked.append(r) or real(r, **k))
    state = ["- u [derive@nobody: test -e a.txt]"] * 2 + ["- o [derive@other: git rev-parse --verify -q HEAD]"]
    r = rederive_text(_continuity(state), PROJECT_SCHEMA, trusted_roots(store.path), max_lines=2)
    assert [x.status for x in r.results] == ["unbound", "unbound", "skipped"]
    assert checked == []
