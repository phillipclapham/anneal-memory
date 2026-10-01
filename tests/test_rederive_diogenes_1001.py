"""Diogenes 2026-10-01: three LOWs, each reproduced at 0206056 before the fix.

1. The include guard was a line-start regex: "[core][include]" on one line,
   or an include in .git/config.worktree, let git read a config outside the
   root while the shape check passed.
2. strip_flag cut at the earliest flag-shaped run, so a claim holding one
   ("  ⚠ BLOCKED (…)") lost its annotation once a real flag was appended.
3. ``derive allow`` bound a linked worktree, where every git/grep/wc line errors.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from anneal_memory.rederive import (
    _check_repo_shape,
    allow_store,
    parse_annotation,
    strip_flag,
)

pytestmark = [
    pytest.mark.skipif(shutil.which("git") is None, reason="needs git"),
    pytest.mark.skipif(os.name != "posix", reason="re-derive is POSIX-only by design"),
]

_GIT = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    run = lambda *a: subprocess.run([*_GIT, *a], cwd=r, check=True, capture_output=True)
    run("init", "-q")
    (r / "a.txt").write_text("hi\n")
    run("add", "a.txt")
    run("commit", "-qm", "i")
    return r


def test_include_shapes_the_regex_missed_are_refused(repo, tmp_path):
    outside = tmp_path / "outside.cfg"
    outside.write_text("[core]\n\tabbrev = 20\n")
    assert _check_repo_shape(str(repo)) is None

    config = repo / ".git" / "config"
    plain = config.read_text()
    config.write_text(plain + f"[core][include]\n\tpath = {outside}\n")
    assert "includes another file" in _check_repo_shape(str(repo))

    config.write_text(plain + f'[includeIf "gitdir:/"]\n\tpath = {outside}\n')
    assert "includes another file" in _check_repo_shape(str(repo))

    config.write_text(plain + "[extensions]\n\tworktreeConfig = true\n")
    (repo / ".git" / "config.worktree").write_text(f"[include]\n\tpath = {outside}\n")
    assert "config.worktree" in _check_repo_shape(str(repo))


def test_a_flag_shaped_claim_survives_strip_flag():
    authored = "Release gate  ⚠ BLOCKED (see Open) [derive: test -e missing.txt]"
    assert parse_annotation(authored).command == "test -e missing.txt"
    for flag in ("  ⚠ STALE (exit 1)", "  ⚠ DERIVE ERROR (x  ✓ (y))", "  ✓"):
        assert strip_flag(authored + flag) == authored
    # an earlier annotation is never what remains
    two = "X [judged: a]  ⚠ BLOCKED (y) [derive: test -e m]"
    assert strip_flag(two + "  ⚠ STALE (exit 1)") == two
    # an unannotated line still loses its flag
    assert strip_flag("bare claim  ⚠ NO DERIVE") == "bare claim"


def test_allow_refuses_a_linked_worktree(repo, tmp_path):
    wt = tmp_path / "wt"
    subprocess.run([*_GIT, "worktree", "add", "-q", str(wt)], cwd=repo, check=True, capture_output=True)
    db = tmp_path / "m.db"
    db.write_text("")
    with pytest.raises(ValueError, match="not a plain directory"):
        allow_store(db, wt, trust_file=tmp_path / "trust.json")
    assert allow_store(db, repo, trust_file=tmp_path / "trust.json") == os.path.realpath(repo)
