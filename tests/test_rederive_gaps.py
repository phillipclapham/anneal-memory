"""spore-1233: four re-derive gaps, each reproduced before its fix.

1. ``derive status`` without ``--json`` raised UnboundLocalError.
2. ``continuity --rederive`` exited 0 with zero flags when re-derive was NOT
   enabled, so "nothing was checked" read the same as "everything holds".
3. A save had no way to require that its State lines were actually checked:
   a trust revoked between a caller's check and the save went through silently.
4. ``prepare_wrap`` showed the composer the stored continuity without its
   re-derive flags, so the model could not see which State lines were stale.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from anneal_memory import prepare_wrap, validated_save_continuity
from anneal_memory.continuity import format_wrap_package_text
from anneal_memory.rederive import rederive_continuity, revoke_store

from .test_rederive import TRUE_STATE, _continuity, _save, project  # noqa: F401

pytestmark = pytest.mark.skipif(os.name != "posix", reason="re-derive is POSIX-only by design")

REPO_ROOT = Path(__file__).resolve().parent.parent


def _cli(store, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    return subprocess.run(
        [sys.executable, "-m", "anneal_memory.cli", "--db", str(store.path), *args],
        capture_output=True, text=True, env=env, timeout=60,
    )


def test_derive_status_prints_without_json(project):
    store, repo = project
    allowed = _cli(store, "derive", "status")
    assert allowed.returncode == 0, allowed.stderr
    assert f"allowed in {os.path.realpath(repo)}" in allowed.stdout

    revoke_store(store.path)
    off = _cli(store, "derive", "status")
    assert off.returncode == 0, off.stderr
    assert "Re-derive: not enabled" in off.stdout


def test_rederive_that_checked_nothing_is_not_a_clean_exit(project):
    store, _repo = project
    _save(store, _continuity(TRUE_STATE))

    clean = _cli(store, "continuity", "--rederive")
    assert clean.returncode == 0, clean.stderr

    revoke_store(store.path)
    off = _cli(store, "continuity", "--rederive")
    assert off.returncode == 3
    assert "not enabled" in off.stderr

    off_json = _cli(store, "continuity", "--rederive", "--json")
    assert off_json.returncode == 3
    assert json.loads(off_json.stdout)["rederive"]["enabled"] is False


def test_save_can_require_rederive(project, tmp_path):
    store, _repo = project
    _save(store, _continuity(TRUE_STATE))
    store.record("next session", "observation")
    prep = prepare_wrap(store)
    revoke_store(store.path)  # the trust goes between the caller's check and the save

    with pytest.raises(ValueError, match="re-derive is not enabled"):
        validated_save_continuity(
            store, _continuity(TRUE_STATE), wrap_token=prep["wrap_token"], require_rederive=True
        )
    assert store.load_wrap_snapshot() is not None  # refused before anything was written

    doc = tmp_path / "c.md"
    doc.write_text(_continuity(TRUE_STATE))
    cli = _cli(store, "save-continuity", str(doc), "--wrap-token", prep["wrap_token"], "--require-rederive")
    assert cli.returncode == 1
    assert "re-derive is not enabled" in cli.stderr

    # Without the requirement the same save still goes through (static checks only).
    validated_save_continuity(store, _continuity(TRUE_STATE), wrap_token=prep["wrap_token"])


def test_prepare_wrap_shows_the_composer_the_stale_flags(project):
    store, repo = project
    _save(store, _continuity(TRUE_STATE))
    (repo / "app.py").write_text('VERSION = "1.4.0"\n')  # the planted drift
    store.record("next session", "observation")

    result = prepare_wrap(store)
    package = result["package"]
    assert package is not None
    flagged = TRUE_STATE[0] + "  ⚠ STALE (now '0', claimed '1')"
    assert flagged in package["continuity"]
    assert package["unconfirmed_state"] == ["line 7: ⚠ STALE (now '0', claimed '1')"]
    assert flagged in format_wrap_package_text(result)
    assert "⚠" not in store.load_continuity()  # the stored text is untouched


def test_prepare_wrap_says_when_state_was_not_checked(project):
    store, _repo = project
    _save(store, _continuity(TRUE_STATE))
    revoke_store(store.path)
    store.record("next session", "observation")

    package = prepare_wrap(store)["package"]
    assert package is not None
    assert "[anneal re-derive]" not in package["continuity"]
    assert "unconfirmed_state" not in package
    assert "no State line was checked" in package["instructions"]


JUDGED_ONLY = ["- Notes read well [judged: t, 2026-09-30, against the draft]"]


def test_an_enabled_store_where_no_command_ran_is_not_a_clean_exit(project):
    """L1 (spore-1233 review): an opted-in store whose State ran no command
    (only [judged:] lines here; a spent load budget reaches the same branch)
    still exited 0."""
    store, _repo = project
    _save(store, _continuity(JUDGED_ONLY))
    assert _cli(store, "continuity", "--rederive").returncode == 3


def test_require_rederive_refuses_a_save_where_no_command_ran(project):
    store, _repo = project
    _save(store, _continuity(TRUE_STATE))
    store.record("next session", "observation")
    prep = prepare_wrap(store)
    with pytest.raises(ValueError, match="no State command ran"):
        validated_save_continuity(
            store, _continuity(JUDGED_ONLY), wrap_token=prep["wrap_token"], require_rederive=True
        )


def test_the_composer_is_never_handed_a_header_line(project):
    """L1 + L3 r2 (spore-1233): a header handed to the composer could be moved
    anywhere and persist a load-time verdict, so prepare_wrap hands over only
    the flagged State lines, opted in or not."""
    store, repo = project
    _save(store, _continuity(TRUE_STATE))
    (repo / "app.py").write_text('VERSION = "1.4.0"\n')
    store.record("next session", "observation")
    on = prepare_wrap(store)
    assert on["package"] is not None
    assert "[anneal re-derive]" not in on["package"]["continuity"]
    assert "⚠ STALE" in on["package"]["continuity"]
    assert "[anneal re-derive]" not in format_wrap_package_text(on)


def test_a_store_whose_commands_all_failed_before_running_checked_nothing(project):
    """codex L3 r1: a line refused at the repository-shape check never spawns its
    command, yet it counted as run, so the load exited 0."""
    store, repo = project
    _save(store, _continuity(TRUE_STATE[1:3]))  # the two git lines
    alternates = repo / ".git" / "objects" / "info" / "alternates"
    alternates.parent.mkdir(parents=True, exist_ok=True)
    alternates.write_text("/tmp\n")  # not a plain repository any more
    out = _cli(store, "continuity", "--rederive", "--json")
    assert out.returncode == 3, out.stdout
    assert json.loads(out.stdout)["rederive"]["clean"] is False


def test_judged_only_state_is_not_reported_clean(project):
    store, _repo = project
    _save(store, _continuity(JUDGED_ONLY))
    out = _cli(store, "continuity", "--rederive", "--json")
    assert json.loads(out.stdout)["rederive"]["clean"] is False


def test_an_authored_preamble_line_matching_the_header_is_kept(project):
    """codex + complement L3 r1/r2: the save strips the header only where
    re-derive writes it (the first line); anywhere else it is authored text."""
    store, _repo = project
    _save(store, _continuity(TRUE_STATE))
    store.record("next session", "observation")
    header = rederive_continuity(store).text.split("\n")[0]
    assert header.startswith("> [anneal re-derive]")
    prepare_wrap(store)
    title, rest = _continuity(TRUE_STATE).split("\n", 1)
    composed = f"{title}\n\nA note quoting the header:\n{header}\n{rest}"
    validated_save_continuity(store, composed)
    assert header in store.load_continuity()
