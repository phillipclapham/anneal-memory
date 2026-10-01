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
from anneal_memory.rederive import revoke_store

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
    assert package["stale_state"] == ["line 7: ⚠ STALE (now '0', claimed '1')"]
    assert flagged in format_wrap_package_text(result)
    assert "⚠" not in store.load_continuity()  # the stored text is untouched


def test_prepare_wrap_says_when_state_was_not_checked(project):
    store, _repo = project
    _save(store, _continuity(TRUE_STATE))
    revoke_store(store.path)
    store.record("next session", "observation")

    package = prepare_wrap(store)["package"]
    assert package is not None
    assert "not enabled for this store; STATE lines were not checked" in package["continuity"]
    assert "stale_state" not in package
