"""The CLI's read-only commands open the store read-only: they write nothing, so they can never migrate a store
to a newer schema generation and lock an older process out of it (the 3.0.0 release note; ``_open_store``)."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import sys

import pytest

from pathlib import Path

from anneal_memory import Store
from anneal_memory.store import register_writer_schema


def _fingerprint(directory) -> dict[str, str]:
    """Every file beside the store: name -> sha256 (the db, its audit trail, sidecars)."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}


_REPO = Path(__file__).resolve().parent.parent


def _cli(db, *argv):
    env = dict(os.environ, PYTHONPATH=str(_REPO))      # this checkout's package, whatever the cwd or an installed copy
    return subprocess.run([sys.executable, "-m", "anneal_memory.cli", "--db", str(db), *argv],
                          capture_output=True, text=True, timeout=120, env=env)


@pytest.fixture
def populated(tmp_path):
    db = tmp_path / "memory.db"
    with Store(db, project_name="t") as store:
        ep = store.record("a decision about the outbox", "decision", source="test")
    # Mark it one generation behind: a write-capable open migrates it forward (the marker goes back to 3); a read-only
    # open leaves it alone. Without this the files are byte-equal either way and the test could not fail. The store
    # keeps its generation-3 tables, so this proves the OPEN MODE, not that each command survives a store that lacks
    # them; that was run by hand on 2026-11-25 against a store built by v2.2.0's own code, every command exit-clean and
    # every file byte-equal.
    conn = sqlite3.connect(db)
    register_writer_schema(conn)
    conn.execute("UPDATE metadata SET value = '2' WHERE key = 'format_version'")
    conn.commit()
    conn.close()
    return db, ep.id


def _marker(db) -> str:
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT value FROM metadata WHERE key = 'format_version'").fetchone()[0]
    finally:
        conn.close()


# (argv, exit code on this one-episode store with no wrap): `continuity` has none yet and exits 1.
READ_ONLY = (
    (("episodes",), 0),
    (("search", "outbox"), 0),
    (("continuity",), 1),
    (("diff",), 0),
    (("stats",), 0),
    (("history",), 0),
)


@pytest.mark.parametrize("argv,code", READ_ONLY, ids=[a[0] for a, _ in READ_ONLY])
def test_a_read_only_command_leaves_every_file_byte_equal(populated, argv, code):
    db, _ = populated
    before = _fingerprint(db.parent)
    r = _cli(db, *argv)
    assert r.returncode == code, r.stderr[-400:]
    assert "Traceback" not in r.stderr, r.stderr[-400:]
    assert _fingerprint(db.parent) == before
    assert _marker(db) == "2"


def test_get_leaves_every_file_byte_equal(populated):
    db, ep_id = populated
    before = _fingerprint(db.parent)
    r = _cli(db, "get", ep_id)
    assert r.returncode == 0, r.stderr[-400:]
    assert _fingerprint(db.parent) == before
    assert _marker(db) == "2"


def test_the_control_a_write_command_does_change_the_files(populated):
    db, _ = populated
    before = _fingerprint(db.parent)
    r = _cli(db, "record", "another decision", "--type", "decision")
    assert r.returncode == 0, r.stderr[-400:]
    assert _fingerprint(db.parent) != before
    assert _marker(db) == "3"
