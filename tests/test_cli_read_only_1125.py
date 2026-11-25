"""The CLI's read-only commands open the store read-only: they write nothing, so they can never migrate a store
to a newer schema generation and lock an older process out of it (the 3.0.0 release note; ``_open_store``)."""

from __future__ import annotations

import hashlib
import sqlite3
import subprocess
import sys

import pytest

from anneal_memory import Store
from anneal_memory.store import register_writer_schema


def _fingerprint(directory) -> dict[str, str]:
    """Every file beside the store: name -> sha256 (the db, its audit trail, sidecars)."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}


def _cli(db, *argv):
    return subprocess.run([sys.executable, "-m", "anneal_memory.cli", "--db", str(db), *argv],
                          capture_output=True, text=True, timeout=120)


@pytest.fixture
def populated(tmp_path):
    db = tmp_path / "memory.db"
    with Store(db, project_name="t") as store:
        ep = store.record("a decision about the outbox", "decision", source="test")
    # Mark it one generation behind: a write-capable open migrates it forward (the marker goes back to 3); a read-only
    # open leaves it alone. Without this the files are byte-equal either way and the test could not fail.
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


READ_ONLY = (
    ("episodes",),
    ("search", "outbox"),
    ("continuity",),
    ("associations", "--stats"),
    ("pattern-associations", "--stats"),
    ("diff",),
    ("stats",),
    ("history",),
    ("graph", "--format", "json"),
)


@pytest.mark.parametrize("argv", READ_ONLY, ids=[a[0] for a in READ_ONLY])
def test_a_read_only_command_leaves_every_file_byte_equal(populated, argv):
    db, _ = populated
    before = _fingerprint(db.parent)
    r = _cli(db, *argv)
    assert r.returncode in (0, 1), r.stderr[-400:]           # 1: "nothing to show" on an empty store is not a write
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
