"""The three 1.0.0rc1 L3 findings fixed before rc2 (Phill 2026-10-14: "fix the
HIGH and MEDs 2 and 4"; source: the whole-diff L3 --final on v0.9.43..rc1).

Each class here was written against a failure reproduced on rc1 (aa79de5)
before the fix: every test in this file fails there and passes on the fix.

1. A new store's consolidate-baton policy was set in a second transaction after
   the schema, so a process killed between the two left the store open to any
   session for good, and ``serve``'s size probe could change an existing
   store's policy. The kill is a real one: the child process ``os._exit``s
   right after ``Store()`` returns, the boundary the old code wrote across.
2. The generation guards covered every table in ``sqlite_master``, so a host's
   FTS5 table made every write-capable open fail ("cannot create triggers on
   virtual tables") and each failure left another pre-migration copy.
3. ``register_writer_schema`` left ``trusted_schema`` as the host had it, so a
   host connection with it OFF crashed on its first guarded write.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import anneal_memory.store as store_mod
from anneal_memory.store import Store, register_writer_schema

ROOT = Path(__file__).resolve().parent.parent


def _child(code: str, *, stdin: str | None = "") -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter that imports this tree's package."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        input=stdin, capture_output=True, text=True, env=env, timeout=120,
    )


# The child dies the instant ``Store()`` returns: the window between creating
# the store and anything its caller does next.
_KILL_AFTER_OPEN = """
    import os, sys
    from anneal_memory import store as S
    _orig = S.Store.__init__
    def _init(self, *a, **k):
        _orig(self, *a, **k)
        os._exit(9)
    S.Store.__init__ = _init
"""


def _requires_baton(db: Path) -> bool:
    s = Store(db)
    try:
        return s.consolidate_requires_baton()
    finally:
        s.close()


class TestNewStoreBatonPolicyIsSetWithTheSchema:
    def test_init_killed_right_after_the_open_still_leaves_the_store_protected(self, tmp_path):
        db = tmp_path / "m.db"
        r = _child(_KILL_AFTER_OPEN + f"""
    from anneal_memory import cli
    sys.argv = ["anneal-memory", "--db", {str(db)!r}, "init"]
    cli.main()
""")
        assert r.returncode == 9, r.stderr
        assert db.exists()
        assert _requires_baton(db) is True

    def test_serve_killed_right_after_the_open_still_leaves_the_store_protected(self, tmp_path):
        db = tmp_path / "m.db"
        r = _child(_KILL_AFTER_OPEN + f"""
    from anneal_memory import cli
    sys.argv = ["anneal-memory", "--db", {str(db)!r}, "serve", "--skip-integrity"]
    cli.main()
""")
        assert r.returncode == 9, r.stderr
        assert _requires_baton(db) is True

    def test_serve_never_changes_an_existing_stores_policy_whatever_a_probe_saw(self, tmp_path):
        # An existing store the operator created without the baton. The child
        # makes every size/existence probe of the path read "empty", which is
        # what a probe sees when another process created the file and had not
        # yet written its schema (the race complement, glm and codex flagged).
        db = tmp_path / "m.db"
        r = _child(f"""
    import sys
    from anneal_memory import cli
    sys.argv = ["anneal-memory", "--db", {str(db)!r}, "init", "--no-baton"]
    cli.main()
""")
        assert r.returncode == 0, r.stderr
        assert _requires_baton(db) is False
        r = _child(f"""
    import os, sys
    _path = os.path.realpath({str(db)!r})
    _getsize, _exists = os.path.getsize, os.path.exists
    os.path.getsize = lambda p: 0 if os.path.realpath(p) == _path else _getsize(p)
    from anneal_memory import cli
    sys.argv = ["anneal-memory", "--db", {str(db)!r}, "serve", "--skip-integrity"]
    cli.main()
""")
        assert r.returncode == 0, r.stderr
        assert _requires_baton(db) is False

    def test_init_refuses_when_another_creator_won_the_race(self, tmp_path):
        # init's exists() check passed, then another process created the store
        # before init's open: init must not report a store it did not create.
        db = tmp_path / "m.db"
        Store(db).close()
        r = _child(f"""
    import sys
    from pathlib import Path
    _real_exists = Path.exists
    Path.exists = lambda self: False if str(self) == {str(db)!r} else _real_exists(self)
    from anneal_memory import cli
    sys.argv = ["anneal-memory", "--db", {str(db)!r}, "init"]
    cli.main()
""")
        assert r.returncode == 1, r.stdout + r.stderr
        assert "already exists" in r.stderr
        assert _requires_baton(db) is False  # the winner's policy, untouched

    def test_the_library_reports_whether_this_open_created_the_store(self, tmp_path):
        db = tmp_path / "m.db"
        first = Store(db, consolidate_requires_baton_on_create=True)
        assert first.created is True
        first.close()
        second = Store(db, consolidate_requires_baton_on_create=False)
        try:
            assert second.created is False
            assert second.consolidate_requires_baton() is True  # never changed by a reopen
        finally:
            second.close()


class TestGenerationGuardsCoverOnlyAnnealTables:
    @staticmethod
    def _host_fts(db: Path) -> None:
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE VIRTUAL TABLE host_notes USING fts5(body)")
        conn.execute("INSERT INTO host_notes (body) VALUES ('a host row')")
        conn.commit()
        conn.close()

    @staticmethod
    def _as_generation_1(db: Path) -> None:
        # a store as a gen-1 release (0.9.43) left it: no guards, marker 1
        conn = sqlite3.connect(str(db))
        conn.execute("BEGIN IMMEDIATE")
        for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            "AND name LIKE 'anneal\\_gen%' ESCAPE '\\'"
        ).fetchall():
            conn.execute(f'DROP TRIGGER "{name}"')
        conn.execute("UPDATE metadata SET value = '1' WHERE key = 'format_version'")
        conn.commit()
        conn.close()

    def test_a_gen1_store_with_a_host_fts5_table_migrates_and_stays_writable(self, tmp_path):
        db = tmp_path / "m.db"
        s = Store(db)
        s.record("before the upgrade", episode_type="observation")
        s.close()
        self._as_generation_1(db)
        self._host_fts(db)
        for _ in range(2):  # the second open is where the failure used to repeat
            s = Store(db)
            s.record("after the upgrade", episode_type="observation")
            s.close()
        backups = sorted(p.name for p in tmp_path.iterdir() if p.name.endswith(".bak"))
        assert len(backups) == 1, backups
        conn = sqlite3.connect(str(db))
        try:
            assert conn.execute("SELECT value FROM metadata WHERE key = 'format_version'").fetchone()[0] == str(
                store_mod._SCHEMA_VERSION
            )
            # the host's own table is the host's: no anneal guard on it or its shadows
            assert conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type = 'trigger' AND tbl_name LIKE 'host\\_notes%' ESCAPE '\\'"
            ).fetchone()[0] == 0
            assert conn.execute("SELECT count(*) FROM host_notes WHERE host_notes MATCH 'host'").fetchone()[0] == 1
        finally:
            conn.close()

    def test_every_table_a_new_store_creates_is_on_the_guard_list(self, tmp_path):
        # The list is explicit, so a table a later release adds must be added
        # to it; this fails until it is, rather than the table going unguarded.
        db = tmp_path / "m.db"
        Store(db).close()
        conn = sqlite3.connect(str(db))
        try:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\'"
            )}
        finally:
            conn.close()
        assert tables <= store_mod._ANNEAL_TABLES, sorted(tables - store_mod._ANNEAL_TABLES)


class TestRegisterWriterSchemaUnderUntrustedSchema:
    def test_a_host_connection_with_trusted_schema_off_can_write(self, tmp_path):
        db = tmp_path / "m.db"
        Store(db).close()
        conn = sqlite3.connect(str(db))
        try:
            conn.execute("PRAGMA trusted_schema=OFF")
            register_writer_schema(conn)
            conn.execute("INSERT INTO metadata (key, value) VALUES ('host_key', 'v')")
            conn.commit()
            assert conn.execute("PRAGMA trusted_schema").fetchone()[0] == 1
        finally:
            conn.close()

    def test_a_connection_that_will_not_take_it_is_refused_at_once(self):
        class _Stuck:
            """A connection whose trusted_schema read-back stays 0."""

            def __init__(self):
                self._c = sqlite3.connect(":memory:")

            def create_function(self, *a, **k):
                return self._c.create_function(*a, **k)

            def execute(self, sql, *a):
                if sql.strip().upper() == "PRAGMA TRUSTED_SCHEMA":
                    return self._c.execute("SELECT 0")
                return self._c.execute(sql, *a)

        with pytest.raises(sqlite3.NotSupportedError, match="trusted_schema"):
            register_writer_schema(_Stuck())  # type: ignore[arg-type]
