"""W3 erasure (ruled "fix both", 10-13): ``secure_delete`` on every connection, a WAL
clear after each erasing write, a one-time VACUUM for a pre-existing store, and a
per-store salt for the content hashes in tombstones and audit events.

Built from the 10-14 RUN (project memory ``w3_erasure_run_1014``): before the fix a
deleted short secret stayed in the database file and in 5 WAL frames, and its
unsalted sha256 sat in the tombstone and two audit events, recovered by guessing in
0.05 s."""
from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import time
import warnings
from pathlib import Path

import pytest

import anneal_memory.store as store_mod
from anneal_memory import Store
from anneal_memory.audit import AuditTrail
from anneal_memory.store import register_writer_schema

SECRET = "my SSN is 078-05-1120"
NEEDLE = SECRET.encode("utf-8")


def _fill(s: Store, n: int = 60) -> None:
    for i in range(n):
        s.record(f"filler observation number {i} about ordinary project work " * 3, "observation")


def _bytes_in(path: Path, needle: bytes = NEEDLE) -> int:
    return path.read_bytes().count(needle) if path.exists() else 0


def _salt(db: Path) -> str | None:
    c = sqlite3.connect(str(db))
    row = c.execute("SELECT value FROM metadata WHERE key = 'content_hash_salt'").fetchone()
    c.close()
    return row[0] if row else None


def _meta(db: Path, key: str) -> str | None:
    c = sqlite3.connect(str(db))
    row = c.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    c.close()
    return row[0] if row else None


def _audit_lines(db: Path) -> list[dict]:
    out = []
    for p in db.parent.glob(db.stem + ".audit*.jsonl"):
        out += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return out


def _hmac(salt: str, content: str) -> str:
    return "hmac-sha256:" + hmac.new(salt.encode("utf-8"), content.encode("utf-8"),
                                     hashlib.sha256).hexdigest()


def _build_pre_w3_store(db: Path, monkeypatch) -> str:
    """A store as a release before this one left it: content deleted without
    secure_delete, unsalted hashes in the tombstone and the audit trail, no salt row,
    at generation 1. Returns the deleted episode's id."""
    orig_connect = store_mod.connect

    def no_secure_delete(*a, **k):
        c = orig_connect(*a, **k)
        c.execute("PRAGMA secure_delete=OFF")
        return c

    with monkeypatch.context() as m:
        m.setattr(store_mod, "connect", no_secure_delete)
        m.setattr(store_mod, "_content_hash",
                  lambda content, salt: hashlib.sha256(content.encode("utf-8")).hexdigest())
        s = Store(db)
        _fill(s)
        ep = s.record(SECRET, "observation")
        _fill(s, 20)
        s.delete(ep.id)
        s.close()
    c = sqlite3.connect(str(db))
    register_writer_schema(c)
    c.execute("BEGIN IMMEDIATE")
    c.execute("DELETE FROM metadata WHERE key IN ('content_hash_salt', 'erasure_vacuum_pending')")
    for (name,) in c.execute(
        "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'anneal\\_gen%' ESCAPE '\\'"
    ).fetchall():
        c.execute(f'DROP TRIGGER "{name}"')
    c.execute("UPDATE metadata SET value = '1' WHERE key = 'format_version'")
    c.commit()
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    c.close()
    assert _bytes_in(db) >= 1, "the simulated old store must hold the deleted text"
    return ep.id


class TestSecureDelete:
    def test_every_anneal_connection_sets_it(self, tmp_path):
        db = tmp_path / "m.db"
        Store(db).close()
        for kwargs in ({}, {"must_exist": True}, {"read_only": True}):
            c = store_mod.connect(db, **kwargs)
            assert c.execute("PRAGMA secure_delete").fetchone()[0] == 1, kwargs
            c.close()

    def test_deleted_text_is_gone_from_db_and_wal_while_open_and_after_close(self, tmp_path):
        db = tmp_path / "m.db"
        wal = tmp_path / "m.db-wal"
        s = Store(db)
        _fill(s)
        ep = s.record(SECRET, "observation")
        _fill(s, 20)
        assert _bytes_in(wal) >= 1  # the precondition the RUN found: live text in the WAL
        assert s.delete(ep.id)
        assert _bytes_in(db) == 0 and _bytes_in(wal) == 0
        assert not s._wal_clear_owed
        s.close()
        assert _bytes_in(db) == 0 and _bytes_in(wal) == 0
        c = sqlite3.connect(str(db))
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        c.close()
        assert _bytes_in(db) == 0

    def test_delete_by_origin_key_and_prune_clear_it_too(self, tmp_path):
        db = tmp_path / "m.db"
        wal = tmp_path / "m.db-wal"
        s = Store(db)
        _fill(s)
        ep = s.record(SECRET, "observation", origin_key="secret-key")
        version = s.read_episode_versioned("secret-key")[1]
        result = s.delete_by_origin_key("secret-key", expected_version=version, effect_id="fx-w3")
        assert result.outcome == "deleted"
        assert _bytes_in(db) == 0 and _bytes_in(wal) == 0
        old = s.record("old secret 555-55-5555", "observation",
                       timestamp="2000-01-01T00:00:00Z")
        assert s.prune(older_than_days=30) >= 1
        assert _bytes_in(db, b"555-55-5555") == 0 and _bytes_in(wal, b"555-55-5555") == 0
        s.close()
        del ep, old

    def test_a_reader_holding_a_snapshot_costs_the_delete_nothing(self, tmp_path):
        """MEASURED 10-14: under the store's busy timeout a TRUNCATE checkpoint
        blocked by a reader took 5.2 s and still failed. The clear does not wait;
        it is owed and paid at close."""
        db = tmp_path / "m.db"
        wal = tmp_path / "m.db-wal"
        s = Store(db)
        _fill(s)
        ep = s.record(SECRET, "observation")
        reader = sqlite3.connect(str(db))
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM episodes").fetchone()
        t0 = time.perf_counter()
        assert s.delete(ep.id)
        assert time.perf_counter() - t0 < 2.0
        assert s._wal_clear_owed
        reader.rollback()
        s.close()  # the reader's connection stays open, so the WAL file stays
        assert wal.exists()
        assert _bytes_in(wal) == 0 and _bytes_in(db) == 0
        reader.close()

    def test_a_delete_inside_a_batch_owes_the_clear(self, tmp_path):
        s = Store(tmp_path / "m.db")
        ep = s.record(SECRET, "observation")
        with s._batch():
            s.delete(ep.id)
            assert s._wal_clear_owed
        s.close()
        assert _bytes_in(tmp_path / "m.db") == 0


class TestSalt:
    def test_a_fresh_store_gets_one_and_no_vacuum(self, tmp_path):
        db = tmp_path / "m.db"
        Store(db).close()
        salt = _salt(db)
        assert salt is not None and len(salt) == 64
        int(salt, 16)
        assert _meta(db, "erasure_vacuum_pending") is None
        Store(db).close()
        assert _salt(db) == salt  # minted exactly once

    def test_two_stores_salt_differently(self, tmp_path):
        Store(tmp_path / "a.db").close()
        Store(tmp_path / "b.db").close()
        assert _salt(tmp_path / "a.db") != _salt(tmp_path / "b.db")

    def test_tombstone_and_audit_carry_the_salted_hash_and_never_the_salt(self, tmp_path):
        db = tmp_path / "m.db"
        s = Store(db)
        ep = s.record(SECRET, "observation")
        s.delete(ep.id)
        s.close()
        salt = _salt(db)
        want = _hmac(salt, SECRET)
        c = sqlite3.connect(str(db))
        assert c.execute("SELECT content_hash FROM tombstones WHERE id = ?",
                         (ep.id,)).fetchone()[0] == want
        c.close()
        events = [e for e in _audit_lines(db) if e.get("data", {}).get("episode_id") == ep.id]
        assert {e["event"] for e in events} == {"record", "delete"}
        assert all(e["data"]["content_hash"] == want for e in events)
        for p in tmp_path.iterdir():
            if p.is_file() and "audit" in p.name:
                assert salt not in p.read_text(errors="replace")
        plain = hashlib.sha256(NEEDLE).hexdigest().encode()
        for p in tmp_path.iterdir():
            if p.is_file():
                assert plain not in p.read_bytes(), p.name

    def test_the_hash_cannot_be_guessed_back_without_the_salt(self, tmp_path):
        """The RUN's attack, on the salted value: the shape and the area number
        known, the last six digits searched. Unsalted it is found; without the
        salt nothing matches. With the salt (it is in the database) it is found:
        the salt protects the audit trail, not a database file in hostile hands."""
        db = tmp_path / "m.db"
        s = Store(db)
        ep = s.record(SECRET, "observation")
        s.delete(ep.id)
        s.close()
        c = sqlite3.connect(str(db))
        stored = c.execute("SELECT content_hash FROM tombstones WHERE id = ?",
                           (ep.id,)).fetchone()[0]
        c.close()
        salt = _salt(db)
        cands = [f"my SSN is 078-{g:02d}-{n:04d}" for g in range(0, 10) for n in range(1100, 1130)]
        assert SECRET in cands
        assert not any(hashlib.sha256(x.encode()).hexdigest() == stored for x in cands)
        assert not any("hmac-sha256:" + hashlib.sha256(x.encode()).hexdigest() == stored
                       for x in cands)
        assert [x for x in cands if _hmac(salt, x) == stored] == [SECRET]

    def test_the_id_cannot_be_guessed_back_without_the_salt(self, tmp_path):
        """L2 1014+22's attack (run, 0.006 s before the fix): the tombstone keeps
        the id and the timestamp, and an unkeyed id is sha256(content+ts+nonce)[:8],
        a verifier for a guess. A recorded episode's id is keyed by the salt."""
        db = tmp_path / "m.db"
        s = Store(db)
        ep = s.record(SECRET, "observation")
        s.delete(ep.id)
        s.close()
        c = sqlite3.connect(str(db))
        tid, ts = c.execute("SELECT id, timestamp FROM tombstones").fetchone()
        c.close()
        assert tid == ep.id
        cands = [f"my SSN is 078-{g:02d}-{n:04d}" for g in range(0, 10) for n in range(1100, 1130)]
        assert not any(hashlib.sha256(f"{x}{ts}0".encode()).hexdigest()[:8] == tid
                       for x in cands)
        key = _salt(db).encode("utf-8")
        assert [x for x in cands
                if hmac.new(key, f"{x}{ts}0".encode(), hashlib.sha256).hexdigest()[:8] == tid
                ] == [SECRET]

    def test_import_keeps_the_exported_id_so_a_reimport_skips(self, tmp_path):
        """L1 1014+22 (run): with salted ids a second store re-derived different
        ids, so a re-import of the same file duplicated every episode."""
        import os
        import subprocess
        import sys
        src, dst, out = tmp_path / "a.db", tmp_path / "b.db", tmp_path / "e.json"
        s = Store(src)
        ids = {s.record(f"kept note {i}", "observation").id for i in range(3)}
        s.close()
        repo = Path(__file__).resolve().parent.parent
        env = {k: v for k, v in os.environ.items() if k != "ANNEAL_OPERATOR"}

        def cli(db, *args):
            r = subprocess.run([sys.executable, "-m", "anneal_memory.cli", "--db", str(db), *args],
                               capture_output=True, text=True, env=env, cwd=str(repo),
                               stdin=subprocess.DEVNULL)
            assert r.returncode == 0, r.stderr
            return r
        cli(src, "export", "--format", "json", "--output", str(out))
        Store(dst).close()
        first = json.loads(cli(dst, "import", str(out), "--json").stdout)
        again = json.loads(cli(dst, "import", str(out), "--json").stdout)
        assert (first["imported"], again["imported"], again["skipped"]) == (3, 0, 3)
        d = Store(dst)
        assert {e.id for e in d.recall(limit=10).episodes} == ids
        d.close()

    def test_a_read_only_handle_has_no_salt_and_cannot_hash(self, tmp_path):
        db = tmp_path / "m.db"
        Store(db).close()
        ro = Store(db, read_only=True)
        assert ro._content_salt is None
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            ro._content_hash("x")
        ro.close()


class TestPreExistingStore:
    def test_migration_salts_once_vacuums_and_keeps_old_hashes(self, tmp_path, monkeypatch):
        db = tmp_path / "m.db"
        old_id = _build_pre_w3_store(db, monkeypatch)
        unsalted = hashlib.sha256(NEEDLE).hexdigest()
        c = sqlite3.connect(str(db))
        rowids_before = c.execute("SELECT rowid, id FROM episodes ORDER BY rowid").fetchall()
        c.close()

        s = Store(db)
        # the one-time VACUUM ran and cleared the text freed before secure_delete
        assert _bytes_in(db) == 0 and _bytes_in(tmp_path / "m.db-wal") == 0
        assert _meta(db, "erasure_vacuum_pending") is None
        salt = _salt(db)
        assert salt is not None and len(salt) == 64
        # the copy the migration takes first still holds it, as the docs say
        copies = list(tmp_path.glob("m.db.gen1-*.bak"))
        assert len(copies) == 1 and _bytes_in(copies[0]) >= 1
        # the old tombstone and audit events keep the unsalted hash they were written with
        c = sqlite3.connect(str(db))
        assert c.execute("SELECT content_hash FROM tombstones WHERE id = ?",
                         (old_id,)).fetchone()[0] == unsalted
        assert [tuple(r) for r in c.execute(
            "SELECT rowid, id FROM episodes ORDER BY rowid").fetchall()] == rowids_before
        c.close()
        old_events = [e for e in _audit_lines(db) if e.get("data", {}).get("episode_id") == old_id]
        assert old_events and all(e["data"]["content_hash"] == unsalted for e in old_events)
        # new erasures are salted, and the mixed chain still verifies
        ep = s.record("second secret 123-45-6789", "observation")
        s.delete(ep.id)
        s.close()
        new_events = [e for e in _audit_lines(db) if e.get("data", {}).get("episode_id") == ep.id]
        assert new_events and all(e["data"]["content_hash"] == _hmac(salt, "second secret 123-45-6789")
                                  for e in new_events)
        assert AuditTrail.verify(db).valid
        Store(db).close()
        assert _salt(db) == salt  # exactly once
        assert len(list(tmp_path.glob("m.db.gen1-*.bak"))) == 1

    def test_a_failed_vacuum_warns_keeps_the_store_and_retries(self, tmp_path, monkeypatch):
        db = tmp_path / "m.db"
        _build_pre_w3_store(db, monkeypatch)

        class NoVacuum(sqlite3.Connection):
            def execute(self, sql, *a, **k):  # type: ignore[override]
                if sql.strip().upper() == "VACUUM":
                    raise sqlite3.OperationalError("database or disk is full")
                return super().execute(sql, *a, **k)

        orig_connect = store_mod.connect
        with monkeypatch.context() as m:
            m.setattr(store_mod, "connect",
                      lambda *a, **k: orig_connect(*a, **{**k, "factory": NoVacuum}))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                s = Store(db)
            assert any("VACUUM" in str(w.message) and "retries" in str(w.message) for w in caught)
            s.record("still works", "observation")
            s.close()
        assert _meta(db, "erasure_vacuum_pending") == "1"
        assert _salt(db) is not None
        assert _bytes_in(db) >= 1  # not cleared yet: said, not hidden
        Store(db).close()  # the next open retries and succeeds
        assert _meta(db, "erasure_vacuum_pending") is None
        assert _bytes_in(db) == 0
