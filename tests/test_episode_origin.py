"""Episode origin keys: the migration, record(origin_key=...) and the column rules
(design ``episode_origin_key_design_1010.md`` r6 §2, §3, §11.5, §12.2; runs 4, 6, 7)."""

from __future__ import annotations

import sqlite3
import tempfile
import multiprocessing
import unittest
from pathlib import Path

from anneal_memory import OriginKeyConflict, Store
from anneal_memory.origin import origin_key_usable
from anneal_memory.store import _ORIGIN_TRIGGER_GEN, _ORIGIN_TRIGGERS

from tests.test_origin import GRAMMAR_FIXTURE


def _record_raced(path: str) -> str:
    s = Store(path)
    try:
        return s.record("raced fact", "observation", origin_key="race").id
    except BaseException as exc:  # noqa: BLE001
        return f"ERR {exc!r}"
    finally:
        s.close()


class _StoreCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "m.db"
        self.store = Store(self.path)

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def raw(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)


class TestMint(_StoreCase):
    def test_every_record_gets_a_distinct_key_in_the_grammar(self) -> None:
        eps = [self.store.record(f"episode number {i}", "observation") for i in range(20)]
        keys = [e.origin_key for e in eps]
        self.assertTrue(all(origin_key_usable(k) for k in keys), keys)
        self.assertEqual(len(set(keys)), 20)
        for e in eps:
            self.assertEqual(self.store.get(e.id).origin_key, e.origin_key)

    def test_a_raw_keyless_insert_is_minted_and_backfilled(self) -> None:
        with self.raw() as c:
            c.execute("INSERT INTO episodes (id, timestamp, type, content) "
                      "VALUES ('aa000001', '2026-10-10T00:00:00Z', 'observation', 'raw')")
            # Simulate a row from before the column: clear the trigger's key.
            c.execute("DROP TRIGGER episode_origin_key_immutable")
            c.execute("UPDATE episodes SET origin_key = NULL WHERE id = 'aa000001'")
        self.store.close()
        self.store = Store(self.path)  # a write-capable open backfills
        self.assertTrue(origin_key_usable(self.store.get("aa000001").origin_key))
        names = {r[0] for r in self.raw().execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
        self.assertTrue(set(_ORIGIN_TRIGGERS) <= names)  # the dropped trigger came back

    def test_delete_retires_the_key(self) -> None:
        e = self.store.record("to be deleted", "observation")
        self.store.delete(e.id)
        row = self.raw().execute(
            "SELECT episode_id, effect_id FROM retired_origin_keys WHERE origin_key = ?",
            (e.origin_key,)).fetchone()
        self.assertEqual(row, (e.id, None))


class TestColumnRules(_StoreCase):
    """Run 7: the CHECK through the real migrated column, and immutability."""

    def test_check_matches_the_grammar(self) -> None:
        e = self.store.record("check target", "observation")
        c = self.raw()
        for value, ok in GRAMMAR_FIXTURE:
            if value is None or isinstance(value, int):
                continue
            with self.subTest(value=value):
                try:
                    with c:
                        c.execute("INSERT INTO episodes (id, timestamp, type, content, origin_key) "
                                  "VALUES ('bb000001', '2026-10-10T00:00:00Z', 'observation', 'x', ?)",
                                  (value,))
                    stored = True
                except sqlite3.IntegrityError:
                    stored = False
                self.assertIs(stored, ok)
                with c:
                    c.execute("DELETE FROM episodes WHERE id = 'bb000001'")
        self.assertEqual(self.store.get(e.id).content, "check target")

    def test_a_set_key_cannot_change(self) -> None:
        e = self.store.record("immutable", "observation")
        c = self.raw()
        for new in ("other", None):
            with self.subTest(new=new), self.assertRaises(sqlite3.DatabaseError) as cm:
                with c:
                    c.execute("UPDATE episodes SET origin_key = ? WHERE id = ?", (new, e.id))
            self.assertIn("immutable", str(cm.exception))
        # Updates that leave the key alone still work.
        with c:
            c.execute("UPDATE episodes SET content = 'changed' WHERE id = ?", (e.id,))
        self.assertEqual(self.store.get(e.id).origin_key, e.origin_key)


class TestTriggerGeneration(_StoreCase):
    """Run 4, in-process half: a higher stored generation is never overwritten,
    and a lower one is replaced."""

    def _mint_sql(self) -> str:
        return self.raw().execute(
            "SELECT sql FROM sqlite_master WHERE name = 'episode_origin_key_mint'").fetchone()[0]

    def test_a_newer_body_survives_this_build(self) -> None:
        newer = _ORIGIN_TRIGGERS["episode_origin_key_mint"].replace(
            "lower(hex(randomblob(16)))", "'n' || lower(hex(randomblob(16)))")
        with self.raw() as c:
            c.execute("DROP TRIGGER episode_origin_key_mint")
            c.execute(newer)
            c.execute("UPDATE metadata SET value = ? WHERE key = 'origin_trigger_gen'",
                      (str(_ORIGIN_TRIGGER_GEN + 1),))
        self.store.close()
        self.store = Store(self.path)
        self.assertIn("'n' ||", self._mint_sql())
        e = self.store.record("under the newer trigger", "observation")
        self.assertTrue(e.origin_key.startswith("n"))
        self.assertEqual(self.raw().execute(
            "SELECT count(*) FROM episodes WHERE origin_key IS NULL").fetchone()[0], 0)

    def test_an_older_body_is_replaced(self) -> None:
        with self.raw() as c:
            c.execute("DROP TRIGGER episode_origin_key_mint")
            c.execute(_ORIGIN_TRIGGERS["episode_origin_key_mint"].replace(
                "lower(hex(randomblob(16)))", "'old' || lower(hex(randomblob(16)))"))
            c.execute("UPDATE metadata SET value = '0' WHERE key = 'origin_trigger_gen'")
        self.store.close()
        self.store = Store(self.path)
        self.assertNotIn("'old'", self._mint_sql())

    def test_two_mint_triggers_give_one_key(self) -> None:
        with self.raw() as c:
            c.execute(_ORIGIN_TRIGGERS["episode_origin_key_mint"].replace(
                "episode_origin_key_mint", "episode_origin_key_mint_other"))
        e = self.store.record("two mints", "observation")
        self.assertTrue(origin_key_usable(e.origin_key))
        self.assertEqual(self.store.get(e.id).origin_key, e.origin_key)


class TestKeyedRecord(_StoreCase):
    """Run 6."""

    def test_retry_returns_the_stored_episode(self) -> None:
        a = self.store.record("keyed fact", "observation", origin_key="pend:x1")
        b = self.store.record("keyed fact", "observation", origin_key="pend:x1")
        self.assertEqual((a.id, a.origin_key), (b.id, "pend:x1"))
        self.assertEqual(self.raw().execute("SELECT count(*) FROM episodes").fetchone()[0], 1)

    def test_retry_with_supersedes_returns_not_refuses(self) -> None:
        old = self.store.record("the meeting is on Monday at noon", "observation")
        new = self.store.record("the meeting is on Tuesday at noon", "observation",
                                supersedes=[old.id], origin_key="k2")
        again = self.store.record("the meeting is on Tuesday at noon", "observation",
                                  supersedes=[old.id], origin_key="k2")
        self.assertEqual(again.id, new.id)
        c = self.raw()
        self.assertEqual(c.execute("SELECT count(*) FROM episodes").fetchone()[0], 2)
        self.assertEqual(c.execute("SELECT count(*) FROM supersessions").fetchone()[0], 1)

    def test_a_different_payload_is_a_conflict(self) -> None:
        self.store.record("keyed fact", "observation", origin_key="k3")
        for kw in ({"content": "other fact"}, {"episode_type": "decision"}, {"source": "user"}):
            args = {"content": "keyed fact", "episode_type": "observation", **kw}
            with self.subTest(kw=kw), self.assertRaises(OriginKeyConflict):
                self.store.record(origin_key="k3", **args)
        self.assertEqual(self.raw().execute("SELECT count(*) FROM episodes").fetchone()[0], 1)

    def test_a_retired_key_is_refused(self) -> None:
        e = self.store.record("short lived", "observation", origin_key="k4")
        self.store.delete(e.id)
        with self.assertRaisesRegex(ValueError, "never reused"):
            self.store.record("short lived", "observation", origin_key="k4")
        self.assertEqual(self.raw().execute("SELECT count(*) FROM episodes").fetchone()[0], 0)

    def test_a_key_outside_the_grammar_writes_nothing(self) -> None:
        for bad in ("a b", "", "k\x00", "é"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.store.record("anything", "observation", origin_key=bad)
        self.assertEqual(self.raw().execute("SELECT count(*) FROM episodes").fetchone()[0], 0)

    def test_concurrent_same_key_gives_one_row(self) -> None:
        # Real processes, as two writers would be (design §5 run 6).
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(4) as pool:
            out = pool.map(_record_raced, [str(self.path)] * 4)
        self.assertEqual([o for o in out if o.startswith("ERR")], [])
        self.assertEqual(len(set(out)), 1)
        self.assertEqual(self.raw().execute(
            "SELECT count(*) FROM episodes WHERE origin_key = 'race'").fetchone()[0], 1)

if __name__ == "__main__":
    unittest.main()


class TestExportImport(unittest.TestCase):
    """Run 10: a SQLite export copies keys; a JSON export imported mints fresh ones
    (a rebuilt store is a new resource)."""

    def test_keys_across_export(self) -> None:
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as d:
            a, b, c, j = (str(Path(d) / n) for n in ("a.db", "b.db", "c.db", "a.json"))
            s = Store(a)
            s.record("exported fact", "observation", origin_key="ex:1")
            s.close()

            def cli(*args: str) -> None:
                subprocess.run([sys.executable, "-m", "anneal_memory", *args],
                               check=True, capture_output=True)

            cli("--db", a, "export", "--format", "json", "--output", j)
            self.assertIn('"origin_key": "ex:1"', Path(j).read_text())
            cli("--db", b, "init")
            cli("--db", b, "import", j)
            cli("--db", a, "export", "--format", "sqlite", "--output", c)
            imported = sqlite3.connect(b).execute("SELECT origin_key FROM episodes").fetchall()
            copied = sqlite3.connect(c).execute("SELECT origin_key FROM episodes").fetchall()
            self.assertEqual(len(imported), 1)
            self.assertNotEqual(imported[0][0], "ex:1")
            self.assertTrue(origin_key_usable(imported[0][0]))
            self.assertEqual(copied, [("ex:1",)])
