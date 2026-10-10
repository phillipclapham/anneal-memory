"""Origin-key reads, the episode version and delete_by_origin_key (design
``episode_origin_key_design_1010.md`` r6 §3, §11.1, §12.1, §12.4; run 8)."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from anneal_memory import Store
from anneal_memory.store import (
    StoreError,
    _EPISODE_VERSION_COVERED,
    _EPISODE_VERSION_EXCLUDED,
)

# Tables that never hold an episode id, each named so a new table is a decision.
UNRELATED = {
    "drift_probes", "drift_results", "metadata", "pattern_aliases", "pattern_associations",
    "pattern_graph_projection_meta", "pattern_history", "pattern_levels",
    "processed_co_surface_events", "sqlite_sequence", "team_snapshot",
    "team_snapshot_enforced", "team_snapshot_notes", "wrap_graduations", "wraps",
}
EPISODE_ID_COLUMNS = {"id", "episode_id", "episode_a", "episode_b", "source_id", "old_id", "new_id"}


class _Case(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "m.db"
        self.store = Store(self.path)

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def raw(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def versioned(self, key: str) -> str:
        got = self.store.read_episode_versioned(key)
        assert got is not None
        return got[1]


class TestEveryTableIsClassified(_Case):
    def test_classification_is_total_and_disjoint(self) -> None:
        tables = {r[0] for r in self.raw().execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
        covered, excluded = set(_EPISODE_VERSION_COVERED), set(_EPISODE_VERSION_EXCLUDED)
        self.assertEqual(covered & excluded, set())
        self.assertEqual(covered & UNRELATED, set())
        self.assertEqual(excluded & UNRELATED, set())
        self.assertEqual(tables, covered | excluded | UNRELATED,
                         "a table appeared or vanished: classify it")

    def test_no_unrelated_table_names_an_episode(self) -> None:
        c = self.raw()
        for t in sorted(UNRELATED):
            cols = {r[1] for r in c.execute(f"PRAGMA table_info({t})")}
            with self.subTest(table=t):
                # 'id' alone is an ordinary primary key in these tables.
                self.assertEqual((cols & EPISODE_ID_COLUMNS) - {"id"}, set())

    def test_covered_columns_exist(self) -> None:
        c = self.raw()
        for t, (ids, values) in _EPISODE_VERSION_COVERED.items():
            cols = {r[1] for r in c.execute(f"PRAGMA table_info({t})")}
            with self.subTest(table=t):
                self.assertTrue(set(ids) | set(values) <= cols)


class TestReads(_Case):
    def test_reads_and_status(self) -> None:
        e = self.store.record("a readable fact", "observation", origin_key="r1")
        self.assertEqual(self.store.get_by_origin_key("r1").id, e.id)
        ep, v = self.store.read_episode_versioned("r1")
        self.assertEqual((ep.id, len(v)), (e.id, 64))
        self.assertEqual(self.store.origin_key_status("r1").state, "live")
        self.assertEqual(self.store.origin_key_status("nope").state, "unknown")
        self.assertIsNone(self.store.get_by_origin_key("nope"))
        self.assertIsNone(self.store.read_episode_versioned("nope"))

    def test_session_bookkeeping_does_not_stale_the_version(self) -> None:
        self.store.record("a stable fact", "observation", origin_key="s1")
        v = self.versioned("s1")
        with self.raw() as c:
            c.execute("UPDATE episodes SET session_id = 'w9' WHERE origin_key = 's1'")
        self.assertEqual(self.versioned("s1"), v)

    def test_read_only_handle_without_the_column_raises(self) -> None:
        legacy = Path(self._tmp.name) / "legacy.db"
        c = sqlite3.connect(legacy)
        c.execute("CREATE TABLE episodes (id TEXT PRIMARY KEY, content TEXT)")
        c.commit()
        c.close()
        ro = Store.__new__(Store)  # a handle on a store with no column
        ro._conn = sqlite3.connect(legacy)
        ro._conn.row_factory = sqlite3.Row
        ro._path = legacy
        with self.assertRaises(StoreError):
            Store._require_origin_key_column(ro, "get_by_origin_key")
        ro._conn.close()


class TestVersionCoversWhatNamesTheEpisode(_Case):
    def setUp(self) -> None:
        super().setUp()
        self.e = self.store.record("the target episode", "observation", origin_key="t1")
        self.other = self.store.record("another episode entirely", "observation")

    def _changes(self, sql: str, args: tuple) -> None:
        before = self.versioned("t1")
        with self.raw() as c:
            c.execute(sql, args)
        self.assertNotEqual(self.versioned("t1"), before, sql)

    def test_each_covered_table_moves_the_version(self) -> None:
        e, o = self.e.id, self.other.id
        cases = [
            ("UPDATE episodes SET content = 'edited target' WHERE id = ?", (e,)),
            ("INSERT INTO episode_trust (episode_id, trust) VALUES (?, 'external')", (e,)),
            ("INSERT INTO episode_derived (episode_id, source_id) VALUES (?, ?)", (o, e)),
            ("INSERT INTO supersessions (old_id, new_id, source) VALUES (?, ?, 'agent')", (o, e)),
            ("INSERT INTO state_keys (episode_id, key) VALUES (?, 'slot')", (e,)),
            ("INSERT INTO pattern_grounding (name, level, earned_on, earning, rule, episode_id) "
             "VALUES ('p', 2, '2026-10-10', 'w:1', 'checked', ?)", (e,)),
            ("INSERT INTO team_entries (entry_id, hash, episode_id) VALUES ('t-1', 'h', ?)", (e,)),
            ("INSERT INTO team_overrides (old_id, new_id) VALUES (?, ?)", (e, o)),
            ("INSERT INTO team_snapshot_rows (key, old_id, new_id) VALUES ('k', ?, ?)", (o, e)),
            ("INSERT INTO rewire_origin (old_id, new_id, standin_old, standin_new, standin_source) "
             "VALUES (?, ?, 'a', 'b', 'agent')", (e, o)),
        ]
        for sql, args in cases:
            with self.subTest(sql=sql):
                self._changes(sql, args)

    def test_a_row_about_other_episodes_leaves_it(self) -> None:
        v = self.versioned("t1")
        third = self.store.record("a third unrelated episode", "observation")
        with self.raw() as c:
            c.execute("INSERT INTO episode_trust (episode_id, trust) VALUES (?, 'external')",
                      (self.other.id,))
            c.execute("INSERT INTO episode_derived (episode_id, source_id) VALUES (?, ?)",
                      (third.id, self.other.id))
        self.assertEqual(self.versioned("t1"), v)


class TestDeleteByOriginKey(_Case):
    def test_outcomes(self) -> None:
        e = self.store.record("to delete by key", "observation", origin_key="d1")
        v = self.versioned("d1")
        stale = self.store.delete_by_origin_key("d1", expected_version="0" * 64, effect_id="fx1")
        self.assertEqual((stale.outcome, stale.version), ("version_mismatch", v))
        self.assertIsNotNone(self.store.get_by_origin_key("d1"))

        done = self.store.delete_by_origin_key("d1", expected_version=v, effect_id="fx1")
        self.assertEqual((done.outcome, done.episode_id), ("deleted", e.id))
        self.assertIsNone(self.store.get(e.id))
        again = self.store.delete_by_origin_key("d1", expected_version=v, effect_id="fx1")
        self.assertEqual(again.outcome, "already_applied")
        other = self.store.delete_by_origin_key("d1", expected_version=v, effect_id="fx2")
        self.assertEqual(other.outcome, "deleted_by_other")
        st = self.store.origin_key_status("d1")
        self.assertEqual((st.state, st.effect_id, st.episode_id), ("retired", "fx1", e.id))
        self.assertEqual(
            self.store.delete_by_origin_key("never", expected_version=v, effect_id="fx1").outcome,
            "unknown")

    def test_plain_delete_first_is_deleted_by_other(self) -> None:
        e = self.store.record("deleted the old way", "observation", origin_key="d2")
        v = self.versioned("d2")
        self.store.delete(e.id)
        self.assertEqual(self.store.delete_by_origin_key(
            "d2", expected_version=v, effect_id="fx3").outcome, "deleted_by_other")

    def test_retry_after_a_crash_between_commit_and_return(self) -> None:
        self.store.record("crash window", "observation", origin_key="d3")
        v = self.versioned("d3")
        orig = self.store._audit_log_after_commit

        def boom(*a, **k):  # the process dies after the commit
            raise KeyboardInterrupt

        self.store._audit_log_after_commit = boom  # type: ignore[method-assign]
        with self.assertRaises(KeyboardInterrupt):
            self.store.delete_by_origin_key("d3", expected_version=v, effect_id="fx4")
        self.store._audit_log_after_commit = orig  # type: ignore[method-assign]
        self.assertEqual(self.store.delete_by_origin_key(
            "d3", expected_version=v, effect_id="fx4").outcome, "already_applied")

    def test_bad_effect_id_writes_nothing(self) -> None:
        self.store.record("guarded", "observation", origin_key="d4")
        v = self.versioned("d4")
        with self.assertRaises(ValueError):
            self.store.delete_by_origin_key("d4", expected_version=v, effect_id="not ok")
        self.assertIsNotNone(self.store.get_by_origin_key("d4"))

    def test_team_operator_marks_the_removal(self) -> None:
        for flag, expect in ((False, "auto"), (True, "operator")):
            with self.subTest(team_operator=flag):
                key = f"tm{int(flag)}"
                e = self.store.record(f"team entry content {flag}", "observation", origin_key=key)
                with self.raw() as c:
                    c.execute("INSERT INTO team_entries (entry_id, hash, episode_id) VALUES (?, 'h', ?)",
                              (f"entry-{flag}", e.id))
                    c.execute("UPDATE episodes SET source = 'team:alice', metadata = ? WHERE id = ?",
                              ('{"team": {"entry_id": "entry-%s", "hash": "h"}}' % flag, e.id))
                v = self.versioned(key)
                r = self.store.delete_by_origin_key(key, expected_version=v, effect_id="fx" + key,
                                                    team_operator=flag)
                self.assertEqual(r.outcome, "deleted")
                removal = self.raw().execute(
                    "SELECT removal FROM team_entries WHERE entry_id = ?", (f"entry-{flag}",)).fetchone()
                self.assertEqual(removal, (expect,))


class TestRewireIsReportedNotVersioned(_Case):
    """codex r5 HIGH 1: A -> E -> C, delete E. An inserted A -> C does not move
    E's version; the result names the rewire in one case and the kept row in the other."""

    def _chain(self) -> tuple[str, str, str]:
        a = self.store.record("the release ships on Monday", "observation",
                              timestamp="2026-10-01T00:00:00Z")
        e = self.store.record("the release ships on Tuesday", "observation",
                              timestamp="2026-10-02T00:00:00Z", supersedes=[a.id], origin_key="E")
        c = self.store.record("the release ships on Wednesday", "observation",
                              timestamp="2026-10-03T00:00:00Z", supersedes=[e.id])
        return a.id, e.id, c.id

    def test_created(self) -> None:
        a, e, c = self._chain()
        r = self.store.delete_by_origin_key("E", expected_version=self.versioned("E"), effect_id="fx")
        self.assertEqual(r.outcome, "deleted")
        self.assertEqual(r.rewires, [{"old_id": a, "past": e, "new_id": c, "outcome": "created"}])
        self.assertEqual(r.links_removed, 2)

    def test_kept_and_version_unchanged(self) -> None:
        a, e, c = self._chain()
        v = self.versioned("E")
        with self.raw() as conn:
            conn.execute("INSERT INTO supersessions (old_id, new_id, source) VALUES (?, ?, 'operator')",
                         (a, c))
        self.assertEqual(self.versioned("E"), v)  # what was shown did not change
        r = self.store.delete_by_origin_key("E", expected_version=v, effect_id="fx")
        self.assertEqual(r.rewires, [{"old_id": a, "past": e, "new_id": c, "outcome": "kept"}])
        src = self.raw().execute("SELECT source FROM supersessions WHERE old_id = ? AND new_id = ?",
                                 (a, c)).fetchone()
        self.assertEqual(src, ("operator",))


if __name__ == "__main__":
    unittest.main()
