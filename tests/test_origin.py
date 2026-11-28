"""Origin-key grammar (design r5 §2, §4, §11.5 item 6)."""

from __future__ import annotations

import sqlite3
import unittest
import uuid

from anneal_memory.origin import (
    ORIGIN_KEY_CHECK_SQL,
    ORIGIN_KEY_MAX_LEN,
    origin_key_usable,
    validate_origin_key,
)

# Every class the grammar decides: (value, in grammar?).
GRAMMAR_FIXTURE: list[tuple[object, bool]] = [
    (uuid.uuid4().hex, True),  # flow's spores.py add stamps uuid4().hex
    (str(uuid.uuid4()), True),
    ("pend:A-b_c.9", True),
    ("tomb:" + uuid.uuid4().hex, True),
    ("a" * ORIGIN_KEY_MAX_LEN, True),
    ("a" * (ORIGIN_KEY_MAX_LEN + 1), False),
    *[(f"k{c}k", True) for c in "._:-"],
    ("", False),
    (" k", False),
    ("k ", False),
    ("a b", False),
    ("k\x00", False),
    ("\x00", False),
    ("k\n", False),
    ("ké", False),
    ("k١", False),  # a non-ASCII digit
    ("k/1", False),
    ("k*", False),
    ("k[a]", False),
    (b"abc", False),
    (7, False),
    (None, False),
]


class TestGrammar(unittest.TestCase):
    def test_python_rule(self) -> None:
        for value, ok in GRAMMAR_FIXTURE:
            with self.subTest(value=value):
                self.assertIs(origin_key_usable(value), ok)
                if ok:
                    self.assertEqual(validate_origin_key(value), value)
                else:
                    with self.assertRaises(ValueError):
                        validate_origin_key(value)

    def test_sql_check_agrees_on_insert_and_update(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute(f"CREATE TABLE t (id INTEGER PRIMARY KEY, origin_key TEXT CHECK ({ORIGIN_KEY_CHECK_SQL}))")
        conn.execute("INSERT INTO t (id, origin_key) VALUES (1, NULL)")
        for value, ok in GRAMMAR_FIXTURE:
            if value is None:
                continue
            with self.subTest(value=value):
                for sql, args in (
                    ("INSERT INTO t (origin_key) VALUES (?)", (value,)),
                    ("UPDATE t SET origin_key = ? WHERE id = 1", (value,)),
                ):
                    try:
                        with conn:
                            conn.execute(sql, args)
                        stored = True
                    except sqlite3.IntegrityError:
                        stored = False
                    # TEXT affinity turns an int into its text before the CHECK
                    # runs, so 7 is stored as the key "7"; every stored key is text.
                    expected = origin_key_usable(str(value)) if isinstance(value, int) else ok
                    self.assertIs(stored, expected, sql)
                self.assertEqual(
                    conn.execute("SELECT count(*) FROM t WHERE origin_key IS NOT NULL "
                                 "AND typeof(origin_key) != 'text'").fetchone()[0], 0)
                with conn:
                    conn.execute("DELETE FROM t WHERE id != 1")
                    conn.execute("UPDATE t SET origin_key = NULL WHERE id = 1")


if __name__ == "__main__":
    unittest.main()
