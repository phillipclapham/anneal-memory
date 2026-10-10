"""Origin-key grammar and section canonical form (design r5 §2, §4, §11.5 item 6)."""

from __future__ import annotations

import random
import sqlite3
import unittest
import uuid

from anneal_memory.graduation import _LINE_TERMINATORS_RE, canonical_continuity_text
from anneal_memory.origin import (
    ORIGIN_KEY_CHECK_SQL,
    ORIGIN_KEY_MAX_LEN,
    canonical_section_markdown,
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


def _generated(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    alphabet = (
        list("ab| x-#\n\t ")
        + ["\r", "\r\n", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", " ", " "]
        + ["\x00", "\x07", "\x7f", "‮", "⁦", "‏", "\U000e0041", "\ud800"]
        + ["١", "٠", "۹", "１"]  # non-ASCII digits
        + [" ", "　", " "]  # exotic spaces
        + ["́", "é", "Å"]  # combining, decomposed, a singleton
        + ["| 1x", "| ١٠x", " (2026-10-10)", "## Patterns"]
    )
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40))) for _ in range(n)]


CASES = [
    "| ‮١x",
    "| ١\x00٠x",
    "line\r\nnext\rlast\x85end",
    "- a | 1x  \n\n\n",
    "",
    "\n\n",
    "é‮",
]


class TestCanonicalSectionMarkdown(unittest.TestCase):
    def _check(self, x: str) -> None:
        f = canonical_section_markdown(x)
        self.assertEqual(canonical_section_markdown(f), f, repr(x))
        self.assertEqual(canonical_continuity_text(f), f, repr(x))
        self.assertFalse(f.endswith("\n"), repr(x))

    def test_named_cases(self) -> None:
        for x in CASES:
            with self.subTest(x=x):
                self._check(x)

    def test_joined_marker_is_ascii_in_one_pass(self) -> None:
        self.assertEqual(canonical_section_markdown("| ‮١x"), "| 1x")
        self.assertEqual(canonical_section_markdown("| ١\x00٠x"), "| 10x")

    def test_generated(self) -> None:
        for x in _generated(3000, seed=1010):
            with self.subTest(x=x):
                self._check(x)

    def test_every_line_terminator_ends_a_line(self) -> None:
        # Enumerated from the continuity grammar's own pattern, not listed by hand.
        terms = [chr(c) for c in range(0x110000) if _LINE_TERMINATORS_RE.fullmatch(chr(c))]
        terms.append("\r\n")
        self.assertIn("\x85", terms)
        for t in terms:
            with self.subTest(t=repr(t)):
                self.assertEqual(canonical_section_markdown(f"a{t}b"), "a\nb")
                self.assertEqual(
                    canonical_section_markdown(f"a{t}b").split("\n"),
                    canonical_continuity_text(f"a{t}b").replace("\r\n", "\n").split("\n"),
                )


if __name__ == "__main__":
    unittest.main()
