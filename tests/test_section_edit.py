"""Section reads and replace_section (design ``episode_origin_key_design_1010.md``
r6 §4, §11.3, §11.4, §12.3, §13; Phill np-e2beb6b7, np-17ebdb7b)."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import unittest
from pathlib import Path

import anneal_memory.store as _store_mod
from anneal_memory import ContinuityLockUnavailable, SectionError, Store, WrapContinuityMovedError
from anneal_memory.continuity import prepare_wrap
from anneal_memory.graduation import _LINE_TERMINATORS_RE
from anneal_memory.origin import canonical_section_markdown

DOC = (
    "# Memory\n\n## State\n\nold state line\n\n## Durable Facts\n\n- fact one\n\n"
    "## Patterns\n\n- p | 1x\n\n## Decisions\n\nd\n\n## Context\n\nctx line\n"
)


# replace_section takes continuity_lock(require=True): with no lock (Windows) it raises.
_NEEDS_LOCK = unittest.skipIf(_store_mod.fcntl is None, "replace_section needs a file lock")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _Case(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self._tmp.name) / "m.db")
        self.write(DOC)

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def write(self, text: str) -> None:
        with open(self.store.continuity_path, "w", encoding="utf-8", newline="") as f:
            f.write(text)

    def raw(self) -> str:
        with open(self.store.continuity_path, encoding="utf-8", newline="") as f:
            return f.read()

    def replace(self, heading: str, body: str):
        got = self.store.read_section(heading)
        assert got is not None
        return self.store.replace_section(heading, body, expected_version=got[1])


class TestLineBreakSets(unittest.TestCase):
    def test_splitlines_breaks_on_exactly_newline_plus_the_canonical_terminators(self) -> None:
        # The raw/canonical line alignment rests on this (L3 r6, glm HIGH refuted).
        breaks = {chr(c) for c in range(0x110000) if len(("a" + chr(c) + "b").splitlines()) == 2}
        mapped = {chr(c) for c in range(0x110000) if _LINE_TERMINATORS_RE.fullmatch(chr(c))}
        self.assertEqual(breaks, mapped | {"\n"})


class TestRead(_Case):
    def test_reads_every_editable_kind(self) -> None:
        self.assertEqual(self.store.read_section("State")[0], "old state line")
        self.assertEqual(self.store.read_section("state")[0], "old state line")
        self.assertEqual(self.store.read_section("Durable Facts")[0], "- fact one")  # optional
        self.assertEqual(self.store.read_section("Context")[0], "ctx line")  # final section
        text, version = self.store.read_section("State")
        self.assertEqual(version, sha(text))

    def test_absent_and_refused(self) -> None:
        self.write(DOC.replace("## Durable Facts\n\n- fact one\n\n", ""))
        self.assertIsNone(self.store.read_section("Durable Facts"))
        with self.assertRaises(SectionError) as cm:
            self.store.read_section("Nonsense")
        self.assertEqual(cm.exception.reason, "no_such_section")
        self.write(DOC + "\n## State\n\nsecond copy\n")
        with self.assertRaises(SectionError) as cm:
            self.store.read_section("State")
        self.assertEqual(cm.exception.reason, "ambiguous_heading")

    @_NEEDS_LOCK
    def test_extended_heading_is_the_section(self) -> None:
        self.write(DOC.replace("## State\n", "## State of Mind\n"))
        self.assertEqual(self.store.read_section("State")[0], "old state line")
        r = self.replace("State", "fresh")
        self.assertEqual(r.outcome, "written")
        self.assertIn("## State of Mind\n\nfresh\n\n## Durable Facts", self.raw())

    def test_missing_file(self) -> None:
        self.store.continuity_path.unlink()
        self.assertIsNone(self.store.read_section("State"))


@_NEEDS_LOCK
class TestReplace(_Case):
    def test_round_trip_version_is_the_canonical_body(self) -> None:
        for heading, body in (("State", "a line  \r\nnext ‮| ١x\n\n"),
                              ("Durable Facts", "\n\n- new fact\n"),
                              ("Context", "last section"),
                              ("Decisions", "")):
            with self.subTest(heading=heading):
                r = self.replace(heading, body)
                self.assertEqual(r.outcome, "written", r)
                self.assertEqual(r.version, sha(canonical_section_markdown(body)))
                self.assertEqual(self.store.read_section(heading),
                                 (canonical_section_markdown(body), r.version))

    def test_lines_outside_the_section_are_kept_byte_for_byte(self) -> None:
        doc = DOC.replace("\n", "\r\n").replace("ctx line", "ctx line  tail")
        self.write(doc)
        before = self.raw()
        r = self.replace("State", "replaced")
        self.assertEqual(r.outcome, "written")
        after = self.raw()
        head, tail = before.split("old state line", 1)
        self.assertTrue(after.startswith(head[: head.index("## State")] + "## State\r\n"))
        self.assertTrue(after.endswith(tail[tail.index("## Durable Facts"):]))
        self.assertEqual(self.store.read_section("State")[0], "replaced")

    def test_mode_is_kept_and_no_tmp_is_left(self) -> None:
        if os.name != "posix":
            self.skipTest("POSIX modes")
        os.chmod(self.store.continuity_path, 0o640)
        self.replace("State", "x")
        self.assertEqual(os.stat(self.store.continuity_path).st_mode & 0o777, 0o640)
        left = [p.name for p in self.store.continuity_path.parent.iterdir() if "section-edit" in p.name]
        self.assertEqual(left, [])

    def test_stale_version_writes_nothing(self) -> None:
        _, v = self.store.read_section("State")
        self.replace("State", "first edit")
        before = self.raw()
        r = self.store.replace_section("State", "second", expected_version=v)
        self.assertEqual((r.outcome, r.version), ("version_mismatch", sha("first edit")))
        self.assertEqual(self.raw(), before)

    def test_refusals_write_nothing(self) -> None:
        before = self.raw()
        _, v = self.store.read_section("Context")
        cases = [
            ("Patterns", "x", "graduating"),
            ("Nonsense", "x", "no_such_section"),
            ("Context", "## Injected", "invalid_body"),
            ("Context", "ok ## Injected", "invalid_body"),  # a terminator, then a header
        ]
        for heading, body, reason in cases:
            with self.subTest(heading=heading, body=body):
                r = self.store.replace_section(heading, body, expected_version=v)
                self.assertEqual((r.outcome, r.reason), ("refused", reason))
        self.assertEqual(self.raw(), before)

    def test_a_pipeline_tmp_refuses(self) -> None:
        stem = self.store.continuity_path.stem
        tmp = self.store.continuity_path.parent / f"{stem}.abcdef123456-0a1b2c3d.md.tmp"
        tmp.write_text("a committed wrap not yet renamed")
        r = self.replace("State", "x")
        self.assertEqual((r.outcome, r.reason), ("refused", "pipeline_tmp_present"))

    def test_an_open_wrap_refuses(self) -> None:
        self.store.record("an episode for the wrap window", "observation")
        prep = prepare_wrap(self.store)
        self.assertEqual(prep["status"], "ready")
        _, v = self.store.read_section("State")
        r = self.store.replace_section("State", "x", expected_version=v)
        self.assertEqual((r.outcome, r.reason), ("refused", "wrap_in_progress"))


@_NEEDS_LOCK
class TestWrapStartSeesAnEdit(_Case):
    """§12.3: an edit landing after prepare's read refuses the wrap's start."""

    def test_wrap_started_compares_the_text_prepare_read(self) -> None:
        composed = self.store.load_continuity() or ""
        self.replace("State", "edited after the read")
        with self.assertRaises(WrapContinuityMovedError):
            self.store.wrap_started(token="t" * 32, episode_ids=[],
                                    expect_continuity_sha256=sha(composed))
        self.assertIsNone(self.store.get_wrap_started_at())

    def test_prepare_downgrades_with_a_retry(self) -> None:
        self.store.record("an episode for the wrap window", "observation")
        real = Store.load_continuity
        calls = {"n": 0}
        store = self.store

        def load_then_edit(self_):  # the edit lands right after prepare's read
            out = real(self_)
            calls["n"] += 1
            if calls["n"] == 1:
                Store.load_continuity = real  # type: ignore[method-assign]
                _, v = store.read_section("State")
                store.replace_section("State", "edited mid-prepare", expected_version=v)
            return out

        Store.load_continuity = load_then_edit  # type: ignore[method-assign]
        try:
            prep = prepare_wrap(self.store)
        finally:
            Store.load_continuity = real  # type: ignore[method-assign]
        self.assertIn("downgraded-continuity-changed", str(prep))
        self.assertIsNone(self.store.get_wrap_started_at())
        self.assertEqual(prepare_wrap(self.store)["status"], "ready")


@unittest.skipIf(_store_mod.fcntl is not None, "this platform has a file lock")
class TestNoLockRefuses(_Case):
    def test_replace_section_raises_without_a_lock(self) -> None:
        _, v = self.store.read_section("State")
        with self.assertRaises(ContinuityLockUnavailable):
            self.store.replace_section("State", "x", expected_version=v)


if __name__ == "__main__":
    unittest.main()
