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
        # Generation 3: the identity is the DB's; a raw save seeds it (and the
        # file it projects). A direct file write would be a foreign edit.
        self.store.save_continuity(text)

    def raw(self) -> str:
        with open(self.store.continuity_path, encoding="utf-8", newline="") as f:
            return f.read()

    def replace(self, heading: str, body: str):
        got = self.store.read_section(heading)
        assert got is not None
        return self.store.replace_section(heading, body, expected_version=got[1])


@unittest.skipIf(_store_mod.fcntl is not None, "this platform has a file lock")
class TestNoLockRefuses(_Case):
    def test_replace_section_raises_without_a_lock(self) -> None:
        _, v = self.store.read_section("State")
        with self.assertRaises(ContinuityLockUnavailable):
            self.store.replace_section("State", "x", expected_version=v)


if __name__ == "__main__":
    unittest.main()
