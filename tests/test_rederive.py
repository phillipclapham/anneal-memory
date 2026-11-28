"""Re-derive at load (spore-1230): the two behaviours a real run reproduced.

1. A planted stale STATE line is flagged inline at load (the first real run
   reported it as a DERIVE ERROR, because ``grep -c`` exits 1 on a count of 0).
2. Crafted malicious STATE lines are refused, at load and at save, with no side
   effect. Containment design: docs/rederive.md.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from anneal_memory import Store, prepare_wrap, validated_save_continuity
from anneal_memory.rederive import (
    DeriveRefused,
    allow_store,
    parse_annotation,
    rederive_continuity,
    validate_command,
)
from anneal_memory.schema import PROJECT_SCHEMA

pytestmark = [
    pytest.mark.skipif(shutil.which("git") is None, reason="needs git"),
    # re-derive refuses non-POSIX systems by design (rederive._SUPPORTED); the
    # `project` fixture calls allow_store, which raises there.
    pytest.mark.skipif(os.name != "posix", reason="re-derive is POSIX-only by design"),
]

_GIT = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]


def _continuity(state_lines: list[str]) -> str:
    return "\n".join([
        "# Scratch — Memory (v1)", "",
        "## Plan", "- Ship 1.3.0.", "",
        "## State", *state_lines, "",
        "## Decisions", "- Tag before bump.", "",
        "## Open", "- Release notes.", "",
        "## Lessons", "- none yet", "",
        "## History", "- Bumped to 1.3.0.", "",
    ])


TRUE_STATE = [
    '- The working tree is at 1.3.0 [derive: grep -c "1.3.0" app.py => 1]',
    "- The latest tag is v1.2.0 [derive: git describe --tags --abbrev=0 @REF => v1.2.0]",
    "- v1.2.0 is behind the pinned ref [derive: git merge-base --is-ancestor v1.2.0 @REF]",
    "- Notes read well [judged: t, 2026-09-30, against the draft]",
]


@pytest.fixture
def project(tmp_path, monkeypatch):
    trust = tmp_path / "trust.json"
    monkeypatch.setenv("ANNEAL_MEMORY_DERIVE_TRUST", str(trust))
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run([*_GIT, *a], cwd=repo, check=True, capture_output=True)
    (repo / "app.py").write_text('VERSION = "1.2.0"\n')
    run("init", "-q")
    run("add", ".")
    run("commit", "-qm", "v1")
    run("tag", "v1.2.0")
    (repo / "app.py").write_text('VERSION = "1.3.0"\n')
    run("commit", "-qam", "bump")
    store = Store(tmp_path / "store" / "proj.db", project_name="Scratch", section_schema=PROJECT_SCHEMA)
    allow_store(store.path, repo)
    yield store, repo
    store.close()


def _save(store: Store, text: str) -> None:
    store.record("a session episode", "observation")
    prepare_wrap(store)
    validated_save_continuity(store, text)




EVIL = [
    "git log -1; touch {canary}",
    'git -c core.fsmonitor="touch {canary}" status',
    'sh -c "touch {canary}"',
    "git log --output={canary} -1",
    "test -e $(touch {canary})",
    "grep -c root /etc/passwd => 1",
    "grep -c x ../../../etc/passwd => 1",
    "grep -f /etc/passwd app.py",
    "git log --format=%G? -1",
    'git grep -O"touch {canary}" VERSION',
    "git diff --no-index /etc/passwd app.py",
    "git tag pwned",
    "git log -1 => x",  # format.pretty in repo config reaches log (codex r2)
    "git cat-file -p HEAD:app.py => x",  # content through alternates (codex r2)
    "git log -1 --format=%+G? => x",  # signature modifiers (L2)
    "git for-each-ref --format=%(*signature:grade) => x",
    "grep -h API .env => x",  # prints content (L2)
    "grep -c root link_out => 1",  # a symlink out of the root
    "grep -c API env_link => 1",  # a TRACKED symlink to the untracked .env (L3)
    "grep -c url .git/config => 1",
    "grep -c url .GIT/config => 1",  # case-insensitive filesystems
]


