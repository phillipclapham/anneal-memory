"""Re-derive at load (spore-1230): the two behaviours a real run reproduced.

1. A planted stale STATE line is flagged inline at load (the first real run
   reported it as a DERIVE ERROR, because ``grep -c`` exits 1 on a count of 0).
2. Crafted malicious STATE lines are refused, at load and at save, with no side
   effect. Containment design: docs/rederive.md.
"""

from __future__ import annotations

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

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

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


def test_planted_stale_line_is_flagged_inline(project):
    store, repo = project
    _save(store, _continuity(TRUE_STATE))

    clean = rederive_continuity(store)
    assert [r.status for r in clean.results] == ["ok", "ok", "ok", "judged"]

    (repo / "app.py").write_text('VERSION = "1.4.0"\n')  # the planted drift
    report = rederive_continuity(store)
    assert [r.status for r in report.results] == ["stale", "ok", "ok", "judged"]
    flagged = [l for l in report.text.split("\n") if "working tree is at 1.3.0" in l]
    assert flagged == [TRUE_STATE[0] + "  ⚠ STALE (now '0', claimed '1')"]
    assert report.ref and len(report.ref) == 40  # one pinned commit per load

    # The loaded text goes back through a save: the stale line is reported,
    # not refused, and no load-time verdict is persisted.
    store.record("another session", "observation")
    prepare_wrap(store)
    result = validated_save_continuity(store, report.text)
    assert result["stale_state"] == ["line 7: ⚠ STALE (now '0', claimed '1')"]
    assert "⚠" not in store.load_continuity() and "✓" not in store.load_continuity()


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


def test_crafted_malicious_lines_are_refused(project, tmp_path):
    store, repo = project
    canary = tmp_path / "pwned"
    (repo / "link_out").symlink_to("/etc/passwd")
    (repo / ".env").write_text("API_KEY=sk-live-secret\n")  # untracked
    (repo / "env_link").symlink_to(".env")
    subprocess.run([*_GIT, "add", "env_link"], cwd=repo, check=True)
    subprocess.run([*_GIT, "commit", "-qm", "link"], cwd=repo, check=True)
    evil_lines = [f"- evil{i} [derive: {c.format(canary=canary)}]" for i, c in enumerate(EVIL)]
    evil_text = _continuity(TRUE_STATE + evil_lines)

    # At save: refused before anything is written, and every line is named.
    # Lexically refusable lines fail the static gate; the rest (symlink,
    # untracked file) fail when the opted-in store runs them.
    def lexically_refused(line):
        try:
            validate_command(parse_annotation(line).command)
        except DeriveRefused:
            return True
        return False

    store.record("a session episode", "observation")
    prepare_wrap(store)
    static = [l for l in evil_lines if lexically_refused(l)]
    runtime = [l for l in evil_lines if l not in static]
    with pytest.raises(ValueError, match="State section refused") as exc:
        validated_save_continuity(store, _continuity(TRUE_STATE + static))
    assert str(exc.value).count(": refused: ") == len(static)
    assert runtime  # the symlink and untracked-file lines pass the lexical gate
    with pytest.raises(ValueError, match="derive commands failed") as exc:
        validated_save_continuity(store, _continuity(TRUE_STATE + runtime))
    assert str(exc.value).count("REFUSED") == len(runtime)
    assert store.load_continuity() is None

    # At load: a memory file edited behind the save gate still runs nothing.
    validated_save_continuity(store, _continuity(TRUE_STATE))
    Path(store.continuity_path).write_text(evil_text)
    report = rederive_continuity(store)
    by_line = {r.line: r.status for r in report.results}
    assert [by_line[l] for l in evil_lines] == ["refused"] * len(EVIL)
    assert not canary.exists()
    tags = subprocess.run(["git", "tag", "-l", "pwned"], cwd=repo, capture_output=True, text=True)
    assert tags.stdout == ""
    assert "sk-live" not in report.text

    # A repository-configured mailmap outside the root is never read (codex L3):
    # plain log output shows the recorded author, not the mapped one.
    outside = tmp_path / "outside.mailmap"
    outside.write_text("OUTSIDE-SECRET <out@x> <t@t>\n")
    subprocess.run(["git", "config", "mailmap.file", str(outside)], cwd=repo, check=True)
    Path(store.continuity_path).write_text(
        _continuity(TRUE_STATE + ["- author [derive: git log -1 => nope]"])
    )
    assert "OUTSIDE-SECRET" not in rederive_continuity(store).text

    # A repository whose object store points outside the root runs no git.
    alt = repo / ".git" / "objects" / "info" / "alternates"
    alt.write_text(str(tmp_path / "elsewhere.git" / "objects") + "\n")
    Path(store.continuity_path).write_text(_continuity(TRUE_STATE))
    git_lines = [r for r in rederive_continuity(store).results if "git " in r.line]
    assert git_lines and all(r.status == "error" and "alternates" in r.detail for r in git_lines)
    alt.unlink()

    # An untracked secret is never read: the claim is judged stale, unread.
    Path(store.continuity_path).write_text(
        _continuity(TRUE_STATE + ["- key [derive: grep -c API .env => 1]"])
    )
    untracked = rederive_continuity(store).results[-1]
    assert (untracked.status, untracked.detail) == ("stale", "a path is not a regular file git tracks in the root")

    # A root inside a larger repo: git must not discover the parent (L1, L2).
    (repo / "sub").mkdir()
    inner = Store(tmp_path / "store" / "inner.db", project_name="S", section_schema=PROJECT_SCHEMA)
    allow_store(inner.path, repo / "sub")
    inner.record("e", "observation")
    prepare_wrap(inner)
    with pytest.raises(ValueError, match="derive commands failed"):
        validated_save_continuity(
            inner, _continuity(["- parent read [derive: git cat-file -e HEAD:app.py]"])
        )
    inner.close()
