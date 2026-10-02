"""spore-1300: the rightmost-opener grammar read a derive line as a judged one.

Reproduced before the fix [run 2026-10-02]: the line
``- unmerged [derive: grep -Fqc '[judged:' README.md => 1]`` parsed as a
judged line, so the validator accepted it and save never executed the derive.
The grammar is bounded rather than the string: a line with more than one
opener is refused, in both directions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import warnings

import pytest

from anneal_memory import Store, prepare_wrap, validated_save_continuity
from anneal_memory.rederive import (
    DeriveRefused,
    allow_store,
    parse_annotation,
    rederive_text,
    trusted_roots,
)
from anneal_memory.schema import PROJECT_SCHEMA

pytestmark = [
    pytest.mark.skipif(shutil.which("git") is None, reason="needs git"),
    pytest.mark.skipif(os.name != "posix", reason="re-derive is POSIX-only by design"),
]

_SPORE_LINE = "- unmerged [derive: grep -Fqc '[judged:' README.md => 1]"
# The same class from the other side, and from the claim: a judged body that
# holds a derive opener would run as a derive; a claim that names an opener
# would move the annotation.
_SIBLINGS = [
    "- vouched [judged: me, today, against a [derive: grep -c x README.md => 1] run]",
    "- the [judged: marker is documented [derive: grep -c x README.md => 1]",
    "- two real markers [judged: me, today, x] [derive: grep -c x README.md => 1]",
    # a malformed derive opener beside a real one was swallowed into its body (codex, L1 2026-10-02)
    "- unmerged [derive@bad label: grep -Fqc '[judged:' README.md => 1]",
    "- unmerged [judged: me, today] [derive : grep -c marker README.md => 1]",
    "- unmerged [judged: me, today] [Derive: grep -c marker README.md => 1]",
    "- unmerged [judged: me, today] [ｄerive: grep -c marker README.md => 1]",
    "- unmerged [judged: me, today] [derive\u200b: grep -c marker README.md => 1]",
    "- unmerged [judged: me, today] [de\u200erive: grep -c marker README.md => 1]",  # bidi mark
    "- unmerged [judged: me, today] [de\u00adrive: grep -c marker README.md => 1]",  # soft hyphen
    "- unmerged [judged: me, today] [de\u034frive: grep -c marker README.md => 1]",  # joiner
    "- unmerged [judged: me, today] [derive\ufe00: grep -c marker README.md => 1]",  # selector
    "- unmerged [judged: me, today] [derive grep -c marker README.md => 1]",  # no colon
]


def _continuity(state: list[str]) -> str:
    return "\n".join([
        "# S — Memory (v1)", "",
        "## Plan", "- p", "",
        "## State", *state, "",
        "## Decisions", "- d", "",
        "## Open", "- o", "",
        "## Lessons", "- l", "",
        "## History", "- h", "",
    ])


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("ANNEAL_MEMORY_DERIVE_TRUST", str(tmp_path / "trust.json"))
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    for args in (["init", "-q"], ["add", "."], ["commit", "-qm", "i", "--allow-empty"]):
        (repo / "README.md").write_text("no marker here\n")
        subprocess.run([*git, *args], cwd=repo, check=True, capture_output=True)
    s = Store(tmp_path / "store" / "m.db", project_name="S", section_schema=PROJECT_SCHEMA)
    allow_store(s.path, repo, visibility="public")
    s.record("e", "observation")
    yield s
    s.close()


def test_the_spore_line_is_refused_at_save_and_at_load(store):
    res = prepare_wrap(store)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=r"line \d+: refused: 2 annotation openers"):
            validated_save_continuity(store, _continuity([_SPORE_LINE]), wrap_token=res["wrap_token"])
    assert store.load_continuity() is None  # nothing written
    # a line already in a stored file is never read as judged either
    report = rederive_text(_continuity([_SPORE_LINE]), PROJECT_SCHEMA, trusted_roots(store.path))
    assert [r.status for r in report.results] == ["refused"]
    assert "judged" not in {r.status for r in report.results}


@pytest.mark.parametrize("line", _SIBLINGS)
def test_any_line_with_a_second_opener_is_refused(line):
    with pytest.raises(DeriveRefused, match="annotation openers"):
        parse_annotation(line)


def test_the_documented_bracket_spelling_still_parses_as_one_derive():
    a = parse_annotation("- no marker [derive: grep -c '[[]judged:' README.md => 0]")
    assert (a.kind, a.command, a.expected) == ("derive", "grep -c '[[]judged:' README.md", "0")
    # prose that merely starts with the keyword is not a look-alike
    b = parse_annotation("- [derived state] shipped [derive: test -e README.md]")
    assert (b.kind, b.command) == ("derive", "test -e README.md")


@pytest.mark.parametrize("sep", ["\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"])
def test_an_unannotated_claim_cannot_ride_the_next_lines_annotation(store, sep):
    """L2 2026-10-02, reproduced at 610aa5f: the gate split on newline only, so
    this one State line was accepted as a single judged claim."""
    line = f"- unverified claim, no annotation{sep}- real [judged: me, now, x]"
    res = prepare_wrap(store)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="line terminator other than the newline"):
            validated_save_continuity(store, _continuity([line]), wrap_token=res["wrap_token"])
    assert store.load_continuity() is None
    assert parse_annotation("- a [judged: me, now, x]\r").kind == "judged"  # CRLF is not one
    # the same character in a NON-State line forges a "## State" heading for any
    # reader that splits on it (codex L3 r3), so the whole text is refused
    forged = _continuity(["- real [judged: me, now, x]"]).replace(
        "## Plan\n- p", f"## Plan\n- p{sep}## State\n- unverified claim, no annotation"
    )
    with pytest.raises(ValueError, match="line terminator other than the newline"):
        validated_save_continuity(store, forged, wrap_token=res["wrap_token"])
