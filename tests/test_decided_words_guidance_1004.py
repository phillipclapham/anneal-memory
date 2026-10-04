"""spore-1328: the Decisions guidance carries the operator-words rule.

A `[decided]` line carries the decider's own quoted words; a seat or desk gloss is a
`[judged: ...]` line and, if it stops work, names its scope. Pinned because a gloss
stored as `[decided]` was obeyed as a standing stop (flow deferral probe, 2026-10-03).
"""
from __future__ import annotations

from anneal_memory.continuity import _marker_reference


def _decisions_block() -> str:
    text = _marker_reference("2026-10-04")
    start = text.index("### Decisions")
    return " ".join(text[start:].split())


def test_decided_line_carries_the_deciders_own_words():
    block = _decisions_block()
    assert "carries the decider's own words, quoted" in block
    assert "[judged: <who>, <when>, <against what>]" in block


def test_a_stopping_judgement_names_its_scope():
    block = _decisions_block()
    assert "name exactly what it stops" in block
    assert "this round, this merge, this release" in block
