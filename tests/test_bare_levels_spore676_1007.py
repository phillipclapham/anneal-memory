"""spore-676 (2026-10-07): a BARE graduation (no [evidence:] tag) at 4x and up was
matched by no regex, so it was neither validated nor demoted at any level. The bare
regex now uses the same 2-and-up atom as the cited one. Phill's ruling (A), the same
day: at every level, a line at or below its high-water mark is HELD whether warm or
cold; a cold one is dated back to its last grounding and flagged for the operator.
No-history and above-mark lines still demote."""
import pytest

from anneal_memory.graduation import validate_graduations

TODAY = "2026-10-07"


def _text(line):
    return f"## State\n.\n## Patterns\n{line}\n## Decisions\n.\n## Context\n.\n"


def _lookup(max_level, last_seen):
    return lambda name: {"max_level_reached": max_level, "last_explanation": "x",
                         "last_seen_at": f"{last_seen}T00:00:00Z", "last_wrap_id": None}


def _run(line, lookup):
    return validate_graduations(text=_text(line), valid_ids=set(), today=TODAY,
                                citations_seen=True, pattern_history_lookup=lookup)


@pytest.mark.parametrize("level", [2, 3, 4, 12, 21])
def test_cold_bare_line_at_or_below_its_mark_is_held_dated_back_and_flagged(level):
    r = _run(f"- p | {level}x ({TODAY})", _lookup(level, "2026-09-20"))
    assert r.bare_demoted == 0
    assert f"- p | {level}x (2026-09-20) (carried-forward)" in r.text
    assert [(c.name, c.cold) for c in r.carried_forward] == [("p", True)]


@pytest.mark.parametrize("level", [2, 12])
def test_no_history_still_demotes(level):
    r = _run(f"- p | {level}x ({TODAY})", lambda name: None)
    assert r.bare_demoted == 1 and f"p | {level - 1}x" in r.text


@pytest.mark.parametrize("level", [4, 12, 29])
def test_warm_bare_line_at_or_below_its_mark_is_held(level):
    r = _run(f"- p | {level}x ({TODAY})", _lookup(level, "2026-10-05"))
    assert r.bare_demoted == 0
    assert f"p | {level}x ({TODAY}) (carried-forward)" in r.text
    assert [c.cold for c in r.carried_forward] == [False]


@pytest.mark.parametrize("seen", ["2026-10-05", "2026-09-20"])
def test_bare_line_above_its_mark_still_demotes(seen):
    r = _run(f"- p | 12x ({TODAY})", _lookup(4, seen))
    assert r.bare_demoted == 1 and "p | 11x" in r.text


def test_malformed_evidence_after_a_4x_plus_marker_is_reported_not_held():
    line = f'- p | 12x ({TODAY}) [provenance: x] [evidence: abcd1234 "why"]'
    r = _run(line, _lookup(12, "2026-10-05"))
    assert r.bare_demoted == 0 and line in r.text
    assert r.malformed_evidence_carries == ["p"]


def test_zero_padded_level_is_not_a_graduation():
    r = _run(f"- p | 04x ({TODAY})", _lookup(4, "2026-09-20"))
    assert r.bare_demoted == 0 and f"p | 04x ({TODAY})" in r.text


# --- L3 1007 round 1 ---------------------------------------------------------------

def test_irregular_spacing_keeps_the_date_whole():
    """glm's HIGH, refuted by this run: the cold hold edits only the date span."""
    for line, want in ((f"- p |  12x  ({TODAY})", "- p |  12x  (2026-09-20) (carried-forward)"),
                       (f"- p|12x ({TODAY})  — note", "- p|12x (2026-09-20) (carried-forward) — note"),
                       (f"- p | 12x\t({TODAY})", "- p | 12x\t(2026-09-20) (carried-forward)")):
        r = _run(line, _lookup(12, "2026-09-20"))
        assert want in r.text


@pytest.mark.parametrize("seen", ["2026-10-05", "2026-09-20"])  # warm and cold
def test_a_hold_never_raises_a_line_above_its_prior_level(seen):
    """complement MED: an eroded pattern (1x in the file being replaced) re-asserted
    bare at its old 5x peak is a re-inflation, not a hold."""
    r = validate_graduations(text=_text(f"- p | 5x ({TODAY})"), valid_ids=set(),
                             today=TODAY, citations_seen=True,
                             pattern_history_lookup=_lookup(5, seen), prior_levels={"p": 1})
    assert r.bare_demoted == 1 and "p | 4x" in r.text and r.carried_forward == []
    held = validate_graduations(text=_text(f"- p | 5x ({TODAY})"), valid_ids=set(),
                                today=TODAY, citations_seen=True,
                                pattern_history_lookup=_lookup(5, seen), prior_levels={"p": 5})
    assert held.bare_demoted == 0 and len(held.carried_forward) == 1
