"""spore-676 (2026-10-07): a BARE graduation (no [evidence:] tag) at 4x and up was
matched by no regex, so it was neither validated nor demoted at any level. The bare
regex now uses the same 2-and-up atom as the cited one; AM-PRESERVE-BARE-PATH decides
hold vs demote exactly as it does at 2x/3x."""
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


@pytest.mark.parametrize("level", [4, 12, 21])
def test_cold_bare_line_at_4x_and_up_demotes_one_level(level):
    r = _run(f"- p | {level}x ({TODAY})", _lookup(level, "2026-09-20"))
    assert r.bare_demoted == 1
    assert f"p | {level - 1}x" in r.text


@pytest.mark.parametrize("level", [4, 12, 29])
def test_warm_bare_line_at_or_below_its_mark_is_held(level):
    r = _run(f"- p | {level}x ({TODAY})", _lookup(level, "2026-10-05"))
    assert r.bare_demoted == 0
    assert f"p | {level}x" in r.text and "(carried-forward)" in r.text


def test_bare_line_above_its_mark_is_not_held():
    r = _run(f"- p | 12x ({TODAY})", _lookup(4, "2026-10-05"))
    assert r.bare_demoted == 1 and "p | 11x" in r.text


def test_malformed_evidence_after_a_4x_plus_marker_is_reported_not_held():
    line = f'- p | 12x ({TODAY}) [provenance: x] [evidence: abcd1234 "why"]'
    r = _run(line, _lookup(12, "2026-10-05"))
    assert r.bare_demoted == 0 and line in r.text
    assert r.malformed_evidence_carries == ["p"]


def test_zero_padded_level_is_not_a_graduation():
    r = _run(f"- p | 04x ({TODAY})", _lookup(4, "2026-09-20"))
    assert r.bare_demoted == 0 and f"p | 04x ({TODAY})" in r.text
