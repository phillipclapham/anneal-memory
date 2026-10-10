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
    # not a graduation, and (L3 r5) no longer left standing: the one normalizer cuts it
    assert r.bare_demoted == 0 and f"p | 1x ({TODAY}) (level-capped)" in r.text


# --- L3 1007 round 1 ---------------------------------------------------------------

def test_irregular_spacing_keeps_the_date_whole():
    """glm's HIGH, refuted by this run: the cold hold edits only the date span."""
    for line, want in ((f"- p |  12x  ({TODAY})", "- p |  12x  (2026-09-20) (carried-forward)"),
                       (f"- p|12x ({TODAY})  — note", "- p|12x (2026-09-20) (carried-forward) — note"),
                       (f"- p | 12x\t({TODAY})", "- p | 12x\t(2026-09-20) (carried-forward)")):
        r = _run(line, _lookup(12, "2026-09-20"))
        assert want in r.text


@pytest.mark.parametrize("seen", ["2026-10-05", "2026-09-20"])  # warm and cold
def test_a_hold_keeps_the_prior_level_whatever_the_composer_wrote(seen):
    """L3 1007 r1+r2 (complement, codex): a held line's level is derived from the file it
    replaces. An eroded pattern (2x there) re-asserted bare at its old 5x peak is held
    at 2x, not demoted one level a wrap toward its peak; a name absent from that file is
    a new claim and is not held at all."""
    r = validate_graduations(text=_text(f"- p | 5x ({TODAY})"), valid_ids=set(),
                             today=TODAY, citations_seen=True,
                             pattern_history_lookup=_lookup(5, seen), prior_levels={"p": 2})
    assert r.bare_demoted == 0 and [c.held_level for c in r.carried_forward] == [2]
    assert "- p | 2x (" in r.text and "(carried-forward)" in r.text
    absent = validate_graduations(text=_text(f"- p | 5x ({TODAY})"), valid_ids=set(),
                                  today=TODAY, citations_seen=True,
                                  pattern_history_lookup=_lookup(5, seen),
                                  prior_levels={"other": 5})
    assert absent.bare_demoted == 1 and absent.carried_forward == []
    same = validate_graduations(text=_text(f"- p | 5x ({TODAY})"), valid_ids=set(),
                                today=TODAY, citations_seen=True,
                                pattern_history_lookup=_lookup(5, seen), prior_levels={"p": 5})
    assert same.bare_demoted == 0 and [c.held_level for c in same.carried_forward] == [5]


def test_a_cited_hold_also_keeps_the_prior_level():
    text = _text(f'- p | 5x ({TODAY}) [evidence: deadbeef "gone"]')
    r = validate_graduations(text=text, valid_ids={"abcd1234"}, today=TODAY,
                             node_content_map={"abcd1234": "x"}, citations_seen=True,
                             pattern_history_lookup=_lookup(5, "2026-10-05"),
                             prior_levels={"p": 3})
    assert [c.held_level for c in r.carried_forward] == [3] and "- p | 3x (" in r.text


def test_rewarm_is_held_at_the_saved_level_and_a_crystal_level_is_never_a_prior(tmp_path):
    """Through the real save, as Phill ruled 2026-10-08 13:17: the held level comes from
    the store's pattern_levels record, never the crystal store (it is caller-writable).
    A pattern the store saved at 5x, dropped and re-added is held at 5x; a crystal
    re-added to the working set is a new claim and re-earns from 1x, as is a pattern
    in no record at all."""
    from anneal_memory import Store, prepare_wrap, validated_save_continuity
    from anneal_memory.crystal import CrystalStore
    from tests.prior_seed import seed_prior_levels
    s = Store(tmp_path / "m.db", project_name="t")
    cs = CrystalStore(tmp_path / "m.crystal.json")
    base = "## State\n.\n\n## Patterns\n{p}\n\n## Decisions\n.\n\n## Context\n.\n"
    try:
        for n in ("kept", "rewarmed", "invented"):
            s.upsert_pattern_history(n, level=5, explanation="x", seen_at="2026-10-05")
        cs.crystallize(name="rewarmed", level=5, explanation="x")
        seed_prior_levels(s, {"kept": 5})
        seed = s.record("the seed pattern grounded substrate observation", "observation")
        cite = f'- seed | 2x (2026-10-06) [evidence: {seed.id[:8]} "seed pattern grounded substrate"]'
        for day, pats in (("2026-10-06", "- kept | 5x (2026-10-06)\n" + cite),
                          ("2026-10-07", "- other | 1x (2026-10-07)"),
                          ("2026-10-08", "- kept | 5x (2026-10-08)\n"
                                         "- rewarmed | 5x (2026-10-08)\n"
                                         "- invented | 5x (2026-10-08)")):
            s.record(f"episode {day}: a substrate observation about the topic.",
                     "observation")
            token = prepare_wrap(s, max_chars=40000)["wrap_token"]
            r = validated_save_continuity(s, base.format(p=pats), today=day,
                                      wrap_token=token, crystal_store=cs)
        saved = s.load_continuity()
        assert s.saved_pattern_levels()[("name", "kept")] == 5
        assert "- kept | 5x (2026-10-08) (carried-forward)" in saved
        assert r["bare_demoted"] == 2  # rewarmed and invented: no saved record, new claims
        for name in ("rewarmed", "invented"):
            line = next(ln for ln in saved.splitlines() if ln.startswith(f"- {name} |"))
            assert line.startswith(f"- {name} | 1x"), line
        assert cs.get("rewarmed")["level"] == 5  # the crystal itself is untouched
    finally:
        s.close()
