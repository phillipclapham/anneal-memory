#!/usr/bin/env python3
"""ACCEPTANCE TEST — a raised FlowScript salience prefix must not make a pattern vanish.

⚖ THIS FILE IS THE CRITERION AND IT IS HELD BY THE OPERATOR SIDE, NOT THE IMPLEMENTER.
Written before the fix and handed to an unattended seat as the definition of done. The seat
implements `anneal_memory/graduation.py`; it MUST NOT edit this file. An implementer that
writes its own acceptance test is not being tested.

THE DEFECT (spore-447, found at the 2026-07-31 EOD wrap, re-reproduced on disk 2026-08-10):
`detect_pattern_omissions` reports a pattern as OMITTED when the only thing that changed is
that its salience prefix was RAISED from `!!` to `!!!`. Deterministic repro at HEAD 25c03dd:

    prior:  - !! thing_one | 2x (2026-08-01) — felt prose
    new:    - !!! thing_one | 3x (2026-08-02) — felt prose
    detect_pattern_omissions(prior, new) -> [OmittedPattern(name='thing_one', prior_level=2)]

⚠ THE SPORE'S DIAGNOSIS IS SLIGHTLY WRONG AND THE REAL ONE IS WIDER — verify for yourself
rather than taking either account on faith. The spore says the detector "matches on something
that includes the salience marker", implying a mangled name. It does not. `_NAMED_PATTERN_RE`
(graduation.py ~:127) offers the alternation `!!|!|\\?|✓|\\*` in BOTH the zero-width branch (c)
and the optional-prefix group, and **there is no `!!!` branch at all**. So a `!!!` line does
not parse to a wrong name — it does not parse AT ALL, and the pattern is INVISIBLE:

    extract_pattern_names("- !! thing_one | 2x")   -> {'thing_one': 2}
    extract_pattern_names("- !!! thing_one | 3x")  -> {}        # the whole line vanishes

WHY THAT IS BIGGER THAN ONE FALSE POSITIVE: `extract_pattern_names` is the shared parser —
its own docstring says it scans "consistent with validate_graduations and
detect_stale_patterns". Every consumer of the `## Patterns` section is blind to a `!!!`
pattern, not just the omission detector. The omission report is merely where it became
visible, because that is the one guard against silently erasing Proven-tier evidence, and it
now emits noise on every re-ranking — a real omission can hide in that noise.

⛔ IMPLEMENT THE PROPERTY, NOT THE ONE FAILING CASE. The right fix is that **a run of one or
more `!` is a salience prefix**, not that `!!!` gets bolted on beside `!!`. `!!!!` must work
too — a test that only pins the reported case would leave the next rank-up to rediscover
this. Whatever construction is used, longest-match ordering matters: an alternation that
tries `!!` before `!!!` consumes two marks and then fails on the third.

⛔ NOT IN SCOPE and MUST NOT change: the omission guard's actual job. A pattern genuinely
dropped between wraps must STILL be reported (test below). A fix that silences the false
positive by weakening the guard trades a noisy check for a blind one, which is worse — this
is the only structural defence against an agent silently erasing load-bearing evidence from
always-loaded memory. Do not touch `min_level` semantics, the `OmittedPattern` shape, the
sort order, or any other detector.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from anneal_memory.graduation import (  # noqa: E402
    detect_pattern_omissions,
    extract_pattern_names,
)


def patterns(*lines: str) -> str:
    body = "\n".join(lines)
    return f"## Patterns\n\n{body}\n"


class TestParserSeesEverySalienceLevel(unittest.TestCase):
    """The shared parser must read a name at ANY salience, since every consumer uses it."""

    def test_single_and_double_still_parse(self):
        """Regression guard: the levels that already worked must keep working."""
        self.assertEqual(
            extract_pattern_names(patterns("- ! alpha | 1x (2026-08-01) — prose")),
            {"alpha": 1})
        self.assertEqual(
            extract_pattern_names(patterns("- !! beta | 2x (2026-08-01) — prose")),
            {"beta": 2})

    def test_triple_is_parsed_at_all(self):
        """THE DEFECT: a `!!!` line is currently invisible to the parser."""
        self.assertEqual(
            extract_pattern_names(patterns("- !!! gamma | 3x (2026-08-02) — prose")),
            {"gamma": 3},
            "a `!!!` pattern line must parse to its name and level like any other")

    def test_any_run_of_bangs_is_a_salience_prefix(self):
        """The PROPERTY, not the reported case. `!!!!` must work or the next rank-up
        rediscovers this bug."""
        self.assertEqual(
            extract_pattern_names(patterns("- !!!! delta | 4x (2026-08-02) — prose")),
            {"delta": 4})

    def test_marker_at_column_zero_also_handles_triple(self):
        """Branch (c) of the regex is a SEPARATE code path — the zero-width lookahead for a
        FlowScript marker at column 0, with no dash bullet. Fixing only the optional-prefix
        group leaves this one broken, and nothing else in the suite would say so."""
        self.assertEqual(
            extract_pattern_names(patterns("!!! epsilon | 3x (2026-08-02) — prose")),
            {"epsilon": 3})
        self.assertEqual(
            extract_pattern_names(patterns("!! zeta | 2x (2026-08-02) — prose")),
            {"zeta": 2})

    def test_other_markers_are_untouched(self):
        """The other FlowScript markers must not regress while `!` is generalised."""
        for marker, name in (("?", "eta"), ("✓", "theta"), ("*", "iota")):
            with self.subTest(marker=marker):
                self.assertEqual(
                    extract_pattern_names(patterns(f"- {marker} {name} | 2x (2026-08-02) — prose")),
                    {name: 2})

    def test_a_bang_run_is_not_mistaken_for_a_name(self):
        """A prefix with no identifier after it is not a pattern. Guards a fix that
        loosens the grammar so far that punctuation parses as a name."""
        self.assertEqual(extract_pattern_names(patterns("- !!! | 3x")), {})


class TestOmissionDetector(unittest.TestCase):
    PRIOR = patterns(
        "- !! thing_one | 2x (2026-08-01) — felt prose",
        "- ! thing_two | 1x (2026-08-01) — felt prose",
    )

    def test_unchanged_reports_nothing(self):
        self.assertEqual(detect_pattern_omissions(self.PRIOR, self.PRIOR), [])

    def test_raising_the_salience_prefix_is_not_an_omission(self):
        """THE BUG spore-447 IS ABOUT. Identical patterns, one re-ranked `!!` -> `!!!`."""
        raised = patterns(
            "- !!! thing_one | 3x (2026-08-02) — felt prose",
            "- ! thing_two | 1x (2026-08-01) — felt prose",
        )
        self.assertEqual(
            detect_pattern_omissions(self.PRIOR, raised), [],
            "thing_one is PRESENT and was promoted — reporting it as omitted is the "
            "false positive that makes this guard noisy on every re-ranking")

    def test_lowering_the_salience_prefix_is_not_an_omission_either(self):
        """The inverse must hold too: demotion is not disappearance."""
        lowered = patterns(
            "- ! thing_one | 2x (2026-08-02) — felt prose",
            "- ! thing_two | 1x (2026-08-01) — felt prose",
        )
        self.assertEqual(detect_pattern_omissions(self.PRIOR, lowered), [])

    def test_a_genuinely_dropped_pattern_is_STILL_reported(self):
        """⛔ THE GUARD MUST NOT BE NEUTERED. Silencing the false positive by weakening the
        detector trades a noisy check for a blind one — and this is the only structural
        defence against an agent silently erasing Proven-tier evidence from always-loaded
        memory. If this test ever goes green-by-deletion, the fix is wrong."""
        dropped = patterns("- ! thing_two | 1x (2026-08-01) — felt prose")
        result = detect_pattern_omissions(self.PRIOR, dropped)
        self.assertEqual(len(result), 1, "thing_one WAS dropped and must be reported")
        self.assertEqual(result[0].name, "thing_one")
        self.assertEqual(result[0].prior_level, 2)

    def test_a_dropped_triple_is_reported(self):
        """The other half of the same coin: once `!!!` parses, a dropped `!!!` pattern must
        be CAUGHT. A fix that merely makes the parser skip `!!!` lines everywhere would pass
        the false-positive tests above and silently stop guarding the highest tier."""
        prior = patterns("- !!! big_one | 3x (2026-08-01) — felt prose")
        after = patterns("- ! small | 1x (2026-08-02) — felt prose")
        result = detect_pattern_omissions(prior, after)
        self.assertEqual(len(result), 1,
                         "a dropped `!!!` pattern must be reported — the highest-salience "
                         "tier is exactly what this guard exists to protect")
        self.assertEqual(result[0].name, "big_one")
        self.assertEqual(result[0].prior_level, 3)

    def test_min_level_semantics_unchanged(self):
        """1x drops stay unreported (normal lifecycle), at any salience prefix."""
        prior = patterns("- !!! only_one | 1x (2026-08-01) — prose")
        self.assertEqual(detect_pattern_omissions(prior, patterns("- ! other | 1x")), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
