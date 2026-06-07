"""Tests for graduation and demotion logic."""

import pytest

from anneal_memory.graduation import (
    CarriedForward,
    CrossSessionCollision,
    GraduationResult,
    OmittedPattern,
    _NAMED_PATTERN_RE,
    _carryforward_decision,
    _is_verbatim_preservation,
    _meaningful_word_overlap,
    _pattern_summary,
    check_explanation_overlap,
    detect_citation_gaming,
    detect_pattern_omissions,
    detect_stale_patterns,
    extract_pattern_names,
    extract_pattern_summaries,
    validate_graduations,
)
from anneal_memory.schema import FLOW_SCHEMA, graduating_headings

# -- Test fixtures --

SAMPLE_CONTINUITY = """# TestAgent — Memory (v1)

## State
Working on database architecture.

## Patterns
{database:
  thought: ACID compliance matters | 2x (2026-03-31) [evidence: abc12345 "PostgreSQL chosen for ACID compliance"]
  thought: connection pooling is bottleneck | 1x (2026-03-31)
  ? horizontal scaling | 1x (2026-03-30)
}

## Decisions
[decided(rationale: "reliability over speed", on: "2026-03-31")] Use PostgreSQL

## Context
Built the database layer. Chose PostgreSQL for ACID compliance.
"""

VALID_IDS = {"abc12345", "def67890", "11111111"}
NODE_CONTENT_MAP = {
    "abc12345": "We chose PostgreSQL for its strong ACID compliance guarantees",
    "def67890": "The API latency improved after adding connection pooling",
    "11111111": "Reviewed horizontal scaling options for the database",
}
TODAY = "2026-03-31"


# -- validate_graduations --


class TestValidateGraduations:
    def test_valid_citation_passes(self):
        result = validate_graduations(
            SAMPLE_CONTINUITY, VALID_IDS, TODAY, NODE_CONTENT_MAP
        )
        assert result.validated == 1
        assert result.demoted == 0

    def test_invalid_id_demotes(self):
        text = SAMPLE_CONTINUITY.replace("abc12345", "ffffffff")
        result = validate_graduations(text, VALID_IDS, TODAY, NODE_CONTENT_MAP)
        assert result.validated == 0
        assert result.demoted == 1
        assert "(ungrounded)" in result.text

    def test_demoted_level_decreases(self):
        text = """## Patterns
  thought: test | 2x (2026-03-31) [evidence: ffffffff "bad cite"]
"""
        result = validate_graduations(text, VALID_IDS, TODAY)
        assert "| 1x" in result.text
        assert "| 2x" not in result.text

    def test_3x_demotes_to_2x(self):
        text = """## Patterns
  thought: test | 3x (2026-03-31) [evidence: ffffffff "bad cite"]
"""
        result = validate_graduations(text, VALID_IDS, TODAY)
        assert "| 2x" in result.text
        assert "| 3x" not in result.text

    def test_carried_forward_not_validated(self):
        """Patterns from previous days pass through unchanged."""
        text = """## Patterns
  thought: old pattern | 2x (2026-03-29) [evidence: ffffffff "old evidence"]
"""
        result = validate_graduations(text, VALID_IDS, TODAY)
        # Should NOT be demoted — it's from a previous day
        assert result.validated == 0
        assert result.demoted == 0
        assert result.skipped_non_today == 1
        assert "(ungrounded)" not in result.text

    def test_only_patterns_section_checked(self):
        """Graduation markers outside ## Patterns are ignored."""
        text = """## State
  thought: state marker | 2x (2026-03-31) [evidence: ffffffff "bad"]

## Patterns
  thought: real pattern | 1x (2026-03-31)

## Decisions
Nothing.

## Context
Nothing.
"""
        result = validate_graduations(text, VALID_IDS, TODAY)
        assert result.validated == 0
        assert result.demoted == 0

    def test_explanation_overlap_check(self):
        """Explanation must reference actual episode content."""
        text = """## Patterns
  thought: test | 2x (2026-03-31) [evidence: abc12345 "completely unrelated gibberish xyz"]
"""
        result = validate_graduations(
            text, VALID_IDS, TODAY, NODE_CONTENT_MAP
        )
        # "completely unrelated gibberish xyz" doesn't overlap with episode content
        assert result.demoted == 1

    def test_explanation_with_overlap_passes(self):
        text = """## Patterns
  thought: test | 2x (2026-03-31) [evidence: abc12345 "PostgreSQL compliance validated"]
"""
        result = validate_graduations(
            text, VALID_IDS, TODAY, NODE_CONTENT_MAP
        )
        assert result.validated == 1
        assert result.demoted == 0

    def test_no_content_map_skips_overlap_check(self):
        """Without content map, only ID validity is checked."""
        text = """## Patterns
  thought: test | 2x (2026-03-31) [evidence: abc12345 "anything goes here"]
"""
        result = validate_graduations(text, VALID_IDS, TODAY)
        assert result.validated == 1

    def test_multiple_cited_ids(self):
        text = """## Patterns
  thought: test | 2x (2026-03-31) [evidence: abc12345, def67890 "PostgreSQL ACID compliance chosen"]
"""
        result = validate_graduations(
            text, VALID_IDS, TODAY, NODE_CONTENT_MAP
        )
        assert result.validated == 1

    def test_citation_reuse_tracking(self):
        text = """## Patterns
  thought: one | 2x (2026-03-31) [evidence: abc12345 "PostgreSQL compliance"]
  thought: two | 2x (2026-03-31) [evidence: abc12345 "PostgreSQL compliance again"]
  thought: three | 2x (2026-03-31) [evidence: abc12345 "PostgreSQL compliance thrice"]
"""
        result = validate_graduations(
            text, VALID_IDS, TODAY, NODE_CONTENT_MAP
        )
        assert result.citation_reuse_max == 3

    def test_bare_graduation_ignored_without_citations_seen(self):
        """Bare graduations pass through when citations_seen is False."""
        text = """## Patterns
  thought: bare pattern | 2x (2026-03-31)
"""
        result = validate_graduations(text, VALID_IDS, TODAY, citations_seen=False)
        assert result.bare_demoted == 0

    def test_bare_graduation_demoted_with_citations_seen(self):
        """Bare graduations are demoted when citations_seen is True."""
        text = """## Patterns
  thought: bare pattern | 2x (2026-03-31)
"""
        result = validate_graduations(text, VALID_IDS, TODAY, citations_seen=True)
        assert result.bare_demoted == 1
        assert "(needs-evidence)" in result.text
        # Demotion actually rewrote the level — guard against the
        # Diogenes 2026-05-22 LOW where the counter incremented while
        # the text retained the old level. Required so future regression
        # in the level-rewrite path can't sneak past a counter-only check.
        assert "| 1x" in result.text
        assert "| 2x" not in result.text

    def test_bare_graduation_demoted_no_space_variant(self):
        """Bare graduations with no space after pipe (|2x) demote correctly.

        Diogenes 2026-05-22 LOW regression test. `_BARE_GRADUATION_RE`
        accepts `\\|\\s*Nx` (space-optional); the prior literal replace
        only matched the single-space form, so `|2x` no-op'd the rewrite
        while incrementing `bare_demoted`. State mismatch fixed by
        mirroring the v0.3.3 `_demote_line` regex substitution.
        """
        text = """## Patterns
  thought: bare pattern |2x (2026-03-31)
"""
        result = validate_graduations(text, VALID_IDS, TODAY, citations_seen=True)
        assert result.bare_demoted == 1
        assert "(needs-evidence)" in result.text
        assert "| 1x" in result.text
        # The original `|2x` should be gone — both as `|2x` and as `| 2x`.
        assert "|2x" not in result.text
        assert "| 2x" not in result.text

    def test_1x_patterns_untouched(self):
        """1x patterns don't need evidence — they're new observations."""
        text = """## Patterns
  thought: new observation | 1x (2026-03-31)
"""
        result = validate_graduations(text, VALID_IDS, TODAY)
        assert result.validated == 0
        assert result.demoted == 0
        assert "1x" in result.text


# -- check_explanation_overlap --


class TestExplanationOverlap:
    def test_overlapping_words_pass(self):
        assert check_explanation_overlap(
            "PostgreSQL chosen for compliance",
            "We chose PostgreSQL for its strong ACID compliance guarantees"
        ) is True

    def test_single_word_overlap_insufficient(self):
        """Single word overlap is too easy to game — requires 2+."""
        assert check_explanation_overlap(
            "database performance",
            "The database handles ACID transactions efficiently"
        ) is False  # only "database" overlaps

    def test_no_overlap_fails(self):
        assert check_explanation_overlap(
            "completely unrelated gibberish",
            "The database handles ACID transactions efficiently"
        ) is False

    def test_stop_words_excluded(self):
        """Stop words don't count as overlap."""
        assert check_explanation_overlap(
            "the is a for",  # all stop words
            "the database is a good choice for the project"
        ) is False

    def test_short_words_excluded(self):
        """Words <= 2 chars don't count."""
        assert check_explanation_overlap(
            "db ok",
            "The db is ok for now"
        ) is False

    def test_case_insensitive(self):
        assert check_explanation_overlap(
            "POSTGRESQL Compliance",
            "postgresql compliance testing"
        ) is True

    def test_empty_explanation(self):
        assert check_explanation_overlap("", "some content") is False

    def test_empty_content(self):
        assert check_explanation_overlap("some explanation", "") is False


# -- detect_stale_patterns --


class TestDetectStalePatterns:
    def test_finds_stale_patterns(self):
        text = """## Patterns
  thought: old pattern | 1x (2026-03-20)
  thought: fresh pattern | 1x (2026-03-31)
"""
        stale = detect_stale_patterns(text, "2026-03-31", staleness_days=7)
        assert len(stale) == 1
        assert stale[0].days_stale == 11
        assert "old pattern" in stale[0].content

    def test_no_stale_patterns(self):
        text = """## Patterns
  thought: fresh | 1x (2026-03-30)
"""
        stale = detect_stale_patterns(text, "2026-03-31", staleness_days=7)
        assert len(stale) == 0

    def test_exactly_at_threshold(self):
        text = """## Patterns
  thought: borderline | 1x (2026-03-24)
"""
        stale = detect_stale_patterns(text, "2026-03-31", staleness_days=7)
        assert len(stale) == 1
        assert stale[0].days_stale == 7

    def test_only_patterns_section(self):
        """Stale markers in other sections are ignored."""
        text = """## State
  thought: state old | 1x (2020-01-01)

## Patterns
  thought: patterns fresh | 1x (2026-03-31)

## Decisions
Nothing.

## Context
Nothing.
"""
        stale = detect_stale_patterns(text, "2026-03-31")
        assert len(stale) == 0

    def test_reports_correct_level(self):
        text = """## Patterns
  thought: evolved | 3x (2026-03-20)
"""
        stale = detect_stale_patterns(text, "2026-03-31")
        assert stale[0].level == 3

    def test_custom_staleness_threshold(self):
        text = """## Patterns
  thought: recent-ish | 1x (2026-03-28)
"""
        # 3 days old, threshold of 2 -> stale
        stale = detect_stale_patterns(text, "2026-03-31", staleness_days=2)
        assert len(stale) == 1

        # 3 days old, threshold of 5 -> not stale
        stale = detect_stale_patterns(text, "2026-03-31", staleness_days=5)
        assert len(stale) == 0


# -- detect_citation_gaming --


class TestDetectCitationGaming:
    def test_no_gaming(self):
        counts = {"abc12345": 1, "def67890": 2}
        assert detect_citation_gaming(counts) == []

    def test_gaming_detected(self):
        counts = {"abc12345": 5, "def67890": 1}
        sus = detect_citation_gaming(counts)
        assert "abc12345" in sus
        assert "def67890" not in sus

    def test_custom_threshold(self):
        counts = {"abc12345": 2}
        assert detect_citation_gaming(counts, threshold=2) == ["abc12345"]
        assert detect_citation_gaming(counts, threshold=3) == []

    def test_empty_counts(self):
        assert detect_citation_gaming({}) == []


# -- extract_pattern_names + detect_pattern_omissions --
#
# These two primitives close the silent-omission gap surfaced by
# Bold Stand Phase 1b probe #1 (2026-05-21). An adversarial agent
# dropped two legitimate Proven-tier patterns across three drift
# sessions by simply not carrying them forward in the wrap text;
# validate_graduations had no visibility into them (it only sees
# patterns the agent wrote INTO the new continuity). The primitives
# below let validated_save_continuity diff the prior continuity
# against the new one and surface 2x+ patterns that disappeared.


class TestExtractPatternNames:
    def test_extracts_names_and_max_levels(self):
        text = """## State
foo.
## Patterns
- alpha_pattern | 3x (2026-05-20) [evidence: abc12345 "explanation"]
- beta_pattern | 2x (2026-05-20) [evidence: def67890 "explanation"]
- gamma_pattern | 1x (2026-05-20) [evidence: 11112222 "explanation"]
## Decisions
- something happened.
## Context
- something else.
"""
        result = extract_pattern_names(text)
        assert result == {
            "alpha_pattern": 3,
            "beta_pattern": 2,
            "gamma_pattern": 1,
        }

    def test_only_patterns_section(self):
        # A line that looks like a pattern but lives in ## Decisions
        # must NOT be picked up — that would mistake free-form decision
        # prose for a graduated pattern.
        text = """## State
foo.
## Patterns
- alpha_pattern | 2x (2026-05-20) [evidence: abc12345 "x"]
## Decisions
- decided_thing | 3x (2026-05-20) — pattern-shaped decision text.
## Context
- contextual.
"""
        result = extract_pattern_names(text)
        assert result == {"alpha_pattern": 2}
        assert "decided_thing" not in result

    def test_empty_text(self):
        assert extract_pattern_names("") == {}

    def test_no_patterns_section(self):
        text = """## State
foo.
## Decisions
- something.
"""
        assert extract_pattern_names(text) == {}

    def test_duplicate_name_takes_max(self):
        # If the same pattern name appears at multiple levels (malformed
        # continuity), highest level wins. This is defensive — the
        # canonical write path doesn't produce duplicates, but external
        # editors might.
        text = """## Patterns
- alpha_pattern | 2x (2026-05-20) [evidence: abc12345 "x"]
- alpha_pattern | 3x (2026-05-21) [evidence: def67890 "y"]
"""
        result = extract_pattern_names(text)
        assert result == {"alpha_pattern": 3}

    def test_ignores_unnamed_pattern_lines(self):
        # Free-form bullets that happen to contain ``| Nx`` but don't
        # match the operator-style identifier grammar must NOT be
        # captured as patterns. This is the discriminator that lets
        # the section coexist with continuity-internal documentation.
        text = """## Patterns
- alpha_pattern | 2x (2026-05-20) [evidence: abc12345 "x"]
- this is some prose that mentions | 2x in passing.
"""
        result = extract_pattern_names(text)
        assert result == {"alpha_pattern": 2}


class TestDetectPatternOmissions:
    PRIOR = """## State
prior.
## Patterns
- alpha_proven_3x | 3x (2026-05-20) [evidence: aaa11111 "explanation"]
- beta_proven_2x | 2x (2026-05-20) [evidence: bbb22222 "explanation"]
- gamma_developing_1x | 1x (2026-05-20) [evidence: ccc33333 "explanation"]
## Decisions
- decision.
## Context
- context.
"""

    def test_no_omissions_when_all_carried_forward(self):
        new = """## State
new.
## Patterns
- alpha_proven_3x | 3x (2026-05-20) [evidence: aaa11111 "x"]
- beta_proven_2x | 2x (2026-05-20) [evidence: bbb22222 "x"]
## Decisions
- decision.
## Context
- context.
"""
        result = detect_pattern_omissions(self.PRIOR, new)
        assert result == []

    def test_detects_3x_dropout(self):
        new = """## State
new.
## Patterns
- beta_proven_2x | 2x (2026-05-20) [evidence: bbb22222 "x"]
## Decisions
- decision.
## Context
- context.
"""
        result = detect_pattern_omissions(self.PRIOR, new)
        assert len(result) == 1
        assert isinstance(result[0], OmittedPattern)
        assert result[0].name == "alpha_proven_3x"
        assert result[0].prior_level == 3

    def test_detects_multiple_dropouts_sorted_by_level_desc(self):
        new = """## State
new.
## Patterns
- gamma_developing_1x | 1x (2026-05-20) [evidence: ccc33333 "x"]
## Decisions
- decision.
## Context
- context.
"""
        result = detect_pattern_omissions(self.PRIOR, new)
        assert len(result) == 2
        # Sort order: highest prior_level first
        assert result[0].name == "alpha_proven_3x"
        assert result[0].prior_level == 3
        assert result[1].name == "beta_proven_2x"
        assert result[1].prior_level == 2

    def test_1x_dropout_not_surfaced_by_default(self):
        # Dropping a 1x ("Developing") pattern is normal lifecycle —
        # patterns that don't accrue further evidence in the next
        # session are supposed to drop out. min_level=2 makes this
        # the default behavior.
        new = """## State
new.
## Patterns
- alpha_proven_3x | 3x (2026-05-20) [evidence: aaa11111 "x"]
- beta_proven_2x | 2x (2026-05-20) [evidence: bbb22222 "x"]
## Decisions
- decision.
## Context
- context.
"""
        # gamma_developing_1x dropped — should NOT be flagged
        result = detect_pattern_omissions(self.PRIOR, new)
        assert result == []

    def test_min_level_1_catches_1x_drops(self):
        # Operators can opt into stricter auditing.
        new = """## State
new.
## Patterns
- alpha_proven_3x | 3x (2026-05-20) [evidence: aaa11111 "x"]
- beta_proven_2x | 2x (2026-05-20) [evidence: bbb22222 "x"]
## Decisions
- decision.
## Context
- context.
"""
        result = detect_pattern_omissions(self.PRIOR, new, min_level=1)
        assert len(result) == 1
        assert result[0].name == "gamma_developing_1x"
        assert result[0].prior_level == 1

    def test_pattern_demoted_not_omitted(self):
        # A pattern that demoted (3x -> 2x or 2x -> 1x) is still PRESENT
        # in the new continuity at a lower level — it must NOT be
        # flagged as omitted. Omission means absent at any level.
        new = """## State
new.
## Patterns
- alpha_proven_3x | 2x (2026-05-20) [evidence: aaa11111 "x"]
- beta_proven_2x | 1x (2026-05-20) [evidence: bbb22222 "x"]
## Decisions
- decision.
## Context
- context.
"""
        result = detect_pattern_omissions(self.PRIOR, new)
        assert result == []

    def test_empty_prior_returns_no_omissions(self):
        # First-wrap case: no prior continuity, nothing to be omitted.
        new = """## State
first.
## Patterns
- alpha_proven_3x | 3x (2026-05-20) [evidence: aaa11111 "x"]
## Decisions
- decision.
## Context
- context.
"""
        assert detect_pattern_omissions("", new) == []

    def test_empty_new_continuity_surfaces_all_proven_dropouts(self):
        # Pathological case: agent writes a continuity with no ## Patterns
        # section. All prior Proven-tier patterns are dropped.
        new = """## State
new state.
## Patterns
## Decisions
- decision.
## Context
- context.
"""
        result = detect_pattern_omissions(self.PRIOR, new)
        names = {op.name for op in result}
        assert names == {"alpha_proven_3x", "beta_proven_2x"}


# -- Cross-session sycophantic-accumulation defense --
#
# Closes the slow-drift attack from Bold Stand Phase 1b probe #1
# (2026-05-21): without this defense, a plausible-sounding pattern
# with plausible-sounding fresh episodes can ride 1x → 2x → 3x across
# three sessions even when each session's explanation simply rephrases
# the same claim. The cross-session check refuses graduations whose
# explanation overlaps too heavily with the pattern's prior-session
# explanation; multi-faceted legitimate evidence uses distinct
# vocabulary across sessions and passes cleanly.


class TestMeaningfulWordOverlap:
    def test_clean_overlap(self):
        a = "the database race condition fixed"
        b = "race condition in our database"
        # Stop words and short tokens excluded; remaining shared:
        # database, race, condition
        assert _meaningful_word_overlap(a, b) == {"database", "race", "condition"}

    def test_no_overlap(self):
        a = "first explanation about widgets"
        b = "second explanation discussing flanges"
        assert _meaningful_word_overlap(a, b) == {"explanation"}

    def test_excludes_stop_words(self):
        a = "this is the database race condition"
        b = "that was the database race condition"
        # "this", "is", "the", "that", "was" are stop words → excluded
        assert _meaningful_word_overlap(a, b) == {"database", "race", "condition"}

    def test_case_insensitive(self):
        a = "Database RACE Condition"
        b = "database race condition"
        assert _meaningful_word_overlap(a, b) == {"database", "race", "condition"}


class TestCrossSessionGraduationCheck:
    """End-to-end coverage of the cross-session-overlap demotion path
    in validate_graduations. Tests pass a stub pattern_history_lookup
    callable instead of wiring through Store — keeps the unit tests
    fast and independent of the SQLite layer."""

    def _stub_lookup(self, **patterns):
        """Build a pattern_history_lookup callback from kwargs:
        ``pattern_name=explanation_string`` → returns a history dict.
        Patterns not in the kwargs return None (no prior history)."""
        def _lookup(name):
            if name in patterns:
                return {
                    "max_level_reached": 2,
                    "last_explanation": patterns[name],
                    "last_seen_at": "2026-05-20T00:00:00Z",
                    "last_wrap_id": None,
                }
            return None
        return _lookup

    def test_no_history_allows_graduation(self):
        """First time a pattern reaches 2x: no prior history,
        graduation must pass cleanly."""
        text = """## State\nfirst graduation.\n## Patterns
- alpha | 2x (2026-05-21) [evidence: abc12345 "concurrency safety invariant"]
## Decisions
.
## Context
.
"""
        node_content = {"abc12345": "we made this concurrency safety invariant a hard rule"}
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-21",
            node_content_map=node_content,
            pattern_history_lookup=self._stub_lookup(),
        )
        assert result.validated == 1
        assert result.demoted == 0
        assert result.cross_session_collisions == []

    def test_distinct_vocabulary_across_sessions_passes(self):
        """Legitimate cross-session graduation: today's explanation
        uses different words than the prior session's. Both contain
        meaningful vocabulary, but overlap is below threshold."""
        text = """## State\ndistinct evidence.\n## Patterns
- alpha | 3x (2026-05-21) [evidence: abc12345 "throughput improvement after batch consolidation"]
## Decisions
.
## Context
.
"""
        # Prior session's evidence was about a different angle of the
        # same pattern. Overlap with today: zero meaningful words.
        prior = "concurrency safety invariant for the migration path"
        node_content = {"abc12345": "throughput improvement after batch consolidation in the pipeline"}
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-21",
            node_content_map=node_content,
            pattern_history_lookup=self._stub_lookup(alpha=prior),
        )
        assert result.validated == 1
        assert result.demoted == 0
        assert result.cross_session_collisions == []

    def test_high_overlap_across_sessions_demotes_with_marker(self):
        """The sycophantic shape: today's explanation rephrases the
        prior session's. The graduation gets demoted and marked
        ``(cross-session-overlap)`` so operators can distinguish this
        failure from the ungrounded path."""
        text = """## State\nsycophantic rephrase.\n## Patterns
- alpha | 3x (2026-05-21) [evidence: abc12345 "standup consensus decision quick agreement architectural"]
## Decisions
.
## Context
.
"""
        # Prior shares 5 meaningful words: standup, consensus,
        # decision, quick, agreement
        prior = "standup consensus decision architecture sync quick agreement"
        node_content = {
            "abc12345":
            "standup consensus decision quick agreement architectural meeting"
        }
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-21",
            node_content_map=node_content,
            pattern_history_lookup=self._stub_lookup(alpha=prior),
        )
        assert result.validated == 0
        assert result.demoted == 1
        assert "(cross-session-overlap)" in result.text
        assert len(result.cross_session_collisions) == 1
        coll = result.cross_session_collisions[0]
        assert isinstance(coll, CrossSessionCollision)
        assert coll.name == "alpha"
        assert coll.today_level == 3
        # Demotion lands at level - 1 in the saved text
        assert "alpha | 2x" in result.text
        # Overlap captures the meaningful words specifically
        assert "standup" in coll.overlap_words
        assert "consensus" in coll.overlap_words

    def test_cross_session_demote_keeps_citation_resolved_flag(self):
        """AM-WARN (v0.4.2) regression: a cross-session-overlap demote
        suppresses ``all_validated_ids`` for that line (the immune gate
        does not strengthen the graph from a suspected sycophantic
        re-graduation), but the cited id STILL resolved to a real store
        episode — the gate fired on the EXPLANATION, not the ids. So
        ``any_citation_resolved`` must be True even though
        ``all_validated_ids`` is empty. AM-WARN Signal A reads the former;
        reading the latter would misdiagnose a healthy immune-gate
        demotion as a dead-namespace graph (the false positive H1)."""
        text = """## State\nsycophantic rephrase.\n## Patterns
- alpha | 3x (2026-05-21) [evidence: abc12345 "standup consensus decision quick agreement architectural"]
## Decisions
.
## Context
.
"""
        prior = "standup consensus decision architecture sync quick agreement"
        node_content = {
            "abc12345":
            "standup consensus decision quick agreement architectural meeting"
        }
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-21",
            node_content_map=node_content,
            pattern_history_lookup=self._stub_lookup(alpha=prior),
        )
        # The line was cross-session-demoted...
        assert result.demoted == 1
        assert result.validated == 0
        assert len(result.cross_session_collisions) == 1
        # ...so the gated co-citation list is empty for it...
        assert result.all_validated_ids == []
        # ...but the cited id DID resolve, so the gate-independent signal
        # AM-WARN reads is True (no false "dead namespace" alarm).
        assert result.any_citation_resolved is True

    def test_threshold_at_boundary_2_words_passes(self):
        """At exactly 2 shared meaningful words, the default threshold
        of 3 does NOT trigger. Within-session check uses 2 as the
        floor for VALID grounding; cross-session check needs more
        than 2 to flag sycophancy."""
        text = """## State\nboundary case.\n## Patterns
- alpha | 2x (2026-05-21) [evidence: abc12345 "standup quick reactive observation pivot detection"]
## Decisions
.
## Context
.
"""
        prior = "standup quick alignment broad agreement consensus building"
        # Shared: standup, quick = 2 words — at threshold boundary,
        # default threshold=3 does NOT fire
        node_content = {
            "abc12345":
            "standup quick reactive observation pivot detection finding"
        }
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-21",
            node_content_map=node_content,
            pattern_history_lookup=self._stub_lookup(alpha=prior),
        )
        assert result.validated == 1
        assert result.cross_session_collisions == []

    def test_custom_threshold_2_catches_2_word_overlap(self):
        """Operators can tighten the cross-session threshold by
        passing a smaller value. Threshold=2 catches even mild
        trope-vocabulary reuse — most legitimate graduations would
        false-positive at this level, but the option exists."""
        text = """## State\nstrict threshold.\n## Patterns
- alpha | 2x (2026-05-21) [evidence: abc12345 "standup quick reactive observation pivot detection"]
## Decisions
.
## Context
.
"""
        prior = "standup quick alignment broad agreement consensus building"
        node_content = {
            "abc12345":
            "standup quick reactive observation pivot detection finding"
        }
        # COLD history (last_seen 20d before today) so the AM-PRESERVE warm+fresh-
        # specific exemption does not apply — this test isolates the operator-tunable
        # THRESHOLD mechanism (at threshold=2 even a 2-word overlap demotes), not
        # warmth. (The warm at-peak case is held — see TestVerbatimPreservation.)
        def _cold_lookup(name):
            if name == "alpha":
                return {"max_level_reached": 2, "last_explanation": prior,
                        "last_seen_at": "2026-05-01T00:00:00Z", "last_wrap_id": None}
            return None
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-21",
            node_content_map=node_content,
            pattern_history_lookup=_cold_lookup,
            cross_session_overlap_threshold=2,
        )
        assert result.validated == 0
        assert result.demoted == 1
        assert len(result.cross_session_collisions) == 1

    def test_no_lookup_skips_check_entirely(self):
        """When pattern_history_lookup is None, the cross-session
        check is skipped — preserves pre-Move-#3 behavior for library
        callers that don't wire history through."""
        text = """## State\nno lookup.\n## Patterns
- alpha | 3x (2026-05-21) [evidence: abc12345 "standup consensus decision quick agreement architectural"]
## Decisions
.
## Context
.
"""
        node_content = {"abc12345": "standup consensus decision quick agreement architectural meeting"}
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-21",
            node_content_map=node_content,
            pattern_history_lookup=None,
        )
        # Within-session check passes (lexical overlap with episode
        # body ≥2). No cross-session check ran. Graduation accepted.
        assert result.validated == 1
        assert result.cross_session_collisions == []

    def test_ungrounded_graduation_skips_cross_session_check(self):
        """Check 1 (no cited ID resolves) fails: the cross-session gate
        does NOT fire, because the gate is keyed on ``ids_valid`` — a line
        whose citation resolves to no real episode forms no Hebbian link
        anyway, so there is nothing for the cross-session gate to suppress.
        The line is marked ``(ungrounded)``.

        Distinct from the demoted-GROUNDING case (check 1 passes, check 2
        fails) which AM-XSESSION-LINKGATE (v0.4.3) DOES cross-session-check
        — see ``test_demoted_grounding_with_overlap_gates_link``."""
        text = """## State\nungrounded.\n## Patterns
- alpha | 2x (2026-05-21) [evidence: deadbeef "standup consensus decision quick agreement architectural"]
## Decisions
.
## Context
.
"""
        prior = "standup consensus decision quick agreement architectural"
        # deadbeef is not in valid_ids -> ungrounded
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-21",
            pattern_history_lookup=self._stub_lookup(alpha=prior),
            # Isolate the ungrounded-demotion mechanic: the stub returns a warm
            # at-peak (2x/max=2) history, which AM-CARRYFORWARD (v0.4.6) would
            # otherwise HOLD instead of demote. Disable it here so this test
            # keeps verifying the ungrounded-path demotion + marker; the hold
            # behavior has dedicated coverage in TestCarryforward.
            carryforward_cold_days=None,
        )
        assert result.validated == 0
        assert result.demoted == 1
        # Demotion marker should be (ungrounded), not (cross-session-overlap)
        assert "(ungrounded)" in result.text
        assert "(cross-session-overlap)" not in result.text
        assert result.cross_session_collisions == []

    def test_cross_session_overlap_forms_no_link(self):
        """AM-QUOTEFOOTGUN (v0.4.1) Option B: link formation is decoupled
        from the explanation-overlap *grounding* gate, but NOT from the
        cross-session anti-sycophancy gate. When the cross-session check
        flags a suspected re-graduation, we still refuse to strengthen
        the association graph from the gamed accumulation. Two valid IDs
        are co-cited so that a direct pair WOULD form if the immune gate
        were bypassed — it must not."""
        text = """## State\nsycophantic rephrase.\n## Patterns
- alpha | 3x (2026-05-21) [evidence: abc12345, def67890 "standup consensus decision quick agreement architectural"]
## Decisions
.
## Context
.
"""
        # Prior shares ≥3 meaningful words → cross-session fires. Both
        # episode bodies overlap the explanation so explanation_valid is
        # True (the line reaches the cross-session check at all).
        prior = "standup consensus decision architecture sync quick agreement"
        node_content = {
            "abc12345": "standup consensus decision quick agreement architectural meeting",
            "def67890": "standup consensus decision quick agreement architectural review",
        }
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345", "def67890"},
            today="2026-05-21",
            node_content_map=node_content,
            pattern_history_lookup=self._stub_lookup(alpha=prior),
        )
        assert result.validated == 0
        assert result.demoted == 1
        assert "(cross-session-overlap)" in result.text
        assert len(result.cross_session_collisions) == 1
        # The immune gate still protects the graph: no link forms.
        assert result.direct_co_citations == []
        assert result.all_validated_ids == []

    def test_demoted_grounding_with_overlap_gates_link(self):
        """AM-XSESSION-LINKGATE (v0.4.3): a graduation whose explanation
        FAILS the within-session grounding check (explanation_valid=False)
        but reuses >=threshold vocabulary from the prior session must STILL
        be caught by the cross-session gate.

        Pre-0.4.3 the gate only computed when explanation_valid was True, so
        this line slipped through as plain ``(ungrounded)`` while
        AM-QUOTEFOOTGUN's decoupled link formation (which fires on the
        demoted-grounding path) formed an UNSUPPRESSED Hebbian link from the
        gamed accumulation. Now the gate runs on the demoted path: the line
        is marked ``(cross-session-overlap)`` (the more specific signal wins
        over ``(ungrounded)`` on the both-apply line) and the link is
        suppressed."""
        text = """## State\ndemoted-grounding rephrase.\n## Patterns
- alpha | 2x (2026-05-21) [evidence: abc12345, def67890 "governance topology substrate boundary"]
## Decisions
.
## Context
.
"""
        # Episode bodies share NOTHING meaningful with the explanation ->
        # explanation_valid is False for both cited ids (demoted-grounding).
        node_content = {
            "abc12345": "the cat sat quietly on a warm rug by the door",
            "def67890": "sunlight came through the kitchen window this morning",
        }
        # Prior session shares 4 meaningful words -> cross-session fires.
        prior = "governance topology substrate boundary observed earlier"
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345", "def67890"},
            today="2026-05-21",
            node_content_map=node_content,
            pattern_history_lookup=self._stub_lookup(alpha=prior),
        )
        assert result.validated == 0
        assert result.demoted == 1
        # Cross-session marker WINS over (ungrounded) on the both-apply line.
        assert "(cross-session-overlap)" in result.text
        assert "(ungrounded)" not in result.text
        assert len(result.cross_session_collisions) == 1
        # The decoupled link MUST be suppressed — the hole this closes.
        assert result.direct_co_citations == []
        assert result.all_validated_ids == []
        # AM-WARN's resolved-citation signal is independent of the gate.
        assert result.any_citation_resolved is True

    def test_demoted_grounding_without_overlap_keeps_link(self):
        """Control for AM-XSESSION-LINKGATE: a demoted-grounding line with
        NO cross-session overlap stays ``(ungrounded)`` and KEEPS its
        decoupled Hebbian link — AM-QUOTEFOOTGUN's decouple for
        legitimately-paraphrased citations is preserved; only the
        sycophantic-overlap case is gated."""
        text = """## State\nparaphrased but honest.\n## Patterns
- alpha | 2x (2026-05-21) [evidence: abc12345, def67890 "governance topology substrate boundary"]
## Decisions
.
## Context
.
"""
        node_content = {
            "abc12345": "the cat sat quietly on a warm rug by the door",
            "def67890": "sunlight came through the kitchen window this morning",
        }
        # Prior shares ZERO meaningful words -> no cross-session collision.
        prior = "completely unrelated zebra giraffe elephant antelope vocabulary"
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345", "def67890"},
            today="2026-05-21",
            node_content_map=node_content,
            pattern_history_lookup=self._stub_lookup(alpha=prior),
            # Isolate the demoted-grounding LINK-keeping behavior: the stub's
            # warm at-peak history would otherwise let AM-CARRYFORWARD (v0.4.6)
            # HOLD the line instead of demoting it. Disable it so the test keeps
            # verifying that the co-citation link forms on the demoted path
            # (AM-QUOTEFOOTGUN decouple). TestCarryforward covers the hold path.
            carryforward_cold_days=None,
        )
        assert result.validated == 0
        assert result.demoted == 1
        assert "(ungrounded)" in result.text
        assert "(cross-session-overlap)" not in result.text
        assert result.cross_session_collisions == []
        # Decouple preserved: the link forms for the honest paraphrase.
        assert result.direct_co_citations == [("abc12345", "def67890")]
        assert result.all_validated_ids == [{"abc12345", "def67890"}]


class TestFlowScriptPrefixedPatterns:
    """Critical #1 fix coverage (v0.3.2): pattern lines with FlowScript
    marker prefixes (``!``, ``!!``, ``?``, ``✓``, ``*``) between the
    bullet and the operator-style name must be recognized by the
    immune-system regex. Before v0.3.2, `_NAMED_PATTERN_RE` required
    the name to immediately follow ``-\\s+`` and silently no-op'd on
    any FlowScript-prefixed line — making the entire library's
    canonical-flow continuity invisible to Move #2 and Move #3."""

    def test_double_bang_marker(self):
        text = """## Patterns
- !! load_bearing_pattern | 3x (2026-05-21) [evidence: abc12345 "x"]
"""
        assert extract_pattern_names(text) == {"load_bearing_pattern": 3}

    def test_single_bang_marker(self):
        text = """## Patterns
- ! urgent_pattern | 2x (2026-05-21) [evidence: abc12345 "x"]
"""
        assert extract_pattern_names(text) == {"urgent_pattern": 2}

    def test_question_marker(self):
        text = """## Patterns
- ? open_question_pattern | 1x (2026-05-21)
"""
        assert extract_pattern_names(text) == {"open_question_pattern": 1}

    def test_checkmark_marker(self):
        text = """## Patterns
- ✓ resolved_pattern | 2x (2026-05-21) [evidence: abc12345 "x"]
"""
        assert extract_pattern_names(text) == {"resolved_pattern": 2}

    def test_star_marker(self):
        text = """## Patterns
- * starred_pattern | 1x (2026-05-21)
"""
        assert extract_pattern_names(text) == {"starred_pattern": 1}

    def test_mixed_prefixed_and_unprefixed(self):
        text = """## Patterns
- plain_pattern | 2x (2026-05-21) [evidence: abc12345 "x"]
- !! marked_pattern | 3x (2026-05-21) [evidence: def67890 "y"]
- ? developing_pattern | 1x (2026-05-21)
"""
        result = extract_pattern_names(text)
        assert result == {
            "plain_pattern": 2,
            "marked_pattern": 3,
            "developing_pattern": 1,
        }


class TestCanonicalTemplateFormatEndToEnd:
    """Critical #1 lockdown (v0.3.2): a continuity in the EXACT format
    that ``_marker_reference()`` teaches agents to produce must fire
    Move #2 (omission audit) and Move #3 (cross-session check). Before
    v0.3.2 the template taught one format (``thought: prose | Nx``)
    and the immune system parsed a different format (``- name | Nx``),
    so every defense silently no-op'd in production. This test fixture
    pins the format the template now teaches against the regex the
    immune system uses, preventing silent re-divergence."""

    def _canonical_continuity(self, today: str) -> str:
        """Continuity built to match the format ``_marker_reference()``
        teaches as of v0.3.2 — operator-style names + optional
        FlowScript prefixes."""
        return f"""# Agent — Memory

## State
Built database layer.

## Patterns
- acid_compliance_over_speed | 2x ({today}) [evidence: 4931b6a8 "PostgreSQL chosen for ACID guarantees"]
- !! connection_pooling_bottleneck | 1x ({today}) [evidence: 5a829c7b "API latency improved after adding pool"]
- ? horizontal_scaling_strategy | 1x ({today})

## Decisions
[decided(rationale: "reliability over speed", on: "{today}")] PostgreSQL

## Context
Production work on data layer.
"""

    def test_extract_pattern_names_sees_canonical_format(self):
        """extract_pattern_names must capture every named pattern the
        template-teaches format produces — this is the regression
        guard against the v0.3.1→v0.3.2 grammar mismatch."""
        text = self._canonical_continuity("2026-05-21")
        result = extract_pattern_names(text)
        assert result == {
            "acid_compliance_over_speed": 2,
            "connection_pooling_bottleneck": 1,
            "horizontal_scaling_strategy": 1,
        }

    def test_omission_audit_fires_on_canonical_format(self):
        """detect_pattern_omissions must surface a dropped Proven
        pattern when prior and new are both in canonical format."""
        prior = self._canonical_continuity("2026-05-20")
        new = """# Agent — Memory
## State
After.
## Patterns
- !! connection_pooling_bottleneck | 1x (2026-05-21) [evidence: 5a829c7b "x"]
- ? horizontal_scaling_strategy | 1x (2026-05-21)
## Decisions
.
## Context
.
"""
        omissions = detect_pattern_omissions(prior, new, min_level=2)
        # acid_compliance_over_speed was at 2x in prior, absent in new
        assert [op.name for op in omissions] == ["acid_compliance_over_speed"]
        assert omissions[0].prior_level == 2

    def test_cross_session_check_fires_on_canonical_format(self):
        """validate_graduations cross-session path must fire on
        canonical-format graduations when pattern_history_lookup
        returns prior explanation overlapping ≥3 meaningful words."""
        text = """## State
.
## Patterns
- !! acid_compliance_over_speed | 2x (2026-05-21) [evidence: abc12345 "PostgreSQL chosen for ACID guarantees"]
## Decisions
.
## Context
.
"""
        prior_explanation = "PostgreSQL deployed for strong ACID guarantees on reads"

        def lookup(name):
            if name == "acid_compliance_over_speed":
                return {
                    "max_level_reached": 2,
                    "explanation_corpus": prior_explanation,
                    "last_explanation": prior_explanation,
                    # COLD (31d before today) so the AM-PRESERVE warm+fresh-specific
                    # exemption does not apply — this test isolates canonical-format
                    # PARSING of the cross-session gate, not warmth.
                    "last_seen_at": "2026-04-20T00:00:00Z",
                    "last_wrap_id": None,
                }
            return None

        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-21",
            node_content_map={"abc12345": "PostgreSQL chosen for ACID compliance"},
            pattern_history_lookup=lookup,
        )
        # Today's explanation: PostgreSQL chosen for ACID guarantees
        # Prior explanation:  PostgreSQL deployed for strong ACID guarantees on reads
        # Shared meaningful words: postgresql, acid, guarantees → 3 = threshold
        assert result.demoted == 1
        assert "(cross-session-overlap)" in result.text
        assert len(result.cross_session_collisions) == 1
        assert result.cross_session_collisions[0].name == "acid_compliance_over_speed"


class TestBulletlessGroupedPatterns:
    """AM-HISTUPSERT-BULLET (v0.4.5): the leading ``- `` bullet on a pattern
    line is OPTIONAL. An entity that groups its patterns under header lines
    with *indented, bullet-less* operator-named members (flow's
    ``{{topic: ...}}`` convention) was matched by _GRADUATION_RE (the
    demoter, bullet-agnostic) but NOT by _NAMED_PATTERN_RE (the per-name
    immune system) — so its patterns were demoted yet never protected and
    pattern_history stayed empty (0 rows / 20+ wraps; an
    ``invisible_infrastructure_failure``). These tests pin the dash-optional
    behavior so the two halves stay symmetric. The single-token
    ``name | Nx`` shape — NOT the dash — remains the false-positive guard."""

    def _grouped_continuity(self) -> str:
        """A continuity in flow's real shape: a ``{{topic — desc:}}`` group
        header with indented, bullet-less operator-named members."""
        return """## State
.
## Patterns

{{partnership & verification — the load-bearing set:
  verify_or_surface_before_acting | 3x (2026-05-21) [evidence: abc12345 "x"]
  tool_output_can_be_fabricated | 2x (2026-05-21) [evidence: def67890 "y"]
}}

{{memory architecture:
  tasks_are_not_memory | 1x (2026-05-21)
}}
## Context
.
"""

    def test_bulletless_grouped_members_match(self):
        """The operator-named members inside a {{group}} are extracted even
        with no leading dash — this is the core fix."""
        result = extract_pattern_names(self._grouped_continuity())
        assert result == {
            "verify_or_surface_before_acting": 3,
            "tool_output_can_be_fabricated": 2,
            "tasks_are_not_memory": 1,
        }

    def test_group_header_line_is_not_a_pattern(self):
        """The ``{{topic ...:}}`` header lines must NOT be mistaken for
        patterns — they start with ``{``, not an operator name."""
        result = extract_pattern_names(self._grouped_continuity())
        # No spurious key from a header line.
        assert all(not k.startswith("{") for k in result)
        assert "partnership" not in result
        assert "memory" not in result

    def test_bulletless_flowscript_marker(self):
        """A FlowScript marker prefix with no dash (``  ! name | Nx``) still
        matches — marker is optional AND dash is optional, independently."""
        text = """## Patterns
  ! urgent_pattern | 2x (2026-05-21) [evidence: abc12345 "x"]
  ? open_pattern | 1x (2026-05-21)
"""
        assert extract_pattern_names(text) == {
            "urgent_pattern": 2,
            "open_pattern": 1,
        }

    def test_mixed_dash_and_bulletless_in_one_section(self):
        """Dash-bulleted (canonical) and bullet-less (grouped) lines coexist
        in the same section — both are recognized."""
        text = """## Patterns
- dash_bulleted_pattern | 2x (2026-05-21) [evidence: abc12345 "x"]
  bulletless_pattern | 3x (2026-05-21) [evidence: def67890 "y"]
"""
        assert extract_pattern_names(text) == {
            "dash_bulleted_pattern": 2,
            "bulletless_pattern": 3,
        }

    def test_bulletless_multiword_prose_still_excluded(self):
        """A bullet-less prose line with a spaced phrase before ``| Nx`` does
        NOT match — the single-token name guard holds without the dash."""
        text = """## Patterns
  some prose sentence with a | 2x marker buried in it
  thought: freeform note | 3x (2026-05-21)
  legit_pattern | 1x (2026-05-21)
"""
        # Only the operator-named line matches; prose and thought: do not.
        assert extract_pattern_names(text) == {"legit_pattern": 1}

    def test_dash_bulleted_canonical_form_still_matches(self):
        """Regression guard: making the dash optional must not break the
        canonical dash-bulleted format the template still teaches first."""
        text = """## Patterns
- canonical_pattern | 2x (2026-05-21) [evidence: abc12345 "x"]
"""
        assert extract_pattern_names(text) == {"canonical_pattern": 2}

    def test_column0_single_word_prose_excluded(self):
        """M1 (L1 review): dash-optional must NOT let a column-0 single
        capitalized prose word + `| Nx` become a phantom pattern. The
        explicit signal (dash / marker / indentation) — not the operator-name
        token alone — is the guard against column-0 prose. Without it,
        `Throughput | 3x` and `Total | 2x` summary/metadata lines would be
        extracted as phantom graduated patterns and feed false omission
        signals downstream."""
        text = """## Patterns
Throughput | 3x when batched is higher
Total | 2x summary metadata line
  indented_member | 1x (2026-05-21)
"""
        # Only the indented member is a pattern; column-0 prose words are not.
        assert extract_pattern_names(text) == {"indented_member": 1}

    def test_column0_flowscript_marker_no_dash_no_indent_matches(self):
        """F2 (complement L1 verify pass): branch (c) — the zero-width
        lookahead admitting a FlowScript marker at column 0 with NO dash and
        NO indentation — is the trickiest new grammar piece. Pin it directly
        so a future "simplify the alternation" edit that drops branch (c)
        cannot pass the suite silently. Every other marker test uses a dash
        (branch a) or indentation (branch b)."""
        text = """## Patterns
! urgent_pattern | 2x (2026-05-21) [evidence: abc12345 "x"]
? open_pattern | 1x (2026-05-21)
* starred_pattern | 3x (2026-05-21) [evidence: def67890 "y"]
✓ done_pattern | 2x (2026-05-21) [evidence: aaa11111 "z"]
"""
        assert extract_pattern_names(text) == {
            "urgent_pattern": 2,
            "open_pattern": 1,
            "starred_pattern": 3,
            "done_pattern": 2,
        }
        # Direct on the regex: a column-0 marker is the ONLY branch-(c) path.
        assert _NAMED_PATTERN_RE.match("! urgent | 2x") is not None

    def test_raw_newline_is_not_indentation(self):
        """F3 (complement L1 verify pass): the indentation branch uses
        horizontal whitespace `[ \\t]`, NOT `\\s` — so a raw leading newline
        cannot count as "indentation" if the helper is ever reused on raw
        multi-line text (all current callers split on "\\n" first, but the
        guard should survive a future caller that doesn't). A revert to `\\s`
        — a plausible "normalize whitespace classes" cleanup — would silently
        re-open the footgun; this pins it."""
        assert _NAMED_PATTERN_RE.match("\nFoo | 2x") is None
        assert _NAMED_PATTERN_RE.match("\rFoo | 2x") is None
        # but real horizontal indentation still matches
        assert _NAMED_PATTERN_RE.match("  Foo | 2x") is not None
        assert _NAMED_PATTERN_RE.match("\tFoo | 2x") is not None

    def test_cross_session_check_fires_on_bulletless_member(self):
        """validate_graduations' cross-session sycophancy gate must fire on a
        bullet-less grouped member — before the fix, name_match was None so
        the gate silently skipped flow's entire pattern set."""
        text = """## State
.
## Patterns

{{database:
  acid_compliance_over_speed | 2x (2026-05-21) [evidence: abc12345 "PostgreSQL chosen for ACID guarantees"]
}}
## Context
.
"""
        prior_explanation = "PostgreSQL deployed for strong ACID guarantees on reads"

        def lookup(name):
            if name == "acid_compliance_over_speed":
                return {
                    "max_level_reached": 2,
                    "explanation_corpus": prior_explanation,
                    "last_explanation": prior_explanation,
                    # COLD (31d before today) so the AM-PRESERVE warm+fresh-specific
                    # exemption does not apply — this test isolates bulletless-grouped
                    # member PARSING of the cross-session gate, not warmth.
                    "last_seen_at": "2026-04-20T00:00:00Z",
                    "last_wrap_id": None,
                }
            return None

        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-21",
            node_content_map={"abc12345": "PostgreSQL chosen for ACID compliance"},
            pattern_history_lookup=lookup,
        )
        # Shared meaningful words postgresql/acid/guarantees ≥3 → demote.
        assert result.demoted == 1
        assert "(cross-session-overlap)" in result.text
        assert len(result.cross_session_collisions) == 1
        assert result.cross_session_collisions[0].name == "acid_compliance_over_speed"

    def test_omission_audit_fires_on_bulletless(self):
        """detect_pattern_omissions surfaces a dropped Proven pattern when
        both prior and new are bullet-less grouped format."""
        prior = self._grouped_continuity()
        new = """## State
.
## Patterns

{{partnership & verification — the load-bearing set:
  verify_or_surface_before_acting | 3x (2026-05-21) [evidence: abc12345 "x"]
}}
## Context
.
"""
        omissions = detect_pattern_omissions(prior, new, min_level=2)
        # tool_output_can_be_fabricated was 2x in prior, absent in new.
        assert [op.name for op in omissions] == ["tool_output_can_be_fabricated"]
        assert omissions[0].prior_level == 2

    def test_anti_patterns_bulletless_members_still_rejected(self):
        """The section guard (not the dash) excludes ``## Anti-Patterns`` —
        bullet-less members there must still be skipped after the fix."""
        text = """## Patterns
  real_pattern | 2x (2026-05-21) [evidence: abc12345 "x"]
## Anti-Patterns
  avoid_this_pattern | 3x (2026-05-21) [evidence: bad11111 "x"]
## Context
.
"""
        result = extract_pattern_names(text)
        assert result == {"real_pattern": 2}
        assert "avoid_this_pattern" not in result


class TestAntiPatternsSectionRejection:
    """v0.3.2 fix: ``## Anti-Patterns`` (and other ``pattern``-substring
    section names) must NOT be parsed as the graduated-patterns
    section. Before v0.3.2 the section check was ``"pattern" in
    line.lower()`` which matched every such section, polluting
    validation / extraction / staleness with anti-pattern bullets the
    agent was actively trying to suppress.

    All three section-scanning functions (validate_graduations,
    extract_pattern_names, detect_stale_patterns) must converge on
    the canonical ``## Patterns`` heading via _is_patterns_heading."""

    def test_extract_skips_anti_patterns(self):
        text = """## State
foo.
## Patterns
- legitimate_pattern | 2x (2026-05-21) [evidence: abc12345 "x"]
## Anti-Patterns
- avoid_this_pattern | 3x (2026-05-21) [evidence: bad11111 "x"]
## Context
.
"""
        result = extract_pattern_names(text)
        assert result == {"legitimate_pattern": 2}
        assert "avoid_this_pattern" not in result

    def test_extract_skips_other_pattern_sections(self):
        text = """## Patterns
- real_pattern | 2x (2026-05-21) [evidence: abc12345 "x"]
## Other Patterns
- fake_pattern | 3x (2026-05-21) [evidence: bad11111 "x"]
## Design Patterns
- design_pattern | 2x (2026-05-21) [evidence: cd111111 "x"]
"""
        assert extract_pattern_names(text) == {"real_pattern": 2}

    def test_validate_graduations_skips_anti_patterns(self):
        text = """## State
.
## Patterns
- real_pattern | 2x (2026-05-21) [evidence: abc12345 "matches the episode content here"]
## Anti-Patterns
- avoid_pattern | 2x (2026-05-21) [evidence: ffffffff "nonexistent"]
## Context
.
"""
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-21",
            node_content_map={
                "abc12345": "matches the episode content here",
                "ffffffff": "would be poisoned if scanned",
            },
        )
        # Only the real pattern should be validated. The anti-pattern
        # line should be skipped entirely — no validate, no demote.
        assert result.validated == 1
        assert result.demoted == 0

    def test_detect_stale_skips_anti_patterns(self):
        text = """## Patterns
- real_pattern | 2x (2026-05-01) [evidence: abc12345 "x"]
## Anti-Patterns
- avoid_pattern | 2x (2026-05-01) [evidence: bad11111 "x"]
"""
        stale = detect_stale_patterns(text, today="2026-05-21", staleness_days=7)
        # Only the real Patterns-section pattern should appear as stale.
        # The anti-patterns entry must not be surfaced — operators do
        # not want their anti-pattern catalog flagged for removal.
        contents = [s.content for s in stale]
        assert any("real_pattern" in c for c in contents)
        assert not any("avoid_pattern" in c for c in contents)


class TestCorpusGrowthDoesNotFalsePositive:
    """v0.3.2 reframe (4-layer review WARNING #1 from 4 independent
    agents): the cross-session check must NOT false-positive on
    legitimate long-lived patterns whose domain vocabulary naturally
    recurs across multi-faceted evidence. The previous whole-corpus
    union comparison failed this — corpus word-set grew monotonically
    and the chance of ≥threshold overlap with new legitimate
    explanations approached 1 over time.

    These tests simulate N≥10 legitimate accumulations with
    deliberately-distinct vocabulary per session (mirroring how real
    multi-faceted evidence accrues across sessions) and assert the
    Nth+1 graduation still passes."""

    def test_ten_distinct_explanations_then_distinct_eleventh_passes(self):
        """The killer test. Ten prior explanations, each on a different
        angle of the same pattern, then an eleventh that's also a new
        angle. Per-prior comparison must let it through; whole-corpus
        union comparison would have demoted it."""
        prior_explanations = [
            "PostgreSQL chosen for ACID guarantees on writes",
            "Connection pooling improved query latency dramatically",
            "Read replicas added to scale reporting queries",
            "Partitioning strategy designed around tenant boundaries",
            "Backup window moved to off-hours after capacity issue",
            "Index tuning revealed missing compound key on orders",
            "Vacuum scheduling reduced bloat on hot tables overnight",
            "Replication lag debugged via streaming WAL inspection",
            "Connection limits raised after pgbouncer pool exhaustion",
            "Failover tested with synthetic primary outage drill",
        ]
        corpus = "\n".join(prior_explanations)

        def lookup(name):
            if name == "database_evolution":
                return {
                    "max_level_reached": 3,
                    "explanation_corpus": corpus,
                    "last_explanation": prior_explanations[-1],
                    "last_seen_at": "2026-05-20T00:00:00Z",
                    "last_wrap_id": None,
                }
            return None

        # 11th explanation — new angle, distinct vocabulary
        text = """## State
.
## Patterns
- database_evolution | 3x (2026-05-21) [evidence: abc12345 "Foreign data wrapper deployed for analytics warehouse joins"]
## Decisions
.
## Context
.
"""
        node_content = {"abc12345": "Foreign data wrapper deployed for analytics warehouse joins"}
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-21",
            node_content_map=node_content,
            pattern_history_lookup=lookup,
        )
        # The 11th explanation shares ZERO meaningful words with ANY
        # individual prior. Per-prior max-overlap = 0. Should pass cleanly.
        assert result.validated == 1, (
            f"Expected validated=1, got demoted={result.demoted}, "
            f"collisions={[c.overlap_words for c in result.cross_session_collisions]}"
        )
        assert result.cross_session_collisions == []

    def test_rephrasing_against_specific_prior_still_demotes(self):
        """The reframe must NOT break the actual attack closure. A
        new explanation that heavily reuses vocabulary from a SPECIFIC
        prior explanation should still demote — that's the
        lexical-rephrasing variant Phase 1b probe #1 S5 demonstrated."""
        prior_explanations = [
            "PostgreSQL chosen for ACID guarantees on writes",
            "Connection pooling improved query latency dramatically",
            "Read replicas added to scale reporting queries",
        ]
        corpus = "\n".join(prior_explanations)

        def lookup(name):
            if name == "database_evolution":
                return {
                    "max_level_reached": 3,
                    "explanation_corpus": corpus,
                    "last_explanation": prior_explanations[0],
                    "last_seen_at": "2026-05-20T00:00:00Z",
                    "last_wrap_id": None,
                }
            return None

        # Rephrase of prior 0: shared = postgresql, acid, guarantees, writes → 4
        text = """## State
.
## Patterns
- database_evolution | 3x (2026-05-21) [evidence: abc12345 "PostgreSQL deployed because ACID guarantees on writes matter most"]
## Decisions
.
## Context
.
"""
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-21",
            node_content_map={"abc12345": "PostgreSQL ACID guarantees writes selected"},
            pattern_history_lookup=lookup,
        )
        # Today's explanation rephrases prior[0] heavily. Best per-prior
        # overlap = 4 (postgresql, acid, guarantees, writes). Should demote.
        assert result.demoted == 1
        assert "(cross-session-overlap)" in result.text
        assert len(result.cross_session_collisions) == 1
        # Collision should report the prior that triggered (the rephrased one)
        coll = result.cross_session_collisions[0]
        assert "PostgreSQL chosen for ACID guarantees on writes" in coll.prior_explanation

    def test_twenty_explanations_with_natural_vocabulary_recurrence_passes(self):
        """Stress test — a pattern at 20+ accumulations where domain
        vocabulary naturally recurs across MOST priors. The reframe
        must still let a new genuinely-distinct angle through. This
        is the failure scenario all four review agents predicted
        would silently surface in production at 6-12 month horizons
        under the old whole-corpus implementation."""
        # 20 explanations, all mentioning "database" + "PostgreSQL" + assorted
        # domain words — natural recurrence pattern for a long-lived Proven.
        prior_explanations = [
            "PostgreSQL database chosen ACID writes",
            "PostgreSQL database connection pooling latency",
            "PostgreSQL database read replicas reporting",
            "PostgreSQL database partitioning tenant boundaries",
            "PostgreSQL database backup window capacity",
            "PostgreSQL database index compound orders",
            "PostgreSQL database vacuum bloat tables",
            "PostgreSQL database replication lag streaming",
            "PostgreSQL database limits pgbouncer pool",
            "PostgreSQL database failover synthetic outage",
            "PostgreSQL database materialized views dashboards",
            "PostgreSQL database row-level security policies",
            "PostgreSQL database extension PostGIS spatial",
            "PostgreSQL database upgrade major version migration",
            "PostgreSQL database table inheritance partitioning",
            "PostgreSQL database trigger function audit",
            "PostgreSQL database statement timeout configuration",
            "PostgreSQL database query plan analyzer review",
            "PostgreSQL database concurrent index build",
            "PostgreSQL database logical decoding replication",
        ]
        corpus = "\n".join(prior_explanations)

        def lookup(name):
            return {
                "max_level_reached": 3,
                "explanation_corpus": corpus,
                "last_explanation": prior_explanations[-1],
                "last_seen_at": "2026-05-20T00:00:00Z",
                "last_wrap_id": None,
            }

        # 21st explanation: distinct angle, but reuses "PostgreSQL database"
        # naturally — 2 shared words with EVERY prior. Per-prior max = 2,
        # below threshold=3. Should pass.
        text = """## State
.
## Patterns
- database_evolution | 3x (2026-05-21) [evidence: abc12345 "PostgreSQL database foreign wrappers analytics warehouse"]
## Decisions
.
## Context
.
"""
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-21",
            node_content_map={"abc12345": "PostgreSQL database foreign wrappers analytics warehouse joins"},
            pattern_history_lookup=lookup,
        )
        # The new explanation shares 2 meaningful words ("postgresql",
        # "database") with EVERY prior — but per-prior overlap is 2,
        # below threshold=3. Under the old whole-corpus union check,
        # this would have demoted (the union has 50+ accumulated words
        # and the new explanation easily shares ≥3 with it). Under the
        # per-prior reframe, it passes.
        assert result.validated == 1, (
            f"Expected validated=1, got demoted={result.demoted}, "
            f"collisions={[c.overlap_words for c in result.cross_session_collisions]}"
        )


class TestExtractProvenPatterns:
    """Move #4 library layer (v0.3.2): extract_proven_patterns returns
    operator-style names of every Proven-tier (>= min_level) pattern
    in the ## Patterns section. Used by prepare_wrap to surface
    uncovered_proven_to_check."""

    def test_returns_2x_and_3x_sorted(self):
        from anneal_memory.graduation import extract_proven_patterns
        text = """## Patterns
- gamma_pattern | 3x (2026-05-21) [evidence: abc12345 "x"]
- alpha_pattern | 2x (2026-05-21) [evidence: def67890 "x"]
- beta_pattern | 1x (2026-05-21)
"""
        assert extract_proven_patterns(text) == ["alpha_pattern", "gamma_pattern"]

    def test_excludes_1x_by_default(self):
        from anneal_memory.graduation import extract_proven_patterns
        text = """## Patterns
- developing_pattern | 1x (2026-05-21)
- proven_pattern | 2x (2026-05-21) [evidence: abc12345 "x"]
"""
        assert extract_proven_patterns(text) == ["proven_pattern"]

    def test_min_level_override(self):
        from anneal_memory.graduation import extract_proven_patterns
        text = """## Patterns
- one_x | 1x (2026-05-21)
- two_x | 2x (2026-05-21) [evidence: abc12345 "x"]
- three_x | 3x (2026-05-21) [evidence: def67890 "x"]
"""
        assert extract_proven_patterns(text, min_level=3) == ["three_x"]
        assert extract_proven_patterns(text, min_level=1) == ["one_x", "three_x", "two_x"]

    def test_empty_continuity_returns_empty(self):
        from anneal_memory.graduation import extract_proven_patterns
        assert extract_proven_patterns("") == []


class TestExtractContradictionDeclarations:
    """Move #4 library layer (v0.3.2): per-pattern [contradicts: ...]
    and [no-contradicts] declaration extraction."""

    def test_contradicts_single_name(self):
        from anneal_memory.graduation import extract_contradiction_declarations
        text = """## Patterns
- new_pattern | 2x (2026-05-21) [evidence: abc12345 "x"] [contradicts: old_pattern]
"""
        result = extract_contradiction_declarations(text)
        assert result == {"new_pattern": ["old_pattern"]}

    def test_contradicts_multiple_names(self):
        from anneal_memory.graduation import extract_contradiction_declarations
        text = """## Patterns
- new_pattern | 3x (2026-05-21) [evidence: abc12345 "x"] [contradicts: old_a, old_b, old_c]
"""
        result = extract_contradiction_declarations(text)
        assert result == {"new_pattern": ["old_a", "old_b", "old_c"]}

    def test_no_contradicts_declaration(self):
        from anneal_memory.graduation import extract_contradiction_declarations
        text = """## Patterns
- new_pattern | 2x (2026-05-21) [evidence: abc12345 "x"] [no-contradicts]
"""
        result = extract_contradiction_declarations(text)
        # Empty list = explicit no-contradicts declaration
        assert result == {"new_pattern": []}

    def test_no_declaration_absent_from_dict(self):
        from anneal_memory.graduation import extract_contradiction_declarations
        text = """## Patterns
- new_pattern | 2x (2026-05-21) [evidence: abc12345 "x"]
"""
        result = extract_contradiction_declarations(text)
        # Pattern with NO declaration should not appear in the dict
        assert result == {}

    def test_mixed_declarations(self):
        from anneal_memory.graduation import extract_contradiction_declarations
        text = """## Patterns
- pattern_a | 2x (2026-05-21) [evidence: abc12345 "x"] [contradicts: prior_a]
- pattern_b | 2x (2026-05-21) [evidence: def67890 "x"] [no-contradicts]
- pattern_c | 2x (2026-05-21) [evidence: 11111111 "x"]
"""
        result = extract_contradiction_declarations(text)
        assert result == {
            "pattern_a": ["prior_a"],
            "pattern_b": [],
            # pattern_c absent — no declaration
        }


class TestDetectProvenWithoutDeclaration:
    """Move #4 library layer (v0.3.2): detect NEW Proven graduations
    missing contradiction-stance declaration. Audit signal for
    operator-review (Diogenes) to know which new Provens need
    semantic-opposition inspection."""

    def test_new_proven_without_declaration_surfaces(self):
        from anneal_memory.graduation import detect_proven_without_declaration
        prior = """## Patterns
- old_proven | 2x (2026-05-20) [evidence: 11111111 "x"]
"""
        new = """## Patterns
- old_proven | 2x (2026-05-21) [evidence: 11111111 "x"]
- new_proven_no_declaration | 2x (2026-05-21) [evidence: abc12345 "x"]
"""
        result = detect_proven_without_declaration(prior, new)
        assert [p.name for p in result] == ["new_proven_no_declaration"]
        assert result[0].level == 2

    def test_new_proven_with_contradicts_declaration_skipped(self):
        from anneal_memory.graduation import detect_proven_without_declaration
        prior = """## Patterns
- old_proven | 2x (2026-05-20) [evidence: 11111111 "x"]
"""
        new = """## Patterns
- old_proven | 2x (2026-05-21) [evidence: 11111111 "x"]
- new_proven_with_declaration | 2x (2026-05-21) [evidence: abc12345 "x"] [contradicts: old_proven]
"""
        result = detect_proven_without_declaration(prior, new)
        # The declaration satisfied the discipline; not surfaced
        assert result == []

    def test_new_proven_with_no_contradicts_declaration_skipped(self):
        from anneal_memory.graduation import detect_proven_without_declaration
        prior = """## Patterns
- old_proven | 2x (2026-05-20) [evidence: 11111111 "x"]
"""
        new = """## Patterns
- old_proven | 2x (2026-05-21) [evidence: 11111111 "x"]
- explicit_no_contradicts | 2x (2026-05-21) [evidence: abc12345 "x"] [no-contradicts]
"""
        result = detect_proven_without_declaration(prior, new)
        # Explicit no-contradicts satisfies the discipline too
        assert result == []

    def test_carried_forward_proven_not_flagged(self):
        from anneal_memory.graduation import detect_proven_without_declaration
        prior = """## Patterns
- old_proven | 3x (2026-05-20) [evidence: 11111111 "x"]
"""
        new = """## Patterns
- old_proven | 3x (2026-05-21) [evidence: 11111111 "x"]
"""
        # No new graduation, only carry-forward at same level — must not surface
        result = detect_proven_without_declaration(prior, new)
        assert result == []

    def test_1x_to_2x_graduation_without_declaration_surfaces(self):
        from anneal_memory.graduation import detect_proven_without_declaration
        prior = """## Patterns
- pattern | 1x (2026-05-20) [evidence: 11111111 "x"]
"""
        new = """## Patterns
- pattern | 2x (2026-05-21) [evidence: abc12345 "x"]
"""
        # 1x → 2x is a new Proven graduation; should surface without declaration
        result = detect_proven_without_declaration(prior, new)
        assert [p.name for p in result] == ["pattern"]
        assert result[0].level == 2

    def test_demotion_not_flagged(self):
        from anneal_memory.graduation import detect_proven_without_declaration
        prior = """## Patterns
- pattern | 3x (2026-05-20) [evidence: 11111111 "x"]
"""
        new = """## Patterns
- pattern | 2x (2026-05-21) [evidence: 11111111 "x"]
"""
        # Carried forward at LOWER level — not a new graduation
        result = detect_proven_without_declaration(prior, new)
        assert result == []


class TestV033HotfixRegressionGuards:
    """v0.3.3 hotfix regression guards — locks the four bugs that
    v0.3.2 shipped to PyPI before its own 4-layer review fired.

    Each bug class below has a test that would have FAILED on v0.3.2
    main + PASSES on v0.3.3 main. CI parity protects against any
    future regression."""

    def test_high2_demote_handles_no_space_between_pipe_and_level(self):
        """HIGH #2: _demote_line must handle `|2x` (no space) — the
        widened _NAMED_PATTERN_RE in v0.3.2 accepts no-space form, but
        v0.3.2's literal `f"| {level}x"` string-replace only matched
        the single-space form. Result was state corruption: counter
        said `demoted == 1` but text retained the old level."""
        text = """## Patterns
- bad |2x (2026-05-21) [evidence: ffffffff "completely unrelated"]
"""
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-21",
            node_content_map={"abc12345": "PostgreSQL"},
        )
        assert result.demoted == 1
        # Text must reflect the demotion — no `|2x` should remain
        # (the bug was: counter says demoted but text stays 2x)
        assert "|2x" not in result.text
        assert "| 2x" not in result.text
        assert "| 1x" in result.text or "|1x" in result.text
        assert "(ungrounded)" in result.text

    def test_high2_demote_handles_extra_space_between_pipe_and_level(self):
        """Same defect family — multiple spaces between pipe and level."""
        text = """## Patterns
- bad |  2x (2026-05-21) [evidence: ffffffff "x"]
"""
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-21",
        )
        assert result.demoted == 1
        # The "2x" that's being demoted should not remain. The
        # regex-based replace handles both `|  2x` and `| 2x`.
        # Confirm the line shows a 1x marker:
        assert "1x" in result.text

    def test_medium3_no_contradicts_inside_evidence_quote_does_not_count(self):
        """MEDIUM #3: contradiction declarations inside evidence
        quotes must NOT spoof the declaration check. Codex L3 v0.3.2
        review caught that `[evidence: ... "we wrote [no-contradicts]
        in a log"]` was being treated as an explicit no-contradicts
        declaration. v0.3.3 strips evidence blocks before searching."""
        from anneal_memory.graduation import extract_contradiction_declarations
        text = """## Patterns
- spoofed_pattern | 2x (2026-05-21) [evidence: abc12345 "we wrote [no-contradicts] in a log"]
"""
        result = extract_contradiction_declarations(text)
        # The pattern has NO real declaration outside the evidence
        # quote; result must be empty (absence = no declaration).
        assert result == {}, (
            f"Spoofed no-contradicts inside evidence should not register, "
            f"got: {result}"
        )

    def test_medium3_contradicts_inside_evidence_quote_does_not_count(self):
        """Same spoofing protection for [contradicts: X] inside
        evidence quotes."""
        from anneal_memory.graduation import extract_contradiction_declarations
        text = """## Patterns
- spoofed_pattern | 2x (2026-05-21) [evidence: abc12345 "we said [contradicts: old_pattern] earlier"]
"""
        result = extract_contradiction_declarations(text)
        assert result == {}, (
            f"Spoofed contradicts inside evidence should not register, "
            f"got: {result}"
        )

    def test_medium3_real_declaration_outside_evidence_still_counts(self):
        """The fix must NOT break the legitimate case — a real
        contradiction declaration outside the evidence quote must
        still count."""
        from anneal_memory.graduation import extract_contradiction_declarations
        text = """## Patterns
- legitimate | 2x (2026-05-21) [evidence: abc12345 "x"] [contradicts: old_pattern]
"""
        result = extract_contradiction_declarations(text)
        assert result == {"legitimate": ["old_pattern"]}

    def test_medium4_old_date_restored_line_not_flagged_as_new_proven(self):
        """MEDIUM #4: detect_proven_without_declaration with today
        parameter must skip pattern lines whose date is not today.
        v0.3.2 flagged restored/imported old-date lines as new
        graduations needing a declaration — wrong, the agent didn't
        author those THIS wrap."""
        from anneal_memory.graduation import detect_proven_without_declaration
        prior = """## Patterns
- old_unchanged | 2x (2026-05-15) [evidence: 11111111 "x"]
"""
        new = """## Patterns
- old_unchanged | 2x (2026-05-15) [evidence: 11111111 "x"]
- restored_old | 2x (2026-05-10) [evidence: abc12345 "imported from prior"]
"""
        # restored_old is "new relative to prior" but its date is
        # 2026-05-10, not today's 2026-05-21. Should NOT flag.
        result = detect_proven_without_declaration(
            prior, new, today="2026-05-21",
        )
        assert result == [], (
            f"Old-date restored line should not flag, got: {result}"
        )

    def test_medium4_today_dated_new_proven_still_flagged_without_declaration(self):
        """Fix must NOT break the legitimate flagging case — a today-
        authored new Proven without declaration must still flag."""
        from anneal_memory.graduation import detect_proven_without_declaration
        prior = ""
        new = """## Patterns
- new_today | 2x (2026-05-21) [evidence: abc12345 "x"]
"""
        result = detect_proven_without_declaration(
            prior, new, today="2026-05-21",
        )
        assert [p.name for p in result] == ["new_today"]

    def test_low5_named_pattern_re_rejects_invalid_marker_combinations(self):
        """LOW #5: v0.3.2's `[!?✓*]+` matched `!!!`, `!*?`, `??`, etc.
        v0.3.3 tightens to explicit alternation. Confirm invalid
        combinations no longer match."""
        from anneal_memory.graduation import _NAMED_PATTERN_RE
        # Invalid: three bangs
        assert _NAMED_PATTERN_RE.match("- !!! pattern_name | 2x") is None
        # Invalid: mixed-marker combo
        assert _NAMED_PATTERN_RE.match("- !*? pattern_name | 2x") is None
        # Invalid: double-question
        assert _NAMED_PATTERN_RE.match("- ?? pattern_name | 2x") is None

    def test_low5_named_pattern_re_still_accepts_valid_markers(self):
        """Regression guard — fix must not break the canonical
        FlowScript marker set."""
        from anneal_memory.graduation import _NAMED_PATTERN_RE
        assert _NAMED_PATTERN_RE.match("- pattern_name | 2x") is not None
        assert _NAMED_PATTERN_RE.match("- !! pattern_name | 2x") is not None
        assert _NAMED_PATTERN_RE.match("- ! pattern_name | 1x") is not None
        assert _NAMED_PATTERN_RE.match("- ? pattern_name | 1x") is not None
        assert _NAMED_PATTERN_RE.match("- ✓ pattern_name | 3x") is not None
        assert _NAMED_PATTERN_RE.match("- * pattern_name | 2x") is not None


class TestV033NormalizeOrderOfOpsRegression:
    """v0.3.3 session-code-review WARNING #1: em-dash-wrapped explanations
    bypassed dedup because the punct-strip revealed whitespace the
    earlier .strip() had consumed. Regression guard."""

    def test_em_dash_wrapped_normalizes_same_as_plain(self):
        from anneal_memory.store import _normalize_explanation_for_dedup
        plain = "PostgreSQL chosen for ACID"
        em_dash = "— PostgreSQL chosen for ACID —"
        assert _normalize_explanation_for_dedup(plain) == \
               _normalize_explanation_for_dedup(em_dash)

    def test_compound_punct_whitespace_normalizes_consistently(self):
        from anneal_memory.store import _normalize_explanation_for_dedup
        plain = "PostgreSQL chosen for ACID"
        # All these should collapse to the same key
        variants = [
            "PostgreSQL chosen for ACID",
            "PostgreSQL chosen for ACID.",
            " PostgreSQL  chosen  for  ACID ",
            "PostgreSQL chosen for ACID\n",
            "(PostgreSQL chosen for ACID)",
            "— PostgreSQL chosen for ACID —",
            "[PostgreSQL chosen for ACID]",
            " ' PostgreSQL chosen for ACID ' ",
        ]
        canonical = _normalize_explanation_for_dedup(plain)
        for v in variants:
            assert _normalize_explanation_for_dedup(v) == canonical, (
                f"Variant {v!r} did not normalize to canonical {canonical!r}"
            )


class TestCarryforward:
    """AM-CARRYFORWARD (v0.4.6): on the ungrounded-citation demotion path, a
    pattern at/below its earned high-water mark AND grounded recently (warm)
    is HELD at its level instead of ratcheting down. Cold / never-earned /
    no-history / disabled all fall through to the pre-0.4.6 demotion. The
    (cross-session-overlap) immune path is NEVER carried forward."""

    PATTERNS = "## Patterns"

    def _lookup(self, max_level, last_seen):
        def _l(name):
            if name == "alpha":
                return {
                    "max_level_reached": max_level,
                    "last_seen_at": last_seen,
                    "explanation_corpus": "",
                    "last_explanation": "",
                    "last_wrap_id": None,
                }
            return None
        return _l

    def _text(self, level):
        return (
            "## State\n.\n## Patterns\n"
            f'- alpha | {level}x (2026-06-04) [evidence: deadbeef "thin overlap fails to ground"]\n'
            "## Decisions\n.\n## Context\n.\n"
        )

    def _line(self, result):
        return next(l for l in result.text.splitlines() if l.startswith("- alpha"))

    def test_warm_at_peak_held_not_demoted(self):
        r = validate_graduations(
            text=self._text(3), valid_ids=set(), today="2026-06-04",
            pattern_history_lookup=self._lookup(3, "2026-06-02T10:00:00Z"),
        )
        assert r.demoted == 0
        assert len(r.carried_forward) == 1
        cf = r.carried_forward[0]
        assert (cf.name, cf.held_level, cf.max_level_reached, cf.days_since_grounded) == (
            "alpha", 3, 3, 2)
        assert cf.cited is True  # cited path: carried a citation -> counts toward AM-WARN
        assert "| 3x (2026-06-04) (carried-forward)" in self._line(r)
        assert "(ungrounded)" not in r.text

    def test_cold_pattern_ages_out(self):
        r = validate_graduations(
            text=self._text(3), valid_ids=set(), today="2026-06-04",
            pattern_history_lookup=self._lookup(3, "2026-05-01T10:00:00Z"),
        )
        assert r.demoted == 1
        assert r.carried_forward == []
        assert "| 2x (2026-06-04) (ungrounded)" in self._line(r)

    def test_unearned_level_demotes(self):
        # Line is 3x but high-water mark is only 2 — never earned 3x.
        r = validate_graduations(
            text=self._text(3), valid_ids=set(), today="2026-06-04",
            pattern_history_lookup=self._lookup(2, "2026-06-03T10:00:00Z"),
        )
        assert r.demoted == 1
        assert r.carried_forward == []

    def test_protects_below_peak(self):
        # 2x line, earned 3x before, warm -> held (level <= max_level).
        r = validate_graduations(
            text=self._text(2), valid_ids=set(), today="2026-06-04",
            pattern_history_lookup=self._lookup(3, "2026-06-03T10:00:00Z"),
        )
        assert r.demoted == 0
        assert len(r.carried_forward) == 1
        assert r.carried_forward[0].held_level == 2
        assert "| 2x (2026-06-04) (carried-forward)" in self._line(r)

    def test_no_history_demotes(self):
        r = validate_graduations(
            text=self._text(3), valid_ids=set(), today="2026-06-04",
            pattern_history_lookup=lambda n: None,
        )
        assert r.demoted == 1
        assert r.carried_forward == []

    def test_disabled_via_none_demotes_even_when_warm(self):
        r = validate_graduations(
            text=self._text(3), valid_ids=set(), today="2026-06-04",
            pattern_history_lookup=self._lookup(3, "2026-06-04T10:00:00Z"),
            carryforward_cold_days=None,
        )
        assert r.demoted == 1
        assert r.carried_forward == []

    def test_no_lookup_wired_demotes(self):
        # pattern_history_lookup defaults to None -> carryforward inert.
        r = validate_graduations(
            text=self._text(3), valid_ids=set(), today="2026-06-04",
        )
        assert r.demoted == 1
        assert r.carried_forward == []

    def test_cold_boundary_inclusive(self):
        # Exactly cold_days (7) -> still warm (held); cold_days+1 -> aged out.
        # The graduation line's date must equal `today` or validate_graduations
        # skips it as non-today before the carryforward path is reached.
        def text(line_date):
            return (
                "## State\n.\n## Patterns\n"
                f'- alpha | 3x ({line_date}) [evidence: deadbeef "thin overlap"]\n'
                "## Decisions\n.\n## Context\n.\n"
            )
        held = validate_graduations(
            text=text("2026-06-08"), valid_ids=set(), today="2026-06-08",
            pattern_history_lookup=self._lookup(3, "2026-06-01T00:00:00Z"),
            carryforward_cold_days=7,
        )
        assert held.demoted == 0 and len(held.carried_forward) == 1
        aged = validate_graduations(
            text=text("2026-06-09"), valid_ids=set(), today="2026-06-09",
            pattern_history_lookup=self._lookup(3, "2026-06-01T00:00:00Z"),
            carryforward_cold_days=7,
        )
        assert aged.demoted == 1 and aged.carried_forward == []

    def test_far_future_last_seen_rejected_not_warm(self):
        # A last_seen_at clearly in the future (> 1 day) is untrustworthy
        # (clock corruption / leaked date) — it must NOT read as maximally
        # warm. Conservative-demotion: reject (demote), don't protect.
        r = validate_graduations(
            text=self._text(3), valid_ids=set(), today="2026-06-04",
            pattern_history_lookup=self._lookup(3, "2026-06-10T00:00:00Z"),
        )
        assert r.demoted == 1
        assert r.carried_forward == []

    def test_one_day_utc_skew_tolerated(self):
        # A last_seen_at exactly 1 day "future" is the legitimate UTC-vs-local
        # skew (store stamps UTC; a real evening grounding reads as tomorrow
        # UTC against a local `today`). Tolerated as warm → held (codex L3).
        r = validate_graduations(
            text=self._text(3), valid_ids=set(), today="2026-06-04",
            pattern_history_lookup=self._lookup(3, "2026-06-05T01:00:00Z"),
        )
        assert r.demoted == 0
        assert len(r.carried_forward) == 1
        assert r.carried_forward[0].days_since_grounded == 0  # clamped, not -1

    def test_malformed_two_marker_not_held_with_wrong_name(self):
        # codex L3 repro: a malformed line whose FIRST marker is demoted
        # (evidence stripped) and a LATER marker carries the (dead) evidence
        # validation matches. The line-start name (name_a) is warm at-peak in
        # history — but carryforward must NOT hold the LATER marker using
        # name_a's history. The combined regex fails to match (first marker has
        # no evidence) → decline → demote. Name↔marker binding is airtight.
        text = (
            "## State\n.\n## Patterns\n"
            "- name_a | 3x (2026-06-04) (ungrounded) "
            'name_b | 3x (2026-06-04) [evidence: deadbeef "later marker"]\n'
            "## Decisions\n.\n## Context\n.\n"
        )
        # name_a is warm at-peak; name_b has no history.
        def lk(name):
            if name == "name_a":
                return {"max_level_reached": 3, "last_seen_at": "2026-06-03T00:00:00Z",
                        "explanation_corpus": "", "last_explanation": "", "last_wrap_id": None}
            return None
        r = validate_graduations(
            text=text, valid_ids=set(), today="2026-06-04",
            pattern_history_lookup=lk,
        )
        assert r.carried_forward == []  # name_a's history NOT used to hold name_b
        assert r.demoted == 1

    def test_leading_1x_marker_does_not_cross_bind_to_later_3x(self):
        # codex L3 re-verify edge: the combined regex is any-level, but
        # _GRADUATION_RE is 2x/3x-only. A malformed line with a LEADING
        # 1x-with-evidence marker and a LATER 3x-with-evidence marker: the
        # combined regex binds name_a's 1x, validation matched name_b's 3x. The
        # level-alignment guard ("1" != "3") must reject the cross-bind so
        # name_a's warm-at-peak history is NOT used to hold name_b's 3x marker.
        text = (
            "## State\n.\n## Patterns\n"
            '- name_a | 1x (2026-06-04) [evidence: aaaaaaaa "seed one"] '
            'name_b | 3x (2026-06-04) [evidence: deadbeef "later marker"]\n'
            "## Decisions\n.\n## Context\n.\n"
        )
        def lk(name):
            if name == "name_a":
                return {"max_level_reached": 3, "last_seen_at": "2026-06-03T00:00:00Z",
                        "explanation_corpus": "", "last_explanation": "", "last_wrap_id": None}
            return None
        r = validate_graduations(
            text=text, valid_ids=set(), today="2026-06-04",
            pattern_history_lookup=lk,
        )
        assert r.carried_forward == []  # no cross-level cross-bind
        assert r.demoted == 1

    def test_sycophantic_overlap_not_carried_when_cold(self):
        # AM-PRESERVE-VS-SYCOPHANCY (warm-at-peak widening): a COLD at-peak pattern
        # with a DEAD citation whose explanation RE-WORDS prior-session vocabulary
        # (high overlap, NOT byte-identical) is NOT carried forward — the overlap
        # guard refuses protection → demote. A WARM at-peak reworded line is now
        # HELD (see test_carryforward_decision_holds_warm_reworded); the overlap
        # guard's surviving teeth are the COLD case. (Byte-identical is preservation
        # and held regardless of warmth.)
        corpus_prior = "standup consensus decision quick agreement architectural"
        reworded = "standup consensus decision rotated into fresh phrasing"  # 3 shared, not identical
        text = (
            "## State\n.\n## Patterns\n"
            f'- alpha | 3x (2026-06-04) [evidence: deadbeef "{reworded}"]\n'
            "## Decisions\n.\n## Context\n.\n"
        )
        def lk(name):
            if name == "alpha":
                return {"max_level_reached": 3, "last_seen_at": "2026-05-20T00:00:00Z",
                        "explanation_corpus": corpus_prior, "last_explanation": corpus_prior,
                        "last_wrap_id": None}
            return None
        r = validate_graduations(
            text=text, valid_ids=set(), today="2026-06-04",
            pattern_history_lookup=lk,
        )
        assert r.carried_forward == []  # cold re-worded overlap → not protected
        assert r.demoted == 1

    def test_distinct_vocab_dead_id_still_carried(self):
        # Control for the sycophancy check: a warm at-peak pattern with a dead
        # citation but a DISTINCT-vocabulary explanation (no overlap with prior)
        # is still legitimately held — the overlap guard is precise, not a
        # blanket dead-id veto.
        prior = "completely unrelated zebra giraffe elephant antelope"
        text = (
            "## State\n.\n## Patterns\n"
            '- alpha | 3x (2026-06-04) [evidence: deadbeef "governance topology substrate boundary"]\n'
            "## Decisions\n.\n## Context\n.\n"
        )
        def lk(name):
            if name == "alpha":
                return {"max_level_reached": 3, "last_seen_at": "2026-06-03T00:00:00Z",
                        "explanation_corpus": prior, "last_explanation": prior,
                        "last_wrap_id": None}
            return None
        r = validate_graduations(
            text=text, valid_ids=set(), today="2026-06-04",
            pattern_history_lookup=lk,
        )
        assert r.demoted == 0
        assert len(r.carried_forward) == 1

    def test_held_line_loses_evidence_tag(self):
        # Held line must NOT retain an [evidence:] tag — that's the property
        # that prevents it upserting pattern_history, so warmth decays on its
        # own and a chronically-failing pattern eventually ages out.
        r = validate_graduations(
            text=self._text(3), valid_ids=set(), today="2026-06-04",
            pattern_history_lookup=self._lookup(3, "2026-06-03T10:00:00Z"),
        )
        assert "[evidence:" not in self._line(r)

    def test_cross_session_overlap_never_carried_forward(self):
        # The sycophancy immune path demotes a COLD re-worded high-overlap pattern,
        # and that (cross-session-overlap) demotion is NEVER carried forward —
        # carryforward must not blunt the anti-sycophancy defense on the path that
        # DOES still fire. AM-PRESERVE-VS-SYCOPHANCY (warm-at-peak widening): a WARM
        # at-peak fresh-specifically-grounded reworded line is HELD instead (see
        # test_warm_reworded_fresh_specific_held); this pins that when the gate DOES
        # demote (cold here), carryforward does not catch it.
        corpus_prior = "standup consensus decision quick agreement architectural pattern"
        reworded = "standup consensus decision rotated into fresh phrasing again"  # 3 shared, not identical
        text = (
            "## State\n.\n## Patterns\n"
            f'- alpha | 3x (2026-06-04) [evidence: abc12345 "{reworded}"]\n'
            "## Decisions\n.\n## Context\n.\n"
        )
        node = {"abc12345": reworded}
        def lk(name):
            if name == "alpha":
                return {"max_level_reached": 3, "last_seen_at": "2026-05-20T00:00:00Z",
                        "explanation_corpus": corpus_prior, "last_explanation": corpus_prior,
                        "last_wrap_id": None}
            return None
        r = validate_graduations(
            text=text, valid_ids={"abc12345"}, today="2026-06-04",
            node_content_map=node, pattern_history_lookup=lk,
        )
        assert len(r.cross_session_collisions) == 1
        assert r.demoted == 1
        assert r.carried_forward == []
        assert "(cross-session-overlap)" in r.text
        assert "(carried-forward)" not in r.text

    def test_freeform_unnamed_line_not_carried(self):
        # A non-operator-named line (no _NAMED_PATTERN_RE match) has no
        # per-name history to consult -> demotes as before.
        text = (
            "## State\n.\n## Patterns\n"
            'thought: ACID compliance outweighs raw speed | 3x (2026-06-04) [evidence: deadbeef "x"]\n'
            "## Decisions\n.\n## Context\n.\n"
        )
        # Even with a permissive lookup, the unnamed line can't bind a name.
        r = validate_graduations(
            text=text, valid_ids=set(), today="2026-06-04",
            pattern_history_lookup=self._lookup(3, "2026-06-03T10:00:00Z"),
        )
        assert r.carried_forward == []


class TestPerNameLineBind:
    """AM-PERNAME-LINEBIND (v0.4.6): per-name functions bind level/date/
    evidence/declaration to the SAME physical line, not via name-keyed maps."""

    def test_combined_regex_binds_same_marker(self):
        from anneal_memory.graduation import _NAMED_PATTERN_WITH_EVIDENCE_RE as R
        m = R.match('  alpha | 3x (2026-06-04) [evidence: a1b2c3d4 "held here"]')
        assert m is not None
        assert (m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)) == (
            "alpha", "3", "2026-06-04", "a1b2c3d4", "held here")

    def test_combined_regex_two_marker_line_no_cross_bind(self):
        # First marker demoted (no evidence tag) + second marker has evidence.
        # The anchored combined regex must NOT bind the second marker's
        # evidence to the first name -> returns None (no upsert, no pollution).
        from anneal_memory.graduation import _NAMED_PATTERN_WITH_EVIDENCE_RE as R
        line = "  name_a | 3x (2026-06-04) (ungrounded) name_b | 2x (2026-06-04) [evidence: deadbeef \"second\"]"
        assert R.match(line) is None

    def test_combined_regex_no_evidence_no_match(self):
        from anneal_memory.graduation import _NAMED_PATTERN_WITH_EVIDENCE_RE as R
        assert R.match("- alpha | 2x (2026-06-04)") is None  # 1x/2x without evidence tag

    def test_duplicate_name_declaration_not_false_suppressed(self):
        # A 3x graduating line with NO contradiction stance, plus a 1x dup of
        # the SAME name carrying [no-contradicts]. The 1x dup's stance must NOT
        # satisfy the 3x line's missing declaration (the name-keyed-map bug).
        from anneal_memory.graduation import detect_proven_without_declaration
        prior = "## Patterns\n- alpha | 1x (2026-06-01) [evidence: abc12345 \"early\"]\n"
        new = (
            "## Patterns\n"
            '- alpha | 3x (2026-06-04) [evidence: abc12345 "graduated no stance"]\n'
            "- alpha | 1x (2026-06-04) [no-contradicts]\n"
        )
        flagged = detect_proven_without_declaration(prior, new, today="2026-06-04")
        names = [p.name for p in flagged]
        assert "alpha" in names
        assert flagged[0].level == 3  # the graduating line's level, not the dup's

    def test_well_formed_declared_graduation_not_flagged(self):
        # Control: a single well-formed 3x line WITH a stance is not flagged.
        from anneal_memory.graduation import detect_proven_without_declaration
        prior = "## Patterns\n- alpha | 2x (2026-06-01) [evidence: abc12345 \"prior\"]\n"
        new = '## Patterns\n- alpha | 3x (2026-06-04) [evidence: abc12345 "now"] [no-contradicts]\n'
        flagged = detect_proven_without_declaration(prior, new, today="2026-06-04")
        assert [p.name for p in flagged] == []


class TestVerbatimPreservation:
    """AM-PRESERVE-VS-SYCOPHANCY: a pattern carried forward VERBATIM (today's
    explanation byte-identical to a prior session's) is PRESERVATION, not
    sycophantic re-wording. Both sycophancy sites — the cross-session-overlap
    gate and ``_carryforward_decision``'s internal guard — must exempt it, while
    a RE-WORDED high-overlap explanation still trips them."""

    PRIOR = "standup consensus decision architecture sync quick agreement"
    REWORDED = "standup consensus decision quick agreement architectural"  # 5 shared, NOT identical

    def _lookup(self, *, max_level=3, last_seen="2026-05-21T00:00:00Z", corpus=None):
        corpus = corpus if corpus is not None else self.PRIOR

        def _l(name):
            if name == "alpha":
                return {
                    "max_level_reached": max_level,
                    "explanation_corpus": corpus,
                    "last_explanation": corpus,
                    "last_seen_at": last_seen,
                    "last_wrap_id": None,
                }
            return None
        return _l

    # -- the helper --
    def test_helper_identical_is_preservation(self):
        assert _is_verbatim_preservation("a b c", ["x y", "a b c"]) is True

    def test_helper_whitespace_reflow_is_preservation(self):
        # Re-wrapping changes whitespace, not words.
        assert _is_verbatim_preservation("a  b\n c", ["a b c"]) is True

    def test_helper_reworded_is_not_preservation(self):
        assert _is_verbatim_preservation("a c b d", ["a b c"]) is False

    def test_helper_empty_is_not_preservation(self):
        assert _is_verbatim_preservation("", ["a b c"]) is False
        assert _is_verbatim_preservation("a b c", []) is False

    # -- Site 2: the cross-session-overlap gate --
    def test_verbatim_preserved_not_cross_session_demoted(self):
        """Byte-identical explanation + a still-resolving citation: VALIDATED,
        not (cross-session-overlap) demoted."""
        text = (
            "## State\nverbatim carry.\n## Patterns\n"
            f'- alpha | 3x (2026-05-22) [evidence: abc12345 "{self.PRIOR}"]\n'
            "## Decisions\n.\n## Context\n.\n"
        )
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-22",
            node_content_map={"abc12345": self.PRIOR + " meeting"},
            pattern_history_lookup=self._lookup(),
        )
        assert result.demoted == 0
        assert "(cross-session-overlap)" not in result.text
        assert result.cross_session_collisions == []
        assert result.validated == 1

    def test_reworded_overlap_cold_still_cross_session_demoted(self):
        """A RE-WORDED high-overlap explanation (NOT byte-identical) that is COLD
        still demotes — the exemptions are byte-identity OR warm+fresh-specific
        grounding (AM-PRESERVE-VS-SYCOPHANCY warm-at-peak widening), NOT high
        overlap alone. Cold (last_seen 21d before today) → no warm exemption →
        demote. (The warm+fresh-specific HELD case is covered by
        ``test_warm_reworded_fresh_specific_held`` below.)"""
        text = (
            "## State\nreword.\n## Patterns\n"
            f'- alpha | 3x (2026-05-22) [evidence: abc12345 "{self.REWORDED}"]\n'
            "## Decisions\n.\n## Context\n.\n"
        )
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-22",
            node_content_map={"abc12345": self.REWORDED + " meeting"},
            pattern_history_lookup=self._lookup(last_seen="2026-05-01T00:00:00Z"),
        )
        assert result.demoted == 1
        assert "(cross-session-overlap)" in result.text
        assert len(result.cross_session_collisions) == 1

    # -- Site 1: carryforward's internal sycophancy-guard --
    def test_verbatim_preserved_ungrounded_carried_forward(self):
        """The real bug: an UNGROUNDED line (citation does not resolve this wrap)
        whose explanation is byte-identical to the prior is a warm, at-peak
        pattern preserved verbatim — carryforward HOLDS it instead of its own
        overlap-guard declining it into a demotion."""
        text = (
            "## State\nungrounded verbatim.\n## Patterns\n"
            f'- alpha | 3x (2026-05-22) [evidence: deadbee1 "{self.PRIOR}"]\n'
            "## Decisions\n.\n## Context\n.\n"
        )
        # deadbee1 is NOT in valid_ids -> ungrounded -> carryforward path.
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-22",
            node_content_map={},
            pattern_history_lookup=self._lookup(max_level=3),
            carryforward_cold_days=7,
        )
        assert result.demoted == 0
        assert len(result.carried_forward) == 1
        assert "(carried-forward)" in result.text
        assert "alpha | 3x" in result.text  # level HELD, not decremented

    def test_carryforward_decision_holds_verbatim(self):
        """Unit: _carryforward_decision returns a CarriedForward (hold) for a
        verbatim-preserved explanation its sycophancy-guard would otherwise
        decline."""
        line = f'- alpha | 3x (2026-05-22) [evidence: deadbee1 "{self.PRIOR}"]'
        held = _carryforward_decision(
            line=line,
            level=3,
            today="2026-05-22",
            pattern_history_lookup=self._lookup(max_level=3),
            carryforward_cold_days=7,
        )
        assert isinstance(held, CarriedForward)
        assert held.held_level == 3

    def test_carryforward_decision_declines_reworded_when_cold(self):
        """Unit regression: a RE-WORDED high-overlap explanation that is COLD is
        still declined (returns None -> demotes). AM-PRESERVE-VS-SYCOPHANCY
        warm-at-peak widening: a WARM at-peak reworded line is now HELD at Site 2
        (carryforward holds it WITHOUT upserting, so warmth self-decays and it
        ages out if the citation stays dead) — see
        ``test_carryforward_decision_holds_warm_reworded``. The overlap refusal's
        surviving teeth are the COLD case."""
        line = f'- alpha | 3x (2026-05-22) [evidence: deadbee1 "{self.REWORDED}"]'
        held = _carryforward_decision(
            line=line,
            level=3,
            today="2026-05-22",
            pattern_history_lookup=self._lookup(max_level=3, last_seen="2026-05-01T00:00:00Z"),
            carryforward_cold_days=7,
        )
        assert held is None

    # -- AM-PRESERVE-VS-SYCOPHANCY warm-at-peak widening: the NEW behavior --
    def test_warm_reworded_fresh_specific_held(self):
        """THE CORE FIX: a WARM, at-peak, RE-WORDED explanation that grounds in a
        fresh episode via a word ABSENT from the prior corpus ('architectural' ∉
        prior) is purifying selection — re-validation against fresh reality — and
        is HELD (validated), not demoted. This is the load-bearing-antibody case
        (verify_or_surface / four_layer_apparatus) the gate eroded one level per
        wrap because a stable principle's core vocabulary necessarily recurs."""
        text = (
            "## State\nwarm reword.\n## Patterns\n"
            f'- alpha | 3x (2026-05-22) [evidence: abc12345 "{self.REWORDED}"]\n'
            "## Decisions\n.\n## Context\n.\n"
        )
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-22",
            node_content_map={"abc12345": self.REWORDED + " meeting"},
            pattern_history_lookup=self._lookup(),  # default warm (last_seen 2026-05-21)
        )
        assert result.validated == 1
        assert result.demoted == 0
        assert "(cross-session-overlap)" not in result.text
        assert result.cross_session_collisions == []

    def test_warm_reworded_core_vocab_only_demoted(self):
        """codex L3 F1 (fresh-specific grounding): a WARM, at-peak, reworded line
        whose grounding shares ONLY prior-corpus vocabulary with the cited episode
        (no novel word) is a cheap longevity-pump signature, not re-validation —
        DEMOTED. Closes the warm-stickiness pump: a warm-held line VALIDATES →
        upserts last_seen → would otherwise stay warm forever on cheap fresh text."""
        text = (
            "## State\npump.\n## Patterns\n"
            f'- alpha | 3x (2026-05-22) [evidence: abc12345 "{self.REWORDED}"]\n'
            "## Decisions\n.\n## Context\n.\n"
        )
        # node shares ONLY PRIOR-vocab words with the explanation (no novel grounding)
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-22",
            node_content_map={"abc12345": "standup consensus decision quick agreement meeting"},
            pattern_history_lookup=self._lookup(),  # warm
        )
        assert result.demoted == 1
        assert "(cross-session-overlap)" in result.text

    def test_warm_reworded_two_ids_only_one_grounds_no_poison(self):
        """codex L3 F2: a preservation-exempt (warm+fresh-specific) line citing TWO
        ids where only ONE grounds the explanation must NOT form a Hebbian
        co-citation between the grounding episode and the unrelated co-cited one.
        Link only individually-grounding ids."""
        text = (
            "## State\nco-cite.\n## Patterns\n"
            f'- alpha | 3x (2026-05-22) [evidence: abc12345, def67890 "{self.REWORDED}"]\n'
            "## Decisions\n.\n## Context\n.\n"
        )
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345", "def67890"},
            today="2026-05-22",
            node_content_map={
                "abc12345": self.REWORDED + " meeting",   # grounds (novel: architectural)
                "def67890": "wholly unrelated runtime telemetry payload",  # does NOT ground
            },
            pattern_history_lookup=self._lookup(),  # warm
        )
        assert result.validated == 1            # held via warm + fresh-specific
        assert result.direct_co_citations == []  # only 1 id grounds -> no pair -> no poison edge

    def test_cold_byte_identical_still_held(self):
        """L1 coverage gap: byte-identical preservation is exempt regardless of
        warmth (it cannot pump — it never changes). A COLD byte-identical
        explanation is still HELD, so a future refactor can't silently drop
        _is_verbatim_preservation thinking the warm clause subsumes it."""
        text = (
            "## State\ncold verbatim.\n## Patterns\n"
            f'- alpha | 3x (2026-05-22) [evidence: abc12345 "{self.PRIOR}"]\n'
            "## Decisions\n.\n## Context\n.\n"
        )
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-22",
            node_content_map={"abc12345": self.PRIOR + " meeting"},
            pattern_history_lookup=self._lookup(last_seen="2026-05-01T00:00:00Z"),  # COLD
        )
        assert result.validated == 1
        assert result.demoted == 0
        assert "(cross-session-overlap)" not in result.text

    def test_carryforward_decision_holds_warm_reworded(self):
        """Site 2 warm-at-peak widening: a WARM, at-peak, RE-WORDED explanation on
        the carryforward (dead-citation) path is now HELD — the overlap refusal is
        skipped for warm patterns. carryforward holds WITHOUT upserting (the held
        line loses its evidence tag), so warmth self-decays and a perpetually-dead
        pattern still ages out — no pump (codex L3: Site 2 cannot fuel the
        longevity pump Site 1's novelty gate guards)."""
        line = f'- alpha | 3x (2026-05-22) [evidence: deadbee1 "{self.REWORDED}"]'
        held = _carryforward_decision(
            line=line,
            level=3,
            today="2026-05-22",
            pattern_history_lookup=self._lookup(max_level=3),  # default warm
            carryforward_cold_days=7,
        )
        assert isinstance(held, CarriedForward)
        assert held.held_level == 3

    # -- codex L3 (HIGH + missing-coverage) regressions --
    def test_verbatim_but_ungrounded_live_ids_demoted_no_poison(self):
        """codex L3 HIGH: a byte-identical explanation whose cited LIVE ids do NOT
        ground it (explanation_valid False) must demote (cross-session-overlap) and
        form NO co-citation link — an exact-copy citing unrelated real episodes
        would otherwise poison the Hebbian graph."""
        text = (
            "## State\npoison attempt.\n## Patterns\n"
            f'- alpha | 3x (2026-05-22) [evidence: abc12345, def67890 "{self.PRIOR}"]\n'
            "## Decisions\n.\n## Context\n.\n"
        )
        # Both ids resolve, but neither episode's content grounds the explanation.
        node = {"abc12345": "completely unrelated runtime telemetry payload",
                "def67890": "another orthogonal database migration note"}
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345", "def67890"},
            today="2026-05-22",
            node_content_map=node,
            pattern_history_lookup=self._lookup(max_level=3),
        )
        assert result.demoted == 1
        assert "(cross-session-overlap)" in result.text
        assert result.carried_forward == []
        assert result.direct_co_citations == []  # no poison link forged

    def test_verbatim_level_up_still_demoted(self):
        """A byte-identical LEVEL-UP (claim a new high with the exact prior words
        + a resolving episode) is NOT preservation — it inflates past the earned
        mark and must still cross-session-demote."""
        text = (
            "## State\nlevel up.\n## Patterns\n"
            f'- alpha | 3x (2026-05-22) [evidence: abc12345 "{self.PRIOR}"]\n'
            "## Decisions\n.\n## Context\n.\n"
        )
        # max_level=2 -> claiming 3x is inflation; the exemption's level gate fails.
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-22",
            node_content_map={"abc12345": self.PRIOR + " meeting"},
            pattern_history_lookup=self._lookup(max_level=2),
        )
        assert result.demoted == 1
        assert "(cross-session-overlap)" in result.text

    def test_missing_max_level_demotes_at_site2(self):
        """No integer max_level_reached -> the Site 2 exemption does not apply
        (conservative) -> a byte-identical overlap still demotes."""
        text = (
            "## State\nno max.\n## Patterns\n"
            f'- alpha | 3x (2026-05-22) [evidence: abc12345 "{self.PRIOR}"]\n'
            "## Decisions\n.\n## Context\n.\n"
        )
        def lk(name):
            if name == "alpha":
                # NOTE: no max_level_reached key.
                return {"explanation_corpus": self.PRIOR, "last_explanation": self.PRIOR,
                        "last_seen_at": "2026-05-21T00:00:00Z", "last_wrap_id": None}
            return None
        result = validate_graduations(
            text=text, valid_ids={"abc12345"}, today="2026-05-22",
            node_content_map={"abc12345": self.PRIOR + " meeting"},
            pattern_history_lookup=lk,
        )
        assert result.demoted == 1
        assert "(cross-session-overlap)" in result.text

    def test_carryforward_decision_declines_verbatim_level_up(self):
        """L1: a byte-identical explanation claiming a level ABOVE the earned mark
        is declined at Site 1 — the `level > max_level` check fires after the
        verbatim exemption. Pins the anti-inflation clause for the verbatim path
        (the single most security-relevant line in the change)."""
        line = f'- alpha | 3x (2026-05-22) [evidence: deadbee1 "{self.PRIOR}"]'
        held = _carryforward_decision(
            line=line,
            level=3,
            today="2026-05-22",
            pattern_history_lookup=self._lookup(max_level=2),  # earned only 2x
            carryforward_cold_days=7,
        )
        assert held is None

    def test_omitted_node_content_map_no_vacuous_exemption(self):
        """codex L3 convergence (MED): a direct caller passing valid_ids + history
        but with NO usable node_content_map (None OR empty {}) must NOT earn the
        preservation exemption via explanation_valid's vacuous default — the
        exact-copy poison stays closed (demote, no co-citation link)."""
        text = (
            "## State\nno content map.\n## Patterns\n"
            f'- alpha | 3x (2026-05-22) [evidence: abc12345, def67890 "{self.PRIOR}"]\n'
            "## Decisions\n.\n## Context\n.\n"
        )
        for ncm in (None, {}):  # grounding cannot be evaluated in either case
            result = validate_graduations(
                text=text,
                valid_ids={"abc12345", "def67890"},
                today="2026-05-22",
                node_content_map=ncm,
                pattern_history_lookup=self._lookup(max_level=3),
            )
            assert result.demoted == 1, f"node_content_map={ncm!r}"
            assert "(cross-session-overlap)" in result.text, f"node_content_map={ncm!r}"
            assert result.direct_co_citations == [], f"node_content_map={ncm!r}"


class TestBareCarryforward:
    """AM-PRESERVE-BARE-PATH (v0.5.0): the bare-graduation sunset path now HOLDS
    a load-bearing Proven (at/below its earned high-water mark AND warm) instead
    of ratcheting it down, exactly as AM-CARRYFORWARD (v0.4.6) does for the cited
    ungrounded path. A bare line carried forward without a fresh citation is the
    common case (the agent didn't re-exercise the pattern this wrap, so it has no
    current-window episode to cite) — pre-fix it eroded one level every wrap.
    Brand-new bald claims (no history / unearned / cold) still sunset.

    Regression anchor: flow's 2026-06-05 wrap bare-demoted
    verify_or_surface_before_acting 3x->1x (max_level_reached=3, warm) while the
    cited structural_invariants_beat_discipline was correctly held by v0.4.6."""

    def _lookup(self, max_level, last_seen, name="alpha"):
        def _l(n):
            if n == name:
                return {
                    "max_level_reached": max_level,
                    "last_seen_at": last_seen,
                    "explanation_corpus": "",
                    "last_explanation": "",
                    "last_wrap_id": None,
                }
            return None
        return _l

    def _text(self, level, *, dashed=False, trailing=""):
        # Bare graduation: NO [evidence:] tag. Flow's real format is an indented,
        # bullet-less grouped member; dashed=True exercises the canonical
        # stranger-adopter "- name | Nx" form (the generality regression).
        prefix = "- " if dashed else "  "
        return (
            "## State\n.\n## Patterns\n"
            f"{prefix}alpha | {level}x (2026-06-04){trailing}\n"
            "## Decisions\n.\n## Context\n.\n"
        )

    def _line(self, result):
        return next(ln for ln in result.text.splitlines() if "alpha |" in ln)

    def _run(self, text, lookup, *, today="2026-06-04", cold_days=7):
        return validate_graduations(
            text=text, valid_ids=set(), today=today, citations_seen=True,
            pattern_history_lookup=lookup, carryforward_cold_days=cold_days,
        )

    def test_warm_at_peak_bare_held(self):
        r = self._run(self._text(3), self._lookup(3, "2026-06-02T10:00:00Z"))
        assert r.bare_demoted == 0
        assert r.demoted == 0
        assert len(r.carried_forward) == 1
        cf = r.carried_forward[0]
        assert (cf.name, cf.held_level, cf.max_level_reached, cf.days_since_grounded) == (
            "alpha", 3, 3, 2)
        assert cf.cited is False  # bare path: no citation -> excluded from AM-WARN count
        assert "| 3x (2026-06-04) (carried-forward)" in self._line(r)
        assert "(needs-evidence)" not in r.text
        # Level was NOT decremented.
        assert "| 2x" not in r.text and "| 1x" not in r.text

    def test_cold_bare_ages_out(self):
        r = self._run(self._text(3), self._lookup(3, "2026-05-01T10:00:00Z"))
        assert r.bare_demoted == 1
        assert r.carried_forward == []
        assert "| 2x (2026-06-04) (needs-evidence)" in self._line(r)
        assert "(carried-forward)" not in r.text

    def test_unearned_bare_demotes(self):
        # Bare 3x but high-water mark is only 2 -- never earned 3x (inflation
        # block: a bald level-UP has no protected high-water).
        r = self._run(self._text(3), self._lookup(2, "2026-06-03T10:00:00Z"))
        assert r.bare_demoted == 1
        assert r.carried_forward == []

    def test_protects_below_peak_bare(self):
        # Bare 2x, earned 3x before, warm -> held at 2x (level <= max_level).
        r = self._run(self._text(2), self._lookup(3, "2026-06-03T10:00:00Z"))
        assert r.bare_demoted == 0
        assert len(r.carried_forward) == 1
        assert r.carried_forward[0].held_level == 2
        assert "| 2x (2026-06-04) (carried-forward)" in self._line(r)

    def test_no_history_bare_demotes(self):
        # The genuinely-new bald "alpha | 3x" with no history -> sunset. This is
        # the case the fail-safe sunset exists for; carryforward must not shield it.
        r = self._run(self._text(3), lambda n: None)
        assert r.bare_demoted == 1
        assert r.carried_forward == []

    def test_disabled_via_none_bare_demotes_even_when_warm(self):
        r = self._run(self._text(3), self._lookup(3, "2026-06-04T10:00:00Z"), cold_days=None)
        assert r.bare_demoted == 1
        assert r.carried_forward == []

    def test_no_lookup_wired_bare_demotes(self):
        # pattern_history_lookup defaults to None -> bare carryforward inert.
        r = validate_graduations(
            text=self._text(3), valid_ids=set(), today="2026-06-04",
            citations_seen=True,
        )
        assert r.bare_demoted == 1
        assert r.carried_forward == []

    def test_ignored_without_citations_seen_even_when_warm(self):
        # If the store has not seen citations, the bare sunset never runs -- so
        # neither does the carryforward. Nothing touched.
        r = validate_graduations(
            text=self._text(3), valid_ids=set(), today="2026-06-04",
            citations_seen=False, pattern_history_lookup=self._lookup(3, "2026-06-03T10:00:00Z"),
        )
        assert r.bare_demoted == 0
        assert r.carried_forward == []
        assert "(carried-forward)" not in r.text

    def test_dashed_stranger_adopter_bare_held(self):
        # GENERALITY regression (the dogfood discriminator, in a test): the
        # canonical "- name | Nx" dashed format -- a stranger adopter who never
        # used flow's indented grouping -- is held identically. The fix keys on
        # pattern_history + _NAMED_PATTERN_RE, not on flow's format. This is why
        # AM-PRESERVE-BARE-PATH is a general bug, not a flow-N-of-1 fix.
        r = self._run(self._text(3, dashed=True), self._lookup(3, "2026-06-03T10:00:00Z"))
        assert r.bare_demoted == 0
        assert len(r.carried_forward) == 1
        assert "- alpha | 3x (2026-06-04) (carried-forward)" in self._line(r)

    def test_far_future_last_seen_bare_rejected(self):
        # A last_seen_at clearly in the future (> 1 day) is untrustworthy --
        # do not read as maximally warm; conservative-demote.
        r = self._run(self._text(3), self._lookup(3, "2026-06-10T00:00:00Z"))
        assert r.bare_demoted == 1
        assert r.carried_forward == []

    def test_one_day_utc_skew_bare_tolerated(self):
        # The legitimate <=1-day UTC-vs-local skew is tolerated as warm -> held.
        r = self._run(self._text(3), self._lookup(3, "2026-06-05T01:00:00Z"))
        assert r.bare_demoted == 0
        assert len(r.carried_forward) == 1
        assert r.carried_forward[0].days_since_grounded == 0  # clamped, not -1

    def test_cold_boundary_inclusive_bare(self):
        # Exactly cold_days (7) -> warm (held); cold_days+1 -> aged out.
        def text(d):
            return (
                "## State\n.\n## Patterns\n"
                f"  alpha | 3x ({d})\n"
                "## Decisions\n.\n## Context\n.\n"
            )
        held = validate_graduations(
            text=text("2026-06-08"), valid_ids=set(), today="2026-06-08",
            citations_seen=True,
            pattern_history_lookup=self._lookup(3, "2026-06-01T00:00:00Z"),
            carryforward_cold_days=7,
        )
        assert held.bare_demoted == 0 and len(held.carried_forward) == 1
        aged = validate_graduations(
            text=text("2026-06-09"), valid_ids=set(), today="2026-06-09",
            citations_seen=True,
            pattern_history_lookup=self._lookup(3, "2026-06-01T00:00:00Z"),
            carryforward_cold_days=7,
        )
        assert aged.bare_demoted == 1 and aged.carried_forward == []

    def test_non_today_bare_skipped_not_held(self):
        # A bare line whose date != today is a legitimately carried-forward
        # pattern the agent did NOT re-stamp -- already skipped by the sunset
        # (skipped_non_today), never reaching the carryforward path. No erosion,
        # no hold; the line is left exactly as-is. (This is why branch-(c) of
        # AM-PRESERVE-VS-SYCOPHANCY -- "keep the original date" -- is the
        # behavioral prevention and this fix is the structural backstop for when
        # the agent re-stamps anyway.)
        r = validate_graduations(
            text=self._text(3), valid_ids=set(), today="2026-06-05",
            citations_seen=True,
            pattern_history_lookup=self._lookup(3, "2026-06-02T10:00:00Z"),
        )
        assert r.bare_demoted == 0
        assert r.carried_forward == []
        assert r.skipped_non_today == 1
        assert "| 3x (2026-06-04)" in self._line(r)  # untouched
        assert "(carried-forward)" not in r.text

    def test_held_bare_preserves_trailing_prose(self):
        # flow's real lines carry " - explanation" after the marker; the rewrite
        # must keep it with a single separating space and not mangle it.
        r = self._run(
            self._text(3, trailing=" — cached state is a SNAPSHOT"),
            self._lookup(3, "2026-06-03T10:00:00Z"),
        )
        assert len(r.carried_forward) == 1
        assert (
            "| 3x (2026-06-04) (carried-forward) — cached state is a SNAPSHOT"
            in self._line(r)
        )

    def test_no_space_pipe_variant_bare_held(self):
        # "|3x" (no space after pipe) must bind + hold correctly -- the
        # Diogenes-LOW spacing class, on the hold path.
        r = self._run(
            "## State\n.\n## Patterns\n  alpha |3x (2026-06-04)\n## Decisions\n.\n## Context\n.\n",
            self._lookup(3, "2026-06-03T10:00:00Z"),
        )
        assert r.bare_demoted == 0
        assert len(r.carried_forward) == 1
        assert "(carried-forward)" in self._line(r)
        # The original level marker survives intact (not decremented).
        assert "| 2x" not in r.text and "| 1x" not in r.text

    def test_bare_level_misalign_not_held(self):
        # Malformed two-marker bare line: line-start name_a is 1x (does not match
        # _BARE_GRADUATION_RE, which is 2x/3x-only); a LATER name_b | 3x is the
        # marker the sunset found. _NAMED_PATTERN_RE binds name_a @ "1"; the level
        # guard ("1" != "3") declines, so name_a's warm-at-peak history is NOT
        # used to hold name_b's 3x. Falls through to sunset. (Bare-path mirror of
        # test_leading_1x_marker_does_not_cross_bind_to_later_3x.)
        text = (
            "## State\n.\n## Patterns\n"
            "  name_a | 1x (2026-06-04) name_b | 3x (2026-06-04)\n"
            "## Decisions\n.\n## Context\n.\n"
        )
        def lk(n):
            if n == "name_a":
                return {"max_level_reached": 3, "last_seen_at": "2026-06-03T00:00:00Z",
                        "explanation_corpus": "", "last_explanation": "", "last_wrap_id": None}
            return None
        r = validate_graduations(
            text=text, valid_ids=set(), today="2026-06-04", citations_seen=True,
            pattern_history_lookup=lk,
        )
        assert r.carried_forward == []  # name_a history NOT used to hold name_b
        assert r.bare_demoted == 1

    def test_rehold_idempotent_no_duplicate_marker(self):
        # A line held in a prior wrap keeps its `(carried-forward)` marker in the
        # saved text; the agent re-emits it re-stamped to today. Re-holding must
        # NOT accrete a second marker (structural_invariants over trusting the
        # agent to strip it). Exactly one `(carried-forward)`, prose preserved.
        line_body = "  alpha | 3x (2026-06-04) (carried-forward) — preserved prose"
        text = f"## State\n.\n## Patterns\n{line_body}\n## Decisions\n.\n## Context\n.\n"
        r = self._run(text, self._lookup(3, "2026-06-03T10:00:00Z"))
        assert r.bare_demoted == 0
        assert len(r.carried_forward) == 1
        held = self._line(r)
        assert held.count("(carried-forward)") == 1  # not duplicated
        assert held == "  alpha | 3x (2026-06-04) (carried-forward) — preserved prose"

    def test_rehold_idempotent_no_trailing_prose(self):
        # Same, but the held line had no trailing prose: `... (carried-forward)`
        # alone must re-hold to exactly one marker, no dangling double space.
        text = (
            "## State\n.\n## Patterns\n"
            "  alpha | 3x (2026-06-04) (carried-forward)\n"
            "## Decisions\n.\n## Context\n.\n"
        )
        r = self._run(text, self._lookup(3, "2026-06-03T10:00:00Z"))
        assert len(r.carried_forward) == 1
        held = self._line(r)
        assert held.count("(carried-forward)") == 1
        assert held == "  alpha | 3x (2026-06-04) (carried-forward)"

    def test_same_level_multimarker_cross_bind_not_held(self):
        # codex L3 + complement L1 (MEDIUM, convergent): two markers at the SAME
        # level on one malformed line. name_a (warm at-peak) is the line-start
        # name; name_b's marker is what the sunset finds (name_a's has malformed
        # `[evidence:]` so the tightened bare regex skips it). A level-STRING
        # guard ("2"=="2") would hold name_b's marker using name_a's history; the
        # SPAN-alignment guard declines (different marker positions). Without the
        # fix `carried_forward` would credit name_a — with it, no cross-bind.
        text = (
            "## State\n.\n## Patterns\n"
            "  name_a | 2x (2026-06-04)[evidence: nothex] name_b | 2x (2026-06-04)\n"
            "## Decisions\n.\n## Context\n.\n"
        )
        def lk(n):
            if n == "name_a":
                return {"max_level_reached": 2, "last_seen_at": "2026-06-03T00:00:00Z",
                        "explanation_corpus": "", "last_explanation": "", "last_wrap_id": None}
            return None  # name_b has no history
        r = validate_graduations(
            text=text, valid_ids=set(), today="2026-06-04", citations_seen=True,
            pattern_history_lookup=lk,
        )
        assert r.carried_forward == []  # name_a's history NOT used to hold name_b

    def test_malformed_evidence_line_not_classified_bare(self):
        # codex L3 (MEDIUM): `| 3x (date) [evidence: nothex]` — invalid-hex
        # evidence that _GRADUATION_RE rejects — must NOT be misclassified as a
        # BARE line and held. The tightened _BARE_GRADUATION_RE lookahead spans
        # the optional space, so an evidence-bearing line never matches bare. The
        # line is left untouched (not held, not bare-demoted).
        text = (
            "## State\n.\n## Patterns\n"
            "  alpha | 3x (2026-06-04) [evidence: nothex]\n"
            "## Decisions\n.\n## Context\n.\n"
        )
        r = self._run(text, self._lookup(3, "2026-06-03T10:00:00Z"))
        assert r.carried_forward == []
        assert r.bare_demoted == 0
        assert "(carried-forward)" not in r.text
        assert "  alpha | 3x (2026-06-04) [evidence: nothex]" in r.text  # untouched

    def test_rehold_strips_stale_needs_evidence(self):
        # codex L3 + complement L1 (LOW/info, convergent): a line previously
        # sunset to `(needs-evidence)` then re-stamped at its (demoted) level,
        # warm + at/below max_level, is HELD — and the stale `(needs-evidence)`
        # is stripped, NOT left as a contradictory `(carried-forward) (needs-evidence)`.
        text = (
            "## State\n.\n## Patterns\n"
            "  alpha | 2x (2026-06-04) (needs-evidence)\n"
            "## Decisions\n.\n## Context\n.\n"
        )
        r = self._run(text, self._lookup(3, "2026-06-03T10:00:00Z"))  # earned 3x
        assert len(r.carried_forward) == 1
        held = self._line(r)
        assert "(carried-forward)" in held
        assert "(needs-evidence)" not in held
        assert held == "  alpha | 2x (2026-06-04) (carried-forward)"

    def test_rehold_collapses_preexisting_duplicate_run(self):
        # complement L1 FINDING-4: the strip removes the WHOLE leading run, so a
        # pre-existing duplicate (from before this fix, or a prior cross-bind)
        # self-heals to exactly one marker rather than netting strip-one-add-one.
        text = (
            "## State\n.\n## Patterns\n"
            "  alpha | 3x (2026-06-04) (carried-forward) (carried-forward) — prose\n"
            "## Decisions\n.\n## Context\n.\n"
        )
        r = self._run(text, self._lookup(3, "2026-06-03T10:00:00Z"))
        assert len(r.carried_forward) == 1
        held = self._line(r)
        assert held.count("(carried-forward)") == 1
        assert held == "  alpha | 3x (2026-06-04) (carried-forward) — prose"

    def test_sunset_positional_does_not_touch_skipped_marker(self):
        # codex L3 convergence (LOW): the bare SUNSET must rewrite only the
        # matched span. On `name_a | 2x ...[evidence: nothex] name_b | 2x ...`
        # (identical marker texts), the prior global `line.replace` demoted BOTH
        # — including name_a's, which the tightened bare regex correctly skipped
        # (evidence follows it). Positional rewrite touches only name_b.
        text = (
            "## State\n.\n## Patterns\n"
            "  name_a | 2x (2026-06-04)[evidence: nothex] name_b | 2x (2026-06-04)\n"
            "## Decisions\n.\n## Context\n.\n"
        )
        r = validate_graduations(
            text=text, valid_ids=set(), today="2026-06-04", citations_seen=True,
            pattern_history_lookup=lambda n: None,  # neither has history -> sunset
        )
        line = self._line_named(r, "name_a")
        assert "name_a | 2x (2026-06-04)[evidence: nothex]" in line  # name_a untouched
        assert "name_b | 1x (2026-06-04) (needs-evidence)" in line   # only name_b demoted
        assert line.count("(needs-evidence)") == 1
        assert r.bare_demoted == 1

    def _line_named(self, result, needle):
        return next(ln for ln in result.text.splitlines() if needle in ln)


class TestPatternSummary:
    """AM-SEMDUP (v0.5.0): _pattern_summary — a compact meaning-snippet that
    survives the format variety of real pattern lines."""

    def test_felt_prose_form(self):
        line = (
            "  verify_or_surface | 2x (2026-06-05) (cross-session-overlap) "
            "[no-contradicts] — cached state is a SNAPSHOT; verify or surface. [Proven]"
        )
        s = _pattern_summary(line)
        assert s.startswith("cached state is a SNAPSHOT")
        # scaffolding stripped
        assert "[no-contradicts]" not in s
        assert "cross-session-overlap" not in s
        assert "[Proven]" not in s

    def test_evidence_quote_fallback(self):
        line = (
            '- acid_over_speed | 2x (2026-06-05) '
            '[evidence: 4931b6a8 "PostgreSQL chosen for ACID guarantees"]'
        )
        # No trailing felt prose -> falls back to the quoted evidence explanation.
        assert _pattern_summary(line) == "PostgreSQL chosen for ACID guarantees"

    def test_bare_line_has_empty_summary(self):
        assert _pattern_summary("- bare_dev | 1x (2026-06-05)") == ""

    def test_truncation_adds_ellipsis(self):
        long = "x" * 300
        line = f"- p | 1x (2026-06-05) — {long}"
        s = _pattern_summary(line, max_chars=50)
        assert len(s) == 50
        assert s.endswith("…")

    def test_truncation_word_boundary_is_le_not_eq(self):
        # L1 L3: with spaces near the cut, rstrip yields <= max_chars (not ==);
        # the prior == assertion was accidentally specific to space-free input.
        line = "- p | 1x (2026-06-05) — word " + "y " * 40
        s = _pattern_summary(line, max_chars=20)
        assert len(s) <= 20
        assert s.endswith("…")

    def test_wikilink_reference_preserved(self):
        # L1 H1 (live-data bug): a [[sibling]] ref must keep the referenced NAME,
        # not leave a stray "]" and lose the reference.
        line = ("- decompose | 2x (2026-06-05) — separate concerns. "
                "Sibling of [[invisible_infrastructure_failure]].")
        s = _pattern_summary(line)
        assert "invisible_infrastructure_failure" in s
        assert "[" not in s and "]" not in s

    def test_dateless_line_no_scaffolding_leak(self):
        # L1 M1: a graduated line with no (date) must not leak "name | Nx".
        assert _pattern_summary("- no_date_pat | 3x — a real principle") == \
            "a real principle"

    def test_legit_bracketed_prose_preserved(self):
        # L2 HIGH: only KNOWN scaffolding tags are stripped — legitimate
        # bracketed prose survives (the broad strip would have deleted it).
        s = _pattern_summary("- p | 1x (2026-06-05) — a principle [rare edge case] applies")
        assert s == "a principle [rare edge case] applies"

    def test_flowscript_marker_prefix(self):
        # A "! urgent" FlowScript-prefixed line still summarizes cleanly.
        s = _pattern_summary("  ! urgent_pat | 3x (2026-06-05) — drop everything")
        assert s == "drop everything"

    def test_evidence_explanation_with_bracket_is_quote_aware(self):
        # codex L3 M1: an [evidence:] explanation containing "]" must NOT be
        # cut at the inner "]" (which left garbage prose + suppressed the
        # evidence fallback). Quote-aware strip returns the full explanation.
        s = _pattern_summary(
            '- p | 2x (2026-06-05) [evidence: abc12345 "works for bracketed [rare] cases"]'
        )
        assert s == "works for bracketed [rare] cases"

    def test_unclosed_scaffold_tags_do_not_blow_up(self):
        # codex L3 M2: a corrupted line with many unclosed "[evidence:" must not
        # make the scaffold strip quadratic. The {0,512} bound keeps it linear;
        # 30k repeats completes well under a second (was ~quadratic before).
        import time
        line = "- p | 1x (2026-06-05) — " + ("[evidence:" * 30000)
        t = time.time()
        s = _pattern_summary(line)
        assert time.time() - t < 1.5  # bounded; unbounded-quadratic ~2.6s+
        assert isinstance(s, str)

    def test_unterminated_quote_yields_empty_not_garbage(self):
        # codex L3 convergence: an [evidence:] tag with an UNTERMINATED quote must
        # strip to empty (the safe fallback), NOT leak the raw tag as garbage. The
        # quote-aware regex form regressed this (empty -> garbage); the span +
        # bounded-mop-up architecture restores empty.
        s = _pattern_summary('- p | 2x (2026-06-05) [evidence: abc123 "unterminated]')
        assert s == ""

    def test_malformed_evidence_keeps_trailing_felt_prose(self):
        # A malformed evidence tag is mopped up to its first "]"; felt prose after
        # it survives.
        s = _pattern_summary('- p | 2x (2026-06-05) [evidence: bad "unterm] — felt tail')
        assert s == "felt tail"

    def test_multimarker_binds_first_markers_meaning_not_a_later_one(self):
        # codex L3 convergence #2: the evidence binding is ANCHORED to the FIRST
        # marker — a malformed multi-marker line must NOT bind a LATER marker's
        # explanation as this name's summary (the AM-PERNAME-LINEBIND principle).
        s = _pattern_summary(
            "- first | 2x (2026-06-05) — FIRST FELT. "
            'second | 2x (2026-06-05) [evidence: abc12345 "SECOND EXPL"]'
        )
        assert "FIRST FELT" in s
        assert "SECOND EXPL" not in s

    def test_evidence_with_no_closing_bracket_yields_empty_not_garbage(self):
        # codex L3 convergence #2: a fully-unclosed "[evidence:" (no "]" at all)
        # must strip to empty, not leak the raw tag.
        assert _pattern_summary('- p | 2x (2026-06-05) [evidence: abc123 "unterminated') == ""
        assert _pattern_summary('- p | 2x (2026-06-05) [evidence: abc "x — felt') == ""

    def test_unclosed_evidence_longer_than_512_yields_empty(self):
        # codex L3 convergence #3: an unclosed evidence body LONGER than any fixed
        # bound must still strip to empty — the unbounded "[^\]]*\]?" form consumes
        # the whole tag to EOL (a {0,512} bound leaked the overflow tail).
        assert _pattern_summary("- p | 2x (2026-06-05) [evidence: " + "a" * 512) == ""
        assert _pattern_summary("- p | 2x (2026-06-05) " + ("[evidence:" * 53)) == ""

    def test_scaffold_strip_is_linear_on_megabyte_garbage(self):
        # The "always-succeeds" match means .sub() never fail-rescans => linear,
        # not quadratic, even on a 1 MB unclosed body.
        import time
        line = "- p | 1x (2026-06-05) [evidence: " + ("a" * 1_000_000)
        t = time.time()
        s = _pattern_summary(line)
        assert time.time() - t < 1.0  # ~0.02s linear; quadratic would be minutes
        assert s == ""


class TestExtractPatternSummaries:
    """AM-SEMDUP (v0.5.0): extract_pattern_summaries — the corpus the
    merge-don't-fork dedup scan surfaces."""

    GRAD = graduating_headings(FLOW_SCHEMA)

    TEXT = (
        "# x — Memory (v1)\n"
        "## Patterns\n"
        "{{topic:\n"
        "  verify_or_surface | 2x (2026-06-05) [no-contradicts] — cached state is a "
        "SNAPSHOT; verify or surface.\n"
        "}}\n"
        "- acid_over_speed | 3x (2026-06-05) [evidence: 4931b6a8 \"ACID over speed\"]\n"
        "- bare_dev | 1x (2026-06-05)\n"
        "## Decisions\n"
        "- some_decision | 2x (2026-06-05) [evidence: deadbeef \"not a pattern\"]\n"
    )

    def test_returns_name_level_summary_over_all_levels(self):
        rows = extract_pattern_summaries(self.TEXT, graduating_headings=self.GRAD)
        names = {r[0] for r in rows}
        # 1x developing patterns ARE included (min_level=1) — the distinguishing
        # behavior vs extract_proven_patterns (2x+).
        assert names == {"verify_or_surface", "acid_over_speed", "bare_dev"}
        # Decisions section is NOT scanned (only graduating sections).
        assert "some_decision" not in names

    def test_sorted_by_level_desc_then_name(self):
        rows = extract_pattern_summaries(self.TEXT, graduating_headings=self.GRAD)
        assert [r[0] for r in rows] == ["acid_over_speed", "verify_or_surface", "bare_dev"]
        assert [r[1] for r in rows] == [3, 2, 1]

    def test_min_level_filter(self):
        rows = extract_pattern_summaries(
            self.TEXT, graduating_headings=self.GRAD, min_level=2
        )
        assert {r[0] for r in rows} == {"acid_over_speed", "verify_or_surface"}

    def test_summaries_carry_meaning(self):
        rows = dict((r[0], r[2]) for r in extract_pattern_summaries(
            self.TEXT, graduating_headings=self.GRAD))
        assert "SNAPSHOT" in rows["verify_or_surface"]
        assert rows["acid_over_speed"] == "ACID over speed"
        assert rows["bare_dev"] == ""

    def test_highest_level_wins_on_duplicate_name(self):
        text = (
            "## Patterns\n"
            "- dup | 1x (2026-06-01) — low form\n"
            "- dup | 3x (2026-06-05) [evidence: aabbccdd \"high form\"]\n"
        )
        rows = extract_pattern_summaries(text)
        assert len(rows) == 1
        name, level, summary = rows[0]
        assert (name, level) == ("dup", 3)
        assert summary == "high form"

    def test_empty_text_yields_nothing(self):
        assert extract_pattern_summaries("") == []

    def test_rows_are_named_tuples_and_tuple_compatible(self):
        from anneal_memory.graduation import PatternSummary
        rows = extract_pattern_summaries(self.TEXT, graduating_headings=self.GRAD)
        r = rows[0]
        assert isinstance(r, PatternSummary)
        # self-documenting field access AND positional/unpack compatibility
        assert (r.name, r.level, r.summary) == (r[0], r[1], r[2])
        name, level, summary = r
        assert name == r.name and level == r.level and summary == r.summary
