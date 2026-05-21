"""Tests for graduation and demotion logic."""

import pytest

from anneal_memory.graduation import (
    CrossSessionCollision,
    GraduationResult,
    OmittedPattern,
    _meaningful_word_overlap,
    check_explanation_overlap,
    detect_citation_gaming,
    detect_pattern_omissions,
    detect_stale_patterns,
    extract_pattern_names,
    validate_graduations,
)

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
        result = validate_graduations(
            text=text,
            valid_ids={"abc12345"},
            today="2026-05-21",
            node_content_map=node_content,
            pattern_history_lookup=self._stub_lookup(alpha=prior),
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
        """Within-session check fails first (no matching episode);
        cross-session check should NOT also fire — operators see one
        failure reason at a time, and the within-session failure is
        more fundamental."""
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
        )
        assert result.validated == 0
        assert result.demoted == 1
        # Demotion marker should be (ungrounded), not (cross-session-overlap)
        assert "(ungrounded)" in result.text
        assert "(cross-session-overlap)" not in result.text
        assert result.cross_session_collisions == []
