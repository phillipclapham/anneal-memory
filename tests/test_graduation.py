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
                    "last_seen_at": "2026-05-20T00:00:00Z",
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
