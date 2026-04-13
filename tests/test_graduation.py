"""Tests for graduation and demotion logic."""

import pytest

from anneal_memory.graduation import (
    GraduationResult,
    check_explanation_overlap,
    detect_citation_gaming,
    detect_stale_patterns,
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
