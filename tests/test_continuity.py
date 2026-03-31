"""Tests for continuity validation and wrap package preparation."""

import pytest

from anneal_memory.continuity import (
    build_engine_prompt,
    format_episodes_for_wrap,
    measure_sections,
    prepare_wrap_package,
    validate_structure,
)
from anneal_memory.types import Episode, EpisodeType


# -- Test data --

VALID_CONTINUITY = """# TestAgent — Memory (v1)

## State
Working on database architecture.

## Patterns
{database:
  thought: ACID compliance matters | 1x (2026-03-31)
}

## Decisions
[decided(rationale: "reliability", on: "2026-03-31")] Use PostgreSQL

## Context
Built the database layer. Chose PostgreSQL.
"""

SAMPLE_EPISODES = [
    Episode(
        id="abc12345",
        timestamp="2026-03-31T10:00:00Z",
        type=EpisodeType.OBSERVATION,
        content="PostgreSQL chosen for ACID compliance",
        source="agent",
    ),
    Episode(
        id="def67890",
        timestamp="2026-03-31T10:05:00Z",
        type=EpisodeType.DECISION,
        content="Use connection pooling to improve latency",
        source="agent",
    ),
    Episode(
        id="aaa11111",
        timestamp="2026-03-31T10:10:00Z",
        type=EpisodeType.TENSION,
        content="Single-writer vs multi-writer for horizontal scaling",
        source="daemon",
    ),
]


# -- validate_structure --


class TestValidateStructure:
    def test_valid_structure(self):
        assert validate_structure(VALID_CONTINUITY) is True

    def test_missing_state(self):
        text = VALID_CONTINUITY.replace("## State", "## Status")
        assert validate_structure(text) is False

    def test_missing_patterns(self):
        text = VALID_CONTINUITY.replace("## Patterns", "## Observations")
        assert validate_structure(text) is False

    def test_missing_decisions(self):
        text = VALID_CONTINUITY.replace("## Decisions", "## Choices")
        assert validate_structure(text) is False

    def test_missing_context(self):
        text = VALID_CONTINUITY.replace("## Context", "## History")
        assert validate_structure(text) is False

    def test_case_insensitive(self):
        text = VALID_CONTINUITY.replace("## State", "## STATE")
        assert validate_structure(text) is True

    def test_extra_sections_ok(self):
        text = VALID_CONTINUITY + "\n## Notes\nSome extra notes.\n"
        assert validate_structure(text) is True

    def test_empty_string(self):
        assert validate_structure("") is False

    def test_no_headers(self):
        assert validate_structure("Just plain text.") is False

    def test_sections_with_extra_text(self):
        text = """## State of Mind
Working.

## Patterns Found
Some patterns.

## Decisions Made
Some decisions.

## Context and History
Some context.
"""
        assert validate_structure(text) is True

    def test_substring_false_positive_rejected(self):
        """A header like '## Interstate' should NOT match 'state'."""
        text = """## Interstate Commerce
Working.

## Patterns
Some patterns.

## Decisions
Some decisions.

## Context
Some context.
"""
        # Missing a real "## State" section — should fail
        assert validate_structure(text) is False


# -- measure_sections --


class TestMeasureSections:
    def test_measures_all_sections(self):
        sizes = measure_sections(VALID_CONTINUITY)
        assert "State" in sizes
        assert "Patterns" in sizes
        assert "Decisions" in sizes
        assert "Context" in sizes

    def test_header_included(self):
        sizes = measure_sections(VALID_CONTINUITY)
        assert "_header" in sizes

    def test_sizes_are_positive(self):
        sizes = measure_sections(VALID_CONTINUITY)
        for name, size in sizes.items():
            assert size > 0, f"Section {name} has size {size}"

    def test_total_approximately_matches(self):
        sizes = measure_sections(VALID_CONTINUITY)
        total = sum(sizes.values())
        # Should be close to the total length (accounting for newline differences)
        assert abs(total - len(VALID_CONTINUITY)) < 10


# -- format_episodes_for_wrap --


class TestFormatEpisodes:
    def test_groups_by_type(self):
        formatted = format_episodes_for_wrap(SAMPLE_EPISODES)
        assert "### Observations" in formatted or "### Observation" in formatted
        assert "### Decisions" in formatted or "### Decision" in formatted
        assert "### Tensions" in formatted or "### Tension" in formatted

    def test_includes_ids(self):
        formatted = format_episodes_for_wrap(SAMPLE_EPISODES)
        assert "(abc12345)" in formatted
        assert "(def67890)" in formatted
        assert "(aaa11111)" in formatted

    def test_includes_content(self):
        formatted = format_episodes_for_wrap(SAMPLE_EPISODES)
        assert "PostgreSQL chosen for ACID compliance" in formatted

    def test_includes_non_agent_source(self):
        """Non-agent sources get a source tag."""
        formatted = format_episodes_for_wrap(SAMPLE_EPISODES)
        assert "[daemon]" in formatted

    def test_agent_source_hidden(self):
        """Default 'agent' source is not shown (noise reduction)."""
        formatted = format_episodes_for_wrap(SAMPLE_EPISODES)
        assert "[agent]" not in formatted

    def test_empty_episodes(self):
        formatted = format_episodes_for_wrap([])
        assert "No episodes" in formatted

    def test_single_episode(self):
        formatted = format_episodes_for_wrap([SAMPLE_EPISODES[0]])
        assert "(abc12345)" in formatted


# -- prepare_wrap_package --


class TestPrepareWrapPackage:
    def test_returns_all_keys(self):
        pkg = prepare_wrap_package(
            SAMPLE_EPISODES, VALID_CONTINUITY, "TestAgent", today="2026-03-31"
        )
        assert "episodes" in pkg
        assert "episode_count" in pkg
        assert "continuity" in pkg
        assert "stale_patterns" in pkg
        assert "instructions" in pkg
        assert "today" in pkg
        assert "max_chars" in pkg

    def test_episode_count(self):
        pkg = prepare_wrap_package(
            SAMPLE_EPISODES, VALID_CONTINUITY, "TestAgent", today="2026-03-31"
        )
        assert pkg["episode_count"] == 3

    def test_passes_existing_continuity(self):
        pkg = prepare_wrap_package(
            SAMPLE_EPISODES, VALID_CONTINUITY, "TestAgent", today="2026-03-31"
        )
        assert pkg["continuity"] == VALID_CONTINUITY

    def test_null_continuity_for_first_session(self):
        pkg = prepare_wrap_package(
            SAMPLE_EPISODES, None, "TestAgent", today="2026-03-31"
        )
        assert pkg["continuity"] is None

    def test_instructions_include_markers(self):
        pkg = prepare_wrap_package(
            SAMPLE_EPISODES, None, "TestAgent", today="2026-03-31"
        )
        instructions = pkg["instructions"]
        assert "thought:" in instructions
        assert "1x" in instructions
        assert "2x" in instructions
        assert "[evidence:" in instructions

    def test_instructions_include_project_name(self):
        pkg = prepare_wrap_package(
            SAMPLE_EPISODES, None, "MyProject", today="2026-03-31"
        )
        assert "MyProject" in pkg["instructions"]

    def test_instructions_include_today(self):
        pkg = prepare_wrap_package(
            SAMPLE_EPISODES, None, "Test", today="2026-03-31"
        )
        assert "2026-03-31" in pkg["instructions"]

    def test_detects_stale_patterns(self):
        stale_continuity = """# Test — Memory (v1)

## State
Active.

## Patterns
  thought: very old pattern | 1x (2026-03-01)
  thought: fresh pattern | 1x (2026-03-31)

## Decisions
None.

## Context
Working.
"""
        pkg = prepare_wrap_package(
            SAMPLE_EPISODES, stale_continuity, "Test", today="2026-03-31"
        )
        assert len(pkg["stale_patterns"]) == 1
        assert pkg["stale_patterns"][0]["days_stale"] == 30

    def test_no_stale_patterns_when_no_continuity(self):
        pkg = prepare_wrap_package(
            SAMPLE_EPISODES, None, "Test", today="2026-03-31"
        )
        assert pkg["stale_patterns"] == []

    def test_max_chars_passed_through(self):
        pkg = prepare_wrap_package(
            SAMPLE_EPISODES, None, "Test", max_chars=10000, today="2026-03-31"
        )
        assert pkg["max_chars"] == 10000


# -- build_engine_prompt --


class TestBuildEnginePrompt:
    def test_includes_session_data(self):
        prompt = build_engine_prompt(
            "Some session data", None, "Test", 20000, "2026-03-31"
        )
        assert "Some session data" in prompt

    def test_includes_existing_continuity(self):
        prompt = build_engine_prompt(
            "Data", VALID_CONTINUITY, "Test", 20000, "2026-03-31"
        )
        assert "Working on database architecture" in prompt

    def test_first_session_message(self):
        prompt = build_engine_prompt(
            "Data", None, "Test", 20000, "2026-03-31"
        )
        assert "first session" in prompt

    def test_includes_marker_reference(self):
        prompt = build_engine_prompt(
            "Data", None, "Test", 20000, "2026-03-31"
        )
        assert "thought:" in prompt
        assert "evidence:" in prompt
        assert "1x" in prompt

    def test_includes_project_name(self):
        prompt = build_engine_prompt(
            "Data", None, "MyAgent", 20000, "2026-03-31"
        )
        assert "MyAgent" in prompt

    def test_includes_max_chars(self):
        prompt = build_engine_prompt(
            "Data", None, "Test", 15000, "2026-03-31"
        )
        assert "15000" in prompt

    def test_quality_rules_present(self):
        prompt = build_engine_prompt(
            "Data", None, "Test", 20000, "2026-03-31"
        )
        assert "PRINCIPLES over facts" in prompt
        assert "DENSITY over length" in prompt
