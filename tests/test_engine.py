"""Tests for anneal_memory.engine — LLM orchestration for programmatic compression."""

from __future__ import annotations

import os
import re
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from anneal_memory import Engine, Store, WrapResult
from anneal_memory.engine import _truncate_to_sections


# -- Fixtures --


@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    """Return a path for a temporary SQLite database."""
    return str(tmp_path / "test.db")


@pytest.fixture
def store(tmp_db: str) -> Store:
    """Create a Store with a few episodes recorded."""
    s = Store(tmp_db, project_name="TestProject")
    s.record("First observation about the system", "observation")
    s.record("Decided to use SQLite for storage", "decision")
    s.record("What's the right graduation threshold?", "question")
    return s


def _valid_continuity(project_name: str = "TestProject") -> str:
    """Return a minimal valid continuity file with all 4 sections."""
    return (
        f"# {project_name} — Memory (v1)\n\n"
        "## State\n\nWorking on engine implementation.\n\n"
        "## Patterns\n\n"
        "{testing:\n"
        "  thought: unit tests catch regressions | 1x (2026-03-31)\n"
        "}\n\n"
        "## Decisions\n\n"
        '[decided(rationale: "SQLite is stdlib", on: "2026-03-31")] Use SQLite\n\n'
        "## Context\n\nBuilding the engine module for programmatic compression.\n"
    )


def _make_llm(output: str | None = None) -> MagicMock:
    """Create a mock LLM that returns valid continuity or custom output."""
    mock = MagicMock()
    mock.return_value = output if output is not None else _valid_continuity()
    return mock


# -- Constructor tests --


class TestEngineConstructor:
    """Engine.__init__ — llm vs api_key vs neither."""

    def test_with_llm_callable(self, store: Store) -> None:
        llm = _make_llm()
        engine = Engine(store, llm=llm)
        assert engine.store is store
        assert engine.max_chars == 20000

    def test_with_custom_max_chars(self, store: Store) -> None:
        engine = Engine(store, llm=_make_llm(), max_chars=5000)
        assert engine.max_chars == 5000

    def test_no_llm_no_key_raises(self, store: Store) -> None:
        # Ensure env var is not set
        env = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="Either llm callable or api_key"):
                Engine(store)
        finally:
            if env is not None:
                os.environ["ANTHROPIC_API_KEY"] = env

    def test_api_key_without_anthropic_raises_import(self, store: Store) -> None:
        """api_key requires the anthropic package — ImportError if missing."""
        # We can't easily unimport anthropic if installed, so we test the
        # _make_anthropic_llm function indirectly via a mock.
        # This test verifies the error message shape.
        from anneal_memory.engine import _make_anthropic_llm

        # If anthropic IS installed, this will succeed (not an error).
        # We test the path that matters: that _make_anthropic_llm uses anthropic.
        # Full import-error testing would require sys.modules manipulation
        # which is brittle — skip if anthropic is installed.
        try:
            import anthropic  # noqa: F401
            pytest.skip("anthropic is installed — can't test ImportError path")
        except ImportError:
            with pytest.raises(ImportError, match="pip install anneal-memory"):
                _make_anthropic_llm("fake-key", "model", 1000)

    def test_env_var_fallback(self, store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
        """ANTHROPIC_API_KEY env var is used when no explicit key/llm given."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        try:
            import anthropic  # noqa: F401
            # If anthropic is installed, constructor succeeds
            engine = Engine(store)
            assert engine.store is store
        except ImportError:
            # If not installed, we get ImportError (not ValueError)
            with pytest.raises(ImportError):
                Engine(store)


# -- Wrap lifecycle tests --


class TestWrapNoEpisodes:
    """wrap() with no episodes to compress."""

    def test_returns_saved_false(self, tmp_db: str) -> None:
        store = Store(tmp_db)
        engine = Engine(store, llm=_make_llm())

        # Record and wrap to clear episodes, then wrap again
        store.record("seed episode", "observation")
        result1 = engine.wrap()
        assert result1.saved is True

        # Second wrap — no new episodes
        result2 = engine.wrap()
        assert result2.saved is False
        assert result2.episodes_compressed == 0
        assert result2.chars == 0

    def test_empty_store(self, tmp_db: str) -> None:
        store = Store(tmp_db)
        engine = Engine(store, llm=_make_llm())
        result = engine.wrap()
        assert result.saved is False

    def test_clears_wrap_started(self, tmp_db: str) -> None:
        store = Store(tmp_db)
        engine = Engine(store, llm=_make_llm())
        engine.wrap()
        assert store._get_metadata("wrap_started_at") == ""


class TestWrapFullCycle:
    """wrap() with episodes — the full compression pipeline."""

    def test_basic_wrap(self, store: Store) -> None:
        llm = _make_llm()
        engine = Engine(store, llm=llm)
        result = engine.wrap()

        assert result.saved is True
        assert result.episodes_compressed == 3
        assert result.chars > 0
        assert result.continuity_text is not None
        assert "## State" in result.continuity_text
        assert "## Patterns" in result.continuity_text

    def test_llm_called_with_prompt(self, store: Store) -> None:
        llm = _make_llm()
        engine = Engine(store, llm=llm)
        engine.wrap()

        llm.assert_called_once()
        prompt = llm.call_args[0][0]
        assert "memory compression engine" in prompt
        assert "Session Data" in prompt
        assert "First observation" in prompt

    def test_continuity_file_saved(self, store: Store) -> None:
        engine = Engine(store, llm=_make_llm())
        engine.wrap()

        text = store.load_continuity()
        assert text is not None
        assert "## State" in text

    def test_wrap_recorded_in_store(self, store: Store) -> None:
        engine = Engine(store, llm=_make_llm())
        engine.wrap()

        status = store.status()
        assert status.total_wraps == 1
        assert status.last_wrap_at is not None
        assert status.wrap_in_progress is False

    def test_metadata_sessions_produced(self, store: Store) -> None:
        engine = Engine(store, llm=_make_llm())
        engine.wrap()

        meta = store.load_meta()
        assert meta["sessions_produced"] == 1

    def test_metadata_increments(self, store: Store) -> None:
        engine = Engine(store, llm=_make_llm())
        engine.wrap()

        # Add more episodes and wrap again
        store.record("New observation after first wrap", "observation")
        engine.wrap()

        meta = store.load_meta()
        assert meta["sessions_produced"] == 2

    def test_project_name_in_prompt(self, tmp_db: str) -> None:
        store = Store(tmp_db, project_name="MyAgent")
        store.record("test episode", "observation")
        llm = _make_llm(_valid_continuity("MyAgent"))
        engine = Engine(store, llm=llm)
        engine.wrap()

        prompt = llm.call_args[0][0]
        assert "MyAgent" in prompt

    def test_existing_continuity_in_prompt(self, store: Store) -> None:
        """Second wrap includes existing continuity in the prompt."""
        llm = _make_llm()
        engine = Engine(store, llm=llm)
        engine.wrap()

        store.record("Another observation", "observation")
        engine.wrap()

        # Second call should include existing continuity
        second_prompt = llm.call_args[0][0]
        assert "Existing Continuity File" in second_prompt

    def test_wrap_started_cleared_on_success(self, store: Store) -> None:
        engine = Engine(store, llm=_make_llm())
        engine.wrap()
        assert store._get_metadata("wrap_started_at") == ""


# -- Validation tests --


class TestStructureValidation:
    """Structure validation — fallback behavior."""

    def test_invalid_structure_with_existing(self, store: Store) -> None:
        """Invalid LLM output + existing continuity = fallback, saved=False."""
        # First, create valid continuity
        existing = _valid_continuity()
        store.save_continuity(existing)

        llm = _make_llm("This is garbage output with no sections")
        engine = Engine(store, llm=llm)
        result = engine.wrap()

        assert result.saved is False
        assert result.continuity_text == existing
        assert result.episodes_compressed == 0  # Nothing was actually compressed

    def test_invalid_structure_clears_wrap_started(self, store: Store) -> None:
        """Fallback path clears wrap_started_at so store doesn't think wrap is running."""
        store.save_continuity(_valid_continuity())
        llm = _make_llm("garbage with no sections")
        engine = Engine(store, llm=llm)
        engine.wrap()
        assert store._get_metadata("wrap_started_at") == ""
        assert store.status().wrap_in_progress is False

    def test_invalid_structure_first_session(self, tmp_db: str) -> None:
        """Invalid LLM output + no existing continuity = use as-is (first session)."""
        store = Store(tmp_db)
        store.record("first episode ever", "observation")

        # LLM returns partial output — missing some sections but has others
        partial = (
            "# Agent — Memory (v1)\n\n"
            "## State\nDoing stuff\n\n"
            "## Patterns\nSome patterns\n\n"
            "## Decisions\nSome decisions\n\n"
            "## Context\nSome context\n"
        )
        llm = _make_llm(partial)
        engine = Engine(store, llm=llm)
        result = engine.wrap()

        # This actually passes validation (has all 4 sections)
        assert result.saved is True

    def test_truly_invalid_first_session_rejects(self, tmp_db: str) -> None:
        """Truly invalid output with no fallback — rejects, episodes preserved for retry."""
        store = Store(tmp_db)
        store.record("first episode", "observation")

        # Missing sections entirely
        llm = _make_llm("Just some random text\nNo sections here")
        engine = Engine(store, llm=llm)
        result = engine.wrap()

        # Engine rejects garbage — episodes stay in store for next attempt
        assert result.saved is False
        assert result.chars == 0
        assert result.continuity_text is None
        assert store.load_continuity() is None
        assert store.status().total_wraps == 0
        # Episodes still available for retry
        assert store.status().total_episodes == 1


class TestGraduationValidation:
    """Graduation citation validation in the engine pipeline."""

    def test_valid_citation_passes(self, store: Store) -> None:
        """Valid citation with matching episode ID passes validation."""
        episodes = store.episodes_since_wrap()
        ep_id = episodes[0].id[:8]
        today = date.today().isoformat()

        output = (
            "# TestProject — Memory (v1)\n\n"
            "## State\nWorking.\n\n"
            "## Patterns\n"
            f'  thought: systems work | 2x ({today}) [evidence: {ep_id} "observation about system"]\n\n'
            "## Decisions\nNone.\n\n"
            "## Context\nBuilding.\n"
        )
        llm = _make_llm(output)
        engine = Engine(store, llm=llm)
        result = engine.wrap()

        assert result.graduations_validated == 1
        assert result.graduations_demoted == 0

    def test_invalid_citation_demotes(self, store: Store) -> None:
        """Citation with non-existent episode ID gets demoted."""
        today = date.today().isoformat()

        output = (
            "# TestProject — Memory (v1)\n\n"
            "## State\nWorking.\n\n"
            "## Patterns\n"
            f'  thought: fake pattern | 2x ({today}) [evidence: deadbeef "made up stuff"]\n\n'
            "## Decisions\nNone.\n\n"
            "## Context\nBuilding.\n"
        )
        llm = _make_llm(output)
        engine = Engine(store, llm=llm)
        result = engine.wrap()

        assert result.graduations_demoted == 1
        # Check the saved text has the demotion
        saved = store.load_continuity()
        assert saved is not None
        assert "1x" in saved
        assert "(ungrounded)" in saved

    def test_citations_seen_updated(self, store: Store) -> None:
        """citations_seen metadata set to True after first valid citation."""
        episodes = store.episodes_since_wrap()
        ep_id = episodes[0].id[:8]
        today = date.today().isoformat()

        output = (
            "# TestProject — Memory (v1)\n\n"
            "## State\nWorking.\n\n"
            "## Patterns\n"
            f'  thought: observation insight | 2x ({today}) [evidence: {ep_id} "observation about system"]\n\n'
            "## Decisions\nNone.\n\n"
            "## Context\nBuilding.\n"
        )
        llm = _make_llm(output)
        engine = Engine(store, llm=llm)
        engine.wrap()

        meta = store.load_meta()
        assert meta["citations_seen"] is True

    def test_citations_seen_not_set_without_valid(self, store: Store) -> None:
        """citations_seen stays False when no valid citations."""
        llm = _make_llm()  # Default output has no citations
        engine = Engine(store, llm=llm)
        engine.wrap()

        meta = store.load_meta()
        assert meta["citations_seen"] is False

    def test_citations_seen_set_on_attempted_citations(self, store: Store) -> None:
        """citations_seen triggers when LLM attempts citations (even invalid ones).

        Aligns with MCP server behavior: if the LLM demonstrates citation
        ability by including [evidence:] tags, the sunset engages.
        """
        today = date.today().isoformat()

        # Citation with a real ID but bad explanation (will fail overlap check)
        episodes = store.episodes_since_wrap()
        ep_id = episodes[0].id[:8]
        output = (
            "# TestProject — Memory (v1)\n\n"
            "## State\nWorking.\n\n"
            "## Patterns\n"
            f'  thought: unrelated claim | 2x ({today}) [evidence: {ep_id} "completely unrelated words xyz"]\n\n'
            "## Decisions\nNone.\n\n"
            "## Context\nBuilding.\n"
        )
        llm = _make_llm(output)
        engine = Engine(store, llm=llm)
        engine.wrap()

        meta = store.load_meta()
        # Even though the citation was demoted, citation_counts is populated
        # because the ID matched a real episode — so citations_seen is True
        assert meta["citations_seen"] is True


# -- Truncation tests --


class TestTruncation:
    """Truncation when LLM output exceeds max_chars."""

    def test_within_budget_unchanged(self) -> None:
        text = _valid_continuity()
        assert _truncate_to_sections(text, 10000) == text

    def test_over_budget_drops_whole_sections(self) -> None:
        """Truncation drops entire sections rather than cutting mid-section."""
        text = _valid_continuity()
        # Set budget to fit title + first section but not all
        # Title + State section should fit, rest dropped
        truncated = _truncate_to_sections(text, 200)
        assert len(truncated) <= 200
        assert "## State" in truncated
        # Verify no unclosed blocks — every { has a matching }
        assert truncated.count("{") == truncated.count("}")

    def test_no_sections_hard_truncate(self) -> None:
        text = "No section headers here, just plain text " * 100
        truncated = _truncate_to_sections(text, 50)
        assert len(truncated) == 50

    def test_pattern_blocks_never_corrupted(self) -> None:
        """Truncation must never leave unclosed {} pattern blocks."""
        text = (
            "# Agent — Memory (v1)\n\n"
            "## State\nShort state.\n\n"
            "## Patterns\n"
            "{database:\n"
            "  thought: ACID matters | 1x (2026-03-31)\n"
            "  thought: pooling is key | 1x (2026-03-31)\n"
            "}\n"
            "{testing:\n"
            "  thought: tests catch bugs | 1x (2026-03-31)\n"
            "}\n\n"
            "## Decisions\nSome decision.\n\n"
            "## Context\nLong context here.\n"
        )
        # Budget fits State but forces Patterns to be dropped entirely
        truncated = _truncate_to_sections(text, 60)
        assert truncated.count("{") == truncated.count("}"), (
            f"Unclosed blocks in truncated output: {truncated!r}"
        )

    def test_engine_truncates_on_wrap(self, store: Store) -> None:
        """Engine truncates when LLM produces oversized output."""
        big_output = _valid_continuity() + "\n" + ("x" * 50000)
        llm = _make_llm(big_output)
        engine = Engine(store, llm=llm, max_chars=500)
        result = engine.wrap()

        assert result.saved is True
        assert result.chars <= 500


# -- Error handling tests --


class TestErrorHandling:
    """Error handling — wrap_started_at cleared on failure."""

    def test_wrap_cancelled_public_api(self, tmp_db: str) -> None:
        """Store.wrap_cancelled() clears wrap-in-progress flag."""
        store = Store(tmp_db)
        store.wrap_started()
        assert store.status().wrap_in_progress is True
        store.wrap_cancelled()
        assert store.status().wrap_in_progress is False

    def test_llm_exception_clears_wrap_started(self, store: Store) -> None:
        def failing_llm(prompt: str) -> str:
            raise RuntimeError("LLM API error")

        engine = Engine(store, llm=failing_llm)

        with pytest.raises(RuntimeError, match="LLM API error"):
            engine.wrap()

        # wrap_started_at should be cleared
        assert store._get_metadata("wrap_started_at") == ""

    def test_llm_exception_preserves_store(self, store: Store) -> None:
        """Store state unchanged after LLM failure."""
        initial_status = store.status()

        def failing_llm(prompt: str) -> str:
            raise RuntimeError("boom")

        engine = Engine(store, llm=failing_llm)

        with pytest.raises(RuntimeError):
            engine.wrap()

        after_status = store.status()
        assert after_status.total_wraps == initial_status.total_wraps
        assert after_status.total_episodes == initial_status.total_episodes


# -- WrapResult field tests --


class TestWrapResultContinuityText:
    """WrapResult.continuity_text — populated by Engine."""

    def test_present_on_success(self, store: Store) -> None:
        engine = Engine(store, llm=_make_llm())
        result = engine.wrap()
        assert result.continuity_text is not None
        assert isinstance(result.continuity_text, str)

    def test_backward_compatible(self) -> None:
        """WrapResult still works without continuity_text (Store.wrap_completed)."""
        result = WrapResult(saved=True, chars=100, section_sizes={"State": 50})
        assert result.continuity_text is None

    def test_section_sizes_populated(self, store: Store) -> None:
        """section_sizes should contain actual measurements, not empty dict."""
        engine = Engine(store, llm=_make_llm())
        result = engine.wrap()
        assert len(result.section_sizes) > 0
        assert "State" in result.section_sizes
        assert "Patterns" in result.section_sizes

    def test_matches_saved_file(self, store: Store) -> None:
        engine = Engine(store, llm=_make_llm())
        result = engine.wrap()

        saved = store.load_continuity()
        assert result.continuity_text == saved


# -- Integration test --


class TestStalePatternDetection:
    """Stale pattern info injected into Engine prompt."""

    def test_stale_patterns_in_prompt(self, store: Store) -> None:
        """When existing continuity has stale patterns, prompt includes them."""
        old_date = (date.today() - timedelta(days=10)).isoformat()
        existing = (
            "# TestProject — Memory (v1)\n\n"
            "## State\nWorking.\n\n"
            "## Patterns\n"
            f"  thought: old pattern | 1x ({old_date})\n\n"
            "## Decisions\nNone.\n\n"
            "## Context\nBuilding.\n"
        )
        store.save_continuity(existing)

        llm = _make_llm()
        engine = Engine(store, llm=llm)
        engine.wrap()

        prompt = llm.call_args[0][0]
        assert "Stale Patterns" in prompt
        assert "days stale" in prompt

    def test_stale_patterns_before_output_instructions(self, store: Store) -> None:
        """Stale patterns should appear BEFORE output format instructions."""
        old_date = (date.today() - timedelta(days=10)).isoformat()
        existing = (
            "# TestProject — Memory (v1)\n\n"
            "## State\nWorking.\n\n"
            "## Patterns\n"
            f"  thought: old pattern | 1x ({old_date})\n\n"
            "## Decisions\nNone.\n\n"
            "## Context\nBuilding.\n"
        )
        store.save_continuity(existing)

        llm = _make_llm()
        engine = Engine(store, llm=llm)
        engine.wrap()

        prompt = llm.call_args[0][0]
        stale_pos = prompt.index("Stale Patterns")
        output_pos = prompt.index("## Output Format")
        assert stale_pos < output_pos, "Stale patterns must appear before output instructions"

    def test_no_stale_patterns_no_section(self, store: Store) -> None:
        """When no stale patterns exist, no extra section in prompt."""
        today = date.today().isoformat()
        existing = (
            "# TestProject — Memory (v1)\n\n"
            "## State\nWorking.\n\n"
            "## Patterns\n"
            f"  thought: fresh pattern | 1x ({today})\n\n"
            "## Decisions\nNone.\n\n"
            "## Context\nBuilding.\n"
        )
        store.save_continuity(existing)

        llm = _make_llm()
        engine = Engine(store, llm=llm)
        engine.wrap()

        prompt = llm.call_args[0][0]
        assert "Stale Patterns" not in prompt


class TestEngineIntegration:
    """Full integration: record -> wrap -> verify -> record -> wrap."""

    def test_two_wrap_cycles(self, tmp_db: str) -> None:
        store = Store(tmp_db, project_name="IntegrationTest")

        # First session
        store.record("Observed pattern A", "observation")
        store.record("Decided to use approach X", "decision")

        llm = _make_llm(_valid_continuity("IntegrationTest"))
        engine = Engine(store, llm=llm)

        result1 = engine.wrap()
        assert result1.saved is True
        assert result1.episodes_compressed == 2

        # Verify store state
        status = store.status()
        assert status.total_wraps == 1
        assert status.episodes_since_wrap == 0

        # Second session
        store.record("New observation in session 2", "observation")

        result2 = engine.wrap()
        assert result2.saved is True
        assert result2.episodes_compressed == 1

        status = store.status()
        assert status.total_wraps == 2
        assert status.episodes_since_wrap == 0

        meta = store.load_meta()
        assert meta["sessions_produced"] == 2
