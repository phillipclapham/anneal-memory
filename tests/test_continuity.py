"""Tests for continuity validation and wrap package preparation."""

import pytest

from anneal_memory.continuity import (
    format_episodes_for_wrap,
    format_wrap_package_text,
    measure_sections,
    prepare_wrap,
    prepare_wrap_package,
    validate_structure,
)
from anneal_memory.store import Store
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


@pytest.mark.filterwarnings(
    "ignore:prepare_wrap_package is deprecated:DeprecationWarning"
)
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


# -- TypedDict return-shape surface (10.5c.3) --


class TestTypedDictReturnShapes:
    """Lock in that the TypedDict return shapes are exported and usable.

    TypedDict is structurally compatible with plain dict at runtime —
    these tests don't verify mypy-level behavior (that's what mypy is
    for). They verify the types are importable from the public
    package surface and that the canonical pipeline returns dicts
    with every declared key.
    """

    def test_types_exported_from_package(self):
        from anneal_memory import (
            PrepareWrapResult,
            SaveContinuityResult,
            StalePatternDict,
            WrapPackageDict,
        )
        # TypedDicts are callable — smoke-test the constructor.
        sp = StalePatternDict(
            line=1, content="x", level=1, last_date="2026-04-10", days_stale=5
        )
        assert sp["line"] == 1

    def test_prepare_wrap_result_has_declared_keys(self, tmp_path):
        """The runtime dict returned by prepare_wrap has every key
        declared in PrepareWrapResult — no drift between type and
        implementation."""
        store = Store(str(tmp_path / "types_shape.db"), project_name="Shape")
        try:
            store.record("An observation", EpisodeType.OBSERVATION)
            result = prepare_wrap(store)
            assert set(result.keys()) == {
                "status",
                "message",
                "episode_count",
                "package",
                "assoc_context",
            }
            assert set(result["package"].keys()) == {
                "episodes",
                "episode_count",
                "continuity",
                "stale_patterns",
                "instructions",
                "today",
                "max_chars",
            }
        finally:
            store.close()

    def test_save_continuity_result_has_declared_keys(self, tmp_path):
        """Same drift-check for validated_save_continuity."""
        from anneal_memory import validated_save_continuity

        store = Store(str(tmp_path / "save_shape.db"), project_name="SaveShape")
        try:
            store.record("An observation", EpisodeType.OBSERVATION)
            text = (
                "# SaveShape — Memory (v1)\n\n"
                "## State\nActive.\n\n"
                "## Patterns\nNone yet.\n\n"
                "## Decisions\nNone.\n\n"
                "## Context\nFirst session.\n"
            )
            prepare_wrap(store)  # mark wrap in progress
            result = validated_save_continuity(store, text)
            expected = {
                "path", "chars", "episodes_compressed",
                "graduations_validated", "graduations_demoted",
                "demoted", "bare_demoted", "citation_reuse_max",
                "gaming_suspects", "associations_formed",
                "associations_strengthened", "associations_decayed",
                "sections", "skipped_prepare", "wrap_result",
            }
            assert set(result.keys()) == expected
        finally:
            store.close()

    def test_save_continuity_result_is_json_serializable(self, tmp_path):
        """Cross-transport parity invariant: the entire
        SaveContinuityResult must be json.dumps-able top-to-bottom
        without the caller touching any dataclass. The pre-fix-pass
        shape embedded ``wrap_result: WrapResult`` which crashed
        json.dumps; the fix converts it to a plain dict via asdict()
        at return time. This test locks the invariant so any future
        refactor that re-introduces a dataclass in the return path
        fails loudly."""
        import json
        from anneal_memory import validated_save_continuity

        store = Store(
            str(tmp_path / "json_parity.db"), project_name="JsonParity"
        )
        try:
            store.record("An observation", EpisodeType.OBSERVATION)
            text = (
                "# JsonParity — Memory (v1)\n\n"
                "## State\nActive.\n\n"
                "## Patterns\nNone yet.\n\n"
                "## Decisions\nNone.\n\n"
                "## Context\nFirst session.\n"
            )
            prepare_wrap(store)
            result = validated_save_continuity(store, text)

            # Must not raise. If this crashes with
            # "Object of type WrapResult is not JSON serializable"
            # the wrap_result → dict conversion regressed.
            rendered = json.dumps(result)
            parsed = json.loads(rendered)

            # Round-trip: parsed dict has the same top-level keys.
            assert set(parsed.keys()) == set(result.keys())
            # wrap_result is a dict, not a dataclass.
            assert isinstance(parsed["wrap_result"], dict)
            assert isinstance(result["wrap_result"], dict)
            # And the wrap_result dict carries the expected keys
            # (from dataclasses.asdict(WrapResult)).
            assert "chars" in result["wrap_result"]
            assert "section_sizes" in result["wrap_result"]
            assert "episodes_compressed" in result["wrap_result"]
        finally:
            store.close()


# -- prepare_wrap_package deprecation (10.5c.3) --


class TestPrepareWrapPackageDeprecation:
    """Lock in the v0.2.0 deprecation warning on ``prepare_wrap_package``.

    The pure implementation lives in the private ``_build_wrap_package``
    helper as of 10.5c.3. The public ``prepare_wrap_package`` remains in
    place for one release cycle as a thin deprecated wrapper that emits
    a :class:`DeprecationWarning` and delegates. v0.3.0 will delete the
    public wrapper.

    Internal callers (``prepare_wrap``) now call ``_build_wrap_package``
    directly — there is no bypass-kwarg on the public wrapper because
    there is no need for one. The canonical pipeline emits zero
    deprecation warnings about anneal-memory's own internal delegation.
    """

    def test_public_call_emits_deprecation_warning(self):
        with pytest.warns(DeprecationWarning, match="prepare_wrap_package is deprecated"):
            prepare_wrap_package(
                SAMPLE_EPISODES,
                VALID_CONTINUITY,
                "TestAgent",
                today="2026-03-31",
            )

    def test_warning_mentions_canonical_entry_point(self):
        """The warning must tell users WHERE to migrate, not just
        that something is deprecated."""
        with pytest.warns(DeprecationWarning) as captured:
            prepare_wrap_package(
                SAMPLE_EPISODES,
                None,
                "TestAgent",
                today="2026-03-31",
            )
        assert any(
            "prepare_wrap(store" in str(w.message) for w in captured
        ), "warning must reference canonical prepare_wrap(store, ...) entry point"

    def test_warning_mentions_since_and_removed_versions(self):
        """CPython idiom: 'deprecated since X, removed in Y' lets
        users grep for 'removed in 0.3.0' across their deprecation
        tech debt."""
        with pytest.warns(DeprecationWarning) as captured:
            prepare_wrap_package(
                SAMPLE_EPISODES,
                None,
                "TestAgent",
                today="2026-03-31",
            )
        messages = [str(w.message) for w in captured]
        assert any("since 0.2.0" in m for m in messages), (
            "warning must say 'since 0.2.0' so users know when "
            "deprecation started"
        )
        assert any("0.3.0" in m for m in messages), (
            "warning must name the removal version explicitly"
        )

    def test_warning_stacklevel_points_at_exact_caller_line(self):
        """stacklevel=2 must surface the warning at the user's source
        line, not inside anneal_memory/continuity.py. The filename-
        endswith check from 10.5c.3's first pass was a false-positive:
        it passed for ANY stacklevel that landed anywhere in the test
        file. Checking the exact lineno catches silent stacklevel
        regressions in future refactors.

        **FRAGILITY NOTE:** the ``call_lineno + 1`` computation below
        requires the ``prepare_wrap_package(...)`` call statement to
        start on the very next source line after the ``call_lineno =``
        assignment. Do NOT insert blank lines, comments, or
        reformatting between them — the test will fail with a
        misleading "stacklevel is wrong" error. If you need to add
        setup, put it BEFORE the ``call_lineno =`` line.
        """
        import warnings as _warnings
        import sys as _sys

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            # Capture the lineno of the CALL SITE below. The next
            # source line is the prepare_wrap_package call; that's
            # the line the warning should report.
            call_lineno = _sys._getframe().f_lineno + 1
            prepare_wrap_package(
                SAMPLE_EPISODES,
                None,
                "TestAgent",
                today="2026-03-31",
            )
        deprecations = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecations) == 1
        w = deprecations[0]
        assert w.filename.endswith("test_continuity.py"), (
            f"warning fired in {w.filename}, not test file — "
            f"stacklevel is too deep"
        )
        assert w.lineno == call_lineno, (
            f"warning reported line {w.lineno} but call site is "
            f"line {call_lineno} — stacklevel is wrong. A different "
            f"value means the warning points at the wrong frame and "
            f"users can't find where to migrate."
        )

    def test_prepare_wrap_canonical_pipeline_emits_no_warning(self, tmp_path):
        """The canonical ``prepare_wrap(store, ...)`` path calls
        ``_build_wrap_package`` directly, NOT ``prepare_wrap_package``.
        Users of the canonical entry point must not see a
        DeprecationWarning about anneal-memory's own internal
        delegation — this is the regression gate that protects the
        canonical path from any future refactor that accidentally
        routes through the public wrapper."""
        import warnings as _warnings

        store = Store(str(tmp_path / "canonical.db"), project_name="Canon")
        try:
            store.record(
                "A test observation for wrap",
                EpisodeType.OBSERVATION,
            )

            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                result = prepare_wrap(store)

            deprecations = [
                w for w in caught
                if issubclass(w.category, DeprecationWarning)
                and "prepare_wrap_package" in str(w.message)
            ]
            assert deprecations == [], (
                "canonical pipeline leaked a prepare_wrap_package "
                f"DeprecationWarning: {[str(w.message) for w in deprecations]}"
            )
            assert result["status"] == "ready"
        finally:
            store.close()

    def test_build_wrap_package_private_helper_emits_no_warning(self):
        """Direct users of the private ``_build_wrap_package`` helper
        (advanced library consumers managing their own lifecycle) do
        not receive a DeprecationWarning — the warning lives on the
        public wrapper, not the implementation."""
        import warnings as _warnings
        from anneal_memory.continuity import _build_wrap_package

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            pkg = _build_wrap_package(
                SAMPLE_EPISODES,
                None,
                "TestAgent",
                today="2026-03-31",
            )
        deprecations = [
            w for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert deprecations == []
        assert pkg["episode_count"] == len(SAMPLE_EPISODES)

    def test_deprecation_warning_raises_under_strict_filter(self):
        """PEP 565 recommended library-test-suite config is
        ``-W error::DeprecationWarning``. Under that filter, calling
        ``prepare_wrap_package`` must raise a ``DeprecationWarning``
        (which subclasses ``Warning``, treated as an exception by
        the strict filter) — not silently swallow it. Protects
        against a future refactor that accidentally catches the
        warning inside the wrapper."""
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error", DeprecationWarning)
            with pytest.raises(DeprecationWarning, match="prepare_wrap_package"):
                prepare_wrap_package(
                    SAMPLE_EPISODES,
                    None,
                    "TestAgent",
                    today="2026-03-31",
                )


# -- prepare_wrap (store-aware library pipeline) --


@pytest.fixture
def wrap_store(tmp_path):
    """Store fixture for prepare_wrap pipeline tests."""
    s = Store(str(tmp_path / "wrap_test.db"), project_name="WrapTest")
    yield s
    s.close()


class TestPrepareWrapLibrary:
    """Tests for the store-aware library prepare_wrap function.

    This is the canonical pipeline entry point that MCP and CLI
    transports delegate to. It must handle both the empty case
    (no episodes → wrap_cancelled) and the ready case (episodes
    present → wrap_started, package built, association context
    attached).
    """

    def test_empty_store_returns_empty_status(self, wrap_store):
        """No episodes → status 'empty', wrap_cancelled called."""
        result = prepare_wrap(wrap_store)
        assert result["status"] == "empty"
        assert result["episode_count"] == 0
        assert result["package"] is None
        assert result["assoc_context"] is None
        # wrap_started should not be set
        assert not wrap_store.status().wrap_in_progress

    def test_empty_clears_stale_wrap_in_progress(self, wrap_store):
        """Empty path must call wrap_cancelled to clear any stale flag.

        A prior abandoned wrap could have left wrap_in_progress=True.
        prepare_wrap on an empty store must clean it up.
        """
        wrap_store.wrap_started()  # Simulate a stale flag
        assert wrap_store.status().wrap_in_progress

        result = prepare_wrap(wrap_store)
        assert result["status"] == "empty"
        assert not wrap_store.status().wrap_in_progress

    def test_ready_status_with_episodes(self, wrap_store):
        """Episodes present → status 'ready' with populated package."""
        wrap_store.record("First observation", EpisodeType.OBSERVATION)
        wrap_store.record("A decision", EpisodeType.DECISION)

        result = prepare_wrap(wrap_store)
        assert result["status"] == "ready"
        assert result["episode_count"] == 2
        assert result["package"] is not None
        # Package should contain the full prepare_wrap_package output
        pkg = result["package"]
        assert "episodes" in pkg
        assert "continuity" in pkg
        assert "instructions" in pkg
        assert "stale_patterns" in pkg
        assert "today" in pkg
        assert "max_chars" in pkg
        assert pkg["episode_count"] == 2

    def test_ready_sets_wrap_in_progress(self, wrap_store):
        """Ready path must call wrap_started on the store."""
        wrap_store.record("Observation", EpisodeType.OBSERVATION)
        assert not wrap_store.status().wrap_in_progress
        prepare_wrap(wrap_store)
        assert wrap_store.status().wrap_in_progress

    def test_first_session_no_existing_continuity(self, wrap_store):
        """First wrap should surface None continuity in the package."""
        wrap_store.record("Observation", EpisodeType.OBSERVATION)
        result = prepare_wrap(wrap_store)
        assert result["package"]["continuity"] is None

    def test_max_chars_and_staleness_days_passed_through(self, wrap_store):
        """Caller-provided config must reach prepare_wrap_package."""
        wrap_store.record("Observation", EpisodeType.OBSERVATION)
        result = prepare_wrap(
            wrap_store, max_chars=5000, staleness_days=14
        )
        assert result["package"]["max_chars"] == 5000
        # staleness_days is not directly surfaced in the package but
        # verifying no crash on custom value is sufficient here

    def test_assoc_context_is_none_when_no_associations(self, wrap_store):
        """No Hebbian links yet → assoc_context is None, not empty string."""
        wrap_store.record("Observation", EpisodeType.OBSERVATION)
        result = prepare_wrap(wrap_store)
        assert result["assoc_context"] is None


# -- format_wrap_package_text (canonical display text) --


class TestFormatWrapPackageText:
    """Tests for the canonical wrap-package text formatter.

    This is the display text MCP and CLI transports hand to the agent.
    Extracting it into a library helper means both transports render
    identical output — no drift risk between MCP prepare_wrap and
    CLI prepare-wrap.
    """

    def test_empty_returns_message_unchanged(self):
        result = {
            "status": "empty",
            "message": "No episodes since last wrap. Nothing to compress.",
            "episode_count": 0,
            "package": None,
            "assoc_context": None,
        }
        text = format_wrap_package_text(result)
        assert text == "No episodes since last wrap. Nothing to compress."

    def test_ready_first_session_includes_first_wrap_notice(self, tmp_path):
        """Ready path with no existing continuity should note 'first wrap'."""
        store = Store(str(tmp_path / "fmt_test.db"), project_name="Test")
        try:
            store.record("Observation A", EpisodeType.OBSERVATION)
            result = prepare_wrap(store)
            text = format_wrap_package_text(result)
            assert "## Episodes This Session" in text
            assert "(No existing continuity file — this is the first wrap.)" in text
            # Compression instructions should lead the output
            assert text.startswith("Compress your session episodes")
        finally:
            store.close()

    def test_ready_with_existing_continuity(self, tmp_path):
        """Existing continuity should appear under its header."""
        store = Store(str(tmp_path / "fmt_test2.db"), project_name="Test")
        try:
            store.save_continuity(
                "# Test — Memory (v1)\n\n"
                "## State\nPrior state.\n\n"
                "## Patterns\nnone yet\n\n"
                "## Decisions\nnone\n\n"
                "## Context\nPrior context.\n"
            )
            store.record("New observation", EpisodeType.OBSERVATION)
            result = prepare_wrap(store)
            text = format_wrap_package_text(result)
            assert "## Current Continuity File" in text
            assert "Prior state." in text
        finally:
            store.close()

    def test_stale_patterns_section_appears_when_present(self):
        """Stale patterns in the package should be rendered as a section."""
        # Hand-built result to isolate formatting from store state
        result = {
            "status": "ready",
            "message": "Ready.",
            "episode_count": 1,
            "package": {
                "instructions": "Compress your episodes.",
                "episodes": "(test)",
                "episode_count": 1,
                "continuity": None,
                "stale_patterns": [
                    {
                        "line": 42,
                        "content": "thought: old pattern | 1x (2026-01-01)",
                        "level": "1x",
                        "last_date": "2026-01-01",
                        "days_stale": 90,
                    }
                ],
                "today": "2026-04-10",
                "max_chars": 20000,
            },
            "assoc_context": None,
        }
        text = format_wrap_package_text(result)
        assert "## Stale Patterns (consider removing)" in text
        assert "Line 42:" in text
        assert "(90d stale)" in text

    def test_assoc_context_appended_when_present(self):
        result = {
            "status": "ready",
            "message": "Ready.",
            "episode_count": 1,
            "package": {
                "instructions": "Compress.",
                "episodes": "(test)",
                "episode_count": 1,
                "continuity": None,
                "stale_patterns": [],
                "today": "2026-04-10",
                "max_chars": 20000,
            },
            "assoc_context": "## Hebbian Association Context\n- some link",
        }
        text = format_wrap_package_text(result)
        assert "## Hebbian Association Context" in text
        assert "some link" in text

    def test_no_stale_patterns_section_when_absent(self):
        """Stale patterns header should only appear when there are stale patterns."""
        result = {
            "status": "ready",
            "message": "Ready.",
            "episode_count": 1,
            "package": {
                "instructions": "Compress.",
                "episodes": "(test)",
                "episode_count": 1,
                "continuity": None,
                "stale_patterns": [],
                "today": "2026-04-10",
                "max_chars": 20000,
            },
            "assoc_context": None,
        }
        text = format_wrap_package_text(result)
        assert "Stale Patterns" not in text

    def test_unknown_status_falls_back_to_message(self):
        """Any status other than 'ready' renders as the bare message.

        Defensive: a future status value (or a malformed hand-built
        result dict) should not crash the formatter by trying to read
        result['package']['instructions'] when package is None.
        """
        result = {
            "status": "future_state_we_dont_know_about",
            "message": "Some status message.",
            "episode_count": 0,
            "package": None,
            "assoc_context": None,
        }
        text = format_wrap_package_text(result)
        assert text == "Some status message."


# -- Cross-transport parity (the Session 10.5c.1 regression guard) --


class TestCrossTransportParity:
    """Assert that library / MCP / CLI paths produce identical wrap metrics.

    This is the structural regression guard against re-introducing the
    three-way divergence that Diogenes caught in Session 10.5c (library
    under-reporting ``bare_demoted`` in the ``graduations_demoted``
    wrap metric vs. MCP and CLI). After Session 10.5c.1 made the
    library canonical and reduced MCP and CLI to thin adapters, this
    test proves all three surfaces produce identical wrap records on
    identical input. If any future change reintroduces divergence, this
    test fails loudly — it's the one test that would catch the exact
    bug class the session was built to eliminate.

    Test methodology: seed three independent stores with the same
    deterministic episodes, run a full wrap through each transport,
    then compare ``get_wrap_history()`` across all three. Non-
    determinstic fields (``id``, ``wrapped_at``) are excluded.
    """

    @staticmethod
    def _deterministic_continuity_text(today: str, evidence_id: str) -> str:
        """Continuity text citing a known episode for graduation to fire."""
        return (
            "# ParityTest — Memory (v1)\n\n"
            "## State\nTesting cross-transport parity.\n\n"
            "## Patterns\n"
            f"thought: parity claim about the testing framework"
            f" | 2x ({today})"
            f" [evidence: {evidence_id} \"testing framework parity assertion\"]\n\n"
            "## Decisions\n"
            f"[decided(rationale: \"identical\", on: \"{today}\")] All three paths match\n\n"
            "## Context\nSeeded episodes for parity test.\n"
        )

    @staticmethod
    def _seed_store(db_path: str) -> tuple[Store, str, str]:
        """Seed a store with two deterministic episodes.

        Returns ``(store, text, episode_prefix)`` — text is the
        canonical continuity document citing the first episode by its
        8-char prefix. Because episode IDs are content-hashed,
        identical (content, type) inputs produce identical prefixes
        across all three stores, which is what makes this parity test
        possible.
        """
        from datetime import date as _date

        store = Store(db_path, project_name="ParityTest")
        ep1 = store.record(
            "testing framework parity assertion",
            EpisodeType.OBSERVATION,
        )
        store.record("supporting observation for parity", EpisodeType.DECISION)
        text = TestCrossTransportParity._deterministic_continuity_text(
            _date.today().isoformat(), ep1.id[:8]
        )
        return store, text, ep1.id[:8]

    @staticmethod
    def _wrap_record_domain_fields(record) -> dict:
        """Extract only the deterministic domain metrics for comparison.

        Excludes ``id`` (autoincrement, differs across stores) and
        ``wrapped_at`` (ISO timestamp, differs by microseconds).
        """
        return {
            "episodes_compressed": record.episodes_compressed,
            "continuity_chars": record.continuity_chars,
            "graduations_validated": record.graduations_validated,
            "graduations_demoted": record.graduations_demoted,
            "citation_reuse_max": record.citation_reuse_max,
            "patterns_extracted": record.patterns_extracted,
            "associations_formed": record.associations_formed,
            "associations_strengthened": record.associations_strengthened,
            "associations_decayed": record.associations_decayed,
        }

    def test_library_mcp_cli_produce_identical_wrap_metrics(self, tmp_path):
        """The three surfaces must emit byte-identical wrap metrics.

        This is the test that would have caught the original Diogenes
        Finding #1 divergence (library under-reporting
        ``graduations_demoted`` by missing ``bare_demoted``) directly
        and automatically — no scheduled review required.
        """
        from argparse import Namespace
        from anneal_memory import validated_save_continuity
        from anneal_memory.server import Server
        from anneal_memory.cli import cmd_save_continuity, cmd_prepare_wrap

        # --- Library path ---
        lib_db = str(tmp_path / "lib.db")
        lib_store, lib_text, _ = self._seed_store(lib_db)
        # Library path: call prepare_wrap + validated_save_continuity
        # directly (this is what framework users would do via the
        # canonical entry points).
        prepare_wrap(lib_store)
        validated_save_continuity(lib_store, lib_text)
        lib_history = lib_store.get_wrap_history()
        lib_store.close()

        # --- MCP path ---
        mcp_db = str(tmp_path / "mcp.db")
        mcp_store, mcp_text, _ = self._seed_store(mcp_db)
        mcp_server = Server(mcp_store)
        mcp_server._tool_prepare_wrap({})
        mcp_server._tool_save_continuity({"text": mcp_text})
        mcp_history = mcp_store.get_wrap_history()
        mcp_store.close()

        # --- CLI path ---
        cli_db = str(tmp_path / "cli.db")
        cli_store, cli_text, _ = self._seed_store(cli_db)
        cli_store.close()  # CLI commands open their own store via args.db

        cli_file = tmp_path / "cli_continuity.md"
        cli_file.write_text(cli_text)

        cli_args = Namespace(
            db=cli_db,
            project_name="ParityTest",
            json=False,
            max_chars=20000,
            staleness_days=7,
            file=str(cli_file),
            affect_tag=None,
            affect_intensity=0.5,
        )
        cmd_prepare_wrap(cli_args)
        cmd_save_continuity(cli_args)

        # Reopen to read history
        cli_store_read = Store(cli_db, project_name="ParityTest")
        try:
            cli_history = cli_store_read.get_wrap_history()
        finally:
            cli_store_read.close()

        # --- Assertions ---
        assert len(lib_history) == 1, "Library path should record exactly one wrap"
        assert len(mcp_history) == 1, "MCP path should record exactly one wrap"
        assert len(cli_history) == 1, "CLI path should record exactly one wrap"

        lib_metrics = self._wrap_record_domain_fields(lib_history[0])
        mcp_metrics = self._wrap_record_domain_fields(mcp_history[0])
        cli_metrics = self._wrap_record_domain_fields(cli_history[0])

        assert lib_metrics == mcp_metrics, (
            f"Library and MCP wrap metrics diverged.\n"
            f"Library: {lib_metrics}\n"
            f"MCP:     {mcp_metrics}"
        )
        assert lib_metrics == cli_metrics, (
            f"Library and CLI wrap metrics diverged.\n"
            f"Library: {lib_metrics}\n"
            f"CLI:     {cli_metrics}"
        )

        # Specifically assert the Diogenes Finding #1 fields — these are
        # the exact fields that diverged in 10.5c and must never diverge
        # again across the three canonical paths.
        assert (
            lib_metrics["graduations_demoted"]
            == mcp_metrics["graduations_demoted"]
            == cli_metrics["graduations_demoted"]
        )
        assert (
            lib_metrics["citation_reuse_max"]
            == mcp_metrics["citation_reuse_max"]
            == cli_metrics["citation_reuse_max"]
        )

        # Graduation must have actually fired in all three paths (not
        # silently skipped due to wall-clock drift or upstream bypass).
        assert lib_metrics["graduations_validated"] >= 1
        assert mcp_metrics["graduations_validated"] >= 1
        assert cli_metrics["graduations_validated"] >= 1


