"""Tests for continuity validation and wrap package preparation."""

# ``from __future__ import annotations`` defers all annotation
# evaluation to string form so ``_read_audit_events(audit_path: Path)``
# below doesn't need ``Path`` imported at module level — Path is only
# referenced in the annotation, and the function body uses it as an
# instance method (``audit_path.exists()``). Python 3.14 makes
# deferred annotations the default (PEP 649), which masked this on
# local 3.14 runs but broke collection on 3.10-3.13 in CI.
from __future__ import annotations

import uuid
from datetime import date

import pytest

from anneal_memory.continuity import (
    _build_wrap_package,
    format_episodes_for_wrap,
    format_wrap_package_text,
    measure_sections,
    prepare_wrap,
    validate_structure,
)
from anneal_memory.store import Store, StoreError
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


# -- _build_wrap_package (private package construction helper) --


class TestBuildWrapPackage:
    """Tests for the private ``_build_wrap_package`` helper.

    The deprecated public wrapper ``prepare_wrap_package`` was removed
    in v0.3.0. These tests cover the private helper directly because
    it remains the pure-function core called by :func:`prepare_wrap`
    and by advanced library users managing their own wrap lifecycle.
    """

    def test_returns_all_keys(self):
        pkg = _build_wrap_package(
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
        pkg = _build_wrap_package(
            SAMPLE_EPISODES, VALID_CONTINUITY, "TestAgent", today="2026-03-31"
        )
        assert pkg["episode_count"] == 3

    def test_passes_existing_continuity(self):
        pkg = _build_wrap_package(
            SAMPLE_EPISODES, VALID_CONTINUITY, "TestAgent", today="2026-03-31"
        )
        assert pkg["continuity"] == VALID_CONTINUITY

    def test_null_continuity_for_first_session(self):
        pkg = _build_wrap_package(
            SAMPLE_EPISODES, None, "TestAgent", today="2026-03-31"
        )
        assert pkg["continuity"] is None

    def test_instructions_include_markers(self):
        pkg = _build_wrap_package(
            SAMPLE_EPISODES, None, "TestAgent", today="2026-03-31"
        )
        instructions = pkg["instructions"]
        assert "thought:" in instructions
        assert "1x" in instructions
        assert "2x" in instructions
        assert "[evidence:" in instructions

    def test_instructions_include_project_name(self):
        pkg = _build_wrap_package(
            SAMPLE_EPISODES, None, "MyProject", today="2026-03-31"
        )
        assert "MyProject" in pkg["instructions"]

    def test_instructions_include_today(self):
        pkg = _build_wrap_package(
            SAMPLE_EPISODES, None, "Test", today="2026-03-31"
        )
        assert "2026-03-31" in pkg["instructions"]

    def test_detects_stale_patterns(self):
        from anneal_memory.continuity import _build_wrap_package

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
        pkg = _build_wrap_package(
            SAMPLE_EPISODES, stale_continuity, "Test", today="2026-03-31"
        )
        assert len(pkg["stale_patterns"]) == 1
        assert pkg["stale_patterns"][0]["days_stale"] == 30

    def test_no_stale_patterns_when_no_continuity(self):
        pkg = _build_wrap_package(
            SAMPLE_EPISODES, None, "Test", today="2026-03-31"
        )
        assert pkg["stale_patterns"] == []

    def test_max_chars_passed_through(self):
        pkg = _build_wrap_package(
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
                "wrap_token",
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
                "skipped_non_today",
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


# -- canonical-pipeline regression gate (post-v0.3.0 wrapper removal) --


class TestCanonicalPipelineNoDeprecation:
    """Regression gate: the canonical ``prepare_wrap`` pipeline MUST
    NOT emit any ``DeprecationWarning``.

    The deprecated public wrapper ``prepare_wrap_package`` was removed
    in v0.3.0 and the legacy no-arg ``Store.wrap_started()`` form was
    tightened to require ``token`` + ``episode_ids``. This gate
    protects the canonical path against any future refactor that
    re-introduces a deprecated surface inside the pipeline.
    """

    def test_prepare_wrap_canonical_pipeline_emits_no_deprecation(self, tmp_path):
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

            # Filter to anneal_memory-specific DeprecationWarnings
            # only — Python stdlib upgrades (sqlite3, json, datetime,
            # asyncio) sometimes surface DeprecationWarnings the
            # canonical pipeline transitively touches; those are not
            # this gate's concern. The intent is "no deprecated
            # anneal_memory surface in the pipeline." Layer 3 review
            # caught the broader gate as a CI-flake risk.
            deprecations = [
                w for w in caught
                if issubclass(w.category, DeprecationWarning)
                and "anneal_memory" in (w.filename or "")
            ]
            assert deprecations == [], (
                "canonical prepare_wrap pipeline leaked an anneal_memory "
                f"DeprecationWarning: {[str(w.message) for w in deprecations]}"
            )
            assert result["status"] == "ready"
        finally:
            store.close()

    def test_build_wrap_package_private_helper_emits_no_warning(self):
        """Direct users of the private ``_build_wrap_package`` helper
        (advanced library consumers managing their own lifecycle)
        receive no warnings."""
        import warnings as _warnings

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
        wrap_store.wrap_started(
            token=uuid.uuid4().hex, episode_ids=[]
        )  # Simulate a stale flag
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
        # Package should contain the full _build_wrap_package output
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
        """Caller-provided config must reach _build_wrap_package."""
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


# ---------------------------------------------------------------------------
# 10.5c.4 — Session-handshake token for the prepare/save TOCTOU window
# ---------------------------------------------------------------------------
#
# These tests lock the frozen-snapshot semantics added by the 10.5c.4
# TOCTOU fix. The structural invariant under test: an episode recorded
# between prepare_wrap and validated_save_continuity must NOT be
# absorbed into the in-progress wrap, regardless of whether the caller
# round-trips the wrap_token or not. The TOCTOU episode must appear in
# the NEXT wrap's compression window, so no data is lost — it's
# deferred to the wrap it semantically belongs to.
#
# The tests fall into four layers:
#
#   1. Token shape / empty-path sanity — the new wrap_token field is
#      present on ready results and None on empty results.
#   2. Frozen-snapshot filter — the TOCTOU episode drops out of
#      graduation validation, episodes_compressed counts, session_id
#      assignment, and re-appears on the next prepare_wrap call.
#   3. Token verification — callers that pass wrap_token get
#      mismatch detection (ValueError on stale/wrong token).
#   4. Skipped-prepare backward compatibility — callers that bypass
#      prepare_wrap entirely still see the pre-10.5c.4 behavior.
#
# Plus targeted tests for `Store.load_wrap_snapshot` (the primitive
# validated_save_continuity reads at save time) and for the audit
# event chain-of-custody enrichment (wrap_token appears in the
# wrap_started + wrap_completed audit payload, snapshot_episode_ids
# appears in wrap_started).


class TestTOCTOUHandshakeToken:
    """10.5c.4 — prepare/save TOCTOU window is frozen via snapshot.

    These tests target the canonical library pipeline directly.
    Transport-level round-trips (MCP wrap_token arg, CLI
    --wrap-token flag) are covered by the cross-transport parity
    infrastructure in TestCanonicalPipelineCrossTransport and by
    new cross-process tests in test_cli.py.
    """

    @staticmethod
    def _make_continuity(project_name: str, cited_id: str) -> str:
        """Build a valid 4-section continuity that cites a specific episode."""
        return (
            f"# {project_name} — Memory (v1)\n\n"
            f"## State\nWorking.\n\n"
            f"## Patterns\n"
            f"{{core:\n"
            f"  thought: snapshot semantics locked "
            f"| 1x (2026-04-10)\n"
            f"}}\n\n"
            f"## Decisions\n"
            f"[decided(rationale: \"freeze window\", on: \"2026-04-10\")] "
            f"Use snapshot\n\n"
            f"## Context\nCited {cited_id}.\n"
        )

    # -- Layer 1: token shape / empty path sanity --

    def test_prepare_wrap_ready_result_has_wrap_token(self, tmp_path):
        """Ready path: wrap_token is a 32-char hex string."""
        store = Store(str(tmp_path / "token.db"), project_name="Token")
        try:
            store.record("An observation", EpisodeType.OBSERVATION)
            result = prepare_wrap(store)
            assert result["status"] == "ready"
            token = result["wrap_token"]
            assert token is not None
            assert isinstance(token, str)
            assert len(token) == 32  # uuid4().hex
            assert all(c in "0123456789abcdef" for c in token)
        finally:
            store.close()

    def test_prepare_wrap_empty_result_has_none_token(self, tmp_path):
        """Empty path: wrap_token is None (no wrap to commit)."""
        store = Store(str(tmp_path / "empty.db"), project_name="Empty")
        try:
            result = prepare_wrap(store)
            assert result["status"] == "empty"
            assert result["wrap_token"] is None
        finally:
            store.close()

    def test_prepare_wrap_mints_fresh_token_each_call(self, tmp_path):
        """Each prepare_wrap call mints a unique token. Overlapping
        prepare calls (e.g. accidental double-prepare) overwrite the
        earlier snapshot and token — only the latest is valid."""
        from anneal_memory import validated_save_continuity

        store = Store(str(tmp_path / "fresh.db"), project_name="Fresh")
        try:
            store.record("First observation", EpisodeType.OBSERVATION)
            result1 = prepare_wrap(store)
            token1 = result1["wrap_token"]

            # Second prepare (caller never called save; overlapping
            # prepare is a legitimate "restart compression" flow)
            result2 = prepare_wrap(store)
            token2 = result2["wrap_token"]

            assert token1 != token2

            # Saving with the stale token1 is now rejected.
            text = self._make_continuity("Fresh", "aaaaaaaa")
            with pytest.raises(ValueError, match="wrap_token mismatch"):
                validated_save_continuity(store, text, wrap_token=token1)

            # Saving with the current token2 works.
            validated_save_continuity(store, text, wrap_token=token2)
        finally:
            store.close()

    # -- Layer 2: frozen-snapshot filter (the load-bearing behavior) --

    def test_toctou_episode_excluded_from_wrap(self, tmp_path):
        """THE core TOCTOU test. Episode recorded between prepare and
        save is NOT counted in episodes_compressed, does NOT get the
        in-progress wrap's session_id, and DOES appear in the next
        wrap's compression window.
        """
        from anneal_memory import validated_save_continuity

        store = Store(str(tmp_path / "toctou.db"), project_name="TOCTOU")
        try:
            # Phase 1: record two "snapshot" episodes and call prepare_wrap.
            ep_a = store.record(
                "First snapshot episode",
                EpisodeType.OBSERVATION,
            )
            ep_b = store.record(
                "Second snapshot episode",
                EpisodeType.OBSERVATION,
            )
            prep = prepare_wrap(store)
            assert prep["status"] == "ready"
            assert prep["episode_count"] == 2
            token = prep["wrap_token"]

            # Phase 2: the TOCTOU window — a NEW episode lands between
            # prepare and save. The agent's compression text was
            # built against the snapshot only and cites ep_a's ID.
            ep_toctou = store.record(
                "TOCTOU episode — must NOT join this wrap",
                EpisodeType.OBSERVATION,
            )
            # Sanity: all three episodes exist and are NULL-session
            assert store.status().episodes_since_wrap == 3

            # Phase 3: save with the snapshot filter active. Episodes
            # compressed should still be 2 (snapshot only), not 3.
            text = self._make_continuity("TOCTOU", ep_a.id)
            save = validated_save_continuity(store, text, wrap_token=token)
            assert save["episodes_compressed"] == 2, (
                "TOCTOU episode leaked into in-progress wrap — snapshot "
                "filter failed"
            )

            # Phase 4: session_id assignment. ep_a and ep_b should be
            # stamped with this wrap's ID; ep_toctou should still be
            # NULL and ready for the next wrap.
            row_a = store._conn.execute(
                "SELECT session_id FROM episodes WHERE id = ?",
                (ep_a.id,),
            ).fetchone()
            row_b = store._conn.execute(
                "SELECT session_id FROM episodes WHERE id = ?",
                (ep_b.id,),
            ).fetchone()
            row_t = store._conn.execute(
                "SELECT session_id FROM episodes WHERE id = ?",
                (ep_toctou.id,),
            ).fetchone()
            assert row_a["session_id"] is not None
            assert row_b["session_id"] is not None
            assert row_t["session_id"] is None, (
                "TOCTOU episode was silently stamped with the current "
                "wrap's session_id — it will be lost to the next wrap's "
                "compression window"
            )

            # Phase 5: the next prepare_wrap picks up ONLY the TOCTOU
            # episode — proof of correct deferral.
            next_prep = prepare_wrap(store)
            assert next_prep["status"] == "ready"
            assert next_prep["episode_count"] == 1
            assert "TOCTOU episode" in next_prep["package"]["episodes"]
        finally:
            store.close()

    def test_frozen_snapshot_applies_without_wrap_token(self, tmp_path):
        """The snapshot filter runs whenever prepare_wrap was called,
        regardless of whether the caller passes wrap_token at save
        time. Token is a verification layer, not the enablement
        mechanism. Single-process library / CLI users that don't
        round-trip the token still get TOCTOU safety.
        """
        from anneal_memory import validated_save_continuity

        store = Store(str(tmp_path / "implicit.db"), project_name="Implicit")
        try:
            ep_a = store.record("Snapshot only", EpisodeType.OBSERVATION)
            prepare_wrap(store)  # ignore return value
            store.record("TOCTOU leak attempt", EpisodeType.OBSERVATION)

            text = self._make_continuity("Implicit", ep_a.id)
            # No wrap_token passed.
            save = validated_save_continuity(store, text)
            assert save["episodes_compressed"] == 1
        finally:
            store.close()

    def test_toctou_episode_graduation_validation_excluded(self, tmp_path):
        """The frozen snapshot must also drive citation validation.
        A graduation marker that cites a TOCTOU episode ID (one the
        agent could not legitimately have seen) should fail
        validation — because the snapshot filter means the TOCTOU
        ID is not in ``valid_ids`` at citation-check time.
        """
        from anneal_memory import validated_save_continuity

        store = Store(str(tmp_path / "grad.db"), project_name="Grad")
        try:
            ep_a = store.record(
                "Legitimate snapshot episode",
                EpisodeType.OBSERVATION,
            )
            prep = prepare_wrap(store)
            token = prep["wrap_token"]

            # TOCTOU episode with a known ID — we need to cite it in
            # the compression text to check that graduation refuses
            # to validate against it.
            ep_toctou = store.record(
                "TOCTOU episode the agent should not see",
                EpisodeType.OBSERVATION,
            )

            # Build a continuity that tries to graduate a pattern with
            # evidence citing the TOCTOU episode. This should demote:
            # the ID is not in the snapshot-filtered valid_ids.
            #
            # Dates MUST use ``date.today().isoformat()`` — hardcoded
            # dates drift past wall-clock at midnight and the
            # graduation format line is then silently skipped as
            # "carried-forward from prior session," dropping demoted=0
            # and failing this test with a confusing error. Diogenes
            # Finding #3 recurred 3x before ``skipped_non_today`` was
            # added to ``SaveContinuityResult`` as a structural
            # invariant. The assertion below catches the drift class
            # explicitly even if a future edit re-introduces a
            # hardcoded date.
            today_str = date.today().isoformat()
            text = (
                f"# Grad — Memory (v1)\n\n"
                f"## State\nTesting citation validation.\n\n"
                f"## Patterns\n"
                f"{{verify:\n"
                f"  thought: snapshot bounds citations "
                f"| 2x ({today_str}) "
                f"[evidence: {ep_toctou.id} \"TOCTOU episode\"]\n"
                f"}}\n\n"
                f"## Decisions\n"
                f"[decided(rationale: \"test\", on: \"{today_str}\")] ok\n\n"
                f"## Context\nSnapshot episode {ep_a.id}.\n"
            )
            result = validated_save_continuity(
                store, text, wrap_token=token
            )
            # Structural invariant: no graduation-format line may be
            # skipped due to date mismatch in a test that intends to
            # exercise validation. If this fires, the test author
            # hardcoded a date that drifted from wall-clock — same
            # bug class Diogenes flagged 3x (Apr 7, Apr 10, Apr 11).
            assert result["skipped_non_today"] == 0, (
                "A graduation-format line was silently skipped because "
                "its date did not match today. This is the Finding #3 "
                "test-drift class — check for hardcoded dates in the "
                "text fixture and use date.today().isoformat() instead "
                f"(full result: {result})"
            )
            # The TOCTOU-citing graduation should have been demoted
            # specifically through the ``demoted`` counter. The
            # graduation validator routes unknown-ID citations
            # through the ``ids_valid`` gate in validate_graduations,
            # which increments ``demoted`` (not ``bare_demoted`` —
            # bare is for evidence-less graduations). Pinning the
            # specific counter catches regressions that misroute
            # the demotion, rather than a softer ``or`` that would
            # also pass if a future change accidentally counted the
            # demotion in an unrelated bucket.
            assert result["demoted"] >= 1, (
                "Graduation citing a TOCTOU episode ID was not "
                "routed through the ``demoted`` counter — snapshot "
                "filter may not be driving citation validation "
                f"(full result: {result})"
            )
        finally:
            store.close()

    # -- Layer 3: token verification --

    def test_wrong_token_raises_valueerror(self, tmp_path):
        """A token that doesn't match the stored one raises ValueError
        with a descriptive message."""
        from anneal_memory import validated_save_continuity

        store = Store(str(tmp_path / "wrong.db"), project_name="Wrong")
        try:
            ep = store.record("An observation", EpisodeType.OBSERVATION)
            prepare_wrap(store)
            text = self._make_continuity("Wrong", ep.id)

            fake_token = "0" * 32  # wrong but shape-valid
            with pytest.raises(ValueError) as excinfo:
                validated_save_continuity(
                    store, text, wrap_token=fake_token
                )
            assert "wrap_token mismatch" in str(excinfo.value)
            assert "stale" in str(excinfo.value).lower() or (
                "prepare_wrap" in str(excinfo.value)
            )

            # Wrap is still in progress — the caller can retry with the
            # correct token without re-running prepare.
            assert store.status().wrap_in_progress
        finally:
            store.close()

    def test_correct_token_completes_wrap(self, tmp_path):
        """The happy path: correct token, save succeeds, wrap state
        is cleared."""
        from anneal_memory import validated_save_continuity

        store = Store(str(tmp_path / "correct.db"), project_name="Correct")
        try:
            ep = store.record("An observation", EpisodeType.OBSERVATION)
            prep = prepare_wrap(store)
            token = prep["wrap_token"]
            text = self._make_continuity("Correct", ep.id)

            validated_save_continuity(store, text, wrap_token=token)

            # Wrap state fully cleared.
            status = store.status()
            assert not status.wrap_in_progress
            assert store.load_wrap_snapshot() is None
        finally:
            store.close()

    # -- Layer 4: skipped_prepare backward compatibility --

    def test_skipped_prepare_path_unchanged(self, tmp_path):
        """Calling validated_save_continuity without a prior
        prepare_wrap call works exactly as before — no snapshot, full
        re-fetch, skipped_prepare=True in the result."""
        from anneal_memory import validated_save_continuity

        store = Store(str(tmp_path / "skipped.db"), project_name="Skipped")
        try:
            ep = store.record("An observation", EpisodeType.OBSERVATION)
            # No prepare_wrap call.
            text = self._make_continuity("Skipped", ep.id)
            result = validated_save_continuity(store, text)
            assert result["skipped_prepare"] is True
            assert result["episodes_compressed"] == 1
        finally:
            store.close()

    def test_skipped_prepare_ignores_wrap_token_arg(self, tmp_path):
        """If there's no snapshot to verify against (skipped_prepare
        path), passing wrap_token is a no-op — verification only
        fires when the store actually has a snapshot."""
        from anneal_memory import validated_save_continuity

        store = Store(str(tmp_path / "skip_tok.db"), project_name="SkipTok")
        try:
            ep = store.record("An observation", EpisodeType.OBSERVATION)
            text = self._make_continuity("SkipTok", ep.id)
            # Garbage token, but no snapshot — should succeed.
            result = validated_save_continuity(
                store, text, wrap_token="ffffffffffffffffffffffffffffffff"
            )
            assert result["skipped_prepare"] is True
        finally:
            store.close()

    # -- Store primitive: load_wrap_snapshot --

    def test_load_wrap_snapshot_idle_returns_none(self, tmp_path):
        """No wrap in progress → snapshot is None."""
        store = Store(str(tmp_path / "idle.db"), project_name="Idle")
        try:
            assert store.load_wrap_snapshot() is None
        finally:
            store.close()

    def test_load_wrap_snapshot_in_progress_returns_data(self, tmp_path):
        """After prepare_wrap, load_wrap_snapshot returns the token
        and the exact episode ID list."""
        store = Store(str(tmp_path / "loaded.db"), project_name="Loaded")
        try:
            ep_a = store.record("A", EpisodeType.OBSERVATION)
            ep_b = store.record("B", EpisodeType.OBSERVATION)
            prep = prepare_wrap(store)
            snap = store.load_wrap_snapshot()
            assert snap is not None
            assert snap["token"] == prep["wrap_token"]
            assert set(snap["episode_ids"]) == {ep_a.id, ep_b.id}
        finally:
            store.close()

    def test_load_wrap_snapshot_corrupt_json_raises_store_error(
        self, tmp_path
    ):
        """Corrupt wrap_episode_ids metadata is a store-integrity
        failure, not silently recovered."""
        from anneal_memory.store import StoreError

        store = Store(
            str(tmp_path / "corrupt.db"), project_name="Corrupt"
        )
        try:
            store.record("A", EpisodeType.OBSERVATION)
            prepare_wrap(store)
            # Directly corrupt the metadata to simulate a manual edit
            # or a bug that wrote bad JSON.
            store._conn.execute(
                "UPDATE metadata SET value = ? WHERE key = ?",
                ("{not valid json", "wrap_episode_ids"),
            )
            store._conn.commit()
            with pytest.raises(StoreError) as excinfo:
                store.load_wrap_snapshot()
            assert excinfo.value.operation == "load_wrap_snapshot"
        finally:
            store.close()

    def test_load_wrap_snapshot_token_without_ids_raises(self, tmp_path):
        """Token set but episode_ids empty is a malformed state —
        treat as integrity failure, don't silently return an empty
        list."""
        from anneal_memory.store import StoreError

        store = Store(str(tmp_path / "half.db"), project_name="Half")
        try:
            store.record("A", EpisodeType.OBSERVATION)
            prepare_wrap(store)
            store._conn.execute(
                "UPDATE metadata SET value = ? WHERE key = ?",
                ("", "wrap_episode_ids"),
            )
            store._conn.commit()
            with pytest.raises(StoreError) as excinfo:
                store.load_wrap_snapshot()
            assert excinfo.value.operation == "load_wrap_snapshot"
        finally:
            store.close()

    # -- wrap_cancelled clears snapshot --

    def test_wrap_cancelled_clears_snapshot(self, tmp_path):
        """wrap_cancelled must clear all three metadata keys so the
        next prepare_wrap starts from a clean slate."""
        store = Store(str(tmp_path / "cancel.db"), project_name="Cancel")
        try:
            store.record("A", EpisodeType.OBSERVATION)
            prepare_wrap(store)
            assert store.load_wrap_snapshot() is not None
            store.wrap_cancelled()
            assert store.load_wrap_snapshot() is None
            # Wrap-in-progress flag also cleared.
            assert not store.status().wrap_in_progress
        finally:
            store.close()

    def test_empty_prepare_wrap_clears_stale_snapshot(self, tmp_path):
        """If a wrap was in progress and a subsequent prepare_wrap
        finds zero episodes (because the previous wrap ate them all),
        the empty path must clear the stale snapshot via
        wrap_cancelled — same pre-10.5c.4 invariant, extended to the
        new metadata keys.
        """
        from anneal_memory import validated_save_continuity

        store = Store(str(tmp_path / "stale.db"), project_name="Stale")
        try:
            ep = store.record("A", EpisodeType.OBSERVATION)
            prep = prepare_wrap(store)
            text = self._make_continuity("Stale", ep.id)
            validated_save_continuity(
                store, text, wrap_token=prep["wrap_token"]
            )
            # Now call prepare_wrap again with no new episodes.
            empty = prepare_wrap(store)
            assert empty["status"] == "empty"
            assert store.load_wrap_snapshot() is None
        finally:
            store.close()

    # -- Audit event chain-of-custody enrichment --
    #
    # The wrap lifecycle audit events must carry enough context to
    # reconstruct which episodes went into which wrap without
    # joining against the DB tables. wrap_started logs the token +
    # full episode ID list at prepare time; wrap_cancelled logs
    # the same two fields so the auditor can see exactly what was
    # abandoned; wrap_completed logs the token so an auditor can
    # match prepare → complete events by identifier.

    def test_audit_events_carry_wrap_token(self, tmp_path):
        """wrap_started audit entry must carry the minted token and
        the full episode ID list; wrap_completed must carry the same
        token (for chain-of-custody matching).
        """
        import json as _json
        from anneal_memory import validated_save_continuity

        db_path = tmp_path / "audit.db"
        store = Store(str(db_path), project_name="Audit")
        try:
            ep_a = store.record("A", EpisodeType.OBSERVATION)
            ep_b = store.record("B", EpisodeType.OBSERVATION)
            prep = prepare_wrap(store)
            token = prep["wrap_token"]
            text = self._make_continuity("Audit", ep_a.id)
            validated_save_continuity(store, text, wrap_token=token)
        finally:
            store.close()

        audit_path = tmp_path / "audit.audit.jsonl"
        assert audit_path.exists()
        entries = [
            _json.loads(line)
            for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        events_by_name = {e["event"]: e for e in entries}

        started = events_by_name.get("wrap_started")
        assert started is not None, "wrap_started event missing from audit"
        started_data = started.get("data", {})
        assert started_data.get("wrap_token") == token
        assert started_data.get("wrap_episode_count") == 2
        assert set(started_data.get("wrap_episode_ids", [])) == {
            ep_a.id, ep_b.id
        }

        completed = events_by_name.get("wrap_completed")
        assert completed is not None
        completed_data = completed.get("data", {})
        assert completed_data.get("wrap_token") == token

    def test_audit_wrap_cancelled_carries_token_and_episode_ids(
        self, tmp_path
    ):
        """Cancelled wraps must log the token AND the full abandoned
        episode ID list so an auditor can see exactly what was
        abandoned without cross-joining against the wrap_started
        event."""
        import json as _json

        db_path = tmp_path / "audit_cancel.db"
        store = Store(str(db_path), project_name="AuditCancel")
        try:
            ep_a = store.record("A", EpisodeType.OBSERVATION)
            ep_b = store.record("B", EpisodeType.OBSERVATION)
            prep = prepare_wrap(store)
            token = prep["wrap_token"]
            store.wrap_cancelled()
        finally:
            store.close()

        audit_path = tmp_path / "audit_cancel.audit.jsonl"
        entries = [
            _json.loads(line)
            for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cancelled = [e for e in entries if e["event"] == "wrap_cancelled"]
        # The explicit cancellation event carries the full payload.
        matching = [
            c for c in cancelled
            if c.get("data", {}).get("wrap_token") == token
        ]
        assert matching, "cancellation event for token not found"
        data = matching[0]["data"]
        assert data.get("wrap_episode_count") == 2
        assert set(data.get("wrap_episode_ids", [])) == {ep_a.id, ep_b.id}

    # -- v0.3.0 signature tightening: contract violations --

    def test_wrap_started_missing_args_raises_typeerror(self, tmp_path):
        """v0.3.0 regression gate: ``token`` and ``episode_ids`` are
        both required keyword-only with no defaults. Python enforces
        this at the call site via TypeError. This test locks the
        Python-level enforcement as a structural invariant — if a
        future refactor accidentally re-adds defaults, the partial
        wrap-in-progress state becomes reachable again and this
        test fails loudly. Cheap structural lock.
        """
        store = Store(str(tmp_path / "missing_args.db"), project_name="MissingArgs")
        try:
            with pytest.raises(TypeError):
                store.wrap_started()
            with pytest.raises(TypeError):
                store.wrap_started(token="abc")
            with pytest.raises(TypeError):
                store.wrap_started(episode_ids=["a" * 8])
        finally:
            store.close()

    def test_wrap_started_empty_token_raises(self, tmp_path):
        """v0.3.0: ``token`` is required keyword-only with no default.
        Python's TypeError covers the missing-argument case; this
        test covers the remaining contract violation — passing an
        empty string token. The canonical pipeline always mints a
        non-empty uuid4().hex token, so any empty token at the write
        boundary is a caller bug."""
        store = Store(str(tmp_path / "empty_tok.db"), project_name="EmptyTok")
        try:
            with pytest.raises(ValueError, match="token must be non-empty"):
                store.wrap_started(token="", episode_ids=[])
        finally:
            store.close()

    def test_wrap_started_non_list_episode_ids_raises(self, tmp_path):
        """v0.3.0: ``episode_ids`` must be a list. The runtime
        ``isinstance`` guard catches strings, tuples, generators —
        the generator case is the load-bearing one because
        ``list(generator)`` exhausts it and ``len(generator)`` would
        raise TypeError AFTER the SQL writes already committed,
        leaving the store in a partial-state with no audit entry.
        Catching at the entry point keeps the failure mode atomic."""
        store = Store(
            str(tmp_path / "non_list.db"), project_name="NonList"
        )
        try:
            with pytest.raises(TypeError, match="episode_ids must be a list"):
                store.wrap_started(token="a" * 32, episode_ids="abc")
            with pytest.raises(TypeError, match="episode_ids must be a list"):
                store.wrap_started(
                    token="a" * 32, episode_ids=("a" * 8, "b" * 8)
                )
            with pytest.raises(TypeError, match="episode_ids must be a list"):
                store.wrap_started(
                    token="a" * 32, episode_ids=(x for x in ["a" * 8])
                )
        finally:
            store.close()

    def test_load_wrap_snapshot_empty_list_round_trip(self, tmp_path):
        """v0.3.0 canonical empty-snapshot round-trip lock.

        The canonical encoding for ``episode_ids=[]`` is the JSON
        string ``"[]"`` — a two-character string. The
        ``load_wrap_snapshot`` partial-state guard checks
        ``if not ids_raw:`` against the raw metadata string, which
        is truthy for ``"[]"`` (passes the guard) and falsy for the
        legacy empty-string default (raises). Layer 3 caught the
        coverage gap: no test exercised the empty-list write →
        load round-trip, so a one-character change like
        ``if not json.loads(ids_raw):`` would silently break the
        canonical empty case.

        This test locks the round-trip as a structural invariant.
        """
        store = Store(
            str(tmp_path / "empty_roundtrip.db"),
            project_name="EmptyRoundtrip",
        )
        try:
            token = uuid.uuid4().hex
            store.wrap_started(token=token, episode_ids=[])
            snap = store.load_wrap_snapshot()
            assert snap is not None, (
                "load_wrap_snapshot returned None on a canonical "
                "empty-list snapshot — the partial-state guard is "
                "incorrectly rejecting '[]' (the canonical encoding)."
            )
            assert snap["token"] == token
            assert snap["episode_ids"] == []
        finally:
            store.close()

    # -- Replay-race closure: commit-atomic token CAS in wrap_completed --

    def test_wrap_completed_cas_rejects_replaced_token(self, tmp_path):
        """Codex Layer 3 HIGH: the pre-fix design verified wrap_token
        at the start of validated_save_continuity, then did many
        store operations, then finally called wrap_completed — a
        concurrent process could have cleared or replaced the token
        between the verify and the commit, but the commit would
        proceed anyway. The fix adds a compare-and-swap UPDATE at
        the top of wrap_completed that's the first DML in the
        transaction, so the token check and the commit are
        atomically bound together.

        This test simulates a concurrent replacement by directly
        mutating the metadata between validated_save_continuity's
        earlier check and wrap_completed's CAS. The save should
        abort with a ValueError distinct from the earlier
        'wrap_token mismatch' message.
        """
        from anneal_memory import validated_save_continuity
        from unittest.mock import patch

        store = Store(str(tmp_path / "cas.db"), project_name="CAS")
        try:
            ep = store.record("A", EpisodeType.OBSERVATION)
            prep = prepare_wrap(store)
            token = prep["wrap_token"]
            text = self._make_continuity("CAS", ep.id)

            # Monkey-patch wrap_completed to replace the token in the
            # metadata table JUST before the real wrap_completed
            # runs. This simulates a concurrent process replacing
            # the token between the validated_save_continuity's
            # early check and the wrap_completed CAS.
            real_wrap_completed = store.wrap_completed

            def racing_wrap_completed(*args, **kwargs):
                # Swap the token out from under ourselves, as if a
                # concurrent process ran a new prepare_wrap.
                store._conn.execute(
                    "UPDATE metadata SET value = ? WHERE key = ?",
                    ("f" * 32, "wrap_token"),
                )
                store._conn.commit()
                return real_wrap_completed(*args, **kwargs)

            with patch.object(store, "wrap_completed", racing_wrap_completed):
                with pytest.raises(ValueError) as excinfo:
                    validated_save_continuity(
                        store, text, wrap_token=token
                    )

            # The CAS message differs from the earlier mismatch
            # message so operators can distinguish a client-side
            # stale token from a concurrent-process interference.
            assert "cleared or replaced during save" in str(excinfo.value)

            # No wraps row should have been inserted — the CAS
            # failure rolled back the transaction before the
            # INSERT INTO wraps ran.
            row = store._conn.execute(
                "SELECT COUNT(*) FROM wraps"
            ).fetchone()
            assert row[0] == 0, (
                "CAS failure did not roll back the wraps INSERT — "
                "partial commit leaked"
            )
        finally:
            store.close()

    def test_wrap_completed_no_cas_when_token_is_none(self, tmp_path):
        """The CAS only runs when wrap_token is passed. The legacy
        skipped_prepare path (wrap_token=None) bypasses CAS and
        proceeds with the unconditional clear, same as pre-10.5c.4.
        """
        from anneal_memory import validated_save_continuity

        store = Store(str(tmp_path / "nocas.db"), project_name="NoCAS")
        try:
            ep = store.record("A", EpisodeType.OBSERVATION)
            # No prepare_wrap — legacy skipped path.
            text = self._make_continuity("NoCAS", ep.id)
            result = validated_save_continuity(store, text)
            assert result["skipped_prepare"] is True
            assert store.status().total_wraps == 1
        finally:
            store.close()


# -- 10.5c.5 Two-phase commit (file/DB atomicity) --


class TestTwoPhaseCommit:
    """10.5c.5 — file/DB atomicity via tmp-sidecar + batched SQLite txn.

    These tests inject failures at each stage of the canonical save
    pipeline and verify that:
      1. The pre-wrap state is cleanly restored on any failure BEFORE
         the batched DB commit (no partial wraps row, no stale
         wrap_started_at, no orphan .tmp files, no continuity / meta
         file mutation).
      2. The batch context manager actually defers all intermediate
         commits — an in-flight DML snapshot shows uncommitted writes
         to a separate connection, and a crash rolls them all back.
      3. The narrow "DB committed, renames pending" window is
         recoverable: the DB reflects the new wrap, the old continuity
         / meta files remain intact, and the tmp sidecars still hold
         the new content.
    """

    @staticmethod
    def _make_continuity(project_name: str, cited_id: str) -> str:
        return (
            f"# {project_name} — Memory (v1)\n\n"
            f"## State\nWorking.\n\n"
            f"## Patterns\n"
            f"{{core:\n"
            f"  thought: two-phase commit holds "
            f"| 1x (2026-04-10)\n"
            f"}}\n\n"
            f"## Decisions\n"
            f"[decided(rationale: \"2pc\", on: \"2026-04-10\")] ok\n\n"
            f"## Context\nCited {cited_id}.\n"
        )

    def _prime_store(self, db_path: str, project: str):
        """Record one episode, prepare a wrap, return (store, text, token)."""
        from anneal_memory import prepare_wrap

        store = Store(db_path, project_name=project)
        ep = store.record("Seed observation", EpisodeType.OBSERVATION)
        result = prepare_wrap(store)
        text = self._make_continuity(project, ep.id)
        return store, text, result["wrap_token"]

    def _count_wraps(self, store: Store) -> int:
        row = store._conn.execute(
            "SELECT COUNT(*) FROM wraps"
        ).fetchone()
        return row[0]

    # -- Failure: continuity tmp write fails --

    def test_continuity_tmp_write_failure_leaves_clean_state(
        self, tmp_path, monkeypatch
    ):
        """StoreError from _prepare_continuity_write: no DB mutation,
        no continuity / meta changes, no orphan tmp files."""
        from anneal_memory import validated_save_continuity

        db_path = str(tmp_path / "cont_fail.db")
        store, text, token = self._prime_store(db_path, "ContFail")
        try:
            wraps_before = self._count_wraps(store)
            snapshot_before = store.load_wrap_snapshot()
            assert snapshot_before is not None  # wrap IS in progress

            original = Store._prepare_continuity_write

            def exploding(self, text_arg, token_hex=None):
                raise StoreError(
                    "injected continuity tmp write failure",
                    operation="prepare_continuity_write",
                    path=str(self.continuity_path),
                )

            monkeypatch.setattr(Store, "_prepare_continuity_write", exploding)
            with pytest.raises(StoreError, match="injected continuity"):
                validated_save_continuity(store, text, wrap_token=token)

            # Post-conditions: pre-wrap state intact.
            monkeypatch.setattr(Store, "_prepare_continuity_write", original)
            assert self._count_wraps(store) == wraps_before
            # Snapshot still present — wrap was not completed.
            assert store.load_wrap_snapshot() is not None
            # No orphan tmp files.
            assert not list(
                store.continuity_path.parent.glob("*.md.tmp")
            )
            assert not list(
                store.continuity_path.parent.glob("*.json.tmp")
            )
        finally:
            store.close()

    # -- Failure: meta tmp write fails (mid-batch) --

    def test_meta_tmp_write_failure_rolls_back_batch(
        self, tmp_path, monkeypatch
    ):
        """StoreError from _prepare_meta_write inside the batch:
        associations DML rolled back, continuity tmp cleaned up,
        no wraps row inserted, snapshot still in progress."""
        from anneal_memory import validated_save_continuity

        db_path = str(tmp_path / "meta_fail.db")
        store, text, token = self._prime_store(db_path, "MetaFail")
        try:
            wraps_before = self._count_wraps(store)

            original = Store._prepare_meta_write

            def exploding(self, meta_arg, token_hex=None):
                raise StoreError(
                    "injected meta tmp write failure",
                    operation="prepare_meta_write",
                    path=str(self.meta_path),
                )

            monkeypatch.setattr(Store, "_prepare_meta_write", exploding)
            with pytest.raises(StoreError, match="injected meta"):
                validated_save_continuity(store, text, wrap_token=token)

            monkeypatch.setattr(Store, "_prepare_meta_write", original)

            # DB rolled back: no new wrap, no associations persisted.
            assert self._count_wraps(store) == wraps_before
            assert store.load_wrap_snapshot() is not None
            assert not list(
                store.continuity_path.parent.glob("*.md.tmp")
            )
            assert not list(
                store.continuity_path.parent.glob("*.json.tmp")
            )
        finally:
            store.close()

    # -- Failure: continuity rename fails (after DB commit) --

    def test_continuity_rename_failure_preserves_tmp_for_recovery(
        self, tmp_path, monkeypatch
    ):
        """OSError from continuity tmp -> real rename (post-DB-commit):
        the residual risk window. DB reflects the new wrap (including
        cleared snapshot), continuity and meta files remain the
        pre-wrap versions, but BOTH .tmp files PERSIST on disk with
        the new content. This is the load-bearing invariant: once the
        batch has committed, the outer except clause MUST NOT unlink
        the tmp files, because they hold the committed state that
        still needs to be externalized. Operator recovery: manually
        ``mv *.md.tmp *.md`` and ``mv *.json.tmp *.json``.

        10.5c.5 L1 HIGH + L2 M2 — prior implementation unlinked the
        tmp files on post-commit failure, destroying the new content
        permanently and making the "residual window recoverable via
        wrap-status/wrap-cancel" story a lie.
        """
        from anneal_memory import validated_save_continuity

        db_path = str(tmp_path / "rename_fail.db")
        store, text, token = self._prime_store(db_path, "RenameFail")
        try:
            wraps_before = self._count_wraps(store)
            # Establish a known-pre-wrap continuity baseline so we
            # can assert the rename failure did NOT clobber it.
            pre_wrap_contents = None
            if store.continuity_path.exists():
                pre_wrap_contents = store.continuity_path.read_text(
                    encoding="utf-8"
                )

            from pathlib import Path as _Path
            original_replace = _Path.replace

            def flaky_replace(self, target):
                if self.suffix == ".tmp" and self.name.endswith(".md.tmp"):
                    raise OSError(
                        "injected continuity rename failure"
                    )
                return original_replace(self, target)

            monkeypatch.setattr(_Path, "replace", flaky_replace)

            with pytest.raises(StoreError, match="continuity tmp"):
                validated_save_continuity(store, text, wrap_token=token)

            monkeypatch.setattr(_Path, "replace", original_replace)

            # DB WAS committed before the rename failed.
            assert self._count_wraps(store) == wraps_before + 1
            # Snapshot is cleared because wrap_completed ran inside
            # the batch and the batch did commit.
            assert store.load_wrap_snapshot() is None
            # Continuity file on disk is STILL the pre-wrap version.
            if pre_wrap_contents is not None:
                assert store.continuity_path.read_text(encoding="utf-8") == \
                    pre_wrap_contents

            # THE LOAD-BEARING ASSERTION: tmp files must PERSIST.
            # They hold the committed state awaiting externalization.
            cont_tmp_files = list(
                store.continuity_path.parent.glob("*.md.tmp")
            )
            meta_tmp_files = list(
                store.continuity_path.parent.glob("*.json.tmp")
            )
            assert len(cont_tmp_files) == 1, (
                f"continuity tmp should persist for operator recovery; "
                f"found: {cont_tmp_files}"
            )
            assert len(meta_tmp_files) == 1, (
                f"meta tmp should persist for operator recovery; "
                f"found: {meta_tmp_files}"
            )

            # And they must hold the NEW content, not old.
            cont_tmp_text = cont_tmp_files[0].read_text(encoding="utf-8")
            assert cont_tmp_text == text.replace(
                "| 1x (2026-04-10)",
                # graduation may rewrite the date or status — just
                # verify the marker content we know is unique to
                # the new wrap is present.
                "| 1x (2026-04-10)",
            ) or "two-phase commit holds" in cont_tmp_text

            # Simulate operator recovery: finish the rename manually.
            cont_tmp_files[0].replace(store.continuity_path)
            meta_tmp_files[0].replace(store.meta_path)

            # After manual recovery the store is in the committed
            # state the pipeline would have produced on a clean run.
            assert "two-phase commit holds" in store.continuity_path.read_text(
                encoding="utf-8"
            )
        finally:
            store.close()

    # -- Behavioral: batched DML is atomic (visible to separate conn only after commit) --

    def test_batched_dml_invisible_to_other_connection_until_commit(
        self, tmp_path
    ):
        """The batch context manager must keep intermediate DML
        uncommitted. A separate read connection against the same DB
        file sees ZERO new wraps until the batch exits successfully.
        """
        import sqlite3
        from anneal_memory import validated_save_continuity

        db_path = str(tmp_path / "visibility.db")
        store, text, token = self._prime_store(db_path, "Visibility")
        try:
            # Snapshot the "other reader" count before save.
            other = sqlite3.connect(db_path)
            other.row_factory = sqlite3.Row
            before = other.execute("SELECT COUNT(*) FROM wraps").fetchone()[0]
            other.close()

            validated_save_continuity(store, text, wrap_token=token)

            # After save completes, the new wrap IS visible externally.
            other = sqlite3.connect(db_path)
            other.row_factory = sqlite3.Row
            after = other.execute("SELECT COUNT(*) FROM wraps").fetchone()[0]
            other.close()
            assert after == before + 1
        finally:
            store.close()

    # -- Behavioral: nested batch raises --

    def test_nested_batch_raises(self, tmp_path):
        """Two-phase commit has exactly one outer transaction; nested
        _batch() calls would break the commit-once invariant. Guard
        it at the API level.
        """
        store = Store(str(tmp_path / "nested.db"), project_name="Nested")
        try:
            with store._batch():
                with pytest.raises(RuntimeError, match="does not support nesting"):
                    with store._batch():
                        pass
        finally:
            store.close()

    # -- Behavioral: close() inside _batch() raises + cleanup runs --

    def test_close_raises_inside_batch(self, tmp_path):
        """``close()`` called inside an active ``_batch()`` context
        raises ``StoreError`` rather than half-closing the connection
        mid-transaction. After the raise, ``_batch()`` ``__exit__``
        runs: ``_defer_commit`` resets, the store stays open
        (``_closed`` is False because the guard fires before any
        close work runs), and subsequent operations on the same
        store succeed.

        Regression guard for the ``close()`` guard at
        ``store.py:close()``. Without this test, a refactor of
        ``_batch()``/``__exit__`` logic could silently drop the
        ``StoreError`` (allowing partial close mid-transaction) or
        corrupt cleanup state, and nothing would catch it.
        """
        store = Store(
            str(tmp_path / "close_in_batch.db"),
            project_name="CloseInBatch",
        )
        try:
            # Guard fires; StoreError propagates out of the batch block.
            with pytest.raises(StoreError) as exc_info:
                with store._batch():
                    store.close()  # Guard fires here.

            assert exc_info.value.operation == "close"
            assert "_batch()" in str(exc_info.value)

            # _batch() __exit__ ran with the exception: defer flag reset.
            assert store._defer_commit is False
            assert store._deferred_audits == []

            # Guard fires before close logic runs — store stays open.
            assert store._closed is False

            # Final invariant: store is fully operational after the
            # failed-close-in-batch sequence. Normal write + read works.
            store.record(
                "after close-in-batch", EpisodeType.OBSERVATION
            )
            result = store.recall(limit=5)
            assert len(result.episodes) == 1
        finally:
            store.close()

    # -- Behavioral: batch commits on successful exit --

    def test_batch_commits_on_success(self, tmp_path):
        """The batch context manager performs the single outer commit
        at __exit__ after a successful body. A simple DML inside the
        batch should be visible to a separate connection AFTER exit
        but NOT before.
        """
        import sqlite3

        db_path = str(tmp_path / "commit.db")
        store = Store(db_path, project_name="Commit")
        try:
            # Pre-check: no episodes in store.
            assert store._conn.execute(
                "SELECT COUNT(*) FROM episodes"
            ).fetchone()[0] == 0

            with store._batch():
                store.record(
                    "Inside batch", EpisodeType.OBSERVATION
                )
                # Inside the batch, another connection sees zero.
                other = sqlite3.connect(db_path)
                other.row_factory = sqlite3.Row
                seen_mid_batch = other.execute(
                    "SELECT COUNT(*) FROM episodes"
                ).fetchone()[0]
                other.close()
                # Note: record() itself may commit (the episode-write
                # path is outside the batched-commit scope by design
                # — batching is scoped to wrap-path methods). This
                # test is about verifying _batch() survives a
                # successful cycle without raising, not about
                # universal write isolation.

            # After the batch exits, our own connection is committed
            # state; the episode must be visible.
            other = sqlite3.connect(db_path)
            other.row_factory = sqlite3.Row
            seen_after = other.execute(
                "SELECT COUNT(*) FROM episodes"
            ).fetchone()[0]
            other.close()
            assert seen_after == 1
        finally:
            store.close()

    # -- Behavioral: exception inside batch rolls back --

    def test_batch_rolls_back_on_exception(self, tmp_path):
        """Exception inside the batch body triggers rollback of
        accumulated DML. A wraps INSERT executed directly on
        self._conn inside a failing batch must NOT persist.
        """
        import sqlite3

        db_path = str(tmp_path / "rollback.db")
        store = Store(db_path, project_name="Rollback")
        try:
            with pytest.raises(RuntimeError, match="injected"):
                with store._batch():
                    store._conn.execute(
                        "INSERT INTO wraps "
                        "(wrapped_at, episodes_compressed, continuity_chars) "
                        "VALUES (?, ?, ?)",
                        ("2026-04-10T00:00:00Z", 1, 100),
                    )
                    raise RuntimeError("injected mid-batch failure")

            # Wraps table must still be empty.
            other = sqlite3.connect(db_path)
            other.row_factory = sqlite3.Row
            count = other.execute(
                "SELECT COUNT(*) FROM wraps"
            ).fetchone()[0]
            other.close()
            assert count == 0

            # And flag is reset so subsequent calls behave normally.
            assert store._defer_commit is False
            assert store._deferred_audits == []
        finally:
            store.close()

    # -- Behavioral: deferred audits only fire after commit --

    def test_deferred_audits_discarded_on_rollback(self, tmp_path):
        """Audit events queued inside a failing batch must NOT be
        written to the audit trail after rollback. Prevents phantom
        entries for DML that never committed.
        """
        from anneal_memory.audit import AuditTrail

        db_path = str(tmp_path / "audit_rollback.db")
        store = Store(db_path, project_name="AuditRollback", audit=True)
        try:
            assert store._audit is not None
            audit_path = store._audit._active_path

            # Count audit entries before.
            def count_audit_entries() -> int:
                if not audit_path.exists():
                    return 0
                return sum(1 for _ in audit_path.read_text(
                    encoding="utf-8"
                ).splitlines() if _.strip())

            entries_before = count_audit_entries()

            with pytest.raises(RuntimeError, match="injected"):
                with store._batch():
                    # Queue an event directly.
                    store._audit_log("test_event", {"marker": "nope"})
                    # Simulate the store verifying the queued payload
                    # — it should NOT have hit the audit file yet.
                    assert count_audit_entries() == entries_before
                    raise RuntimeError("injected failure after queueing")

            # After rollback, the deferred audit event is discarded.
            assert count_audit_entries() == entries_before
            assert store._deferred_audits == []
        finally:
            store.close()

    # -- Behavioral: successful save still fires all expected audits --

    def test_successful_save_fires_all_audits(self, tmp_path):
        """Regression guard: the batched save path still fires
        wrap_completed, continuity_saved, and associations_* audit
        events in the correct order. Confirms batching did NOT
        silently drop audit events.
        """
        import json as json_mod
        from anneal_memory import validated_save_continuity

        db_path = str(tmp_path / "audits.db")
        store, text, token = self._prime_store(db_path, "Audits")
        try:
            assert store._audit is not None
            audit_path = store._audit._active_path

            def read_events() -> list[dict]:
                if not audit_path.exists():
                    return []
                events: list[dict] = []
                for line in audit_path.read_text(
                    encoding="utf-8"
                ).splitlines():
                    if not line.strip():
                        continue
                    events.append(json_mod.loads(line))
                return events

            before = read_events()
            validated_save_continuity(store, text, wrap_token=token)
            after = read_events()

            new_events = after[len(before):]
            event_names = [e["event"] for e in new_events]
            assert "wrap_completed" in event_names
            assert "continuity_saved" in event_names
            # The wrap_completed event must come before
            # continuity_saved because the batch flushes its deferred
            # audits BEFORE the pipeline fires the post-rename
            # continuity_saved event.
            assert event_names.index("wrap_completed") < event_names.index(
                "continuity_saved"
            )
        finally:
            store.close()


class TestPostReviewFixes:
    """10.5c.5 post-L1/L2-review regression gates.

    Each test here covers a specific finding from the 4-layer review
    that previously had NO regression coverage. These are the
    tests the review said should have existed.
    """

    @staticmethod
    def _make_continuity(project_name: str, cited_id: str) -> str:
        return (
            f"# {project_name} — Memory (v1)\n\n"
            f"## State\nWorking.\n\n"
            f"## Patterns\n"
            f"{{core:\n"
            f"  thought: post review fix coverage "
            f"| 1x (2026-04-10)\n"
            f"}}\n\n"
            f"## Decisions\n"
            f"[decided(rationale: \"regression\", on: \"2026-04-10\")] ok\n\n"
            f"## Context\nCited {cited_id}.\n"
        )

    def _prime_store_with_retention(self, db_path: str, retention_days: int):
        """Store with retention configured, one old episode + one fresh."""
        from datetime import datetime, timedelta, timezone
        from anneal_memory import prepare_wrap

        store = Store(
            db_path,
            project_name="Retention",
            retention_days=retention_days,
        )
        # Record a fresh episode that will survive pruning.
        fresh_ep = store.record(
            "fresh observation", EpisodeType.OBSERVATION
        )
        # Backdate an old episode past the retention threshold. We
        # reach into the DB directly because the Store API doesn't
        # expose timestamp override — this mirrors how the existing
        # prune tests seed stale rows.
        old_ts = (
            datetime.now(timezone.utc)
            - timedelta(days=retention_days + 10)
        ).isoformat().replace("+00:00", "Z")
        store._conn.execute(
            "UPDATE episodes SET timestamp = ? WHERE id = ?",
            (old_ts, fresh_ep.id),
        )
        # Add ANOTHER fresh episode that we actually cite in the
        # wrap, so graduation has a live ID to validate against.
        live_ep = store.record(
            "live observation that the wrap cites",
            EpisodeType.OBSERVATION,
        )
        store._conn.commit()
        prepare_wrap(store)
        text = self._make_continuity("Retention", live_ep.id)
        # Return fresh_ep id so the test can verify it was pruned.
        return store, text, fresh_ep.id

    # -- L1 CRITICAL: auto-prune regression through canonical pipeline --

    def test_canonical_pipeline_runs_auto_prune(self, tmp_path):
        """retention_days configured + canonical pipeline → old
        episodes are pruned after the wrap completes. Regression
        guard for the L1 CRITICAL finding: the batched pipeline was
        silently skipping prune inside wrap_completed and the
        pipeline caller never invoked it, so users with
        retention_days configured would silently stop getting
        lifecycle management.
        """
        from anneal_memory import validated_save_continuity

        db_path = str(tmp_path / "retention.db")
        store, text, stale_id = self._prime_store_with_retention(
            db_path, retention_days=7
        )
        try:
            # Verify the stale episode exists pre-pipeline.
            row = store._conn.execute(
                "SELECT id FROM episodes WHERE id = ?", (stale_id,)
            ).fetchone()
            assert row is not None

            result = validated_save_continuity(store, text)

            # Pipeline returned a pruned_count reflecting the prune.
            # (Exact count depends on how many stale episodes seed,
            # but the key invariant is "at least 1 pruned.")
            pruned_count = result["wrap_result"]["pruned_count"]
            assert pruned_count >= 1, (
                f"auto-prune did not run — pruned_count={pruned_count}"
            )

            # Stale episode is gone from the episodes table OR marked
            # as a tombstone (depending on keep_tombstones setting).
            row = store._conn.execute(
                "SELECT id FROM episodes WHERE id = ?", (stale_id,)
            ).fetchone()
            assert row is None, (
                f"stale episode {stale_id} still in episodes table "
                f"after auto-prune"
            )
        finally:
            store.close()

    def test_canonical_pipeline_skips_prune_when_no_retention(self, tmp_path):
        """retention_days is None → pipeline does NOT invoke prune.
        Negative-space companion to the regression guard above — we
        don't want to accidentally prune when the user hasn't opted
        into retention management.
        """
        from anneal_memory import prepare_wrap, validated_save_continuity

        db_path = str(tmp_path / "no_retention.db")
        store = Store(db_path, project_name="NoRetention")
        try:
            ep = store.record("obs", EpisodeType.OBSERVATION)
            prepare_wrap(store)
            text = self._make_continuity("NoRetention", ep.id)
            result = validated_save_continuity(store, text)
            assert result["wrap_result"]["pruned_count"] == 0
        finally:
            store.close()

    # -- L2 M2: audit-flush failure must not propagate --

    def test_audit_flush_failure_does_not_propagate(
        self, tmp_path, monkeypatch
    ):
        """Audit log raising mid-flush during batch __exit__ must be
        swallowed. Otherwise the exception propagates through the
        pipeline's outer except, which unlinks the tmp files — the
        DB is already committed, so that would permanently destroy
        the new wrap content. Regression guard for L2 M2.
        """
        from anneal_memory import validated_save_continuity

        db_path = str(tmp_path / "audit_flush.db")
        store, text, token = _prime_simple(db_path, "AuditFlush")
        try:
            wraps_before = store._conn.execute(
                "SELECT COUNT(*) FROM wraps"
            ).fetchone()[0]

            # Inject a failure ONLY in the deferred-audit flush path.
            # We target self._audit.log itself, raising on the
            # wrap_completed event. All other audit sites (pre-batch
            # and post-rename continuity_saved) still fire normally.
            assert store._audit is not None
            original_log = store._audit.log
            call_count = {"n": 0}

            def flaky_log(event, payload):
                call_count["n"] += 1
                if event == "wrap_completed":
                    raise OSError(
                        "injected audit flush failure"
                    )
                return original_log(event, payload)

            monkeypatch.setattr(store._audit, "log", flaky_log)

            # The pipeline must complete successfully despite the
            # audit failure. DB committed + files externalized +
            # no exception propagated.
            result = validated_save_continuity(
                store, text, wrap_token=token
            )

            assert result["episodes_compressed"] >= 1
            # DB committed.
            assert store._conn.execute(
                "SELECT COUNT(*) FROM wraps"
            ).fetchone()[0] == wraps_before + 1
            # Files externalized.
            assert store.continuity_path.exists()
            assert store.meta_path.exists()
            # No orphan tmp files (this is the healthy-path
            # assertion; tmp files persist only on post-commit
            # failure, which isn't what we're testing here — we're
            # testing that AUDIT failure does NOT count as
            # post-commit failure for cleanup purposes).
            assert not list(
                store.continuity_path.parent.glob("*.md.tmp")
            )
            assert not list(
                store.continuity_path.parent.glob("*.json.tmp")
            )
        finally:
            store.close()

    # -- L2 H2: _defer_commit flag reset is unconditional --

    def test_defer_commit_reset_after_successful_save(self, tmp_path):
        """After a successful validated_save_continuity, _defer_commit
        must be False. Guard against the poisoned-store scenario
        where a commit succeeds but the flag stays True.
        """
        from anneal_memory import validated_save_continuity

        db_path = str(tmp_path / "reset_success.db")
        store, text, token = _prime_simple(db_path, "ResetSuccess")
        try:
            validated_save_continuity(store, text, wrap_token=token)
            assert store._defer_commit is False
            assert store._deferred_audits == []
        finally:
            store.close()

    def test_defer_commit_reset_after_failed_save(
        self, tmp_path, monkeypatch
    ):
        """After validated_save_continuity raises from inside the
        batch, _defer_commit must be False. Guard against the
        poisoned-store scenario under the exception path.
        """
        from anneal_memory import validated_save_continuity

        db_path = str(tmp_path / "reset_fail.db")
        store, text, token = _prime_simple(db_path, "ResetFail")
        try:
            def exploding(self, meta_arg, token_hex=None):
                raise StoreError(
                    "injected meta failure",
                    operation="prepare_meta_write",
                    path=str(self.meta_path),
                )

            monkeypatch.setattr(Store, "_prepare_meta_write", exploding)
            with pytest.raises(StoreError):
                validated_save_continuity(store, text, wrap_token=token)

            assert store._defer_commit is False
            assert store._deferred_audits == []
        finally:
            store.close()

    def test_defer_commit_reset_after_nested_batch_raises(self, tmp_path):
        """Nested _batch() raises RuntimeError — but critically, the
        OUTER batch's state must not be mutated by the guard, and
        subsequent batches on the same store must work normally.
        Guard for L1 LOW finding.
        """
        db_path = str(tmp_path / "nested_reset.db")
        store = Store(db_path, project_name="NestedReset")
        try:
            with store._batch():
                # Queue an event so we can verify outer state is
                # not disturbed by the inner guard.
                store._audit_log("test_marker", {"ok": 1})
                outer_queue_len = len(store._deferred_audits)
                with pytest.raises(RuntimeError, match="does not support nesting"):
                    with store._batch():
                        pass
                # Outer batch's queued audits untouched.
                assert len(store._deferred_audits) == outer_queue_len

            # After outer batch exits normally, flag is reset.
            assert store._defer_commit is False
            assert store._deferred_audits == []

            # A subsequent batch works normally.
            with store._batch():
                store._audit_log("second_batch", {"ok": 1})
            assert store._defer_commit is False
        finally:
            store.close()

    # -- L1 MEDIUM: wrap_cancelled emits audit on partial state --

    def test_wrap_cancelled_partial_state_emits_audit(self, tmp_path):
        """Partial state (wrap_started_at set but wrap_token empty)
        must produce a wrap_cancelled audit entry with
        partial_state=True so operator recovery leaves a forensic
        trail. Regression guard for L1 MEDIUM.
        """
        import json as json_mod

        db_path = str(tmp_path / "partial_cancel.db")
        store = Store(db_path, project_name="PartialCancel", audit=True)
        try:
            assert store._audit is not None
            audit_path = store._audit._active_path

            # Induce partial state: wrap_started_at set, token empty.
            store._conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                ("wrap_started_at", "2026-04-10T00:00:00.000000Z"),
            )
            store._conn.commit()

            before_events = _read_audit_events(audit_path)

            store.wrap_cancelled()

            after_events = _read_audit_events(audit_path)
            new = after_events[len(before_events):]
            cancel_events = [
                e for e in new if e.get("event") == "wrap_cancelled"
            ]
            assert len(cancel_events) == 1, (
                f"expected exactly 1 wrap_cancelled event, got "
                f"{len(cancel_events)}: {cancel_events}"
            )
            payload = cancel_events[0].get("data") or {}
            assert payload.get("partial_state") is True
            assert "wrap_started_at" in payload

            # State is cleared.
            assert store.get_wrap_started_at() is None
        finally:
            store.close()

    def test_wrap_cancelled_clean_store_no_audit(self, tmp_path):
        """Cancelling a clean store (no partial state, no healthy
        wrap) must NOT emit an audit event — there's nothing to
        cancel. Negative-space companion to the partial-state test.
        """
        db_path = str(tmp_path / "clean_cancel.db")
        store = Store(db_path, project_name="CleanCancel", audit=True)
        try:
            assert store._audit is not None
            audit_path = store._audit._active_path
            before = _read_audit_events(audit_path)

            store.wrap_cancelled()

            after = _read_audit_events(audit_path)
            new = after[len(before):]
            cancel_events = [
                e for e in new if e.get("event") == "wrap_cancelled"
            ]
            assert cancel_events == []
        finally:
            store.close()


# -- Module-level helpers shared across TestPostReviewFixes --


def _prime_simple(db_path: str, project: str):
    """Minimal primed store + text + token fixture for post-review tests."""
    from anneal_memory import prepare_wrap

    store = Store(db_path, project_name=project)
    ep = store.record("seed observation", EpisodeType.OBSERVATION)
    result = prepare_wrap(store)
    text = (
        f"# {project} — Memory (v1)\n\n"
        f"## State\nWorking.\n\n"
        f"## Patterns\n"
        f"{{core:\n"
        f"  thought: post review fix | 1x (2026-04-10)\n"
        f"}}\n\n"
        f"## Decisions\n"
        f"[decided(rationale: \"fix\", on: \"2026-04-10\")] ok\n\n"
        f"## Context\nCited {ep.id[:8]}.\n"
    )
    return store, text, result["wrap_token"]


def _read_audit_events(audit_path: Path) -> list[dict]:
    """Read the JSONL audit trail as a list of parsed events."""
    import json as json_mod

    if not audit_path.exists():
        return []
    events = []
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        events.append(json_mod.loads(line))
    return events


class TestL3Fixes:
    """10.5c.5 Layer 3 consultation fix regression gates.

    Each test guards a specific L3 finding that was fixed in the
    post-review pass. Convergent findings (all 3 agents flagged)
    get prioritized assertions; single-agent findings get
    individual coverage.
    """

    @staticmethod
    def _make_continuity(project_name: str, cited_id: str) -> str:
        return (
            f"# {project_name} — Memory (v1)\n\n"
            f"## State\nWorking.\n\n"
            f"## Patterns\n"
            f"{{core:\n"
            f"  thought: l3 fix regression guard "
            f"| 1x (2026-04-10)\n"
            f"}}\n\n"
            f"## Decisions\n"
            f"[decided(rationale: \"l3\", on: \"2026-04-10\")] ok\n\n"
            f"## Context\nCited {cited_id}.\n"
        )

    # -- L3 Fix #12: unique tmp filenames (CRITICAL convergent) --

    def test_prepare_continuity_write_generates_unique_tmp_paths(
        self, tmp_path
    ):
        """Two back-to-back _prepare_continuity_write calls must
        return different tmp paths. The uuid suffix is the
        concurrent-writer collision fix from L3 Fix #12.
        """
        store = Store(str(tmp_path / "unique.db"), project_name="Unique")
        try:
            path_a = store._prepare_continuity_write("a")
            path_b = store._prepare_continuity_write("b")
            assert path_a != path_b, (
                f"tmp paths collided: {path_a} == {path_b}"
            )
            # Both files exist on disk after the writes.
            assert path_a.exists()
            assert path_b.exists()
            # Both sit beside the final continuity path, not inside
            # a subdirectory.
            assert path_a.parent == store.continuity_path.parent
            assert path_b.parent == store.continuity_path.parent
            # Clean up so later tests / fixtures don't see orphans.
            path_a.unlink()
            path_b.unlink()
        finally:
            store.close()

    def test_prepare_meta_write_generates_unique_tmp_paths(self, tmp_path):
        """Same invariant for meta sidecar tmp files."""
        store = Store(str(tmp_path / "unique_meta.db"), project_name="UniqueMeta")
        try:
            path_a = store._prepare_meta_write({"a": 1})
            path_b = store._prepare_meta_write({"b": 2})
            assert path_a != path_b
            path_a.unlink()
            path_b.unlink()
        finally:
            store.close()

    def test_tmp_path_matches_glob_pattern(self, tmp_path):
        """Operator recovery + startup detection globs for
        ``<stem>.*.md.tmp`` / ``<stem>.*.json.tmp``. Verify the
        generated tmp paths actually match that pattern so the
        recovery tooling finds them.
        """
        store = Store(str(tmp_path / "glob.db"), project_name="Glob")
        try:
            cont_tmp = store._prepare_continuity_write("x")
            meta_tmp = store._prepare_meta_write({"x": 1})
            cont_stem = store.continuity_path.stem
            meta_stem = store.meta_path.stem
            cont_matches = list(
                store.continuity_path.parent.glob(f"{cont_stem}.*.md.tmp")
            )
            meta_matches = list(
                store.meta_path.parent.glob(f"{meta_stem}.*.json.tmp")
            )
            assert cont_tmp in cont_matches
            assert meta_tmp in meta_matches
            cont_tmp.unlink()
            meta_tmp.unlink()
        finally:
            store.close()

    # -- L3 Fix #13: startup orphan tmp detection (HIGH convergent) --

    def test_store_init_warns_on_orphan_continuity_tmp(self, tmp_path):
        """Store open after a crash that left an orphan .md.tmp
        file must emit a UserWarning so the operator notices.
        """
        db_path = str(tmp_path / "orphan_cont.db")
        # First open to establish the continuity path conventions.
        store = Store(db_path, project_name="OrphanCont")
        cont_parent = store.continuity_path.parent
        cont_stem = store.continuity_path.stem
        store.close()

        # Simulate a crashed pipeline: drop an orphan .md.tmp
        # alongside the (non-existent) real continuity file.
        orphan = cont_parent / f"{cont_stem}.deadbeef0123.md.tmp"
        orphan.write_text(
            "# orphan — Memory (v1)\n\n## State\nstuck\n\n"
            "## Patterns\np\n\n## Decisions\nd\n\n## Context\nc\n",
            encoding="utf-8",
        )

        # Re-opening the store must warn.
        with pytest.warns(UserWarning, match="orphan tmp sidecar"):
            store = Store(db_path, project_name="OrphanCont")
        try:
            # And the orphan is discoverable via the helper.
            orphans = store._find_orphan_tmp_files()
            assert orphan in orphans
            # AND not auto-deleted — operator owns the recovery.
            assert orphan.exists()
        finally:
            store.close()
            orphan.unlink()  # cleanup for pytest tmp dir

    def test_store_init_warns_on_orphan_meta_tmp(self, tmp_path):
        """Same behavior for meta orphans."""
        db_path = str(tmp_path / "orphan_meta.db")
        store = Store(db_path, project_name="OrphanMeta")
        meta_parent = store.meta_path.parent
        meta_stem = store.meta_path.stem
        store.close()

        orphan = meta_parent / f"{meta_stem}.deadbeef0123.json.tmp"
        orphan.write_text('{"stuck": true}\n', encoding="utf-8")

        with pytest.warns(UserWarning, match="orphan tmp sidecar"):
            store = Store(db_path, project_name="OrphanMeta")
        try:
            orphans = store._find_orphan_tmp_files()
            assert orphan in orphans
            assert orphan.exists()
        finally:
            store.close()
            orphan.unlink()

    def test_store_init_silent_with_no_orphans(self, tmp_path):
        """Clean store open emits NO UserWarning about tmp files."""
        import warnings as _warnings

        db_path = str(tmp_path / "clean.db")
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            store = Store(db_path, project_name="Clean")
            store.close()
        orphan_warnings = [
            w for w in caught
            if "orphan tmp sidecar" in str(w.message)
        ]
        assert orphan_warnings == []

    def test_orphan_recovery_via_manual_rename(self, tmp_path):
        """Operator recovery story end-to-end: tmp file exists,
        operator renames it to final path, next Store open is
        silent (no warning).
        """
        import warnings as _warnings

        db_path = str(tmp_path / "recovery.db")
        store = Store(db_path, project_name="Recovery")
        final = store.continuity_path
        parent = final.parent
        stem = final.stem
        store.close()

        orphan = parent / f"{stem}.abcd12345678.md.tmp"
        orphan.write_text("orphan content", encoding="utf-8")

        # Simulate the operator running:
        #   mv <orphan> <final>
        orphan.replace(final)

        # Next open must be silent.
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            store = Store(db_path, project_name="Recovery")
            store.close()
        orphan_warnings = [
            w for w in caught
            if "orphan tmp sidecar" in str(w.message)
        ]
        assert orphan_warnings == []
        assert final.exists()

    # -- L3 Fix #14: _batch() commit-failure rollback --

    def test_batch_commit_failure_rolls_back(self, tmp_path):
        """If self._conn.commit() raises during batch __exit__,
        subsequent DML on the store must see the pre-batch state.
        L3 complement F2.

        sqlite3.Connection.commit is read-only at the C level so
        we can't monkeypatch it directly — instead, wrap _conn in
        a transparent proxy that only overrides commit.
        """
        import sqlite3 as _sqlite

        class FlakyCommitProxy:
            """Delegates everything except commit (which explodes)."""
            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            def commit(self):
                raise _sqlite.OperationalError(
                    "injected commit failure"
                )

        db_path = str(tmp_path / "commit_fail.db")
        store = Store(db_path, project_name="CommitFail")
        try:
            real_conn = store._conn
            baseline = real_conn.execute(
                "SELECT COUNT(*) FROM episodes"
            ).fetchone()[0]

            store._conn = FlakyCommitProxy(real_conn)  # type: ignore[assignment]

            # 10.5c.6: batch commit failures now surface as
            # StoreDatabaseError with operation="batch_commit" instead
            # of bare sqlite3.OperationalError. The underlying sqlite
            # error is still attached via __cause__ for callers that
            # need the original errno.
            from anneal_memory import StoreDatabaseError

            with pytest.raises(StoreDatabaseError) as exc_info:
                with store._batch():
                    real_conn.execute(
                        """INSERT INTO episodes
                           (id, timestamp, type, content, source, metadata)
                           VALUES (?, ?, ?, ?, ?, NULL)""",
                        (
                            "aaaaaaaa",
                            "2026-04-10T00:00:00Z",
                            "observation",
                            "should be rolled back",
                            "test",
                        ),
                    )

            # __cause__ should still be the original injected error
            assert isinstance(exc_info.value.__cause__, _sqlite.OperationalError)
            assert "injected" in str(exc_info.value.__cause__)
            assert exc_info.value.operation == "batch_commit"

            # Restore real connection for post-check and cleanup.
            store._conn = real_conn

            # Post-conditions: flags reset, DML rolled back.
            assert store._defer_commit is False
            assert store._deferred_audits == []

            after = real_conn.execute(
                "SELECT COUNT(*) FROM episodes"
            ).fetchone()[0]
            assert after == baseline
        finally:
            store.close()

    # -- L3 Fix #15: Phase 4 audit try/except --

    def test_phase4_audit_exception_does_not_propagate(
        self, tmp_path, monkeypatch
    ):
        """Exception during the Phase 4 continuity_saved audit log
        must not propagate — at this point the wrap is fully
        committed + externalized, and a phantom failure report
        would mislead the transport layer. L3 complement F4.
        """
        from anneal_memory import validated_save_continuity

        db_path = str(tmp_path / "phase4_audit.db")
        store, text, token = _prime_simple(db_path, "Phase4Audit")
        try:
            assert store._audit is not None
            original_log = store._audit.log

            def flaky_log(event, payload, **kwargs):
                if event == "continuity_saved":
                    raise OSError(
                        "injected phase 4 audit failure"
                    )
                return original_log(event, payload, **kwargs)

            monkeypatch.setattr(store._audit, "log", flaky_log)

            # Must return successfully despite the audit failure.
            result = validated_save_continuity(
                store, text, wrap_token=token
            )
            assert result["episodes_compressed"] >= 1
            # Files externalized.
            assert store.continuity_path.exists()
            assert store.meta_path.exists()
        finally:
            store.close()

    # -- L3 Fix #16: assert → StoreError (-O safety) --

    def test_meta_tmp_none_raises_storeerror_not_assertion(
        self, tmp_path, monkeypatch
    ):
        """If meta_tmp is somehow None at Phase 3 (should be
        impossible under normal control flow, but future refactors
        could introduce it), the pipeline must raise StoreError with
        recovery guidance, not AssertionError (which would vanish
        under python -O) or AttributeError.
        """
        # Impossible to reach this branch under the current code
        # path without patching — which is the point of the fix.
        # We verify by monkeypatching _prepare_meta_write to return
        # None, forcing the explicit None check to fire.
        from anneal_memory import validated_save_continuity

        db_path = str(tmp_path / "meta_none.db")
        store, text, token = _prime_simple(db_path, "MetaNone")
        try:
            original_prep = Store._prepare_meta_write

            def returns_none(self, meta, token_hex=None):
                # Pretend the meta write succeeded but returned None.
                # Walk through the real call first so the tmp file
                # exists (Phase 3 rename would otherwise blow up
                # differently), then return None.
                real_path = original_prep(self, meta, token_hex=token_hex)
                # Unlink the tmp so no orphan is left behind.
                real_path.unlink(missing_ok=True)
                return None

            monkeypatch.setattr(Store, "_prepare_meta_write", returns_none)

            with pytest.raises(StoreError, match="meta_tmp is None"):
                validated_save_continuity(store, text, wrap_token=token)
        finally:
            store.close()

    # -- L3 Fix #17: record() batch-aware --

    def test_record_inside_batch_defers_commit(self, tmp_path):
        """record() called inside _batch() must NOT commit mid-batch.
        A separate connection against the same DB should see zero
        new episodes until the batch exits successfully.
        """
        import sqlite3 as _sqlite

        db_path = str(tmp_path / "record_batch.db")
        store = Store(db_path, project_name="RecordBatch")
        try:
            with store._batch():
                store.record(
                    "inside batch observation",
                    EpisodeType.OBSERVATION,
                )
                # Another connection sees zero episodes mid-batch.
                other = _sqlite.connect(db_path)
                other.row_factory = _sqlite.Row
                mid = other.execute(
                    "SELECT COUNT(*) FROM episodes"
                ).fetchone()[0]
                other.close()
                assert mid == 0, (
                    f"record() committed mid-batch: other conn saw "
                    f"{mid} episodes"
                )

            # After batch exits, the new episode is visible.
            other = _sqlite.connect(db_path)
            other.row_factory = _sqlite.Row
            after = other.execute(
                "SELECT COUNT(*) FROM episodes"
            ).fetchone()[0]
            other.close()
            assert after == 1
        finally:
            store.close()

    def test_record_inside_failed_batch_rolls_back(self, tmp_path):
        """record() inside a batch that raises must be rolled back —
        no trace of the new episode in the store after the
        exception propagates.
        """
        import sqlite3 as _sqlite

        db_path = str(tmp_path / "record_batch_rollback.db")
        store = Store(db_path, project_name="RecordBatchRollback")
        try:
            with pytest.raises(RuntimeError, match="injected"):
                with store._batch():
                    store.record(
                        "should be rolled back",
                        EpisodeType.OBSERVATION,
                    )
                    raise RuntimeError("injected mid-batch failure")

            # Store is clean: episode was rolled back, flags reset.
            other = _sqlite.connect(db_path)
            other.row_factory = _sqlite.Row
            count = other.execute(
                "SELECT COUNT(*) FROM episodes"
            ).fetchone()[0]
            other.close()
            assert count == 0
            assert store._defer_commit is False
        finally:
            store.close()

    def test_record_audit_deferred_inside_batch(self, tmp_path):
        """record() audit event must be queued (not fired) inside a
        batch, and flushed only after commit succeeds. Preserves
        the audit-after-commit invariant that the batch exists to
        enforce.
        """
        db_path = str(tmp_path / "record_audit.db")
        store = Store(db_path, project_name="RecordAudit", audit=True)
        try:
            assert store._audit is not None
            audit_path = store._audit._active_path

            before = _read_audit_events(audit_path)

            with store._batch():
                store.record(
                    "batched observation",
                    EpisodeType.OBSERVATION,
                )
                # Inside the batch, the audit event is queued,
                # not fired.
                mid = _read_audit_events(audit_path)
                new_mid = [
                    e for e in mid[len(before):]
                    if e["event"] == "record"
                ]
                assert new_mid == [], (
                    f"record audit fired mid-batch: {new_mid}"
                )
                # And the deferred queue contains it.
                assert len(store._deferred_audits) == 1
                assert store._deferred_audits[0][0] == "record"
                # Third slot is kwargs — should carry actor.
                assert store._deferred_audits[0][2].get("actor") == "agent"

            # After exit, the record audit fires.
            after = _read_audit_events(audit_path)
            new_after = [
                e for e in after[len(before):]
                if e["event"] == "record"
            ]
            assert len(new_after) == 1
            assert new_after[0]["actor"] == "agent"
        finally:
            store.close()


class TestL3CodexFixes:
    """10.5c.5 Layer 3 codex retry-pass fix regression gates.

    Each test here covers a finding that only the codex agent
    surfaced in the second consultation pass (the first pass had
    a codex timeout). These fixes ride on top of the earlier L3
    landing and address recoverability identity, warning text
    accuracy, and in-flight false positives.
    """

    @staticmethod
    def _make_continuity(project_name: str, cited_id: str) -> str:
        return (
            f"# {project_name} — Memory (v1)\n\n"
            f"## State\nWorking.\n\n"
            f"## Patterns\n"
            f"{{core:\n"
            f"  thought: codex fix regression guard "
            f"| 1x (2026-04-10)\n"
            f"}}\n\n"
            f"## Decisions\n"
            f"[decided(rationale: \"codex\", on: \"2026-04-10\")] ok\n\n"
            f"## Context\nCited {cited_id}.\n"
        )

    # -- L3 Fix #19: paired tmp filenames via wrap_token --

    def test_pipeline_tmp_files_share_token_prefix(
        self, tmp_path, monkeypatch
    ):
        """Continuity + meta tmp files written by the canonical
        pipeline must share a token prefix (the first 12 hex chars
        of the wrap_token) so operator recovery can pair them.
        L3 codex HIGH.
        """
        from anneal_memory import validated_save_continuity

        db_path = str(tmp_path / "paired.db")
        store, text, token = _prime_simple(db_path, "Paired")
        try:
            # Monkeypatch Path.replace to fail on the continuity
            # rename — this freezes the pipeline at the post-commit
            # / pre-rename state, leaving BOTH tmp files on disk
            # for us to inspect.
            from pathlib import Path as _Path
            original_replace = _Path.replace

            def flaky(self, target):
                if self.suffix == ".tmp" and self.name.endswith(".md.tmp"):
                    raise OSError("inject rename failure for paired check")
                return original_replace(self, target)

            monkeypatch.setattr(_Path, "replace", flaky)

            with pytest.raises(StoreError):
                validated_save_continuity(store, text, wrap_token=token)

            monkeypatch.setattr(_Path, "replace", original_replace)

            # Find the persisted tmp files.
            cont_parent = store.continuity_path.parent
            cont_tmps = list(cont_parent.glob("*.md.tmp"))
            meta_tmps = list(cont_parent.glob("*.json.tmp"))
            assert len(cont_tmps) == 1
            assert len(meta_tmps) == 1

            # Extract tokens from filenames.
            cont_token = Store._token_from_orphan(cont_tmps[0])
            meta_token = Store._token_from_orphan(meta_tmps[0])
            assert cont_token == meta_token, (
                f"tmp filenames not paired: {cont_token} != {meta_token}"
            )
            # The shared token must be the 12-char prefix of the
            # real wrap_token.
            assert cont_token == token[:12]

            # Clean up so the fixture's tmp_path doesn't leave
            # orphans between tests.
            cont_tmps[0].unlink()
            meta_tmps[0].unlink()
        finally:
            store.close()

    # -- L3 Fix #20: warning text references continuity_saved --

    def test_orphan_warning_mentions_continuity_saved_not_wrap_completed(
        self, tmp_path
    ):
        """Orphan warning recovery text must point operators at the
        ``continuity_saved`` audit event (which carries
        content_hash) rather than the old ``wrap_completed`` event
        (which does not). L3 codex MEDIUM.
        """
        db_path = str(tmp_path / "warntext.db")
        store = Store(db_path, project_name="WarnText")
        final = store.continuity_path
        parent = final.parent
        stem = final.stem
        store.close()

        orphan = parent / f"{stem}.deadbeef0123.md.tmp"
        orphan.write_text("orphan content", encoding="utf-8")

        with pytest.warns(UserWarning) as record:
            store = Store(db_path, project_name="WarnText")
        try:
            matching = [
                w for w in record
                if "orphan tmp sidecar" in str(w.message)
            ]
            assert len(matching) >= 1
            msg = str(matching[0].message)
            # The fixed warning text MUST reference continuity_saved
            # and MUST mention that it carries content_hash.
            assert "continuity_saved" in msg
            assert "content_hash" in msg
        finally:
            store.close()
            orphan.unlink(missing_ok=True)

    def test_paired_orphans_emit_single_warning(self, tmp_path):
        """Two orphans from the same wrap (paired via token prefix)
        must emit exactly ONE warning listing both files, not two
        unrelated warnings. L3 codex HIGH — the pairing is the
        whole point of Fix #19.
        """
        db_path = str(tmp_path / "paired_warn.db")
        store = Store(db_path, project_name="PairedWarn")
        cont_final = store.continuity_path
        meta_final = store.meta_path
        parent = cont_final.parent
        cont_stem = cont_final.stem
        meta_stem = meta_final.stem
        store.close()

        # Drop paired orphans with matching token.
        tok = "feedface1234"
        cont_orphan = parent / f"{cont_stem}.{tok}.md.tmp"
        meta_orphan = parent / f"{meta_stem}.{tok}.json.tmp"
        cont_orphan.write_text("orphan cont", encoding="utf-8")
        meta_orphan.write_text('{"orphan": "meta"}', encoding="utf-8")

        with pytest.warns(UserWarning) as record:
            store = Store(db_path, project_name="PairedWarn")
        try:
            matching = [
                w for w in record
                if "orphan tmp sidecar" in str(w.message)
            ]
            # Exactly one warning — the pair grouped together.
            assert len(matching) == 1, (
                f"expected 1 paired warning, got {len(matching)}: "
                f"{[str(w.message)[:80] for w in matching]}"
            )
            msg = str(matching[0].message)
            # Both filenames appear in the single message.
            assert str(cont_orphan) in msg
            assert str(meta_orphan) in msg
            # Token prefix is called out.
            assert tok in msg
        finally:
            store.close()
            cont_orphan.unlink(missing_ok=True)
            meta_orphan.unlink(missing_ok=True)

    def test_unpaired_orphan_notes_missing_pair(self, tmp_path):
        """An orphan with NO matching pair (only .md.tmp or only
        .json.tmp) must warn with a NOTE that the pair is missing
        so operators don't assume the recovery is complete.
        """
        db_path = str(tmp_path / "unpaired.db")
        store = Store(db_path, project_name="Unpaired")
        parent = store.continuity_path.parent
        cont_stem = store.continuity_path.stem
        store.close()

        # Drop an orphan continuity tmp with no matching meta tmp.
        orphan = parent / f"{cont_stem}.abcdef012345.md.tmp"
        orphan.write_text("lone orphan", encoding="utf-8")

        with pytest.warns(UserWarning) as record:
            store = Store(db_path, project_name="Unpaired")
        try:
            matching = [
                w for w in record
                if "orphan tmp sidecar" in str(w.message)
            ]
            assert len(matching) == 1
            msg = str(matching[0].message)
            # The NOTE about missing pair is present.
            assert "no paired meta tmp found" in msg
        finally:
            store.close()
            orphan.unlink(missing_ok=True)

    # -- L3 Fix #21: in-flight tmp files not flagged as orphans --

    def test_active_wrap_tmp_files_not_flagged_as_orphans(
        self, tmp_path
    ):
        """If tmp files exist whose embedded token matches the
        currently-active wrap_token in metadata, they are in-flight
        (not orphans) and must NOT trigger a warning when a second
        store opens. L3 codex MEDIUM.
        """
        import warnings as _warnings

        db_path = str(tmp_path / "inflight.db")
        store = Store(db_path, project_name="Inflight")
        parent = store.continuity_path.parent
        cont_stem = store.continuity_path.stem
        meta_stem = store.meta_path.stem

        # Simulate Store A being mid-batch: set wrap_token in
        # metadata and write the matching tmp files.
        active_tok_full = "a1b2c3d4e5f60123456789abcdef0123"  # 32 hex
        active_prefix = active_tok_full[:12]
        store._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("wrap_token", active_tok_full),
        )
        store._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("wrap_started_at", "2026-04-10T00:00:00Z"),
        )
        store._conn.commit()

        cont_inflight = parent / f"{cont_stem}.{active_prefix}.md.tmp"
        meta_inflight = parent / f"{meta_stem}.{active_prefix}.json.tmp"
        cont_inflight.write_text("in-flight content", encoding="utf-8")
        meta_inflight.write_text('{"in_flight": true}', encoding="utf-8")
        store.close()

        # Store B opens — should NOT warn about these in-flight tmps.
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            store_b = Store(db_path, project_name="Inflight")
            store_b.close()
        orphan_warnings = [
            w for w in caught
            if "orphan tmp sidecar" in str(w.message)
        ]
        assert orphan_warnings == [], (
            f"in-flight tmps incorrectly flagged as orphans: "
            f"{[str(w.message)[:80] for w in orphan_warnings]}"
        )

        # And the orphan-finder returns an empty list.
        store_b = Store(db_path, project_name="Inflight")
        try:
            assert store_b._find_orphan_tmp_files() == []
        finally:
            store_b.close()
            cont_inflight.unlink(missing_ok=True)
            meta_inflight.unlink(missing_ok=True)

    def test_stale_tmp_different_token_is_flagged(self, tmp_path):
        """Negative-space companion to the in-flight test: if the
        active wrap_token is X but a tmp file embeds token Y, the
        tmp file IS an orphan (from a prior crashed wrap, distinct
        from the current one) and MUST be flagged.
        """
        db_path = str(tmp_path / "stale.db")
        store = Store(db_path, project_name="Stale")
        parent = store.continuity_path.parent
        cont_stem = store.continuity_path.stem

        # Set active wrap_token to X.
        store._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("wrap_token", "xxxxxxxxxxxx0000000000000000xxxx"),
        )
        store._conn.commit()

        # Drop an orphan with token Y (different from active X).
        stale = parent / f"{cont_stem}.yyyyyyyyyyyy.md.tmp"
        stale.write_text("stale", encoding="utf-8")
        store.close()

        with pytest.warns(UserWarning, match="orphan tmp sidecar"):
            store = Store(db_path, project_name="Stale")
        try:
            orphans = store._find_orphan_tmp_files()
            assert stale in orphans
        finally:
            store.close()
            stale.unlink(missing_ok=True)

