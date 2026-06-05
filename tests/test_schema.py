"""Tests for the v0.3.4 pluggable continuity section schema.

Covers schema.py (roles, defaults, validation, helpers), Store-level schema
persistence, the schema-aware validate_structure / _build_wrap_instructions /
graduation gate, and an end-to-end FLOW_SCHEMA wrap cycle. The load-bearing
invariant throughout: DEFAULT_SCHEMA reproduces the exact pre-0.3.4 behavior.
"""

from __future__ import annotations

import json
import sqlite3
from typing import get_args

import pytest

from anneal_memory import schema as S
from anneal_memory.schema import (
    DEFAULT_GRADUATING,
    DEFAULT_SCHEMA,
    FLOW_SCHEMA,
    SCHEMA_NAMES,
    SectionRole,
    _SCHEMAS_BY_NAME,
    default_max_chars,
    graduating_headings,
    heading_marker,
    name_for_schema,
    required_headings,
    schema_by_name,
    schema_role_warning,
    sections_by_role,
    validate_schema,
)
from anneal_memory.store import Store, StoreError
from anneal_memory.continuity import (
    _build_wrap_instructions,
    prepare_wrap,
    validate_structure,
    validated_save_continuity,
)
from anneal_memory.graduation import extract_pattern_names
from anneal_memory.types import EpisodeType


# -- named schemas (AM-INITSCHEMA) --


class TestNamedSchemas:
    def test_registry_maps_names_to_constants(self):
        # The registry is module-private (LOW-2): selection is via schema_by_name.
        assert SCHEMA_NAMES == ("default", "partnership")
        assert _SCHEMAS_BY_NAME["default"] is DEFAULT_SCHEMA
        assert _SCHEMAS_BY_NAME["partnership"] is FLOW_SCHEMA

    def test_registry_not_in_public_api(self):
        import anneal_memory
        assert not hasattr(anneal_memory, "SCHEMAS_BY_NAME")

    def test_schema_by_name_resolves(self):
        assert [s["heading"] for s in schema_by_name("default")] == \
            [s["heading"] for s in DEFAULT_SCHEMA]
        assert [s["heading"] for s in schema_by_name("partnership")] == \
            [s["heading"] for s in FLOW_SCHEMA]

    def test_schema_by_name_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown schema name"):
            schema_by_name("nope")

    def test_schema_by_name_returns_independent_copy(self):
        # Public adapter entry point: a caller mutating the result (list OR the
        # section dicts) must not corrupt the shared module constants.
        got = schema_by_name("partnership")
        got.append({"heading": "Injected", "role": "narrative"})
        got[0]["heading"] = "MUTATED"
        assert len(FLOW_SCHEMA) == 6
        assert FLOW_SCHEMA[0]["heading"] == "State"

    def test_name_for_schema_roundtrip(self):
        assert name_for_schema(DEFAULT_SCHEMA) == "default"
        assert name_for_schema(FLOW_SCHEMA) == "partnership"
        assert name_for_schema(schema_by_name("partnership")) == "partnership"

    def test_name_for_schema_custom_returns_none(self):
        custom = [{"heading": "Notes", "role": "graduating"}]
        assert name_for_schema(custom) is None

    def test_set_section_schema_is_audited(self, tmp_path):
        # A schema migration changes which sections are required + whether the
        # felt-gate fires — it belongs in the audit chain (anneal audit-everything).
        db = tmp_path / "m.db"
        store = Store(db)  # audit on by default
        store.set_section_schema(FLOW_SCHEMA)
        store.close()
        lines = (tmp_path / "m.audit.jsonl").read_text(encoding="utf-8").strip().split("\n")
        entries = [json.loads(l) for l in lines]
        schema_events = [e for e in entries if e["event"] == "section_schema_set"]
        assert len(schema_events) == 1
        payload = schema_events[0]["data"]
        assert payload["from"] == [s["heading"] for s in DEFAULT_SCHEMA]
        assert payload["to"] == [s["heading"] for s in FLOW_SCHEMA]

    def test_audit_failure_is_best_effort(self, tmp_path):
        # MEDIUM-1 (codex L3): the schema write is durable before the audit
        # append; an audit OSError must warn, not raise, and must NOT undo the
        # persisted schema.
        db = tmp_path / "m.db"
        store = Store(db)

        def boom(*a, **k):
            raise OSError("audit disk full")

        store._audit.log = boom  # type: ignore[union-attr]
        with pytest.warns(UserWarning, match="audit log append failed"):
            store.set_section_schema(FLOW_SCHEMA)
        # The schema is persisted despite the audit failure.
        assert [s["heading"] for s in store.section_schema] == \
            [s["heading"] for s in FLOW_SCHEMA]
        store.close()


# -- schema module --


class TestSchemaSnapshot:
    """AM-SCHEMASNAPSHOT: the section schema is frozen for a wrap's duration so
    prepare and save agree even if the live schema changes mid-wrap."""

    def _raw_set_live_schema(self, store, schema):
        # Change the LIVE persisted schema directly, bypassing set_section_schema
        # (whose guard refuses mid-wrap) — simulates the concurrent/cross-process
        # writer that the freeze defends against.
        store._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("section_schema", json.dumps(schema)),
        )
        store._conn.commit()

    def test_frozen_schema_used_during_wrap(self, tmp_path):
        store = Store(tmp_path / "s.db", section_schema=FLOW_SCHEMA)
        store.wrap_started(token="a" * 32, episode_ids=[], section_schema=FLOW_SCHEMA)
        # A concurrent writer flips the LIVE schema to ops mid-wrap:
        self._raw_set_live_schema(store, DEFAULT_SCHEMA)
        # section_schema_for_wrap still returns the FROZEN partnership schema...
        wrap_headings = [s["heading"] for s in store.section_schema_for_wrap()]
        assert "Understanding" in wrap_headings
        # ...while the plain live read reflects the change (proves they diverged
        # and the freeze is what protected the wrap).
        live_headings = [s["heading"] for s in store.section_schema]
        assert "Understanding" not in live_headings
        store.close()

    def test_frozen_schema_cleared_on_complete(self, tmp_path):
        store = Store(tmp_path / "s.db", section_schema=FLOW_SCHEMA)
        ep = store.record("e", "observation")
        store.wrap_started(token="b" * 32, episode_ids=[ep.id], section_schema=FLOW_SCHEMA)
        store.wrap_completed(episodes_compressed=1, continuity_chars=10,
                             wrap_token="b" * 32, episode_ids=[ep.id])
        # Wrap done -> no freeze; a live change is now reflected by section_schema_for_wrap.
        self._raw_set_live_schema(store, DEFAULT_SCHEMA)
        assert [s["heading"] for s in store.section_schema_for_wrap()] == \
            [s["heading"] for s in DEFAULT_SCHEMA]
        store.close()

    def test_frozen_schema_cleared_on_cancel(self, tmp_path):
        store = Store(tmp_path / "s.db", section_schema=FLOW_SCHEMA)
        store.wrap_started(token="c" * 32, episode_ids=[], section_schema=FLOW_SCHEMA)
        store.wrap_cancelled()
        self._raw_set_live_schema(store, DEFAULT_SCHEMA)
        assert [s["heading"] for s in store.section_schema_for_wrap()] == \
            [s["heading"] for s in DEFAULT_SCHEMA]
        store.close()

    def test_no_frozen_schema_fails_closed(self, tmp_path):
        # A wrap started by a pre-AM-SCHEMASNAPSHOT version (wrap_started_at set,
        # wrap_section_schema empty) FAILS CLOSED rather than silently reverting
        # to the live schema (codex L3 LOW-1) — an invariant, not a discipline.
        store = Store(tmp_path / "s.db", section_schema=FLOW_SCHEMA)
        store.wrap_started(token="d" * 32, episode_ids=[], section_schema=FLOW_SCHEMA)
        store._conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('wrap_section_schema', '')"
        )
        store._conn.commit()
        with pytest.raises(StoreError, match="predates AM-SCHEMASNAPSHOT"):
            store.section_schema_for_wrap()
        store.close()

    def test_allow_restart_overwrites_frozen_schema(self, tmp_path):
        # B (L1): a restart over an in-flight wrap must OVERWRITE the prior
        # freeze, not leave the old wrap's schema stranded.
        store = Store(tmp_path / "s.db", section_schema=FLOW_SCHEMA)
        store.wrap_started(token="a" * 32, episode_ids=[], section_schema=FLOW_SCHEMA)
        store.wrap_started(token="b" * 32, episode_ids=[],
                           section_schema=DEFAULT_SCHEMA, allow_restart=True)
        assert [s["heading"] for s in store.section_schema_for_wrap()] == \
            [s["heading"] for s in DEFAULT_SCHEMA]
        store.close()

    def test_save_enforces_frozen_schema_after_live_flip(self, tmp_path):
        # C (L1): the load-bearing end-to-end claim — validated_save_continuity
        # validates against the FROZEN schema, not the live one. Freeze
        # partnership, flip live to ops mid-wrap, then a save with ops-only
        # content (missing Active Threads + Understanding) must be REJECTED
        # because the wrap is still held to the frozen partnership schema.
        from anneal_memory.continuity import validated_save_continuity
        store = Store(tmp_path / "s.db", section_schema=FLOW_SCHEMA)
        store.record("a session episode", "observation")
        prepare_wrap(store)  # freezes partnership
        self._raw_set_live_schema(store, DEFAULT_SCHEMA)  # concurrent flip to ops
        ops_only = "# C\n## State\n.\n## Patterns\n.\n## Decisions\n.\n## Context\n.\n"
        with pytest.raises(ValueError, match="(?i)section|structure|required|understanding"):
            validated_save_continuity(store, ops_only)
        store.close()

    def test_corrupt_frozen_schema_fails_closed(self, tmp_path):
        store = Store(tmp_path / "s.db", section_schema=FLOW_SCHEMA)
        store.wrap_started(token="e" * 32, episode_ids=[], section_schema=FLOW_SCHEMA)
        store._conn.execute(
            "UPDATE metadata SET value='{not valid json' WHERE key='wrap_section_schema'"
        )
        store._conn.commit()
        with pytest.raises(StoreError, match="Frozen wrap section_schema"):
            store.section_schema_for_wrap()
        store.close()

    def test_prepare_wrap_freezes_the_read_schema(self, tmp_path):
        # End-to-end: prepare_wrap freezes the partnership schema it read, and a
        # concurrent live flip does not change what the wrap sees.
        store = Store(tmp_path / "s.db", section_schema=FLOW_SCHEMA)
        store.record("a session episode", "observation")
        result = prepare_wrap(store)
        assert result["status"] == "ready"
        self._raw_set_live_schema(store, DEFAULT_SCHEMA)
        assert "Understanding" in [s["heading"] for s in store.section_schema_for_wrap()]
        store.close()


class TestSchemaModule:
    def test_section_role_literal_matches_valid_roles(self):
        # The runtime _VALID_ROLES set must stay in sync with the Literal.
        assert set(get_args(SectionRole)) == S._VALID_ROLES

    def test_default_schema_is_the_historical_four_sections(self):
        assert [s["heading"] for s in DEFAULT_SCHEMA] == [
            "State",
            "Patterns",
            "Decisions",
            "Context",
        ]

    def test_default_graduating_is_just_patterns(self):
        assert DEFAULT_GRADUATING == frozenset({"## patterns"})
        assert graduating_headings(DEFAULT_SCHEMA) == frozenset({"## patterns"})

    def test_flow_schema_graduating_is_patterns(self):
        # flow's only graduating section is Patterns -> identical gate to default.
        assert graduating_headings(FLOW_SCHEMA) == frozenset({"## patterns"})

    def test_flow_schema_required_headings(self):
        assert required_headings(FLOW_SCHEMA) == [
            "State",
            "Active Threads",
            "Patterns",
            "Decisions",
            "Context",
            "Understanding",
        ]

    def test_heading_marker_normalizes(self):
        assert heading_marker("Active Threads") == "## active threads"
        assert heading_marker("  Patterns ") == "## patterns"

    def test_sections_by_role(self):
        nt = sections_by_role(FLOW_SCHEMA, "narrative-timeless")
        assert [s["heading"] for s in nt] == ["Understanding"]
        ls = sections_by_role(FLOW_SCHEMA, "live-state")
        assert [s["heading"] for s in ls] == ["State", "Active Threads"]

    def test_validate_schema_accepts_builtins_and_roundtrips(self):
        assert validate_schema(DEFAULT_SCHEMA) == DEFAULT_SCHEMA
        assert validate_schema(FLOW_SCHEMA) == FLOW_SCHEMA

    def test_validate_schema_strips_whitespace(self):
        norm = validate_schema([{"heading": "  Spaced  ", "role": "graduating"}])
        assert norm[0]["heading"] == "Spaced"

    @pytest.mark.parametrize(
        "bad",
        [
            [],
            "notalist",
            None,
            [{"heading": "State"}],  # missing role
            [{"role": "graduating"}],  # missing heading
            [{"heading": "", "role": "graduating"}],  # empty heading
            [{"heading": "X", "role": "bogus"}],  # unknown role
            [{"heading": "State", "role": "live-state"}],  # no graduating
            [  # duplicate heading (case-insensitive)
                {"heading": "A", "role": "graduating"},
                {"heading": "a", "role": "live-state"},
            ],
        ],
    )
    def test_validate_schema_rejects(self, bad):
        with pytest.raises(ValueError):
            validate_schema(bad)

    def test_validate_schema_rejects_ambiguous_substring_headings(self):
        # One heading word-bounded inside another would let one header line
        # satisfy both required sections in validate_structure (codex M2).
        with pytest.raises(ValueError, match="word-bounded substring"):
            validate_schema(
                [
                    {"heading": "State", "role": "live-state"},
                    {"heading": "State Machine", "role": "graduating"},
                ]
            )
        with pytest.raises(ValueError, match="word-bounded substring"):
            validate_schema(
                [
                    {"heading": "Patterns", "role": "graduating"},
                    {"heading": "Anti-Patterns", "role": "frozen"},
                ]
            )


# -- Store-level persistence --


class TestStoreSectionSchema:
    def test_fresh_store_defaults_to_default_schema(self, tmp_path):
        s = Store(str(tmp_path / "m.db"), project_name="ops")
        assert s.section_schema == DEFAULT_SCHEMA
        s.close()

    def test_explicit_schema_persists_and_survives_reconstruction(self, tmp_path):
        p = str(tmp_path / "flow.db")
        s = Store(p, project_name="flow", section_schema=FLOW_SCHEMA)
        assert s.section_schema == FLOW_SCHEMA
        s.close()
        # reconstruct WITHOUT the arg — persisted schema is authoritative.
        s2 = Store(p)
        assert s2.section_schema == FLOW_SCHEMA
        s2.close()

    def test_set_section_schema_overwrites_live(self, tmp_path):
        p = str(tmp_path / "m.db")
        s = Store(p, section_schema=FLOW_SCHEMA)
        norm = s.set_section_schema(DEFAULT_SCHEMA)
        assert norm == DEFAULT_SCHEMA
        s.close()
        assert Store(p).section_schema == DEFAULT_SCHEMA

    def test_legacy_store_without_schema_key_falls_back(self, tmp_path):
        p = str(tmp_path / "m.db")
        s = Store(p, section_schema=FLOW_SCHEMA)
        s.close()
        # Simulate a pre-0.3.4 store: drop the metadata key entirely.
        conn = sqlite3.connect(p)
        conn.execute("DELETE FROM metadata WHERE key='section_schema'")
        conn.commit()
        conn.close()
        # Reading directly via a fresh raw connection would be empty; the Store
        # re-inserts the default on construction and the property returns it.
        s2 = Store(p)
        assert s2.section_schema == DEFAULT_SCHEMA
        s2.close()

    def test_corrupt_schema_falls_back_not_crash(self, tmp_path):
        p = str(tmp_path / "m.db")
        Store(p).close()
        conn = sqlite3.connect(p)
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES "
            "('section_schema', '{not valid json')"
        )
        conn.commit()
        conn.close()
        assert Store(p).section_schema == DEFAULT_SCHEMA

    def test_invalid_schema_at_construction_raises_no_db(self, tmp_path):
        p = tmp_path / "bad.db"
        with pytest.raises(ValueError):
            Store(str(p), section_schema=[{"heading": "X", "role": "nope"}])
        assert not p.exists()

    def test_section_schema_returns_fresh_copy(self, tmp_path):
        # Mutating a returned schema must not corrupt the shared singleton
        # (codex L1 / L1-L2 M3 — fallback path returned DEFAULT_SCHEMA by ref).
        s = Store(str(tmp_path / "m.db"))  # default schema
        sch = s.section_schema
        sch[0]["heading"] = "MUTATED"
        assert s.section_schema[0]["heading"] != "MUTATED"
        assert DEFAULT_SCHEMA[0]["heading"] == "State"
        s.close()

    def test_set_section_schema_refused_during_active_wrap(self, tmp_path):
        # The schema is frozen for a wrap's duration so prepare and save agree
        # on routing (codex M1).
        s = Store(str(tmp_path / "m.db"), section_schema=FLOW_SCHEMA)
        s.record("x", EpisodeType.OBSERVATION)
        prepare_wrap(s)  # wrap now in progress
        with pytest.raises(ValueError, match="while a wrap is in progress"):
            s.set_section_schema(DEFAULT_SCHEMA)
        s.close()


# -- validate_structure --


DEFAULT_TEXT = (
    "# M\n## State\nx\n## Patterns\ny\n## Decisions\nz\n## Context\nw\n"
)
FLOW_TEXT = (
    "# flow — Memory (v1)\n"
    "## State\na\n## Active Threads\nb\n## Patterns\nc\n"
    "## Decisions\nd\n## Context\ne\n## Understanding\nf\n"
)


class TestValidateStructureSchema:
    def test_default_behavior_unchanged(self):
        assert validate_structure(DEFAULT_TEXT) is True
        assert validate_structure(DEFAULT_TEXT, DEFAULT_SCHEMA) is True
        assert validate_structure("## State only") is False

    def test_flow_schema_requires_all_six_including_understanding(self):
        assert validate_structure(FLOW_TEXT, FLOW_SCHEMA) is True
        missing_understanding = FLOW_TEXT.replace("## Understanding\nf\n", "")
        assert validate_structure(missing_understanding, FLOW_SCHEMA) is False

    def test_default_text_fails_flow_schema(self):
        # Missing Active Threads + Understanding.
        assert validate_structure(DEFAULT_TEXT, FLOW_SCHEMA) is False

    def test_multiword_heading_matches(self):
        sch = [
            {"heading": "Active Threads", "role": "graduating"},
        ]
        assert validate_structure("## Active Threads\nx", sch) is True
        assert validate_structure("## State\nx", sch) is False

    def test_regex_special_heading_matches(self):
        # Headings ending in non-word chars must match (codex M2 — `\b` fails on
        # `## C++`; the `(?<!\w)…(?!\w)` form handles it).
        sch = [{"heading": "C++", "role": "graduating"}]
        assert validate_structure("## C++\nnotes", sch) is True
        assert validate_structure("## C\nnotes", sch) is False

    def test_extra_descriptive_words_preserved(self):
        # Historical leniency preserved: a descriptive header satisfies the
        # required section; an embedded substring does not.
        sch = [{"heading": "State", "role": "graduating"}]
        assert validate_structure("## State of the Project\nx", sch) is True
        assert validate_structure("## Interstate\nx", sch) is False


# -- _build_wrap_instructions --


class TestWrapInstructionsSchema:
    def test_default_preserves_required_substrings(self):
        d = _build_wrap_instructions("MyProj", 20000, "2026-03-31")
        assert d.startswith("Compress your session episodes")
        for sub in ["MyProj", "2026-03-31", "thought:", "1x", "2x", "[evidence:",
                    "`## State`", "`## Context`"]:
            assert sub in d
        # default has a narrative section (Context) -> PM discipline block present
        assert "Compression discipline" in d
        assert "Implementation-claims guardrail" in d

    def test_flow_schema_includes_timeless_guidance(self):
        f = _build_wrap_instructions("flow", 20000, "2026-03-31", FLOW_SCHEMA)
        for sub in ["`## Active Threads`", "`## Understanding`", "TIMELESS",
                    "knowing someone, not a dossier", "FULL arc of the partnership"]:
            assert sub in f

    def test_default_omits_timeless_guidance(self):
        d = _build_wrap_instructions("MyProj", 20000, "2026-03-31")
        assert "knowing someone, not a dossier" not in d

    def test_custom_graduating_section_in_marker_reference(self):
        # The pattern-line marker reference must point at the schema's actual
        # graduating section, not a hardcoded ## Patterns (codex M4).
        sch = [
            {"heading": "State", "role": "live-state"},
            {"heading": "Proven", "role": "graduating"},
            {"heading": "Context", "role": "narrative"},
        ]
        instr = _build_wrap_instructions("e", 20000, "2026-03-31", sch)
        assert "Pattern lines in `## Proven`" in instr
        assert "Pattern lines in `## Patterns`" not in instr

    def test_contradiction_scan_emitted_with_proven(self):
        """AM-CONTRASCAN-EMIT (v0.4.3): a non-empty uncovered_proven list +
        a graduating schema emits the scan block — names + marker contract."""
        out = _build_wrap_instructions(
            "p", 20000, "2026-06-02", None, ["alpha_proven", "beta_proven"]
        )
        assert "Contradiction Scan" in out
        assert "alpha_proven" in out and "beta_proven" in out
        assert "[contradicts:" in out and "[no-contradicts]" in out

    def test_contradiction_scan_absent_when_no_proven_or_omitted(self):
        """No scan block when uncovered_proven is omitted (default None) or
        empty — backward-compatible for direct callers + first wraps."""
        none_out = _build_wrap_instructions("p", 20000, "2026-06-02")
        empty_out = _build_wrap_instructions("p", 20000, "2026-06-02", None, [])
        assert "Contradiction Scan" not in none_out
        assert "Contradiction Scan" not in empty_out

    def test_contradiction_scan_absent_without_graduating_section(self):
        """A schema with no graduating section emits no scan block even when
        Proven names are passed — there is no Proven tier to graduate into,
        so the scan discipline does not apply."""
        sch = [
            {"heading": "State", "role": "live-state"},
            {"heading": "Context", "role": "narrative"},
        ]
        out = _build_wrap_instructions(
            "p", 20000, "2026-06-02", sch, ["alpha_proven"]
        )
        assert "Contradiction Scan" not in out


# -- Tier 2: graduation gate honors the schema --


GRAD_TEXT = (
    "## Patterns\n"
    "- alpha_in_patterns | 2x (2026-01-01) [evidence: 1234abcd \"x\"]\n"
    "## Proven\n"
    "- beta_in_proven | 2x (2026-01-01) [evidence: 5678ef01 \"y\"]\n"
)


class TestGraduationGateSchema:
    def test_default_gate_scans_patterns_only(self):
        names = extract_pattern_names(GRAD_TEXT)
        assert "alpha_in_patterns" in names
        assert "beta_in_proven" not in names

    def test_custom_gate_scans_named_graduating_section(self):
        gh = graduating_headings(
            [
                {"heading": "State", "role": "live-state"},
                {"heading": "Proven", "role": "graduating"},
            ]
        )
        names = extract_pattern_names(GRAD_TEXT, gh)
        assert "beta_in_proven" in names
        assert "alpha_in_patterns" not in names

    def test_multi_graduating_scans_all(self):
        gh = graduating_headings(
            [
                {"heading": "Patterns", "role": "graduating"},
                {"heading": "Proven", "role": "graduating"},
            ]
        )
        names = extract_pattern_names(GRAD_TEXT, gh)
        assert "alpha_in_patterns" in names
        assert "beta_in_proven" in names

    def test_anti_patterns_still_not_matched(self):
        # The v0.3.2 Anti-Patterns fix must survive: "## Anti-Patterns" is not
        # the graduating section under the default gate.
        txt = (
            "## Anti-Patterns\n"
            "- should_not_match | 2x (2026-01-01) [evidence: abcd1234 \"z\"]\n"
        )
        assert "should_not_match" not in extract_pattern_names(txt)


# -- End-to-end FLOW_SCHEMA wrap cycle --


FLOW_CONTINUITY = (
    "# flow — Memory (v1)\n\n"
    "## State\nVerifying the pluggable schema end to end.\n\n"
    "## Active Threads\nThe 0.3.4 build is live.\n\n"
    "## Patterns\n{testing:\n  schema_drives_the_pipeline | 1x (2026-05-31)\n}\n\n"
    "## Decisions\n[decided(rationale: \"dogfood\", on: \"2026-05-31\")] Ship 0.3.4.\n\n"
    "## Context\nBuilt and verified the schema feature today.\n\n"
    "## Understanding\nDirect, depth-first partner. Working together is thinking out loud.\n"
)


class TestFlowSchemaEndToEnd:
    def test_prepare_wrap_describes_flow_schema(self, tmp_path):
        s = Store(str(tmp_path / "flow.db"), project_name="flow",
                  section_schema=FLOW_SCHEMA)
        s.record("did a thing", EpisodeType.OBSERVATION)
        pkg = prepare_wrap(s)
        assert pkg["status"] == "ready"
        instr = pkg["package"]["instructions"]
        assert "`## Understanding`" in instr
        assert "`## Active Threads`" in instr
        assert "knowing someone, not a dossier" in instr
        s.close()

    def test_full_wrap_cycle_saves_flow_continuity(self, tmp_path):
        s = Store(str(tmp_path / "flow.db"), project_name="flow",
                  section_schema=FLOW_SCHEMA)
        s.record("did a thing", EpisodeType.OBSERVATION)
        prepare_wrap(s)
        result = validated_save_continuity(s, FLOW_CONTINUITY, today="2026-05-31")
        assert result is not None
        saved = s.load_continuity()
        assert saved is not None
        assert "## Understanding" in saved
        assert "## Active Threads" in saved
        s.close()

    def test_default_store_rejects_flow_text_missing_sections(self, tmp_path):
        # A default-schema store must still reject a 4-section save that omits
        # nothing — but a 6-section flow text saved to a default store is fine
        # (default only REQUIRES its four; extras are allowed).
        s = Store(str(tmp_path / "d.db"), project_name="ops")
        s.record("x", EpisodeType.OBSERVATION)
        prepare_wrap(s)
        # missing ## Context -> rejected under default schema
        bad = "# M\n## State\na\n## Patterns\nb\n## Decisions\nc\n"
        with pytest.raises(ValueError, match="must contain all sections"):
            validated_save_continuity(s, bad, today="2026-05-31")
        s.close()

    def test_custom_graduating_section_full_wrap_cycle(self, tmp_path):
        # End-to-end with a NON-Patterns graduating section: prepare_wrap's
        # instructions point at ## Proven, and a full save cycle succeeds with
        # the graduating section named Proven (Tier 2 routing, end to end).
        custom_schema = [
            {"heading": "State", "role": "live-state"},
            {"heading": "Proven", "role": "graduating"},
            {"heading": "Decisions", "role": "decisions"},
            {"heading": "Context", "role": "narrative"},
        ]
        custom_continuity = (
            "# e — Memory (v1)\n\n"
            "## State\nx\n\n"
            "## Proven\n- a_principle | 1x (2026-05-31)\n\n"
            '## Decisions\n[decided(rationale: "r", on: "2026-05-31")] d\n\n'
            "## Context\nshape, not transcript\n"
        )
        s = Store(str(tmp_path / "c.db"), project_name="e",
                  section_schema=custom_schema)
        s.record("y", EpisodeType.OBSERVATION)
        pkg = prepare_wrap(s)
        assert "Pattern lines in `## Proven`" in pkg["package"]["instructions"]
        validated_save_continuity(s, custom_continuity, today="2026-05-31")
        saved = s.load_continuity()
        assert saved is not None and "## Proven" in saved
        s.close()


# -- AM-SCHEMA-BUDGET (v0.4.2): schema-aware default max_chars --


class TestDefaultMaxChars:
    def test_default_schema_is_exactly_20000(self):
        # The load-bearing byte-compat invariant: ops entities are unchanged.
        assert default_max_chars(DEFAULT_SCHEMA) == 20000

    def test_flow_schema_gets_headroom(self):
        # +1500 for the extra live-state (Active Threads) + 4000 for the
        # narrative-timeless felt floor (Understanding).
        assert default_max_chars(FLOW_SCHEMA) == 25500
        assert default_max_chars(FLOW_SCHEMA) > default_max_chars(DEFAULT_SCHEMA)

    def test_extra_graduating_section_adds_budget(self):
        sch = validate_schema([
            {"heading": "State", "role": "live-state"},
            {"heading": "P1", "role": "graduating"},
            {"heading": "P2", "role": "graduating"},
            {"heading": "Ctx", "role": "narrative"},
        ])
        assert default_max_chars(sch) == 22500  # 20000 + 2500 for the 2nd graduating

    def test_narrative_timeless_adds_felt_floor(self):
        sch = validate_schema([
            {"heading": "State", "role": "live-state"},
            {"heading": "Patterns", "role": "graduating"},
            {"heading": "Decisions", "role": "decisions"},
            {"heading": "Context", "role": "narrative"},
            {"heading": "Understanding", "role": "narrative-timeless"},
        ])
        assert default_max_chars(sch) == 24000  # 20000 + 4000

    def test_extra_live_state_adds_budget(self):
        sch = validate_schema([
            {"heading": "State", "role": "live-state"},
            {"heading": "Threads", "role": "live-state"},
            {"heading": "Patterns", "role": "graduating"},
            {"heading": "Context", "role": "narrative"},
        ])
        assert default_max_chars(sch) == 21500  # 20000 + 1500

    def test_frozen_section_adds_budget(self):
        sch = validate_schema([
            {"heading": "State", "role": "live-state"},
            {"heading": "Patterns", "role": "graduating"},
            {"heading": "Context", "role": "narrative"},
            {"heading": "Charter", "role": "frozen"},
        ])
        assert default_max_chars(sch) == 21000  # 20000 + 1000

    def test_prepare_wrap_default_uses_schema_budget(self, tmp_path):
        # No explicit max_chars -> schema-aware default flows into the package.
        flow = Store(str(tmp_path / "flow.db"), project_name="flow",
                     section_schema=FLOW_SCHEMA)
        flow.record("did a thing", EpisodeType.OBSERVATION)
        assert prepare_wrap(flow)["package"]["max_chars"] == 25500
        flow.close()

        ops = Store(str(tmp_path / "ops.db"), project_name="ops")
        ops.record("did a thing", EpisodeType.OBSERVATION)
        assert prepare_wrap(ops)["package"]["max_chars"] == 20000  # byte-compat
        ops.close()

    def test_prepare_wrap_explicit_max_chars_overrides(self, tmp_path):
        flow = Store(str(tmp_path / "flow.db"), project_name="flow",
                     section_schema=FLOW_SCHEMA)
        flow.record("did a thing", EpisodeType.OBSERVATION)
        assert prepare_wrap(flow, max_chars=9999)["package"]["max_chars"] == 9999
        flow.close()

    def test_build_wrap_instructions_none_resolves_per_schema(self):
        # None max_chars resolves to the schema-aware default at the instruction layer.
        d = _build_wrap_instructions("ops", None, "2026-06-02")
        assert "within 20000 characters" in d
        f = _build_wrap_instructions("flow", None, "2026-06-02", FLOW_SCHEMA)
        assert "within 25500 characters" in f


class TestSchemaRoleWarning:
    """AM-ROLECHECK (v0.5.0): schema_role_warning — surface a VALID-but-mis-roled
    schema that would silently thin the prepare_wrap package, without
    false-warning on the known-good or benign-reorder cases."""

    def _misrole(self, schema, heading, new_role):
        out = [dict(s) for s in schema]
        for s in out:
            if s["heading"] == heading:
                s["role"] = new_role
        return out

    def test_known_good_named_schemas_silent(self):
        # Exact named schemas are correct by construction — never warn.
        assert schema_role_warning(DEFAULT_SCHEMA) is None
        assert schema_role_warning(FLOW_SCHEMA) is None

    def test_reorder_only_is_benign(self):
        # A reordering of a named schema (e.g. a future Understanding-at-primacy
        # profile) keeps every role — must NOT warn. Guards the CHIPSCHEMA path.
        reordered = [s for s in FLOW_SCHEMA if s["heading"] == "Understanding"] + [
            s for s in FLOW_SCHEMA if s["heading"] != "Understanding"
        ]
        assert name_for_schema(reordered) is None  # not an exact named match
        assert schema_role_warning(reordered) is None

    def test_felt_misrole_is_the_load_bearing_catch(self):
        # The prepare-reachable, load-bearing case: Understanding demoted from
        # narrative-timeless to narrative — keeps a graduating section (passes
        # validate_schema) but silently drops the felt proportion-check + the
        # shrink-gate's felt protection.
        misroled = self._misrole(FLOW_SCHEMA, "Understanding", "narrative")
        validate_schema(misroled)  # still valid (has graduating, valid roles)
        warning = schema_role_warning(misroled)
        assert warning is not None
        assert "Understanding" in warning
        assert "narrative-timeless" in warning
        assert "partnership" in warning

    def test_misrole_off_default_schema(self):
        # Heading-set match against DEFAULT with a differing role also fires.
        misroled = self._misrole(DEFAULT_SCHEMA, "Context", "decisions")
        assert name_for_schema(misroled) is None
        warning = schema_role_warning(misroled)
        assert warning is not None
        assert "Context" in warning and "default" in warning

    def test_no_graduating_floor_is_a_backstop(self):
        # validate_schema HARD-REFUSES a 0-graduating schema at persist, so the
        # floor only fires for a raw schema passed directly to this function.
        raw = [
            {"heading": "State", "role": "live-state"},
            {"heading": "Notes", "role": "narrative"},
        ]
        with pytest.raises(ValueError):
            validate_schema(raw)  # the store path would never persist this
        warning = schema_role_warning(raw)
        assert warning is not None
        assert "no 'graduating' section" in warning

    def test_valid_custom_novel_headings_silent(self):
        # A legitimate custom schema (novel headings, has graduating) — no
        # named-schema heading-set match and a graduating section → no warning.
        custom = [
            {"heading": "Now", "role": "live-state"},
            {"heading": "Principles", "role": "graduating"},
            {"heading": "Log", "role": "narrative"},
        ]
        validate_schema(custom)
        assert schema_role_warning(custom) is None
