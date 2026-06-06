"""Tests for AM-CRYSTAL-MIGRATE — the crystallize/compost wrap integration.

Two halves:
  1. The SHRINK-GATE CREDIT (safety-critical): a wrap that crystallizes patterns
     OUT of ## Patterns is credited (read as a MOVE, not a collapse) so the gate
     stays ON without a blanket allow_shrink — while a recency-trap, which earns
     ZERO credit (the vanished patterns are not in the crystal store), still trips.
  2. The PREPARE-SIDE surfacing: prepare_wrap surfaces cold-Proven crystallization
     candidates + hot re-warm candidates + extends the dedup/contradiction scans to
     read the crystal corpus.
"""

from __future__ import annotations

from datetime import date

import pytest

from anneal_memory import (
    CrystalStore,
    FLOW_SCHEMA,
    Store,
    prepare_wrap,
    validated_save_continuity,
)
from anneal_memory.continuity import (
    _check_no_catastrophic_shrink,
    _crystallization_credit,
    _role_section_body,
)

T0 = date(2026, 6, 6)
TODAY = "2026-06-06"

# Three named patterns, each padded so ## Patterns clears the 500-char gate floor.
_PAT_NAMES = [
    "structural_invariants_beat_discipline",
    "verify_or_surface_before_acting",
    "play_first_is_the_engine",
]


def _patterns_section(names: list[str]) -> str:
    # Graduation-format lines (the immune-system format the owner-based credit keys
    # on), each padded so the ## Patterns section clears the 500-char gate floor.
    # Dated 2026-06-01 (not today) so graduation validation skips them on save.
    pad = " felt prose padding clause to push this line's mass past the floor." * 3
    return "\n".join(
        f'- {n} | 3x (2026-06-01) [evidence: abcd1234 "why {n}"]{pad}' for n in names
    )


def _flow_doc(patterns_body: str) -> str:
    understanding = (
        "Phill is a paradox-holding ensemble mind; play is the engine not the mood; "
        "the work is high-bandwidth and for the record. "
    ) * 16
    return (
        "# flow — Memory (test)\n\n"
        "## State\nstore-half build in progress.\n\n"
        "## Active Threads\n- crystal\n- argus\n\n"
        "## Patterns\n" + patterns_body + "\n\n"
        "## Decisions\n[decided] crystallized tier ships first.\n\n"
        "## Context\nbuilding the AM-CRYSTAL store-half.\n\n"
        "## Understanding\n" + understanding + "\n"
    )


# -- _role_section_body ----------------------------------------------------


class TestRoleSectionBody:
    def test_extracts_graduating_body_excludes_header(self):
        doc = _flow_doc(_patterns_section(_PAT_NAMES))
        body = "\n".join(_role_section_body(doc, FLOW_SCHEMA, "graduating"))
        assert "structural_invariants_beat_discipline" in body
        assert "## Patterns" not in body  # header line excluded
        assert "store-half build in progress" not in body  # other sections excluded

    def test_unknown_role_empty(self):
        doc = _flow_doc(_patterns_section(_PAT_NAMES))
        assert _role_section_body(doc, FLOW_SCHEMA, "no-such-role") == []


# -- _crystallization_credit ----------------------------------------------


class TestCrystallizationCredit:
    @pytest.fixture
    def crystal(self, tmp_path):
        return CrystalStore(tmp_path / "mem.crystal.json")

    def test_departed_and_crystallized_today_is_credited(self, crystal):
        for n in _PAT_NAMES:
            crystal.crystallize(name=n, level=3, explanation="x", today=T0)
        prior = _flow_doc(_patterns_section(_PAT_NAMES))
        new = _flow_doc("a single fresh pattern line remains here after crystallization.")
        credit = _crystallization_credit(prior, new, FLOW_SCHEMA, crystal)
        assert credit.get("graduating", 0) > 0

    def test_still_present_in_new_not_credited(self, crystal):
        for n in _PAT_NAMES:
            crystal.crystallize(name=n, level=3, explanation="x", today=T0)
        prior = _flow_doc(_patterns_section(_PAT_NAMES))
        new = prior  # nothing departed → no credit
        assert _crystallization_credit(prior, new, FLOW_SCHEMA, crystal) == {}

    def test_credited_by_recoverability_not_date(self, crystal):
        # Provenance is by RECOVERABILITY, not date: a pattern crystallized long ago,
        # still LIVE in the store, that departs ## Patterns this wrap IS credited (it
        # is recoverable from the store — re-warm after re-cool, old origin date).
        for n in _PAT_NAMES:
            crystal.crystallize(name=n, level=3, explanation="x", today=date(2026, 1, 1))
        prior = _flow_doc(_patterns_section(_PAT_NAMES))
        new = _flow_doc("a single fresh line remains here after crystallizing the rest out.")
        assert _crystallization_credit(prior, new, FLOW_SCHEMA, crystal).get("graduating", 0) > 0

    def test_not_in_store_not_credited(self, crystal):
        # A pattern that departed but is NOT in the crystal store (a recency-trap) is
        # never credited — it is NOT recoverable.
        prior = _flow_doc(_patterns_section(_PAT_NAMES))  # nothing crystallized
        new = _flow_doc("a single fresh line remains here.")
        assert _crystallization_credit(prior, new, FLOW_SCHEMA, crystal) == {}

    def test_survivor_dodging_anchored_regex_not_credited(self, crystal):
        # HIGH (L3 complement+kimi): a SURVIVING crystallized pattern present in new
        # but DODGING the anchored first-marker regex — (a) a column-0 bare line, and
        # (b) a marker merged as the SECOND marker on another line — must still be seen
        # as present by the loose all-markers scan → NOT credited as departed.
        crystal.crystallize(name="col0_survivor", level=3, explanation="x", today=T0)
        crystal.crystallize(name="merged_survivor", level=3, explanation="x", today=T0)
        prior = _flow_doc(
            '- col0_survivor | 3x (2026-06-01) [evidence: abcd1234 "a"] prose one.\n'
            '- merged_survivor | 3x (2026-06-01) [evidence: efgh5678 "b"] prose two.'
        )
        # new: col0_survivor as a COLUMN-0 bare line (no bullet); merged_survivor as a
        # SECOND marker on the keeper's line. Both dodge `_NAMED_PATTERN_RE.match`.
        new = _flow_doc(
            'col0_survivor | 2x (2026-06-06) [evidence: a1a1a1a1 "still here"] surviving.\n'
            '- keeper | 3x (2026-06-06) ... and also merged_survivor | 2x trailing.'
        )
        # Neither survivor departed → zero credit (the loose scan caught both).
        assert _crystallization_credit(prior, new, FLOW_SCHEMA, crystal) == {}

    def test_no_crystal_store_or_no_prior_is_empty(self, crystal):
        prior = _flow_doc(_patterns_section(_PAT_NAMES))
        assert _crystallization_credit(prior, "x", FLOW_SCHEMA, None) == {}
        assert _crystallization_credit(None, "x", FLOW_SCHEMA, crystal) == {}

    def test_corrupt_crystal_store_no_credit_no_raise(self, tmp_path):
        path = tmp_path / "bad.crystal.json"
        path.write_text("{not valid json", encoding="utf-8")
        crystal = CrystalStore(path)
        prior = _flow_doc(_patterns_section(_PAT_NAMES))
        # must NOT raise, and must NOT credit (gate stays strict on a store fault)
        assert _crystallization_credit(prior, "x", FLOW_SCHEMA, crystal) == {}

    def test_oserror_crystal_store_no_credit_no_raise_warns(self):
        # codex L3 (2026-06-06): the credit path caught CrystalError ONLY, so a raw
        # OSError (PermissionError, disk I/O) ESCAPED and broke the SAVE — violating
        # the crystal-fault-never-breaks-a-wrap invariant the corrupt-JSON case above
        # honors. It must degrade to no-credit AND surface a diagnostic (degrade-but
        # -warn; a guard must not silently mask the fault it protects against).
        class _FaultStore:
            def active(self):
                raise PermissionError("simulated EACCES on memory.crystal.json")

        prior = _flow_doc(_patterns_section(_PAT_NAMES))
        with pytest.warns(UserWarning, match="crystal store unreadable"):
            assert _crystallization_credit(prior, "x", FLOW_SCHEMA, _FaultStore()) == {}


# -- credit OWNER accounting (regression for the L1 HIGH over-credit) ------


class TestCreditOwnerAccounting:
    """Credit must key on each prior line's OWNER pattern (the ``name | Nx`` the
    immune system parses), counted ONCE — never on a merely-mentioned name, a
    substring, or per-name. Regression for the over-credit hole where a recency-trap
    that crystallized one substring-y / sibling-referenced name could mask the gate.
    """

    @pytest.fixture
    def crystal(self, tmp_path):
        return CrystalStore(tmp_path / "mem.crystal.json")

    def test_sibling_reference_does_not_leak_credit(self, crystal):
        # kept_pattern survives and references [[departed_pattern]] in its line.
        # Only departed_pattern's OWN line is credited, not the referencing line.
        crystal.crystallize(name="departed_pattern", level=3, explanation="x", today=T0)
        kept = ('- kept_pattern | 3x (2026-06-01) [evidence: abcd1234 "k"] felt. '
                'Sibling of [[departed_pattern]].')
        dep = '- departed_pattern | 3x (2026-06-01) [evidence: efgh5678 "d"] felt prose here.'
        prior = _flow_doc(kept + "\n" + dep)
        new = _flow_doc(kept)  # departed_pattern's line gone; kept (w/ sibling ref) stays
        credit = _crystallization_credit(prior, new, FLOW_SCHEMA, crystal)
        assert credit.get("graduating", 0) == len(dep) + 1

    def test_substring_owner_not_credited(self, crystal):
        # 'verify' departs; 'verify_or_surface' stays. The substring must not cause
        # verify_or_surface's (surviving) line to be credited.
        crystal.crystallize(name="verify", level=3, explanation="x", today=T0)
        vshort = '- verify | 3x (2026-06-01) [evidence: abcd1234 "x"] short owner line here now.'
        vlong = ('- verify_or_surface | 3x (2026-06-01) [evidence: efgh5678 "y"] a longer '
                 'surviving owner line.')
        prior = _flow_doc(vshort + "\n" + vlong)
        new = _flow_doc(vlong)
        credit = _crystallization_credit(prior, new, FLOW_SCHEMA, crystal)
        assert credit.get("graduating", 0) == len(vshort) + 1

    def test_recency_trap_with_substring_crystal_still_trips_gate(self, crystal):
        # The L1 scenario: 5 provens, the wrap guts ALL of them, but crystallizes ONE
        # whose name ('proven') is a substring of the others. The gate must STILL trip
        # (credit covers only the one departed owner's line, nowhere near the collapse).
        from anneal_memory.continuity import _schema_section_masses
        crystal.crystallize(name="proven", level=3, explanation="x", today=T0)
        names = ["proven_alpha", "proven_beta", "proven_gamma", "proven_delta", "proven"]
        # pad ×3 so the FOUR recency-trapped patterns alone clear the 500-char floor
        # (the (prior - credit) baseline excludes the one crystallized 'proven' line).
        pad = " felt prose padding clause to push this line's mass past the floor." * 3
        body = "\n".join(
            f'- {n} | 3x (2026-06-01) [evidence: abcd1234 "why{n}"]{pad}'
            for n in names
        )
        prior = _flow_doc(body)
        collapsed = _flow_doc("- lone_survivor | 1x (2026-06-06) one short line.")
        credit = _crystallization_credit(prior, collapsed, FLOW_SCHEMA, crystal)
        pm = _schema_section_masses(prior, FLOW_SCHEMA)["patterns"]
        # credit covers only 'proven's line; the non-crystallized baseline still huge
        assert credit.get("graduating", 0) < pm * 0.5
        with pytest.raises(ValueError, match="Patterns"):
            _check_no_catastrophic_shrink(
                prior, collapsed, FLOW_SCHEMA, allow_shrink=False,
                crystallized_credit=credit,
            )

    def test_crystallization_credit_is_neutral_to_nongraduating_collapse(self):
        # HIGH-2 (L3 complement): graduating credit must NOT be fungible across the
        # whole-doc backstop — a legit Patterns crystallization can't excuse a
        # recency-trap of an UNPROTECTED section (Decisions/Context). The doc backstop
        # excludes graduating, so a non-graduating collapse trips regardless of credit.
        big_decisions = ("[decided] a substantial accumulated decision line. " * 30)
        big_context = ("recent work narrative across the arc, substantial. " * 30)
        prior = (
            "# m\n\n## State\ns\n\n## Active Threads\n- x\n\n"
            "## Patterns\n- p | 3x (2026-06-01) [evidence: abcd1234 \"x\"] some mass here now.\n\n"
            "## Decisions\n" + big_decisions + "\n\n"
            "## Context\n" + big_context + "\n\n"
            "## Understanding\n" + ("the felt relationship shape, preserved. " * 20) + "\n"
        )
        # new: Patterns fully crystallized out (huge credit) AND Decisions+Context gutted.
        new = (
            "# m\n\n## State\ns\n\n## Active Threads\n- x\n\n"
            "## Patterns\n\n## Decisions\ngutted.\n\n## Context\ngutted.\n\n"
            "## Understanding\n" + ("the felt relationship shape, preserved. " * 20) + "\n"
        )
        # Even with a large (truthful) graduating credit, the non-graduating collapse trips.
        with pytest.raises(ValueError, match="excl. graduating"):
            _check_no_catastrophic_shrink(
                prior, new, FLOW_SCHEMA, allow_shrink=False,
                crystallized_credit={"graduating": 100000},
            )

    def test_non_pattern_lines_never_credited(self, crystal):
        # A prior body line with no owner (prose / group header) is never credited.
        crystal.crystallize(name="real_pattern", level=3, explanation="x", today=T0)
        prose = "Just a group-header or prose line mentioning real_pattern in passing."
        prior = _flow_doc(prose)
        new = _flow_doc("nothing here.")
        # real_pattern is crystallized + absent from new, but its only prior mention
        # is a non-owner prose line → no credit.
        assert _crystallization_credit(prior, new, FLOW_SCHEMA, crystal) == {}

    def test_merged_line_hitchhiking_not_credited(self, crystal):
        # codex convergence HIGH: a prior line with TWO markers — a recoverable owner
        # (`alpha`, in store) + a SECOND non-recoverable pattern (`beta`, NOT in store)
        # whose recency-trapped mass hitchhikes the line — must earn ZERO credit (a
        # multi-marker line is ambiguous). So beta's mass stays in the protected
        # baseline and a co-occurring collapse trips.
        from anneal_memory.continuity import _schema_section_masses
        crystal.crystallize(name="alpha", level=3, explanation="x", today=T0)
        filler = " non-recoverable recency-trapped prose mass that must stay gated." * 12
        merged = f'- alpha | 3x (2026-06-01) short. beta | 1x (2026-06-01){filler}'
        prior = _flow_doc(merged)
        collapsed = _flow_doc("- lone_survivor | 1x (2026-06-06) one short line.")
        # zero credit (the line has 2 markers) → beta's mass stays in the baseline
        assert _crystallization_credit(prior, collapsed, FLOW_SCHEMA, crystal) == {}
        pm = _schema_section_masses(prior, FLOW_SCHEMA)["patterns"]
        assert pm >= 500  # the merged line clears the floor on its own
        with pytest.raises(ValueError, match="Patterns"):
            _check_no_catastrophic_shrink(
                prior, collapsed, FLOW_SCHEMA, allow_shrink=False,
                crystallized_credit=_crystallization_credit(prior, collapsed, FLOW_SCHEMA, crystal),
            )

    def test_credit_requires_owner_in_store_not_just_any_marker(self, crystal):
        # codex MED #4: a prior line that matches the immune owner-regex but is NOT a
        # crystallized pattern (e.g. an indented continuation line that happens to read
        # as `name | Nx`) earns NO credit unless `name` is actually live in the store.
        # The store membership requirement bounds the immune-regex ambiguity: only a
        # genuinely-recoverable (in-store) owner can ever be credited.
        prior = _flow_doc(
            '- real_pattern | 3x (2026-06-01) [evidence: abcd1234 "x"] padding mass here.\n'
            '  notation_only | 3x is just an inline notation example, not a real pattern row.'
        )
        new = _flow_doc("fresh line, both gone.")
        # neither real_pattern nor notation_only is in the store → zero credit
        assert _crystallization_credit(prior, new, FLOW_SCHEMA, crystal) == {}


# -- shrink gate WITH credit (unit) ----------------------------------------


class TestShrinkGateCredit:
    def _prior_new_patterns_only_shrink(self):
        # Understanding identical in both → only ## Patterns shrinks, so the gate's
        # graduating-role check is what the credit must rescue.
        prior = _flow_doc(_patterns_section(_PAT_NAMES))
        new = _flow_doc("one short remaining pattern line after the crystallization.")
        return prior, new

    def test_no_credit_trips_on_patterns(self):
        prior, new = self._prior_new_patterns_only_shrink()
        with pytest.raises(ValueError, match="Patterns"):
            _check_no_catastrophic_shrink(prior, new, FLOW_SCHEMA, allow_shrink=False)

    def test_credit_covering_shortfall_passes(self):
        prior, new = self._prior_new_patterns_only_shrink()
        # credit ≥ the graduating shortfall → must not raise
        from anneal_memory.continuity import _schema_section_masses
        pm = _schema_section_masses(prior, FLOW_SCHEMA)["patterns"]
        nm = _schema_section_masses(new, FLOW_SCHEMA)["patterns"]
        _check_no_catastrophic_shrink(
            prior, new, FLOW_SCHEMA, allow_shrink=False,
            crystallized_credit={"graduating": pm - nm},
        )

    def test_credit_capped_cannot_manufacture_phantom_mass(self):
        # An absurd credit cannot rescue a genuine Understanding (felt) collapse —
        # crystallization only credits graduating, and credit is capped at the
        # observed shortfall.
        prior = _flow_doc(_patterns_section(_PAT_NAMES))
        collapsed = (
            "# flow — Memory (test)\n\n## State\ns\n\n## Active Threads\n- x\n\n"
            "## Patterns\np\n\n## Decisions\nd\n\n## Context\nc\n\n"
            "## Understanding\nshort.\n"
        )
        with pytest.raises(ValueError, match="Understanding"):
            _check_no_catastrophic_shrink(
                prior, collapsed, FLOW_SCHEMA, allow_shrink=False,
                crystallized_credit={"graduating": 10_000_000, "narrative-timeless": 0},
            )

    def test_credit_none_is_byte_identical_old_behavior(self):
        prior, new = self._prior_new_patterns_only_shrink()
        with pytest.raises(ValueError):
            _check_no_catastrophic_shrink(
                prior, new, FLOW_SCHEMA, allow_shrink=False, crystallized_credit=None
            )


# -- end-to-end: validated_save_continuity with crystal_store --------------


class TestSaveWithCrystalStore:
    def _store(self, tmp_path):
        store = Store(tmp_path / "mem.db", project_name="flow")
        store.set_section_schema(FLOW_SCHEMA)
        return store

    def _wrap(self, store, text, *, crystal_store=None, n_eps=3):
        for i in range(n_eps):
            store.record(f"episode {TODAY} #{i}: a substrate observation about topic {i} here.",
                         "observation")
        result = prepare_wrap(store, max_chars=60000, crystal_store=crystal_store)
        return validated_save_continuity(
            store, text, today=TODAY, wrap_token=result["wrap_token"],
            crystal_store=crystal_store,
        )

    def test_legit_crystallization_passes_with_credit(self, tmp_path):
        store = self._store(tmp_path)
        crystal = CrystalStore(tmp_path / "mem.crystal.json")
        # wrap 1: establish a prior with a big ## Patterns (no gate on first wrap)
        self._wrap(store, _flow_doc(_patterns_section(_PAT_NAMES)), crystal_store=crystal)
        # crystallize the three patterns OUT, today
        for n in _PAT_NAMES:
            crystal.crystallize(name=n, level=3, explanation="proven principle", today=T0)
        # wrap 2: ## Patterns now drops them (moved to crystal), Understanding kept.
        new = _flow_doc("one short remaining working-set pattern line after crystallizing out.")
        # passes BECAUSE crystal_store grounds the credit
        res = self._wrap(store, new, crystal_store=crystal)
        assert res["chars"] > 0  # saved, not refused

    def test_recency_trap_still_trips_without_crystal_grounding(self, tmp_path):
        store = self._store(tmp_path)
        crystal = CrystalStore(tmp_path / "mem.crystal.json")
        self._wrap(store, _flow_doc(_patterns_section(_PAT_NAMES)), crystal_store=crystal)
        # patterns NOT crystallized — they just vanish (the recency-trap)
        new = _flow_doc("one short remaining working-set pattern line.")
        with pytest.raises(ValueError, match="Patterns"):
            self._wrap(store, new, crystal_store=crystal)


# -- prepare-side surfacing ------------------------------------------------


class TestPrepareSurfacing:
    def _store(self, tmp_path):
        store = Store(tmp_path / "mem.db", project_name="flow")
        store.set_section_schema(FLOW_SCHEMA)
        return store

    def test_rewarm_candidates_surface_hot_crystal(self, tmp_path):
        store = self._store(tmp_path)
        crystal = CrystalStore(tmp_path / "mem.crystal.json")
        crystal.crystallize(name="hot_pattern", level=3, explanation="x", today=T0)
        crystal.crystallize(name="cold_pattern", level=2, explanation="x", today=date(2026, 1, 1))
        store.record("an episode about substrate topics worth at least eighty characters here now.",
                     "observation")
        result = prepare_wrap(store, crystal_store=crystal)
        # only the hot one is a re-warm candidate
        assert "hot_pattern" in result["rewarm_candidates"]
        assert "cold_pattern" not in result["rewarm_candidates"]

    def test_crystallization_block_emitted_in_instructions(self, tmp_path):
        store = self._store(tmp_path)
        crystal = CrystalStore(tmp_path / "mem.crystal.json")
        crystal.crystallize(name="hot_pattern", level=3, explanation="x", today=T0)
        store.record("an episode about substrate topics worth at least eighty characters here now.",
                     "observation")
        result = prepare_wrap(store, crystal_store=crystal)
        assert "Crystallization Routing" in result["package"]["instructions"]

    def test_crystal_corpus_extends_contradiction_scan(self, tmp_path):
        # A crystallized Proven that's no longer in ## Patterns must still appear in
        # the contradiction-scan set so a wrap can't graduate a contradicting pattern.
        store = self._store(tmp_path)
        crystal = CrystalStore(tmp_path / "mem.crystal.json")
        crystal.crystallize(name="crystallized_only_pattern", level=3, explanation="x", today=T0)
        # prior continuity has its own ## Patterns proven, distinct from the crystal one
        store.save_continuity(_flow_doc("existing_inline_pattern | 2x (2026-06-01) [evidence: abcd1234 \"x\"]"))
        store.record("an episode about substrate topics worth at least eighty characters here now.",
                     "observation")
        result = prepare_wrap(store, crystal_store=crystal)
        assert "crystallized_only_pattern" in result["uncovered_proven_to_check"]

    def test_no_crystal_store_no_surfacing(self, tmp_path):
        store = self._store(tmp_path)
        store.record("an episode about substrate topics worth at least eighty characters here now.",
                     "observation")
        result = prepare_wrap(store)  # no crystal_store
        assert result["rewarm_candidates"] == []
        assert result["crystallization_candidates"] == []

    def test_oserror_on_active_does_not_break_prepare(self, tmp_path, monkeypatch):
        # codex L3 (2026-06-06): an OSError reading the crystal store during prepare
        # (active() in the package build) must NOT abort prepare_wrap — aborting would
        # strand a wrap-in-progress over a fault in an ADDITIVE tier. _crystal_active_safe
        # now catches (CrystalError, OSError) + warns; the wrap proceeds with the
        # crystallized tier degraded to empty rather than breaking.
        store = self._store(tmp_path)
        crystal = CrystalStore(tmp_path / "mem.crystal.json")
        crystal.crystallize(name="hot_pattern", level=3, explanation="x", today=T0)
        store.record("an episode about substrate topics worth at least eighty characters here now.",
                     "observation")

        def _boom(self, *a, **k):
            raise PermissionError("simulated EACCES on memory.crystal.json")

        monkeypatch.setattr(CrystalStore, "active", _boom)
        with pytest.warns(UserWarning, match="crystal store unreadable"):
            result = prepare_wrap(store, crystal_store=crystal)
        assert result["status"] == "ready"        # the wrap is NOT broken by the fault
        assert result["rewarm_candidates"] == []  # crystal tier degraded, not fatal
