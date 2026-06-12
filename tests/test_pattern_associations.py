"""Tests for the cortical pattern-association graph (AM-LINKGATE-DECAY, Slice B).

The pattern-level sibling of the episode Hebbian layer: keyed on stable pattern
NAMES, strengthened on co-RETRIEVAL (the idempotent co-surface drain), decayed on
calendar-DISUSE (lazy-on-touch). SHADOW MODE — nothing reads it for recall yet;
these tests lock the producer/consumer mechanics + the lifecycle (rename / homonym
guard) + the telemetry so Slice C can be gated on a clean graph.

Covers: schema, canonical ordering, calendar decay math, co-graduation seeding,
the idempotent drain (formation / strengthening / per-session burst-damp /
provenance-gating / event_id dedup), homeostatic normalization, GC thresholds,
rename (re-key + merge + self-pair drop), the homonym sever + generation bump,
reads/telemetry, the Store wrappers, and the end-to-end wrap-pipeline seeding.
"""

import math

import pytest

from anneal_memory.pattern_associations import (
    CO_SURFACE_BASE,
    DECAY_PER_DAY,
    GC_THRESHOLD,
    NODE_OUTGOING_BUDGET,
    RETRIEVAL_THRESHOLD,
    SEED_STRENGTH,
    canonical_pair,
    effective_strength,
    _current_generation,
)
from anneal_memory.store import Store


# -- Helpers --

def _store(tmp_path) -> Store:
    return Store(str(tmp_path / "m.db"), project_name="test")


def _events(*specs, basis="keyword"):
    """Build co-surface event dicts. Each spec = (event_id, session, [names])."""
    return [
        {"event_id": eid, "ts": "2026-06-12T10:00:00Z", "session": sess,
         "names": names, "basis": basis}
        for eid, sess, names in specs
    ]


def _pair(store, a, b, today="2026-06-12"):
    """Effective strength of a specific pair, or None if absent."""
    for p in store.pattern_association_stats(today=today, top_n=999).strongest_pairs:
        if {p.name_a, p.name_b} == {a, b}:
            return p
    return None


# -- Pure helpers --

class TestCanonicalPair:
    def test_orders_lexicographically(self):
        assert canonical_pair("zebra", "alpha") == ("alpha", "zebra")
        assert canonical_pair("alpha", "zebra") == ("alpha", "zebra")

    def test_self_pair_is_none(self):
        assert canonical_pair("same", "same") is None


class TestEffectiveStrength:
    def test_no_decay_same_day(self):
        assert effective_strength(1.0, "2026-06-12", "2026-06-12") == 1.0

    def test_decays_by_calendar_days(self):
        # 10 days at 0.96/day
        assert effective_strength(1.0, "2026-06-02", "2026-06-12") == pytest.approx(0.96 ** 10)

    def test_backwards_span_does_not_grow(self):
        # a backdated / evening-skew today must never run decay in reverse (spore-081)
        assert effective_strength(1.0, "2026-06-12", "2026-06-02") == 1.0

    def test_accepts_full_iso_datetime(self):
        assert effective_strength(1.0, "2026-06-02T23:59:59Z", "2026-06-12T00:00:01Z") == pytest.approx(0.96 ** 10)


# -- Schema --

class TestSchema:
    def test_tables_created(self, tmp_path):
        s = _store(tmp_path)
        names = {r[0] for r in s._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert {"pattern_associations", "pattern_aliases", "processed_co_surface_events"} <= names
        s.close()

    def test_canonical_check_constraint(self, tmp_path):
        s = _store(tmp_path)
        with pytest.raises(Exception):
            s._conn.execute(
                "INSERT INTO pattern_associations (name_a, name_b, strength, first_linked_at, "
                "last_strengthened_at, last_co_surfaced_at, last_decay_at, formation_source) "
                "VALUES ('zzz','aaa',1.0,'2026-06-12','2026-06-12','2026-06-12','2026-06-12','x')")
        s.close()


# -- Co-graduation seeding --

class TestCoGraduationSeeding:
    def test_seeds_all_pairs(self, tmp_path):
        s = _store(tmp_path)
        n = s.seed_pattern_co_graduation(["a", "b", "c"], today="2026-06-12")
        assert n == 3  # ab, ac, bc
        assert _pair(s, "a", "b").strength == pytest.approx(SEED_STRENGTH)
        assert _pair(s, "a", "b").formation_source == "co_graduation"
        s.close()

    def test_does_not_overwrite_existing(self, tmp_path):
        s = _store(tmp_path)
        # strengthen a-b via co-surface first
        s.drain_co_surface_events(_events(("e1", "s1", ["a", "b"])), today="2026-06-12")
        before = _pair(s, "a", "b").strength
        s.seed_pattern_co_graduation(["a", "b", "c"], today="2026-06-12")
        # a-b keeps its (stronger) co-surface strength; only a-c / b-c are seeded
        assert _pair(s, "a", "b").strength == pytest.approx(before)
        assert _pair(s, "a", "c").strength == pytest.approx(SEED_STRENGTH)
        s.close()

    def test_dedups_and_ignores_singletons(self, tmp_path):
        s = _store(tmp_path)
        assert s.seed_pattern_co_graduation(["solo"], today="2026-06-12") == 0
        assert s.seed_pattern_co_graduation(["a", "a", "b"], today="2026-06-12") == 1  # one pair
        s.close()


# -- The idempotent drain --

class TestDrain:
    def test_forms_pairs_from_co_surface(self, tmp_path):
        s = _store(tmp_path)
        m = s.drain_co_surface_events(_events(("e1", "s1", ["a", "b", "c"])), today="2026-06-12")
        assert m["pairs_formed"] == 3  # ab, ac, bc
        assert _pair(s, "a", "b").formation_source == "co_surface"
        s.close()

    def test_burst_damp_is_per_session_log1p(self, tmp_path):
        s = _store(tmp_path)
        # a-b co-surfaces 3x in ONE session → damped to log1p(3), not 3x base
        s.drain_co_surface_events(_events(
            ("e1", "s1", ["a", "b"]), ("e2", "s1", ["a", "b"]), ("e3", "s1", ["a", "b"]),
        ), today="2026-06-12")
        assert _pair(s, "a", "b").strength == pytest.approx(CO_SURFACE_BASE * math.log1p(3))
        s.close()

    def test_distinct_sessions_each_count(self, tmp_path):
        s = _store(tmp_path)
        s.drain_co_surface_events(_events(
            ("e1", "s1", ["a", "b"]), ("e2", "s2", ["a", "b"]),
        ), today="2026-06-12")
        # two sessions, one co-surface each → 2 * base*log1p(1)
        assert _pair(s, "a", "b").strength == pytest.approx(2 * CO_SURFACE_BASE * math.log1p(1))
        assert _pair(s, "a", "b").co_surface_session_count == 2
        s.close()

    def test_idempotent_by_event_id(self, tmp_path):
        s = _store(tmp_path)
        ev = _events(("e1", "s1", ["a", "b"]))
        s.drain_co_surface_events(ev, today="2026-06-12")
        before = _pair(s, "a", "b").strength
        m2 = s.drain_co_surface_events(ev, today="2026-06-12")  # re-feed SAME event
        assert m2["events_applied"] == 0
        assert _pair(s, "a", "b").strength == pytest.approx(before)
        s.close()

    def test_singleton_and_empty_events_form_no_pairs(self, tmp_path):
        s = _store(tmp_path)
        m = s.drain_co_surface_events(_events(
            ("e1", "s1", ["solo"]), ("e2", "s1", []),
        ), today="2026-06-12")
        assert m["pairs_formed"] == 0
        # but the event_ids are still recorded (idempotency / "absence is unambiguous")
        assert m["events_seen"] == 2
        s.close()

    def test_within_batch_duplicate_event_id_counts_once(self, tmp_path):
        """Regression for codex/kimi L3: a duplicate event_id within ONE batch is
        deduped (the cross-drain idempotency table only covers committed ids)."""
        s = _store(tmp_path)
        ev = {"event_id": "dup", "session": "s1", "names": ["a", "b"], "basis": "keyword"}
        s.drain_co_surface_events([dict(ev), dict(ev)], today="2026-06-12")
        # counted ONCE → log1p(1), not log1p(2)
        assert _pair(s, "a", "b").strength == pytest.approx(CO_SURFACE_BASE * math.log1p(1))
        s.close()

    def test_graph_mediated_basis_does_not_reinforce(self, tmp_path):
        s = _store(tmp_path)
        # basis 'graph' is gated to 0 reinforcement (Slice-C echo-chamber guard)
        m = s.drain_co_surface_events(
            _events(("e1", "s1", ["a", "b"]), basis="graph"), today="2026-06-12")
        assert m["pairs_formed"] == 0  # zero-weight delta → no edge formed
        assert _pair(s, "a", "b") is None
        s.close()

    def test_strengthen_decays_existing_first(self, tmp_path):
        s = _store(tmp_path)
        s.drain_co_surface_events(_events(("e1", "s1", ["a", "b"])), today="2026-06-01")
        base = _pair(s, "a", "b", today="2026-06-01").stored_strength
        # 10 days later, strengthen again: the stored value decays to today FIRST,
        # then the new delta is added.
        s.drain_co_surface_events(_events(("e2", "s2", ["a", "b"])), today="2026-06-11")
        p = _pair(s, "a", "b", today="2026-06-11")
        expected = base * (DECAY_PER_DAY ** 10) + CO_SURFACE_BASE * math.log1p(1)
        assert p.stored_strength == pytest.approx(expected)
        s.close()


# -- Homeostatic normalization + GC --

class TestNormalizationAndGC:
    def test_node_outgoing_budget_caps_concentration(self, tmp_path):
        s = _store(tmp_path)
        # Hammer a hub node 'h' with many strong distinct-session co-surfaces.
        specs = []
        for i in range(40):
            specs.append((f"e{i}", f"sess{i}", ["h", f"leaf{i:02d}"]))
        s.drain_co_surface_events(_events(*specs), today="2026-06-12")
        # h's outgoing effective sum must be bounded by the budget (+ epsilon).
        total = sum(p.strength for p in s.get_pattern_associations(
            ["h"], today="2026-06-12", min_strength=0.0, limit=999))
        assert total <= NODE_OUTGOING_BUDGET + 1e-6
        s.close()

    def test_two_hubs_sharing_edge_both_bounded_single_pass(self, tmp_path):
        """Regression for L1 LOW-1 / L2 M1: two over-budget hubs that share an
        edge must each stay <= budget, with the shared edge scaled ONCE (by the
        min of the two factors), deterministically — not double-scaled by
        sequential per-node passes."""
        s = _store(tmp_path)
        specs = [(f"e{i}", f"s{i}", ["h1", f"a{i:02d}"]) for i in range(40)]
        specs += [(f"f{i}", f"t{i}", ["h2", f"b{i:02d}"]) for i in range(40)]
        specs += [(f"g{i}", f"u{i}", ["h1", "h2"]) for i in range(20)]  # strong shared edge
        s.drain_co_surface_events(_events(*specs), today="2026-06-12")
        for hub in ("h1", "h2"):
            total = sum(p.strength for p in s.get_pattern_associations(
                [hub], today="2026-06-12", min_strength=0.0, limit=999))
            assert total <= NODE_OUTGOING_BUDGET + 1e-6
        # deterministic: re-reading the same graph gives identical effective values
        a = [(p.name_a, p.name_b, round(p.strength, 6))
             for p in s.pattern_association_stats(today="2026-06-12", top_n=999).strongest_pairs]
        b = [(p.name_a, p.name_b, round(p.strength, 6))
             for p in s.pattern_association_stats(today="2026-06-12", top_n=999).strongest_pairs]
        assert a == b
        s.close()

    def test_gc_deletes_decayed_edge_on_touch(self, tmp_path):
        s = _store(tmp_path)
        # weak seed, then let it decay far below GC threshold, then touch a NEIGHBOR
        s.seed_pattern_co_graduation(["x", "y"], today="2026-01-01")  # 0.3
        # 0.3 * 0.96^days < 0.03 → days > ln(0.1)/ln(0.96) ≈ 56 days
        # touch node x via a new co-surface ~200 days later → opportunistic GC fires on x-y
        s.drain_co_surface_events(_events(("e1", "s1", ["x", "z"])), today="2026-07-20")
        assert _pair(s, "x", "y", today="2026-07-20") is None  # GC'd
        s.close()

    def test_retrieval_threshold_separate_from_gc(self, tmp_path):
        s = _store(tmp_path)
        s.seed_pattern_co_graduation(["x", "y"], today="2026-06-01")  # 0.3 stored
        # ~25 days: 0.3*0.96^25 ≈ 0.108 → above GC (0.03), below... no, above retrieval (0.2)? 0.108<0.2
        # pick a day where effective is between GC and retrieval
        eff = effective_strength(SEED_STRENGTH, "2026-06-01", "2026-06-26")
        assert GC_THRESHOLD < eff < RETRIEVAL_THRESHOLD
        # get_pattern_associations (min_strength=retrieval) hides it...
        assert s.get_pattern_associations(["x"], today="2026-06-26", min_strength=RETRIEVAL_THRESHOLD) == []
        # ...but it is NOT gc'd (still in the table, surfaces at min_strength=0)
        assert s.get_pattern_associations(["x"], today="2026-06-26", min_strength=0.0)
        s.close()


# -- Lifecycle: rename + homonym guard --

class TestRename:
    def test_carries_edges_to_new_name(self, tmp_path):
        s = _store(tmp_path)
        s.seed_pattern_co_graduation(["a", "b", "c"], today="2026-06-12")
        rk = s.rename_pattern_association("a", "z", today="2026-06-12")
        assert rk == 2  # a-b, a-c
        assert _pair(s, "a", "b") is None
        assert _pair(s, "z", "b") is not None
        assert _pair(s, "z", "c") is not None
        s.close()

    def test_merge_on_collision_and_drop_self_pair(self, tmp_path):
        s = _store(tmp_path)
        s.seed_pattern_co_graduation(["a", "b", "c"], today="2026-06-12")  # ab, ac, bc
        # rename b -> c: b-c becomes a self-pair (dropped); a-b merges into a-c
        s.rename_pattern_association("b", "c", today="2026-06-12")
        remaining = {(p.name_a, p.name_b) for p in s.pattern_association_stats(today="2026-06-12").strongest_pairs}
        assert remaining == {("a", "c")}
        s.close()

    def test_records_alias(self, tmp_path):
        s = _store(tmp_path)
        s.seed_pattern_co_graduation(["a", "b"], today="2026-06-12")
        s.rename_pattern_association("a", "z", today="2026-06-12")
        row = s._conn.execute(
            "SELECT canonical_name, kind FROM pattern_aliases WHERE alias='a'").fetchone()
        assert (row["canonical_name"], row["kind"]) == ("z", "rename")
        s.close()

    def test_preserves_first_linked_at_on_insert_branch(self, tmp_path):
        """Regression for L2 H1: the no-collision INSERT branch must PRESERVE the
        original first_linked_at, not reset it to `today`. The base suite missed
        this because seeded edges have all dates equal."""
        s = _store(tmp_path)
        s.seed_pattern_co_graduation(["a", "b"], today="2026-01-01")  # first_linked = 2026-01-01
        # strengthen 2 months later → last_* advance, first_linked stays 2026-01-01
        s.drain_co_surface_events(_events(("e1", "s1", ["a", "b"])), today="2026-03-01")
        # rename to a fresh name (no collision → INSERT branch) much later
        s.rename_pattern_association("a", "z", today="2026-06-01")
        p = _pair(s, "z", "b", today="2026-06-01")
        assert p.first_linked_at == "2026-01-01"  # preserved, NOT 2026-06-01
        s.close()


class TestHomonymGuard:
    def test_sever_deletes_edges_and_bumps_generation(self, tmp_path):
        s = _store(tmp_path)
        s.seed_pattern_co_graduation(["a", "b", "c"], today="2026-06-12")
        severed = s.sever_pattern_concept("a", today="2026-06-12")
        assert severed == 2
        assert _pair(s, "a", "b") is None
        assert _pair(s, "a", "c") is None
        assert _pair(s, "b", "c") is not None  # untouched
        assert _current_generation(s._conn, "a") == 2
        s.close()

    def test_reused_name_starts_clean_after_sever(self, tmp_path):
        s = _store(tmp_path)
        s.seed_pattern_co_graduation(["a", "b"], today="2026-06-12")
        s.sever_pattern_concept("a", today="2026-06-12")  # concept-A leaves
        # concept-B reuses the name 'a' — forms fresh edges, no contamination
        s.drain_co_surface_events(_events(("e1", "s1", ["a", "d"])), today="2026-06-13")
        assert _pair(s, "a", "b") is None  # old concept's edge stayed severed
        assert _pair(s, "a", "d") is not None  # new concept's edge
        s.close()


# -- Telemetry --

class TestStats:
    def test_concentration_and_sources(self, tmp_path):
        s = _store(tmp_path)
        s.seed_pattern_co_graduation(["a", "b"], today="2026-06-12")  # co_graduation
        s.drain_co_surface_events(_events(("e1", "s1", ["c", "d"])), today="2026-06-12")  # co_surface
        st = s.pattern_association_stats(today="2026-06-12")
        assert st.total_links == 2
        assert st.by_formation_source == {"co_graduation": 1, "co_surface": 1}
        assert 0.0 < st.concentration <= 1.0
        s.close()

    def test_retrievable_count_uses_effective(self, tmp_path):
        s = _store(tmp_path)
        s.seed_pattern_co_graduation(["a", "b"], today="2026-06-01")  # 0.3 > 0.2 today
        st_now = s.pattern_association_stats(today="2026-06-01")
        assert st_now.retrievable_links == 1
        # 30 days later it has decayed below the retrieval threshold
        st_later = s.pattern_association_stats(today="2026-07-01")
        assert st_later.total_links == 1
        assert st_later.retrievable_links == 0
        s.close()


# -- Store integration: read_only rejection + audit --

class TestStoreIntegration:
    def test_read_only_rejects_writes(self, tmp_path):
        s = _store(tmp_path)
        s.seed_pattern_co_graduation(["a", "b"], today="2026-06-12")
        s.close()
        ro = Store(str(tmp_path / "m.db"), read_only=True)
        # reads work
        assert ro.pattern_association_stats(today="2026-06-12").total_links == 1
        # writes are rejected (query_only)
        with pytest.raises(Exception):
            ro.drain_co_surface_events(_events(("e1", "s1", ["a", "b"])), today="2026-06-12")
        ro.close()


# -- End-to-end: co-graduation seeding through the real wrap pipeline --

class TestWrapPipelineSeeding:
    def test_two_graduations_seed_a_pattern_link(self, tmp_path):
        from anneal_memory.continuity import validated_save_continuity, prepare_wrap

        s = Store(str(tmp_path / "m.db"), project_name="test")
        # two episodes that two patterns will cite as fresh evidence today
        e1 = s.record("first grounding episode about the alpha mechanism", "observation").id
        e2 = s.record("second grounding episode about the beta mechanism", "observation").id
        today = "2026-06-12"
        text = (
            "## State\n\ncurrent focus.\n\n"
            "## Patterns\n\n"
            f"- alpha_mechanism_pattern | 2x ({today}) [evidence: {e1} \"first grounding episode alpha mechanism\"]\n"
            f"- beta_mechanism_pattern | 2x ({today}) [evidence: {e2} \"second grounding episode beta mechanism\"]\n"
            "\n## Decisions\n\nnone.\n"
            "\n## Context\n\nwrap.\n"
        )
        prep = prepare_wrap(s)
        validated_save_continuity(s, text, today=today, wrap_token=prep["wrap_token"])
        # both graduated this wrap → a co-graduation seed links them
        p = _pair(s, "alpha_mechanism_pattern", "beta_mechanism_pattern", today=today)
        assert p is not None
        assert p.formation_source == "co_graduation"
        assert p.strength == pytest.approx(SEED_STRENGTH)
        s.close()

    def test_seeding_fault_does_not_roll_back_wrap_hebbian_dml(self, tmp_path):
        """Regression for codex L3 HIGH-1: co-graduation seeding now runs as a
        POST-COMMIT separate transaction, so a seeding fault must NOT roll back
        the wrap's episode-Hebbian DML (which inside the batch a swallowed fault
        would have discarded via _db_boundary rollback) and the wrap must still
        succeed."""
        from anneal_memory.continuity import validated_save_continuity, prepare_wrap

        s = Store(str(tmp_path / "m.db"), project_name="test")
        e1 = s.record("episode one about alpha grounding", "observation").id
        e2 = s.record("episode two about alpha grounding", "observation").id
        e3 = s.record("episode three about beta grounding", "observation").id
        e4 = s.record("episode four about beta grounding", "observation").id
        today = "2026-06-12"
        # two graduations, each co-citing 2 episodes → forms episode Hebbian links
        # AND triggers co-graduation seeding (2+ graduated names)
        text = (
            "## State\n\nfocus.\n\n## Patterns\n\n"
            f"- alpha_pattern | 2x ({today}) [evidence: {e1}, {e2} \"episode alpha grounding\"]\n"
            f"- beta_pattern | 2x ({today}) [evidence: {e3}, {e4} \"episode beta grounding\"]\n"
            "\n## Decisions\n\nnone.\n\n## Context\n\nwrap.\n"
        )

        def _boom(*a, **k):
            raise RuntimeError("seeding blew up")

        s.seed_pattern_co_graduation = _boom  # type: ignore[method-assign]
        prep = prepare_wrap(s)
        result = validated_save_continuity(s, text, today=today, wrap_token=prep["wrap_token"])
        assert result is not None  # wrap succeeded despite the seeding fault
        # the episode Hebbian DML survived (was NOT rolled back by the fault)
        assert s.association_stats().total_links > 0
        s.close()
