"""Tests for the unified on-demand recall surface (retrieval.py, AM-CRYSTAL-RECALL).

Covers keyword extraction, the precision-biased weighted-overlap scoring over BOTH
crystallized patterns and episodes, ranking, caps, the high-signal type boost, the
recent-exclusion cutoff (incl. fail-open on a bad clock), and an L4-style proof that
a recall hook can consume the result with zero adapter (the v2-consumer contract).
"""

from __future__ import annotations

from datetime import date

import pytest

from anneal_memory import (
    CrystalStore,
    RelevantResult,
    Store,
    extract_keywords,
    retrieve_patterns,
    retrieve_relevant,
)
from anneal_memory.types import EpisodeType
from anneal_memory import retrieval as _retrieval


T0 = date(2026, 6, 6)
# A fixed "now" well after the seeded episodes so nothing is recent-excluded by default.
NOW = "2026-06-06T18:00:00Z"


@pytest.fixture
def stores(tmp_path):
    store = Store(tmp_path / "mem.db")
    crystal = CrystalStore(tmp_path / "mem.crystal.json")
    return store, crystal


def _seed_episodes(store):
    # Content ≥ MIN_EPISODE_LEN (80) and carrying ≥2 distinctive keywords each.
    store.record(
        "Resolved the apparatus routing drift: the cross_substrate review path could "
        "silently skip the non-replaceable codex reviewer for state-machine code.",
        EpisodeType.DECISION, source="flow", timestamp="2026-06-06T09:00:00Z",
    )
    store.record(
        "Observed that grinding breaks productivity while play sustains the trickster "
        "engine across long build sessions — energy is the real constraint here.",
        EpisodeType.OBSERVATION, source="flow", timestamp="2026-06-05T09:00:00Z",
    )
    store.record(
        "A short note.", EpisodeType.OBSERVATION, source="flow",
        timestamp="2026-06-05T10:00:00Z",
    )


def _seed_crystal(crystal):
    crystal.crystallize(
        name="structural_invariants_beat_discipline", level=3,
        explanation="discipline drifts then is bypassed; an invariant refuses",
        tags=["apparatus", "verification"], today=T0,
    )
    crystal.crystallize(
        name="play_first", level=2,
        explanation="grinding breaks productivity; play is the engine",
        tags=["cognition"], today=T0,
    )
    crystal.crystallize(
        name="unrelated_pattern", level=2,
        explanation="something about sourdough baking and bread proofing entirely",
        tags=["food"], today=T0,
    )


# -- keyword extraction ----------------------------------------------------


class TestExtractKeywords:
    def test_drops_stopwords_and_short(self):
        kws = extract_keywords("the apparatus is a routing drift")
        assert "apparatus" in kws and "routing" in kws and "drift" in kws
        assert "the" not in kws and "is" not in kws and "a" not in kws

    def test_preserves_snake_case_and_hyphen(self):
        kws = extract_keywords("the cross_substrate review and non-replaceable codex")
        assert "cross_substrate" in kws
        assert "non-replaceable" in kws

    def test_dedup_order_preserving(self):
        kws = extract_keywords("codex codex apparatus codex")
        assert kws == ["codex", "apparatus"]

    def test_caps_breadth(self):
        many = " ".join(f"term{i}word" for i in range(40))
        assert len(extract_keywords(many)) <= _retrieval.MAX_KEYWORDS


# -- retrieve_relevant: precision bias -------------------------------------


class TestPrecision:
    def test_too_few_keywords_returns_empty(self, stores):
        store, crystal = stores
        _seed_episodes(store)
        _seed_crystal(crystal)
        r = retrieve_relevant(store, crystal, "apparatus", now=NOW, today=T0)  # 1 keyword
        assert r.patterns == [] and r.episodes == []
        assert r.query_keywords == ["apparatus"]

    def test_irrelevant_query_surfaces_nothing(self, stores):
        store, crystal = stores
        _seed_episodes(store)
        _seed_crystal(crystal)
        r = retrieve_relevant(store, crystal, "quantum chromodynamics lagrangian", now=NOW, today=T0)
        assert r.patterns == [] and r.episodes == []

    def test_short_episode_excluded(self, stores):
        store, crystal = stores
        _seed_episodes(store)  # "A short note." is < MIN_EPISODE_LEN
        r = retrieve_relevant(store, None, "short note here", now=NOW, today=T0)
        assert all("short note" not in e.content.lower() or len(e.content) >= 80 for e in r.episodes)


# -- retrieve_relevant: patterns -------------------------------------------


class TestPatterns:
    def test_snake_case_name_substring_match(self, stores):
        store, crystal = stores
        _seed_crystal(crystal)
        # query words match INSIDE the snake_case name as substrings
        r = retrieve_relevant(store, crystal, "structural invariants discipline", now=NOW, today=T0)
        names = [p.name for p in r.patterns]
        assert "structural_invariants_beat_discipline" in names

    def test_pattern_fields_and_activation(self, stores):
        store, crystal = stores
        _seed_crystal(crystal)
        r = retrieve_relevant(store, crystal, "grinding productivity play", now=NOW, today=T0)
        p = next(p for p in r.patterns if p.name == "play_first")
        assert p.level == 2
        assert p.activation == "hot"  # crystallized today
        assert p.score > 0 and "cognition" in p.tags

    def test_ranked_score_then_level(self, stores):
        store, crystal = stores
        # two patterns that both match "apparatus verification" — 3x should win ties
        crystal.crystallize(name="a_pat", level=2, explanation="apparatus verification matters", tags=[], today=T0)
        crystal.crystallize(name="b_pat", level=3, explanation="apparatus verification matters", tags=[], today=T0)
        r = retrieve_relevant(store, crystal, "apparatus verification", now=NOW, today=T0,
                              max_patterns=5)
        # equal score → higher level first
        assert r.patterns[0].level == 3

    def test_max_patterns_cap(self, stores):
        store, crystal = stores
        for i in range(5):
            crystal.crystallize(name=f"pat_{i}", level=2,
                               explanation="apparatus verification routing codex", today=T0)
        r = retrieve_relevant(store, crystal, "apparatus verification routing", now=NOW, today=T0,
                              max_patterns=2)
        assert len(r.patterns) == 2

    def test_no_crystal_store_means_no_patterns(self, stores):
        store, _ = stores
        _seed_episodes(store)
        r = retrieve_relevant(store, None, "apparatus routing codex", now=NOW, today=T0)
        assert r.patterns == []
        assert len(r.episodes) >= 1  # episodes still surface


# -- retrieve_relevant: episodes -------------------------------------------


class TestEpisodes:
    def test_high_signal_type_boost(self, stores):
        store, crystal = stores
        # identical content (≥80 chars), different type — DECISION outscores OBSERVATION
        body = ("apparatus routing drift codex review path matters a great deal here "
                "indeed and at length too.")
        store.record(body, EpisodeType.OBSERVATION, timestamp="2026-06-05T08:00:00Z")
        store.record(body, EpisodeType.DECISION, timestamp="2026-06-05T08:00:01Z")
        r = retrieve_relevant(store, None, "apparatus routing codex", now=NOW, today=T0, max_episodes=5)
        assert r.episodes[0].type == "decision"
        assert r.episodes[0].score > r.episodes[1].score

    def test_ranked_by_score_then_recency(self, stores):
        store, crystal = stores
        _seed_episodes(store)
        r = retrieve_relevant(store, None, "apparatus routing cross_substrate codex", now=NOW, today=T0)
        assert len(r.episodes) >= 1
        scores = [e.score for e in r.episodes]
        assert scores == sorted(scores, reverse=True)

    def test_exclude_recent_minutes(self, stores):
        store, crystal = stores
        # episode at 17:30, now 18:00 → within a 45-min exclusion window
        store.record("apparatus routing drift codex review path is the live session echo right "
                     "here and now in this very session.",
                     EpisodeType.DECISION, timestamp="2026-06-06T17:30:00Z")
        # without exclusion: surfaces
        r1 = retrieve_relevant(store, None, "apparatus routing codex", now=NOW, today=T0)
        assert len(r1.episodes) == 1
        # with a 45-min exclusion: dropped
        r2 = retrieve_relevant(store, None, "apparatus routing codex", now=NOW, today=T0,
                               exclude_recent_minutes=45)
        assert r2.episodes == []

    def test_max_episodes_cap(self, stores):
        store, crystal = stores
        for i in range(5):
            store.record(f"apparatus routing drift codex review number {i} with plenty of extra "
                         f"descriptive text padding this out past the floor.",
                         EpisodeType.DECISION, timestamp=f"2026-06-05T0{i}:00:00Z")
        r = retrieve_relevant(store, None, "apparatus routing codex", now=NOW, today=T0, max_episodes=2)
        assert len(r.episodes) == 2


# -- recent cutoff helper --------------------------------------------------


class TestRecentCutoff:
    def test_none_when_disabled(self):
        assert _retrieval._recent_cutoff(None, NOW) is None
        assert _retrieval._recent_cutoff(0, NOW) is None

    def test_bad_now_fails_open_to_none(self):
        # fail-open: a bogus clock must NOT produce a cutoff that drops all episodes
        assert _retrieval._recent_cutoff(45, "not-a-timestamp") is None

    def test_valid_cutoff_subtracts(self):
        cut = _retrieval._recent_cutoff(60, "2026-06-06T18:00:00Z")
        assert cut == "2026-06-06T17:00:00Z"

    def test_naive_now_assumed_utc(self):
        cut = _retrieval._recent_cutoff(60, "2026-06-06T18:00:00")
        assert cut == "2026-06-06T17:00:00Z"


# -- L4: the v2-consumer contract (zero-adapter hook usage) ----------------


class TestHookConsumerContract:
    def test_hook_can_render_with_zero_adapter(self, stores):
        """Mimic a recall hook: call ONE function, render both kinds into a block.
        Proves the RelevantResult shape carries everything the hook needs."""
        store, crystal = stores
        _seed_episodes(store)
        _seed_crystal(crystal)
        r = retrieve_relevant(store, crystal, "apparatus routing cross_substrate codex", now=NOW, today=T0)
        assert isinstance(r, RelevantResult)

        # the kind of formatting flow's recall_injection_hook does — patterns first
        lines = []
        for p in r.patterns:
            lines.append(f"[pattern {p.level}x/{p.activation}] {p.name}: {p.explanation}")
        for e in r.episodes:
            lines.append(f"[{e.timestamp[:10]}] {e.source}/{e.type} ({e.id}): {e.content[:80]}")
        block = "\n".join(lines)
        # the apparatus-routing decision episode is the strongest episode signal
        assert "apparatus" in block.lower()
        # every surfaced item exposes a numeric score for thresholding/measurement
        assert all(isinstance(p.score, float) for p in r.patterns)
        assert all(isinstance(e.score, float) for e in r.episodes)

    def test_determinism(self, stores):
        store, crystal = stores
        _seed_episodes(store)
        _seed_crystal(crystal)
        a = retrieve_relevant(store, crystal, "apparatus routing codex", now=NOW, today=T0)
        b = retrieve_relevant(store, crystal, "apparatus routing codex", now=NOW, today=T0)
        assert [p.name for p in a.patterns] == [p.name for p in b.patterns]
        assert [e.id for e in a.episodes] == [e.id for e in b.episodes]


# -- retrieve_patterns: the patterns-only entry (no episodic Store) ---------
class TestRetrievePatterns:
    def test_parity_with_retrieve_relevant_patterns(self, stores):
        """The contract: retrieve_patterns(crystal, q) is byte-for-byte the pattern
        half of retrieve_relevant(..., max_episodes=0) — so a hook can switch to it
        without changing what surfaces, while constructing no episodic Store."""
        store, crystal = stores
        _seed_crystal(crystal)
        q = "structural invariants discipline verification apparatus"
        only = retrieve_patterns(crystal, q, today=T0)
        full = retrieve_relevant(store, crystal, q, max_episodes=0, today=T0).patterns

        # Total field tuple — a forever-public parity guarantee must cover EVERY
        # RelevantPattern field, incl. explanation + tags (a future _score_patterns
        # refactor could otherwise silently diverge on a content field) AND source (both
        # paths are keyword-only here → both must tag "keyword").
        def _tot(p):
            return (p.name, p.level, p.score, p.activation, p.explanation, tuple(p.tags), p.source)
        assert [_tot(p) for p in only] == [_tot(p) for p in full]
        assert only  # non-empty — the query genuinely matches seeded patterns

    def test_needs_no_store(self, tmp_path):
        """The whole point: works from a CrystalStore alone, no Store anywhere."""
        crystal = CrystalStore(tmp_path / "mem.crystal.json")
        _seed_crystal(crystal)
        out = retrieve_patterns(crystal, "structural invariants discipline", today=T0)
        names = [p.name for p in out]
        assert "structural_invariants_beat_discipline" in names

    def test_none_crystal_store_returns_empty(self):
        assert retrieve_patterns(None, "structural invariants discipline", today=T0) == []

    def test_too_few_keywords_returns_empty(self, stores):
        _, crystal = stores
        _seed_crystal(crystal)
        assert retrieve_patterns(crystal, "apparatus", today=T0) == []  # 1 keyword < MIN_KEYWORDS

    def test_nonpositive_cap_returns_empty(self, stores):
        _, crystal = stores
        _seed_crystal(crystal)
        assert retrieve_patterns(crystal, "structural invariants discipline", max_patterns=0, today=T0) == []

    def test_irrelevant_query_surfaces_nothing(self, stores):
        _, crystal = stores
        _seed_crystal(crystal)
        assert retrieve_patterns(crystal, "quantum chromodynamics lagrangian", today=T0) == []

    def test_max_patterns_cap_and_ranking(self, tmp_path):
        crystal = CrystalStore(tmp_path / "mem.crystal.json")
        crystal.crystallize(name="a_pat", level=2, explanation="apparatus verification routing", tags=[], today=T0)
        crystal.crystallize(name="b_pat", level=3, explanation="apparatus verification routing", tags=[], today=T0)
        crystal.crystallize(name="c_pat", level=2, explanation="apparatus verification routing", tags=[], today=T0)
        out = retrieve_patterns(crystal, "apparatus verification routing", max_patterns=2, today=T0)
        assert len(out) == 2
        assert out[0].level == 3  # equal score → higher graduation level wins the tie

    def test_malformed_tags_coerced_not_crashed(self, tmp_path):
        """A hand-edited store with non-str / non-list tags must NOT raise a raw
        TypeError from the read path, and RelevantPattern.tags stays list[str]
        (codex L3). Structural store corruption still raises upstream in _load — this
        only guards a malformed FIELD inside an otherwise-valid row."""
        import json
        p = tmp_path / "mem.crystal.json"
        crystal = CrystalStore(p)
        crystal.crystallize(name="int_tag_pat", level=2,
                            explanation="apparatus verification routing", tags=["keep"], today=T0)
        crystal.crystallize(name="bare_tag_pat", level=2,
                            explanation="apparatus verification routing", tags=["x"], today=T0)
        data = json.loads(p.read_text())
        for row in data["crystal"]:
            if row["name"] == "int_tag_pat":
                row["tags"] = [1, "keep", 2]    # mixed int/str — non-str members dropped
            elif row["name"] == "bare_tag_pat":
                row["tags"] = "apparatus"        # bare string — ONE tag, not char-split
        p.write_text(json.dumps(data))

        out = retrieve_patterns(crystal, "apparatus verification routing", max_patterns=10, today=T0)
        by_name = {pp.name: pp for pp in out}
        assert all(isinstance(t, str) for pp in out for t in pp.tags)  # no list[int] leak
        assert by_name["int_tag_pat"].tags == ["keep"]
        assert by_name["bare_tag_pat"].tags == ["apparatus"]  # NOT "a p p a r a t u s"


# -- associative (Hebbian) pattern retrieval -------------------------------


class TestAssociativeRetrieval:
    """The Hebbian backend: a pattern grounded in a keyword-matched episode (or one
    co-cited with it) surfaces even with ZERO query-keyword overlap with the pattern's
    own compressed text — the keyword-orthogonal miss Step C measured. Precision is
    inherited from the episode tier: no episode match → no associative pattern."""

    # An episode the query matches strongly; its DISTINCTIVE keywords (master_plan /
    # document / drifted / reality / scheduler) appear in NONE of the pattern texts
    # below — so a keyword pass over the patterns scores 0 and the link is the only
    # path. ≥ MIN_EPISODE_LEN, ≥ 2 distinctive keywords.
    _DRIFT_EPISODE = (
        "The master_plan document drifted from reality after the retired scheduler "
        "file kept being referenced by eleven downstream readers."
    )
    _QUERY = "why did the master_plan document drift from reality"

    def test_keyword_orthogonal_pattern_surfaces_via_evidence(self, stores):
        store, crystal = stores
        e = store.record(
            self._DRIFT_EPISODE, EpisodeType.DECISION, source="flow",
            timestamp="2026-06-06T09:00:00Z",
        )
        crystal.crystallize(
            name="invisible_infrastructure_failure", level=3,
            explanation="healthy by surface, broken by structure; a feature inert across many surfaces",
            evidence=[e.id], tags=["substrate"], today=T0,
        )
        # keyword-only: the pattern's compressed text shares no query keyword → missed.
        kw = retrieve_relevant(store, crystal, self._QUERY, now=NOW, today=T0, associative=False)
        assert "invisible_infrastructure_failure" not in {p.name for p in kw.patterns}
        # associative: surfaces via the evidence edge to the matched episode.
        a = retrieve_relevant(store, crystal, self._QUERY, now=NOW, today=T0, associative=True)
        by_name = {p.name: p for p in a.patterns}
        assert "invisible_infrastructure_failure" in by_name
        # ... and its provenance is the DIRECT evidence edge (the pattern cites the
        # episode the query keyword-matched), not a graph hop.
        assert by_name["invisible_infrastructure_failure"].source == "evidence_edge"

    def test_associative_is_on_by_default(self, stores):
        store, crystal = stores
        e = store.record(
            self._DRIFT_EPISODE, EpisodeType.DECISION, source="flow",
            timestamp="2026-06-06T09:00:00Z",
        )
        crystal.crystallize(
            name="invisible_infrastructure_failure", level=3,
            explanation="healthy by surface, broken by structure; a feature inert across many surfaces",
            evidence=[e.id], today=T0,
        )
        r = retrieve_relevant(store, crystal, self._QUERY, now=NOW, today=T0)  # no flag
        assert "invisible_infrastructure_failure" in {p.name for p in r.patterns}

    def test_precision_inherited_no_episode_match_no_pattern(self, stores):
        store, crystal = stores
        e = store.record(
            self._DRIFT_EPISODE, EpisodeType.DECISION, source="flow",
            timestamp="2026-06-06T09:00:00Z",
        )
        crystal.crystallize(
            name="invisible_infrastructure_failure", level=3,
            explanation="healthy by surface, broken by structure; a feature inert",
            evidence=[e.id], today=T0,
        )
        # Query matches NO episode → no seed → associative reach is empty.
        r = retrieve_relevant(
            store, crystal, "quantum chromodynamics lagrangian gauge symmetry", now=NOW, today=T0
        )
        assert r.patterns == []

    def test_hebbian_hop_reaches_co_cited_episode_pattern(self, stores):
        store, crystal = stores
        e_match = store.record(
            self._DRIFT_EPISODE, EpisodeType.DECISION, source="flow",
            timestamp="2026-06-06T09:00:00Z",
        )
        e_other = store.record(
            "A separate consolidation about governance ledgers, audit chains, and "
            "provenance proofs across the precedent registry tier.",
            EpisodeType.DECISION, source="flow", timestamp="2026-06-06T09:05:00Z",
        )
        # Strongly associate the two episodes (3 direct co-citations ≥ ASSOC_STRENGTH_NORM).
        for _ in range(3):
            store.record_associations(direct_pairs={(e_match.id, e_other.id)})
        # The pattern is grounded ONLY in e_other and shares no query keyword.
        crystal.crystallize(
            name="memory_is_governance", level=3,
            explanation="audit ledger chains prove provenance across the registry",
            evidence=[e_other.id], today=T0,
        )
        r = retrieve_relevant(store, crystal, self._QUERY, now=NOW, today=T0)
        # e_other is reached via the strong Hebbian hop from the matched seed e_match.
        by_name = {p.name: p for p in r.patterns}
        assert "memory_is_governance" in by_name
        # ... and its provenance is the graph hop (the pattern's evidence episode was
        # reached only THROUGH the Hebbian link, not by a direct keyword match).
        assert by_name["memory_is_governance"].source == "graph_hop"

    def test_weak_hop_alone_does_not_clear_gate(self, stores):
        store, crystal = stores
        e_match = store.record(
            self._DRIFT_EPISODE, EpisodeType.DECISION, source="flow",
            timestamp="2026-06-06T09:00:00Z",
        )
        e_other = store.record(
            "A separate consolidation about governance ledgers, audit chains, and "
            "provenance proofs across the precedent registry tier.",
            EpisodeType.OBSERVATION, source="flow", timestamp="2026-06-06T09:05:00Z",
        )
        # A single SESSION co-citation = weak link (0.3) → the one-hop reach is far
        # below the precision gate, so a pattern grounded only there must NOT surface.
        store.record_associations(direct_pairs=set(), session_pairs={(e_match.id, e_other.id)})
        crystal.crystallize(
            name="memory_is_governance", level=3,
            explanation="audit ledger chains prove provenance across the registry",
            evidence=[e_other.id], today=T0,
        )
        r = retrieve_relevant(store, crystal, self._QUERY, now=NOW, today=T0)
        assert "memory_is_governance" not in {p.name for p in r.patterns}

    def test_no_duplicate_when_both_keyword_and_associative(self, stores):
        store, crystal = stores
        e = store.record(
            "The apparatus routing drift let the cross_substrate review path skip the "
            "non-replaceable codex reviewer on state-machine code.",
            EpisodeType.DECISION, source="flow", timestamp="2026-06-06T09:00:00Z",
        )
        # Pattern keyword-matches the query AND cites the matched episode → must appear ONCE.
        crystal.crystallize(
            name="apparatus_routing", level=3,
            explanation="the apparatus routing drift on the cross_substrate review path",
            evidence=[e.id], tags=["apparatus"], today=T0,
        )
        r = retrieve_relevant(
            store, crystal, "apparatus routing drift cross_substrate review", now=NOW, today=T0
        )
        names = [p.name for p in r.patterns]
        assert names.count("apparatus_routing") == 1

    def test_union_respects_max_patterns_cap(self, stores):
        store, crystal = stores
        e = store.record(
            self._DRIFT_EPISODE, EpisodeType.DECISION, source="flow",
            timestamp="2026-06-06T09:00:00Z",
        )
        for i in range(4):
            crystal.crystallize(
                name=f"orthogonal_pattern_{i}", level=3,
                explanation="healthy by surface broken by structure inert feature substrate",
                evidence=[e.id], today=T0,
            )
        r = retrieve_relevant(store, crystal, self._QUERY, now=NOW, today=T0, max_patterns=2)
        assert len(r.patterns) == 2

    def test_corrupt_evidence_field_is_skipped(self, stores, tmp_path):
        store, crystal = stores
        e = store.record(
            self._DRIFT_EPISODE, EpisodeType.DECISION, source="flow",
            timestamp="2026-06-06T09:00:00Z",
        )
        crystal.crystallize(
            name="invisible_infrastructure_failure", level=3,
            explanation="healthy by surface, broken by structure; a feature inert",
            evidence=[e.id], today=T0,
        )
        # Hand-corrupt the evidence field to a non-list — the read path must not crash.
        import json
        data = json.loads(crystal.path.read_text())
        data["crystal"][0]["evidence"] = "not-a-list"
        crystal.path.write_text(json.dumps(data))
        r = retrieve_relevant(store, crystal, self._QUERY, now=NOW, today=T0)
        assert "invisible_infrastructure_failure" not in {p.name for p in r.patterns}

    def test_evidence_idf_prevents_hub_episode_flood(self, stores):
        store, crystal = stores
        hub = store.record(
            "The apparatus routing drift in the cross_substrate review path skipped "
            "the codex reviewer on the state-machine code path entirely.",
            EpisodeType.DECISION, source="flow", timestamp="2026-06-06T09:00:00Z",
        )
        distinct = store.record(
            "A distinct governance ledger about provenance chains and audit trails "
            "across the precedent registry tier in this consolidation.",
            EpisodeType.DECISION, source="flow", timestamp="2026-06-06T09:05:00Z",
        )
        # 5 patterns cite ONLY the hub episode; 1 cites the distinctively-cited one.
        for i in range(5):
            crystal.crystallize(
                name=f"hub_citer_{i}", level=3,
                explanation="unrelated distilled wisdom inert across surfaces here",
                evidence=[hub.id], today=T0,
            )
        crystal.crystallize(
            name="distinct_citer", level=3,
            explanation="separate distilled wisdom inert across surfaces here",
            evidence=[distinct.id], today=T0,
        )
        # Query brushes BOTH episodes at the minimum (2 keyword hits each).
        names = {
            p.name for p in retrieve_relevant(
                store, crystal, "apparatus routing governance ledger",
                now=NOW, today=T0, max_patterns=10,
            ).patterns
        }
        # The hub episode (cited by 5 patterns) is IDF-down-weighted below the gate →
        # none of the hub-only citers flood; the distinctively-cited pattern surfaces.
        assert not any(n.startswith("hub_citer_") for n in names)
        assert "distinct_citer" in names

    def test_keyword_pattern_not_displaced_by_associative(self, stores):
        store, crystal = stores
        e = store.record(
            "The apparatus routing drift on the cross_substrate review path with the "
            "codex reviewer skipped on state-machine code here.",
            EpisodeType.DECISION, source="flow", timestamp="2026-06-06T09:00:00Z",
        )
        # A keyword-matching pattern (matches the query on its OWN text)...
        crystal.crystallize(
            name="apparatus_keyword_hit", level=2,
            explanation="the apparatus routing review path discipline",
            today=T0,
        )
        # ...and an associative-only pattern grounded in the matched episode.
        crystal.crystallize(
            name="orthogonal_assoc", level=3,
            explanation="totally separate distilled wisdom about proofing sourdough",
            evidence=[e.id], today=T0,
        )
        out = retrieve_relevant(
            store, crystal, "apparatus routing review", now=NOW, today=T0, max_patterns=1
        )
        # 1 slot, keyword-first: the keyword hit takes it; associative can't displace it.
        assert [p.name for p in out.patterns] == ["apparatus_keyword_hit"]

    def test_max_aggregation_not_sum(self, stores):
        store, crystal = stores
        e1 = store.record(
            "The apparatus routing concern on the cross_substrate review path here "
            "with assorted detail to clear the length floor comfortably.",
            EpisodeType.OBSERVATION, source="flow", timestamp="2026-06-06T09:00:00Z",
        )
        e2 = store.record(
            "A governance ledger about provenance and audit trails across the registry "
            "tier with assorted detail to clear the length floor comfortably.",
            EpisodeType.OBSERVATION, source="flow", timestamp="2026-06-06T09:05:00Z",
        )
        # Pattern cites BOTH matched episodes; under SUM it'd score ~5, under MAX ~2.5.
        crystal.crystallize(
            name="multi_cite", level=3,
            explanation="separate distilled wisdom inert across surfaces here",
            evidence=[e1.id, e2.id], today=T0,
        )
        out = retrieve_relevant(
            store, crystal, "apparatus routing governance ledger", now=NOW, today=T0
        )
        p = next(pp for pp in out.patterns if pp.name == "multi_cite")
        # Strongest single reach (~2.5) + a small multi-hit bonus — NOT the sum (~5.0).
        assert p.score < 4.0


# -- corpus-IDF keyword weighting (the precision fix, spore-141/spore-104 dep-2) ----


class TestCorpusIDF:
    """The precision fix: on a real-sized corpus, a high-document-frequency PROCESS word
    is deflated by corpus-IDF so it can't float a pattern grounded in an episode it
    merely brushed; below ``IDF_MIN_CORPUS`` the length-proxy is used (so small fixtures
    and the Store-free :func:`retrieve_patterns` path stay byte-unchanged)."""

    # Two ubiquitous words present in EVERY seeded episode → DF == corpus size → the most
    # common possible terms. A per-episode distinctive token keeps each independently
    # recallable. The grounding episode is a DECISION so the +0.5 type boost is in play.
    _COMMON = "session convo"

    def _seed_big_corpus(self, store, n=60):
        # Two unique per-episode tokens (`qa{i}z`/`qb{i}z`) — the trailing `z` makes them
        # collision-free under substring match (`qa3z` is NOT inside `qa30z`), so their
        # DF is exactly 1, unlike a bare `widget_3` which substrings `widget_30..39`.
        ids = []
        for i in range(n):
            e = store.record(
                f"A session convo note number {i} about qa{i}z and qb{i}z in the ongoing "
                f"build, recorded with enough body to clear the episode length floor.",
                EpisodeType.DECISION, source="flow",
                timestamp=f"2026-06-05T{i // 60:02d}:{i % 60:02d}:00Z",
            )
            ids.append(e.id)
        return ids

    def test_idf_weight_floors_ubiquitous_lifts_rare(self):
        # A term in (nearly) every episode floors; a unique term tops out near 1.0.
        assert _retrieval._idf_weight(1000, 1000) == _retrieval.IDF_FLOOR  # in ALL → floor
        assert _retrieval._idf_weight(1, 1000) > 0.8                       # unique → high
        assert _retrieval._idf_weight(500, 1000) < _retrieval._idf_weight(5, 1000)

    def test_query_weights_uses_idf_above_min_corpus(self, stores):
        store, _ = stores
        self._seed_big_corpus(store, n=60)
        assert store.recall(limit=0).total_matching >= _retrieval.IDF_MIN_CORPUS
        weights, used_idf = _retrieval._query_weights(
            store, ["session", "qa3z"], {"session": 60, "qa3z": 1}
        )
        assert used_idf is True
        # the ubiquitous word is floored; the rare one outweighs it decisively.
        assert weights["session"] == pytest.approx(_retrieval.IDF_FLOOR, abs=0.01)
        assert weights["qa3z"] > weights["session"]

    def test_query_weights_falls_back_below_min_corpus(self, stores):
        store, _ = stores
        store.record(
            "One apparatus-and-verification episode, comfortably past the length floor "
            "so it is a valid candidate but the corpus stays tiny.",
            EpisodeType.OBSERVATION, source="flow", timestamp="2026-06-05T09:00:00Z",
        )
        weights, used_idf = _retrieval._query_weights(
            store, ["apparatus", "verification"], {"apparatus": 1, "verification": 1}
        )
        assert used_idf is False  # below IDF_MIN_CORPUS → length-proxy
        assert weights["apparatus"] == _retrieval._keyword_weight("apparatus")

    def test_common_word_no_longer_floods_pattern(self, stores, monkeypatch):
        store, crystal = stores
        ids = self._seed_big_corpus(store, n=60)
        # A pattern grounded in ONE seeded episode, whose own text shares NO query word —
        # so only the evidence edge from that episode can surface it.
        crystal.crystallize(
            name="some_unrelated_principle", level=3,
            explanation="a distilled lesson with no lexical overlap with the query terms",
            evidence=[ids[0]], today=T0,
        )
        q = "session convo"  # ONLY the two ubiquitous words
        # Length-proxy regime (forced by lifting the corpus floor): 1.0+1.0 + 0.5 boost
        # = 2.5 ⇒ the episode clears the bar, seeds, and FLOODS the pattern.
        monkeypatch.setattr(_retrieval, "IDF_MIN_CORPUS", 10**9)
        proxy = retrieve_relevant(store, crystal, q, now=NOW, today=T0)
        assert "some_unrelated_principle" in {p.name for p in proxy.patterns}
        # Corpus-IDF regime (default): both words floor (0.3+0.3 + 0.5 = 1.1 < 1.6) ⇒ the
        # episode never seeds ⇒ the pattern is correctly NOT surfaced. The fix.
        monkeypatch.undo()
        idf = retrieve_relevant(store, crystal, q, now=NOW, today=T0)
        assert "some_unrelated_principle" not in {p.name for p in idf.patterns}

    def test_distinctive_query_still_reaches_on_big_corpus(self, stores):
        store, crystal = stores
        ids = self._seed_big_corpus(store, n=60)
        crystal.crystallize(
            name="some_unrelated_principle", level=3,
            explanation="a distilled lesson with no lexical overlap with the query terms",
            evidence=[ids[3]], today=T0,
        )
        # qa3z + qb3z are unique (df=1) → high IDF; two such hits clear the bar and reach
        # the pattern grounded in ids[3] (one distinctive hit + boost would not — that is
        # the MIN_HITS=2 precision floor, intact under IDF).
        r = retrieve_relevant(store, crystal, "qa3z qb3z", now=NOW, today=T0)
        assert "some_unrelated_principle" in {p.name for p in r.patterns}

    def test_keyword_source_tag(self, stores):
        store, crystal = stores
        _seed_crystal(crystal)
        r = retrieve_relevant(store, crystal, "structural invariants discipline", now=NOW, today=T0)
        p = next(p for p in r.patterns if p.name == "structural_invariants_beat_discipline")
        assert p.source == "keyword"  # matched the pattern's OWN text

    def test_anchor_gate_kills_fat_common_word_query_above_min_corpus(self, stores):
        store, crystal = stores
        ids = self._seed_big_corpus(store, n=60)  # all carry "session"/"convo"/"build"
        crystal.crystallize(
            name="some_unrelated_principle", level=3,
            explanation="a distilled lesson with no lexical overlap with the query terms",
            evidence=[ids[0]], today=T0,
        )
        # A FAT query of MANY ubiquitous words (the L2-flagged accumulation case): under a
        # floor-only fix 6 floored words (6×0.30=1.8) would clear the 1.6 bar and re-leak.
        # The √N distinctiveness anchor kills it structurally — no term is distinctive, so
        # no episode seeds, regardless of how many common words pile up.
        q = "session convo build session convo build session convo"
        r = retrieve_relevant(store, crystal, q, now=NOW, today=T0)
        assert r.patterns == []

    def test_distinctive_name_surfaces_as_keyword_source_under_idf(self, stores):
        store, crystal = stores
        self._seed_big_corpus(store, n=60)  # corpus ≥ IDF_MIN_CORPUS → IDF regime active
        # A pattern whose NAME carries two distinctive (corpus-absent) tokens; a query of
        # those tokens clears the √N anchor AND matches the pattern's own text → the
        # keyword tier fires under IDF and the source tag is "keyword" (not evidence_edge).
        crystal.crystallize(
            name="zylophonic_qismetric_invariant", level=3,
            explanation="a principle whose distilled text shares the query's rare tokens",
            tags=[], today=T0,
        )
        r = retrieve_relevant(store, crystal, "zylophonic qismetric invariant", now=NOW, today=T0)
        p = next(p for p in r.patterns if p.name == "zylophonic_qismetric_invariant")
        assert p.source == "keyword"
