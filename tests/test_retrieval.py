"""Tests for the unified on-demand recall surface (retrieval.py, AM-CRYSTAL-RECALL).

Covers keyword extraction, the precision-biased weighted-overlap scoring over BOTH
crystallized patterns and episodes, ranking, caps, the high-signal type boost, the
recent-exclusion cutoff (incl. fail-open on a bad clock), and an L4-style proof that
a recall hook can consume the result with zero adapter (the v2-consumer contract).
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from anneal_memory import (
    CrystalStore,
    RelevantResult,
    Store,
    extract_keywords,
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
