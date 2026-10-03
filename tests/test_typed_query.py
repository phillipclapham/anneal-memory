"""Typed-query recall: ``mode="query"`` on the retrieval functions, ``search_episodes``,
and the MCP ``recall`` word-by-word fallback / ``crystal_recall`` ``mode`` parameter.

The scenario fixtures follow what a real agent did: it asked MCP ``recall`` for a
whole PHRASE ("bank export fmt_row64 CLI nightly rows"), the exact-substring match
found nothing, and the store did hold single-word hits (fmt_row52, cutover, nightly).
"""

from __future__ import annotations

from datetime import date

import pytest

from anneal_memory import (
    CrystalStore,
    EpisodeMatch,
    Store,
    retrieve_patterns,
    retrieve_relevant,
    search_episodes,
)
from anneal_memory import retrieval as _retrieval
from anneal_memory.integrity import TOOLS
from anneal_memory.server import Server
from anneal_memory.types import EpisodeType

T0 = date(2026, 10, 3)
PHRASE = "bank export fmt_row64 CLI nightly rows"


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "mem.db", project_name="TypedQuery")
    yield s
    s.close()


@pytest.fixture
def server(store):
    return Server(store)


def _call(server, name, arguments=None):
    return server._handle_tools_call({"name": name, "arguments": arguments or {}})


def _text(result):
    return result["content"][0]["text"]


def _seed_mykhailo(store):
    """Three episodes, none containing the whole phrase, each holding some of its words."""
    a = store.record(
        "Rounded the invoice amounts in fmt_row52 so the fixed-width bank file carries "
        "whole cents and the totals row still adds up.",
        EpisodeType.DECISION, source="agent", timestamp="2026-10-01T09:00:00Z",
    )
    b = store.record(
        "The cutover to the new billing run happened on Friday; the old totals script "
        "was retired after the first clean reconciliation.",
        EpisodeType.OBSERVATION, source="user", timestamp="2026-10-01T10:00:00Z",
    )
    c = store.record(
        "Scheduled the nightly export as a cron job and wrote the output rows to the "
        "shared drive for the finance team to pick up each morning.",
        EpisodeType.CONTEXT, source="agent", timestamp="2026-10-02T09:00:00Z",
    )
    return a.id, b.id, c.id


# -- retrieve_relevant / retrieve_patterns: the mode --------------------------


class TestMode:
    def test_query_mode_surfaces_single_hit_episode_prompt_mode_does_not(self, store):
        a, _b, _c = _seed_mykhailo(store)
        # Two distinctive keywords, but the store holds only ONE of them (fmt_row52).
        q = "fmt_row52 zebraquartz"
        prompt = retrieve_relevant(store, None, q)
        query = retrieve_relevant(store, None, q, mode="query")
        assert prompt.episodes == []
        assert [e.id for e in query.episodes] == [a]

    def test_query_mode_accepts_a_single_keyword_query(self, store):
        a, _b, _c = _seed_mykhailo(store)
        assert retrieve_relevant(store, None, "fmt_row52").episodes == []
        r = retrieve_relevant(store, None, "fmt_row52", mode="query")
        assert [e.id for e in r.episodes] == [a]
        assert r.query_keywords == ["fmt_row52"]

    def test_prompt_mode_is_the_default_and_unchanged(self, store):
        _seed_mykhailo(store)
        q = "nightly export rows finance team"
        default = retrieve_relevant(store, None, q)
        explicit = retrieve_relevant(store, None, q, mode="prompt")
        assert default == explicit
        assert len(default.episodes) == 1  # 2+ hits: the gates it keeps still pass

    def test_query_mode_keeps_ranking_more_hits_first(self, store):
        a, b, c = _seed_mykhailo(store)
        r = retrieve_relevant(
            store, None, "nightly export rows cutover", mode="query", max_episodes=5
        )
        assert [e.id for e in r.episodes][0] == c  # three hits beat one
        assert {e.id for e in r.episodes} == {b, c}

    def test_query_mode_still_skips_short_episodes(self, store):
        store.record("fmt_row52 rounding.", EpisodeType.OBSERVATION, source="agent")
        r = retrieve_relevant(store, None, "fmt_row52", mode="query")
        assert r.episodes == []

    def test_query_mode_opens_the_pattern_tier(self, tmp_path):
        crystal = CrystalStore(tmp_path / "mem.crystal.json")
        crystal.crystallize(
            name="fmt_row52_rounding", level=2,
            explanation="the bank export rounds to whole cents", tags=[], today=T0,
        )
        store = Store(tmp_path / "mem.db")
        try:
            assert retrieve_patterns(crystal, "fmt_row52", today=T0) == []
            assert retrieve_patterns(crystal, "fmt_row52", today=T0, mode="query")[0].name \
                == "fmt_row52_rounding"
            assert retrieve_relevant(store, crystal, "fmt_row52", today=T0).patterns == []
            assert retrieve_relevant(
                store, crystal, "fmt_row52", today=T0, mode="query"
            ).patterns[0].name == "fmt_row52_rounding"
        finally:
            store.close()

    @pytest.mark.parametrize("bad", ["Query", "", "prompts", None, 3, ["query"]])
    def test_bad_mode_raises(self, store, bad):
        with pytest.raises(ValueError, match="mode must be one of"):
            retrieve_relevant(store, None, "nightly export rows", mode=bad)
        with pytest.raises(ValueError, match="mode must be one of"):
            retrieve_patterns(None, "nightly export rows", mode=bad)

    def test_query_mode_does_not_mutate_module_constants(self, store):
        _seed_mykhailo(store)
        before = (
            _retrieval.MIN_KEYWORDS, _retrieval.MIN_HITS,
            _retrieval.SCORE_THRESHOLD, _retrieval.IDF_SCORE_THRESHOLD,
            _retrieval.IDF_ANCHOR_WEIGHT,
        )
        retrieve_relevant(store, None, "fmt_row52", mode="query")
        search_episodes(store, "fmt_row52 nightly")
        after = (
            _retrieval.MIN_KEYWORDS, _retrieval.MIN_HITS,
            _retrieval.SCORE_THRESHOLD, _retrieval.IDF_SCORE_THRESHOLD,
            _retrieval.IDF_ANCHOR_WEIGHT,
        )
        assert before == after == (2, 2, 2.5, 1.6, 0.5)
        # and a prompt-mode call straight after is still gated
        assert retrieve_relevant(store, None, "fmt_row52").episodes == []

    def test_query_mode_opens_the_anchor_under_idf(self, store):
        """A corpus past IDF_MIN_CORPUS turns the distinctive anchor on in prompt mode;
        query mode drops it, so a query of only common words still returns episodes."""
        for i in range(60):
            store.record(
                f"Session {i}: worked on the build and the convo about the plan, "
                f"then reviewed the session notes for item {i}.",
                EpisodeType.OBSERVATION, source="agent",
                timestamp=f"2026-09-{(i % 28) + 1:02d}T09:00:00Z",
            )
        q = "session convo build"
        assert retrieve_relevant(store, None, q).episodes == []
        assert len(retrieve_relevant(store, None, q, mode="query").episodes) == 3


# -- search_episodes -----------------------------------------------------------


class TestSearchEpisodes:
    def test_returns_ranked_matches_with_matched_words(self, store):
        a, b, c = _seed_mykhailo(store)
        out = search_episodes(store, PHRASE)
        assert all(isinstance(m, EpisodeMatch) for m in out)
        by_id = {m.episode.id: m.matched for m in out}
        assert set(by_id) == {a, c}
        assert by_id[a] == ("bank",)
        assert by_id[c] == ("export", "nightly", "rows")
        assert out[0].episode.id == c  # three matched words outrank one

    def test_filters_are_applied_to_the_candidates(self, store):
        a, b, c = _seed_mykhailo(store)
        q = "bank export nightly cutover"
        assert {m.episode.id for m in search_episodes(store, q)} == {a, b, c}
        assert {m.episode.id for m in search_episodes(store, q, source="user")} == {b}
        assert {m.episode.id for m in search_episodes(store, q, episode_type="context")} == {c}
        assert {m.episode.id for m in search_episodes(
            store, q, episode_type=EpisodeType.DECISION)} == {a}
        assert {m.episode.id for m in search_episodes(
            store, q, since="2026-10-02T00:00:00Z")} == {c}
        assert {m.episode.id for m in search_episodes(
            store, q, until="2026-10-01T09:30:00Z")} == {a}

    def test_limit_caps_and_nonpositive_returns_nothing(self, store):
        _seed_mykhailo(store)
        q = "bank export nightly cutover"
        assert len(search_episodes(store, q, limit=2)) == 2
        assert search_episodes(store, q, limit=0) == []
        assert search_episodes(store, q, limit=-1) == []

    def test_no_keywords_or_no_hit_returns_empty(self, store):
        _seed_mykhailo(store)
        assert search_episodes(store, "the and of") == []
        assert search_episodes(store, "zebraquartz marmalade") == []

    def test_superseded_episode_hidden_unless_asked(self, store):
        old = store.record(
            "The nightly export wrote its rows to the old shared drive location, which finance read.",
            EpisodeType.CONTEXT, source="agent", timestamp="2026-10-01T09:00:00Z",
        )
        new = store.record(
            "The nightly export now writes its rows to the finance share instead, which they read daily.",
            EpisodeType.CONTEXT, source="agent", timestamp="2026-10-02T09:00:00Z",
            supersedes=[old.id],
        )
        ids = {m.episode.id for m in search_episodes(store, "nightly export rows")}
        assert ids == {new.id}
        ids = {m.episode.id for m in search_episodes(
            store, "nightly export rows", include_superseded=True)}
        assert ids == {old.id, new.id}

    def test_bad_episode_type_raises(self, store):
        _seed_mykhailo(store)
        with pytest.raises(ValueError):
            search_episodes(store, "nightly export", episode_type="nonsense")


# -- MCP recall: exact first, words second -------------------------------------


class TestMcpRecallFallback:
    def test_phrase_miss_falls_back_to_ranked_words(self, server, store):
        a, b, c = _seed_mykhailo(store)
        r = _call(server, "recall", {"keyword": PHRASE, "limit": 10})
        assert not r.get("isError")
        text = _text(r)
        assert "No matching" not in text
        assert text.startswith(
            "No episode contains the exact phrase; ranked by matching words "
            "(bank, export, fmt_row64, nightly, rows)."
        )
        assert f"({c})" in text and f"({a})" in text
        assert f"({b})" not in text
        assert "(matched: export, nightly, rows)" in text
        assert text.index(f"({c})") < text.index(f"({a})")  # best first

    def test_exact_phrase_hit_is_unchanged(self, server, store):
        a, b, c = _seed_mykhailo(store)
        r = _call(server, "recall", {"keyword": "nightly export"})
        text = _text(r)
        assert text.startswith("Found 1 episodes (showing 1):")
        assert "ranked by matching words" not in text
        assert f"({c})" in text and f"({a})" not in text

    def test_single_word_miss_is_unchanged(self, server, store):
        _seed_mykhailo(store)
        assert _text(_call(server, "recall", {"keyword": "fmt_row99"})) == \
            "No matching episodes found."

    def test_single_word_hit_is_unchanged(self, server, store):
        a, _b, _c = _seed_mykhailo(store)
        text = _text(_call(server, "recall", {"keyword": "fmt_row52"}))
        assert text.startswith("Found 1 episodes (showing 1):")
        assert f"({a})" in text

    def test_phrase_with_no_matching_word_stays_a_miss(self, server, store):
        _seed_mykhailo(store)
        assert _text(_call(server, "recall", {"keyword": "zebraquartz marmalade"})) == \
            "No matching episodes found."

    def test_filters_are_respected_in_the_fallback(self, server, store):
        a, b, c = _seed_mykhailo(store)
        text = _text(_call(server, "recall", {"keyword": PHRASE, "source": "user"}))
        assert text == "No matching episodes found."  # a, c are source=agent
        text = _text(_call(server, "recall", {"keyword": PHRASE, "source": "agent",
                                              "episode_type": "context"}))
        assert f"({c})" in text and f"({a})" not in text
        text = _text(_call(server, "recall", {"keyword": PHRASE,
                                              "since": "2026-10-02T00:00:00Z"}))
        assert f"({c})" in text and f"({a})" not in text
        text = _text(_call(server, "recall", {"keyword": PHRASE,
                                              "until": "2026-10-01T23:59:59Z"}))
        assert f"({a})" in text and f"({c})" not in text

    def test_limit_is_respected_in_the_fallback(self, server, store):
        _seed_mykhailo(store)
        text = _text(_call(server, "recall", {"keyword": PHRASE, "limit": 1}))
        assert "Showing 1:" in text
        assert text.count("\n- (") == 1

    def test_offset_past_a_miss_does_not_fall_back(self, server, store):
        _seed_mykhailo(store)
        assert _text(_call(server, "recall", {"keyword": PHRASE, "offset": 1})) == \
            "No matching episodes found."

    def test_no_keyword_still_lists_recent(self, server, store):
        _seed_mykhailo(store)
        assert _text(_call(server, "recall", {"limit": 100})).startswith("Found 3 episodes")

    @pytest.mark.parametrize("bad", ["message", "", "Decision", 5, ["decision"]])
    def test_bad_episode_type_names_the_valid_values(self, server, bad):
        r = _call(server, "recall", {"episode_type": bad})
        assert r.get("isError") is True
        text = _text(r)
        assert f"episode_type {bad!r} is not one of: " in text
        for t in EpisodeType:
            assert t.value in text

    def test_good_episode_type_still_filters(self, server, store):
        a, _b, _c = _seed_mykhailo(store)
        text = _text(_call(server, "recall", {"episode_type": "decision"}))
        assert text.startswith("Found 1 episodes") and f"({a})" in text

    def test_episode_type_is_an_enum_in_the_schema(self):
        recall = next(t for t in TOOLS if t["name"] == "recall")
        assert recall["inputSchema"]["properties"]["episode_type"]["enum"] == [
            t.value for t in EpisodeType
        ]

    def test_description_tells_the_agent_about_word_matching(self):
        recall = next(t for t in TOOLS if t["name"] == "recall")
        assert "word" in recall["description"]
        assert "word by word" in recall["inputSchema"]["properties"]["keyword"]["description"]


# -- MCP crystal_recall: mode ---------------------------------------------------


class TestMcpCrystalRecallMode:
    def _seed(self, server):
        CrystalStore(server._crystal_path).crystallize(
            name="fmt_row52_rounding", level=2,
            explanation="the bank export rounds to whole cents", tags=[],
        )

    def test_schema_declares_mode(self):
        cr = next(t for t in TOOLS if t["name"] == "crystal_recall")
        mode = cr["inputSchema"]["properties"]["mode"]
        assert mode["enum"] == ["prompt", "query"] and mode["default"] == "prompt"

    def test_default_mode_keeps_the_gates(self, server):
        self._seed(server)
        assert _text(_call(server, "crystal_recall", {"query": "fmt_row52"})) == (
            "No crystallized patterns matched (query too thin — give it at least 2 "
            "distinctive keywords, or check crystal_index for what exists)."
        )
        assert _text(_call(server, "crystal_recall",
                           {"query": "fmt_row52", "mode": "prompt"})).startswith(
            "No crystallized patterns matched")

    def test_query_mode_surfaces_the_single_keyword_pattern(self, server):
        self._seed(server)
        text = _text(_call(server, "crystal_recall",
                           {"query": "fmt_row52", "mode": "query"}))
        assert "fmt_row52_rounding" in text

    def test_query_mode_with_associative_off(self, server):
        self._seed(server)
        text = _text(_call(server, "crystal_recall",
                           {"query": "fmt_row52", "mode": "query", "associative": False}))
        assert "fmt_row52_rounding" in text

    @pytest.mark.parametrize("bad", ["Query", "", "open", 1, True, None, ["query"]])
    def test_bad_mode_is_an_error_result(self, server, bad):
        r = _call(server, "crystal_recall", {"query": "fmt_row52 rounding", "mode": bad})
        assert r.get("isError") is True
        assert "mode must be one of" in _text(r)
