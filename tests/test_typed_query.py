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
from anneal_memory import extract_keywords
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

    def test_query_mode_has_no_length_floor_prompt_mode_keeps_it(self, store):
        short = store.record(
            "User is allergic to tree nuts.", EpisodeType.OBSERVATION, source="agent",
            timestamp="2026-10-02T09:00:00Z",
        )
        assert len(short.content) < _retrieval.MIN_EPISODE_LEN
        q = "allergic tree nuts"
        assert retrieve_relevant(store, None, q).episodes == []
        r = retrieve_relevant(store, None, q, mode="query")
        assert [e.id for e in r.episodes] == [short.id]
        assert [m.episode.id for m in search_episodes(store, q)] == [short.id]

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
            "(bank, export, fmt_row64, cli, nightly, rows). Showing 2:"
        )
        assert f"({c})" in text and f"({a})" in text
        assert f"({b})" not in text
        assert "(matched 3/6: export, nightly, rows)" in text
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
        assert "Showing top 1 of 2 word matches; pass a rarer word or a higher limit" in text
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


# -- fix round -------------------------------------------------------------------


class TestShortEpisodeFallback:
    def test_mcp_fallback_finds_a_short_episode(self, server, store):
        short = store.record("User is allergic to tree nuts.", EpisodeType.OBSERVATION)
        text = _text(_call(server, "recall", {"keyword": "allergic tree nuts"}))
        assert f"({short.id})" in text
        assert "ranked by matching words" in text


class TestLimitCoercion:
    def _seed(self, store):
        for i in range(4):
            store.record(f"Episode {i} about the nightly export of billing rows.",
                         EpisodeType.OBSERVATION)

    def test_whole_number_float_limit_works_on_both_paths(self, server, store):
        self._seed(store)
        exact = _text(_call(server, "recall", {"keyword": "export", "limit": 3.0}))
        assert exact.startswith("Found 4 episodes (showing 3):")
        fallback = _text(_call(server, "recall",
                               {"keyword": "export zzzzz nightly", "limit": 3.0}))
        assert "ranked by matching words" in fallback
        assert fallback.count("\n- (") == 3

    @pytest.mark.parametrize("bad", [2.5, True, False, "5", None, [3], float("inf")])
    def test_non_integer_limit_is_a_named_error(self, server, store, bad):
        self._seed(store)
        for keyword in ("export", "export zzzzz nightly"):
            r = _call(server, "recall", {"keyword": keyword, "limit": bad})
            assert r.get("isError") is True
            assert _text(r) == "Error: limit must be an integer"

    @pytest.mark.parametrize("bad", [1.5, True, "1", None])
    def test_non_integer_offset_is_a_named_error(self, server, store, bad):
        self._seed(store)
        r = _call(server, "recall", {"keyword": "export", "offset": bad})
        assert r.get("isError") is True
        assert _text(r) == "Error: offset must be an integer"

    def test_whole_number_float_offset_works(self, server, store):
        self._seed(store)
        assert _text(_call(server, "recall", {"keyword": "export", "offset": 1.0,
                                              "limit": 2})).startswith("Found 4 episodes (showing 2):")


class TestSupersessionInFallback:
    def test_include_superseded_marks_the_replaced_episode(self, server, store):
        old = store.record(
            "The nightly export wrote its rows to the old shared drive location.",
            EpisodeType.CONTEXT, source="agent", timestamp="2026-10-01T09:00:00Z")
        new = store.record(
            "The nightly export now writes its rows to the finance share instead.",
            EpisodeType.CONTEXT, source="agent", timestamp="2026-10-02T09:00:00Z",
            supersedes=[old.id])
        text = _text(_call(server, "recall", {"keyword": "export rows zzzzz",
                                              "include_superseded": True}))
        old_line = next(ln for ln in text.splitlines() if f"({old.id})" in ln)
        assert f"(superseded by {new.id})" in old_line
        new_line = next(ln for ln in text.splitlines() if f"({new.id})" in ln)
        assert "superseded by" not in new_line
        hidden = _text(_call(server, "recall", {"keyword": "export rows zzzzz"}))
        assert f"({old.id})" not in hidden

    def test_episode_match_carries_superseded_by(self, store):
        old = store.record(
            "The nightly export wrote its rows to the old shared drive location.",
            EpisodeType.CONTEXT, timestamp="2026-10-01T09:00:00Z")
        new = store.record(
            "The nightly export now writes its rows to the finance share instead.",
            EpisodeType.CONTEXT, timestamp="2026-10-02T09:00:00Z", supersedes=[old.id])
        by_id = {m.episode.id: m.superseded_by for m in search_episodes(
            store, "nightly export rows", include_superseded=True)}
        assert by_id == {old.id: new.id, new.id: None}


class TestQueryModeShortTokens:
    def test_all_caps_and_symbol_tokens_survive_in_query_mode(self):
        q = "SQL API s3 k8s v2 my-db plain lowercase cli"
        assert extract_keywords(q, mode="query") == [
            "sql", "api", "s3", "k8s", "v2", "my-db", "plain", "lowercase"]
        assert extract_keywords("CLI sql Api", mode="query") == ["cli"]

    def test_stopwords_stay_out_however_cased(self):
        assert extract_keywords("IT THE Do", mode="query") == []

    def test_prompt_mode_extraction_is_unchanged(self):
        q = "SQL API s3 k8s v2 my-db plain lowercase cli"
        assert extract_keywords(q) == ["my-db", "plain", "lowercase"]
        assert extract_keywords(q, mode="prompt") == ["my-db", "plain", "lowercase"]

    def test_bad_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            extract_keywords("nightly export", mode="x")

    def test_search_finds_an_episode_by_its_short_term(self, store):
        hit = store.record("Moved the report queries from SQL views into the warehouse.",
                           EpisodeType.DECISION)
        store.record("Moved the report queries from the old views into the warehouse.",
                     EpisodeType.DECISION)
        assert [m.episode.id for m in search_episodes(store, "SQL")] == [hit.id]
        assert [e.id for e in retrieve_relevant(
            store, None, "SQL", mode="query").episodes] == [hit.id]
        assert retrieve_relevant(store, None, "SQL").episodes == []


class TestFallbackTrigger:
    def test_phrase_reducing_to_one_word_falls_back_on_that_word(self, server, store):
        a, b, c = _seed_mykhailo(store)
        text = _text(_call(server, "recall", {"keyword": "cutover the"}))
        assert "ranked by matching words (cutover)" in text
        assert f"({b})" in text
        assert "(matched 1/1: cutover)" in text

    def test_single_token_never_falls_back(self, server, store):
        _seed_mykhailo(store)
        assert _text(_call(server, "recall", {"keyword": "cutoverr"})) == \
            "No matching episodes found."

    def test_phrase_reducing_to_nothing_does_not_fall_back(self, server, store):
        _seed_mykhailo(store)
        assert _text(_call(server, "recall", {"keyword": "the and"})) == \
            "No matching episodes found."

    def test_short_caps_term_counts_toward_the_trigger(self, server, store):
        hit = store.record("Moved the report queries from SQL views into the warehouse.",
                           EpisodeType.DECISION)
        text = _text(_call(server, "recall", {"keyword": "SQL zzzzz"}))
        assert f"({hit.id})" in text and "ranked by matching words (sql, zzzzz)" in text


class TestFallbackCap:
    def _seed_many(self, store, n=12):
        for i in range(n):
            store.record(
                f"Row {i}: the nightly export of the bank rows ran clean on that day.",
                EpisodeType.OBSERVATION, timestamp=f"2026-09-{i + 1:02d}T09:00:00Z")

    def test_default_is_capped_at_ten_and_says_so(self, server, store):
        self._seed_many(store)
        text = _text(_call(server, "recall", {"keyword": "nightly export bank zzzzz"}))
        assert "Showing top 10 of 12 word matches; pass a rarer word or a higher limit " \
            "for more." in text
        assert text.count("\n- (") == 10
        assert "(matched 3/4: nightly, export, bank)" in text

    def test_explicit_limit_is_honoured_above_the_default_cap(self, server, store):
        self._seed_many(store)
        text = _text(_call(server, "recall", {"keyword": "nightly export bank zzzzz",
                                              "limit": 12}))
        assert text.count("\n- (") == 12
        assert "Showing 12:" in text and "word matches" not in text

    def test_explicit_limit_below_the_matches_says_so(self, server, store):
        self._seed_many(store)
        text = _text(_call(server, "recall", {"keyword": "nightly export bank zzzzz",
                                              "limit": 4}))
        assert text.count("\n- (") == 4
        assert "Showing top 4 of 12 word matches" in text


class TestAlsoMatchingByWords:
    def test_small_exact_result_is_topped_up_with_word_matches(self, server, store):
        a, b, c = _seed_mykhailo(store)
        exact = store.record(
            "Scheduled the nightly export for the bank at midnight with the finance rows.",
            EpisodeType.DECISION, timestamp="2026-10-03T09:00:00Z")
        text = _text(_call(server, "recall", {"keyword": "nightly export for the bank"}))
        head, _, tail = text.partition("\n\nAlso matching by words:")
        # exact results first and unchanged
        assert head.startswith("Found 1 episodes (showing 1):")
        assert f"({exact.id})" in head
        assert tail and f"({exact.id})" not in tail
        assert f"({c})" in tail and "(matched 2/3: nightly, export)" in tail

    def test_no_topup_when_exact_has_three_or_more(self, server, store):
        for i in range(3):
            store.record(f"Row {i}: nightly export bank rows were fine on that day ok.",
                         EpisodeType.OBSERVATION)
        store.record("Other nightly export note about something else entirely here.",
                     EpisodeType.OBSERVATION)
        text = _text(_call(server, "recall", {"keyword": "nightly export bank"}))
        assert "Also matching" not in text

    def test_no_topup_for_a_two_word_keyword(self, server, store):
        _seed_mykhailo(store)
        assert "Also matching" not in _text(_call(server, "recall",
                                                  {"keyword": "nightly export"}))

    def test_topup_is_capped_at_five(self, server, store):
        store.record("Exact: alpha beta gamma delta sits together in this one episode.",
                     EpisodeType.DECISION)
        for i in range(8):
            store.record(f"Row {i}: alpha only appears in this other episode number {i}.",
                         EpisodeType.OBSERVATION)
        text = _text(_call(server, "recall", {"keyword": "alpha beta gamma delta"}))
        tail = text.partition("Also matching by words:")[2]
        assert tail.count("\n- (") == 5


class TestCrystalRecallDescriptions:
    def test_mode_description_is_agent_language_with_no_hardcoded_gates(self):
        cr = next(t for t in TOOLS if t["name"] == "crystal_recall")
        mode = cr["inputSchema"]["properties"]["mode"]["description"]
        assert mode == (
            "'query': for a question you are asking on purpose; one keyword is "
            "enough, and weaker matches come back too. 'prompt' (default): strict, "
            "built for automatic per-turn injection; may return nothing."
        )
        assert not any(ch.isdigit() for ch in mode)

    def test_main_description_does_not_contradict_query_mode(self):
        cr = next(t for t in TOOLS if t["name"] == "crystal_recall")
        d = cr["description"]
        assert "In the default 'prompt' mode it is precision-biased" in d
        assert "Pass mode='query' when you are asking explicitly." in d

    def test_thin_query_message_grammar_in_query_mode(self, server):
        text = _text(_call(server, "crystal_recall", {"query": "the", "mode": "query"}))
        assert "at least 1 distinctive keyword, or check crystal_index" in text


class TestEvidenceEdgeKeepsThePromptBar:
    def _corpus(self, store):
        ids = []
        for i in range(60):
            ids.append(store.record(
                f"Session {i}: reviewed the plan and the notes for item {i}"
                + (" including the zylophonic qismetric calibration." if i == 0 else "."),
                EpisodeType.OBSERVATION, timestamp=f"2026-09-{(i % 28) + 1:02d}T09:00:00Z",
            ).id)
        return ids

    def _crystal(self, tmp_path, ids):
        crystal = CrystalStore(tmp_path / "mem.crystal.json")
        crystal.crystallize(
            name="calibration_is_the_gate", level=3,
            explanation="an unrelated distilled lesson with no overlap",
            evidence=[ids[0]], today=T0)
        return crystal

    def test_one_brushed_word_does_not_float_a_citing_pattern(self, store, tmp_path):
        ids = self._corpus(store)
        crystal = self._crystal(tmp_path, ids)
        # "session" brushes every episode, including the one the pattern cites.
        r = retrieve_relevant(store, crystal, "session", today=T0, mode="query")
        assert r.patterns == []
        r = retrieve_relevant(store, crystal, "zylophonic", today=T0, mode="query")
        assert r.patterns == []  # one rare word still scores under the prompt bar

    def test_two_rare_words_still_reach_it_through_the_edge(self, store, tmp_path):
        ids = self._corpus(store)
        crystal = self._crystal(tmp_path, ids)
        r = retrieve_relevant(store, crystal, "zylophonic qismetric", today=T0,
                              mode="query")
        assert [p.name for p in r.patterns] == ["calibration_is_the_gate"]
        assert r.patterns[0].source == "evidence_edge"


class TestSearchEpisodesValidation:
    def test_bad_episode_type_raises_even_with_a_nonpositive_limit(self, store):
        with pytest.raises(ValueError):
            search_episodes(store, "nightly export", episode_type="nonsense", limit=0)


# -- prompt mode, frozen -------------------------------------------------------------

_PARITY_QUERIES = [
    "billing database invoice tables",
    "nightly export bank cutover",
    "export",
    "finance reconciliation script year-end",
    "database capacity weekly growth",
    "unrelated zebra quartz marmalade",
    "work plan project schedule",
]

# query -> (keywords, [(seed index, score)], [(pattern, score, source)], [(pattern, score)]
# from retrieve_patterns). Recorded from 54250bb, before any mode existed.
_PARITY_EXPECTED = {
    'billing database invoice tables': (['billing', 'database', 'invoice', 'tables'], [(0, 3.26), (2, 2.76), (3, 1.79)], [('billing_database_acid', 2.09, 'keyword')], [('billing_database_acid', 3.5)]),
    'nightly export bank cutover': (['nightly', 'export', 'bank', 'cutover'], [(6, 2.98), (1, 2.15)], [('export_cutover_discipline', 2.24, 'keyword')], [('export_cutover_discipline', 3.0)]),
    'export': (['export'], [], [], []),
    'finance reconciliation script year-end': (['finance', 'reconciliation', 'script', 'year-end'], [(7, 3.74)], [], []),
    'database capacity weekly growth': (['database', 'capacity', 'weekly', 'growth'], [], [], []),
    'unrelated zebra quartz marmalade': (['unrelated', 'zebra', 'quartz', 'marmalade'], [], [], []),
    'work plan project schedule': (['work', 'plan', 'project', 'schedule'], [], [], []),
}


def _parity_store(tmp_path):
    st = Store(tmp_path / "m.db", audit=False)
    cs = CrystalStore(tmp_path / "m.crystal.json")
    texts = [
        ("decision", "Chose PostgreSQL for the billing database because ACID guarantees outweigh raw write speed on the invoice tables."),
        ("observation", "The nightly export job writes fixed-width bank rows to the shared drive and finance reads them each morning."),
        ("tension", "Latency versus consistency in the billing database cannot both be optimized without sharding the invoice tables."),
        ("outcome", "Migration finished: invoice queries on the hot path run three times faster after the database index change."),
        ("context", "Production database sits at eighty percent capacity and grows about five percent every week."),
        ("observation", "Short note about export."),
        ("question", "Should the nightly export move to the new bank layout before the cutover, or after the quarterly audit window?"),
        ("decision", "Retired the old tax table only after finance confirmed the year-end reconciliation script no longer imports it."),
    ]
    ids = [
        st.record(c, t, timestamp=f"2026-09-{10 + i:02d}T09:00:00Z").id
        for i, (t, c) in enumerate(texts)
    ]
    for i in range(58):  # past IDF_MIN_CORPUS, so the IDF regime is the one frozen
        st.record(
            f"Filler session {i}: reviewed the work plan and discussed the project "
            f"schedule with the team lead for item {i}.",
            "observation", timestamp=f"2026-08-{(i % 28) + 1:02d}T12:00:00Z")
    cs.crystallize(name="billing_database_acid", level=3,
                   explanation="prefer ACID guarantees for billing tables",
                   tags=["billing"], evidence=[ids[0]], today=T0)
    cs.crystallize(name="export_cutover_discipline", level=2,
                   explanation="verify the layout change before the nightly job switches",
                   tags=["export"], evidence=[ids[1], ids[6]], today=T0)
    return st, cs, ids


class TestPromptModeFrozen:
    def test_prompt_mode_output_matches_the_recording_from_before_modes_existed(
        self, tmp_path
    ):
        st, cs, ids = _parity_store(tmp_path)
        try:
            index = {i: n for n, i in enumerate(ids)}
            for q in _PARITY_QUERIES:
                for kwargs in ({}, {"mode": "prompt"}):
                    r = retrieve_relevant(st, cs, q, today=T0, max_episodes=3, **kwargs)
                    got = (
                        r.query_keywords,
                        [(index.get(e.id, -1), e.score) for e in r.episodes],
                        [(p.name, p.score, p.source) for p in r.patterns],
                        [(p.name, p.score)
                         for p in retrieve_patterns(cs, q, today=T0, **kwargs)],
                    )
                    kw, eps, pats, pat_only = _PARITY_EXPECTED[q]
                    want = (
                        kw,
                        [tuple(x) for x in eps],
                        [tuple(x) for x in pats],
                        [tuple(x) for x in pat_only],
                    )
                    assert got == want, (q, kwargs)
        finally:
            st.close()
