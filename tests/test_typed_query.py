"""Typed-query recall: ``mode="query"`` on ``extract_keywords``, ``search_episodes``,
and the MCP ``recall`` word-by-word fallback.

The scenario fixtures follow what a real agent did: it asked MCP ``recall`` for a
whole PHRASE ("bank export fmt_row64 CLI nightly rows"), the exact-substring match
found nothing, and the store did hold single-word hits (fmt_row52, cutover, nightly).
"""

from __future__ import annotations

import pytest

from anneal_memory import (
    EpisodeMatch,
    Store,
    search_episodes,
)
from anneal_memory import retrieval as _retrieval
from anneal_memory import extract_keywords
from anneal_memory.integrity import TOOLS
from anneal_memory.server import Server
from anneal_memory.types import EpisodeType

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


# -- the retrieval mode -------------------------------------------------------


class TestMode:
    def test_query_mode_has_no_length_floor_prompt_mode_keeps_it(self, store):
        short = store.record(
            "User is allergic to tree nuts.", EpisodeType.OBSERVATION, source="agent",
            timestamp="2026-10-02T09:00:00Z",
        )
        assert len(short.content) < _retrieval.MIN_EPISODE_LEN
        q = "allergic tree nuts"
        assert [m.episode.id for m in search_episodes(store, q)] == [short.id]

    @pytest.mark.parametrize("bad", ["Query", "", "prompts", None, 3, ["query"]])
    def test_bad_mode_raises(self, store, bad):
        with pytest.raises(ValueError, match="mode must be one of"):
            extract_keywords("nightly export rows", mode=bad)

    def test_query_mode_does_not_mutate_module_constants(self, store):
        _seed_mykhailo(store)
        before = (
            _retrieval.MIN_HITS, _retrieval.SCORE_THRESHOLD,
            _retrieval.IDF_SCORE_THRESHOLD, _retrieval.IDF_ANCHOR_WEIGHT,
        )
        search_episodes(store, "fmt_row52 nightly")
        after = (
            _retrieval.MIN_HITS, _retrieval.SCORE_THRESHOLD,
            _retrieval.IDF_SCORE_THRESHOLD, _retrieval.IDF_ANCHOR_WEIGHT,
        )
        assert before == after == (2, 2.5, 1.6, 0.5)

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
        assert len(search_episodes(store, q, limit=3)) == 3


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




class TestSearchEpisodesValidation:
    def test_bad_episode_type_raises_even_with_a_nonpositive_limit(self, store):
        with pytest.raises(ValueError):
            search_episodes(store, "nightly export", episode_type="nonsense", limit=0)
