"""L3 fixes for typed-query recall: uncapped candidate fetch in query mode, the
top-up honouring ``limit``, a negative offset, and the recall ``limit`` description."""

from __future__ import annotations

import pytest

from anneal_memory import Store, retrieve_relevant, search_episodes
from anneal_memory import retrieval as _retrieval
from anneal_memory.integrity import TOOLS
from anneal_memory.server import Server
from anneal_memory.types import EpisodeType


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "m.db", project_name="T", audit=False)
    yield s
    s.close()


@pytest.fixture
def server(store):
    return Server(store)


def _call(server, name, arguments=None):
    return server._handle_tools_call({"name": name, "arguments": arguments or {}})


def _text(r):
    return r["content"][0]["text"]


def _seed_flood(store, per_word):
    """One OLD episode holding both words, then more than the per-keyword fetch cap of
    NEWER episodes holding only one of them."""
    old = store.record(
        "Old note: alpha and beta both appear here together in one episode.",
        EpisodeType.OBSERVATION, timestamp="2026-01-01T00:00:00Z")
    for i in range(per_word):
        store.record(f"New alpha-only note number {i} about the plan.",
                     EpisodeType.OBSERVATION, timestamp=f"2026-06-01T00:{i // 60:02d}:{i % 60:02d}Z")
        store.record(f"New beta-only note number {i} about the plan.",
                     EpisodeType.OBSERVATION, timestamp=f"2026-07-01T00:{i // 60:02d}:{i % 60:02d}Z")
    return old.id


class TestUncappedFetchInQueryMode:
    PER_WORD = _retrieval.CANDIDATE_LIMIT_PER_KEYWORD + 5

    def test_search_episodes_finds_the_older_two_word_episode(self, store):
        old = _seed_flood(store, self.PER_WORD)
        top = search_episodes(store, "alpha beta", limit=1)
        assert [m.episode.id for m in top] == [old]
        assert top[0].matched == ("alpha", "beta")

    def test_retrieve_relevant_query_mode_too_prompt_mode_keeps_the_cap(self, store):
        old = _seed_flood(store, self.PER_WORD)
        q = "alpha beta"
        got = retrieve_relevant(store, None, q, mode="query", max_episodes=1)
        assert [e.id for e in got.episodes] == [old]
        # prompt mode still fetches only the newest candidates per keyword (and has the
        # 80-character floor, so these short notes never surface there anyway)
        assert old not in [e.id for e in retrieve_relevant(
            store, None, q, max_episodes=3).episodes]

    def test_the_mcp_count_is_exact(self, server, store):
        _seed_flood(store, self.PER_WORD)
        total = 2 * self.PER_WORD + 1
        text = _text(_call(server, "recall", {"keyword": "alpha beta zzzzz"}))
        assert f"Showing top 10 of {total} word matches" in text
        assert text.splitlines()[1].count("(matched 2/3: alpha, beta)") == 1


class TestAlsoMatchingHonoursLimit:
    def _seed(self, store):
        store.record("Exact: alpha beta gamma delta sit together in this one episode.",
                     EpisodeType.DECISION, timestamp="2026-10-03T09:00:00Z")
        for i in range(8):
            store.record(f"Row {i}: alpha only appears in this other episode number {i}.",
                         EpisodeType.OBSERVATION, timestamp=f"2026-10-02T09:00:{i:02d}Z")

    def test_extras_are_capped_by_the_remaining_limit(self, server, store):
        self._seed(store)
        text = _text(_call(server, "recall", {"keyword": "alpha beta gamma delta", "limit": 2}))
        assert text.count("\n- (") == 2  # one exact + one extra, not six
        assert "Also matching by words:" in text

    def test_no_room_means_no_topup(self, server, store):
        self._seed(store)
        text = _text(_call(server, "recall", {"keyword": "alpha beta gamma delta", "limit": 1}))
        assert "Also matching" not in text and text.count("\n- (") == 1

    def test_default_limit_still_gives_five(self, server, store):
        self._seed(store)
        text = _text(_call(server, "recall", {"keyword": "alpha beta gamma delta"}))
        assert text.partition("Also matching by words:")[2].count("\n- (") == 5


class TestNegativeOffsetIsZero:
    def test_exact_with_topup(self, server, store):
        TestAlsoMatchingHonoursLimit()._seed(store)
        base = _text(_call(server, "recall", {"keyword": "alpha beta gamma delta"}))
        assert "Also matching by words:" in base
        assert _text(_call(server, "recall",
                           {"keyword": "alpha beta gamma delta", "offset": -3})) == base

    def test_fallback(self, server, store):
        TestAlsoMatchingHonoursLimit()._seed(store)
        base = _text(_call(server, "recall", {"keyword": "alpha zzzzz qqqqq"}))
        assert "ranked by matching words" in base
        assert _text(_call(server, "recall",
                           {"keyword": "alpha zzzzz qqqqq", "offset": -1})) == base

    def test_negative_limit_is_zero(self, server, store):
        TestAlsoMatchingHonoursLimit()._seed(store)
        assert _text(_call(server, "recall", {"keyword": "alpha", "limit": -5})) == \
            "No matching episodes found."
        assert _text(_call(server, "recall",
                           {"keyword": "alpha zzzzz qqqqq", "limit": -5})) == \
            "No matching episodes found."


class TestRecallLimitDescription:
    def test_the_limit_property_names_the_fallback_default(self):
        recall = next(t for t in TOOLS if t["name"] == "recall")
        d = recall["inputSchema"]["properties"]["limit"]["description"]
        assert "Default 100" in d and "default is 10" in d
