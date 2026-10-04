"""L3 fixes for the cue wiring: bounded query-mode fetch, CRLF continuity hash, fail-closed
inert key, one normalisation of limit/offset, the long-prompt count, split cue/fact matches."""

from __future__ import annotations

import warnings

import pytest

from anneal_memory import Store, prepare_wrap, retrieve_relevant, search_episodes, validated_save_continuity
from anneal_memory import continuity as _continuity_mod
from anneal_memory import retrieval as _retrieval
from anneal_memory.integrity import TOOLS
from anneal_memory.server import Server
from anneal_memory.types import EpisodeType

from .cue_helpers import save_cont, write_inert_key
from .test_cue_save_path import _episodes, default_text, save, stored_key
from .test_cue_wiring import _continuity

FACT = "- release process — cues: deploy, rollback"


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "m.db", project_name="T")
    yield s
    s.close()


def _call(server, name, arguments=None):
    return server._handle_tools_call({"name": name, "arguments": arguments or {}})


def _text(r):
    return r["content"][0]["text"]


def _spy_limits(store, monkeypatch):
    limits = []
    real = store.recall

    def spy(*a, **k):
        if k.get("keyword"):
            limits.append(k.get("limit"))
        return real(*a, **k)

    monkeypatch.setattr(store, "recall", spy)
    return limits


class TestQueryFetchIsBounded:
    def _seed(self, store, n):
        for i in range(n):
            store.record(f"Note {i}: the alpha thing happened in the plan again.",
                         EpisodeType.OBSERVATION, timestamp=f"2026-06-01T00:{i // 60:02d}:{i % 60:02d}Z")

    def test_the_named_ceiling_is_applied_in_query_mode_and_prompt_mode_keeps_400(
        self, store, monkeypatch
    ):
        assert _retrieval.QUERY_CANDIDATE_LIMIT == 5000
        self._seed(store, 30)
        limits = _spy_limits(store, monkeypatch)
        search_episodes(store, "alpha plan")
        retrieve_relevant(store, None, "alpha plan", mode="query", durable=False)
        assert set(limits) == {5000}
        limits.clear()
        retrieve_relevant(store, None, "alpha plan", durable=False)
        assert set(limits) == {400}

    def test_nothing_asks_for_an_unbounded_fetch(self, store, monkeypatch):
        self._seed(store, 30)
        server = Server(store)
        limits = _spy_limits(store, monkeypatch)
        _call(server, "recall", {"keyword": "alpha plan zzzzz"})
        assert limits and all(isinstance(n, int) and n <= 5000 for n in limits if n)

    def test_a_lowered_ceiling_bounds_what_is_read(self, store, monkeypatch):
        old = store.record("Old: alpha and beta both here in one place together.",
                           EpisodeType.OBSERVATION, timestamp="2026-01-01T00:00:00Z")
        for i in range(12):
            store.record(f"New alpha-only note {i} about the plan.", EpisodeType.OBSERVATION,
                         timestamp=f"2026-06-01T00:00:{i:02d}Z")
            store.record(f"New beta-only note {i} about the plan.", EpisodeType.OBSERVATION,
                         timestamp=f"2026-07-01T00:00:{i:02d}Z")
        assert [m.episode.id for m in search_episodes(store, "alpha beta", limit=1)] == [old.id]
        monkeypatch.setattr(_retrieval, "QUERY_CANDIDATE_LIMIT", 5)
        assert old.id not in [m.episode.id for m in search_episodes(store, "alpha beta")]

    def test_the_count_says_at_least_when_a_keyword_exceeded_the_ceiling(
        self, store, monkeypatch
    ):
        self._seed(store, 30)
        server = Server(store)
        text = _text(_call(server, "recall", {"keyword": "alpha plan zzzzz"}))
        assert "of at least" not in text and "Showing 10:" not in text  # 30 matches, exact
        assert "Showing top 10 of 30 word matches" in text
        monkeypatch.setattr(_retrieval, "QUERY_CANDIDATE_LIMIT", 8)
        text = _text(_call(server, "recall", {"keyword": "alpha plan zzzzz"}))
        assert "Showing top 8 of at least 8 word matches" in text


class TestFallbackCountsNothingExtra:
    def test_the_word_fallback_takes_no_count_beyond_the_searchs_own(self, store, monkeypatch):
        for i in range(30):
            store.record(f"Note {i}: the alpha thing happened in the plan again.",
                         EpisodeType.OBSERVATION)
        server = Server(store)
        calls = []
        real = store.recall

        def spy(*a, **k):
            calls.append(dict(k))
            return real(*a, **k)

        monkeypatch.setattr(store, "recall", spy)
        text = _text(_call(server, "recall", {"keyword": "alpha plan zzzzz"}))
        assert "ranked by matching words" in text
        # the exact-phrase recall and the corpus count (no keyword) aside, every keyword
        # read is the search's own fetch, never a limit-0 count
        assert [c for c in calls if c.get("keyword") and c.get("limit") == 0] == []

    def test_the_search_reports_truncation_from_its_own_counts(self, store, monkeypatch):
        for i in range(12):
            store.record(f"Note {i}: the alpha thing happened in the plan again.",
                         EpisodeType.OBSERVATION)
        matches, truncated = _retrieval.search_episodes_counted(store, "alpha plan")
        assert len(matches) == 10 and truncated is False
        monkeypatch.setattr(_retrieval, "QUERY_CANDIDATE_LIMIT", 5)
        _matches, truncated = _retrieval.search_episodes_counted(store, "alpha plan")
        assert truncated is True


class TestCrlfContinuity:
    def test_the_hash_ignores_line_ending_style(self):
        a, b, c = "x\ny\n", "x\r\ny\r\n", "x\ry\r"
        assert _retrieval.continuity_hash(a) == _retrieval.continuity_hash(b) == \
            _retrieval.continuity_hash(c)
        assert _retrieval.continuity_hash(a) != _retrieval.continuity_hash("x\ny")

    def test_a_crlf_save_passes_its_own_check_on_reload(self, store):
        _episodes(store)
        save(store, default_text(FACT, "- b — cues: hotfix").replace("\n", "\r\n"))
        key = stored_key(store)
        assert key["continuity_hash"] == _retrieval.continuity_hash(store.load_continuity())
        assert "deploy" in key["tokens"]
        assert retrieve_relevant(store, None, "deploy", max_episodes=0).facts == []
        assert [f.matched for f in retrieve_relevant(store, None, "hotfix", max_episodes=0).facts] \
            == [("hotfix",)]


class TestFailClosed:
    def test_a_failed_computation_removes_the_old_key(self, store, monkeypatch):
        _episodes(store)
        save(store, default_text(FACT))
        assert stored_key(store) is not None
        monkeypatch.setattr(_continuity_mod, "compute_durable_inert_tokens",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        store.record("another episode before the second wrap", EpisodeType.OBSERVATION)
        _result, _msgs = save(store, default_text(FACT, "- b — cues: hotfix"))
        assert stored_key(store) is None
        assert retrieve_relevant(store, None, "hotfix", max_episodes=0).facts == []

    def test_a_cold_store_still_serves_facts_through_its_valid_empty_key(self, store):
        _episodes(store, n=10)
        save(store, default_text(FACT))
        assert stored_key(store)["tokens"] == []
        assert [f.matched for f in retrieve_relevant(store, None, "deploy", max_episodes=0).facts] \
            == [("deploy",)]

    def test_mcp_recall_and_crystal_recall_follow_the_same_rule(self, store):
        store.save_continuity(_continuity(FACT))  # no key
        server = Server(store)
        assert "Durable facts" not in _text(_call(server, "recall", {"keyword": "rollback"}))
        assert "Durable facts" not in _text(
            _call(server, "crystal_recall", {"query": "rollback", "mode": "query"}))
        write_inert_key(store)
        assert _text(_call(server, "recall", {"keyword": "rollback"})).startswith(
            "Durable facts matching your words:")
        assert _text(_call(server, "crystal_recall",
                           {"query": "rollback", "mode": "query"})).startswith(
            "Durable facts matching your words:")


class TestPagingNormalisedOnce:
    def _server(self, store):
        save_cont(store, _continuity("- tree nut allergy — cues: restaurant"))
        store.record("Booked the Friday restaurant for the team with a long menu and room.",
                     EpisodeType.DECISION, timestamp="2026-10-02T09:00:00Z")
        return Server(store)

    def test_a_negative_offset_behaves_exactly_like_zero_facts_included(self, store):
        server = self._server(store)
        base = _text(_call(server, "recall", {"keyword": "restaurant"}))
        assert base.startswith("Durable facts matching your words:")
        assert _text(_call(server, "recall", {"keyword": "restaurant", "offset": -3})) == base
        assert _text(_call(server, "recall", {"keyword": "restaurant", "offset": -3.0})) == base

    def test_a_negative_limit_returns_nothing_at_all(self, store):
        server = self._server(store)
        assert _text(_call(server, "recall", {"keyword": "restaurant", "limit": -2})) == \
            "No matching episodes found."

    def test_the_descriptions_say_zero_returns_nothing_facts_included(self):
        recall = next(t for t in TOOLS if t["name"] == "recall")["description"]
        crystal = next(t for t in TOOLS if t["name"] == "crystal_recall")["description"]
        assert "limit=0 returns nothing at all, facts included" in recall
        assert "max_patterns=0 returns nothing at all, facts included" in crystal
        assert "two distinctive words of its text" in recall
        assert "two distinctive words of its text" in crystal


class TestLongPromptCountSkipsInertTokens:
    def test_inert_words_do_not_make_a_short_prompt_long(self, store):
        save_cont(store, _continuity("- tree nut allergy — cues: restaurant, time, work"))
        write_inert_key(store, tokens={"time", "work"})
        q = "what time should I book the restaurant for work"
        got = retrieve_relevant(store, None, q, max_episodes=0).facts
        assert [f.cue_matched for f in got] == [("restaurant",)]

    def test_without_inert_words_the_long_prompt_still_needs_two(self, store):
        save_cont(store, _continuity("- tree nut allergy — cues: restaurant"))
        q = "what time should I book the restaurant for work"
        assert retrieve_relevant(store, None, q, max_episodes=0).facts == []


class TestCueAndFactMatchesStaySeparate:
    FACT2 = "- the bank layout fmt_row64 starts at cutover — cues: nightly"

    def test_the_fields_and_the_union(self, store):
        save_cont(store, _continuity(self.FACT2))
        (f,) = retrieve_relevant(store, None, "nightly bank layout", max_episodes=0).facts
        assert (f.source, f.cue_matched, f.fact_matched) == ("cue", ("nightly",), ("bank", "layout"))
        assert f.matched == ("nightly", "bank", "layout")

    def test_a_fact_only_match_has_no_cue_words(self, store):
        save_cont(store, _continuity("- the bank layout fmt_row64 starts at cutover"))
        (f,) = retrieve_relevant(store, None, "bank cutover", max_episodes=0).facts
        assert (f.source, f.cue_matched, f.fact_matched) == ("fact", (), ("bank", "cutover"))

    def test_mcp_prints_both_when_both_took_part(self, store):
        save_cont(store, _continuity(self.FACT2))
        text = _text(_call(Server(store), "recall", {"keyword": "nightly bank layout"}))
        assert ("- the bank layout fmt_row64 starts at cutover "
                "(cue: nightly; matches: bank, layout)") in text


class TestInertTokensDropBeforeTheCap:
    def test_twelve_inert_tokens_then_a_cue_still_cues(self, store):
        save_cont(store, _continuity("- tree nut allergy — cues: restaurant"))
        inert = {f"inert{i}" for i in range(12)}
        write_inert_key(store, tokens=inert)
        prompt = " ".join(f"inert{i}" for i in range(12)) + " restaurant"
        got = retrieve_relevant(store, None, prompt, max_episodes=0).facts
        assert [f.cue_matched for f in got] == [("restaurant",)]

    def test_without_the_inert_set_the_cap_still_cuts_the_cue_off(self, store):
        save_cont(store, _continuity("- tree nut allergy — cues: restaurant"))
        prompt = " ".join(f"inert{i}" for i in range(12)) + " restaurant"
        assert retrieve_relevant(store, None, prompt, max_episodes=0).facts == []


class TestKeyFieldTypes:
    @pytest.mark.parametrize("episodes", [True, False, -1, "5", 1.5, None])
    def test_a_bad_episode_count_withholds_the_tier(self, store, episodes):
        import json
        save_cont(store, _continuity(FACT))
        raw = json.loads(store._get_metadata(_retrieval.INERT_TOKENS_KEY))
        assert [f.matched for f in retrieve_relevant(store, None, "rollback", max_episodes=0).facts] \
            == [("rollback",)]
        raw["episodes"] = episodes
        store._conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                            (_retrieval.INERT_TOKENS_KEY, json.dumps(raw)))
        store._conn.commit()
        assert retrieve_relevant(store, None, "rollback", max_episodes=0).facts == []
