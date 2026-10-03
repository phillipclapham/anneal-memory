"""The precision guard of the durable-fact cue tier: generic words, long prompts,
the fact-text path. The tier runs on every prompt, so a stray common word must not
bring a fact up."""

from __future__ import annotations

import json
import time

import pytest

from anneal_memory import Store, retrieve_relevant
from anneal_memory import retrieval as _retrieval
from anneal_memory.integrity import TOOLS
from anneal_memory.server import Server
from anneal_memory.types import EpisodeType

FACT = "- tree nut allergy — cues: restaurant, dinner, menu, time, work"


def _continuity(*facts: str) -> str:
    return "\n".join([
        "# T — Memory (v1)", "", "## State", "", "Working.", "",
        "## Durable Facts", "", *facts, "",
        "## Patterns", "", "## Decisions", "", "## Context", "", "Nothing yet.", "",
    ])


def _fill(store, n, *, every=None):
    """``n`` episodes; the ones whose index is divisible by ``every`` (or all, for 1)
    mention "time" and "work"."""
    for i in range(n):
        common = every is not None and i % every == 0
        store.record(
            f"Episode {i}: " + ("spent time on work for the plan " if common else "")
            + f"and reviewed the notes for item {i} with the group.",
            EpisodeType.OBSERVATION, timestamp=f"2026-09-{(i % 28) + 1:02d}T09:00:00Z")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "m.db", project_name="T")
    yield s
    s.close()


def _matched(store, q):
    return [f.matched for f in retrieve_relevant(store, None, q, max_episodes=0).facts]


def _write_inert(store) -> set[str]:
    """What the save path will do: compute the inert tokens for the store's current
    continuity and store them under the metadata key, tied to the continuity's hash."""
    facts = _retrieval.load_durable_facts(store)
    tokens = _retrieval.compute_durable_inert_tokens(store, facts)
    value = json.dumps({
        "tokens": sorted(tokens),
        "continuity_hash": _retrieval.continuity_hash(store.load_continuity()),
        "episodes": store.recall(limit=0).total_matching,
        "threshold": _retrieval.DURABLE_GENERIC_DF,
    })
    store._conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        (_retrieval.INERT_TOKENS_KEY, value))
    store._conn.commit()
    return tokens


@pytest.fixture
def busy(store):
    """A store past IDF_MIN_CORPUS where "time" and "work" are in every other episode,
    with its inert tokens computed and stored."""
    _fill(store, 80, every=2)
    store.save_continuity(_continuity(FACT))
    _write_inert(store)
    return store


class TestGenericWordsNeverCue:
    def test_a_generic_cue_word_alone_never_surfaces_a_fact(self, busy):
        assert _matched(busy, "time") == []
        assert _matched(busy, "what time") == []
        assert _matched(busy, "work") == []

    def test_a_specific_cue_still_surfaces_it(self, busy):
        assert _matched(busy, "restaurant") == [("restaurant",)]
        assert _matched(busy, "any good restaurant downtown?") == [("restaurant",)]

    def test_a_generic_word_beside_a_specific_one_does_not_count_as_a_match(self, busy):
        assert _matched(busy, "restaurant time") == [("restaurant",)]

    def test_the_computed_set_is_the_stores_own_frequency(self, tmp_path, busy):
        assert {"time", "work"} <= _retrieval.compute_durable_inert_tokens(
            busy, _retrieval.load_durable_facts(busy))
        quiet = Store(tmp_path / "quiet.db", project_name="T")
        _fill(quiet, 80, every=None)
        quiet.save_continuity(_continuity(FACT))
        try:
            assert _retrieval.compute_durable_inert_tokens(
                quiet, _retrieval.load_durable_facts(quiet)) == set()
            _write_inert(quiet)
            assert _matched(quiet, "time") == [("time",)]
        finally:
            quiet.close()

    def test_no_stored_set_means_no_filter_and_nothing_is_counted_at_prompt_time(
        self, store, monkeypatch
    ):
        _fill(store, 80, every=2)
        store.save_continuity(_continuity(FACT))  # no key written
        real = store.recall
        calls = []

        def spy(*a, **k):
            calls.append(k)
            return real(*a, **k)

        monkeypatch.setattr(store, "recall", spy)
        assert _matched(store, "time") == [("time",)]
        assert [c for c in calls if "keyword" in c and c.get("limit") == 0] == []

    def test_a_stale_set_is_ignored(self, busy):
        assert _matched(busy, "time") == []
        busy.save_continuity(_continuity(FACT, "- another fact — cues: spaceship"))
        assert _matched(busy, "time") == [("time",)]  # the key was for the old continuity

    def test_a_store_too_small_to_tell_has_an_empty_set(self, store):
        _fill(store, _retrieval.IDF_MIN_CORPUS - 1, every=1)
        store.save_continuity(_continuity(FACT))
        assert _retrieval.compute_durable_inert_tokens(
            store, _retrieval.load_durable_facts(store)) == set()
        assert _matched(store, "time") == [("time",)]

    def test_document_frequency_counts_whole_words(self, store):
        for i in range(60):
            store.record(
                f"Note {i}: the category of concatenate calls is {'a cat' if i < 5 else 'x'}.",
                EpisodeType.OBSERVATION)
            store.record(
                f"Memo {i}: the current parent account {'pays rent' if i < 5 else 'is x'}.",
                EpisodeType.OBSERVATION)
        store.save_continuity(_continuity("- pet — cues: cat, category", "- lease — cues: rent, current"))
        inert = _retrieval.compute_durable_inert_tokens(
            store, _retrieval.load_durable_facts(store))
        # "cat" is a whole word in 5 of 120 episodes (4%); a substring count would say 60+.
        assert "cat" not in inert and "rent" not in inert
        assert "category" in inert and "current" in inert  # whole words in half the store

    def test_the_threshold_is_a_module_constant_not_a_hidden_number(self, tmp_path, monkeypatch):
        s = Store(tmp_path / "t.db", project_name="T")
        _fill(s, 80, every=2)
        s.save_continuity(_continuity(FACT))
        facts = _retrieval.load_durable_facts(s)
        try:
            assert "time" in _retrieval.compute_durable_inert_tokens(s, facts)
            monkeypatch.setattr(_retrieval, "DURABLE_GENERIC_DF", 0.9)
            assert _retrieval.compute_durable_inert_tokens(s, facts) == set()
        finally:
            s.close()

    def test_mcp_recall_applies_it_too(self, busy):
        server = Server(busy)
        r = server._handle_tools_call({"name": "recall", "arguments": {"keyword": "time"}})
        assert "Durable facts" not in r["content"][0]["text"]
        r = server._handle_tools_call({"name": "recall", "arguments": {"keyword": "restaurant"}})
        assert r["content"][0]["text"].startswith("Durable facts matching your words:")


class TestLongPromptsNeedTwoTokens:
    FACT2 = "- tree nut allergy — cues: dinner, menu, restaurant"

    def _s(self, store):
        store.save_continuity(_continuity(self.FACT2))
        return store

    def test_one_stray_cue_word_in_a_long_prompt_surfaces_nothing(self, store):
        self._s(store)
        q = "plan the quarterly review dinner agenda for eight senior managers"
        assert _matched(store, q) == []

    def test_two_distinct_cue_words_in_a_long_prompt_do(self, store):
        self._s(store)
        q = "plan the quarterly review dinner menu for eight senior managers"
        assert _matched(store, q) == [("dinner", "menu")]

    def test_a_short_prompt_cues_on_one_token(self, store):
        self._s(store)
        for q in ("restaurant?", "dinner plans", "any good restaurant downtown?"):
            assert len(_matched(store, q)) == 1, q

    def test_the_cutoff_is_a_module_constant(self, store, monkeypatch):
        self._s(store)
        q = "plan the quarterly review dinner agenda for eight senior managers"
        assert _matched(store, q) == []
        monkeypatch.setattr(_retrieval, "DURABLE_SHORT_PROMPT_TOKENS", 50)
        assert _matched(store, q) == [("dinner",)]


class TestFactTextPath:
    def test_one_word_of_the_fact_text_does_not_cue_it(self, store):
        store.save_continuity(_continuity("- the bank layout fmt_row64 starts at cutover"))
        assert _matched(store, "cutover") == []
        assert _matched(store, "bank notes") == []

    def test_two_words_of_the_fact_text_do(self, store):
        store.save_continuity(_continuity("- the bank layout fmt_row64 starts at cutover"))
        assert _matched(store, "bank cutover") == [("bank", "cutover")]

    def test_a_cue_match_is_unaffected_and_shows_only_the_cue_words(self, store):
        store.save_continuity(_continuity(
            "- the bank layout fmt_row64 starts at cutover — cues: nightly"))
        r = retrieve_relevant(store, None, "nightly bank")
        assert [(f.source, f.matched) for f in r.facts] == [("cue", ("nightly",))]


class TestDistinctQueryTokens:
    def test_a_token_matching_a_cue_and_a_fact_word_counts_once(self, store):
        store.save_continuity(_continuity("- rotate the API tokens weekly — cues: token"))
        q = "explain token bucket algorithm"
        assert _matched(store, q) == []  # one query token, in a 4-token prompt
        assert _matched(store, "token") == [("token",)]  # a one-word prompt still cues
        assert _matched(store, "explain token tokens") == []  # inflections of one token

    def test_two_different_query_tokens_still_count(self, store):
        store.save_continuity(_continuity("- rotate the API tokens weekly — cues: token, rotation"))
        assert _matched(store, "explain token rotation policy") == [("token", "rotation")]


class TestStemmingCollisions:
    @pytest.mark.parametrize("word, collides_with", [
        ("rating", "rat"), ("files", "fil"), ("lines", "lin"),
    ])
    def test_no_three_letter_stem_from_es_or_ing(self, word, collides_with):
        assert collides_with not in _retrieval._token_forms(word)
        assert not (_retrieval._token_forms(word) & _retrieval._token_forms(collides_with))

    def test_the_real_inflections_still_agree(self):
        for a, b in (("files", "file"), ("lines", "line"), ("ratings", "rating"),
                     ("recipes", "recipe"), ("restaurants", "restaurant"), ("menus", "menu"),
                     ("meetings", "meeting"), ("running", "runn")):
            assert _retrieval._token_forms(a) & _retrieval._token_forms(b), (a, b)

    def test_news_does_not_cue_new_because_new_is_a_stopword(self, store):
        store.save_continuity(_continuity("- tone — cues: new, rat, fil, lin"))
        for q in ("news", "rating", "files", "lines"):
            assert _matched(store, q) == [], q


class TestCostIsBounded:
    def test_a_160_token_prompt_on_a_12k_episode_store_is_fast(self, tmp_path):
        s = Store(tmp_path / "big.db", project_name="T", audit=False)
        words = ("work time project deploy review commit branch merge test build ship "
                 "release server config script agent memory session wrap").split()
        for i in range(12000):
            s.record(" ".join(words[(i + j) % len(words)] for j in range(30)) + f" ep{i}",
                     EpisodeType.OBSERVATION,
                     timestamp=f"2026-0{1 + i % 9}-{1 + i % 28:02d}T10:{i % 60:02d}:00Z")
        s.save_continuity(_continuity(FACT, "- bank — cues: cutover, nightly"))
        prompt = " ".join(f"word{i}" for i in range(20)) + " restaurant dinner " + " ".join(
            f"more{i}" for i in range(140))
        assert len(prompt.split()) == 162
        try:
            retrieve_relevant(s, None, prompt, max_episodes=0)  # warm
            t0 = time.perf_counter()
            r = retrieve_relevant(s, None, prompt, max_episodes=0)
            ms = (time.perf_counter() - t0) * 1000
            assert ms < 50, ms
            assert r.facts == []  # the cue words sit past the 12th distinct token
        finally:
            s.close()

    def test_only_the_first_twelve_distinct_tokens_are_considered(self, store):
        store.save_continuity(_continuity(FACT))
        early = "restaurant dinner " + " ".join(f"filler{i}" for i in range(10))
        late = " ".join(f"filler{i}" for i in range(12)) + " restaurant dinner"
        assert _matched(store, early) == [("restaurant", "dinner")]
        assert _matched(store, late) == []


class TestTierNeverRaises:
    def test_any_failure_gives_an_empty_tier(self, store, monkeypatch):
        store.save_continuity(_continuity(FACT))
        monkeypatch.setattr(_retrieval, "match_durable_facts",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        assert retrieve_relevant(store, None, "restaurant").facts == []
        assert _retrieval.durable_facts_for(store, "restaurant") == []

    def test_a_corrupt_stored_set_is_ignored(self, store):
        store.save_continuity(_continuity(FACT))
        store._conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                            (_retrieval.INERT_TOKENS_KEY, "{not json"))
        store._conn.commit()
        assert _matched(store, "restaurant") == [("restaurant",)]


def test_descriptions_still_describe_the_tier():
    assert "Durable facts matching your words" in next(
        t for t in TOOLS if t["name"] == "recall")["description"]


class TestMcpRecallFactsBlock:
    def _server(self, store):
        store.save_continuity(_continuity(
            "- tree nut allergy — cues: restaurant, restaurants, dinner",
            "- the bank layout fmt_row64 starts at cutover"))
        store.record("Booked the Friday restaurant for the team dinner with a long menu "
                     "and a private room.", EpisodeType.DECISION,
                     source="agent", timestamp="2026-10-02T09:00:00Z")
        return Server(store)

    def _text(self, server, **args):
        return server._handle_tools_call({"name": "recall", "arguments": args})["content"][0]["text"]

    def test_the_block_shows_the_fact_not_the_cue_list_and_each_family_once(self, store):
        text = self._text(self._server(store), keyword="restaurant dinner")
        head = text.partition("\n\n")[0]
        assert head == ("Durable facts matching your words:\n"
                        "- tree nut allergy (cue: restaurant, dinner)")
        assert "cues:" not in head

    def test_a_fact_text_match_is_labelled_matches(self, store):
        text = self._text(self._server(store), keyword="bank cutover")
        assert "- the bank layout fmt_row64 starts at cutover (matches: bank, cutover)" in text

    def test_no_facts_for_limit_zero_or_any_episode_filter(self, store):
        server = self._server(store)
        for args in ({"limit": 0}, {"since": "2026-01-01T00:00:00Z"},
                     {"until": "2027-01-01T00:00:00Z"}, {"source": "agent"},
                     {"episode_type": "decision"}):
            text = self._text(server, keyword="restaurant dinner", **args)
            assert "Durable facts" not in text, args

    def test_the_no_match_line_stays_after_the_block(self, store):
        server = self._server(store)
        assert self._text(server, keyword="cutover bank") == (
            "Durable facts matching your words:\n"
            "- the bank layout fmt_row64 starts at cutover (matches: bank, cutover)"
            "\n\nNo matching episodes found.")
