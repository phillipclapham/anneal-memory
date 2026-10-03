"""The precision guard of the durable-fact cue tier: generic words, long prompts,
the fact-text path. The tier runs on every prompt, so a stray common word must not
bring a fact up."""

from __future__ import annotations

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


@pytest.fixture
def busy(store):
    """A store past IDF_MIN_CORPUS where "time" and "work" are in every other episode."""
    _fill(store, 80, every=2)
    store.save_continuity(_continuity(FACT))
    return store


def _matched(store, q):
    return [f.matched for f in retrieve_relevant(store, None, q, max_episodes=0).facts]


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

    def test_the_filter_is_the_stores_own_frequency(self, tmp_path):
        """The same word is generic in a store that says it constantly and a cue in one
        that does not."""
        quiet = Store(tmp_path / "quiet.db", project_name="T")
        _fill(quiet, 80, every=None)
        quiet.save_continuity(_continuity(FACT))
        try:
            assert _matched(quiet, "time") == [("time",)]
        finally:
            quiet.close()

    def test_a_store_too_small_to_tell_applies_no_filter(self, store):
        _fill(store, _retrieval.IDF_MIN_CORPUS - 1, every=1)
        store.save_continuity(_continuity(FACT))
        assert _matched(store, "time") == [("time",)]

    def test_the_threshold_is_a_module_constant_not_a_hidden_number(self, busy, monkeypatch):
        assert _matched(busy, "time") == []
        monkeypatch.setattr(_retrieval, "DURABLE_GENERIC_DF", 0.9)
        assert _matched(busy, "time") == [("time",)]

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

    def test_a_cue_match_is_unaffected_and_counts_the_text_words_too(self, store):
        store.save_continuity(_continuity(
            "- the bank layout fmt_row64 starts at cutover — cues: nightly"))
        r = retrieve_relevant(store, None, "nightly bank")
        assert [(f.source, f.matched) for f in r.facts] == [("cue", ("nightly", "bank"))]


def test_descriptions_still_describe_the_tier():
    assert "Durable facts matching your words" in next(
        t for t in TOOLS if t["name"] == "recall")["description"]
