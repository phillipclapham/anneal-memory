"""Cue wiring: the durable facts a query cues surface on the recall paths.

``retrieve_relevant`` returns them as ``RelevantResult.facts``; MCP ``recall`` and
``crystal_recall`` list them first. The continuity under test carries the
``## Durable Facts`` section the schema's ``durable`` role names.
"""

from __future__ import annotations

from datetime import date

import pytest

from anneal_memory import (
    CrystalStore,
    RelevantFact,
    RelevantResult,
    Store,
    retrieve_relevant,
)
from anneal_memory import retrieval as _retrieval
from anneal_memory.integrity import TOOLS
from anneal_memory.server import Server
from anneal_memory.types import EpisodeType

T0 = date(2026, 10, 3)
ALLERGY = "- tree nut allergy — cues: restaurant, dinner, recipe, food, menu"


def _continuity(*fact_lines: str, section: bool = True) -> str:
    parts = ["# T — Memory (v1)", "", "## State", "", "Working.", ""]
    if section:
        parts += ["## Durable Facts", "", *fact_lines, ""]
    parts += ["## Patterns", "", "## Decisions", "", "## Context", "", "Nothing yet.", ""]
    return "\n".join(parts)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "m.db", project_name="T")
    yield s
    s.close()


@pytest.fixture
def server(store):
    return Server(store)


def _call(server, name, arguments=None):
    return server._handle_tools_call({"name": name, "arguments": arguments or {}})


def _text(r):
    return r["content"][0]["text"]


def _seed_episodes(store):
    store.record(
        "Booked the Friday team dinner at a Thai place with a long vegetarian menu "
        "and a private room for eight people.",
        EpisodeType.DECISION, timestamp="2026-10-02T09:00:00Z")
    store.record(
        "Reviewed the nightly export job and the finance reconciliation rows for the "
        "quarter, nothing unusual to report this week.",
        EpisodeType.OBSERVATION, timestamp="2026-10-01T09:00:00Z")


class TestRetrieveRelevantFacts:
    def test_a_cue_word_in_the_prompt_surfaces_the_fact(self, store):
        store.save_continuity(_continuity(ALLERGY))
        _seed_episodes(store)
        on = retrieve_relevant(store, None, "any good restaurant downtown?")
        assert on.facts == [RelevantFact(
            fact="tree nut allergy", line=ALLERGY, matched=("restaurant",), source="cue")]
        off = retrieve_relevant(store, None, "any good restaurant downtown?", durable=False)
        assert off.facts == []
        assert (on.patterns, on.episodes, on.query_keywords) == (
            off.patterns, off.episodes, off.query_keywords)

    def test_a_one_word_prompt_still_cues_and_stems(self, store):
        store.save_continuity(_continuity(ALLERGY))
        for q in ("restaurants", "restaurant?", "Recipes", "menus"):
            for mode in ("prompt", "query"):
                r = retrieve_relevant(store, None, q, mode=mode)
                assert [f.source for f in r.facts] == ["cue"], (q, mode)
                assert r.episodes == [] and r.patterns == []
        assert retrieve_relevant(store, None, "recipes").facts[0].matched == ("recipe",)

    def test_whole_token_equality_not_substring(self, store):
        store.save_continuity(_continuity(ALLERGY))
        for q in ("restaurateur", "menuet", "foodie", "restaurateurs"):
            assert retrieve_relevant(store, None, q).facts == [], q

    def test_stopwords_and_short_tokens_never_match(self, store):
        store.save_continuity(_continuity("- a rule — cues: the, for, ox, door"))
        assert retrieve_relevant(store, None, "the for ox").facts == []
        assert [f.matched for f in retrieve_relevant(store, None, "doors").facts] == [("door",)]

    def test_fact_text_alone_cues_only_through_two_distinct_words(self, store):
        store.save_continuity(_continuity("- the bank layout fmt_row64 starts at cutover"))
        r = retrieve_relevant(store, None, "cutover and bank layout notes")
        assert [(f.source, f.matched) for f in r.facts] == [
            ("fact", ("bank", "layout", "cutover"))]
        # one word of the fact text is not enough
        assert retrieve_relevant(store, None, "tell me about the cutover plan").facts == []
        assert retrieve_relevant(store, None, "cutover").facts == []

    def test_cap_of_two_ranked_by_distinct_tokens_then_section_order(self, store):
        store.save_continuity(_continuity(
            "- first fact — cues: dinner",
            "- second fact — cues: dinner, menu",
            "- third fact — cues: dinner",
            "- fourth fact — cues: dinner, menu, food",
        ))
        assert _retrieval.MAX_DURABLE_FACTS == 2
        r = retrieve_relevant(store, None, "dinner menu food")
        assert [f.fact for f in r.facts] == ["fourth fact", "second fact"]
        r = retrieve_relevant(store, None, "dinner")
        assert [f.fact for f in r.facts] == ["first fact", "second fact"]

    def test_no_continuity_or_no_section_is_unchanged(self, store):
        _seed_episodes(store)
        q = "team dinner menu at a restaurant"
        base = retrieve_relevant(store, None, q, durable=False)
        assert retrieve_relevant(store, None, q).facts == []  # no continuity at all
        store.save_continuity(_continuity(ALLERGY, section=False))
        again = retrieve_relevant(store, None, q)
        assert again.facts == []
        assert again == base

    def test_durable_false_leaves_facts_empty(self, store):
        store.save_continuity(_continuity(ALLERGY))
        assert retrieve_relevant(store, None, "restaurant", durable=False).facts == []

    def test_an_unreadable_continuity_gives_no_facts_and_no_exception(self, store, monkeypatch):
        def boom():
            raise OSError("disk gone")
        monkeypatch.setattr(store, "load_continuity", boom)
        assert retrieve_relevant(store, None, "restaurant").facts == []

    def test_result_default_is_empty_facts(self):
        assert RelevantResult(patterns=[], episodes=[]).facts == []


class TestMcpRecallFacts:
    def test_facts_block_comes_first_then_the_episodes(self, server, store):
        store.save_continuity(_continuity(ALLERGY))
        _seed_episodes(store)
        text = _text(_call(server, "recall", {"keyword": "dinner plans"}))
        assert text.startswith(
            "Durable facts matching your words:\n" + ALLERGY + " (cue: dinner)\n\n")
        assert text.partition("\n\n")[2].startswith(
            "No episode contains the exact phrase; ranked by matching words")

    def test_a_fact_with_no_episode_replaces_no_matching(self, server, store):
        store.save_continuity(_continuity(ALLERGY))
        text = _text(_call(server, "recall", {"keyword": "dinner plans"}))
        assert text == "Durable facts matching your words:\n" + ALLERGY + " (cue: dinner)"

    def test_no_keyword_is_unchanged(self, server, store):
        store.save_continuity(_continuity(ALLERGY))
        assert _text(_call(server, "recall", {})) == "No matching episodes found."
        _seed_episodes(store)
        assert "Durable facts" not in _text(_call(server, "recall", {}))

    def test_a_later_page_does_not_repeat_the_facts(self, server, store):
        store.save_continuity(_continuity(ALLERGY))
        _seed_episodes(store)
        text = _text(_call(server, "recall", {"keyword": "dinner", "offset": 1}))
        assert "Durable facts" not in text

    def test_no_section_leaves_recall_exactly_as_before(self, server, store):
        store.save_continuity(_continuity(ALLERGY, section=False))
        _seed_episodes(store)
        text = _text(_call(server, "recall", {"keyword": "dinner plans"}))
        assert text.startswith("No episode contains the exact phrase; ranked by")
        assert "Durable facts" not in text
        assert _text(_call(server, "recall", {"keyword": "zebraquartz"})) == \
            "No matching episodes found."

    def test_error_results_pass_through(self, server, store):
        store.save_continuity(_continuity(ALLERGY))
        r = _call(server, "recall", {"keyword": "dinner", "limit": 2.5})
        assert r.get("isError") is True

    def test_description_mentions_the_facts_block(self):
        recall = next(t for t in TOOLS if t["name"] == "recall")
        assert "Durable facts matching your words" in recall["description"]


class TestMcpCrystalRecallFacts:
    def test_facts_listed_first_with_patterns(self, server, store):
        store.save_continuity(_continuity(ALLERGY))
        CrystalStore(server._crystal_path).crystallize(
            name="dinner_menu_discipline", level=2,
            explanation="check the dinner menu against every allergy", tags=[])
        text = _text(_call(server, "crystal_recall", {"query": "dinner menu allergy check"}))
        assert text.startswith("Durable facts matching your words:\n" + ALLERGY)
        assert "Found 1 crystallized pattern(s):" in text
        assert "dinner_menu_discipline" in text

    def test_a_fact_with_no_pattern_replaces_no_match(self, server, store):
        store.save_continuity(_continuity(ALLERGY))
        text = _text(_call(server, "crystal_recall", {"query": "restaurant", "mode": "query"}))
        assert text == "Durable facts matching your words:\n" + ALLERGY + " (cue: restaurant)"

    def test_no_section_is_unchanged(self, server, store):
        assert _text(_call(server, "crystal_recall", {"query": "restaurant dinner"})) == \
            "No crystallized patterns matched."

    def test_description_mentions_the_facts_block(self):
        cr = next(t for t in TOOLS if t["name"] == "crystal_recall")
        assert "Durable facts matching your words" in cr["description"]
