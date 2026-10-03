"""The save path writes the recall tier's inert-token set, and warns about cues that
cannot work. The set is computed from the store's own episodes for the exact text saved."""

from __future__ import annotations

import hashlib
import json
import warnings

import pytest

from anneal_memory import Store, prepare_wrap, retrieve_relevant, validated_save_continuity
from anneal_memory import retrieval as _retrieval
from anneal_memory.types import EpisodeType

TODAY = "2026-10-03"
FACT = "- release process — cues: deploy, rollback, db"


def default_text(*durable: str, section: bool = True) -> str:
    block = ("## Durable Facts\n" + "\n".join(durable) + "\n\n") if section else ""
    return (
        "# T — Memory (v1)\n\n## State\nworking\n\n"
        f"{block}"
        "## Patterns\n- x | 1x (2026-10-03)\n\n## Decisions\n- d\n\n## Context\nc\n"
    )


def _episodes(store, n=60):
    """n episodes; every other one mentions "deploy" (well over 10% of the store)."""
    for i in range(n):
        store.record(
            f"Episode {i}: " + ("the deploy went out after the review" if i % 2 == 0
                                else "reviewed the notes for the group")
            + f" for item {i}.", EpisodeType.OBSERVATION)


def save(store, text):
    assert prepare_wrap(store)["status"] == "ready"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = validated_save_continuity(store, text, today=TODAY)
    return result, [str(w.message) for w in caught]


def stored_key(store):
    raw = store._get_metadata(_retrieval.INERT_TOKENS_KEY)
    return json.loads(raw) if raw else None


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "m.db", project_name="T")
    yield s
    s.close()


class TestKeyWrittenAtSave:
    def test_the_save_writes_the_key_for_the_exact_saved_text(self, store):
        _episodes(store)
        assert stored_key(store) is None
        save(store, default_text(FACT))
        key = stored_key(store)
        saved = store.load_continuity()
        assert key["continuity_hash"] == hashlib.sha256(saved.encode("utf-8")).hexdigest()
        assert "deploy" in key["tokens"] and "rollback" not in key["tokens"]
        assert key["episodes"] == store.recall(limit=0).total_matching
        assert key["threshold"] == _retrieval.DURABLE_GENERIC_DF

    def test_the_next_prompt_reads_it(self, store):
        _episodes(store)
        save(store, default_text(FACT))
        assert retrieve_relevant(store, None, "deploy", max_episodes=0).facts == []
        got = retrieve_relevant(store, None, "rollback", max_episodes=0).facts
        assert [f.matched for f in got] == [("rollback",)]

    def test_a_stale_hash_turns_the_filter_off(self, store):
        _episodes(store)
        save(store, default_text(FACT))
        assert retrieve_relevant(store, None, "deploy", max_episodes=0).facts == []
        # the continuity changes outside the save path; the stored set no longer applies
        store.save_continuity(store.load_continuity() + "\nextra\n")
        got = retrieve_relevant(store, None, "deploy", max_episodes=0).facts
        assert [f.matched for f in got] == [("deploy",)]

    def test_each_save_replaces_the_key(self, store):
        _episodes(store)
        save(store, default_text(FACT))
        first = stored_key(store)["continuity_hash"]
        store.record("another episode before the second wrap", EpisodeType.OBSERVATION)
        save(store, default_text(FACT, "- another fact — cues: spaceship"))
        key = stored_key(store)
        assert key["continuity_hash"] != first
        assert key["continuity_hash"] == hashlib.sha256(
            store.load_continuity().encode("utf-8")).hexdigest()

    def test_a_small_store_gets_an_empty_set_with_a_current_hash(self, store):
        _episodes(store, n=10)
        save(store, default_text(FACT))
        key = stored_key(store)
        assert key["tokens"] == [] and key["episodes"] >= 10

    def test_a_schema_without_a_durable_section_writes_nothing(self, tmp_path):
        old = [{"heading": "State", "role": "live-state"},
               {"heading": "Patterns", "role": "graduating"},
               {"heading": "Decisions", "role": "decisions"},
               {"heading": "Context", "role": "narrative"}]
        s = Store(tmp_path / "old.db", project_name="T", section_schema=old)
        try:
            _episodes(s)
            _result, msgs = save(s, default_text(section=False))
            assert stored_key(s) is None
            assert not [m for m in msgs if "cue " in m]
        finally:
            s.close()


class TestCueWarnings:
    def test_an_inert_cue_and_a_too_short_cue_are_warned_about(self, store):
        _episodes(store)
        result, msgs = save(store, default_text(FACT))
        inert = ("cue 'deploy' appears in more than 10% of this store's episodes, so it "
                 "will not cue anything; add a more specific cue")
        short = "cue 'db' is too short to match; spell it out"
        assert inert in msgs and short in msgs
        assert inert in result["durable_warnings"] and short in result["durable_warnings"]
        assert not any("rollback" in m for m in msgs)

    def test_each_cue_is_named_once_across_facts(self, store):
        _episodes(store)
        _result, msgs = save(store, default_text(
            "- a — cues: deploy, db", "- b — cues: deploy, db"))
        assert sum("cue 'deploy'" in m for m in msgs) == 1
        assert sum("cue 'db'" in m for m in msgs) == 1

    def test_a_clean_cue_list_adds_no_cue_warning(self, store):
        _episodes(store)
        _result, msgs = save(store, default_text("- release — cues: rollback, hotfix"))
        assert not [m for m in msgs if m.startswith("cue ")]
