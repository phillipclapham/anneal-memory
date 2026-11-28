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




