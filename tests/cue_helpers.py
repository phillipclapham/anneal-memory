"""Helpers for the durable-fact cue tests: save a continuity the way the wrap does,
including the store's inert-token key, so the recall tier has a valid set to read."""

from __future__ import annotations

import json

from anneal_memory import Store
from anneal_memory import retrieval as _retrieval


def write_inert_key(store: Store, *, tokens: set[str] | None = None) -> set[str]:
    """Write a valid ``durable_inert_tokens`` key for the store's current continuity:
    the computed set, or ``tokens`` when given."""
    facts = _retrieval.load_durable_facts(store)
    chosen = (
        _retrieval.compute_durable_inert_tokens(store, facts) if tokens is None else tokens
    )
    store._conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        (_retrieval.INERT_TOKENS_KEY, json.dumps({
            "tokens": sorted(chosen),
            "continuity_hash": _retrieval.continuity_hash(store.load_continuity()),
            "episodes": store.recall(limit=0).total_matching,
            "threshold": _retrieval.DURABLE_GENERIC_DF,
        })))
    store._conn.commit()
    return set(chosen)


def save_cont(store: Store, text: str) -> None:
    """``store.save_continuity(text)`` plus a valid inert-token key for it."""
    store.save_continuity(text)
    write_inert_key(store)
