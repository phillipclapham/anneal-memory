"""1.0.2: a save forms no association link, and says so on the store.

Phill ruled the cut 2026-10-23 (both Hebbian graphs and the affect weight stop
forming; the public association APIs stay as legacy until a later MAJOR release removes them).
"""
from __future__ import annotations

import warnings

import pytest

from anneal_memory import (
    AssociationFormationOffError,
    prepare_wrap,
    validated_save_continuity,
)
from anneal_memory.store import Store
from anneal_memory.types import AffectiveState, EpisodeType
from tests.prior_seed import seed_prior_levels

TODAY = "2026-06-02"

# Two real ids co-cited on one line: before 1.0.2 this formed a direct link.
CO_CITED = (
    "## State\nactive.\n\n"
    "## Patterns\n"
    '- real_pattern | 2x (2026-06-02) [evidence: {ep0}, {ep1} '
    '"genuine substrate observation discipline rotation co-citation"]\n'
    '- other_pattern | 2x (2026-06-02) [evidence: {ep1} '
    '"substrate observation discipline rotation memory topic"]\n\n'
    "## Decisions\n- d.\n\n"
    "## Context\n- c.\n"
)


def _episodes(store: Store, n: int = 2) -> list[str]:
    return [
        store.record(
            f"substrate observation about discipline rotation memory topic {i}",
            EpisodeType.OBSERVATION,
        ).id
        for i in range(n)
    ]


def _save(store: Store, ids: list[str], **kwargs):
    res = prepare_wrap(store)
    text = CO_CITED
    for i, ep_id in enumerate(ids):
        text = text.replace(f"{{ep{i}}}", ep_id)
    return validated_save_continuity(
        store, text, today=TODAY, wrap_token=res["wrap_token"], **kwargs
    )


def _links(store: Store) -> list[tuple]:
    return store._conn.execute(
        "SELECT episode_a, episode_b, strength, co_citations, last_strengthened "
        "FROM associations ORDER BY episode_a, episode_b"
    ).fetchall()








