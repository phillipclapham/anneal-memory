"""1.0.2: a save forms no association link, and says so on the store.

Phill ruled the cut 2026-10-23 (both Hebbian graphs and the affect weight stop
forming; the public association APIs stay as legacy until 2.0).
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


def test_a_save_forms_no_link_and_records_the_capability(tmp_path):
    store = Store(tmp_path / "m.db", project_name="Cut")
    try:
        seed_prior_levels(store, {"real_pattern": 1, "other_pattern": 1})
        assert store.association_formation() == {
            "mode": "enabled", "off_at_wrap": None, "off_by_version": None}
        ids = _episodes(store)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _save(store, ids,
                           affective_state=AffectiveState(tag="dreaming", intensity=0.9))
        assert result["graduations_validated"] >= 1  # the immune system still ran
        assert (result["associations_formed"], result["associations_strengthened"],
                result["associations_decayed"]) == (0, 0, 0)
        assert _links(store) == []
        assert store._conn.execute(
            "SELECT COUNT(*) FROM pattern_associations").fetchone()[0] == 0
        wrap_id = store._conn.execute("SELECT MAX(id) FROM wraps").fetchone()[0]
        assert result["wrap_id"] == wrap_id == result["wrap_result"]["wrap_id"]
        formation = store.association_formation()
        assert formation["mode"] == "disabled"
        assert formation["off_at_wrap"] == wrap_id
        assert formation["off_by_version"]
        # The deprecated affect input is accepted, ignored, and said so.
        assert any("affective_state is deprecated" in str(w.message) for w in caught)

        # A later save leaves the first wrap as the record of when it turned off.
        ids2 = _episodes(store)
        result2 = _save(store, ids2)
        assert result2["wrap_id"] == wrap_id + 1
        assert store.association_formation()["off_at_wrap"] == wrap_id
    finally:
        store.close()


def test_legacy_apis_answer_and_refuse_to_form(tmp_path):
    store = Store(tmp_path / "m.db", project_name="Cut")
    try:
        a, b = _episodes(store)
        # A store no 1.0.2 save has touched still forms on the legacy API.
        assert store.record_associations({(a, b)}) == (1, 0)
        _save(store, _episodes(store))
        for call in (
            lambda: store.record_associations({(a, b)}),
            lambda: store.decay_associations(),
            lambda: store.seed_pattern_co_graduation(["x_pattern", "y_pattern"]),
        ):
            with pytest.raises(AssociationFormationOffError, match="removed in 2.0"):
                call()
        # A caught refusal must not cost the caller's own writes in an open batch
        # (L2 1023+25, reproduced: raised inside the store boundary, the refusal
        # rolled the whole batch back, so the episode below was silently lost).
        with store._batch():
            kept = store.record("written in the same batch", EpisodeType.OBSERVATION)
            with pytest.raises(AssociationFormationOffError):
                store.record_associations({(a, b)})
        assert store.get(kept.id) is not None
        # Reads still answer.
        assert [(p.episode_a, p.episode_b) for p in store.get_associations([a])] == [
            tuple(sorted((a, b)))]
        assert store.association_stats().total_links == 1
        # The compatibility drain discards and reports zero work.
        metrics = store.drain_co_surface_events(
            [{"event_id": "e1", "session_id": "s1", "names": ["x_pattern", "y_pattern"]}],
            today=TODAY,
        )
        assert metrics["events_applied"] == metrics["pairs_formed"] == 0
        assert metrics["discarded"] == 1
        assert store._conn.execute(
            "SELECT COUNT(*) FROM pattern_associations").fetchone()[0] == 0
    finally:
        store.close()


def test_an_existing_graph_reads_unchanged_after_the_first_save(tmp_path):
    store = Store(tmp_path / "m.db", project_name="Cut")
    try:
        ids = _episodes(store, 4)
        store.record_associations({(ids[0], ids[1])}, {(ids[2], ids[3])})
        before = _links(store)
        assert len(before) == 2
        _save(store, _episodes(store))
        # Not decayed, not strengthened, not removed: 1.0.1 decayed every link on
        # every save it did not reinforce.
        assert _links(store) == before
    finally:
        store.close()
    reopened = Store(tmp_path / "m.db", project_name="Cut")
    try:
        assert _links(reopened) == before
        assert reopened.association_formation()["mode"] == "disabled"
    finally:
        reopened.close()


def test_prepare_wrap_gives_the_composer_no_association_context(tmp_path):
    store = Store(tmp_path / "m.db", project_name="Cut")
    try:
        a, b = _episodes(store)
        # A strong link between two window episodes: before 1.0.2 this reached the
        # composer as "## Episode Associations".
        store.record_associations({(a, b)})
        store.record_associations({(a, b)})
        res = prepare_wrap(store)
        assert res["status"] == "ready"
        assert res["assoc_context"] is None
        instructions = res["package"]["instructions"]
        assert "Hebbian" not in instructions
        assert "affective_state" not in instructions
        assert "Episode Associations" not in (res["package"].get("episodes") or "")
    finally:
        store.close()
