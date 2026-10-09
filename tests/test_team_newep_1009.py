"""Slice (A), Phill 2026-10-09: a team snapshot that changes an entry's text makes a
NEW episode, linked over the old one by a team-owned link; the old episode keeps its
id, text and everything its text earned. Both tests are the CAP-08 L3 r2 codex HIGHs
(1009+22), each run on 5227fc4 before the fix (anneal project_memory
seat_1009_A/before/residue_1009-22-cap08.txt)."""
from __future__ import annotations

import json

from anneal_memory import EpisodeType, Store
from anneal_memory.continuity import prepare_wrap, validated_save_continuity
from anneal_memory.store import SupersessionError
from anneal_memory.team import import_ledger
from tests.test_team_import import ledger
from tests.test_team_snapshot_v3 import v3

import pytest

X = "deploy gate: the nightly export runs at 02:00 and needs two reviewers"
Y = "the cafeteria menu changes on mondays and the soup rotates weekly"
EXPL = "nightly export runs at 02:00 and needs two reviewers"
HEAD = "# T — Memory (v1)\n\n## State\nActive.\n\n"
TAIL = "## Decisions\nNone.\n\n## Context\nSession.\n"


def entry(words):
    return ledger("bob", [{"type": "decision", "kind": "practice", "owner": "client:acme",
                           "words": words}])[0]


D_X, D_Y = entry(X), entry(Y)
D_ID = json.loads(D_X)["id"]


def held(s):
    return s._conn.execute("SELECT episode_id FROM team_entries WHERE entry_id = ?",
                           (D_ID,)).fetchone()[0]


def save(s, pattern, today):
    return validated_save_continuity(s, HEAD + "## Patterns\n" + pattern + "\n\n" + TAIL,
                                     today=today)


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "m.db", project_name="T")
    s.record("Session start.", EpisodeType.OBSERVATION)
    prepare_wrap(s)
    save(s, "- deploy_gate | 1x (2026-10-07)", "2026-10-07")
    import_ledger(s, v3([(D_X, True, [])], seq=1))
    yield s
    s.close()


def _line(old, own):
    return f'- deploy_gate | 2x (2026-10-08) [evidence: {old}, {own} "{EXPL}"]'


def test_grounding_earned_by_the_old_text_stays_with_the_old_text(store):
    """HIGH 1: a rung grounded on X stayed attached to the id the import rewrote to Y."""
    old = held(store)
    own = store.record(f"I checked it myself: {EXPL}.", EpisodeType.OBSERVATION)
    prepare_wrap(store)
    assert save(store, _line(old, own.id), "2026-10-08")["graduations_validated"] == 1
    r = import_ledger(store, v3([(D_Y, True, [])], seq=2))
    new = held(store)
    assert r.replaced == [{"id": D_ID, "old": old, "new": new, "linked": True}]
    assert new != old and X in store.get(old).content and Y in store.get(new).content
    grounded = {i for rung in store.pattern_grounding()["deploy_gate"].values()
                for g in rung for i in g["episodes"]}
    assert old in grounded and new not in grounded
    assert store.team_owned(old_id=old, new_id=new)
    assert [e.id for e in store.recall(limit=20).episodes if e.id in (old, new)] == [new]
    # the retired row no longer claims the entry: a third snapshot is "already present"
    again = import_ledger(store, v3([(D_Y, True, [])], seq=3))
    assert again.already_present == [D_ID] and not again.replaced and not again.conflicts


def test_a_save_validated_against_the_old_text_cannot_commit_on_the_new(store):
    """HIGH 2: the import landed between the save's read of X and its lock, and the
    save committed grounding on an id that then read Y."""
    old = held(store)
    own = store.record(f"I checked it myself: {EXPL}.", EpisodeType.OBSERVATION)
    prepare_wrap(store)
    real, calls = store.ground_state, []

    def racing(ids):
        if not calls:
            calls.append(1)
            with Store(store.path) as other:
                import_ledger(other, v3([(D_Y, True, [])], seq=2))
        return real(ids)

    store.ground_state = racing  # type: ignore[method-assign]
    with pytest.raises(SupersessionError, match="superseded by another writer"):
        save(store, _line(old, own.id), "2026-10-08")
    assert store.status().wrap_in_progress
    assert X in store.get(old).content and store.pattern_grounding() == {}
