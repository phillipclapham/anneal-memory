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
    assert r.replaced == [{"id": D_ID, "old": old, "new": new, "revived": False}]
    assert [(l["old"], l["new"]) for l in r.links_added] == [(old, new)]
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


def test_the_link_and_the_ids_are_derived_again_on_every_replace(tmp_path, store):
    """L1 r1 (run): a deleted head re-imported left the old text shown with no link,
    for good. L2 r1 (run): X -> Y -> X minted a nonce twin of X, so this store's
    head id differed from a store that only ever saw X. L3 r1: below."""
    old = held(store)
    import_ledger(store, v3([(D_Y, True, [])], seq=2))
    new = held(store)
    assert store.delete(new)  # a library delete: the stream brings the entry back
    r = import_ledger(store, v3([(D_Y, True, [])], seq=3))
    back = held(store)
    assert r.reimported == [D_ID] and store.supersession_exists(old_id=old, new_id=back)
    r = import_ledger(store, v3([(D_X, True, [])], seq=4))
    assert r.replaced == [{"id": D_ID, "old": back, "new": old, "revived": True}]
    with Store(tmp_path / "fresh.db", project_name="T") as fresh:
        import_ledger(fresh, v3([(D_X, True, [])], seq=1))
        assert held(fresh) == held(store) == old
    assert [e.id for e in store.recall(limit=20).episodes
            if e.id in (old, back)] == [old]
    # L3 r1 complement (run): a released key leaves its link unowned, so the flip
    # is refused as a cycle. L3 r2 codex (two HIGHs): removing it by its team:
    # label also removed another ledger's and an operator's link; it stays, the
    # refusal is reported every replace, and the operator's unsupersede ends it.
    store.team_forget_key("k1")
    for seq in (5, 6):
        r = import_ledger(store, v3([(D_Y, True, [])], seq=seq))
        assert held(store) == back and any(
            l["old"] == old and "cycle" in l["reason"] for l in r.links_refused)
    assert store.unsupersede(old_id=back, new_id=old)
    import_ledger(store, v3([(D_Y, True, [])], seq=7))
    assert [e.id for e in store.recall(limit=20).episodes if e.id in (old, back)] == [back]
    # L3 r1 codex (run): a head deleted, then an earlier copy enforced, minted a
    # twin through the re-import path instead of reviving the kept row.
    assert store.delete(back)
    r = import_ledger(store, v3([(D_X, True, [])], seq=8))
    assert r.reimported == [D_ID] and held(store) == old
