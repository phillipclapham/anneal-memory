"""0.9.40: a pruned team entry stays pruned, and the owner of the call may link.

Both tests come from a reproduced run, not a reasoned path (Diogenes 2026-10-05
MEDIUM; levain ledger supersede plan, question F).
"""
from __future__ import annotations

import json

from anneal_memory.store import Store
from anneal_memory.team import import_ledger

from tests.test_team_import import RULING, _prefix, ledger, seal


def test_import_prune_import_is_stable(tmp_path):
    s = Store(tmp_path / "m.db", project_name="p", retention_days=30, audit=False)
    try:
        old = seal({"v": 1, "id": f"{_prefix('alice')}-20260801120000-00000000",
                    "ts": "2026-08-01T12:00:00Z", "author": "alice", "type": "decision",
                    "kind": "practice", "words": "use tabs", "paths": [], "supersedes": []}, "")
        lines = [json.dumps(old)]
        first = import_ledger(s, lines)
        assert len(first.imported) == 1
        assert s.prune() == 1
        for _ in range(2):
            again = import_ledger(s, lines)
            assert not again.imported and again.clean
            assert again.already_removed == [old["id"]]
            assert again.to_dict()["already_removed"] == 1
            assert again.links_to_removed == []
            assert s.prune() == 0
        assert s._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 0
        # the same id with another hash is a conflict, not a fresh entry
        forged = seal({**{k: v for k, v in old.items() if k not in ("hash", "prev")},
                       "words": "use spaces"}, "")
        clash = import_ledger(s, [json.dumps(forged)])
        assert not clash.imported and clash.conflicts[0]["stored_hash"] == old["hash"]
        # a delete holds too, and a link onto the removed entry is reported
        rul = ledger("bo", [RULING, {"type": "retire", "supersedes": [f"{_prefix('bo')}-20261004120000-00000000"]}])
        r = import_ledger(s, rul[:1])
        assert s.delete(r.imported[0]["episode"])
        r = import_ledger(s, rul)
        assert len(r.imported) == 1 and r.already_removed and r.clean
        assert r.links_to_removed == [{"id": f"{_prefix('bo')}-20261004120000-00000001",
                                       "target": f"{_prefix('bo')}-20261004120000-00000000"}]
    finally:
        s.close()
    # keep_tombstones=False erases the record with the episode: the entry comes back
    e = Store(tmp_path / "e.db", project_name="p", retention_days=30, audit=False,
              keep_tombstones=False)
    try:
        import_ledger(e, lines)
        assert e.prune() == 1
        back = import_ledger(e, lines).imported
        assert len(back) == 1
        assert e.delete(back[0]["episode"])
        assert len(import_ledger(e, lines).imported) == 1
    finally:
        e.close()


def test_owner_of_the_call_may_link(tmp_path):
    target = "alice-20261004120000-00000000"
    ruling = {**RULING, "owner": "bob"}

    def run(name, linker, owner_field, **kw):
        s = Store(tmp_path / f"{name}.db", project_name="p", audit=False)
        try:
            lines = ledger("alice", [{**ruling, "owner": owner_field}]) + ledger(
                linker, [{"type": "retire", "supersedes": [target]}])
            return import_ledger(s, lines, **kw)
        finally:
            s.close()

    # bob owns alice's call and is a current member: honoured, and says why
    rep = run("yes", "bob", "bob", call_owners=["bob", "carol"])
    assert rep.clean and rep.links_made[0]["authority"] == "call_owner"
    assert rep.to_dict()["cross_author_links"][0]["authority"] == "call_owner"
    # without the membership list, 0.9.39 behaviour: reported, names who owns the call
    rep = run("nolist", "bob", "bob")
    assert not rep.links_made and rep.links_unauthorized[0]["target_owner"] == "bob"
    # a member who does not own the call
    assert not run("carol", "carol", "bob", call_owners=["bob", "carol"]).links_made
    # owns the call but is no longer a member
    assert not run("left", "bob", "bob", call_owners=["carol"]).links_made
    # 'lead' resolves to the team owner, never to a member handle named lead
    assert not run("lead", "lead", "lead", call_owners=["lead"]).links_made
    assert run("leadok", "lead", "lead", link_authority=["lead"]).links_made[0]["authority"] == "link_authority"
