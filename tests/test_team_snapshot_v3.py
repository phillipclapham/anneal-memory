"""v3 team-import: the store keeps levain's latest verdict as a snapshot (spore-1344
seam design, flow projects/levain/reference/spore1344_seam_design_1005.md §3b, §6).
Test ids (S1b, S3, ...) are the design's."""
from __future__ import annotations

import json

import pytest

from anneal_memory.store import Store
from anneal_memory.team import import_ledger
from tests.test_team_import import RULING, _prefix, ledger

A0 = f"{_prefix('alice')}-20261004120000-00000000"   # alice's ruling
B0 = f"{_prefix('bob')}-20261004120000-00000000"     # bob's retire of A0
RETIRE = {"type": "retire", "supersedes": [A0]}


def v3(items, *, key="k1", root="r1", epoch="e1", repin_n=0, pos=1, seq=1,
       judged="full", end=True):
    """items: (ledger line, enforced, honours)."""
    out = [json.dumps({"anneal_team_stream": 3, "key": key, "root": root, "epoch": epoch,
                       "repin_n": repin_n, "pos": pos, "seq": seq, "judged": judged})]
    for n, (line, enf, hon) in enumerate(items, 1):
        out.append(json.dumps({"frame": "f", "n": n, "line": line,
                               "enforced": enf, "honours": hon}))
    if end:
        out.append(json.dumps({"anneal_team_stream_end": len(items)}))
    return out


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "m.db", project_name="p", audit=False)
    yield s
    s.close()


def lines():
    return ledger("alice", [RULING])[0], ledger("bob", [RETIRE])[0]


def hidden(s):
    return {e.content for e in s.recall(limit=50).episodes}


def ep(s, entry):
    return s._conn.execute(
        "SELECT episode_id FROM team_entries WHERE entry_id = ?", (entry,)).fetchone()[0]


def links(s):
    return {(r[0], r[1]) for r in s._conn.execute("SELECT old_id, new_id FROM supersessions")}


def test_s1b_and_s3_snapshot_follows_the_verdict(store):
    a, b = lines()
    # a 0.9.39-style first import pinned bob's link (the 1344 shape)
    import_ledger(store, [a, b], link_authority=["bob"])
    pair = (ep(store, A0), ep(store, B0))
    assert pair in links(store)
    # first v3 import: levain does not honour bob's retire -> adopted and removed
    r = import_ledger(store, v3([(a, True, []), (b, True, [])]))
    assert r.snapshot == "replaced" and r.links_adopted == 1
    assert [l["target"] for l in r.links_removed] == [A0]
    assert pair not in links(store)
    # levain honours it now -> added back; then the linker is withdrawn -> removed
    r = import_ledger(store, v3([(a, True, []), (b, True, [A0])], seq=2))
    assert len(r.links_added) == 1 and pair in links(store)
    r = import_ledger(store, v3([(a, True, []), (b, False, [])], seq=3))
    assert len(r.links_removed) == 1 and pair not in links(store)
    # a withdrawn TARGET keeps its links (nothing to un-hide)
    import_ledger(store, v3([(a, True, []), (b, True, [A0])], seq=4))
    r = import_ledger(store, v3([(a, False, []), (b, True, [])], seq=5))
    assert not r.links_removed and pair in links(store)


def test_s4_s12_operator_override(store, monkeypatch):
    a, b = lines()
    import_ledger(store, v3([(a, True, []), (b, True, [A0])]))
    old, new = ep(store, A0), ep(store, B0)
    assert store.team_owned(old_id=old, new_id=new)
    # library calls are a no-op on an owned row
    assert store.unsupersede(old_id=old, new_id=new) is False
    assert (old, new) in links(store)
    assert store.supersede(old_id=old, new_id=new) is False
    # the confirmed override sticks through every later replace
    assert store.unsupersede(old_id=old, new_id=new, team_override=True) is True
    r = import_ledger(store, v3([(a, True, []), (b, True, [A0])], seq=2))
    assert (old, new) not in links(store) and not r.links_added
    # an OLD binary's removal (raw delete) is caught as an operator act
    s2 = (old, new)
    store._conn.execute("DELETE FROM team_overrides"); store._conn.commit()
    import_ledger(store, v3([(a, True, []), (b, True, [A0])], seq=3))
    assert s2 in links(store)
    store._conn.execute("DELETE FROM supersessions WHERE old_id=? AND new_id=?", s2)
    store._conn.commit()
    r = import_ledger(store, v3([(a, True, []), (b, True, [A0])], seq=4))
    assert r.overrides_recorded and s2 not in links(store)
    # an operator supersede over an owned row takes it out of every snapshot
    store._conn.execute("DELETE FROM team_overrides"); store._conn.commit()
    import_ledger(store, v3([(a, True, []), (b, True, [A0])], seq=5))
    assert store.supersede(old_id=old, new_id=new, source="me", team_override=True)
    assert not store.team_owned(old_id=old, new_id=new)
    import_ledger(store, v3([(a, True, []), (b, False, [])], seq=6))
    assert s2 in links(store)  # the operator's now: the withdrawn linker leaves it


def test_s5_s11_s13_s15_s17_ordering_and_keys(store):
    a, b = lines()
    on = v3([(a, True, []), (b, True, [A0])], key="k1", pos=5, seq=10)
    off = lambda **kw: v3([(a, True, []), (b, True, [])], **kw)  # noqa: E731
    import_ledger(store, on)
    pair = (ep(store, A0), ep(store, B0))
    # a lagging clone (lower pos) is stale and changes nothing (S13)
    assert import_ledger(store, off(key="k2", pos=4, seq=99)).snapshot == "stale_stream"
    assert pair in links(store)
    # a newer key of the same root takes over and the store follows it (S5/S11)
    r = import_ledger(store, off(key="k2", pos=6, seq=1))
    assert r.snapshot == "replaced" and pair not in links(store)
    st = {k["key"]: k for k in store.team_snapshot_status()["keys"]}
    assert st["k2"]["active"] and not st["k1"]["active"]
    # a second ledger (another root) is independent
    import_ledger(store, v3([(a, True, []), (b, True, [A0])], key="x", root="r2"))
    # a new epoch lands even with a lower pos (S15)
    assert import_ledger(store, off(key="k2", epoch="e2", pos=1, seq=1)).snapshot == "replaced"
    # same key, same epoch, higher repin_n wins over a stored (pos, seq) (S17)
    r = import_ledger(store, v3([(a, True, []), (b, True, [A0])], key="k2", epoch="e2",
                                repin_n=1, pos=1, seq=0))
    assert r.snapshot == "replaced"


def test_s6_incomplete_partial_streams_import_episodes_only(store):
    a, b = lines()
    for stream, why in [
        (v3([(a, True, []), (b, True, [A0])], end=False), "incomplete_stream"),
        (v3([(a, True, []), (b, True, [A0])], judged="partial"), "partial_stream"),
        (v3([(a, True, []), (b, True, ["nope"])]), "incomplete_stream"),
        (v3([(a, True, []), (b, False, [A0])]), "incomplete_stream"),
        (v3([(a, True, []), (b.replace('"bob"', '"bo"'), True, [A0])]), "incomplete_stream"),
    ]:
        s = Store(store.path.parent / f"{why}{len(stream)}{id(stream)}.db", audit=False)
        try:
            r = import_ledger(s, stream)
            assert r.snapshot == why, (why, r.chain_problems)
            assert not links(s)
            assert r.imported  # episodes still import
        finally:
            s.close()


def test_s7_s8_legacy_and_twins(store):
    a, b = lines()
    # a pre-v3 store where bob's link was refused (no authority) -> legacy adds it once
    r0 = import_ledger(store, [a, b])
    assert r0.links_unauthorized
    r = import_ledger(store, v3([(a, True, []), (b, True, [A0])]))
    assert len(r.links_added_legacy) == 1 and not r.links_added
    # a stored copy whose hash differs maps to nothing: its pair is removed
    store._conn.execute("UPDATE team_entries SET hash='x' WHERE entry_id=?", (B0,))
    store._conn.execute(
        "UPDATE episodes SET metadata=json_set(metadata,'$.team.hash','x') WHERE id=?",
        (ep(store, B0),))
    store._conn.commit()
    r = import_ledger(store, v3([(a, True, []), (b, True, [A0])], seq=2))
    assert len(r.links_removed) == 1


def test_s10_s14_s16_s18_prune_and_rewire(tmp_path):
    a = ledger("alice", [{**RULING, "ts": "2026-01-01T00:00:00Z"}])[0]
    b = ledger("bob", [{"type": "retire", "supersedes": [A0], "ts": "2026-01-02T00:00:00Z"}])[0]
    c0 = f"{_prefix('cy')}-20261004120000-00000000"
    c = ledger("cy", [{"type": "retire", "supersedes": [B0]}])[0]
    c_both = ledger("cy", [{"type": "retire", "supersedes": [B0, A0]}])[0]
    # retention keeps B, which links an owned row; the old ruling A itself may go
    s = Store(tmp_path / "p.db", project_name="p", audit=False, retention_days=30)
    try:
        import_ledger(s, v3([(a, True, []), (b, True, [A0]), (c, True, [B0])]))
        eb, ec = ep(s, B0), ep(s, c0)
        s.prune()
        assert s.get(eb) is not None and (eb, ec) in links(s)
        assert not s._conn.execute("SELECT 1 FROM team_overrides").fetchone()
    finally:
        s.close()
    # an explicit delete of B: A -> C rewired, owned, standing in for (A, B)
    s = Store(tmp_path / "q.db", project_name="p", audit=False)
    try:
        import_ledger(s, v3([(a, True, []), (b, True, [A0]), (c, True, [B0])]))
        ea, eb, ec = ep(s, A0), ep(s, B0), ep(s, c0)
        assert s.delete(eb)
        assert (ea, ec) in links(s) and s.team_owned(old_id=ea, new_id=ec)
        assert not s._conn.execute("SELECT 1 FROM team_overrides").fetchone()
        # kept while levain honours (A, B) (S16)
        r = import_ledger(s, v3([(a, True, []), (b, True, [A0]), (c, True, [B0])], seq=2))
        assert (ea, ec) in links(s) and not r.links_removed
        # levain stops honouring (A, B): removed
        r = import_ledger(s, v3([(a, True, []), (b, True, []), (c, True, [B0])], seq=3))
        assert (ea, ec) not in links(s) and len(r.links_removed) == 1
    finally:
        s.close()
    # S18: a rewired row whose pair is also honoured directly stays while either holds
    s = Store(tmp_path / "r.db", project_name="p", audit=False)
    try:
        import_ledger(s, v3([(a, True, []), (b, True, [A0]), (c_both, True, [B0, A0])]))
        ea, eb, ec = ep(s, A0), ep(s, B0), ep(s, c0)
        assert s.delete(eb)
        assert (ea, ec) in links(s)
        r = import_ledger(s, v3([(a, True, []), (b, True, []), (c_both, True, [B0, A0])], seq=2))
        assert (ea, ec) in links(s) and not r.links_removed
        r = import_ledger(s, v3([(a, True, []), (b, True, []), (c_both, True, [B0])], seq=3))
        assert (ea, ec) not in links(s)
    finally:
        s.close()


def test_cli_refuses_authority_flags_with_v3(tmp_path, capsys, monkeypatch):
    import io
    import sys

    from anneal_memory.cli import main
    a, b = lines()
    db = tmp_path / "c.db"
    Store(db, audit=False).close()
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(
        "\n".join(v3([(a, True, []), (b, True, [A0])])).encode())))
    monkeypatch.setattr(sys, "argv", ["anneal-memory", "--db", str(db), "team-import",
                                      "--link-authority", "ana", "-"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
