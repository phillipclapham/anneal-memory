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


def v3(items, *, key="k1", root="r1", prev_root=None, epoch="e1", repin_n=0, pos=1,
       seq=1, judged="full", end=True):
    """items: (ledger line, enforced, honours)."""
    out = [json.dumps({"anneal_team_stream": 3, "key": key, "root": root,
                       "prev_root": root if prev_root is None else prev_root,
                       "epoch": epoch, "repin_n": repin_n, "pos": pos, "seq": seq,
                       "judged": judged})]
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
    # a key new to the store takes over, even at a lower pos: it is the clone that
    # exists now (S5/S11, design §3b.3 "a key's FIRST replace takes over")
    r = import_ledger(store, off(key="k2", pos=4, seq=1))
    assert r.snapshot == "replaced" and pair not in links(store)
    st = {k["key"]: k for k in store.team_snapshot_status()["keys"]}
    assert st["k2"]["active"] and not st["k1"]["active"]
    # S13: the lagging KNOWN clone k1 exports a view older than the active k2's
    # (pos, seq): stale, and it changes nothing
    assert import_ledger(store, v3([(a, True, []), (b, True, [A0])], key="k1", pos=3,
                                   seq=99)).snapshot == "stale_stream"
    assert pair not in links(store)
    # k2 moves on, newer than its own stored view
    r = import_ledger(store, off(key="k2", pos=6, seq=1))
    assert r.snapshot == "replaced"
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
        # a per-line hash mismatch is UNMAPPABLE, never stream-breaking (design §3a 3b)
        (v3([(a, True, []), (b.replace('"bob"', '"bo"'), True, [A0])]), "replaced"),
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
    # S31: no enforced line carries the stored hash: replaced IN PLACE (same episode),
    # so the real line's pair maps again and stays
    before = ep(store, B0)
    r = import_ledger(store, v3([(a, True, []), (b, True, [A0])], seq=2))
    assert r.replaced_in_place == [B0] and not r.links_removed
    assert ep(store, B0) == before and (ep(store, A0), before) in links(store)


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
        assert s.delete(eb, team_operator=True)   # final: B stays out
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
        assert s.delete(eb, team_operator=True)
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


def test_degenerate_profile_without_tenure(store):
    """Phill 2026-10-05 (seam first): levain exports with a TOFU root, epoch =
    digest(root), repin_n = 0 and a first-parent pos. Successive syncs replace in order."""
    import hashlib
    a, b = lines()
    prof = dict(key="clone-a", root="genesis-sha", repin_n=0,
                epoch=hashlib.sha256(b"genesis-sha").hexdigest())
    r = import_ledger(store, v3([(a, True, []), (b, True, [A0])], pos=2, seq=10, **prof))
    assert r.snapshot == "replaced" and len(r.links_added_legacy) == 1
    r = import_ledger(store, v3([(a, True, []), (b, True, [])], pos=3, seq=11, **prof))
    assert r.snapshot == "replaced" and len(r.links_removed) == 1
    # the same view again, and an older one, change nothing
    assert import_ledger(store, v3([(a, True, []), (b, True, [])], pos=3, seq=11,
                                   **prof)).snapshot == "stale_stream"
    assert import_ledger(store, v3([(a, True, []), (b, True, [A0])], pos=2, seq=12,
                                   **prof)).snapshot == "stale_stream"
    assert not links(store)


def test_review_gaps_twins_roots_trailer_rewired(tmp_path):
    a, b = lines()
    b_twin = ledger("bob", [{**RETIRE, "reason": "twin"}])[0]   # same id B0, other bytes
    # S8: two enforced lines with one id (a live twin) replace, but map no linker:
    # the pair is never pinned, and the twin is a reported conflict
    s = Store(tmp_path / "t.db", audit=False)
    try:
        r = import_ledger(s, v3([(a, True, []), (b, True, [A0]), (b_twin, True, [A0])]))
        assert r.snapshot == "replaced" and not links(s) and r.conflicts
    finally:
        s.close()
    # S6: a malformed trailer, a count mismatch, content after the end line
    for stream in (
        v3([(a, True, []), (b, True, [A0])], end=False) + ['{"anneal_team_stream_end": "2"}'],
        v3([(a, True, []), (b, True, [A0])], end=False) + ['{"anneal_team_stream_end": 3}'],
        v3([(a, True, []), (b, True, [A0])]) + [json.dumps({"x": 1})],
    ):
        s = Store(tmp_path / f"m{id(stream)}.db", audit=False)
        try:
            assert import_ledger(s, stream).snapshot == "incomplete_stream" and not links(s)
        finally:
            s.close()
    s = Store(tmp_path / "x.db", audit=False)
    try:
        # S5/S33: a second ledger (other root) that shares only a copied ruling (a
        # target-only overlap) is never taken over by the first one's replaces
        d0 = f"{_prefix('dan')}-20261004120000-00000000"
        d = ledger("dan", [RETIRE])[0]
        import_ledger(s, v3([(a, True, []), (b, True, [A0])], key="k1", root="r1"))
        pair = (ep(s, A0), ep(s, B0))
        import_ledger(s, v3([(a, True, []), (d, True, [A0])], key="x", root="r2"))
        dpair = (ep(s, A0), ep(s, d0))
        assert {pair, dpair} <= links(s)
        r = import_ledger(s, v3([(a, True, []), (b, True, [])], key="k1", root="r1", seq=2))
        assert pair not in links(s) and dpair in links(s)
        st = {k["key"]: k for k in s.team_snapshot_status()["keys"]}
        assert st["x"]["active"] and st["k1"]["active"]
        # a verbatim copy of the LINKER line is a linker match: a takeover (design §5)
        import_ledger(s, v3([(a, True, []), (d, True, [A0])], key="k1", root="r1", seq=3))
        st = {k["key"]: k for k in s.team_snapshot_status()["keys"]}
        assert not st["x"]["active"]
    finally:
        s.close()
    # S16: an operator's rewired row stays the operator's; an old binary's is never removed
    c0 = f"{_prefix('cy')}-20261004120000-00000000"
    c = ledger("cy", [{"type": "retire", "supersedes": [B0]}])[0]
    s = Store(tmp_path / "o.db", audit=False)
    try:
        import_ledger(s, v3([(a, True, []), (b, True, []), (c, True, [])]))
        ea, eb, ec = ep(s, A0), ep(s, B0), ep(s, c0)
        s.supersede(old_id=ea, new_id=eb, source="me")          # operator links
        s.supersede(old_id=eb, new_id=ec, source="me")
        s.delete(eb, team_operator=True)                        # -> rewired A -> C, operator's
        assert (ea, ec) in links(s) and not s.team_owned(old_id=ea, new_id=ec)
        import_ledger(s, v3([(a, True, []), (b, True, []), (c, True, [])], seq=2))
        assert (ea, ec) in links(s)
        # an old binary's rewired row: no rewire_origin, never adopted or removed
        s._conn.execute("DELETE FROM rewire_origin")
        s._conn.execute("UPDATE supersessions SET source='rewired'")
        s._conn.commit()
        s.team_forget_key("k1")                                 # next import runs legacy
        import_ledger(s, v3([(a, True, []), (b, True, []), (c, True, [])], seq=3))
        assert (ea, ec) in links(s)
        assert s.team_snapshot_status()["unmanaged_rewired"] == 1
    finally:
        s.close()
    # S7: legacy adopts a team:-labelled row whose linker was never ENFORCED in a v3
    # stream (eve's stranger line: levain exports it unenforced), and removes it
    e0 = f"{_prefix('eve')}-20261004120000-00000000"
    e = ledger("eve", [RETIRE])[0]
    s = Store(tmp_path / "e.db", audit=False)
    try:
        import_ledger(s, [a, e], link_authority=["eve"])
        assert (ep(s, A0), ep(s, e0)) in links(s)
        r = import_ledger(s, v3([(a, True, []), (e, False, [])]))
        assert r.links_adopted == 1 and not links(s)
    finally:
        s.close()
    # diogenes-20261006-020941 (design §3b.5, legacy r6 complement L5-4): a row whose
    # linker is NOT of this stream (another ledger's link onto a copied ruling) is
    # never adopted on the target alone, so this ledger's first replace keeps it
    s = Store(tmp_path / "f.db", audit=False)
    try:
        import_ledger(s, [a, b], link_authority=["bob"])
        pair = (ep(s, A0), ep(s, B0))
        r = import_ledger(s, v3([(a, True, [])]))
        assert r.snapshot == "replaced" and r.links_adopted == 0
        assert pair in links(s) and not r.links_removed
    finally:
        s.close()


def test_l1_same_key_older_repin_is_stale_and_override_of_missing_owned_row(store):
    a, b = lines()
    import_ledger(store, v3([(a, True, []), (b, True, [A0])], epoch="e1", repin_n=0, pos=5))
    pair = (ep(store, A0), ep(store, B0))
    import_ledger(store, v3([(a, True, []), (b, True, [])], epoch="e2", repin_n=1, pos=1))
    assert pair not in links(store)
    # a LOWER repin_n than stored: the clone's state was restored or copied, so it is
    # a first replace and lands (design §3b.3, legacy r5 complement F5; this reverses
    # the 1005+4 L1 stale guard)
    r = import_ledger(store, v3([(a, True, []), (b, True, [A0])], epoch="e1", repin_n=0, pos=9))
    assert r.snapshot == "replaced" and pair in links(store)
    # an override supersede over an owned row that went missing makes it the operator's
    import_ledger(store, v3([(a, True, []), (b, True, [A0])], epoch="e2", repin_n=1, pos=2))
    store._conn.execute("DELETE FROM supersessions"); store._conn.commit()
    assert store.supersede(old_id=pair[0], new_id=pair[1], source="me", team_override=True)
    assert not store.team_owned(old_id=pair[0], new_id=pair[1])
    import_ledger(store, v3([(a, True, []), (b, True, [])], epoch="e2", repin_n=1, pos=3))
    assert pair in links(store)


def test_l3r1_replay_after_epoch_change_and_takeover_override(store):
    a, b = lines()
    on = v3([(a, True, []), (b, True, [A0])], key="k1", epoch="e1", pos=5, seq=10)
    import_ledger(store, on)
    pair = (ep(store, A0), ep(store, B0))
    import_ledger(store, v3([(a, True, []), (b, True, [])], key="k2", epoch="e2", pos=1, seq=1))
    assert pair not in links(store)
    # k1 replays the exact view it already sent: stale, the revoked link stays revoked
    assert import_ledger(store, on).snapshot == "stale_stream"
    assert pair not in links(store)
    # takeover by supersede, then an ordinary unsupersede: no snapshot re-adds it
    import_ledger(store, v3([(a, True, []), (b, True, [A0])], key="k2", epoch="e2", pos=2, seq=2))
    assert store.supersede(old_id=pair[0], new_id=pair[1], source="me", team_override=True)
    assert store.unsupersede(old_id=pair[0], new_id=pair[1]) is True
    import_ledger(store, v3([(a, True, []), (b, True, [A0])], key="k2", epoch="e2", pos=3, seq=3))
    assert pair not in links(store)


# -- 1006+4: the landed legacy-profile § (design §3a 3b, §3b.1-§3b.7) ----------------

def test_s21_per_line_results_never_break_the_stream(store):
    a = ledger("alice", [RULING])[0]
    # unsafe text in an enforced line: imported sanitised and flagged, mapped by the
    # line's own hash, so its supersede still hides its target
    b_zwj = ledger("bob", [{**RETIRE, "reason": "per‍sonal"}])[0]
    r = import_ledger(store, v3([(a, True, []), (b_zwj, True, [A0])]))
    assert r.snapshot == "replaced" and r.sanitised == [B0]
    pair = (ep(store, A0), ep(store, B0))
    assert pair in links(store)
    meta = json.loads(store._conn.execute(
        "SELECT metadata FROM episodes WHERE id = ?", (pair[1],)).fetchone()[0])
    assert meta["team"]["sanitised"] is True
    assert meta["team"]["hash"] == json.loads(b_zwj)["hash"]
    assert "\\u200d" in meta["team"]["reason"]
    # a "v": true line is UNMAPPABLE: reported and noted, the rest replaces, and the
    # ruling it would supersede stays visible
    c0 = f"{_prefix('cy')}-20261004120000-00000000"
    c_bad = ledger("cy", [{**RETIRE, "v": True}])[0]
    r = import_ledger(store, v3([(a, True, []), (b_zwj, True, []), (c_bad, True, [A0])],
                                seq=2))
    assert r.snapshot == "replaced" and [u["id"] for u in r.unmappable] == [c0]
    assert not r.clean and not links(store)
    notes = store.team_snapshot_status()["notes"]
    assert [(n["kind"], n["entry_id"]) for n in notes] == [("unmappable", c0)]
    # an UNENFORCED bad line is ignored entirely
    r = import_ledger(store, v3([(a, True, []), (b_zwj, True, []), (c_bad, False, [])],
                                seq=3))
    assert r.snapshot == "replaced" and not r.unmappable
    assert store.team_snapshot_status()["notes"] == []


def test_s35_stale_twin_of_an_unmappable_line_is_flagged(store):
    a, b = lines()
    import_ledger(store, v3([(a, True, []), (b, True, [A0])]))
    pair = (ep(store, A0), ep(store, B0))
    # levain's current B0 is a line anneal refuses: the stored copy is a stale twin
    b_bad = ledger("bob", [{**RETIRE, "v": True}])[0]
    r = import_ledger(store, v3([(a, True, []), (b_bad, True, [A0])], seq=2))
    assert r.unmappable and r.unmappable[0]["stale_twin"] == pair[1]
    assert pair in links(store)  # design §3b.1: rows an unmappable linker made stay
    text = store.get(pair[1]).content
    assert text.startswith("[stale:")
    import_ledger(store, v3([(a, True, []), (b_bad, True, [A0])], seq=3))
    assert store.get(pair[1]).content.count("[stale:") == 1  # flagged once
    # the real line (the stored copy's own hash) is enforced again: the flag goes
    # and the pair maps again
    r = import_ledger(store, v3([(a, True, []), (b, True, [A0])], seq=4))
    assert r.already_present == [A0, B0] and pair in links(store)
    assert not store.get(pair[1]).content.startswith("[stale:")


def test_s22_root_move_needs_prev_root_and_a_collision_changes_nothing(store):
    a, b = lines()
    import_ledger(store, v3([(a, True, []), (b, True, [A0])], key="k1", root="r1"))
    pair = (ep(store, A0), ep(store, B0))
    # the same key under another root, prev_root not the stored one: a collision
    r = import_ledger(store, v3([(a, True, []), (b, True, [])], key="k1", root="r9",
                                prev_root="rX", seq=2))
    assert r.snapshot == "partial_stream" and r.chain_problems and pair in links(store)
    assert {k["root"] for k in store.team_snapshot_status()["keys"]} == {"r1"}
    # a proven move (prev_root = the stored root): moved, replaced, nothing co-owned
    r = import_ledger(store, v3([(a, True, []), (b, True, [])], key="k1", root="r2",
                                prev_root="r1", epoch="e2", seq=2))
    assert r.snapshot == "replaced" and pair not in links(store)
    assert not r.links_added_legacy and r.links_adopted == 0  # a move is not legacy
    keys = store.team_snapshot_status()["keys"]
    assert [(k["key"], k["root"]) for k in keys] == [("k1", "r2")]


def test_s30_s32_retention_keeps_enforced_and_non_final_removals_heal(tmp_path):
    a = ledger("alice", [{**RULING, "ts": "2026-01-01T00:00:00Z"}])[0]
    b = ledger("bob", [{"type": "retire", "supersedes": [A0], "ts": "2026-01-02T00:00:00Z"}])[0]
    s = Store(tmp_path / "ret.db", project_name="p", audit=False, retention_days=30)
    try:
        import_ledger(s, v3([(a, True, []), (b, True, [A0])]))
        pair = (ep(s, A0), ep(s, B0))
        # S30: withdraw the honour, retention runs, re-honour: nothing was pruned
        import_ledger(s, v3([(a, True, []), (b, True, [])], seq=2))
        assert s.prune() == 0
        import_ledger(s, v3([(a, True, []), (b, True, [A0])], seq=3))
        assert pair in links(s)
        # S32: bob's line goes unenforced (1g in a forged window), retention prunes
        # it, the owner reverts: it is re-imported at the next replace
        import_ledger(s, v3([(a, True, []), (b, False, [])], seq=4))
        assert s.prune() == 1 and s.get(pair[1]) is None
        r = import_ledger(s, v3([(a, True, []), (b, True, [A0])], seq=5))
        assert r.reimported == [B0]
        assert (ep(s, A0), ep(s, B0)) in links(s)
        # the 0.9.40 loop does not return: import, prune, import is stable
        assert s.prune() == 0
        r = import_ledger(s, v3([(a, True, []), (b, True, [A0])], seq=6))
        assert not r.imported and not r.reimported
        assert s.team_snapshot_status()["protected_episodes"] == 2
    finally:
        s.close()


def test_s34_only_an_operator_delete_is_final(store, tmp_path, monkeypatch, capsys):
    a, b = lines()
    import_ledger(store, v3([(a, True, []), (b, True, [A0])]))
    # a library / MCP delete of an enforced linker: re-imported, target hidden again
    assert store.delete(ep(store, B0))
    r = import_ledger(store, v3([(a, True, []), (b, True, [A0])], seq=2))
    assert r.reimported == [B0] and (ep(store, A0), ep(store, B0)) in links(store)
    # a stale or partial stream does not bring it back: only a replace does
    assert store.delete(ep(store, B0))
    r = import_ledger(store, v3([(a, True, []), (b, True, [A0])], seq=2))
    assert r.snapshot == "stale_stream" and not r.reimported and not r.imported
    # the operator's delete is final; the v2 path never re-imports either way
    r = import_ledger(store, v3([(a, True, []), (b, True, [A0])], seq=3))
    assert store.delete(ep(store, B0), team_operator=True)
    r = import_ledger(store, v3([(a, True, []), (b, True, [A0])], seq=4))
    assert r.already_removed == [B0] and not r.reimported
    # the CLI: --force without the override is non-final, and says so
    import sys

    from anneal_memory.cli import main
    db = tmp_path / "cli.db"
    s = Store(db, audit=False)
    import_ledger(s, v3([(a, True, []), (b, True, [A0])]))
    eb = ep(s, B0)
    s.close()
    monkeypatch.delenv("ANNEAL_TEAM_OVERRIDE", raising=False)
    monkeypatch.setattr(sys, "argv", ["anneal-memory", "--db", str(db), "delete", eb,
                                      "--force"])
    main()
    assert "comes back" in capsys.readouterr().err
    s = Store(db, audit=False)
    try:
        assert s._conn.execute("SELECT removal FROM team_entries WHERE entry_id = ?",
                               (B0,)).fetchone()[0] == "auto"
        r = import_ledger(s, v3([(a, True, []), (b, True, [A0])], seq=2))
        assert r.reimported == [B0]
        eb = ep(s, B0)
    finally:
        s.close()
    monkeypatch.setenv("ANNEAL_TEAM_OVERRIDE", "1")
    monkeypatch.setattr(sys, "argv", ["anneal-memory", "--db", str(db), "delete", eb,
                                      "--force"])
    main()
    s = Store(db, audit=False)
    try:
        assert s._conn.execute("SELECT removal FROM team_entries WHERE entry_id = ?",
                               (B0,)).fetchone()[0] == "operator"
    finally:
        s.close()


def test_v3_add_checks_are_existence_and_cycle_only(store):
    # bob's retire is OLDER than alice's ruling (a teammate's clock skew): levain
    # honours it in history order, so anneal adds it (no older-than check)
    a = ledger("alice", [{**RULING, "ts": "2026-10-04T12:00:30Z"}])[0]
    b = ledger("bob", [{**RETIRE, "ts": "2026-10-04T12:00:01Z"}])[0]
    r = import_ledger(store, v3([(a, True, []), (b, True, [A0])]))
    assert len(r.links_added_legacy) == 1 and not r.links_refused
    # a ruling superseded by a retire-less, words-less finding: levain's may_link
    # already ruled, so anneal does not re-apply the ruling-words rule
    f0 = f"{_prefix('fay')}-20261004120000-00000000"
    f = ledger("fay", [{"type": "finding", "summary": "replaced by the new job",
                         "supersedes": [A0]}])[0]
    r = import_ledger(store, v3([(a, True, []), (b, True, []), (f, True, [A0])], seq=2))
    assert len(r.links_added) == 1 and (ep(store, A0), ep(store, f0)) in links(store)
    # a cycle is refused, reported, and noted for team-status
    store._conn.execute("DELETE FROM supersessions WHERE old_id = ? AND new_id = ?",
                        (ep(store, A0), ep(store, f0)))
    store._conn.execute("DELETE FROM team_snapshot_rows")
    store._conn.execute("INSERT INTO supersessions (old_id, new_id, source) "
                        "VALUES (?, ?, 'me')", (ep(store, f0), ep(store, A0)))
    store._conn.commit()
    r = import_ledger(store, v3([(a, True, []), (b, True, []), (f, True, [A0])], seq=3))
    assert r.links_refused and "cycle" in r.links_refused[0]["reason"]
    assert [n["kind"] for n in store.team_snapshot_status()["notes"]] == ["refused"]


def test_operator_supersede_relabels_to_operator_and_rotation_is_not_legacy(store):
    a, b = lines()
    import_ledger(store, v3([(a, True, []), (b, True, [A0])], key="k1"))
    pair = (ep(store, A0), ep(store, B0))
    assert store.supersede(old_id=pair[0], new_id=pair[1], source="me", team_override=True)
    assert store._conn.execute("SELECT source FROM supersessions WHERE old_id = ? AND "
                               "new_id = ?", pair).fetchone()[0] == "operator"
    # a rotated key on the same root is a takeover, never a legacy step: the
    # operator's row is not adopted again and nothing is re-added as legacy
    r = import_ledger(store, v3([(a, True, []), (b, True, [])], key="k2"))
    assert r.snapshot == "replaced" and r.links_adopted == 0
    assert not r.links_added_legacy and pair in links(store)


def test_l1_l2_round_fixes(store):
    a, b = lines()
    import_ledger(store, v3([(a, True, []), (b, True, [A0])]))
    pair = (ep(store, A0), ep(store, B0))
    # L1-2: the same line enforced twice is one line, not a live twin
    r = import_ledger(store, v3([(a, True, []), (b, True, [A0]), (b, True, [A0])], seq=2))
    assert r.snapshot == "replaced" and not r.links_removed and pair in links(store)
    # L2-1: a stale stream behind an active key names it, as a note (not a problem)
    r = import_ledger(store, v3([(a, True, []), (b, True, [])], key="k2", pos=9, seq=9))
    assert r.snapshot == "replaced"
    r = import_ledger(store, v3([(a, True, []), (b, True, [A0])], key="k1", pos=2, seq=1))
    assert r.snapshot == "stale_stream" and "team-forget-key k2" in r.snapshot_notes[0]
    assert not r.chain_problems


def test_l1_handle_alone_never_adopts(store):
    # ledger 1: bob's B0 links over alice's A0 (a v2 import, operator-authorised)
    a, b = lines()
    import_ledger(store, [a, b], link_authority=["bob"])
    pair = (ep(store, A0), ep(store, B0))
    # ledger 2's first stream: a copy of A0 and ANOTHER bob line, B0 not in it
    other_bob = ledger("bob", [{"type": "finding", "summary": "first"},
                               {"type": "finding", "summary": "unrelated note"}])[1]
    assert json.loads(other_bob)["id"] != B0
    r = import_ledger(store, v3([(a, True, []), (other_bob, True, [])], key="z", root="r9"))
    assert r.links_adopted == 0 and pair in links(store)


def test_l3r1_1006_fixes(store, tmp_path):
    a, b = lines()
    # codex #3: an enforced linker whose line fails its hash check keeps its id, and
    # the row it linked stays (design §3b.1)
    import_ledger(store, v3([(a, True, []), (b, True, [A0])]))
    pair = (ep(store, A0), ep(store, B0))
    bad = json.loads(b); bad["reason"] = "edited"; b_edit = json.dumps(bad)  # stale hash
    r = import_ledger(store, v3([(a, True, []), (b_edit, True, [A0])], seq=2))
    assert r.snapshot == "replaced" and [u["id"] for u in r.unmappable] == [B0]
    assert pair in links(store) and not r.links_removed
    # codex #7: a library supersede over an owned row that went missing writes nothing
    store._conn.execute("DELETE FROM supersessions"); store._conn.commit()
    assert store.supersede(old_id=pair[0], new_id=pair[1], source="agent") is False
    assert pair not in links(store) and store.team_owned(old_id=pair[0], new_id=pair[1])
    assert not store._conn.execute("SELECT 1 FROM team_overrides").fetchone()
    # codex #9: header values SQLite cannot bind are refused, never a crash
    for bad_head in ({"seq": 2 ** 64}, {"key": "\ud800"}):
        stream = v3([(a, True, [])], **{k: v for k, v in bad_head.items()})
        r = import_ledger(store, stream)
        assert r.framing == "unknown" and r.chain_problems


def test_l3r1_1006_twin_adoption_and_cross_root_rewrite(tmp_path):
    a, b = lines()
    b_twin = ledger("bob", [{**RETIRE, "reason": "twin"}])[0]   # same id B0, other bytes
    # codex #1 (consensus): an unenforced TWIN of the legacy linker is not provenance
    s = Store(tmp_path / "tw.db", audit=False)
    try:
        import_ledger(s, [a, b], link_authority=["bob"])
        pair = (ep(s, A0), ep(s, B0))
        r = import_ledger(s, v3([(a, True, []), (b_twin, False, [])], key="z", root="r9"))
        assert r.links_adopted == 0 and pair in links(s)
    finally:
        s.close()
    # gemini HIGH: a stream of another root never rewrites a copy that an active key
    # of another ledger enforces
    s = Store(tmp_path / "xr.db", audit=False)
    try:
        import_ledger(s, v3([(a, True, []), (b, True, [A0])], key="k1", root="r1"))
        before = s.get(ep(s, B0)).content
        r = import_ledger(s, v3([(a, True, []), (b_twin, True, [A0])], key="k2", root="r2"))
        assert not r.replaced_in_place and r.conflicts
        assert s.get(ep(s, B0)).content == before
    finally:
        s.close()
    # codex #6: two enforced twins, neither the stored hash: the stored copy stays
    s = Store(tmp_path / "tt.db", audit=False)
    try:
        import_ledger(s, v3([(a, True, []), (b, True, [A0])]))
        before = s.get(ep(s, B0)).content
        b_twin2 = ledger("bob", [{**RETIRE, "reason": "twin two"}])[0]
        r = import_ledger(s, v3([(a, True, []), (b_twin, True, [A0]), (b_twin2, True, [A0])],
                                seq=2))
        assert not r.replaced_in_place and s.get(ep(s, B0)).content == before
    finally:
        s.close()


def test_l3r1_1006_rewired_row_takeover_through_its_standin(tmp_path):
    a = ledger("alice", [{**RULING, "ts": "2026-01-01T00:00:00Z"}])[0]
    b = ledger("bob", [{"type": "retire", "supersedes": [A0], "ts": "2026-01-02T00:00:00Z"}])[0]
    s = Store(tmp_path / "rw.db", audit=False)
    try:
        import_ledger(s, v3([(a, True, []), (b, True, [A0])], key="x", root="r1"))
        ea, eb = ep(s, A0), ep(s, B0)
        c = s.record("a local note that replaces bob's retire", "observation")
        s._conn.execute("INSERT INTO supersessions (old_id, new_id, source) VALUES (?, ?, 'me')",
                        (eb, c.id))
        s._conn.commit()
        assert s.delete(eb, team_operator=True)        # A -> C rewired, owned by x
        assert s.team_owned(old_id=ea, new_id=c.id)
        # a new key of ANOTHER root whose stream carries the exact line B takes x over
        r = import_ledger(s, v3([(a, True, []), (b, True, [])], key="y", root="r2"))
        st = {k["key"]: k for k in s.team_snapshot_status()["keys"]}
        assert not st["x"]["active"] and (ea, c.id) not in links(s)
    finally:
        s.close()
