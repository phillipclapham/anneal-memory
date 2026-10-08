"""KL-09 (2026-10-07): 13 of flow's 29 crystals had empty evidence (11 from one
2026-08-31 wrap), so the evidence edge in retrieval could never surface them.
Two paths produced it: a crystallize from a carried-forward line (no [evidence:] tag)
and a re-crystallize, which REPLACED good evidence with that empty list."""
from anneal_memory import Store
from anneal_memory.crystal import CrystalStore


def test_recrystallize_without_evidence_keeps_the_old_evidence(tmp_path):
    cs = CrystalStore(tmp_path / "c.json")
    cs.crystallize(name="derive_dont_invent", level=3, explanation="x", evidence=["aaaa1111"])
    cs.crystallize(name="derive_dont_invent", level=4, explanation="y", evidence=None)
    assert cs.get("derive_dont_invent")["evidence"] == ["aaaa1111"]
    cs.crystallize(name="derive_dont_invent", level=4, explanation="z",
                   evidence=["bbbb2222", "aaaa1111"])
    assert cs.get("derive_dont_invent")["evidence"] == ["aaaa1111", "bbbb2222"]
    cs.update("derive_dont_invent", evidence=[])  # the explicit path can still clear
    assert cs.get("derive_dont_invent")["evidence"] == []


def test_ground_empty_evidence(tmp_path):
    s = Store(tmp_path / "m.db", audit=False)
    cs = CrystalStore(tmp_path / "c.json")
    try:
        hit = s.record("the seat re-derived it: derive_dont_invent applied", "observation")
        s.record("derive_dont_invent_more is a different name", "observation")
        s.record("deriveXdontXinvent must not match through LIKE wildcards", "observation")
        old = s.record("old seam note naming derive_dont_invent, review rounds", "observation")
        new = s.record("corrected seam note, review rounds", "observation")
        assert s.supersede(old_id=old.id, new_id=new.id)
        s.record("harness_before_model, cited", "observation")
        cs.crystallize(name="derive_dont_invent", level=3, explanation="x")
        cs.crystallize(name="has_evidence", level=3, explanation="x", evidence=["cccc3333"])
        cs.crystallize(name="nobody_names_me", level=3, explanation="x")

        dry = cs.ground_empty_evidence(s, dry_run=True)
        assert dry == {"derive_dont_invent": [hit.id[:8]], "nobody_names_me": []}
        assert cs.get("derive_dont_invent")["evidence"] == []

        assert cs.ground_empty_evidence(s) == dry
        got = cs.get("derive_dont_invent")
        assert got["evidence"] == [hit.id[:8]] and "ground_empty_evidence" in got["notes"][-1]
        assert cs.get("has_evidence")["evidence"] == ["cccc3333"]
        assert cs.ground_empty_evidence(s) == {"nobody_names_me": []}  # idempotent
    finally:
        s.close()
