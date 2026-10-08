"""KL-09 (2026-10-07): 13 of flow's 29 crystals had empty evidence (11 from one
2026-08-31 wrap), so the evidence edge in retrieval could never surface them.
Two paths produced it: a crystallize from a carried-forward line (no [evidence:] tag)
and a re-crystallize, which REPLACED good evidence with that empty list."""
from anneal_memory import Store
from anneal_memory.crystal import PROVISIONAL_EVIDENCE, CrystalStore


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


def test_revive_without_evidence_keeps_the_retired_rows(tmp_path):
    cs = CrystalStore(tmp_path / "c.json")
    cs.crystallize(name="p", level=3, explanation="x", evidence=["aaaa1111"])
    cs.retire("p", kind="obsolete", reason="t")
    cs.crystallize(name="p", level=3, explanation="x")
    assert cs.get("p")["evidence"] == ["aaaa1111"]
    cs.retire("p", kind="obsolete", reason="t")
    cs.crystallize(name="p", level=3, explanation="x", evidence=["cccc3333"])
    assert cs.get("p")["evidence"] == ["cccc3333"]  # fresh evidence starts fresh


def _store(tmp_path):
    return Store(tmp_path / "m.db", audit=False), CrystalStore(tmp_path / "c.json")


def test_ground_empty_evidence(tmp_path):
    s, cs = _store(tmp_path)
    try:
        first = s.record("the incident: we derive_dont_invent from PyPI", "observation")
        s.record("derive_dont_invent_more is a different name", "observation")
        s.record("deriveXdontXinvent must not match through LIKE wildcards", "observation")
        s.record("derive_dont_invent-ish is hyphen-joined, not the name", "observation")
        old = s.record("old seam note naming derive_dont_invent, review rounds", "observation")
        new = s.record("corrected seam note, review rounds", "observation")
        assert s.supersede(old_id=old.id, new_id=new.id)
        later = s.record("again derive_dont_invent, later", "observation")
        cs.crystallize(name="derive_dont_invent", level=3, explanation="x")
        cs.crystallize(name="has_evidence", level=3, explanation="x", evidence=["cccc3333"])
        cs.crystallize(name="nobody_names_me", level=3, explanation="x")

        dry = cs.ground_empty_evidence(s, dry_run=True, limit=1)
        assert dry["derive_dont_invent"].status == "would_ground"
        assert dry["derive_dont_invent"].evidence == [first.id[:8]]   # OLDEST first
        assert dry["nobody_names_me"].status == "no_episode"
        assert "has_evidence" not in dry
        assert cs.get("derive_dont_invent")["evidence"] == []

        got = cs.ground_empty_evidence(s)
        assert got["derive_dont_invent"].evidence == [first.id[:8], later.id[:8]]
        rec = cs.get("derive_dont_invent")
        assert rec["evidence"] == [first.id[:8], later.id[:8]]
        assert rec[PROVISIONAL_EVIDENCE] == rec["evidence"]
        assert cs.get("has_evidence")["evidence"] == ["cccc3333"]
        assert list(cs.ground_empty_evidence(s)) == ["nobody_names_me"]   # idempotent

        # the first REAL evidence replaces the provisional ids
        cs.crystallize(name="derive_dont_invent", level=4, explanation="y",
                       evidence=["dddd4444"])
        rec = cs.get("derive_dont_invent")
        assert rec["evidence"] == ["dddd4444"] and PROVISIONAL_EVIDENCE not in rec
    finally:
        s.close()


def test_an_episode_naming_several_live_patterns_is_not_used(tmp_path):
    """L2 1007 [run]: an end-of-day log naming many patterns, used as everyone's
    evidence, becomes a hub the evidence edge discounts for all of them."""
    s, cs = _store(tmp_path)
    try:
        s.record("EOD: alpha_pattern, beta_pattern and gamma_pattern all fired", "observation")
        own = s.record("the beta_pattern incident itself", "observation")
        for n in ("alpha_pattern", "beta_pattern", "gamma_pattern"):
            cs.crystallize(name=n, level=3, explanation="x")
        got = cs.ground_empty_evidence(s)
        assert got["beta_pattern"].evidence == [own.id[:8]]
        assert got["beta_pattern"].hubs_skipped == 1
        assert got["alpha_pattern"].status == "no_episode"
        assert got["alpha_pattern"].hubs_skipped == 1
    finally:
        s.close()


def test_a_working_set_pattern_name_also_makes_a_hub(tmp_path):
    """L2 1007 follow-up [run]: a desk log naming one crystal and working-set patterns
    (pattern_history only) must not be picked over the incident."""
    s, cs = _store(tmp_path)
    try:
        s.record("desk: beta_pattern fired, plus ws_one and ws_two", "observation")
        own = s.record("the beta_pattern incident", "observation")
        s._conn.executemany(
            "INSERT INTO pattern_history (pattern_name, max_level_reached, "
            "explanation_corpus, last_explanation, last_seen_at) "
            "VALUES (?, 2, 'x', 'x', '2026-10-01')", [("ws_one",), ("ws_two",)])
        s._conn.commit()
        cs.crystallize(name="beta_pattern", level=3, explanation="x")
        got = cs.ground_empty_evidence(s, limit=1)["beta_pattern"]
        assert got.evidence == [own.id[:8]] and got.hubs_skipped == 1
    finally:
        s.close()


def test_an_explicit_update_clears_the_provisional_mark(tmp_path):
    s, cs = _store(tmp_path)
    try:
        e = s.record("the alpha_pattern incident", "observation")
        cs.crystallize(name="alpha_pattern", level=3, explanation="x")
        cs.ground_empty_evidence(s)
        cs.update("alpha_pattern", evidence=[e.id[:8]])     # the operator confirms it
        assert PROVISIONAL_EVIDENCE not in cs.get("alpha_pattern")
        cs.crystallize(name="alpha_pattern", level=4, explanation="y", evidence=["eeee5555"])
        assert cs.get("alpha_pattern")["evidence"] == [e.id[:8], "eeee5555"]
    finally:
        s.close()


def test_a_revive_keeps_the_provisional_mark(tmp_path):
    """L1 1007 (mutation M14): a revived pattern's grounded ids stay provisional, so
    the first real evidence still replaces them."""
    s, cs = _store(tmp_path)
    try:
        s.record("the alpha_pattern incident", "observation")
        cs.crystallize(name="alpha_pattern", level=3, explanation="x")
        cs.ground_empty_evidence(s)
        cs.retire("alpha_pattern", kind="obsolete", reason="t")
        cs.crystallize(name="alpha_pattern", level=3, explanation="x")
        assert cs.get("alpha_pattern")[PROVISIONAL_EVIDENCE]
        cs.crystallize(name="alpha_pattern", level=3, explanation="x", evidence=["ffff6666"])
        assert cs.get("alpha_pattern")["evidence"] == ["ffff6666"]
    finally:
        s.close()


def test_a_dotted_name_does_not_match_inside_another(tmp_path):
    s, cs = _store(tmp_path)
    try:
        s.record("see foo.bar for the incident", "observation")
        cs.crystallize(name="foo", level=3, explanation="x")
        assert cs.ground_empty_evidence(s)["foo"].status == "no_episode"
    finally:
        s.close()
