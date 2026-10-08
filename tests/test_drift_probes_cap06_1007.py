"""CAP-06 (2026-10-07): drift probes, the operator's instrument for meaning drift.

The operator declares what must survive; every save checks it lexically against the
saved text, records the verdict with the wrap, and never blocks or rewrites."""
import subprocess
import sys

import pytest

from anneal_memory import Store, prepare_wrap, validated_save_continuity
from anneal_memory.crystal import CrystalStore
from anneal_memory.drift import evaluate_probes


def _doc(patterns, facts="", understanding="who we are."):
    return (f"# t\n\n## State\ns.\n\n## Patterns\n{patterns}\n\n## Decisions\nd.\n\n"
            f"## Context\n{facts}\n")


def _wrap(store, text, today):
    store.record(f"episode {today}: a substrate observation about the topic.", "observation")
    token = prepare_wrap(store, max_chars=40000)["wrap_token"]
    return validated_save_continuity(store, text, today=today, wrap_token=token)


def test_evaluate_statuses():
    probes = [
        {"id": 1, "kind": "pattern", "name": "alpha", "min_level": 3},
        {"id": 2, "kind": "pattern", "name": "beta", "min_level": 4},
        {"id": 3, "kind": "pattern", "name": "gamma", "min_level": 2},
        {"id": 4, "kind": "pattern", "name": "delta", "min_level": 2},
        {"id": 5, "kind": "fact", "text": "Desi is a shepherd mix", "section": "Context"},
        {"id": 6, "kind": "fact", "text": "the baton belongs to the EOD seat"},
    ]
    text = _doc("- alpha | 5x (2026-10-01)\n- beta | 3x (2026-10-01)",
                facts="Desi, the dog, is a shepherd mix.")
    out = {r["probe_id"]: r["status"] for r in evaluate_probes(
        text, probes, pattern_levels={"alpha": 5, "beta": 3}, live_crystals=["gamma"])}
    assert out == {1: "held", 2: "weakened", 3: "crystallized", 4: "lost",
                   5: "held", 6: "lost"}


def test_a_fact_in_the_wrong_section_is_lost():
    probes = [{"id": 1, "kind": "fact", "text": "Desi is a shepherd mix", "section": "State"}]
    text = _doc("", facts="Desi is a shepherd mix")
    assert evaluate_probes(text, probes, pattern_levels={})[0]["status"] == "lost"


def test_add_validation(tmp_path):
    s = Store(tmp_path / "m.db", audit=False)
    try:
        for kw in ({}, {"pattern": "a", "fact": "b c"}, {"pattern": "a", "min_level": 1},
                   {"pattern": "a", "section": "S"}, {"fact": "the of a"},
                   {"fact": "real words", "min_level": 3}, {"pattern": " "}):
            with pytest.raises(ValueError):
                s.add_drift_probe(**kw)
    finally:
        s.close()


def test_every_save_records_the_verdict_and_never_blocks(tmp_path):
    s = Store(tmp_path / "m.db", project_name="t")
    try:
        p1 = s.add_drift_probe(pattern="alpha", min_level=2)
        s.add_drift_probe(fact="the hub runs on soupcan", section="Context")
        r = _wrap(s, _doc("- alpha | 2x (2026-10-01)", facts="the hub runs on soupcan"),
                  "2026-10-06")
        assert r["drift"]["counts"]["held"] == 2 and r["drift"]["not_held"] == []
        # alpha drops out and the fact is rewritten away: the save still commits
        r = _wrap(s, _doc("- beta | 2x (2026-10-01)", facts="moved to a new box"),
                  "2026-10-07")
        assert r["drift"]["counts"]["lost"] == 2
        st = s.drift_status()
        assert st["wrap_id"] == 2
        assert [p["status"] for p in st["probes"]] == ["lost", "lost"]
        assert st["probes"][0]["since_wrap"] == 2
        assert s.retire_drift_probe(p1) and not s.retire_drift_probe(p1)
        assert [p["id"] for p in s.drift_status()["probes"]] == [p1 + 1]
    finally:
        s.close()


def test_no_probes_no_key(tmp_path):
    s = Store(tmp_path / "m.db", project_name="t")
    try:
        assert s.drift_status()["wrap_id"] is None
        r = _wrap(s, _doc("- a | 2x (2026-10-01)"), "2026-10-07")
        assert "drift" not in r
        st = s.drift_status()
        assert st["wrap_id"] == 1 and st["probes"] == []
    finally:
        s.close()


def test_a_crystallized_pattern_is_a_move_not_a_loss(tmp_path):
    s = Store(tmp_path / "m.db", project_name="t")
    cs = CrystalStore(tmp_path / "m.crystal.json")
    try:
        s.add_drift_probe(pattern="alpha")
        cs.crystallize(name="alpha", level=3, explanation="x")
        s.record("episode: a substrate observation about the topic.", "observation")
        token = prepare_wrap(s, max_chars=40000)["wrap_token"]
        r = validated_save_continuity(s, _doc("- beta | 2x (2026-10-01)"),
                                      today="2026-10-07", wrap_token=token, crystal_store=cs)
        assert r["drift"]["counts"]["crystallized"] == 1
    finally:
        s.close()


def test_cli_round_trip(tmp_path):
    db = str(tmp_path / "c.db")
    run = lambda *a: subprocess.run([sys.executable, "-m", "anneal_memory.cli", "--db", db, *a],
                                    capture_output=True, text=True)
    Store(db).close()
    assert run("probe", "add", "--pattern", "alpha").returncode == 0
    assert run("probe", "add", "--fact", "the of").returncode == 1
    assert "pattern alpha >= 2x" in run("probe", "list").stdout
    assert "No save yet" in run("probe", "status").stdout
    assert run("probe", "retire", "9").returncode == 1
    assert run("probe", "retire", "9", "--json").returncode == 1   # L1 1007



@pytest.mark.parametrize("fact,saved,status", [
    # L2 1007 [run]: each read "held" before
    ("client data does not go to Google", "client data does go to Google now.", "changed"),
    ("client data goes to Google", "client data never goes to Google.", "changed"),
    ("the rate is $85 an hour", "the rate is $65 an hour.", "lost"),
    ("rent due 10-22", "rent due 10-29.", "lost"),
    ("Arlington audit is billed on her trigger",
     "Arlington was audited. The hub is billed monthly. Her trigger is a new site.", "lost"),
    # and false "lost" before
    ("he decided after the call", "He decides, after the call.", "held"),
    ("rent is due on the twenty second of the month",
     "rent is due on the twenty second\nof the month.", "held"),
    # residue run on flow's store [run]: emphasis around a sentence end, and an
    # unrelated "no" later in the same sentence
    ("the EOD is his; he runs it every day and calls it sacred",
     "**Some rituals are his, not the harness's.** The EOD is one: he runs it every day "
     "and calls it sacred, and no seat offers to run it for him.", "held"),
])
def test_fact_matching(fact, saved, status):
    probes = [{"id": 1, "kind": "fact", "text": fact}]
    assert evaluate_probes(_doc("", facts=saved), probes, pattern_levels={})[0]["status"] \
        == status


def test_a_bad_probe_is_unchecked_never_a_gate(tmp_path):
    """L1 1007 [run]: an unknown kind (a newer version's) or a NULL text refused every save."""
    s = Store(tmp_path / "m.db", project_name="t")
    try:
        s.add_drift_probe(fact="the hub runs on soupcan")
        s._conn.execute("INSERT INTO drift_probes (kind, name) VALUES ('mood', 'x')")
        s._conn.execute("INSERT INTO drift_probes (kind, text) VALUES ('fact', NULL)")
        s._conn.commit()
        r = _wrap(s, _doc("", facts="the hub runs on soupcan."), "2026-10-07")
        assert r["drift"]["counts"] == {"held": 1, "changed": 0, "weakened": 0,
                                        "crystallized": 0, "lost": 0, "unchecked": 2}
    finally:
        s.close()


def test_pattern_probe_defaults_to_its_current_level(tmp_path):
    s = Store(tmp_path / "m.db", project_name="t")
    try:
        _wrap(s, _doc("- alpha | 7x (2026-10-01)"), "2026-10-06")
        pid = s.add_drift_probe(pattern="alpha")
        assert [p["min_level"] for p in s.list_drift_probes() if p["id"] == pid] == [7]
        assert s.list_drift_probes()[0]["min_level"] == 7
        r = _wrap(s, _doc("- alpha | 3x (2026-10-01)"), "2026-10-07")
        assert r["drift"]["counts"]["weakened"] == 1
    finally:
        s.close()


def test_drift_is_in_the_audit_and_the_worklist_lists_graduations(tmp_path):
    import json as _json
    s = Store(tmp_path / "m.db", project_name="t")
    try:
        s.add_drift_probe(fact="the hub runs on soupcan")
        ep = s.record("we moved the hub to soupcan and it runs there now", "observation")
        token = prepare_wrap(s, max_chars=40000)["wrap_token"]
        line = (f'- hub_location_is_soupcan | 2x (2026-10-07) '
                f'[evidence: {ep.id[:8]} "the hub runs on soupcan now"]')
        validated_save_continuity(s, _doc(line, facts="the hub runs on soupcan."),
                                  today="2026-10-07", wrap_token=token)
        events = [_json.loads(x) for x in s._audit._active_path.read_text().splitlines() if x]
        saved = [e for e in events if e["event"] == "continuity_saved"][-1]["data"]
        assert saved["drift"]["counts"]["held"] == 1
        added = [e for e in events if e["event"] == "drift_probe_added"]
        assert added and added[0]["data"]["kind"] == "fact"
        g = s.drift_status()["graduated"]
        assert [(x["name"], x["level"]) for x in g] == [("hub_location_is_soupcan", 2)]
    finally:
        s.close()
