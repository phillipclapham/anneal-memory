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
        r = _wrap(s, _doc("- a | 2x (2026-10-01)"), "2026-10-07")
        assert "drift" not in r and s.drift_status()["wrap_id"] is None
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
    assert "No save has checked" in run("probe", "status").stdout
    assert run("probe", "retire", "9").returncode == 1
