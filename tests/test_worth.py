"""Outcome write-back, surfaced fold, Worth counters, and the re-warm dedup.

Each test replays a path that was run by hand on a copy of a real store first
(anneal-sota-0930 §3.1(b) / §3.2 build, 2026-10-02).
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from anneal_memory import CrystalStore, FLOW_SCHEMA, Store, prepare_wrap
from anneal_memory.crystal import CrystalError
from anneal_memory.worth import (
    ExposedRef,
    ExposureLabel,
    OutcomeLog,
    compute_worth,
    fold_surfaced,
    outcome_log_path,
)


def test_outcome_log_merges_by_exposure_and_skips_bad_lines(tmp_path):
    log = OutcomeLog(outcome_log_path(tmp_path / "mem.db"))
    assert log.path.name == "mem.outcomes.jsonl"
    log.record("ev1", [ExposureLabel("crystal", "p", "followed"),
                       ExposureLabel("crystal", "q", "ignored")])
    log.record("ev1", [ExposureLabel("crystal", "p", "not_applicable")])
    log.record("ev1", [], outcome="failure")  # outcome written later, labels kept
    log.record("ev1", [ExposureLabel("crystal", "q", "ignored")])  # keeps the outcome
    with open(log.path, "a", encoding="utf-8") as f:
        f.write('{"v": 1, "exposure_id": "torn"\n')  # a torn line from a crash
    latest, bad = log.latest()
    assert bad == 1
    assert latest["ev1"]["outcome"] == "failure"
    assert latest["ev1"]["items"] == [
        {"kind": "crystal", "ref": "p", "followed": "not_applicable"},
        {"kind": "crystal", "ref": "q", "followed": "ignored"},
    ]
    for bad_label in ("maybe", "", None):
        with pytest.raises(ValueError):
            ExposureLabel("crystal", "p", bad_label)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        log.record("ev\n2", [ExposureLabel("crystal", "p", "followed")])
    with pytest.raises(ValueError):
        log.record("ev2", [])


def test_worth_counts_retrieved_by_label_and_credits_each_episode_once(tmp_path):
    crystal = CrystalStore(tmp_path / "mem.crystal.json")
    crystal.crystallize(name="p", level=3, explanation="x", evidence=["e1", "e2"])
    crystal.crystallize(name="q", level=2, explanation="y", evidence=["e1", "e3"])
    log = OutcomeLog(tmp_path / "mem.outcomes.jsonl")
    log.record("a", [ExposureLabel("crystal", "p", "followed"),
                     ExposureLabel("crystal", "q", "ignored")], outcome="success")
    log.record("b", [ExposureLabel("crystal", "p", "followed")], outcome="failure")
    log.record("c", [ExposureLabel("crystal", "p", "followed")])  # outcome unknown
    log.record("d", [ExposureLabel("crystal", "gone", "followed"),
                     ExposureLabel("episode", "e9", "followed")], outcome="success")
    # p and q both cite e1 and e1 is also labelled directly: one exposure, one count
    log.record("e", [ExposureLabel("crystal", "p", "ignored"),
                     ExposureLabel("crystal", "q", "followed"),
                     ExposureLabel("episode", "e1", "followed")], outcome="failure")

    report = compute_worth(log, crystal)
    rows = {r.ref: r for r in report.crystals}
    p = rows["p"]
    assert (p.followed, p.ignored, p.success, p.failure) == (3, 1, 1, 2)
    assert p.table["followed"] == {"success": 1, "failure": 1, "unknown": 1}
    assert p.table["ignored"] == {"success": 0, "failure": 1, "unknown": 0}
    # an ignored exposure still counts as retrieved-with-outcome (Memory Worth)
    assert (rows["q"].success, rows["q"].failure) == (1, 1)
    assert rows["gone"].live is False and rows["gone"].success == 1
    eps = {r.ref: r for r in report.episodes}
    # exposures a (success, cited), b (failure, cited), e (failure, direct): 3, not 5
    assert (eps["e1"].success, eps["e1"].failure) == (1, 2)
    assert (eps["e1"].credited_success, eps["e1"].credited_failure) == (1, 1)
    assert eps["e1"].table["followed"]["failure"] == 1
    assert (eps["e3"].success, eps["e3"].failure) == (1, 1)
    assert (eps["e9"].success, eps["e9"].credited_success) == (1, 0)
    assert report.exposures == 5
    # report-only: computing Worth writes nothing
    assert crystal.get("p").get("surfaced_count") is None
    assert rows["p"].surfaced_count is None  # no fold has run: unknown, not zero


def test_fold_counts_once_leaves_activation_alone_and_defers_fresh_receipts(tmp_path):
    crystal = CrystalStore(tmp_path / "mem.crystal.json")
    crystal.crystallize(name="p", level=3, explanation="x", today=date(2026, 6, 1))
    crystal.crystallize(name="q", level=3, explanation="y", today=date(2026, 6, 1))
    crystal.crystallize(name="r", level=3, explanation="w", today=date(2026, 6, 1))
    now = datetime(2026, 10, 2, 12, 0, 0, tzinfo=timezone.utc)

    def rec(ts, names, qd=None):
        r = {"event_id": ts, "ts": ts,
             "exposed": [{"pattern": n, "rank": i, "source": "evidence_edge"}
                         for i, n in enumerate(names)]}
        if qd:
            r["query_date"] = qd
        return json.dumps(r)

    receipts = tmp_path / "receipts.jsonl"
    receipts.write_text("\n".join([
        rec("2026-10-01T10:00:00Z", ["p", "p", "gone"], qd="2026-10-01"),
        rec("2026-10-01T10:00:00Z", ["p"], qd="2026-10-01"),  # same event_id: rotation
        rec("2026-10-02T11:58:30Z", ["p", "q"]),
        rec("2026-10-02T11:59:30Z", ["q"]),  # inside the 60s skew: next fold
        "not json",
        rec("2026-10-02T09:00:00Z", []),
        rec("2026-10-02T11:00:00Z", ["q"], qd="9999-99-99"),  # bogus local date
        rec("2026-10-02T11:01:00Z", ["q"], qd="2099-01-01"),  # valid date, far future
        rec("2026-10-02T11:02:00Z", ["r"], qd="2000-01-01"),  # valid date, far past
        json.dumps({"ts": "2026-10-02T11:03:00Z", "exposed": [{"pattern": "p"}]}),  # no id
    ]) + "\n")

    with pytest.raises(FileNotFoundError):
        fold_surfaced(crystal, [tmp_path / "typo.jsonl"], now=now)
    assert "surfaced_fold" not in json.loads(crystal.path.read_text())  # mark not moved

    r1 = fold_surfaced(crystal, [receipts, tmp_path / "rotated-away.jsonl"], now=now)
    assert (r1.receipts_folded, r1.exposures_counted, r1.duplicates_skipped) == (5, 6, 1)
    assert r1.names_unknown == {"gone": 1} and r1.lines_skipped == 1
    assert r1.event_id_missing == 1
    assert r1.paths_missing == [str(tmp_path / "rotated-away.jsonl")]
    assert r1.mark == "2026-10-02T11:59:00Z"
    p = crystal.get("p")
    assert (p["surfaced_count"], p["last_surfaced_on"]) == (2, "2026-10-02")
    assert p["last_activated_on"] == "2026-06-01"  # exposure is not activation

    r2 = fold_surfaced(crystal, [receipts], now=now)
    assert r2.receipts_folded == 0 and crystal.get("p")["surfaced_count"] == 2

    assert crystal.get("q")["last_surfaced_on"] == "2026-10-02"  # bogus date ignored
    assert crystal.get("r")["last_surfaced_on"] == "2026-10-02"  # out-of-range date ignored
    r3 = fold_surfaced(crystal, [receipts], now=now + timedelta(minutes=5))
    assert r3.receipts_folded == 1 and crystal.get("q")["surfaced_count"] == 4
    crystal.retire("p", kind="superseded")
    crystal.crystallize(name="p", level=3, explanation="x")
    assert crystal.get("p")["surfaced_count"] == 2  # a revive keeps its history

    doc = json.loads(crystal.path.read_text())
    for bad_mark in (None, {"through": "garbage"}):
        doc["surfaced_fold"] = bad_mark
        crystal.path.write_text(json.dumps(doc))
        with pytest.raises(CrystalError):  # an unreadable mark never re-counts history
            fold_surfaced(crystal, [receipts], now=now + timedelta(days=1))
        assert crystal.get("p")["surfaced_count"] == 2
    crystal.crystallize(name="fresh", level=3, explanation="v")
    rows = {r.ref: r for r in compute_worth(OutcomeLog(tmp_path / "x.jsonl"), crystal).crystals}
    assert rows["fresh"].surfaced_count is None  # unknown under a bad mark, not 0
    assert rows["p"].surfaced_count == 2  # a stored count is still reported


def test_rewarm_candidates_skip_patterns_already_in_the_working_set(tmp_path):
    store = Store(tmp_path / "mem.db", project_name="flow")
    store.set_section_schema(FLOW_SCHEMA)
    crystal = CrystalStore(tmp_path / "mem.crystal.json")
    crystal.crystallize(name="hot_in_set", level=3, explanation="x", today=date.today())
    crystal.crystallize(name="hot_out_of_set", level=3, explanation="y", today=date.today())
    # a name the graduation regex's alphabet cannot parse is still recognised
    crystal.crystallize(name="2fast/név", level=3, explanation="z", today=date.today())
    pad = " felt prose padding clause to push this line's mass past the floor." * 3
    store.save_continuity(
        "# flow — Memory (test)\n\n## State\nbuilding.\n\n## Active Threads\n- a\n\n"
        f"## Patterns\n- hot_in_set | 3x (2026-06-01) [evidence: abcd1234 \"why\"]{pad}\n"
        f"- !! 2fast/név | 3x (2026-06-01) [evidence: abcd1234 \"why\"]{pad}\n\n"
        "## Decisions\n[decided] x.\n\n## Context\ny.\n\n## Understanding\nz.\n"
    )
    store.record("an episode about substrate topics worth at least eighty characters here now.",
                 "observation")
    result = prepare_wrap(store, crystal_store=crystal)
    assert "hot_out_of_set" in result["rewarm_candidates"]
    assert "hot_in_set" not in result["rewarm_candidates"]
    assert "2fast/név" not in result["rewarm_candidates"]


def test_unlabelled_exposures_count_in_their_own_column(tmp_path):
    """1002+13 flow-seat: flow labels only verbatim cites, so most real outcomes
    carried no label and were invisible to Worth. Phill 2026-10-02: count
    exposed + outcome + no label in a separate column, never in success/failure."""
    crystal = CrystalStore(tmp_path / "mem.crystal.json")
    crystal.crystallize(name="p", level=3, explanation="x", evidence=["e1"])
    crystal.crystallize(name="q", level=2, explanation="y", evidence=["e2"])
    log = OutcomeLog(tmp_path / "mem.outcomes.jsonl")
    seen = [ExposedRef("crystal", "p"), ExposedRef("crystal", "q"), ExposedRef("episode", "e7")]
    log.record("a", [], outcome="success", exposed=seen)
    log.record("b", [ExposureLabel("crystal", "p", "followed")], outcome="failure", exposed=seen)
    log.record("c", [ExposureLabel("crystal", "q", "ignored")])  # outcome unknown, no exposed
    log.record("c", [], outcome="failure", exposed=[ExposedRef("crystal", "p")])
    report = compute_worth(log, crystal)
    rows = {r.ref: r for r in report.crystals}
    p, q = rows["p"], rows["q"]
    # a: p unlabelled+success. b: p labelled, so not unlabelled. c: p unlabelled+failure.
    assert (p.unlabelled_success, p.unlabelled_failure, p.unlabelled_unknown) == (1, 1, 0)
    assert (p.success, p.failure) == (0, 1)  # only the labelled exposure b
    # a: q unlabelled+success. b: q unlabelled+failure. c: q labelled.
    assert (q.unlabelled_success, q.unlabelled_failure) == (1, 1)
    assert (q.success, q.failure) == (0, 1)
    eps = {r.ref: r for r in report.episodes}
    assert (eps["e7"].unlabelled_success, eps["e7"].unlabelled_failure) == (1, 1)
    assert (eps["e7"].success, eps["e7"].failure) == (0, 0)
    # p's evidence e1 is credited by the labelled exposure b only; the unlabelled
    # exposures a and c credit nothing to it
    assert (eps["e1"].credited_success, eps["e1"].credited_failure) == (0, 1)
    assert eps["e1"].unlabelled_success == eps["e1"].unlabelled_failure == 0
    # L1: an episode already credited through a labelled crystal's evidence in the
    # same exposure is not also unlabelled (it was counted once, as credited)
    log.record("f", [ExposureLabel("crystal", "p", "followed")], outcome="success",
               exposed=[ExposedRef("crystal", "p"), ExposedRef("episode", "e1")])
    e1 = {r.ref: r for r in compute_worth(log, crystal).episodes}["e1"]
    assert (e1.credited_success, e1.unlabelled_success) == (1, 0)
    latest, _ = log.latest()
    assert {(e["kind"], e["ref"]) for e in latest["c"]["exposed"]} == {("crystal", "p")}
    # an unreadable exposed entry is dropped, not the record with its labels
    with open(log.path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"v": 1, "exposure_id": "g", "ts": "2026-10-02T00:00:00Z",
                            "outcome": "success", "items": [],
                            "exposed": [{"kind": "future", "ref": "x"},
                                        {"kind": "crystal", "ref": "q"}]}) + "\n")
    latest, bad = log.latest()
    assert bad == 0 and latest["g"]["exposed"] == [{"kind": "crystal", "ref": "q"}]
    with pytest.raises(ValueError):
        log.record("d", [], exposed=seen)  # exposed alone never makes a record
    with pytest.raises(ValueError):
        ExposedRef("note", "p")

