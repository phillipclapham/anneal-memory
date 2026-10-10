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
    FOLLOWED_VALUES,
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



def test_cli_exposed_records_any_ref_and_warns_on_a_label_suffix(capsys):
    """L3 flipped this twice: refusing a ref ending in a label rejected a valid
    crystal named foo=followed; accepting it silently recorded pasted --item
    syntax. Phill 2026-10-02: record it, warn on stderr."""
    from anneal_memory.cli import _parse_exposed

    assert _parse_exposed("crystal:a=b") == ExposedRef("crystal", "a=b")
    assert capsys.readouterr().err == ""
    for raw in ("crystal:foo=followed", "crystal:x=ignored", "crystal:y=not_applicable",
                "crystal:z=Followed ", "crystal:=followed"):
        kind, _, ref = raw.partition(":")
        assert _parse_exposed(raw) == ExposedRef(kind, ref)
        assert "with --item" in capsys.readouterr().err
    with pytest.raises(ValueError):
        _parse_exposed("no-colon")


def test_record_if_missing_keeps_a_human_label_written_mid_check(tmp_path):
    """flow's labeller did latest() then record(): two lock spans. A human
    correction landing between them was overwritten in the merge (1003 design (a)).
    Reproduced first, then the same interleaving through record_if_missing."""
    fcntl = pytest.importorskip("fcntl")
    import os
    import threading
    import time

    human = [ExposureLabel("crystal", "p", "ignored")]
    machine = [ExposureLabel("crystal", "p", "followed")]
    seen = [ExposedRef("crystal", "p"), ExposedRef("crystal", "q")]

    # BEFORE: the check-then-append pattern loses the human's label
    log = OutcomeLog(tmp_path / "before.outcomes.jsonl")
    cur = log.latest()[0].get("ev")  # the machine checks: nothing recorded yet
    log.record("ev", human)  # the human corrects in the gap
    if not cur:
        log.record("ev", machine, outcome="success", exposed=seen)
    assert log.latest()[0]["ev"]["items"][0]["followed"] == "followed"  # human lost

    # AFTER: hold the log's lock as record() does, so the human's write is in
    # flight while record_if_missing starts; it must wait, then see and keep it
    log = OutcomeLog(tmp_path / "after.outcomes.jsonl")
    fd = os.open(log.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    got: list = []
    t = threading.Thread(target=lambda: got.append(
        log.record_if_missing("ev", machine, outcome="success", exposed=seen)))
    t.start()
    time.sleep(0.2)
    assert t.is_alive()  # blocked on the lock, has not read the log yet
    os.write(fd, (json.dumps({"v": 1, "exposure_id": "ev", "ts": "2026-10-03T00:00:00Z",
                              "outcome": None, "items": [
                                  {"kind": "crystal", "ref": "p", "followed": "ignored"}]})
                  + "\n").encode())
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
    t.join(5)
    # labels were present, so only the missing outcome (and exposed) went in
    assert got[0]["items"] == [] and got[0]["outcome"] == "success"
    merged = log.latest()[0]["ev"]
    assert merged["items"] == [{"kind": "crystal", "ref": "p", "followed": "ignored"}]
    assert merged["outcome"] == "success" and len(merged["exposed"]) == 2
    # nothing missing now: no write, and only exposed being new never writes alone
    size = log.path.stat().st_size
    assert log.record_if_missing("ev", machine, outcome="failure", exposed=seen) is None
    assert log.record_if_missing(
        "ev", machine, outcome="success", exposed=[ExposedRef("crystal", "new")]) is None
    assert log.path.stat().st_size == size
    with pytest.raises(ValueError):  # validated as record() validates
        log.record_if_missing("ev2", [])
    # a fresh exposure gets everything
    assert log.record_if_missing("ev2", machine)["items"] == [
        {"kind": "crystal", "ref": "p", "followed": "followed"}]


def test_worth_counts_receipt_exposures_with_no_record(tmp_path):
    """Design (b), option 4b: an exposure with no cite and no outcome never
    reaches the log; the receipts still prove it. Report-only, nothing written."""
    from anneal_memory.worth import load_receipts

    crystal = CrystalStore(tmp_path / "mem.crystal.json")
    crystal.crystallize(name="p", level=3, explanation="x", evidence=["e1"])
    crystal.crystallize(name="q", level=2, explanation="y", evidence=["e2"])
    log = OutcomeLog(tmp_path / "mem.outcomes.jsonl")
    log.record("r1", [ExposureLabel("crystal", "p", "followed")], outcome="success")
    live = tmp_path / "receipts.jsonl"
    rows = [
        {"event_id": "r1", "exposed": [{"pattern": "p"}, {"pattern": "q"}]},  # recorded
        {"event_id": "r2", "exposed": [{"pattern": "p"}, {"pattern": "q"}]},
        {"event_id": "r3", "exposed": [{"pattern": "q"}, {"pattern": "gone"}]},
        {"event_id": "r4", "exposed": []},  # not an exposure
        {"exposed": [{"pattern": "p"}]},  # no event_id: cannot be matched
    ]
    live.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    backup = tmp_path / "receipts.jsonl.1"
    backup.write_text(json.dumps(rows[1]) + "\n")  # a rotated copy of r2
    before = log.path.read_bytes()

    receipts, bad, missing = load_receipts([live, backup, tmp_path / "absent.jsonl"])
    assert bad == 1 and missing == [str(tmp_path / "absent.jsonl")]
    report = compute_worth(log, crystal, receipts=receipts)
    by = {r.ref: r for r in report.crystals}
    assert (by["p"].exposed_unrecorded, by["q"].exposed_unrecorded) == (1, 2)
    assert by["gone"].exposed_unrecorded == 1 and by["gone"].live is False
    assert (report.receipts_read, report.receipts_skipped) == (3, 1)
    assert by["p"].success == 1  # the recorded counters are untouched
    assert all(r.exposed_unrecorded is None for r in report.episodes)
    assert log.path.read_bytes() == before

    # receipts omitted: the report and its dict are exactly as before
    plain = compute_worth(log, crystal).as_dict()
    assert "receipts_read" not in plain
    assert all("exposed_unrecorded" not in r for r in plain["crystals"])
    with pytest.raises(FileNotFoundError):
        load_receipts([tmp_path / "nope.jsonl"])


def test_fold_on_a_store_with_no_crystal_file_does_not_create_it(tmp_path):
    """Diogenes 10-03: the crystal file's existence is the wrap path's persistent
    opt-in to the crystal tier, and a fold used to create it (the transaction
    saves on clean exit). Nothing is live to count, so the fold leaves it absent."""
    receipts = tmp_path / "receipts.jsonl"
    receipts.write_text('{"ts":"2026-10-01T00:00:00Z","event_id":"e1","exposed":[]}\n')
    path = tmp_path / "mem.crystal.json"
    result = fold_surfaced(CrystalStore(path), [receipts])
    assert not path.exists()
    assert result.store_missing and result.mark is None and result.previous_mark is None
    assert result.receipts_folded == 0
    with pytest.raises(FileNotFoundError):  # a wrong receipt path still refuses
        fold_surfaced(CrystalStore(path), [tmp_path / "nope.jsonl"])
    assert not path.exists()


def test_a_record_after_a_torn_line_starts_its_own_line(tmp_path):
    """A crash can leave the last line without its newline; the next append used
    to be glued onto it, so a good record was lost with the torn one."""
    log = OutcomeLog(tmp_path / "mem.outcomes.jsonl")
    log.record("a", [ExposureLabel("crystal", "p", "followed")])
    log.path.write_bytes(log.path.read_bytes()[:-1])  # the newline never made it
    log.record("b", [ExposureLabel("crystal", "p", "ignored")])
    assert log.latest()[0].keys() == {"a", "b"}
    with open(log.path, "ab") as f:
        f.write(b'{"v": 1, "exposure_id": "torn"')  # torn mid-record
    log.record_if_missing("c", [ExposureLabel("crystal", "q", "followed")])
    latest, bad = log.latest()
    assert latest.keys() == {"a", "b", "c"} and bad == 1  # only the torn line lost


@pytest.mark.parametrize("v,read", [(1, True), (True, False), (1.0, False), ("1", False),
                                    (2, False)])
def test_a_record_whose_version_is_not_the_int_1_is_skipped(tmp_path, v, read):
    # True == 1 and 1.0 == 1 in Python; only the int 1 is this log's version.
    log = OutcomeLog(tmp_path / "x.outcomes.jsonl")
    log.path.write_text(json.dumps({
        "v": v, "exposure_id": "e1", "ts": "2026-10-03T00:00:00Z", "outcome": "success",
        "items": [{"kind": "crystal", "ref": "a", "followed": "followed"}]}) + "\n")
    records, bad = log.read()
    assert (len(records), bad) == ((1, 0) if read else (0, 1))


# ⛔ FROZEN, NEVER EDITED: version-1 outcome-log lines in the exact shapes released
# code writes. A later anneal that changes a record's shape writes a NEW "v" and
# must still read these (Phill 2026-10-03, "3A", V-LOG). That is what makes an
# older process's late append after a newer anneal's migration a correct v1
# record rather than a corruption (the race named in the store-identity
# CHANGELOG entry). If this fails, the change dropped a released format.
_FROZEN_V1_LINES = (
    # 0.9.20-0.9.23: no "store" key (an unbound record)
    '{"v": 1, "exposure_id": "x1", "ts": "2026-10-02T00:00:00Z", "outcome": "success", '
    '"items": [{"kind": "crystal", "ref": "p", "followed": "followed"}], '
    '"exposed": [{"kind": "crystal", "ref": "p"}]}',
    # store identity: the same record stamped with its store
    '{"v": 1, "exposure_id": "x2", "ts": "2026-10-03T00:00:00Z", "outcome": "failure", '
    '"items": [{"kind": "episode", "ref": "e9", "followed": "ignored"}], "store": "s1"}',
)


def test_released_v1_outcome_records_stay_readable(tmp_path):
    log = OutcomeLog(tmp_path / "x.outcomes.jsonl", store_id="s1")
    log.path.write_text("\n".join(_FROZEN_V1_LINES) + "\n")
    latest, bad = log.latest()
    assert bad == 0
    assert latest["x1"]["outcome"] == "success"
    assert latest["x1"]["items"] == [{"kind": "crystal", "ref": "p", "followed": "followed"}]
    assert latest["x2"]["outcome"] == "failure"
    assert latest["x2"]["items"] == [{"kind": "episode", "ref": "e9", "followed": "ignored"}]


# -- `crystal get` records a followed label (a pull is the one non-guessed label) --

def _pull_cli(db, *args, timeout=60):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    env = dict(os.environ, PYTHONPATH=str(root))
    return subprocess.run(
        [sys.executable, "-m", "anneal_memory.cli", "--db", str(db), *args],
        capture_output=True, text=True, env=env, timeout=timeout, cwd=str(root),
    )


def _pull_store(tmp_path, *, with_id=True):
    db = tmp_path / "mem.db"
    with Store(db, audit=False) as store:
        sid = store.store_id
    assert sid
    if not with_id:
        import sqlite3

        conn = sqlite3.connect(str(db))
        conn.execute("DELETE FROM metadata WHERE key = 'store_id'")
        conn.commit()
        conn.close()
        sid = None
    CrystalStore(tmp_path / "mem.crystal.json").crystallize(
        name="derive_dont_invent", level=3, explanation="read the live surface first")
    return db, sid


def _log_lines(db):
    path = outcome_log_path(db)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_crystal_get_found_pull_writes_one_followed_record(tmp_path):
    db, sid = _pull_store(tmp_path)
    got = _pull_cli(db, "crystal", "get", "derive_dont_invent")
    assert got.returncode == 0, got.stderr
    assert "derive_dont_invent" in got.stdout and got.stderr == ""
    (rec,) = _log_lines(db)
    assert rec["exposure_id"].startswith("pull:") and len(rec["exposure_id"]) == len("pull:") + 32
    assert rec["items"] == [{"kind": "crystal", "ref": "derive_dont_invent", "followed": "followed"}]
    assert rec["outcome"] is None and rec["store"] == sid and rec["pull"] is True
    assert "exposed" not in rec
    # --json prints the item alone on stdout and records too
    again = _pull_cli(db, "crystal", "get", "derive_dont_invent", "--json")
    assert again.returncode == 0 and json.loads(again.stdout)["name"] == "derive_dont_invent"
    assert len(_log_lines(db)) == 2


def test_crystal_get_without_an_episodic_db_writes_nothing_and_exits_zero(tmp_path):
    db = tmp_path / "mem.db"  # never created: a crystal-only deployment
    CrystalStore(tmp_path / "mem.crystal.json").crystallize(
        name="derive_dont_invent", level=3, explanation="x")
    got = _pull_cli(db, "crystal", "get", "derive_dont_invent")
    assert got.returncode == 0 and "derive_dont_invent" in got.stdout
    assert got.stderr == ""
    assert not outcome_log_path(db).exists() and not db.exists()
    # control: the same command beside a real store does record (a guard that
    # also holds on a build that never records would prove nothing)
    with Store(db, audit=False):
        pass
    assert _pull_cli(db, "crystal", "get", "derive_dont_invent").returncode == 0
    assert len(_log_lines(db)) == 1


def test_crystal_get_unwritable_log_says_so_on_stderr_and_still_prints(tmp_path):
    import os

    db, _ = _pull_store(tmp_path)
    # a directory where the log file belongs: unwritable for every uid
    outcome_log_path(db).mkdir()
    got = _pull_cli(db, "crystal", "get", "derive_dont_invent")
    assert got.returncode == 0, got.stderr
    assert "derive_dont_invent" in got.stdout
    assert got.stderr.count("\n") == 1 and "pull not recorded" in got.stderr
    assert os.path.isdir(outcome_log_path(db))


def test_crystal_get_no_record_flag_writes_nothing(tmp_path):
    db, _ = _pull_store(tmp_path)
    got = _pull_cli(db, "crystal", "get", "derive_dont_invent", "--no-record")
    assert got.returncode == 0 and "derive_dont_invent" in got.stdout and got.stderr == ""
    assert _log_lines(db) == []


def test_crystal_get_not_found_writes_nothing_and_keeps_its_exit(tmp_path):
    db, _ = _pull_store(tmp_path)
    got = _pull_cli(db, "crystal", "get", "no_such_pattern")
    assert got.returncode == 1 and "not found" in got.stderr
    assert _log_lines(db) == []
    # control: a found name in the same store does record
    assert _pull_cli(db, "crystal", "get", "derive_dont_invent").returncode == 0
    assert len(_log_lines(db)) == 1


def test_crystal_get_on_a_store_with_no_id_writes_nothing_and_never_mints_one(tmp_path):
    import sqlite3

    db, _ = _pull_store(tmp_path, with_id=False)
    got = _pull_cli(db, "crystal", "get", "derive_dont_invent")
    assert got.returncode == 0 and "derive_dont_invent" in got.stdout
    assert "no store id" in got.stderr and got.stderr.count("\n") == 1
    assert _log_lines(db) == []
    conn = sqlite3.connect(str(db))
    try:
        assert conn.execute(
            "SELECT 1 FROM metadata WHERE key = 'store_id'").fetchone() is None
    finally:
        conn.close()


def test_crystal_get_a_held_log_lock_is_bounded_not_a_hang(tmp_path):
    fcntl = pytest.importorskip("fcntl")
    import subprocess
    import sys
    import time

    db, _ = _pull_store(tmp_path)
    log = outcome_log_path(db)
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import fcntl, os, sys, time\n"
         "fd = os.open(sys.argv[1], os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)\n"
         "fcntl.flock(fd, fcntl.LOCK_EX)\n"
         "print('held', flush=True)\n"
         "time.sleep(60)\n", str(log)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        started = time.monotonic()
        got = _pull_cli(db, "crystal", "get", "derive_dont_invent", timeout=20)
        elapsed = time.monotonic() - started
    finally:
        holder.kill()
        holder.wait()
    assert elapsed < 8, elapsed
    assert got.returncode == 0, got.stderr
    assert "derive_dont_invent" in got.stdout
    assert got.stderr == "crystal get: pull not recorded (outcome log busy)\n"
    assert log.read_bytes() == b""


def test_outcome_log_record_lock_timeout_raises_busy_and_default_is_unchanged(tmp_path):
    fcntl = pytest.importorskip("fcntl")
    import os

    from anneal_memory.worth import OutcomeLogBusy

    log = OutcomeLog(tmp_path / "mem.outcomes.jsonl")
    held = os.open(log.path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(OutcomeLogBusy):
            log.record("a", [ExposureLabel("crystal", "p", "followed")], lock_timeout=0.1)
        assert log.path.read_bytes() == b""
    finally:
        os.close(held)  # releases the lock
    log.record("a", [ExposureLabel("crystal", "p", "followed")], lock_timeout=0.1)
    log.record("b", [ExposureLabel("crystal", "p", "followed")])  # blocking default
    assert sorted(log.latest()[0]) == ["a", "b"]


def test_crystal_get_a_closed_stdout_is_not_a_received_pull(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    db, _ = _pull_store(tmp_path)
    root = Path(__file__).resolve().parent.parent
    r, w = os.pipe()
    os.close(r)  # nobody reads: the flush fails
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "anneal_memory.cli", "--db", str(db),
             "crystal", "get", "derive_dont_invent"],
            stdout=w, stderr=subprocess.PIPE, text=True, timeout=60, cwd=str(root),
            env=dict(os.environ, PYTHONPATH=str(root)),
        )
    finally:
        os.close(w)
    assert proc.returncode != 0
    assert _log_lines(db) == []


def test_crystal_get_a_retired_pattern_is_printed_but_not_recorded(tmp_path):
    db, _ = _pull_store(tmp_path)
    assert _pull_cli(db, "crystal", "get", "derive_dont_invent").returncode == 0
    assert len(_log_lines(db)) == 1  # control: live records
    CrystalStore(tmp_path / "mem.crystal.json").retire("derive_dont_invent", kind="obsolete")
    got = _pull_cli(db, "crystal", "get", "derive_dont_invent")
    assert got.returncode == 0 and "retired" in got.stdout and got.stderr == ""
    assert len(_log_lines(db)) == 1


def test_crystal_get_refusal_text_is_one_line_without_the_error_prefix(tmp_path):
    import sqlite3

    db = tmp_path / "mem.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE unrelated (x)")
    conn.commit()
    conn.close()
    CrystalStore(tmp_path / "mem.crystal.json").crystallize(
        name="derive_dont_invent", level=3, explanation="x")
    got = _pull_cli(db, "crystal", "get", "derive_dont_invent")
    assert got.returncode == 0 and "derive_dont_invent" in got.stdout
    assert got.stderr.count("\n") == 1
    assert got.stderr.startswith("crystal get: pull not recorded (cannot read the store id")
    assert "Error:" not in got.stderr
    assert _log_lines(db) == []


def test_a_pull_moves_pulled_only_and_never_a_judged_cell_or_an_episode_row(tmp_path):
    crystal = CrystalStore(tmp_path / "mem.crystal.json")
    crystal.crystallize(name="p", level=3, explanation="x", evidence=["e1", "e2"])
    log = OutcomeLog(tmp_path / "mem.outcomes.jsonl")
    log.record("judged", [ExposureLabel("crystal", "p", "followed")], outcome="success")
    before = compute_worth(log, crystal)
    log.record("pull:" + "a" * 32, [ExposureLabel("crystal", "p", "followed")], pull=True)
    log.record("pull:" + "b" * 32, [ExposureLabel("crystal", "gone", "followed")], pull=True)
    after = compute_worth(log, crystal)

    rows = {r.ref: r for r in after.crystals}
    was = {r.ref: r for r in before.crystals}
    assert rows["p"].pulled == 1 and was["p"].pulled == 0
    for field_name in ("followed", "ignored", "not_applicable", "success", "failure",
                       "unlabelled_success", "unlabelled_failure", "unlabelled_unknown"):
        assert getattr(rows["p"], field_name) == getattr(was["p"], field_name), field_name
    assert rows["p"].table == was["p"].table
    assert rows["p"].table["followed"]["unknown"] == 0
    # a pull of a name that is not live still shows as pulled, in no judged cell
    assert rows["gone"].pulled == 1 and rows["gone"].live is False
    assert rows["gone"].followed == 0 and rows["gone"].table["followed"]["unknown"] == 0
    assert [(r.ref, r.success, r.credited_success) for r in after.episodes] == \
        [(r.ref, r.success, r.credited_success) for r in before.episodes]
    assert rows["p"].as_dict()["pulled"] == 1


def test_a_pull_alone_creates_no_episode_row(tmp_path):
    crystal = CrystalStore(tmp_path / "mem.crystal.json")
    crystal.crystallize(name="p", level=3, explanation="x", evidence=["e1", "e2"])
    log = OutcomeLog(tmp_path / "mem.outcomes.jsonl")
    log.record("pull:" + "a" * 32, [ExposureLabel("crystal", "p", "followed")], pull=True)
    report = compute_worth(log, crystal)
    assert report.episodes == []
    assert {r.ref: r.pulled for r in report.crystals}["p"] == 1


def test_an_ordinary_record_named_pull_keeps_all_its_counts(tmp_path):
    crystal = CrystalStore(tmp_path / "mem.crystal.json")
    crystal.crystallize(name="p", level=3, explanation="x", evidence=["e1"])
    log = OutcomeLog(tmp_path / "mem.outcomes.jsonl")
    log.record("pull:manual", [ExposureLabel("crystal", "p", "followed")], outcome="success")
    log.record("pull:both", [ExposureLabel("crystal", "p", "followed")], pull=True)
    log.record("pull:both", [], outcome="failure")  # a later judgement makes it ordinary
    rows = {r.ref: r for r in compute_worth(log, crystal).crystals}
    p = rows["p"]
    assert p.pulled == 0
    assert p.followed == 2 and p.ignored == 0
    assert p.success == 1 and p.failure == 1
    assert p.table["followed"]["success"] == 1
    erows = {r.ref: r for r in compute_worth(log, crystal).episodes}
    assert erows["e1"].success == 1 and erows["e1"].failure == 1


def test_the_pull_field_is_ignored_by_a_reader_that_does_not_know_it(tmp_path):
    log = OutcomeLog(tmp_path / "mem.outcomes.jsonl")
    log.record("pull:x", [ExposureLabel("crystal", "p", "followed")], pull=True)
    line = json.loads(log.path.read_text())
    assert line["pull"] is True and line["v"] == 1 and line["items"][0]["followed"] == "followed"
    # the released parser reads only v, exposure_id, outcome, items, exposed and store
    assert set(line) == {"v", "exposure_id", "ts", "outcome", "items", "pull"}


def test_crystal_get_a_locked_db_is_bounded_not_a_stall(tmp_path):
    import sqlite3
    import subprocess
    import sys
    import time

    db, _ = _pull_store(tmp_path)
    # A rollback-journal db: there an exclusive writer blocks readers (a WAL db
    # keeps serving them, so it would not exercise the bound).
    conn = sqlite3.connect(str(db))
    assert conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    conn.close()
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sqlite3, sys, time\n"
         "c = sqlite3.connect(sys.argv[1], isolation_level=None)\n"
         "c.execute('BEGIN EXCLUSIVE')\n"
         "print('held', flush=True)\n"
         "time.sleep(60)\n", str(db)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        started = time.monotonic()
        got = _pull_cli(db, "crystal", "get", "derive_dont_invent", timeout=30)
        elapsed = time.monotonic() - started
    finally:
        holder.kill()
        holder.wait()
    assert elapsed < 4, elapsed
    assert got.returncode == 0, got.stderr
    assert "derive_dont_invent" in got.stdout
    assert got.stderr == "crystal get: pull not recorded (store busy)\n"
    assert _log_lines(db) == []
    sqlite3.connect(str(db)).close()


def test_a_store_replaced_before_the_append_gets_no_pull_from_the_old_one(tmp_path, capsys):
    from argparse import Namespace

    from anneal_memory import cli

    db, sid = _pull_store(tmp_path)
    real = cli._read_store_id_bounded
    reads = []

    def swapped(db_path, timeout):
        reads.append(timeout)
        return real(db_path, timeout) if len(reads) == 1 else ("b" * 32, None)  # store B now

    cli._read_store_id_bounded = swapped
    try:
        cli._record_pull_label(Namespace(db=str(db)), "derive_dont_invent")
    finally:
        cli._read_store_id_bounded = real
    assert len(reads) == 2
    assert _log_lines(db) == []
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "pull not recorded" in err and "changed" in err


def test_crystal_get_diagnostics_never_reach_stdout_or_change_the_exit(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    pytest.importorskip("fcntl")
    db, _ = _pull_store(tmp_path, with_id=False)  # a store with no id says so on stderr
    root = Path(__file__).resolve().parent.parent
    base = dict(cwd=str(root), env=dict(os.environ, PYTHONPATH=str(root)),
                capture_output=False, timeout=60)
    cmd = [sys.executable, "-m", "anneal_memory.cli", "--db", str(db),
           "crystal", "get", "derive_dont_invent"]
    # stderr closed at startup (sys.stderr is None): print(file=None) would use stdout
    closed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, preexec_fn=lambda: os.close(2),
                            **{k: v for k, v in base.items() if k != "capture_output"})
    assert closed.returncode == 0
    assert "derive_dont_invent" in closed.stdout and "not recorded" not in closed.stdout
    # stderr a pipe nobody reads: the write fails and the exit is still 0
    r, w = os.pipe()
    os.close(r)
    try:
        broken = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=w, text=True,
                                **{k: v for k, v in base.items() if k != "capture_output"})
    finally:
        os.close(w)
    assert broken.returncode == 0
    assert "derive_dont_invent" in broken.stdout and "not recorded" not in broken.stdout


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, -0.5, True, "2", object()])
def test_record_lock_timeout_is_validated_before_anything_is_touched(tmp_path, bad):
    log = OutcomeLog(tmp_path / "sub" / "mem.outcomes.jsonl")
    with pytest.raises(ValueError, match="lock_timeout"):
        log.record("a", [ExposureLabel("crystal", "p", "followed")], lock_timeout=bad)
    assert not (tmp_path / "sub").exists()


def test_worthrow_positional_construction_keeps_its_old_arity():
    from anneal_memory.worth import WorthRow

    tbl = {k: {"success": 0, "failure": 0, "unknown": 0} for k in FOLLOWED_VALUES}
    row = WorthRow("crystal", "p", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, tbl, 12, "s", "a", False)
    assert (row.exposed_unrecorded, row.surfaced_count, row.last_surfaced_on) == (11, 12, "s")
    assert row.last_activated_on == "a" and row.live is False and row.pulled == 0


def test_any_failure_of_the_label_path_is_one_stderr_line_and_exit_zero(tmp_path, capsys):
    from anneal_memory import cli

    db, _ = _pull_store(tmp_path)
    args = cli.build_parser().parse_args(
        ["--db", str(db), "crystal", "get", "derive_dont_invent"])

    def boom(*a, **kw):
        raise RuntimeError("symlink loop")

    real = cli._read_store_id_bounded
    cli._read_store_id_bounded = boom
    try:
        args.func(args)  # a RuntimeError escaping here would be exit 1 after the print
    finally:
        cli._read_store_id_bounded = real
    out, err = capsys.readouterr()
    assert "derive_dont_invent" in out
    assert err == "crystal get: pull not recorded (symlink loop)\n"
    assert _log_lines(db) == []


@pytest.mark.parametrize("bad", ["false", "true", 1, 0, None, "yes"])
def test_record_pull_must_be_a_real_bool(tmp_path, bad):
    log = OutcomeLog(tmp_path / "sub" / "mem.outcomes.jsonl")
    with pytest.raises(ValueError, match="pull must be a bool"):
        log.record("a", [ExposureLabel("crystal", "p", "followed")], pull=bad)
    assert not (tmp_path / "sub").exists()


@pytest.mark.parametrize("raw", ['"true"', "1", '"false"', "null", "[true]", "0"])
def test_only_json_true_marks_a_pull_so_any_other_value_keeps_its_counts(tmp_path, raw):
    crystal = CrystalStore(tmp_path / "mem.crystal.json")
    crystal.crystallize(name="p", level=3, explanation="x")
    log = OutcomeLog(tmp_path / "mem.outcomes.jsonl")
    log.path.write_text(
        '{"v": 1, "exposure_id": "pull:h", "ts": "2026-10-03T00:00:00Z", "outcome": "success", '
        '"items": [{"kind": "crystal", "ref": "p", "followed": "followed"}], '
        '"pull": ' + raw + "}\n")
    row = {r.ref: r for r in compute_worth(log, crystal).crystals}["p"]
    assert row.pulled == 0 and row.followed == 1 and row.success == 1


def test_the_pull_note_is_one_physical_line_whatever_the_reason(capsys):
    from anneal_memory import cli

    cli._pull_note("disk said:\r\nforged second line\nand a third")
    err = capsys.readouterr().err
    assert err == "crystal get: pull not recorded (disk said: forged second line and a third)\n"
    assert err.count("\n") == 1 and "\r" not in err


@pytest.mark.parametrize("kwargs", [
    {"items": [ExposureLabel("episode", "e", "followed")]},
    {"items": [ExposureLabel("crystal", "p", "ignored")]},
    {"items": [ExposureLabel("crystal", "p", "followed")], "outcome": "failure"},
    {"items": [ExposureLabel("crystal", "p", "followed"), ExposureLabel("crystal", "q", "followed")]},
    {"items": [ExposureLabel("crystal", "p", "followed")], "exposed": [ExposedRef("episode", "e")]},
])
def test_pull_refuses_anything_but_one_followed_crystal(tmp_path, kwargs):
    """c-pull-label L3 r1 (codex + complement MED): a pull record carrying a
    label, an outcome or exposed refs was accepted and compute_worth then
    dropped that content."""
    log = OutcomeLog(tmp_path / "mem.outcomes.jsonl")
    items = kwargs.pop("items")
    with pytest.raises(ValueError, match="pull=True takes exactly one crystal"):
        log.record("x", items, pull=True, **kwargs)
    assert not (tmp_path / "mem.outcomes.jsonl").exists() or log.latest()[0] == {}


def test_a_hand_made_pull_with_an_outcome_is_counted_as_what_it_carries(tmp_path):
    """c-pull-label L3 r1: compute_worth applies the pull rule only to the pull
    shape, so a written outcome is never discarded."""
    import json
    path = tmp_path / "mem.outcomes.jsonl"
    path.write_text(json.dumps({
        "v": 1, "exposure_id": "x", "ts": "2026-10-10T00:00:00Z", "pull": True,
        "outcome": "failure", "items": [{"kind": "crystal", "ref": "p", "followed": "followed"}],
    }) + "\n", encoding="utf-8")
    crystal = CrystalStore(tmp_path / "c.crystal.json")
    rows = {r.ref: r for r in compute_worth(OutcomeLog(path), crystal).crystals}
    assert rows["p"].pulled == 0
    assert rows["p"].failure == 1


_NO_FIFO = not hasattr(__import__("os"), "mkfifo")


@pytest.mark.skipif(_NO_FIFO, reason="no FIFOs on this platform")
@pytest.mark.parametrize("act", ["read", "record", "record_if_missing", "adopt"])
def test_a_fifo_at_the_outcome_log_refuses_instead_of_hanging(tmp_path, act):
    """c-pull-label L3 r1 (codex HIGH), reproduced on main 0655b72: `worth` on a
    FIFO outcome log hung until killed. Every open checks the descriptor."""
    import os
    path = tmp_path / "mem.outcomes.jsonl"
    os.mkfifo(path)
    log = OutcomeLog(path, store_id="s1", bind=True)
    calls = {
        "read": lambda: log.binding(),
        "record": lambda: log.record("e1", [], outcome="success"),
        "record_if_missing": lambda: log.record_if_missing("e1", [], outcome="success"),
        "adopt": lambda: log.adopt_unbound(),
    }
    with pytest.raises(OSError, match="not a regular file"):
        calls[act]()


@pytest.mark.skipif(not hasattr(__import__("os"), "O_NOFOLLOW"), reason="no O_NOFOLLOW")
@pytest.mark.parametrize("act", ["record", "record_if_missing", "adopt"])
def test_a_write_never_follows_a_symlinked_outcome_log(tmp_path, act):
    """c-pull-label L3 r1 (codex HIGH), reproduced on main 0655b72: `outcome`
    appended through a symlink into the file it named (a 7-byte canary grew to
    147 bytes)."""
    canary = tmp_path / "canary.txt"
    canary.write_bytes(b"CANARY\n")
    path = tmp_path / "mem.outcomes.jsonl"
    path.symlink_to(canary)
    log = OutcomeLog(path, store_id="s1", bind=True)
    calls = {
        "record": lambda: log.record("e1", [], outcome="success"),
        "record_if_missing": lambda: log.record_if_missing("e1", [], outcome="success"),
        "adopt": lambda: log.adopt_unbound(),
    }
    with pytest.raises(OSError, match="symlink"):
        calls[act]()
    assert canary.read_bytes() == b"CANARY\n"


def test_a_symlinked_outcome_log_is_still_read(tmp_path):
    """Reading follows a symlink: it writes nothing, and a log kept elsewhere
    still reports."""
    real = tmp_path / "real.outcomes.jsonl"
    OutcomeLog(real).record("e1", [], outcome="success")
    link = tmp_path / "mem.outcomes.jsonl"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("cannot create a symlink here")
    assert "e1" in OutcomeLog(link).latest()[0]



@pytest.mark.skipif(_NO_FIFO, reason="no FIFOs on this platform")
def test_a_fifo_receipt_file_refuses_instead_of_hanging(tmp_path):
    """outcomes-open L3 r2 (codex HIGH): a receipt read under the crystal
    store's lock blocked every crystal write when the receipt was a FIFO."""
    import os
    from anneal_memory.worth import load_receipts
    fifo = tmp_path / "receipts.jsonl"
    os.mkfifo(fifo)
    with pytest.raises(OSError, match="receipt file is not a regular file"):
        load_receipts([fifo])


@pytest.mark.skipif(_NO_FIFO, reason="no FIFOs on this platform")
def test_fold_surfaced_refuses_a_fifo_receipt_before_anything_else(tmp_path):
    """outcomes-open L3 r3 (codex + complement LOW): the preflight called a FIFO
    "missing", so a lone FIFO read as "none exists" and a mixed list could
    return early without the FIFO ever being checked."""
    import os
    from anneal_memory.worth import fold_surfaced
    good = tmp_path / "good.jsonl"
    good.write_text("", encoding="utf-8")
    fifo = tmp_path / "r.jsonl"
    os.mkfifo(fifo)
    store = CrystalStore(tmp_path / "c.crystal.json")
    for paths in ([fifo], [good, fifo]):
        with pytest.raises(OSError, match="receipt file is not a regular file"):
            fold_surfaced(store, paths)


def test_fold_surfaced_never_moves_the_mark_past_receipts_it_did_not_read(tmp_path, monkeypatch):
    """outcomes-open L3 r4 (codex MED): a receipt gone between the preflight and
    the locked read was skipped while the mark moved past it."""
    import anneal_memory.worth as worth
    rec = tmp_path / "r.jsonl"
    rec.write_text("", encoding="utf-8")
    store = CrystalStore(tmp_path / "c.crystal.json")
    store.crystallize(name="p", level=3, explanation="x", evidence=["e1"])
    real = worth._read_regular

    def vanished(path, *, what):
        rec.unlink(missing_ok=True)
        return real(path, what=what)

    monkeypatch.setattr(worth, "_read_regular", vanished)
    before = store.path.read_bytes()
    with pytest.raises(FileNotFoundError, match="disappeared during the fold"):
        worth.fold_surfaced(store, [rec])
    assert store.path.read_bytes() == before
    # L3 r5 (codex MED): one source vanishing while another reads fine also
    # refuses; the mark never moves past the vanished one.
    rec.write_text("", encoding="utf-8")
    other = tmp_path / "other.jsonl"
    other.write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="disappeared during the fold"):
        worth.fold_surfaced(store, [other, rec])
    assert store.path.read_bytes() == before


def test_fold_surfaced_calls_a_dangling_symlink_missing(tmp_path):
    """outcomes-open L3 r4 (complement): the preflight called a dangling symlink
    "not a regular file"; one stat makes it missing, as the read would."""
    from anneal_memory.worth import fold_surfaced
    good = tmp_path / "good.jsonl"
    good.write_text("", encoding="utf-8")
    dangling = tmp_path / "gone.jsonl"
    try:
        dangling.symlink_to(tmp_path / "nowhere.jsonl")
    except OSError:
        pytest.skip("cannot create a symlink here")
    store = CrystalStore(tmp_path / "c.crystal.json")
    store.crystallize(name="p", level=3, explanation="x", evidence=["e1"])
    result = fold_surfaced(store, [good, dangling])
    assert str(dangling) in result.paths_missing


def test_fold_surfaced_tracks_the_preflight_per_entry_not_per_name(tmp_path, monkeypatch):
    """outcomes-open L3 r6 (codex MED): with [rec, rec], rec missing at the first
    stat, present at the second, gone at the read, both entries matched "was
    missing" by name and the mark moved having read nothing."""
    import os
    import anneal_memory.worth as worth
    rec = tmp_path / "r.jsonl"
    store = CrystalStore(tmp_path / "c.crystal.json")
    store.crystallize(name="p", level=3, explanation="x", evidence=["e1"])
    real_stat = worth.os.stat
    calls = []

    def flicker(p, *a, **k):
        if str(p) == str(rec):
            calls.append(p)
            if len(calls) == 2:
                rec.write_text("", encoding="utf-8")
                st = real_stat(p, *a, **k)
                rec.unlink()
                return st
            raise FileNotFoundError(2, "No such file", str(p))
        return real_stat(p, *a, **k)

    monkeypatch.setattr(worth.os, "stat", flicker)
    before = store.path.read_bytes()
    # r8: a repeated path is one entry, so the flicker between two stats of one
    # name can no longer happen: one stat, an honest "none exists", no mark move.
    with pytest.raises(FileNotFoundError, match="none of the receipt paths exists"):
        worth.fold_surfaced(store, [rec, rec])
    assert len(calls) == 1
    assert store.path.read_bytes() == before


def test_fold_surfaced_reads_a_repeated_path_once(tmp_path):
    """outcomes-open L3 r8 (codex LOW): duplicate path arguments were the race
    surface of r6-r8; a path named twice is one entry."""
    import json
    from anneal_memory.worth import fold_surfaced
    rec = tmp_path / "r.jsonl"
    rec.write_text(json.dumps({"event_id": "e", "ts": "2026-10-01T00:00:00Z",
                               "exposed": [{"pattern": "p"}]}) + "\n", encoding="utf-8")
    store = CrystalStore(tmp_path / "c.crystal.json")
    store.crystallize(name="p", level=3, explanation="x", evidence=["e1"])
    import os
    entry = next(e for e in os.scandir(tmp_path) if e.name == "r.jsonl")
    # L3 r9 (codex): a path-like (DirEntry) dedupes by its path, not its repr.
    result = fold_surfaced(store, [rec, str(rec), entry, rec],
                           now=datetime(2026, 10, 2, tzinfo=timezone.utc))
    assert result.receipts_folded == 1
    assert result.duplicates_skipped == 0  # read once: no event seen twice
    assert result.paths_missing == []



def test_crystal_get_records_a_pull_for_a_store_path_with_uri_characters(tmp_path):
    """c-pull-label: the store id is read through store.connect's escaped URI."""
    sub = tmp_path / "a #?% b"
    sub.mkdir()
    db, sid = _pull_store(sub)
    got = _pull_cli(db, "crystal", "get", "derive_dont_invent")
    assert got.returncode == 0 and got.stderr == "", got.stderr
    (rec,) = _log_lines(db)
    assert rec["store"] == sid
