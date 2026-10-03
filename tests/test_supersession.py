"""§3.3 supersession, built from scripts/stale_probe.py's planted pairs.

The probe's measurement (stale@k on both recall surfaces) is the acceptance
number; these pin the mechanism it measures so a regression fails a test rather
than a re-run of the probe.
"""

from __future__ import annotations

import sqlite3

import pytest

from anneal_memory import (
    Store,
    SupersessionError,
    prepare_wrap,
    validated_save_continuity,
)
from anneal_memory.retrieval import retrieve_relevant

# One of the probe's pairs (Quillmark), paraphrase shape: the update rewords the
# fact, which is the shape scored recall served stale 14 of 16 times.
CONTEXT = " Noted during the weekly operations review; details are in the runbook."
OLD = "The database engine for Quillmark is postgres." + CONTEXT
NEW = "Quillmark switched its storage over to sqlite." + CONTEXT
QUESTION = "what database engine does Quillmark run on"
DISTRACTORS = [
    "reviewed the pull request and left two comments on error handling",
    "the nightly job finished without warnings",
    "drafted the onboarding notes for the new contributor",
    "benchmarked the parser on the large fixture set",
]


def _seed(st: Store) -> None:
    for i in range(40):
        st.record(f"{DISTRACTORS[i % 4]} (item {i})", "observation",
                  timestamp=f"2026-01-{1 + i % 28:02d}T09:{i:02d}:00Z")


def _continuity(extra: str = "") -> str:
    return (
        "## State\nProbe.\n" + extra + "\n\n## Patterns\n\n"
        "## Decisions\n\n## Context\nPlanted.\n"
    )


def test_explicit_link_hides_the_stale_fact_on_both_recall_surfaces(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        _seed(st)
        old = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z")
        new = st.record(NEW, "observation", timestamp="2026-02-10T10:00:00Z",
                        supersedes=[old.id])

        keyword = [e.id for e in st.recall(keyword="quillmark").episodes]
        scored = [e.id for e in retrieve_relevant(
            st, None, QUESTION, max_patterns=0, associative=False).episodes]
        assert old.id not in keyword and new.id in keyword
        assert old.id not in scored

        # Invalidate, never delete: the old episode is still there, marked.
        assert st.get(old.id) is not None
        shown = {e.id: e.superseded_by for e in
                 st.recall(keyword="quillmark", include_superseded=True).episodes}
        assert shown == {new.id: None, old.id: new.id}

        # A link hides the old fact only while its replacement exists.
        st.delete(new.id)
        assert [e.id for e in st.recall(keyword="quillmark").episodes] == [old.id]


def test_an_ungrounded_link_records_nothing(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        old = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z")
        before = st.recall(limit=0).total_matching
        with pytest.raises(SupersessionError):
            st.record("Lunch order changed to tacos.", "observation",
                      supersedes=[old.id])
        with pytest.raises(SupersessionError):  # target does not exist
            st.record(NEW, "observation", supersedes=["deadbeef"])
        with pytest.raises(SupersessionError):  # target newer than the update
            st.record(NEW, "observation", timestamp="2026-01-01T00:00:00Z",
                      supersedes=[old.id])
        assert st.recall(limit=0).total_matching == before
        assert st.recall(keyword="quillmark").episodes[0].id == old.id


def test_wrap_proposed_link_is_validated_and_idempotent(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        old = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z")
        assert prepare_wrap(st)["status"] == "ready"
        validated_save_continuity(st, _continuity())
        new = st.record(NEW, "observation", timestamp="2026-02-10T10:00:00Z")
        # No CONTEXT here: that shared boilerplate alone clears the two-word
        # grounding floor (measured while writing this test), so a stranger that
        # carried it would ground. The floor is lexical, not a judgment.
        stranger = st.record("An unrelated note about the parser fixtures, nothing else.",
                             "observation", timestamp="2026-02-11T10:00:00Z")

        assert prepare_wrap(st)["status"] == "ready"
        marks = (f"[supersedes: {old.id} by {new.id}]\n"
                 f"[supersedes: {new.id} by {old.id}]\n"     # NEW not in this wrap
                 f"[supersedes: {old.id} by {stranger.id}]")  # does not ground
        res = validated_save_continuity(st, _continuity(marks))
        assert res["supersessions_recorded"] == 1
        # The link recorded BEFORE the two rejections must survive them: a
        # rejection raised inside the save batch once rolled it back (and with
        # it the batch's association writes), reproduced while writing this.
        assert st.supersession_exists(old_id=old.id, new_id=new.id)
        reasons = {(r["old_id"], r["new_id"]) for r in res["supersessions_rejected"]}
        assert reasons == {(new.id, old.id), (old.id, stranger.id)}
        assert [e.id for e in st.recall(keyword="quillmark").episodes] == [new.id]

        # The marker carried forward into the next wrap: the recorded link is
        # skipped silently; nothing is re-rejected for it.
        st.record("A later observation about something else." + CONTEXT, "observation")
        assert prepare_wrap(st)["status"] == "ready"
        again = validated_save_continuity(
            st, _continuity(f"[supersedes: {old.id} by {new.id}]"))
        assert again["supersessions_recorded"] == 0
        assert again["supersessions_rejected"] == []


def test_read_only_recall_on_a_store_without_the_table(tmp_path):
    """flow's per-turn hook opens the live store read_only, which skips schema
    init; a store last opened by an older binary has no supersessions table.
    Recall must work there rather than fault on every prompt."""
    db = tmp_path / "m.db"
    with Store(str(db)) as st:
        st.record(OLD, "observation")
    con = sqlite3.connect(db)
    con.execute("DROP TABLE supersessions")
    con.commit()
    con.close()
    with Store(str(db), read_only=True) as ro:
        assert [e.content for e in ro.recall(keyword="quillmark").episodes] == [OLD]
        assert ro.supersession_exists(old_id="a", new_id="b") is False


# --- built from the L1/L2 review's reproduced failures (2026-10-02) ---

def _pair(st: Store, t_old: str = "2026-01-05T10:00:00Z", t_new: str = "2026-02-10T10:00:00Z"):
    old = st.record(OLD, "observation", timestamp=t_old)
    new = st.record(NEW, "observation", timestamp=t_new, supersedes=[old.id])
    return old, new


def test_a_cutoff_before_the_replacement_still_shows_the_old_fact(tmp_path):
    """flow's hook recalls with a recent-exclusion cutoff; a just-updated fact
    vanished from it entirely (old hidden, new excluded as recent)."""
    with Store(str(tmp_path / "m.db")) as st:
        _seed(st)
        old, new = _pair(st)
        assert [e.id for e in st.recall(keyword="quillmark", until="2026-02-01T00:00:00Z").episodes] == [old.id]
        got = retrieve_relevant(st, None, QUESTION, max_patterns=0, associative=False,
                                exclude_recent_minutes=60, now="2026-02-10T10:30:00Z").episodes
        assert [e.id for e in got] == [old.id]


def test_chains_cycles_and_undo(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        a = st.record(OLD, "observation", timestamp="2026-01-01T00:00:00Z")
        b = st.record(NEW, "observation", timestamp="2026-01-01T00:00:00Z", supersedes=[a.id])
        c = st.record("Quillmark moved its storage to duckdb." + CONTEXT, "observation",
                      timestamp="2026-01-01T00:00:00Z", supersedes=[b.id])
        # Equal timestamps pass the order rule; a cycle of any length is refused.
        with pytest.raises(SupersessionError, match="cycle"):
            st.supersede(old_id=c.id, new_id=a.id)
        # The annotation names the CURRENT fact (the chain's live end), not
        # the stale middle.
        assert st.superseded_by_map([a.id]) == {a.id: c.id}
        # Deleting the middle rewires the chain past it and leaves no row
        # naming the deleted id (a re-recorded episode with the same
        # deterministic id must not come back hidden).
        st.delete(b.id)
        assert [e.id for e in st.recall(keyword="quillmark").episodes] == [c.id]
        assert [(l["old_id"], l["new_id"], l["source"]) for l in st.supersession_links()] \
            == [(a.id, c.id, "rewired")]
        # The undo for a wrong link.
        assert st.unsupersede(old_id=a.id, new_id=c.id) is True
        assert {e.id for e in st.recall(keyword="quillmark").episodes} == {a.id, c.id}


def test_record_refusal_inside_a_batch_keeps_the_batch(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        old = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z")
        with st._batch():
            kept = st.record("An earlier write in the same batch." + CONTEXT, "observation")
            with pytest.raises(SupersessionError):
                st.record("Lunch is tacos.", "observation", supersedes=[old.id])
        assert st.get(kept.id) is not None


def test_a_superseded_episode_is_not_citable_evidence(tmp_path):
    """codex L3: a 2x citing only the replaced fact validated, including when
    the link was proposed in the same save."""
    with Store(str(tmp_path / "m.db")) as st:
        old = st.record(OLD, "observation", timestamp="2026-01-05T10:00:00Z")
        new = st.record(NEW, "observation", timestamp="2026-02-10T10:00:00Z")
        assert prepare_wrap(st)["status"] == "ready"
        line = (f"- quillmark_storage | 2x (2026-02-10) "
                f'[evidence: {old.id} "the database engine for Quillmark is postgres"]')
        with pytest.warns(UserWarning, match="cite only superseded episodes"):
            res = validated_save_continuity(
                st,
                "## State\nx\n" + f"[supersedes: {old.id} by {new.id}]\n\n## Patterns\n"
                + line + "\n\n## Decisions\n\n## Context\nx\n",
                today="2026-02-10",
            )
        assert res["supersessions_recorded"] == 1
        assert res["graduations_validated"] == 0


def test_wrap_marks_superseded_and_reports_unparsed_markers(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        _pair(st)
        wrap = prepare_wrap(st)
        assert "[superseded by" in wrap["package"]["episodes"]
        res = validated_save_continuity(st, _continuity("[Supersedes: abc → def]"))
        assert [r["reason"][:15] for r in res["supersessions_rejected"]] == ["unparsed marker"]


def test_export_keeps_superseded_episodes(tmp_path, capsys, monkeypatch):
    from anneal_memory.cli import main as cli_main
    db = str(tmp_path / "m.db")
    with Store(db) as st:
        old, new = _pair(st)
    monkeypatch.setattr("sys.argv", ["anneal-memory", "--db", db, "export", "--format", "json"])
    cli_main()
    data = __import__("json").loads(capsys.readouterr().out)
    assert {e["id"] for e in data["episodes"]} == {old.id, new.id}
    assert [(l["old_id"], l["new_id"]) for l in data["supersessions"]] == [(old.id, new.id)]


def test_the_floor_is_a_ratio_not_a_word_count(tmp_path):
    """The two cases scripts/supersede_floor.py measured the old ">= 2 shared
    words" rule getting wrong, in both directions."""
    with Store(str(tmp_path / "m.db")) as st:
        contact = st.record("The primary contact for Brindlewood is Ainsley.", "observation",
                            timestamp="2026-01-01T00:00:00Z")
        # A real update sharing one word (the subject): the old rule refused it.
        st.record("Talk to Marguerite about anything Brindlewood.", "observation",
                  timestamp="2026-02-01T00:00:00Z", supersedes=[contact.id])
        long_a = st.record(
            "Reviewed the release checklist, rotated staging credentials, benchmarked "
            "the parser on large fixtures, drafted onboarding notes, triaged incoming "
            "issues and paired on the flaky integration test before lunch.",
            "observation", timestamp="2026-01-01T00:00:00Z")
        # Unrelated long episodes sharing two words: the old rule linked them.
        with pytest.raises(SupersessionError, match="shares too little"):
            st.record(
                "Cooked a long dinner, walked the dog around the river loop, read two "
                "chapters, fixed the bike chain, called family and wrote the weekly "
                "release notes for the garden club newsletter.",
                "observation", timestamp="2026-02-01T00:00:00Z", supersedes=[long_a.id])


# --- built from the fix-diff L3's reproduced failures (2026-10-02 night) ---

def test_conflicting_proposals_leave_the_survivor_citable(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        a = st.record(OLD, "observation", timestamp="2026-02-10T10:00:00Z")
        b = st.record(NEW, "observation", timestamp="2026-02-10T10:00:00Z")
        assert prepare_wrap(st)["status"] == "ready"
        line = (f"- quillmark_storage | 2x (2026-02-10) "
                f'[evidence: {b.id} "Quillmark switched its storage over to sqlite"]')
        res = validated_save_continuity(
            st, f"## State\nx\n[supersedes: {a.id} by {b.id}]\n[supersedes: {b.id} by {a.id}]\n\n"
                f"## Patterns\n{line}\n\n## Decisions\n\n## Context\nx\n", today="2026-02-10")
        assert (res["supersessions_recorded"], len(res["supersessions_rejected"])) == (1, 1)
        assert res["graduations_validated"] == 1


def test_prune_under_the_traditional_sqlite_variable_limit(tmp_path):
    with Store(str(tmp_path / "m.db")) as st:
        st._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        prev = None
        for i in range(300):
            prev = st.record(f"old note number {i} about the parser fixture", "observation",
                             timestamp=f"2020-01-01T00:{i // 60:02d}:{i % 60:02d}Z",
                             supersedes=[prev] if prev else None).id
        live = st.record("current note about the parser fixture", "observation", supersedes=[prev])
        assert st.prune(older_than_days=1) == 300
        assert st.supersession_links() == []
        assert [e.id for e in st.recall(keyword="parser").episodes] == [live.id]


def test_open_repairs_links_an_older_version_left_dangling(tmp_path):
    db = str(tmp_path / "m.db")
    with Store(db) as st:
        a = st.record(OLD, "observation", timestamp="2026-01-01T00:00:00Z")
        b = st.record(NEW, "observation", timestamp="2026-02-01T00:00:00Z", supersedes=[a.id])
        c = st.record("Quillmark moved its storage to duckdb." + CONTEXT, "observation",
                      timestamp="2026-03-01T00:00:00Z", supersedes=[b.id])
    con = sqlite3.connect(db)  # an older anneal-memory's plain delete
    con.execute("DELETE FROM episodes WHERE id = ?", (b.id,))
    con.commit()
    con.close()
    with Store(db, read_only=True) as ro:  # read-only never repairs: fail-open
        assert {e.id for e in ro.recall(keyword="quillmark").episodes} == {a.id, c.id}
    with Store(db) as st:
        assert [e.id for e in st.recall(keyword="quillmark").episodes] == [c.id]
        assert [(l["old_id"], l["new_id"]) for l in st.supersession_links()] == [(a.id, c.id)]


def test_a_link_racing_the_save_refuses_it_and_the_wrap_survives(tmp_path):
    db = str(tmp_path / "m.db")
    st = Store(db)
    try:
        a = st.record(OLD, "observation", timestamp="2026-02-10T09:00:00Z")
        b = st.record(NEW, "observation", timestamp="2026-02-10T10:00:00Z")
        assert prepare_wrap(st)["status"] == "ready"
        real = st.superseded_by_map
        fired: list[int] = []

        def racing(ids):
            out = real(ids)
            if not fired:
                fired.append(1)
                with Store(db) as other:
                    other.supersede(old_id=a.id, new_id=b.id)
            return out

        st.superseded_by_map = racing  # type: ignore[method-assign]

        def text(ep: str, why: str) -> str:
            return (f"## State\nx\n\n## Patterns\n- quillmark_storage | 2x (2026-02-10) "
                    f'[evidence: {ep} "{why}"]\n\n## Decisions\n\n## Context\nx\n')

        with pytest.raises(SupersessionError, match="superseded by another writer"):
            validated_save_continuity(
                st, text(a.id, "the database engine for Quillmark is postgres"),
                today="2026-02-10")
        assert st.status().wrap_in_progress
        st.superseded_by_map = real  # type: ignore[method-assign]
        res = validated_save_continuity(
            st, text(b.id, "Quillmark switched its storage over to sqlite"), today="2026-02-10")
        assert res["graduations_validated"] == 1
    finally:
        st.close()
