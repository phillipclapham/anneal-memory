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
        assert st.supersession_exists(old.id, new.id)
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
        assert ro.supersession_exists("a", "b") is False
