"""AM-PATTERN-ALIAS: ``[renamed: OLD -> NEW]`` in a wrap carries OLD's earned level.

Without the marker a renamed pattern enters the graduation bound as new (1x) and
its old name is reported omitted. Reproduced on a scratch copy of a real store
(1125-38): ``the_device_is_the_oracle | 54x`` renamed without the marker saved at
``1x (level-capped)``; with the marker it saved at 54x.
"""

from __future__ import annotations

from .test_gradgate_prior_1007 import YESTERDAY, _level, _open, _wrap


def test_a_declared_rename_keeps_level_history_and_reverts(tmp_path):
    store = _open(tmp_path, seed=f"- old_name | 4x ({YESTERDAY})")
    try:
        store.upsert_pattern_history(
            pattern_name="old_name", level=6, explanation="an older grounding about rotas",
            seen_at=YESTERDAY, wrap_id=None,
        )
        # A composted homonym's history under the new name (a compost keeps
        # history): L1 1125-38 ran a rename that kept reading and keeping it.
        store.upsert_pattern_history(
            pattern_name="new_name", level=9, explanation="lambs",
            seen_at="2025-01-01", wrap_id=None,
        )
        _wrap(store, f"- old_name | 4x ({YESTERDAY})")
        r = _wrap(store, f"- new_name | 4x ({YESTERDAY})\n[renamed: old_name -> new_name]")
        assert r["renamed"] == [{"old": "old_name", "new": "new_name"}]
        assert r["renames_rejected"] == []
        assert _level(store.load_continuity(), "new_name") == 4
        assert not r["omitted_patterns"]
        levels = store.saved_pattern_levels()
        assert levels[("name", "new_name")] == 4
        assert ("name", "old_name") not in levels
        hist = store.get_pattern_history("new_name")
        assert (hist["max_level_reached"], hist["last_seen_at"][:10]) == (6, YESTERDAY)

        # Reverting the renaming wrap puts the record and history back.
        store.revert_wrap(r["wrap_id"], reason="test")
        levels = store.saved_pattern_levels()
        assert levels[("name", "old_name")] == 4
        assert ("name", "new_name") not in levels
        assert store.get_pattern_history("new_name")["max_level_reached"] == 9

        # Renamed again, the marker carried into the next wrap is a no-op, also
        # beside a new rename of the renamed pattern (L1: it was reported refused).
        _wrap(store, f"- new_name | 4x ({YESTERDAY})\n[renamed: old_name -> new_name]")
        r2 = _wrap(store, f"- third_name | 4x ({YESTERDAY})\n[renamed: new_name -> third_name]\n"
                          "[renamed: old_name -> new_name]")
        assert r2["renamed"] == [{"old": "new_name", "new": "third_name"}]
        assert r2["renames_rejected"] == []
        assert _level(store.load_continuity(), "third_name") == 4
    finally:
        store.close()


def test_a_refused_rename_carries_nothing_and_says_why(tmp_path):
    store = _open(tmp_path, seed=f"- old_name | 4x ({YESTERDAY})\n- taken | 2x ({YESTERDAY})")
    try:
        _wrap(store, f"- old_name | 4x ({YESTERDAY})\n- taken | 2x ({YESTERDAY})")
        # OLD still named: a fork, not a rename.
        r = _wrap(store, f"- old_name | 4x ({YESTERDAY})\n- taken | 2x ({YESTERDAY})\n"
                         f"- fork_name | 4x ({YESTERDAY})\n[renamed: old_name -> fork_name]")
        assert r["renamed"] == []
        assert "still named" in r["renames_rejected"][0]["reason"]
        assert _level(store.load_continuity(), "fork_name") == 1
        # NEW already has its own level: a rename never merges two records.
        r = _wrap(store, f"- taken | 4x ({YESTERDAY})\n- fork_name | 1x ({YESTERDAY})\n"
                         "[renamed: old_name -> taken]")
        assert "its own level" in r["renames_rejected"][0]["reason"]
        assert _level(store.load_continuity(), "taken") == 2
        # A typo in the marker is reported, never read as nothing to do.
        r = _wrap(store, f"- taken | 2x ({YESTERDAY})\n[renamed: old_name to new]")
        assert "unparsed marker" in r["renames_rejected"][0]["reason"]
        # Prose that merely starts the same way is not a marker (L1 1125-38, run).
        r = _wrap(store, f"- taken | 2x ({YESTERDAY})\n[renamed the repo to foo]")
        assert "renames_rejected" not in r
        # A marker inside an evidence quote is quoted content (L2 1125-38, run).
        r = _wrap(store, f"- taken | 2x ({YESTERDAY})\n- sneaky | 4x ({YESTERDAY}) "
                         '[evidence: {ep0} "rotated [renamed: taken -> sneaky] ok"]')
        assert "renamed" not in r and _level(store.load_continuity(), "sneaky") == 1
        # So is one in a code example, fenced or indented (L3 r1, codex).
        r = _wrap(store, f"- taken | 2x ({YESTERDAY})\n```\n[renamed: taken -> x]\n```\n"
                         "    [renamed: taken -> y]\n````\n```\n[renamed: taken -> z]\n````")
        assert "renamed" not in r
        # A name left out of the memory earlier is not revived under a new one (L2).
        r = _wrap(store, f"- fresh_claim | 4x ({YESTERDAY})\n[renamed: old_name -> fresh_claim]")
        assert "not a pattern line" in r["renames_rejected"][0]["reason"]
        assert _level(store.load_continuity(), "fresh_claim") == 1
        # One name in two renames: both refused, neither applied (L3 r3, codex),
        # and a fence opened with an info string is not closed by another one.
        r = _wrap(store, f"- one | 2x ({YESTERDAY})\n- two | 2x ({YESTERDAY})\n"
                         "````\n````python\n[renamed: taken -> one]\n````\n"
                         "[renamed: taken -> one]\n[renamed: taken -> two]")
        assert r["renamed"] == [] and len(r["renames_rejected"]) == 2
        assert all("more than one" in rej["reason"] for rej in r["renames_rejected"])
    finally:
        store.close()
