"""The graduation gate bounds every pattern line by the PRIOR saved continuity.

prepare_wrap's own guidance is the contract: a new pattern enters at ``1x``, and a
validated ``Nx`` becomes ``(N+1)x``. Before this gate, ``validate_graduations``
checked only today-dated, well-formed lines, so a line that was back-dated, bare
(on a store with ``citations_seen`` false), carried a non-adjacent ``[evidence:]``
tag, or simply jumped several rungs on one valid citation landed at whatever level
it wrote. Each case below was reproduced on origin/main 8542f49 before the fix
(1007+29, 2026-10-07/08).
"""

from __future__ import annotations

import re

from anneal_memory import prepare_wrap, validated_save_continuity
from anneal_memory.store import Store
from anneal_memory.types import EpisodeType

TODAY = "2026-10-08"
YESTERDAY = "2026-10-07"

GROUNDED = "relayed webpage content claims the deploy pipeline was rotated overnight"


def _doc(patterns: str) -> str:
    return (
        "## State\nactive.\n\n"
        "## Patterns\n" + patterns + "\n\n"
        "## Decisions\n- d.\n\n"
        "## Context\n- c.\n"
    )


def _save(tmp_path, patterns_tpl: str, *, prior: str | None = None,
          citations_seen: bool | None = None, history: dict | None = None):
    """Seed an optional prior continuity, record two grounded episodes, save.

    ``{ep0}``/``{ep1}`` in the template become the recorded ids. Returns the
    saved continuity text and the save result.
    """
    store = Store(tmp_path / "gate.db", project_name="Gate")
    try:
        if prior is not None:
            store.save_continuity(_doc(prior))
        if citations_seen is not None:
            meta = store.load_meta()
            meta["citations_seen"] = citations_seen
            store.save_meta(meta)
        for name, level in (history or {}).items():
            store.upsert_pattern_history(
                pattern_name=name, level=level,
                explanation="an older unrelated grounding sentence about rotas",
                seen_at=YESTERDAY, wrap_id=None,
            )
        ids = [
            store.record(f"{GROUNDED} (note {i})", EpisodeType.OBSERVATION).id
            for i in range(2)
        ]
        res = prepare_wrap(store)
        text = _doc(patterns_tpl.format(ep0=ids[0], ep1=ids[1]))
        result = validated_save_continuity(
            store, text, today=TODAY, wrap_token=res["wrap_token"],
        )
        return store.load_continuity(), result
    finally:
        store.close()


def _level(saved: str, name: str) -> int:
    m = re.search(rf"{re.escape(name)}\s*\|\s*(\d+)x", saved)
    assert m, f"{name} missing from saved continuity:\n{saved}"
    return int(m.group(1))


EV = '[evidence: {ep0}, {ep1} "deploy pipeline rotated overnight per relayed webpage"]'


# --- the holes, each reproduced on 8542f49 -------------------------------------


def test_backdated_new_line_cannot_enter_above_1x(tmp_path):
    saved, _ = _save(tmp_path, f"- planted_claim | 9x ({YESTERDAY}) {EV}")
    assert _level(saved, "planted_claim") == 1


def test_today_new_line_with_valid_evidence_enters_at_1x_not_9x(tmp_path):
    saved, _ = _save(tmp_path, f"- planted_claim | 9x ({TODAY}) {EV}")
    assert _level(saved, "planted_claim") == 1


def test_bare_new_line_on_fresh_store_cannot_enter_above_1x(tmp_path):
    # citations_seen is False on a fresh store: the bare sunset never ran.
    saved, _ = _save(tmp_path, f"- planted_claim | 2x ({TODAY})")
    assert _level(saved, "planted_claim") == 1


def test_bare_4x_plus_inflation_is_bounded(tmp_path):
    # _BARE_GRADUATION_RE only sees 2x/3x, so a bare 5x over a prior 4x passed.
    saved, _ = _save(
        tmp_path, f"- mature_claim | 5x ({TODAY})",
        prior=f"- mature_claim | 4x ({YESTERDAY})", citations_seen=True,
    )
    assert _level(saved, "mature_claim") == 4


def test_non_adjacent_evidence_cannot_inflate(tmp_path):
    saved, _ = _save(
        tmp_path,
        f'- claim_x | 3x ({TODAY}) [provenance: relay] '
        f'[evidence: {{ep0}} "deploy pipeline rotated overnight per relayed webpage"]',
        prior=f"- claim_x | 2x ({YESTERDAY})", citations_seen=True,
    )
    assert _level(saved, "claim_x") == 2


def test_backdated_inflation_over_prior_is_bounded(tmp_path):
    saved, _ = _save(
        tmp_path, f"- claim_y | 9x ({YESTERDAY}) {EV}",
        prior=f"- claim_y | 3x ({YESTERDAY})", citations_seen=True,
    )
    assert _level(saved, "claim_y") == 3


def test_validated_jump_is_bounded_to_one_rung(tmp_path):
    saved, _ = _save(
        tmp_path, f"- claim_z | 5x ({TODAY}) {EV}",
        prior=f"- claim_z | 2x ({YESTERDAY})", citations_seen=True,
    )
    assert _level(saved, "claim_z") == 3


def test_freeform_unnamed_line_is_bounded_too(tmp_path):
    saved, _ = _save(
        tmp_path, f"- thought: the pipeline rotated overnight | 9x ({YESTERDAY}) {EV}",
    )
    m = re.search(r"overnight \|\s*(\d+)x", saved)
    assert m and int(m.group(1)) == 1


# --- what must NOT change -------------------------------------------------------


def test_carried_line_keeps_its_level_and_date(tmp_path):
    line = f"- carried_claim | 5x ({YESTERDAY})"
    saved, result = _save(tmp_path, line, prior=line, citations_seen=True)
    assert line in saved
    assert not result.get("level_capped")


def test_validated_single_rung_is_untouched(tmp_path):
    saved, result = _save(
        tmp_path, f"- claim_v | 3x ({TODAY}) {EV}",
        prior=f"- claim_v | 2x ({YESTERDAY})", citations_seen=True,
    )
    assert _level(saved, "claim_v") == 3
    assert not result.get("level_capped")


def test_new_1x_line_is_untouched(tmp_path):
    saved, result = _save(tmp_path, f"- fresh_claim | 1x ({TODAY})")
    assert _level(saved, "fresh_claim") == 1
    assert not result.get("level_capped")


def test_reinsert_returns_to_the_level_it_was_saved_at(tmp_path):
    # Dropped from the file after the store saved it at 5x: re-adding it at 5x is
    # a tombstone carry (store.saved_pattern_levels), not an inflation.
    store = Store(tmp_path / "gate.db", project_name="Gate")
    try:
        store.save_continuity(_doc(f"- returning_claim | 5x ({YESTERDAY})"))
        _wrap(store, f"- returning_claim | 5x ({YESTERDAY})")
        _wrap(store, "- other_claim | 1x (2026-10-01)")
        _wrap(store, f"- returning_claim | 5x ({YESTERDAY})")
        assert _level(store.load_continuity(), "returning_claim") == 5
    finally:
        store.close()


def test_cap_is_reported(tmp_path):
    _, result = _save(tmp_path, f"- planted_claim | 9x ({YESTERDAY}) {EV}")
    capped = result.get("level_capped")
    assert capped and capped[0]["name"] == "planted_claim"
    assert capped[0]["written_level"] == 9 and capped[0]["capped_to"] == 1


# --- L1 + L2 round 1 (1007+29): the record the bound reads, and its receipts ----


def _wrap(store, patterns_tpl: str, *, today: str = TODAY):
    """One real wrap on an open store: two grounded episodes, prepare, save."""
    ids = [
        store.record(f"{GROUNDED} (wrap note {i})", EpisodeType.OBSERVATION).id
        for i in range(2)
    ]
    res = prepare_wrap(store)
    return validated_save_continuity(
        store, _doc(patterns_tpl.format(ep0=ids[0], ep1=ids[1])),
        today=today, wrap_token=res["wrap_token"],
    )


def _open(tmp_path, seed: str | None = None) -> Store:
    store = Store(tmp_path / "gate.db", project_name="Gate")
    if seed is not None:
        store.save_continuity(_doc(seed))
    return store


def test_drop_and_readd_returns_to_saved_level_not_high_water(tmp_path):
    # L1 #1 / L2 #2 (run): history's high-water mark (5) let a demoted pattern
    # (saved at 2x) come back at 5x by being left out for one wrap.
    store = _open(tmp_path, seed=f"- foo | 2x ({YESTERDAY})")
    try:
        store.upsert_pattern_history(
            pattern_name="foo", level=5, explanation="an older grounding about rotas",
            seen_at=YESTERDAY, wrap_id=None,
        )
        _wrap(store, f"- foo | 2x ({YESTERDAY})")           # the store records foo=2
        _wrap(store, "- other_claim | 1x (2026-10-01)")      # foo dropped
        _wrap(store, "- foo | 5x (2026-09-01)")              # re-added at the old mark
        assert _level(store.load_continuity(), "foo") == 2
    finally:
        store.close()


def test_out_of_band_file_raise_is_not_a_prior(tmp_path):
    # L1 #3: the file is writable by anyone; the store's own record is the prior.
    store = _open(tmp_path, seed=f"- foo | 2x ({YESTERDAY})")
    try:
        _wrap(store, f"- foo | 2x ({YESTERDAY})")
        store.save_continuity(_doc(f"- foo | 6x ({YESTERDAY})"))   # hand-raised
        _wrap(store, f"- foo | 6x ({YESTERDAY})")
        assert _level(store.load_continuity(), "foo") == 2
    finally:
        store.close()


def test_operator_hand_demotion_in_the_file_stands(tmp_path):
    store = _open(tmp_path, seed=f"- foo | 3x ({YESTERDAY})")
    try:
        _wrap(store, f"- foo | 3x ({YESTERDAY})")
        store.save_continuity(_doc(f"- foo | 1x ({YESTERDAY})"))   # operator lowers
        _wrap(store, f"- foo | 3x ({YESTERDAY})")                  # composer restores
        assert _level(store.load_continuity(), "foo") == 1
    finally:
        store.close()


def test_line_added_to_the_file_out_of_band_counts_as_new(tmp_path):
    store = _open(tmp_path, seed=f"- foo | 2x ({YESTERDAY})")
    try:
        _wrap(store, f"- foo | 2x ({YESTERDAY})")
        store.save_continuity(_doc(f"- foo | 2x ({YESTERDAY})\n- bar | 7x ({YESTERDAY})"))
        _wrap(store, f"- foo | 2x ({YESTERDAY})\n- bar | 7x ({YESTERDAY})")
        assert _level(store.load_continuity(), "bar") == 1
    finally:
        store.close()


def test_wrap_crossing_midnight_keeps_its_rung(tmp_path):
    # L2 #4 (run): prepared on day D, saved after midnight, `today` was D+1, so a
    # correctly stamped D graduation was skipped and then cut.
    from datetime import date, datetime, timedelta, timezone
    prep_day = (date.today() - timedelta(days=1)).isoformat()
    store = _open(tmp_path, seed=f"- foo | 1x ({prep_day})")
    try:
        ids = [store.record(f"{GROUNDED} (n{i})", EpisodeType.OBSERVATION).id
               for i in range(2)]
        res = prepare_wrap(store)
        noon_local = datetime.fromisoformat(f"{prep_day}T12:00:00").astimezone()
        store._conn.execute(
            "UPDATE metadata SET value = ? WHERE key = 'wrap_started_at'",
            (noon_local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),),
        )
        # prepare ran on prep_day: the day it gave the composer is that day.
        store._conn.execute(
            "UPDATE metadata SET value = ? WHERE key = 'wrap_today'", (prep_day,))
        store._conn.commit()
        result = validated_save_continuity(
            store, _doc(f"- foo | 2x ({prep_day}) " + EV.format(ep0=ids[0], ep1=ids[1])),
            wrap_token=res["wrap_token"],
        )
        assert _level(store.load_continuity(), "foo") == 2
        assert result["graduations_validated"] == 1
    finally:
        store.close()


def test_stale_level_capped_mark_is_dropped_once_the_level_is_earned(tmp_path):
    saved, result = _save(
        tmp_path, f"- foo | 3x ({TODAY}) {EV} (level-capped)",
        prior=f"- foo | 2x ({YESTERDAY})", citations_seen=True,
    )
    assert "(level-capped)" not in saved
    assert not result.get("level_capped")


def test_cut_carried_line_loses_its_carried_forward_mark(tmp_path):
    # L1 #5 (run): a hold the bound overrode still said "(carried-forward)".
    saved, _ = _save(
        tmp_path, f"- foo | 3x ({TODAY}) [evidence: deadbeef \"no such episode\"]",
        prior=f"- foo | 2x ({YESTERDAY})", citations_seen=True,
        history={"foo": 3},
    )
    line = next(l for l in saved.split("\n") if "foo |" in l)
    assert "(carried-forward)" not in line and "(level-capped)" in line
    assert _level(saved, "foo") == 2


def test_oversized_level_is_cut_not_a_crash(tmp_path):
    # L1 #4 (run): int() refuses more than 4300 digits.
    saved, result = _save(tmp_path, f"- huge | {'9' * 5000}x ({YESTERDAY})")
    assert _level(saved, "huge") == 1
    assert result["level_capped"][0]["capped_to"] == 1


def test_new_validated_lines_cut_to_1x_are_not_graduations(tmp_path):
    # L1 #2 (run): validated=2 and graduated_names=['foo','bar'] for two lines
    # that entered at 1x.
    _, result = _save(tmp_path, f"- foo | 9x ({TODAY}) {EV}\n- bar | 7x ({TODAY}) {EV}")
    assert result["graduations_validated"] == 0


def test_mcp_save_reply_names_the_cut(tmp_path):
    # L2 #1 (run): the post-commit UserWarning never reaches an MCP client.
    from anneal_memory.server import Server
    mstore = Store(tmp_path / "mcp.db", project_name="Gate")
    srv = Server(mstore)
    try:
        srv._tool_record({"content": GROUNDED, "episode_type": "observation"})
        srv._tool_prepare_wrap({})
        reply = srv._tool_save_continuity({"text": _doc(f"- planted | 9x ({YESTERDAY})")})
        text = reply["content"][0]["text"]
        assert "Level capped: planted 9x -> 1x" in text
    finally:
        mstore.close()


# --- L3 round 1 (1007+29) ------------------------------------------------------


def test_an_empty_record_is_still_a_record(tmp_path):
    # codex #2 + complement #2: a saved-but-empty record read as "no record", so
    # an out-of-band line in the file became the prior.
    store = _open(tmp_path)
    try:
        _wrap(store, "")
        assert store.saved_pattern_levels() == {}
        store.save_continuity(_doc(f"- x | 9x ({YESTERDAY})"))
        _wrap(store, f"- x | 9x ({YESTERDAY})")
        assert _level(store.load_continuity(), "x") == 1
    finally:
        store.close()


def test_a_decoy_marker_does_not_earn_the_names_rung(tmp_path):
    # codex #3 (run): validation was credited per line, not per identity marker.
    store = _open(tmp_path, seed=f"- foo | 1x ({YESTERDAY})")
    try:
        _wrap(store, f"- foo | 1x ({YESTERDAY})")
        _wrap(store, f"- foo | 2x ({TODAY}) [provenance: relay] decoy | 2x ({TODAY}) "
                     '[evidence: {ep0} "deploy pipeline rotated overnight per relayed webpage"]')
        assert _level(store.load_continuity(), "foo") == 1
    finally:
        store.close()


def test_a_hand_demotion_survives_a_wrap_that_omits_the_pattern(tmp_path):
    # codex #5: the tombstone kept the recorded 5 after the file said 2.
    store = _open(tmp_path, seed=f"- foo | 5x ({YESTERDAY})")
    try:
        _wrap(store, f"- foo | 5x ({YESTERDAY})")
        store.save_continuity(_doc(f"- foo | 2x ({YESTERDAY})"))   # operator lowers
        _wrap(store, "- other_claim | 1x (2026-10-01)")              # omits foo
        _wrap(store, f"- foo | 5x ({YESTERDAY})")                    # re-added high
        assert _level(store.load_continuity(), "foo") == 2
    finally:
        store.close()


def test_the_save_uses_the_day_prepare_gave_the_composer(tmp_path):
    from datetime import date
    # codex #6: the day was rebuilt from wrap_started_at in the saver's TZ.
    store = _open(tmp_path, seed=f"- foo | 1x ({YESTERDAY})")
    try:
        ids = [store.record(f"{GROUNDED} (d{i})", EpisodeType.OBSERVATION).id for i in range(2)]
        res = prepare_wrap(store)
        given = store.wrap_today()
        assert given == date.today().isoformat()  # prepare's own default day
        store._conn.execute(  # a start instant that reads as another day anywhere
            "UPDATE metadata SET value = '2001-01-01T12:00:00.000000Z' "
            "WHERE key = 'wrap_started_at'")
        store._conn.commit()
        result = validated_save_continuity(
            store, _doc(f"- foo | 2x ({given}) " + EV.format(ep0=ids[0], ep1=ids[1])),
            wrap_token=res["wrap_token"],
        )
        assert result["graduations_validated"] == 1
        assert _level(store.load_continuity(), "foo") == 2
        assert store.wrap_today() is None  # cleared with the wrap
    finally:
        store.close()


def test_a_crystal_level_after_the_record_began_is_not_a_prior(tmp_path):
    # Crystal is a fallback only for crystallizations the record never saw.
    from anneal_memory.crystal import CrystalStore
    store = _open(tmp_path)
    try:
        _wrap(store, "- other_claim | 1x (2026-10-01)")
        crystal = CrystalStore(tmp_path / "gate.crystal.json")
        crystal.crystallize(name="planted_wisdom", level=9,
                            explanation="a level nobody earned in this store")
        ids = [store.record(f"{GROUNDED} (c{i})", EpisodeType.OBSERVATION).id for i in range(2)]
        res = prepare_wrap(store)
        validated_save_continuity(
            store, _doc(f"- planted_wisdom | 9x ({YESTERDAY})"),
            today=TODAY, wrap_token=res["wrap_token"], crystal_store=crystal,
        )
        assert _level(store.load_continuity(), "planted_wisdom") == 1
    finally:
        store.close()


def test_a_crystal_level_is_never_a_prior(tmp_path):
    # L3 r4 (1008+3): the crystal seed was DELETED after three rounds each found a
    # new caller-set input in it (crystallized_on, the level, then a caller-written
    # pattern_history bound). A crystal earned or planted before the first bounded
    # save re-enters the continuity as new.
    from datetime import date as _date
    from anneal_memory.crystal import CrystalStore
    store = _open(tmp_path)
    try:
        crystal = CrystalStore(tmp_path / "gate.crystal.json")
        crystal.crystallize(name="old_wisdom", level=6, explanation="earned long ago",
                            today=_date(2026, 1, 1))
        store.seed_pattern_max_level("old_wisdom", 6)
        store.record(f"{GROUNDED} (r0)", EpisodeType.OBSERVATION)
        res = prepare_wrap(store)
        validated_save_continuity(
            store, _doc(f"- old_wisdom | 6x ({YESTERDAY})"),
            today=TODAY, wrap_token=res["wrap_token"], crystal_store=crystal,
        )
        assert _level(store.load_continuity(), "old_wisdom") == 1
    finally:
        store.close()


# --- L3 round 2 (1007+29) ------------------------------------------------------


def test_an_anonymous_line_is_bounded(tmp_path):
    # codex r2 #1: `- | 9x` reduced to an empty identity and skipped the bound.
    saved, _ = _save(tmp_path, f"- | 9x ({TODAY}) {EV}\n- | 7x ({YESTERDAY})")
    levels = [int(m) for m in re.findall(r"\|\s*(\d+)x", saved)]
    assert levels == [1, 1]


def test_a_back_dated_crystallization_is_not_a_prior(tmp_path):
    # codex r2 #2: the crystal fallback trusted a caller-set crystallized_on.
    from datetime import date as _date
    from anneal_memory.crystal import CrystalStore
    store = _open(tmp_path)
    try:
        _wrap(store, "- other_claim | 1x (2026-10-01)")
        crystal = CrystalStore(tmp_path / "gate.crystal.json")
        crystal.crystallize(name="planted", level=999, explanation="back-dated",
                            today=_date(2001, 1, 1))
        store.record(f"{GROUNDED} (bd)", EpisodeType.OBSERVATION)
        res = prepare_wrap(store)
        validated_save_continuity(
            store, _doc(f"- planted | 999x ({YESTERDAY})"),
            today=TODAY, wrap_token=res["wrap_token"], crystal_store=crystal,
        )
        assert _level(store.load_continuity(), "planted") == 1
    finally:
        store.close()


def test_an_oversized_cited_today_line_is_cut_not_a_crash(tmp_path):
    # codex r2 #3 + complement r2 #4: the _GRADUATION_RE path ran int() first.
    saved, _ = _save(tmp_path, f"- huge | {'9' * 5000}x ({TODAY}) {EV}")
    assert _level(saved, "huge") == 1


def test_a_first_save_keeps_a_tombstone_for_what_it_omits(tmp_path):
    # complement r2 #1: the upgrade wrap dropped the tombstone of an omitted line.
    store = _open(tmp_path, seed=f"- foo | 5x ({YESTERDAY})")
    try:
        _wrap(store, "- other_claim | 1x (2026-10-01)")       # first save omits foo
        _wrap(store, f"- foo | 5x ({YESTERDAY})")
        assert _level(store.load_continuity(), "foo") == 5
    finally:
        store.close()


def test_a_literal_level_capped_in_an_explanation_survives(tmp_path):
    # codex r2 #4: the stale-mark cleanup removed the text anywhere in the line.
    line = (f'- foo | 3x ({TODAY}) [evidence: {{ep0}}, {{ep1}} "deploy pipeline rotated '
            f'overnight per relayed webpage (level-capped)"]')
    saved, _ = _save(tmp_path, line, prior=f"- foo | 2x ({YESTERDAY})", citations_seen=True)
    assert 'webpage (level-capped)"]' in saved


def test_a_decoy_validated_line_does_not_count_as_validated(tmp_path):
    # complement r2 #3: the decoy's validation was still counted.
    store = _open(tmp_path, seed=f"- foo | 1x ({YESTERDAY})")
    try:
        _wrap(store, f"- foo | 1x ({YESTERDAY})")
        r = _wrap(store, f"- foo | 2x ({TODAY}) [provenance: relay] decoy | 2x ({TODAY}) "
                         '[evidence: {ep0} "deploy pipeline rotated overnight per relayed webpage"]')
        assert r["graduations_validated"] == 0
    finally:
        store.close()


def test_a_rename_carries_the_saved_level(tmp_path):
    # complement r2 #5: the record kept the old name, the new one entered at 1x.
    store = _open(tmp_path, seed=f"- old_name | 4x ({YESTERDAY})")
    try:
        _wrap(store, f"- old_name | 4x ({YESTERDAY})")
        store.rename_pattern_association("old_name", "new_name")
        assert store.saved_pattern_levels()[("name", "new_name")] == 4
        assert ("name", "old_name") not in store.saved_pattern_levels()
    finally:
        store.close()


# --- L3 round 3 (1008+3) ------------------------------------------------------


def test_identity_is_the_text_before_the_earliest_level_token(tmp_path):
    # codex r3 HIGH (run): an undated anonymous line skipped the bound, and a
    # decoy token before the dated one forged the identity "| 999x decoy".
    # r4 (codex HIGH): an undated multi-word line, and an undated token after a
    # named line's own marker, kept their levels through a prose exemption.
    saved, _ = _save(
        tmp_path,
        f"- | 9x\n- | 999x decoy | 9x ({YESTERDAY})\n- multi word | 999x\n"
        f"- foo | 9x ({YESTERDAY}) then | 999x",
    )
    levels = [int(m) for m in re.findall(r"\|\s*(\d+)x", saved)]
    assert levels == [1, 1, 1, 1, 1, 1]
    # codex r3 HIGH + complement (run): renaming a name to itself deleted its
    # saved level, so the next wrap cut it to 1x as new.
    (tmp_path / "rn").mkdir()
    store = _open(tmp_path / "rn")
    try:
        _wrap(store, "- foo | 1x (2026-10-01)")
        before = store.saved_pattern_levels()
        assert store.rename_pattern_association("foo", "foo") == 0
        assert store.saved_pattern_levels() == before
    finally:
        store.close()


# --- L3 r5 (codex HIGH, run on 023b700): the bound's grammar must cover every reader --
import pytest  # noqa: E402


@pytest.mark.parametrize("ws", [" ", " ", "\v", "\f", "\x1c", "　"])
def test_bound_grammar_covers_every_reader_whitespace(tmp_path, ws):
    """``_GRADUATION_RE`` reads ``|<any Unicode space>999x``; the bound must too."""
    saved, _ = _save(tmp_path, "- planted |" + ws + "999x (" + TODAY + ") " + EV,
                     prior="- planted | 1x (" + YESTERDAY + ")")
    levels = [int(m.group(1)) for m in re.finditer(r"\|\s*(\d+)x", saved)]
    assert levels and max(levels) <= 2, saved


def test_bound_reads_non_ascii_digits(tmp_path):
    """``\\d`` readers and ``int`` read Arabic-Indic ``٩٩٩`` as 999."""
    saved, _ = _save(tmp_path, "- planted | ٩٩٩x (" + TODAY + ") " + EV,
                     prior="- planted | 1x (" + YESTERDAY + ")")
    levels = [int(m.group(1)) for m in re.finditer(r"\|\s*(\d+)x", saved)]
    assert levels and max(levels) <= 2, saved
