"""The reference floor: a protected layer may not fall below REFERENCE_FLOOR_PERCENT of
the largest it was in the last 30 days, which a save-to-save comparison cannot see.

One test per behaviour. The end-to-end ones run the real save pipeline on a
FLOW_SCHEMA store (the same shape ``TestCatastrophicShrinkGate`` uses)."""

from __future__ import annotations

import pytest

from anneal_memory import FLOW_SCHEMA, Store, prepare_wrap, validated_save_continuity
from anneal_memory import continuity as C


def _doc(understanding: int, patterns: int = 900, context: int = 20) -> str:
    """A FLOW_SCHEMA continuity whose Understanding is ~`understanding` chars."""
    return (
        "# flow — Memory (test)\n\n"
        "## State\nparity-observe mode.\n\n"
        "## Active Threads\n- migration\n\n"
        "## Patterns\n" + ("verify_or_surface grounds truth. " * (patterns // 33)) + "\n\n"
        "## Decisions\n[decided] diet-first.\n\n"
        "## Context\n" + ("recent work. " * (context // 13 + 1)) + "\n\n"
        "## Understanding\n" + ("the felt layer, who we are together. " * (understanding // 37)) + "\n"
    )


def _masses(text: str) -> dict[str, int]:
    return C._floor_role_masses(text, FLOW_SCHEMA)


def _save(store: Store, text: str, today: str, **kw):
    store.record(f"episode {today}: observation.", "observation")
    pkg = prepare_wrap(store, max_chars=40000)
    return validated_save_continuity(
        store, text, today=today, wrap_token=pkg["wrap_token"], **kw)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "s.db", project_name="flow")
    s.set_section_schema(FLOW_SCHEMA)
    yield s
    s.close()


def test_a_slow_shrink_over_several_saves_is_refused(store):
    """Each save keeps ~80% of the one before, which the floor accepts on its own;
    the second lands below the floor's share of the largest: refused, and the wrap
    stays open."""
    _save(store, _doc(3000), "2026-11-01")
    _save(store, _doc(2400), "2026-11-02")
    before = store.load_continuity()
    with pytest.raises(ValueError, match="reference floor") as exc:
        _save(store, _doc(1900), "2026-11-03")
    assert "Understanding" in str(exc.value) and "allow_shrink" in str(exc.value)
    assert store.get_wrap_started_at() is not None
    assert store.load_continuity() == before


def test_allow_shrink_passes_the_floor_and_resets_only_that_section(store):
    """The override saves; the felt layer's record restarts from that save, so the
    next ordinary save at the new size is not refused against the pre-diet size.
    Nothing here belongs to another section."""
    _save(store, _doc(3000), "2026-11-01")
    _save(store, _doc(2400), "2026-11-02")
    _save(store, _doc(1900), "2026-11-03", allow_shrink=True)
    log = store.load_meta()["section_mass_log"]
    felt = [e["m"]["narrative-timeless"] for e in log if "narrative-timeless" in e["m"]]
    assert len(felt) == 1  # only the saved size is left of the felt layer's record
    _save(store, _doc(1900), "2026-11-04")  # no override needed


def test_only_the_felt_layer_is_floored(store):
    """Patterns and Context fall hard across saves without a refusal: the floor is
    the felt layer's alone (the identity layer keeps the save-to-save check)."""
    _save(store, _doc(3000, patterns=3000, context=3000), "2026-11-01")
    _save(store, _doc(3000, patterns=2100, context=2000), "2026-11-02")
    _save(store, _doc(3000, patterns=1500, context=1000), "2026-11-03")
    _save(store, _doc(3000, patterns=1100, context=100), "2026-11-04")


def test_the_mass_log_is_bounded_and_a_burst_of_saves_cannot_flush_it():
    """One entry per date (the day's largest mass), at most the cap kept: 200 saves in
    one day leave the history in place, so shrinking after them is still checked."""
    big, small = _doc(3000), _doc(2000)
    log: object = C._updated_mass_log(None, None, big, FLOW_SCHEMA, None, "2026-11-01")
    for _ in range(200):
        log = C._updated_mass_log(log, big, big, FLOW_SCHEMA, None, "2026-11-02")
    log = C._updated_mass_log(log, big, small, FLOW_SCHEMA, None, "2026-11-02")
    assert [e["d"] for e in log] == ["2026-11-01", "2026-11-02"]
    assert C._reference_mass(log, "narrative-timeless", "2026-11-03",
                             _masses(small)["narrative-timeless"], 0)[0] >= 3000
    many: object = None
    for day in range(C._MASS_LOG_MAX_ENTRIES * 2):
        many = C._updated_mass_log(many, big, big, FLOW_SCHEMA, None,
                                   f"2026-{1 + day // 28:02d}-{1 + day % 28:02d}")
    assert len(many) <= C._MASS_LOG_MAX_ENTRIES


def test_a_hostile_stored_log_is_read_and_rewritten_within_bounds():
    """A meta that already carries a huge list of wide entries costs the bound, not
    its own size: only the tail is read, only the floor roles survive, and a size
    above the cap is not a size."""
    wide = {"d": "2026-11-20", "m": {f"role{i}": 5 for i in range(500)} | {"narrative-timeless": 4000}}
    hostile = [wide] * 5000 + [{"d": "2026-11-21", "m": {"narrative-timeless": 10**30}}]
    kept = C._updated_mass_log(hostile, None, _doc(3000), FLOW_SCHEMA, None, "2026-11-22")
    assert len(kept) <= C._MASS_LOG_MAX_ENTRIES
    assert all(set(e["m"]) <= set(C._REFERENCE_FLOOR_ROLES) for e in kept)
    assert all(v <= C._MASS_LOG_MAX_MASS for e in kept for v in e["m"].values())


def test_a_layer_that_shrank_to_under_the_size_floor_is_still_checked():
    """The per-save gate skips a prior under 500 chars; the reference floor must not."""
    log = [{"d": "2026-11-01", "m": {"narrative-timeless": 3000, "graduating": 900}}]
    prior, new = _doc(450), _doc(300)
    assert _masses(prior)["narrative-timeless"] < 500
    with pytest.raises(ValueError, match="reference floor"):
        C._check_no_catastrophic_shrink(
            prior, new, FLOW_SCHEMA, allow_shrink=False,
            mass_log=log, today="2026-11-20")


def test_a_pattern_crystallized_out_since_is_not_counted_against_the_floor():
    """The reference counts only what should still be here: identity that moved to
    the crystal store since the largest entry leaves the reference."""
    log = [
        {"d": "2026-11-01", "m": {"narrative-timeless": 3000}},
        {"d": "2026-11-10", "m": {"narrative-timeless": 2400}, "c": {"narrative-timeless": 600}},
    ]
    ref, _ = C._reference_mass(log, "narrative-timeless", "2026-11-20", prior_mass=2400,
                               credit_now=0)
    assert ref == 2400  # 3000 at the largest, less the 600 credited since
    uncredited = [{"d": e["d"], "m": e["m"]} for e in log]
    assert C._reference_mass(uncredited, "narrative-timeless", "2026-11-20", 2400, 0)[0] == 3000
    # a malformed or out-of-window log is ignored, never raised on
    for bad in (None, "x", [1, {"d": 5}, {"d": "no", "m": {}}],
                [{"d": "2020-01-01", "m": {"narrative-timeless": 99999}}]):
        assert C._reference_mass(bad, "narrative-timeless", "2026-11-20", 2400, 0)[0] == 2400


def test_a_future_dated_entry_is_not_a_reference_and_a_bad_today_keeps_the_log():
    """A hand-edited or clock-skewed entry dated after today does not pin the
    reference, and a ``today`` the window cannot be counted from drops nothing."""
    future = [{"d": "2099-01-01", "m": {"narrative-timeless": 99999}}]
    assert C._reference_mass(future, "narrative-timeless", "2026-11-20", 2400, 0)[0] == 2400
    log = [{"d": "2026-11-01", "m": {"narrative-timeless": 3000}}]
    kept = C._updated_mass_log(log, None, _doc(3000), FLOW_SCHEMA, None,
                               "2026-11-24T10:00")
    assert kept == log
