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


def test_reinsert_up_to_history_high_water_is_allowed(tmp_path):
    # Dropped from the prior continuity but earned 5x before (pattern_history):
    # re-adding it at its earned level is a tombstone carry, not an inflation.
    saved, _ = _save(
        tmp_path, f"- returning_claim | 5x ({YESTERDAY})",
        prior="- other_claim | 1x (2026-10-01)", citations_seen=True,
        history={"returning_claim": 5},
    )
    assert _level(saved, "returning_claim") == 5


def test_cap_is_reported(tmp_path):
    _, result = _save(tmp_path, f"- planted_claim | 9x ({YESTERDAY}) {EV}")
    capped = result.get("level_capped")
    assert capped and capped[0]["name"] == "planted_claim"
    assert capped[0]["written_level"] == 9 and capped[0]["capped_to"] == 1
