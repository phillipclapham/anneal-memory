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


















# --- what must NOT change -------------------------------------------------------












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





















# --- L3 round 1 (1007+29) ------------------------------------------------------














# --- L3 round 2 (1007+29) ------------------------------------------------------
















# --- L3 round 3 (1008+3) ------------------------------------------------------




# --- L3 r5 (codex HIGH, run on 023b700): the bound's grammar must cover every reader --
import pytest  # noqa: E402


# \v, \f and \x1c are line terminators, refused at the save entry since r6.




# --- L3 r6/r7 (1008+11, run): one text grammar for the final text ----------------












# --- L3 r8 (1008+11), Phill 12:13 "(A)": nothing un-canonical enters the pipeline ----




# --- L3 r9 (1008+11, desk ruling inside Phill's (A)): every load point is canonical ---




# --- gradgate-merge L3 r1 (1009+18): each reproduced by codex at runtime on c932674 --




def test_a_cut_decoy_line_does_not_unname_a_real_graduation():
    # codex MED 3: graduated_names.remove("foo") dropped the FIRST foo line's name.
    from anneal_memory.graduation import validate_graduations
    expl = "deploy pipeline rotated overnight per relayed webpage"
    ids = {"aaaa1111": GROUNDED, "bbbb2222": GROUNDED, "cccc3333": GROUNDED}
    text = _doc(
        f'- foo | 2x ({TODAY}) [evidence: aaaa1111 "{expl}"]\n'
        f'- foo | 9x then | 2x ({TODAY}) [evidence: bbbb2222 "{expl}"]\n'
        f'- bar | 2x ({TODAY}) [evidence: cccc3333 "{expl}"]'
    )
    prior = _doc(f"- foo | 1x ({YESTERDAY})\n- bar | 1x ({YESTERDAY})")
    result = validate_graduations(
        text, set(ids), TODAY, node_content_map=ids, citations_seen=True,
        prior_text=prior, saved_levels={("name", "foo"): 1, ("name", "bar"): 1},
    )
    assert sorted(result.graduated_names) == ["bar", "foo"]




def test_a_cut_carried_line_keeps_prose_that_names_the_mark():
    # codex MED 5: every "(carried-forward)" on the line was removed, prose included.
    from anneal_memory.graduation import DEFAULT_GRADUATING, _apply_prior_bound
    lines = ["## Patterns",
             f"- foo | 3x ({YESTERDAY}) (carried-forward) keeps literal (carried-forward) text"]
    _apply_prior_bound(
        lines, prior_text=_doc(f"- foo | 3x ({YESTERDAY})"),
        saved_levels={("name", "foo"): 2}, validated_lines={}, carried_lines={1},
        graduating_headings=DEFAULT_GRADUATING,
    )
    assert lines[1] == (f"- foo | 2x ({YESTERDAY}) keeps literal (carried-forward) text"
                        " (level-capped)")
