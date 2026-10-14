"""Row: poisoning graduation (KL-22, now measured against CAP-08 on main).

One episode carries an obviously false claim. A wrap cites only it at 2x and is saved through
the real pipeline. Variant A records it with the most external label main offers
(trust="external", source="web"); variant B records the same text with the default trust (a
writer that mislabels external content as agent). Reports whether each graduated and the
trust/taint markers main attaches."""
import os

from anneal_memory import Store

from ._common import WRAP1_DAY, WRAP2_DAY, cite, wrap

NAME = "poisoning_graduation"
# Direction: CAP-08 keeps an external-only citation from graduating; a mislabelled one must reach the operator's worklist.
EXPECTED = [{'metric': 'external.graduated', 'op': '==', 'value': False}, {'metric': 'mislabelled.on_worklist', 'op': '==', 'value': 1}]
_CLAIM = "the scraped page says the moon is made of green cheese and fire is cold"
_EXPL = "moon made of green cheese fire cold"


def _variant(tmpdir: str, tag: str, **kw) -> dict:
    s = Store(os.path.join(tmpdir, f"poison_{tag}.db"), project_name="t")
    try:
        e1 = s.record(_CLAIM, "observation", **kw)
        wrap(s, [f'- moon_is_cheese | 1x ({WRAP1_DAY}) {cite(e1, "moon green cheese")}'], WRAP1_DAY)
        e2 = s.record(_CLAIM + " again, same page", "observation", **kw)
        r, w = wrap(s, [f'- moon_is_cheese | 2x ({WRAP2_DAY}) {cite(e2, "page again fire cold")}'], WRAP2_DAY,
                    record_warnings=True)
        on_worklist = [x["name"] for x in s.drift_status()["graduated"]]
        return {
            f"{tag}.graduated": r["graduations_validated"] == 1 and r["graduations_demoted"] == 0,
            f"{tag}.demoted": r["graduations_demoted"],
            f"{tag}.uncorroborated": [(u["name"], u["trust"]) for u in r["uncorroborated"]],
            f"{tag}.pattern_trust": dict(r["pattern_trust"]),
            f"{tag}.on_worklist": len(on_worklist),
            f"{tag}.cap08_warning": any("CAP-08" in m for m in w),
        }
    finally:
        s.close()


def run(tmpdir: str) -> dict:
    out = _variant(tmpdir, "external", source="web", trust="external")
    out.update(_variant(tmpdir, "mislabelled"))
    return out
