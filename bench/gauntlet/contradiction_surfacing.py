"""Row: contradiction surfacing.

Two episodes contradict each other (X is true / X is false) and one 2x pattern cites both.
Reports whatever main surfaces. Variant A leaves the pattern without a contradiction stance;
variant B adds an explicit [no-contradicts], the declaration a careless wrap would write."""
import os

from anneal_memory import Store

from ._common import WRAP1_DAY, WRAP2_DAY, cite, wrap

NAME = "contradiction_surfacing"
# Direction: anneal does not judge whether cited
# episodes agree (semantic truth is the operator's); the graduation must reach the
# operator's worklist in both variants. "graduated" stays a measured count, not an expectation.
EXPECTED = [{"metric": "undeclared.on_worklist", "op": "==", "value": 1},
            {"metric": "no_contradicts.on_worklist", "op": "==", "value": 1}]


def _variant(tmpdir: str, tag: str, suffix: str) -> dict:
    s = Store(os.path.join(tmpdir, f"contra_{tag}.db"), project_name="t")
    try:
        f = s.record("nightly export job rerun policy came up in standup", "observation")
        wrap(s, [f'- nightly_export_rerun_policy | 1x ({WRAP1_DAY}) '
                 f'{cite(f, "export rerun standup")}'], WRAP1_DAY)
        a = s.record("the nightly export job is idempotent and safe to rerun", "observation")
        b = s.record("the nightly export job is not idempotent and rerunning it duplicates rows",
                     "observation")
        r, w = wrap(s, [f'- nightly_export_rerun_policy | 2x ({WRAP2_DAY}) '
                        f'{cite(a, "nightly export job idempotent rerun")} '
                        f'{cite(b, "nightly export job idempotent rerun duplicates")}{suffix}'],
                    WRAP2_DAY, record_warnings=True)
        on_worklist = [x["name"] for x in s.drift_status()["graduated"]]
        return {
            f"{tag}.on_worklist": int("nightly_export_rerun_policy" in on_worklist),
            f"{tag}.graduated": r["graduations_validated"],
            f"{tag}.demoted": r["graduations_demoted"],
            f"{tag}.proven_without_decl": len(r["proven_without_contradicts_declaration"]),
            f"{tag}.warnings": [m[:100] for m in w],
        }
    finally:
        s.close()


def run(tmpdir: str) -> dict:
    out = _variant(tmpdir, "undeclared", "")
    out.update(_variant(tmpdir, "no_contradicts", " [no-contradicts]"))
    return out
