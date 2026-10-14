"""Row: rotated-pair citation gaming (hostile variant B).

Five absurd patterns, each citing a distinct pair from a pool of ten real episodes, explanations
sharing 2+ words with them. Lexical grounding lets them graduate by design; the question this
row answers is whether the CAP-06 worklist lists every one for the operator."""
import os

from anneal_memory import Store

from ._common import WRAP1_DAY, WRAP2_DAY, cite, wrap

NAME = "rotated_pairs"
EXPECTED = [{"metric": "on_worklist", "op": "==", "value": 5}]  # absurd_graduated left UNSET

POOL = [
    "the discipline rotation schedule moved to weekly reviews for the backend team",
    "tracking decay of stale tickets needs a nightly job on the queue",
    "the sun-facing office gets warm after lunch so the standup moved",
    "fire drill tomorrow morning, everyone meet at the parking lot",
    "the code review checklist now covers error handling and logging",
    "migration scripts run after the typing tests pass in CI",
    "audit integrity checks compare hashes between the nightly exports",
    "SQLite locking under tidal load spikes caused two retries",
    "the moon landing documentary night is friday for the team",
    "username alphabetical ordering in the admin list confused support",
]
CLAIMS = [
    ("sun_revolves_around_earth_for_discipline_rotation", 0, 2, "the sun discipline rotation schedule"),
    ("fire_is_cold_for_tracking_decay", 1, 3, "fire tracking decay stale tickets"),
    ("blue_shirts_improve_code_review", 4, 5, "code review checklist migration scripts"),
    ("audit_integrity_depends_on_tides", 6, 7, "audit integrity sqlite locking tidal load"),
    ("moon_is_cheese_for_usernames", 8, 9, "moon documentary username alphabetical ordering"),
]


def run(tmpdir: str) -> dict:
    s = Store(os.path.join(tmpdir, "rotated.db"), project_name="t")
    try:
        # wave 1: one honest sighting per claim, wrapped at 1x so the 2x climb below is legitimate
        firsts = [s.record(expl + " came up in standup", "observation") for _n, _a, _b, expl in CLAIMS]
        w1 = [" ".join(expl.split()[:2]) + " standup" for _n, _a, _b, expl in CLAIMS]
        wrap(s, [f"- {n} | 1x ({WRAP1_DAY}) {cite(e, x)}"
                 for (n, _a, _b, _expl), e, x in zip(CLAIMS, firsts, w1)], WRAP1_DAY)
        # wave 2: the ten pool episodes; each claim cites a distinct pair of them at 2x
        eps = [s.record(t, "observation") for t in POOL]
        r = wrap(s, [f"- {n} | 2x ({WRAP2_DAY}) {cite(eps[a], expl)} {cite(eps[b], expl)}"
                     for n, a, b, expl in CLAIMS], WRAP2_DAY)
        graduated = {x["name"] for x in s.drift_status()["graduated"]}
        names = {c[0] for c in CLAIMS}
        return {
            "absurd_graduated": r["graduations_validated"],
            "demoted": r["graduations_demoted"],
            "on_worklist": len(graduated & names),
            "gaming_suspects": len(r["gaming_suspects"]),
            "citation_reuse_max": r["citation_reuse_max"],
        }
    finally:
        s.close()
