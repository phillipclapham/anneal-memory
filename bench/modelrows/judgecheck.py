#!/usr/bin/env python3
"""Re-judge a balanced sample of saved answers with a second judge (Sonnet 5.5, or an OpenRouter
model with ``--judge or:<model id>``) and report agreement with the Haiku 5.5 judge that scored
them: raw agreement, Cohen's kappa, and where the two disagree. A low agreement downgrades the
row to "judge-limited".

  python -m bench.modelrows.judgecheck --kind a --data DATA --runs RUN [RUN...] --n 120 --out OUT --budget 3
  ... --judge or:deepseek/deepseek-chat-v3.1 --or-budget 0.5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from . import orouter, rows
from .common import SONNET, Client, Ledger, cluster_bootstrap  # noqa: F401


def kappa(pairs: list[tuple[int, int]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    pa1 = sum(a for a, _ in pairs) / n
    pb1 = sum(b for _, b in pairs) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return None if pe == 1 else (po - pe) / (1 - pe)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("a", "b"), required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--judge", default="sonnet", help="sonnet | or:<OpenRouter model id>")
    ap.add_argument("--or-budget", type=float, default=0.0)
    args = ap.parse_args()

    data = Path(args.data)
    scs = {s["id"]: s for s in json.loads((data / f"scenarios_{args.kind}.json").read_text())}
    units = [json.loads(f.read_text()) for r in args.runs for f in sorted((Path(r) / "units").glob("*.json"))]
    arms = rows.A_ARMS if args.kind == "a" else rows.B_ARMS
    pool = []   # (unit, arm, probe_index or None)
    for u in units:
        for arm in arms:
            if args.kind == "a":
                pool += [(u, arm, i) for i in range(len(u["arms"][arm]["probes"]))]
            else:
                pool.append((u, arm, None))
    rng = random.Random(args.seed)

    def hscore(x):
        u, arm, i = x
        return (u["arms"][arm]["probes"][i] if i is not None else u["arms"][arm])["score"]
    # balanced by the Haiku judge's own call (most A answers are 0, so a plain sample has no positives)
    pos = [x for x in pool if hscore(x) == 1]
    neg = [x for x in pool if hscore(x) == 0]
    pick = rng.sample(pos, min(args.n // 2, len(pos))) + rng.sample(neg, min(args.n - args.n // 2, len(neg)))

    if args.judge.startswith("or:"):
        ledger = orouter.ORLedger()
        judge = orouter.Client(args.judge[3:], ledger, args.or_budget)
    else:
        ledger = Ledger()
        judge = Client(SONNET, ledger, args.budget)
    out_rows = []
    for u, arm, i in pick:
        sc = scs[u["scenario"]]
        if args.kind == "a":
            p = u["arms"][arm]["probes"][i]
            _, ev = rows.sessions_a(sc, u["variant"])
            new = rows._judge_a(judge, sc, ev, p["answer"])
            old = p
        else:
            old = u["arms"][arm]
            new = rows._judge_b(judge, sc, old["answer"])
        out_rows.append({"uid": u["uid"], "arm": arm, "i": i, "haiku": old["score"], "check": new["score"],
                         "haiku_why": old.get("why"), "check_why": new.get("why")})
    ok = [r for r in out_rows if r["haiku"] in (0, 1) and r["check"] in (0, 1)]
    pairs = [(r["haiku"], r["check"]) for r in ok]
    summary = {"kind": args.kind, "sampled": len(out_rows), "scored_by_both": len(ok),
               "agreement": sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else None,
               "kappa": kappa(pairs),
               "haiku_mean": sum(a for a, _ in pairs) / len(pairs) if pairs else None,
               "check_mean": sum(b for _, b in pairs) / len(pairs) if pairs else None,
               "agree_when_haiku_1": (lambda x: sum(1 for r in x if r["check"] == 1) / len(x) if x else None)(
                   [r for r in ok if r["haiku"] == 1]),
               "agree_when_haiku_0": (lambda x: sum(1 for r in x if r["check"] == 0) / len(x) if x else None)(
                   [r for r in ok if r["haiku"] == 0]),
               "judge": args.judge,
               "note": "sample balanced on the Haiku call, so kappa is for a 50/50 mix, not for the run's prevalence",
               "ledger": ledger.summary()["total"]}
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": out_rows}, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
