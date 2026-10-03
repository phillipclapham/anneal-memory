#!/usr/bin/env python3
"""Merge run.py result directories and print per-condition scores with Wilson 95% CIs,
both over every task each condition ran and over the tasks ALL listed conditions share.

  python bench/inmind/aggregate.py RUN_DIR [RUN_DIR ...] [--json OUT]

A later directory wins when the same (condition, task) appears twice.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

METRICS = ("naive", "target-recall", "application", "answer-only", "target-recall@inject")
ORDER = ("none", "oracle", "anneal-recall", "anneal-continuity", "paper-probe")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def load(dirs: list[Path]) -> dict[str, dict[int, dict]]:
    rows: dict[str, dict[int, dict]] = {}
    for d in dirs:
        for f in sorted(d.glob("results.*.jsonl")):
            cond = f.name[len("results."):-len(".jsonl")]
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    rows.setdefault(cond, {})[int(r["task_id"])] = r
    return rows


def score(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for m in METRICS:
        s = [r["judgements"][m]["score"] for r in rows if m in r.get("judgements", {})]
        if s:
            ok = [x for x in s if x is not None]
            k = sum(ok)
            lo, hi = wilson(k, len(ok))
            out[m] = {"k": k, "n": len(ok), "pct": round(100 * k / len(ok), 1) if ok else None,
                      "ci95": [round(lo, 1), round(hi, 1)], "unparsed": len(s) - len(ok)}
    return out


def fmt(e: dict, m: str) -> str:
    if m not in e:
        return "-"
    v = e[m]
    return f"{v['k']}/{v['n']} {v['pct']:.1f}% [{v['ci95'][0]:.0f},{v['ci95'][1]:.0f}]"


def table(title: str, res: dict) -> None:
    print(f"\n{title}")
    hdr = f"{'condition':<18}{'n':>4}  " + "  ".join(f"{m:>22}" for m in METRICS)
    print(hdr)
    for c in [c for c in ORDER if c in res] + [c for c in res if c not in ORDER]:
        e = res[c]
        print(f"{c:<18}{e['n']:>4}  " + "  ".join(f"{fmt(e, m):>22}" for m in METRICS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    rows = load(args.dirs)
    every = {c: score(list(v.values())) for c, v in rows.items()}
    common = set.intersection(*(set(v) for v in rows.values())) if rows else set()
    shared = {c: score([v[t] for t in sorted(common)]) for c, v in rows.items()}
    table("ALL TASKS EACH CONDITION RAN (n differs by condition)", every)
    table(f"SHARED TASKS ONLY (n={len(common)}: {sorted(common)})", shared)
    print("\ncells: k/n pct% [Wilson 95% CI]; application/target-recall/naive use the repo's "
          "context-aware judges, answer-only the optional answer-only judge.")
    if args.json:
        args.json.write_text(json.dumps({"every": every, "shared": shared,
                                         "shared_tasks": sorted(common)}, indent=2))


if __name__ == "__main__":
    main()
