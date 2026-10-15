#!/usr/bin/env python3
"""Driver for the two model-call rows. Key: $ANTHROPIC_API_KEY in this process's own environment
only (the launcher script sets it on the exec line); never printed, never written.

  python bench/modelrows/run.py gen --kind a --per-call 5 --calls 30 --out DATA --budget 4
  python bench/modelrows/run.py row-a --data DATA --offset 0 --limit 6 --out RUN --budget 2
  python bench/modelrows/run.py row-b --data DATA --offset 0 --limit 6 --out RUN --budget 2
  python bench/modelrows/run.py reread-a --data DATA --from-run RUN --out RUN2 --budget 8 --reader sonnet
  python bench/modelrows/run.py report --kind a RUN [RUN ...]

A run is refused up front when its projected cost would pass --budget; the ledger stops it if the
projection was wrong. Each finished unit is one file under RUN/units, so a rerun skips what is done.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import HAIKU, SONNET, BudgetExceeded, Client, Ledger  # noqa: E402
import gen  # noqa: E402
import rows  # noqa: E402

MODELS = {"haiku": HAIKU, "sonnet": SONNET}


def clients(args, ledger: Ledger):
    main = Client(HAIKU, ledger, args.budget)
    judge = Client(HAIKU, ledger, args.budget)
    reader = Client(MODELS[args.reader], ledger, args.budget) if getattr(args, "reader", "haiku") != "haiku" else None
    return main, judge, reader


def write_ledger(out: Path, ledger: Ledger, extra: dict | None = None) -> None:
    (out / "ledger.json").write_text(json.dumps({"ledger": ledger.summary(), **(extra or {})}, indent=1))


def run_units(args, kind: str) -> None:
    out = Path(args.out)
    (out / "units").mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    scs = json.loads((data / f"scenarios_{kind}.json").read_text())[args.offset:args.offset + args.limit]
    filler = json.loads((data / "filler.json").read_text()) if kind == "b" else None
    variants = args.variants.split(",") if kind == "a" else [None]
    jobs = [(sc, v) for sc in scs for v in variants]
    ledger = Ledger()
    main, judge, reader = clients(args, ledger)
    per_unit = args.est_unit_usd
    if per_unit * len(jobs) > args.budget:
        sys.exit(f"projected ${per_unit * len(jobs):.2f} for {len(jobs)} units passes budget ${args.budget:.2f}; not started")
    errors, lock = [], threading.Lock()
    t0 = time.time()

    def one(job):
        sc, v = job
        uid = f"{sc['id']}_{v}" if v else sc["id"]
        f = out / "units" / f"{uid}.json"
        if f.exists():
            return
        try:
            if kind == "a":
                res = rows.run_unit_a(sc, v, out, main, judge, reader)
            else:
                res = rows.run_unit_b(sc, filler, args.seed, out, main, judge, reader)
            f.write_text(json.dumps(res))
        except BudgetExceeded:
            raise
        except Exception:
            with lock:
                errors.append({"uid": uid, "trace": traceback.format_exc()[-1500:]})

    try:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(one, jobs))
    except BudgetExceeded as e:
        errors.append({"stopped": str(e)})
    done = len(list((out / "units").glob("*.json")))
    write_ledger(out, ledger, {"errors": errors, "jobs": len(jobs), "units_on_disk": done,
                               "seconds": round(time.time() - t0), "args": vars(args)})
    tot = ledger.summary()["total"]
    print(f"units on disk {done}/{len(jobs)} · errors {len(errors)} · calls {tot['calls']} · ${tot['usd']:.4f}")


def cmd_gen(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger()
    c = Client(SONNET, ledger, args.budget)
    name = {"a": "scenarios_a.json", "b": "scenarios_b.json", "filler": "filler.json"}[args.kind]
    r = gen.generate(args.kind, c, args.per_call, args.calls, out / name)
    (out / f"gen_{args.kind}.ledger.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r))


def cmd_reread(args, kind: str) -> None:
    out = Path(args.out)
    (out / "units").mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    scs = {s["id"]: s for s in json.loads((data / f"scenarios_{kind}.json").read_text())}
    src = sorted((Path(args.from_run) / "units").glob("*.json"))[args.offset:args.offset + args.limit]
    ledger = Ledger()
    main, judge, reader = clients(args, ledger)
    assert reader is not None, "--reader sonnet required"
    if args.est_unit_usd * len(src) > args.budget:
        sys.exit(f"projected ${args.est_unit_usd * len(src):.2f} passes budget ${args.budget:.2f}; not started")
    errors = []

    def one(path):
        f = out / "units" / path.name
        if f.exists():
            return
        u = json.loads(path.read_text())
        try:
            r = (rows.reread_a if kind == "a" else rows.reread_b)(u, scs[u["scenario"]], reader, judge)
            f.write_text(json.dumps(r))
        except BudgetExceeded:
            raise
        except Exception:
            errors.append({"uid": path.stem, "trace": traceback.format_exc()[-1500:]})
    try:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(one, src))
    except BudgetExceeded as e:
        errors.append({"stopped": str(e)})
    write_ledger(out, ledger, {"errors": errors, "args": vars(args)})
    tot = ledger.summary()["total"]
    print(f"reread {len(list((out / 'units').glob('*.json')))}/{len(src)} · errors {len(errors)} · ${tot['usd']:.4f}")


def fmt(ci) -> str:
    if not ci:
        return "n/a"
    return f"{ci['mean']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}] (n {ci['n']}, {ci['clusters']} scen)"


def cmd_report(args) -> None:
    units = []
    for r in args.runs:
        for f in sorted((Path(r) / "units").glob("*.json")):
            units.append(json.loads(f.read_text()))
    s = (rows.summarize_a if args.kind == "a" else rows.summarize_b)(units)
    print(f"units {s['units']}")
    if args.kind == "a":
        for tag, o in s["by_kind"].items():
            print(f"[{tag}] endorsement (score 1 = goes along)")
            for arm, ci in o["arms"].items():
                print(f"  {arm:20s} {fmt(ci)}   raises_evidence {fmt(o['raises_evidence'][arm])}")
            for k, ci in o["diffs"].items():
                print(f"  diff {k:44s} {fmt(ci)}")
            for k, ci in o["raises_diffs"].items():
                print(f"  raises_evidence diff {k:36s} {fmt(ci)}")
        rest = ("state", "by_variant", "unscored_answers", "wrap")
    else:
        for arm, ci in s["arms"].items():
            print(f"  {arm:20s} {fmt(ci)}  ctx chars {ci['mean_context_chars']:.0f}")
        for k, ci in s["diffs"].items():
            print(f"  diff {k:44s} {fmt(ci)}")
        rest = ("recall_fact_retrieved", "unscored_answers", "wrap")
    for k in rest:
        if k in s:
            print(f"  {k}: {json.dumps(s[k], default=str)}")
    if args.json:
        Path(args.json).write_text(json.dumps(s, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--kind", choices=("a", "b", "filler"), required=True)
    g.add_argument("--per-call", type=int, default=5)
    g.add_argument("--calls", type=int, default=20)
    g.add_argument("--out", required=True)
    g.add_argument("--budget", type=float, required=True)
    for name in ("row-a", "row-b", "reread-a", "reread-b"):
        r = sub.add_parser(name)
        r.add_argument("--data", required=True)
        r.add_argument("--out", required=True)
        r.add_argument("--budget", type=float, required=True)
        r.add_argument("--offset", type=int, default=0)
        r.add_argument("--limit", type=int, default=6)
        r.add_argument("--workers", type=int, default=6)
        r.add_argument("--est-unit-usd", type=float, required=True, help="projected $ per unit; the run is refused if units x this passes --budget")
        r.add_argument("--variants", default="light,heavy")
        r.add_argument("--seed", type=int, default=0)
        r.add_argument("--reader", choices=("haiku", "sonnet"), default="haiku")
        r.add_argument("--from-run")
    rp = sub.add_parser("report")
    rp.add_argument("--kind", choices=("a", "b"), required=True)
    rp.add_argument("--json")
    rp.add_argument("runs", nargs="+")
    args = ap.parse_args()
    if args.cmd == "gen":
        cmd_gen(args)
    elif args.cmd == "row-a":
        run_units(args, "a")
    elif args.cmd == "row-b":
        run_units(args, "b")
    elif args.cmd == "reread-a":
        cmd_reread(args, "a")
    elif args.cmd == "reread-b":
        cmd_reread(args, "b")
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
