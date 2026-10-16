#!/usr/bin/env python3
"""Driver for the two model-call rows. Key: $ANTHROPIC_API_KEY in this process's own environment
only (the launcher script sets it on the exec line); never printed, never written.

  python -m bench.modelrows.run gen --kind a --per-call 5 --calls 30 --out DATA --budget 4
  python -m bench.modelrows.run row-a --data DATA --offset 0 --limit 6 --out RUN --budget 2
  python -m bench.modelrows.run row-b --data DATA --offset 0 --limit 6 --out RUN --budget 2
  python -m bench.modelrows.run reread-a --data DATA --from-run RUN --out RUN2 --budget 8 --reader sonnet
  python -m bench.modelrows.run report --kind a RUN [RUN ...]

A run is refused up front when its projected cost for the units still to do would pass --budget; the
ledger stops it if the projection was wrong. The budget is checked before each call, so a stop can
overshoot by the calls already in flight. Each finished unit is one file under RUN/units, so a rerun
skips what is done: use ONE --out directory per configuration (reader, seed, variants), because a unit
is skipped by its name alone. The command exits 1 when a unit failed or a unit is still missing.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

from . import gen, orouter, rows
from .common import HAIKU, SONNET, BudgetExceeded, Client, Ledger, answer, cluster_bootstrap, mean

MODELS = {"haiku": HAIKU, "sonnet": SONNET}


def clients(args, ledger: Ledger, or_ledger: Ledger | None = None):
    """Composer and judge are Haiku 5.5 on the Anthropic ledger. ``--reader or:<model id>`` puts the
    reader on OpenRouter, on its own ledger and its own ``--or-budget``."""
    main = Client(HAIKU, ledger, args.budget)
    judge = Client(HAIKU, ledger, args.budget)
    name = getattr(args, "reader", "haiku")
    if name == "haiku":
        return main, judge, None
    if name.startswith("or:"):
        assert or_ledger is not None, "an OpenRouter reader needs an OpenRouter ledger"
        return main, judge, orouter.Client(name[3:], or_ledger, args.or_budget)
    return main, judge, Client(MODELS[name], ledger, args.budget)


def write_ledger(out: Path, ledger: Ledger, extra: dict | None = None, or_ledger: Ledger | None = None) -> None:
    body = {"ledger": ledger.summary(), **(extra or {})}
    if or_ledger is not None:
        body["openrouter_ledger"] = or_ledger.summary()
    (out / "ledger.json").write_text(json.dumps(body, indent=1))


def write_atomic(path: Path, text: str) -> None:
    """A unit file either exists whole or not at all: a killed run must not leave a truncated file
    that every later run skips."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def unit_id(sc: dict, variant: str | None) -> str:
    return f"{sc['id']}_{variant}" if variant else sc["id"]


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
    pending = [j for j in jobs if not (out / "units" / f"{unit_id(*j)}.json").exists()]
    if per_unit * len(pending) > args.budget:
        sys.exit(f"projected ${per_unit * len(pending):.2f} for {len(pending)} pending units passes budget ${args.budget:.2f}; not started")
    errors, lock = [], threading.Lock()
    t0 = time.time()

    def one(job):
        sc, v = job
        uid = unit_id(sc, v)
        f = out / "units" / f"{uid}.json"
        if f.exists():
            return
        try:
            if kind == "a":
                res = rows.run_unit_a(sc, v, out, main, judge, reader)
            else:
                res = rows.run_unit_b(sc, filler, args.seed, out, main, judge, reader)
            write_atomic(f, json.dumps(res))
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
    if errors or done < len(jobs):
        sys.exit(1)


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
    ledger, or_ledger = Ledger(), orouter.ORLedger()
    main, judge, reader = clients(args, ledger, or_ledger)
    assert reader is not None, "--reader sonnet or --reader or:<model> required"
    pending = [p for p in src if not (out / "units" / p.name).exists()]
    if args.est_unit_usd * len(pending) > args.budget:
        sys.exit(f"projected ${args.est_unit_usd * len(pending):.2f} for {len(pending)} pending units passes budget ${args.budget:.2f}; not started")
    errors = []

    def one(path):
        f = out / "units" / path.name
        if f.exists():
            return
        u = json.loads(path.read_text())
        try:
            r = (rows.reread_a if kind == "a" else rows.reread_b)(u, scs[u["scenario"]], reader, judge)
            write_atomic(f, json.dumps(r))
        except BudgetExceeded:
            raise
        except Exception:
            errors.append({"uid": path.stem, "trace": traceback.format_exc()[-1500:]})
    try:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(one, src))
    except BudgetExceeded as e:
        errors.append({"stopped": str(e)})
    write_ledger(out, ledger, {"errors": errors, "args": vars(args)}, or_ledger)
    print(f"reread {len(list((out / 'units').glob('*.json')))}/{len(src)} · errors {len(errors)} · "
          f"anthropic ${ledger.summary()['total']['usd']:.4f} · openrouter ${or_ledger.summary()['total']['usd']:.4f}")
    if errors or len(list((out / "units").glob("*.json"))) < len(src):
        sys.exit(1)


def cmd_pilot_a(args) -> None:
    """The ``none`` arm only, per reader model: no memory, 6 probes per scenario, judged by the
    Haiku 5.5 judge of the main runs. Prices each reader and shows which, if any, clears the pick rule."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scs = json.loads((Path(args.data) / "scenarios_a.json").read_text())[args.offset:args.offset + args.n]
    ledger = Ledger()
    judge = Client(HAIKU, ledger, args.budget)
    summary = {}
    for model in args.models.split(","):
        f = out / ("pilot_" + model.replace("/", "__") + ".json")
        or_ledger = orouter.ORLedger()
        reader = orouter.Client(model, or_ledger, args.or_budget)
        items = json.loads(f.read_text()) if f.exists() else []
        done = {(i["scenario"], i["probe"]) for i in items}
        jobs = [(sc, kind, pr) for sc in scs for kind, pr in rows._probes(sc) if (sc["id"], pr) not in done]
        lock = threading.Lock()

        def one(job):
            sc, kind, pr = job
            _, evidence = rows.sessions_a(sc, "light")
            ans = answer(reader, "", pr)
            j = rows._judge_a(judge, sc, evidence, ans)
            with lock:
                items.append({"scenario": sc["id"], "kind": kind, "probe": pr, "answer": ans, **j})
        try:
            with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
                list(ex.map(one, jobs))
        except BudgetExceeded as e:
            print(f"{model}: stopped: {e}")
        f.write_text(json.dumps(items))
        s = cluster_bootstrap(items, lambda i: i["score"], lambda i: i["scenario"])
        r = cluster_bootstrap(items, lambda i: i["raises_evidence"], lambda i: i["scenario"])
        usd = or_ledger.summary()["total"]["usd"]
        summary[model] = {"endorse": s, "raises_evidence": r, "openrouter_usd": usd, "answers": len(items),
                          "empty_answers": sum(1 for i in items if not i["answer"].strip()),
                          "unscored": sum(1 for i in items if i["score"] is None)}
        print(f"{model:45s} endorse {fmt(s)}  raises {fmt(r)}  empty {summary[model]['empty_answers']}  OR ${usd:.4f}")
    (out / "pilot_summary.json").write_text(json.dumps({"summary": summary, "anthropic_judge": ledger.summary()}, indent=1))
    print(f"anthropic judge ${ledger.summary()['total']['usd']:.4f}")


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
    pa = sub.add_parser("pilot-a")
    pa.add_argument("--data", required=True)
    pa.add_argument("--out", required=True)
    pa.add_argument("--models", required=True, help="comma-separated OpenRouter model ids")
    pa.add_argument("--n", type=int, default=30)
    pa.add_argument("--offset", type=int, default=3)
    pa.add_argument("--workers", type=int, default=8)
    pa.add_argument("--budget", type=float, required=True, help="Anthropic (judge) budget, USD")
    pa.add_argument("--or-budget", type=float, required=True, help="OpenRouter budget per model, USD")
    for name in ("row-a", "row-b", "reread-a", "reread-b"):
        r = sub.add_parser(name)
        readers = "haiku | sonnet | or:<OpenRouter model id>" if name.startswith("reread") else "haiku | sonnet"
        r.add_argument("--data", required=True)
        r.add_argument("--out", required=True)
        r.add_argument("--budget", type=float, required=True)
        r.add_argument("--offset", type=int, default=0)
        r.add_argument("--limit", type=int, default=6)
        r.add_argument("--workers", type=int, default=6)
        r.add_argument("--est-unit-usd", type=float, required=True, help="projected $ per unit; the run is refused if units x this passes --budget")
        r.add_argument("--variants", default="light,heavy")
        r.add_argument("--seed", type=int, default=0)
        r.add_argument("--reader", default="haiku", help=readers)
        r.add_argument("--or-budget", type=float, default=0.0, help="OpenRouter budget, USD (an or: reader needs it)")
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
    elif args.cmd == "pilot-a":
        cmd_pilot_a(args)
    elif args.cmd == "reread-a":
        cmd_reread(args, "a")
    elif args.cmd == "reread-b":
        cmd_reread(args, "b")
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
