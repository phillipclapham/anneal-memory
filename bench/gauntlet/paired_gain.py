"""Row: paired gain (memory-on versus memory-off, one reader).

A fact is stated once, early, in an 80-session run of unrelated exchanges; a later request depends on
the fact without naming it. Six arms answer that request: no memory, the fact exchange alone (the
ceiling), a full dump of all 80 sessions, anneal's retrieval path, anneal's wrap-composed continuity
state, and the paper's rolling memory. A judge scores whether each answer applies the fact.

This is the one gauntlet row that calls a model, so it never runs by accident: it needs
ANTHROPIC_API_KEY in the environment AND ANNEAL_BENCH_PAID=1. Without both it prints SKIP and
spends nothing. Composer, reader and judge are one model (bench.modelrows.common.HAIKU). The scenarios
are synthetic and ship with the repo (bench/modelrows/data); no outside dataset is used.
`ANNEAL_BENCH_UNITS` sizes the slice and `ANNEAL_BENCH_BUDGET_USD` caps it; the run refuses to start
when units x EST_UNIT_USD passes the cap. EXPECTED is unset on
purpose: the measured values are model output, and a slice this small cannot support a pass or fail.
"""
import json
import os
import concurrent.futures as cf
from pathlib import Path

NAME = "paired_gain"
EXPECTED = None

DATA = Path(__file__).resolve().parents[1] / "modelrows" / "data"
FIRST_SCENARIO = 6      # b000-b005 were the pricing pilots in the original runs
DEFAULT_UNITS = 4
DEFAULT_BUDGET_USD = 2.0
EST_UNIT_USD = 0.08     # per unit; the 2026-10-15 main runs measured 0.062 (their ledger.json), with headroom


def requires():
    missing = []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if os.environ.get("ANNEAL_BENCH_PAID") != "1":
        missing.append("ANNEAL_BENCH_PAID=1 (this row spends API credit)")
    return missing


def summarize(units: list[dict]) -> dict:
    """Plain means per arm over the units that ran. No interval: a few units cannot support one."""
    from bench.modelrows.rows import B_ARMS

    def mean_of(arm, key):
        xs = [u["arms"][arm][key] for u in units if u["arms"][arm][key] is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    out = {"units": len(units)}
    for arm in B_ARMS:
        out[f"application.{arm}"] = mean_of(arm, "score")
    out["context_chars.anneal-continuity"] = round(mean_of("anneal-continuity", "context_chars"))
    out["context_chars.full-dump"] = round(mean_of("full-dump", "context_chars"))
    a, b = out["application.anneal-continuity"], out["application.none"]
    out["gain.continuity_minus_none"] = None if a is None or b is None else round(a - b, 3)
    out["recall_fact_retrieved"] = round(
        sum(1 for u in units if u["diag"]["recall"]["fact_retrieved"]) / len(units), 3)
    out["unscored_answers"] = sum(1 for u in units for arm in B_ARMS if u["arms"][arm]["score"] is None)
    out["wraps_saved"] = sum(u["wrap"]["anneal"]["saved"] for u in units)
    out["wraps_cancelled"] = sum(u["wrap"]["anneal"]["cancelled"] for u in units)
    return out


def run(tmpdir: str) -> dict:
    from bench.modelrows import rows
    from bench.modelrows.common import HAIKU, BudgetExceeded, Client, Ledger

    n = int(os.environ.get("ANNEAL_BENCH_UNITS", DEFAULT_UNITS))
    budget = float(os.environ.get("ANNEAL_BENCH_BUDGET_USD", DEFAULT_BUDGET_USD))
    if n * EST_UNIT_USD > budget:
        raise RuntimeError(f"{n} units project to ${n * EST_UNIT_USD:.2f}, over the ${budget:.2f} cap; "
                           "raise ANNEAL_BENCH_BUDGET_USD or lower ANNEAL_BENCH_UNITS")
    scenarios = json.loads((DATA / "scenarios_b.json").read_text())[FIRST_SCENARIO:FIRST_SCENARIO + n]
    if n < 1 or not scenarios:
        raise RuntimeError(f"ANNEAL_BENCH_UNITS={n} selects no scenarios")
    filler = json.loads((DATA / "filler.json").read_text())
    ledger = Ledger()
    client, judge = Client(HAIKU, ledger, budget), Client(HAIKU, ledger, budget)
    work = Path(tmpdir)
    units, errors, stopped = [], [], None

    def one(sc):
        return rows.run_unit_b(sc, filler, 0, work, client, judge)

    with cf.ThreadPoolExecutor(max_workers=min(len(scenarios), 4)) as ex:
        futures = [ex.submit(one, sc) for sc in scenarios]
        for f in futures:
            try:
                units.append(f.result())
            except BudgetExceeded as e:
                stopped = str(e)
            except Exception as e:   # a unit that crashed is counted, never hidden
                errors.append(f"{type(e).__name__}: {e}"[:200])
    if not units:
        raise RuntimeError(f"no unit finished; errors={errors} stopped={stopped}")
    out = summarize(units)
    out["scenarios_requested"] = len(scenarios)
    out["unit_errors"] = len(errors)
    if stopped:
        out["stopped"] = stopped
    out["spend_usd"] = round(ledger.summary()["total"]["usd"], 4)
    return out
