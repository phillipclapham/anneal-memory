"""AM-GAUNTLET runner: python -m bench.gauntlet.run  (from the repo root, PYTHONPATH=.)

Each row module declares NAME, EXPECTED (list of {"metric","op","value"} or None) and
run(tmpdir) -> dict of counts. No row calls a model unless the caller has opted in: paired_gain is
the one row that does, and it prints SKIP without its API key and its opt-in. A row with
EXPECTED=None prints UNSET with its measured value; a row that declares missing inputs prints SKIP
with the path; a row that raises prints ERROR with the exception. Nothing fails silently."""
import importlib
import operator
import sys
import tempfile
import traceback

ROWS = ["rotated_pairs", "audit_multiwriter", "poisoning_graduation", "false_demotion",
        "contradiction_surfacing", "paired_gain"]
NOT_BUILT = [("sycophancy_drift", "needs a model call")]
OPS = {"==": operator.eq, "!=": operator.ne, "<=": operator.le, ">=": operator.ge,
       "<": operator.lt, ">": operator.gt}


def evaluate(expected, counts):
    if expected is None:
        return "UNSET"
    for e in expected:
        try:
            if e["metric"] not in counts or not OPS[e["op"]](counts[e["metric"]], e["value"]):
                return "FAIL"
        except (TypeError, KeyError):  # a None metric or an unknown op is a failed expectation
            return "FAIL"
    return "PASS"


def fmt_expected(expected):
    if expected is None:
        return "(none)"
    return " and ".join(f'{e["metric"]} {e["op"]} {e["value"]}' for e in expected)


def main() -> int:
    results = []
    for name in ROWS:
        try:
            mod = importlib.import_module(f"bench.gauntlet.{name}")
            missing = mod.requires() if hasattr(mod, "requires") else None
            if missing:
                results.append((name, f"missing: {missing}", fmt_expected(mod.EXPECTED), "SKIP"))
                continue
            with tempfile.TemporaryDirectory() as td:
                counts = mod.run(td)
            results.append((name, counts, fmt_expected(mod.EXPECTED), evaluate(mod.EXPECTED, counts)))
        except Exception as exc:  # a crashed row (or one that cannot import) is a result, not silence
            traceback.print_exc(file=sys.stderr)
            results.append((name, f"{type(exc).__name__}: {exc}", "(none)", "ERROR"))
    for name, why in NOT_BUILT:
        results.append((name, why, "(none)", "NOT BUILT"))
    for name, counts, exp, verdict in results:
        print(f"{verdict:9} | {name}")
        print(f"    counts   : {counts}")
        print(f"    expected : {exp}")
    bad = [r for r in results if r[3] in ("FAIL", "ERROR")]
    print(f"\n{len(results)} rows: " + ", ".join(
        f"{v}={sum(1 for r in results if r[3] == v)}"
        for v in ("PASS", "FAIL", "ERROR", "UNSET", "SKIP", "NOT BUILT")))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
