# AM-GAUNTLET

AM-GAUNTLET runs scenarios against the real anneal code on temporary stores and prints counts. One row, `audit_multiwriter`, makes no model or API call. One row, `paired_gain`, does, and it only runs when you give it a key and say so (see below); without them it prints SKIP and spends nothing. Each row declares what it expects; the runner exits 1 on a FAIL.

The 3.x gauntlet also had four rows that exercised graduation (`rotated_pairs`, `poisoning_graduation`, `false_demotion`, `contradiction_surfacing`). 4.0 has no graduation, so those rows are gone with it.

Run (repo root): `PYTHONPATH=. python -W ignore -m bench.gauntlet.run`. Exit code 1 on any FAIL or ERROR.

Each row module declares `NAME`, `EXPECTED` (a list of `{"metric","op","value"}`, or `None`) and `run(tmpdir) -> dict`. `EXPECTED=None` prints UNSET with the measured value; a row that declares missing inputs prints SKIP; a crashing row prints ERROR.

| row | measures |
|---|---|
| audit_multiwriter | 3 processes x 200 audited writes to one store: does the audit chain verify, does `status()` admit a dropped write |
| paired_gain | a fact stated once among 80 unrelated sessions, then a request that depends on it without naming it: does the reader apply the fact with no memory, the full transcript, anneal's search path, anneal's wrap-composed memory file, and a rolling memory (needs `ANTHROPIC_API_KEY` and `ANNEAL_BENCH_PAID=1`; SKIP otherwise) |
| sycophancy_drift | NOT BUILT (needs a model call) |

## Result on 4.0.0

```
PASS      | audit_multiwriter
    counts   : {'episodes': 600, 'verify_valid': True, 'status_write_failures': 0, 'audit_entries': 600, 'writer_failures': 0}
    expected : episodes == 600 and verify_valid == True and status_write_failures == 0 and audit_entries >= 600 and writer_failures == 0
SKIP      | paired_gain
    counts   : missing: ['ANTHROPIC_API_KEY', 'ANNEAL_BENCH_PAID=1 (this row spends API credit)']
    expected : (none)
NOT BUILT | sycophancy_drift
    counts   : needs a model call
    expected : (none)

3 rows: PASS=1, FAIL=0, ERROR=0, UNSET=0, SKIP=1, NOT BUILT=1
```

## The model-call row

`paired_gain` is the only row that calls a model. It replays a synthetic 80-session history (scenarios and filler ship in `bench/modelrows/data`; no outside dataset) into scratch stores, composes anneal's memory file with the real wrap loop (`prepare_wrap`, a model run of the step's prompt, `validated_save_continuity`), and asks a reader a question that depends on a fact stated once, early. Composer, reader and judge are `claude-haiku-5-5`.

Run it: `ANTHROPIC_API_KEY=... ANNEAL_BENCH_PAID=1 PYTHONPATH=. python -W ignore -m bench.gauntlet.run`. It runs 4 scenarios by default and refuses to start if the projected cost passes `ANNEAL_BENCH_BUDGET_USD` (default $2). `ANNEAL_BENCH_UNITS` sizes the slice. Its expectation is unset on purpose: the output is model behaviour, and a slice this small supports neither pass nor fail. `python -m bench.modelrows.run row-b --data bench/modelrows/data --out RUN1 --offset 6 --limit 68 --budget 7 --est-unit-usd 0.08` runs a larger chunk, and `python -m bench.modelrows.run report --kind b RUN1` prints the arms and the paired intervals.

**The figures below were measured on the 3.x wrap, not on 4.0.** They were taken on 2026-10-15 with the 3.x consolidation (a package with graduation instructions), 136 scenarios in two chunks, about $8.90 of API credit. The 4.0 wrap composes the memory file with one fixed prompt, so its numbers have not been measured and the run has to be repeated before any of these is claimed for 4.0.

On those 136 synthetic scenarios (95% intervals are cluster bootstraps by scenario), a reader with no memory applied the planted fact 15% of the time [10%, 21%]. With the 3.x composed state it applied it 82% of the time [76%, 88%], at about 3,100 characters of context. With the full transcript of all 80 sessions it applied it 81% [74%, 87%], at about 34,000 characters. With the fact exchange alone (the ceiling) it applied it 84%. A rolling memory (a model rewrites one memory file, capped at 200 lines, after every session) scored 78% at about 600 characters. The paired differences (+1.5 points against the full dump, +4.4 against the rolling memory) had intervals that include zero. Reader and judge are one model family on synthetic data written by one model, with the fact at one position (the 17th of 80 sessions) and one horizon (80 sessions).

**A named limitation.** anneal's 3.x retrieval path (`retrieve_relevant`, lexical; 4.0 removed it, and the arm now calls `search_episodes`, which has not been measured here) sat at the no-memory level on these queries (21% against 15%, difference +5 points, interval [-1, +12]) because each query is written to share no vocabulary with the fact. It found the planted fact in 9% of scenarios. Requests that name their subject are the other regime, and were not run.
