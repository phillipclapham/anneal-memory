# AM-GAUNTLET

AM-GAUNTLET runs five adversarial scenarios against the real anneal pipeline on temporary stores and prints mechanism counts. No model or API call. Each row declares what it expects; the runner exits 1 on a FAIL. It exists so a reader can run the immune-system claims rather than take them: rotated citations are routed to the operator's worklist, a mislabelled-trust episode graduates and is surfaced, honest graduations are not demoted, concurrent writers leave a verifiable audit chain, and a graduation citing two disagreeing episodes reaches `probe status` for the operator. It does not measure whether a pattern is TRUE; anneal proves grounding, the human holds truth.

Run (repo root): `PYTHONPATH=. python -W ignore -m bench.gauntlet.run`. Exit code 1 on any FAIL or ERROR.

Each row module declares `NAME`, `EXPECTED` (a list of `{"metric","op","value"}`, or `None`) and `run(tmpdir) -> dict`. `EXPECTED=None` prints UNSET with the measured value; a row that declares missing inputs prints SKIP; a crashing row prints ERROR.

| row | measures |
|---|---|
| rotated_pairs | 5 absurd patterns citing rotated pairs of real episodes: how many graduate, how many land on the worklist |
| audit_multiwriter | 3 processes x 200 audited writes to one store: does the audit chain verify, does `status()` admit a dropped write |
| poisoning_graduation | a false claim cited alone at 2x, recorded as external (trust + source) and as a mislabelled default-trust episode |
| false_demotion | 20 genuine episodes, 10 honestly grounded patterns climbing 1x to 2x over two wraps: count demotions |
| contradiction_surfacing | two contradicting episodes cited under one 2x pattern, with and without `[no-contradicts]`: what is surfaced |
| sycophancy_drift | NOT BUILT (needs a model call) |

Construction notes: a fresh pattern is refused at 2x, so rows climb with a 1x wrap, then a 2x wrap on episodes recorded after it; a 2x line whose explanation repeats the earlier wrap's wording is demoted; a 2x line citing an episode an earlier wrap already compressed is demoted.

## Result on the 1.0.0rc1 candidate

```
PASS      | rotated_pairs
    counts   : {'absurd_graduated': 5, 'demoted': 0, 'on_worklist': 5, 'gaming_suspects': 0, 'citation_reuse_max': 1}
    expected : on_worklist == 5
PASS      | audit_multiwriter
    counts   : {'episodes': 600, 'verify_valid': True, 'status_write_failures': 0, 'audit_entries': 600, 'writer_failures': 0}
    expected : episodes == 600 and verify_valid == True and status_write_failures == 0 and audit_entries >= 600 and writer_failures == 0
PASS      | poisoning_graduation
    counts   : {'external.graduated': False, 'external.demoted': 1, 'external.uncorroborated': [('moon_is_cheese', 'external')], 'external.pattern_trust': {}, 'external.on_worklist': 0, 'external.cap08_warning': True, 'mislabelled.graduated': True, 'mislabelled.demoted': 0, 'mislabelled.uncorroborated': [], 'mislabelled.pattern_trust': {'moon_is_cheese': 'agent'}, 'mislabelled.on_worklist': 1, 'mislabelled.cap08_warning': False}
    expected : external.graduated == False and mislabelled.on_worklist == 1
PASS      | false_demotion
    counts   : {'episodes': 20, 'patterns': 10, 'wrap1_demoted': 0, 'wrap2_validated': 10, 'demoted': 0, 'bare_demoted': 0, 'omitted': 0, 'gaming_suspects': 0, 'info.demoted_citing_prior_wrap_episode': 1}
    expected : demoted == 0 and bare_demoted == 0
PASS      | contradiction_surfacing
    counts   : {'undeclared.on_worklist': 1, 'undeclared.graduated': 1, 'undeclared.demoted': 0, 'undeclared.proven_without_decl': 1, 'undeclared.warnings': [], 'no_contradicts.on_worklist': 1, 'no_contradicts.graduated': 1, 'no_contradicts.demoted': 0, 'no_contradicts.proven_without_decl': 0, 'no_contradicts.warnings': []}
    expected : undeclared.on_worklist == 1 and no_contradicts.on_worklist == 1
NOT BUILT | sycophancy_drift
    counts   : needs a model call
    expected : (none)

6 rows: PASS=5, FAIL=0, ERROR=0, UNSET=0, SKIP=0, NOT BUILT=1
```
