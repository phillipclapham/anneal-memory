# Slice-C Gain Instrument — the design-nail (2026-06-17)

> **Canonical home** for the AM-LINKGATE-DECAY Slice-C GO/NO-GO *gain* methodology.
> Turns the oracle from a discrete edge-quality gate into a continuous gain-vs-baseline
> instrument. Pre-work for the Slice-C build; locked 2026-06-17 (Phill + flow), grounded
> in daemon's overnight retrieval-receipt design + CL-Bench + the existing Step-C infra.
> Build-order at the bottom is the spec for the build-start session.

---

## 1. The problem this fixes

`scripts/pattern_graph_oracle.py` measures three signals — **graph health** (links / strengths / formation-mix), the **echo-chamber canary** (concentration trend), **neocortex co-mention** (do top pairs co-appear in the Patterns working set). **All three are graph-*intrinsic*: "do the edges look meaningful?"** That is exactly what CL-Bench (arXiv:2606.05661) shows is *insufficient* — "use-shaped associations exist is NOT evidence of learning unless **outcome gain** appears." Naive in-context-learning (dump-everything) beat dedicated memory systems in their setup; an intrinsically-pretty graph can change recall outcomes by **zero** and still pass the current oracle. Proxy-not-contract, at the validation layer.

**Concrete blind spot (today's oracle run, 2026-06-17):** 14 of 15 top pairs are NOT co-mentioned in the neocortex. The oracle flags this "inspect — noise or a real cross-domain discovery." That ambiguity is unresolvable intrinsically. The gain instrument resolves it: *when those 14 un-co-mentioned pairs are surfaced via graph-consuming recall, do they demonstrably HELP on a real query, or not?* Discovery vs noise becomes a measurement, not a guess.

## 2. The reframe

**GO/NO-GO = intrinsic hygiene (NECESSARY) AND gain-vs-baseline (SUFFICIENT).** The three intrinsic signals stay — as **hygiene gates** (an echo-canary climbing monotonically, or retrieval-health excluding events, is still a hard NO). They are no longer the GO; they are the floor. The GO is **measured outcome gain** of graph-consuming recall over the best baseline.

**June-26 dissolves.** It was a date because the gate was discrete ("look at the graph ~Jun-26, decide"). A gain instrument is **continuous**: receipts accumulate, the harness runs on demand, GO = a *threshold crossed on a running gauge*. Slice-C builds when the gain shows up — measured while building, not gated at the end. This is the "instrument-while-building" reframe made literal, and it is the **Protocol-Memory antidote**: kill/keep/promote is never a blind binary again.

## 3. The unit — the retrieval receipt (daemon, extends the existing spool)

Production recall already logs **co-surface events** to `state/pattern_cosurface_spool.jsonl` (`recall_injection_hook.log_co_surface_event`). The receipt **extends** that record from "which patterns co-surfaced" to the full provenance-and-outcome unit:

```
RetrievalReceipt {
  query_id / context_id            # the prompt/context that fired recall
  query_text (capped 16KB) + query_truncated  # the VERBATIM query — §9.2 replay NEEDS it (can't replay a query you didn't keep; codex L3 2026-06-17); lone-surrogate-sanitized
  exposed: [{pattern, rank, score, source: keyword|evidence_edge|graph_hop}]
  graph_version + high_water_mark  # projection version (blue/green-safe replay)
  cited_used: [pattern, ...]       # the subset actually used downstream (wrap citation / response reference)
  downstream_claim_or_action_id    # what the recall fed
  outcome_signal                   # labeled relevance/usefulness (probe) OR observed-use proxy
  provenance_spans                 # which source episodes grounded the surfaced patterns (influence trace)
}
```

**Negative evidence is first-class** (daemon): a *surfaced-but-unused* exposure and a *used-but-bad-outcome* are negative signals, not neutral. This is what lets associations strengthen by **useful downstream influence**, not mere exposure — and it is the same data the gain measure reads.

## 4. The measurement — offline A/B replay (the echo-chamber-safe resolution)

**The chicken-and-egg:** to measure whether *graph-consuming* recall helps, recall must consume the graph — but production recall is graph-independent *by construction* (the Slice-B echo-chamber guard; a graph-mediated co-surface reinforces at 0). Consuming the graph in production before it's proven would be the echo chamber forming.

**Resolution: measure gain OFFLINE, in a sandbox, before production consumes anything.**

- **Production recall stays graph-independent** (unchanged; echo-chamber guard intact) and logs full receipts. The receipt corpus is the replay set.
- A **measurement harness** replays the receipt corpus' queries through **N recall paths in parallel** (one is graph-consuming, the rest are baselines) and scores each path's surfaced set against the outcome metric.
- **Gain = graph-consuming path's outcome score − best baseline's outcome score**, on the same queries. Positive, noise-clearing margin = the graph earns Slice-C; ≤ 0 = the graph is ornamental and Slice-C is NO-GO (the CL-Bench failure mode, *caught* instead of shipped).

The graph-consuming recall path is therefore **built in the harness sandbox first** (measured), and only **promoted to production (Slice-C proper) when gain clears** — instrument-while-building, with the echo-chamber guard never relaxed pre-GO.

## 5. The baselines (what graph-consuming must beat)

| Baseline | What it isolates |
|---|---|
| **(a) no-recall / stateless** | the floor — does recall help at all |
| **(b) keyword-only** (pre-Hebbian backend) | the lexical baseline (flow's conceptual corpus already fails this — Step-C) |
| **(c) direct evidence-edge associative** (current 0.8.0 backend, NO graph hop) | **the load-bearing one** — proves the *graph HOP* adds value, not just the evidence edges already shipped |
| **(d) full-context / naive-ICL** (dump the whole working set) | the CL-Bench challenger — proves *selective* recall beats dump-everything (the cost-parity argument: selectivity is the point) |

The two that matter most are **(c)** and **(d)**: beating (c) proves the *new* thing (the pattern→pattern hop) earns its complexity; beating (d) proves the *whole memory thesis* against the naive baseline CL-Bench wielded.

## 6. The outcome metric (minimum-viable → gold)

- **Minimum-viable (gates Slice-C):** *labeled relevance/usefulness* on a small probe set — the Step-C `relevant-recall` measure (fired-when-should / should-have-fired), extended from keyword to associative, judged by a Phill-labeled sample + a cheap LLM-judge for volume. This already has partial infra (the retired `crystal_hitrate.py` analyzer + the 06-08 recall-lift method: boot-prompt exclusion, score-banding, spurious-dominant-pattern taxonomy, evidence-lift-vs-backend-lift decomposition). The AM-GAUNTLET recall-quality row (`spore-048`) is its permanent home.
- **Gold-standard (follow-on, not gating):** *task-probe behavior* — does the recalled pattern improve **procedure / gotcha / state-mutation / premise** behavior on task-like probes (STATE-Bench pass^5 / LongMemEval-V2 axes; daemon `31b9eaf0`). Expensive (needs a probe harness + evaluator); deferred behind minimum-viable.

**Bias correction is structural, not a patch (Joachims):** receipts are exposure-biased — a surfaced pattern is likelier to be "used" *because it was shown*. Observational "used-rate" would reward whatever the graph surfaces, circularly. The **controlled A/B** (same queries, different recall paths, a *labeled* outcome) dodges exposure bias by construction — which is the whole reason the measure is A/B replay and not an observational use-rate.

## 7. GO / NO-GO criteria (reframed)

**Hygiene gates (NECESSARY — any fail = hard NO):**
- echo-canary concentration trend non-monotonic (no echo chamber forming);
- retrievable-links healthy + retrieval-health drain-log clean (0 excluded-by-error) over the measured window, scoped `--drain-since <shadow-start>` (the existing 4bd56c8 hardening).

**Gain gate (SUFFICIENT — the new arm):**
- graph-consuming recall shows **positive labeled-usefulness gain over the best baseline** (esp. (c) direct-evidence and (d) full-context) on the probe set, by a margin that clears the labeling noise floor;
- the gain is **not** an artifact of the self-match / boot-prompt inflation the Step-C read already exposed (exclude consult-boot self-matches from the denominator).

**NO-GO is a real, expected outcome:** gain ≤ baseline means the graph hop is ornamental *for this corpus regime* → ship keyword/evidence-edge as the adequate default, do NOT wire the hop. That is the instrument working, not failing.

## 8. What this is NOT (boundaries / anti-scope-creep)

- **Does NOT relax production recall's graph-independence pre-GO** (the echo-chamber guard is the whole integrity of the shadow phase).
- **Does NOT reimplement STATE-Bench** — minimum-viable labeled-relevance first; the task-probe gold is a deliberate follow-on.
- **Is NOT gated on a date** — a threshold on a running instrument.
- **Does NOT change any shipped behavior** — the receipt is an additive log; the harness is offline; production is untouched until GO.

## 9. Build order (the build-start spec)

1. **Extend the co-surface spool → full retrieval receipts** (`recall_injection_hook` writes the receipt fields; additive to the existing `pattern_cosurface_spool.jsonl` path; provenance-spans + cited/used populated post-hoc from the wrap).
2. **The A/B replay harness** — replays the receipt-corpus queries through the 4 baseline paths + the graph-consuming sandbox path (the latter = the un-shipped Slice-C recall, run offline only).
3. **The labeled probe set** — small, Phill-labeled core + LLM-judge for volume; the AM-GAUNTLET row (`spore-048`) is its home; reuse the Step-C analyzer's transferable parts.
4. **The oracle gain-arm** — `pattern_graph_oracle.py` gains a gain section that reads the harness output and reports gain-over-best-baseline alongside the three hygiene signals; GO criteria wired per §7.
5. **Promotion path** — when the gain gate clears, the sandbox graph-consuming path is promoted to production recall (Slice-C proper), with the provenance gate binding (graph-mediated co-surfaces reinforce at 0 until they've proven downstream-useful — the influence half).

## 10. Couplings

- **`spore-085`** (the GO/NO-GO gate) — reframed: hygiene AND gain, not a date. Points here.
- **`spore-048`** (AM-GAUNTLET recall-quality row) — the probe set + labeled-relevance measure live here.
- **`scripts/pattern_graph_oracle.py`** — gains the gain-arm (§9.4); the 3 intrinsic signals become hygiene.
- **daemon's retrieval-receipt fold** (`anneal/next.md`, 2026-06-17; ids 31b9eaf0 / 103febd2 / bc7f2370 / 30285a29) — the source design this operationalizes.
- **anneal Slice-C build** — this is its measurement spine; the build is gated on the gain gate, measured continuously.

---

## 11. Build progress + surfaced upstream dependencies (2026-06-17)

**Step 1 (receipts) — SHIPPED to main, additive, full 4-layer apparatus clean (2026-06-17).** `recall_injection_hook.log_co_surface_event` extended from the co-surface event → the §3 receipt shape; recall-time fields populated (`query_id`/`context_id`, **`query_text` (capped 16KB) + `query_truncated`**, `exposed[]` with rank/score/source, `outcome_signal` scaffold), post-hoc fields (`cited_used` / `downstream_claim_or_action_id` / `provenance_spans`) scaffolded-nullable for the wrap side. `RECEIPT_VERSION = 1`.

- **Apparatus (full 4-layer):** L1 session-review (APPROVE-with-notes → empty-name leak in `exposed` filtered, contiguous `rank` among kept) · L2 back-compat (SAFE — every consumer reads only Slice-B keys via `.get()`, verified at each site) · **L3 codex caught 2 the Claude layers missed** (`cross_substrate_review_codex_nonreplaceable`): **HIGH** — receipts had no replayable query (only an irreversible `prompt_hash`) so §9.2 replay was impossible → **added `query_text`, resolving the §3↔§4 inconsistency**; **MED** — `assoc_degraded` windows serve keyword results but `exposed[].source` was `None` → now labels `assoc_degraded` as `keyword` (only the live `assoc_hop` stays `None`). Codex round 2: **LOW** — a lone-surrogate prompt would make the `ensure_ascii=False` write raise and silently drop the whole record → `query_text` sanitized via `encode('utf-8','ignore')`. · **L4** — producer behavior (no-drop / sanitize / truncation / `exposed`) + a real anneal `Store.drain_co_surface_events` consuming the enriched records (`events_applied=2, pairs_formed=1`, no error) + 21/21 regression on final code.

**Two upstream anneal/library deps surfaced (`spore-104`) — gate steps 2+, NOT step 1:**
1. **anneal pattern-graph PROJECTION-VERSION / HIGH-WATER-MARK primitive (anneal-side; the bigger one).** Receipts log `graph_version`/`high_water_mark` as **null** because anneal exposes no such primitive — only crystal-file `schema_version` exists. This is the unbuilt CQRS "remainder" already noted in `next.md` (daemon `d8b9834c`: "explicit store path + high-water mark + projection version + shadow↔active switch, rebuilt blue/green"). **Gates §9.2's blue/green-safe A/B replay harness** — it can't pin which graph version produced a receipt without it. → an anneal build item, upstream of Slice-C step 2.
   - **✅ SHIPPED 2026-06-21 (THIN — pin-not-rebuild).** anneal `pattern_graph_projection_meta` singleton + `Store.pattern_graph_projection_meta()` + the flow receipt stamp. `high_water_mark` = a monotonic revision counter bumped on every committed graph-edge mutation (seed/drain/rename/sever/gc); `projection_version` = constant 1 (the future blue/green rebuild bumps it). It STAMPS state (pins a receipt to a snapshot), it does NOT rebuild — the graph isn't rebuildable-from-log today (raw co-surface events truncated post-drain, decay is wall-calendar, GC deletes), so as-of reconstruction stays deferred.
   - **▶ CONSUMER CONTRACT the §9.2 harness MUST honor (L2):** (a) **bucket receipts on `(high_water_mark, query_date)`, NOT hwm alone** — lazy decay makes effective strength a pure function of `(stored, today)`, so two same-hwm receipts on different days saw different effective graphs; (b) **pin-not-rebuild is sufficient ONLY IF receipts-per-hwm-bucket gives A/B power** — hwm bumps ≈ every wrap, so a single pinned snapshot's usable sample may be just one day's receipts; **count receipts-per-hwm before trusting the thin path**, and if the bucket is too thin, build the as-of rebuild (the deferred remainder). Divergent store lineage + concept-generation boundaries are deliberately un-stamped (out of scope for the sovereign single-writer store + edge-state harness).
2. **`retrieve_relevant` per-pattern SOURCE tagging (library return-shape).** `exposed[].source` can't distinguish keyword vs evidence-edge per-pattern for an associative basis (the backend doesn't tag it; `graph_hop` is structurally impossible in shadow). **Gates §5 baseline isolation** (separating baseline (c) direct-evidence-edge from the graph hop). → a `retrieve_relevant` return-shape change, upstream of the baselines.

## 12. The recall-PRECISION blocker — the regime read, 2026-06-21 (measure BEFORE building the consumer)

Running the gain-regime read on the **real receipt corpus** (the step-1 `query_text`, 39 receipts — the ground-truth query distribution `spore-085` said the synthetic-inconclusive differentiation needed) surfaced that **the underlying recall is too NOISY to measure a graph gain against.** The instrument caught a noise-amplification trap — exactly its §2 job (the Step-C / `positioning_is_architecture` discipline: measure the retrieval regime before committing the consumer).

- **Recall is net-noise.** Production recall surfaces a fixed promiscuous ~6-pattern set (positioning_is_architecture / tasks_are_not_memory / watermark / anneal_FOUR_LAYER / sourdough / compression) on NEARLY EVERY query, incl. poker chatter ("scary flop top pair" → `positioning_is_architecture`@3.5) and `<task-notification>` blobs. The Slice-B graph's strongest edges (positioning↔watermark 3.16, tasks↔watermark 2.97) ARE these promiscuous co-fires = the §7 / `spore-085` spurious class, NOT discoveries. `exposed[].source` is `None` on all 49 live exposures (dep-2 confirmed binding).
- **Root cause = degree-bias in `_associative_patterns` (`retrieval.py`).** A surfaced pattern's score = its grounding EPISODE's keyword-reach, so high-evidence flow-meta patterns are reachable from too many episodes; the IDF / max-agg / hop guards are insufficient at flow's corpus. A hook-side score floor can't separate it — signal and noise scores overlap (2.5–5.5).
- **→ Slice C is NO-GO until recall precision improves.** The fix **CONVERGES with dep-2**: per-pattern source-tagging is BOTH the §5 baseline-isolation primitive AND the precision diagnostic (it lets recall filter the incidental episode-groundings that drive the promiscuity). **Reframed build order:** dep-2 source-tagging + a pattern-side **degree penalty** (`spore-141`) FIRST → re-measure the cleaned graph → THEN the §9.2 replay harness (steps 2+). The hygiene gates (echo-canary non-monotonic, retrieval-health clean) stay GREEN — this is a precision problem in the *input* to the graph, not an echo-chamber.
- **Flow-side HYGIENE half shipped** (`84409e8`, apparatus-clean): `recall_injection_hook` skips `<task-notification>` payloads before both injection and the co-surface log (they were poisoning the graph). The degree-bias half (`spore-141`) is the anneal-library build.

### ✅ RESOLVED — AM-RECALL-IDF shipped 2026-06-21 (anneal `57511eb` 0.9.3 + flow `3ffaae7`, full 4-layer apparatus)

**The "degree-bias" diagnosis above was REFUTED by measurement before any code shipped** (`measured_ground_truth_beats_layout_theory` — and a clean example of why the precision read had to precede the consumer). Measuring flow's live crystal store: all 14 patterns cite ~4 evidence episodes (uniform degree — the promiscuous set is NOT higher-degree than the genuine set), and there are no hub episodes (max citation = 2, so the existing episode-IDF is inert). A pattern-side degree penalty would down-weight everything uniformly and change no ordering. **The true mechanism is TERM-FREQUENCY bias:** `_keyword_weight` was a corpus-BLIND length/punctuation proxy, so high-document-frequency PROCESS words (work 29% / load 21% / commit 17% / project 12% of the episode corpus) scored like distinctive terms; any prompt with a couple of scaffolding words floated whatever flow-meta episode they brushed.

**The shipped fix** (replaces the planned degree penalty): (1) **per-pattern `RelevantPattern.source`** (`keyword|evidence_edge|graph_hop`) — dep-2, the baseline-isolation primitive AND the diagnostic that confirmed everything surfaces via `evidence_edge` (the graph hop is dead); wired to the receipt `exposed[].source`. (2) **corpus-IDF keyword weight** (`_idf_weight`, exact df from `RecallResult.total_matching`, captured free in the candidate fetch) — engages > `IDF_MIN_CORPUS`=50, proxy/Store-free paths byte-unchanged. (3) **a √N distinctiveness ANCHOR** (`IDF_ANCHOR_WEIGHT`=0.5 ⟺ df ≤ √N, corpus-RELATIVE — L2 forced it: the floor alone re-leaked on FAT process prompts, structurally killed by requiring ≥1 genuinely-distinctive matched term). Calibrated on the real 53-query receipt corpus (`IDF_SCORE_THRESHOLD`=1.6). **Measured outcome:** surfacings 115→~16, process/noise prompts → empty, genuine queries (reminders→`tasks_are_not_memory`) retained.

**▶ The recall input to the Slice-B graph is now clean.** Next return-to-anneal: re-run the gain-regime read (§9.2 replay harness; `spore-104` dep-1 graph projection-version primitive still upstream) on the cleaned graph → does the hop earn Slice C, or NO-GO. Deferred enhancements (per-call config / distribution-relative bar / single-snapshot stats read) = `spore-146`, none blocking. `spore-141` resolved (done); `spore-104` dep-2 done.

---

## 13. receipts-per-bucket MEASURED → pooled-multi-snapshot path → CAPTURE layer shipped (2026-06-21)

dep-1 shipped THIN (pin-not-rebuild). Its success-condition (a single pinned snapshot has A/B power only if enough same-bucket receipts exist) was MEASURED before building the harness — disk/source-verified, not theorized.

**The measurement.** Cadence: 22 mutating drains / 9 days = **~2.4/day**; hwm bumps **once per drain** (`pattern_associations.py:621`, gated on `formed or strengthened or gced`), NOT per-edge → constant within a session, so a bucket = one inter-drain window. Density: ~75 receipts/active-day, **53% empty-exposed** (noise → recall correctly empty post-IDF), **~24 usable (≥2-exposed)/day**. → a single natural `(hwm, query_date)` bucket ≈ **8 usable comparison-queries = SPARSE** (Step-C drew conclusions at 39–53; one natural snapshot is below A/B power). Gotcha confirmed: the live `projection_meta` row was empty → all stampable receipts were `(1,0)` until the next mutating drain (the first bump lazily inserts the row → hwm=1).

**Two findings reframed the thin-vs-as-of binary.**
1. **Receipts were TRANSIENT (blocks ANY replay path).** The drain rotates + UNLINKS the spool (`anneal_dualwrite:481`), extracting only pattern-pair co-surfaces and discarding `query_text`/`exposed`/`hwm`/`source`. The §12 read's corpus was a *lucky failed-drain orphan*. So step-0 of any harness = durable receipt retention.
2. **Pooled multi-snapshot reaches power WITHOUT the as-of rebuild.** Pin the graph (cheap DB `VACUUM INTO`) at EVERY hwm advance; the `(hwm, query_date)` date axis does NOT fragment a bucket because lazy decay is closed-form (`strength × decay^Δdays`) and within one hwm the edge SET is fixed; gain is a PAIRED per-query diff (baselines graph-independent/stable), so pooling paired diffs across heterogeneous-graph snapshots is valid. ~17 snapshots/week × ~8 usable ≈ **~135 pooled comparison-queries/week** — powered — without the as-of CQRS rebuild (blocked anyway: co-surface events truncated post-drain, decay wall-calendar, GC deletes). **Phill RATIFIED pooled-multi-snapshot.**

**Build-order consequence.** The §9 build-order's "step-1 receipts SHIPPED" shipped them to a transient sink. The real prerequisites — built this session, full 4-layer apparatus — are:
- **step-0 (durable receipt corpus):** `recall_injection_hook._append_durable_receipt` tees every receipt to `state/retrieval_receipts.jsonl` — append-only, spool-relative (`_receipt_log_for`; the spool is the single isolation handle so tests can't pollute the real corpus), `flock(LOCK_EX|LOCK_NB)`-serialized on a stable `.lock` sidecar (non-blocking — rotation only when the lock is won; the ≤16KB O_APPEND is atomic, so a receipt is never lost to contention), rotating to one `.1.jsonl` backup, fail-silent. Receipts gain **`query_date`** (local; `RECEIPT_VERSION=2`).
- **step-1 (per-drain graph snapshot):** `anneal_dualwrite._snapshot_graph_if_advanced` (called in `cmd_save` after the drain) `VACUUM INTO` a `.pending.*` temp → reads `(pv, hwm)` FROM the snapshot (definitionally consistent tag, closes the TOCTOU) → copies `<stem>.crystal.json` → `os.replace`-publishes → appends `state/graph_snapshots/manifest.jsonl` (the commit point). Idempotent by `(pv, hwm)` with file-existence; prune keeps last 40 hwm + sweeps crash/temp orphans.

**§9.2 HARNESS CONTRACT (step-2, gated on ~1 week accumulation — `spore-149`):**
- Bucket on `(projection_version, high_water_mark, query_date)`. A receipt whose `(pv, hwm)` has NO exact snapshot is EXCLUDED, never nearest-matched.
- Replay each receipt's `query_text` against its hwm-snapshot (decay-projected to its `query_date`) through the 4 baselines + the graph-consuming sandbox path; gain = pooled paired (graph-consuming − best-baseline).
- hwm is bumped by 5 paths (seed/drain/rename/sever/gc); in flow's loop all run inside the wrap (snapshotted), but an out-of-band anneal mutation could advance hwm un-snapshotted → those receipts DROP.

**Apparatus highlight:** codex L3 caught 2 HIGH the Claude-lineage pass (L1+L2+the in-context fixes) missed — a *blocking* `flock` on the every-prompt hot path, and a missing local `query_date` (UTC-derived date mis-buckets across midnight). `cross_substrate_review_codex_nonreplaceable`, on PROSE-adjacent measurement code this time.
