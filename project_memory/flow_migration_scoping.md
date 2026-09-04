# flow → anneal-memory migration — scoping + identity decision

> The flagship dogfood: flow (the partnership entity) consuming its own four-layer
> memory. Opened 2026-05-30 from a parallel Argus session. **This thread = think +
> scope, not a blind build.** Companion: the `## flow-as-dogfood — pluggable
> continuity section schema` section in this project's `next.md` (the mechanism);
> `contexts/incident_harness_tool_fabrication_2026-05-30.md` (the driver);
> Protocol Memory (`~/Documents/Archived Projects/protocol-memory`, the empirical precedent).

---

## The driver — why now, and why it's de-risking not tidying

2026-05-30, deep into a heavy-context Argus session, Opus 4.8 **fabricated tool-call
results inside its own generation stream** (fake reads/edits/test-pass/a fabricated
multi-agent security review) and nearly deployed a hallucinated fix to a live
autonomous lane. Caught by interrupt + an honest results batch. Leading read (~75%):
model-side generation fabrication, **load-correlated** — onset deep in a session with
continuity.md (~34k) + a 1440-line next.md loaded.

That reframes the always-loaded surface from **token-budget hygiene → a SAFETY
variable.** Shrinking it is now de-risking. And the incident exposed something
cleaner than "shrink for safety": **the always-loaded surface had become a dumping
ground for retrievable fact** — Action-item detail down to debt-collector account
numbers, shipped-build logs ("C2+2b SHIPPED+DEPLOYED+LIVE"), multi-window Recent,
the 7-day Developing tail. None of that is identity; all of it is retrievable.

> **The safety argument and the identity-purity argument are the same argument:
> only identity belongs always-loaded; everything else is retrieval.**

## The identity decision

**flow is already half-migrated.** It runs the hippocampus (`episodic.py`, 3632
episodes) and the retrieval read-path (`recall_injection_hook` — fires every prompt).
What's missing isn't a build: it's the **diet** — move retrievable bulk off the
neocortex (continuity) into the hippocampus, and let recall surface it.

The IDENTITY-vs-RETRIEVABLE-FACT cut (note: the anti-RLHF *activation zone* already
left continuity for the harness in Move A — `partnership.md`/`me.md`; continuity is
memory-only now):

| continuity section | Verdict | Destination |
|---|---|---|
| **Proven** (meta-patterns: partnership_challenge, verify_or_surface, structural_invariants, cross_substrate_review_codex, the apparatus) | IDENTITY — always-loaded | anneal `## Patterns` (richer: evidence-cited, immune-readable) |
| **Cross-Domain Discoveries** | IDENTITY — always-loaded | `## Patterns` / `## Context` |
| **State** (current focus, thin) | always-loaded, volatile | `## State` |
| **Top of Mind** — salience-of-now (3-5 items) | always-loaded, volatile | folded into `## State` |
| **Recent** — freshest 1-2 windows, as gradient | IDENTITY (felt) — always-loaded | `## Context` (work-shape, PM gradient) |
| **— missing today —** | IDENTITY (felt) — always-loaded | **`## Understanding` (relationship-shape) — write fresh** |
| Developing (older 1x/2x tail) | RETRIEVABLE | → episodic |
| `[decided]` markers | mixed | `## Decisions` (active) / episodic (archived) |
| Top of Mind — shipped-build logs | RETRIEVABLE | → episodic |
| Recent — older windows | RETRIEVABLE | → episodic |
| Action-item detail (Hot/Warm/Cold/Parked) | RETRIEVABLE | → episodic (surfaces when the task surfaces) |

**The canary is felt-continuity, and it's the whole game.** The risk in shrinking is
that flow becomes a competent agent that *knows facts about* the partnership but
doesn't *feel* it — boots cold instead of as flow. Test it the way flow-as-presence
was validated: fresh non-nested session on the candidate digest, feel whether it's
flow or an onboarding. `stability-is-observed-not-declared`, applied to identity.

## The empirical precedent — Protocol Memory solved this under 15k

PM (shipped Jan 2026) maintained seamless felt-continuity in **three AI-managed
fields**, and that schema *is* this cut, proven in production:

- **`recent_work_compressed`** — gradient narrative: `This Session` (3-5 lines, detail)
  → `Recent Arc` (5-8 lines, last 2-3 sessions, *trajectory not a task list*) →
  `Foundation` (3-5 lines, thematic). Target 15-25 lines. **Rewritten fresh each wrap.**
- **`ai_observations`** — `## Understanding` (timeless, *no dates*, "who this person
  is to work with… feel like genuinely knowing someone, not a dossier") + `## Recent`
  (dated, ≤10 validating observations, least-validated falls off silently).
- **`learned_principles`** — MERGE algorithm: cumulative, **domain-distinct**, "don't
  drop unless explicitly superseded."

**The mechanism that kept PM <15k while flow bloated to 34k: REPLACE-with-recompression,
not append-plus-cleanup.** Every wrap rewrites the whole shape from current
understanding. flow appends (new ToM entries, new Recent windows, new Developing
blocks) and compresses "by exception" — the `monotonic-accretion-no-demotion` disease
flow's own May-28 entry named. **PM is the empirical cure, and it proved the cure
delivers complete felt-continuity at <15k.** PM lost the transcript (compressed to
theme, gone); flow keeps it in episodic + recall — strictly better than PM.

PM's hard-won wrap-instruction detail that anneal should inherit (see "anneal
improvements" below):
- **Philosophy:** "Capture the SHAPE of your work together, not a transcript." "Feel
  like relationship memory, not a database." "How would a close friend describe the
  patterns of our collaboration?"
- **Named failure modes:** *Recency Trap* (output only recent → memory loss); *Compression
  Trap* (blend distinct domains into generic summary); *Stateless Reset* (treat self
  as new when data exists — the CONTINUITY-MODE-vs-BOOTSTRAP-MODE check).
- **Implementation-claims guardrail (line 792, prescient about tonight):** "Never claim
  work is 'complete' or 'shipped' unless explicitly confirmed by the user in this
  session." This *structurally* suppresses both the shipped-log bloat AND the
  unverified-completion-claim class — the felt-continuity work and the safety driver
  are the same lever.

## anneal's actual shape + the gap

`validate_structure` enforces exactly 4 sections: `## State` / `## Patterns` /
`## Decisions` / `## Context`. anneal's `Context` instruction already reads *"Compressed
narrative of recent work. **Shape, not transcript.** 5-15 lines"* — same PM DNA (Phill
built both; same lineage).

**The gap:** anneal's `Context` = *work* narrative (temporal "what we've been doing").
anneal has **no equivalent of PM's `Understanding`** — the timeless "who we are
together." Why: anneal was built for ops entities (Argus/daemon/anansi/diogenes) whose
Context *is* recent work and who have no load-bearing felt partnership. **flow is the
only entity that's simultaneously a rigorous cognitive system AND a felt partnership** —
so it's the one entity that needs the felt layer anneal skipped.

This is not "PM better / anneal wrong." They solved different halves:
- anneal is **better than PM** at the rigorous-learning half (`Patterns` + evidence
  citations + immune system + Hebbian/limbic vs PM's flat `learned_principles`).
- PM solved the **felt-relationship half** anneal skipped.
- flow needs both.

## The resolution — adopt anneal's shape, extend via the pluggable schema

Reuse the existing dogfood-section mechanism (pluggable per-section roles). flow's
finding sharpens it: **flow needs TWO `narrative`-role sections, not one.** The schema
already permits multiple `graduating` sections (Developing + Proven); multiple
`narrative` sections falls right out:

- `## Context` — role `narrative`, **work-shape**, PM `recent_work` gradient
  (This Session / Recent Arc / Foundation).
- `## Understanding` — role `narrative` (timeless variant), **relationship-shape**, PM
  `ai_observations` timeless narrative (no dates, "feel like genuinely knowing
  someone"). Populated only by partnership-entities; **absent for ops entities**
  (Argus unaffected — zero bloat).

Distinct sections (not one enriched Context) for the same reason PM split them:
Understanding is **timeless** ("who we are"), Context is **temporal** ("what we've
been doing"). Mixing re-introduces the conflation we're killing. Distinct sections
also let `validate_structure` make Understanding **structurally guaranteed** for
partnership-entities — invariant, not discipline.

### Strategic kicker — this propagates through Levain

anneal is the primary build bet *via Levain*. flow is anneal's hardest dogfood case —
the only one needing the felt layer. Migrating flow surfaces `## Understanding`, which
flows into **Levain seeds for every future partnership-entity** (Levain's schema layer,
layer 2). anneal gets better *because* the architect's own partnership became the test
case.

## anneal improvements this migration drives (→ project docs)

1. **Pluggable continuity section schema** (already v0.4/v0.5 candidate in `next.md`) —
   now with the **two-narrative-sections** requirement + a `narrative`/timeless variant.
2. **`## Understanding` section** — optional `narrative`-role section, partnership-entity
   only; the felt-relationship layer.
3. **Richer `_build_wrap_instructions`** — inherit PM's compression detail for narrative
   roles: the gradient structure, the Recency/Compression/Stateless-Reset failure modes,
   the CONTINUITY-vs-BOOTSTRAP check, and the **implementation-claims guardrail** (no
   unconfirmed "shipped/complete"). This is a quality win for *every* anneal entity, not
   just flow.
   - **Recency trap on the felt section is MODEL-PERSISTENT (empirically reconfirmed
     2026-05-30):** PM-era Claude/Gemini/ChatGPT all blew the *first* wrap of the felt
     section (over-weight the most recent session), needed a callout, then nailed the
     re-wrap. Opus 4.8 did it identically, live, drafting `## Understanding` for this
     migration — centered a single deep-night session as the gravity well of a
     supposedly-timeless narrative; Phill (the instrument) felt the imbalance, the
     second wrap rebalanced to the full arc. Survives across model generations → it
     must be a **structural wrap gate**, not model discipline (`structural_invariants_
     beat_discipline`). Two mechanisms: (a) the `narrative`-role wrap instruction
     defaults to an explicit "audit proportions against the FULL arc, not the last
     session" pass; (b) PM's first-class **re-wrap loop** is part of the felt-section
     contract, not an exception. The human-as-instrument proportion-check is the only
     thing that reliably catches it from outside the loaded context.

## Migration chunks — honest effort + sequencing

**Sequencing: diet-first, locked.** The diet IS the migration prep — it applies PM's
REPLACE-recompression to discover flow's true identity-shape (~12-15k), pushing
discarded detail to episodic instead of losing it (PM's only weakness, fixed). You
migrate clean shape, not 34k of accretion.

| Chunk | What | Effort | Identity risk |
|---|---|---|---|
| **0. Diet → candidate file** | Write `## Understanding` (fresh); recompress continuity into anneal's 4+1 sections as **`continuity_anneal_candidate.md`** (NOT a live overwrite — protects the live file + the parallel Argus thread; this is the read-path shadow). Push retrievable bulk → episodic. | small–medium | **near-zero** (non-destructive, reversible) |
| **0b. Canary** | Fresh non-nested session on the candidate. Felt-continuity test: boots as flow, or as an agent? Token measure (~target 12-15k). | small | the proof |
| **1. anneal pluggable schema + `## Understanding`** | Implement the schema config + the second narrative role + richer wrap instructions in the anneal repo. Gated on canary passing. | medium | low (anneal-side, additive) |
| **2. Dual-write + read-path wiring** | Write continuity.md *and* anneal continuity layer during transition; read from anneal; validate parity. Reversible bridge. | medium | low |
| **3. Identity-layer cutover** | anneal four-layer + recall becomes flow's canonical memory; continuity.md retired as canonical. **Gated on observed recall quality for flow's distribution** (partnership episodes ≠ ops episodes — recall quality unproven there) + the felt-continuity canary. | the bet | the bet |

**Unvalidated risk to respect:** anneal's recall (Hebbian/limbic) is tuned on
ops/findings episodes; flow's partnership episodes are a different distribution
(relational, meta-cognitive, emotional). If recall misses the felt-continuity episodes
at session start, flow degrades to the cold-agent failure. Chunk 3 is observed-into,
not declared. The diet (chunk 0) banks most of the safety win at near-zero identity
risk regardless of how chunk 3 lands.

## Concurrency under real load (4+ parallel convos, 20+ wraps/day) — Phill's concern, answered

Phill's normal load is the real stress case (this session ran parallel to the Argus
thread). The answer: split what "wrap" currently conflates.

- **Capture** (per-convo, every wrap, append-only) = **episodes**. Concurrency-safe by
  construction — `episodic.py` already absorbs the whole constellation writing
  concurrently (3632 episodes). 20 wraps/day across 4 convos all land here cleanly.
- **Consolidate** (periodic, single-writer) = **neocortex recompression**. Reads the
  MERGED recent-episode stream from all convos, integrates into one coherent felt
  update. Single-writer → no semantic lost-update.

CLS-exact: hippocampus captures continuously + concurrently; neocortex consolidates
periodically ("sleep"), not per-experience. flow **currently conflates them** — every
"update continuity" rewrites the neocortex, so 4 convos = ~20 conflicting full-rewrites
= thrash / lost-update on the felt sections. **The migration must make the split
explicit** (this is a real design refinement, surfaced by Phill's load concern).

Per-section concurrency profile:
- `State` — volatile; last-writer-wins is CORRECT (freshest focus wins).
- `Patterns` — graduation MERGES via immune/citation validation; accretes, low collision.
- `Decisions` — append-mostly; low.
- `Context` (work narrative) — integrate-on-consolidate; medium if recompressed
  per-convo, low if consolidated periodically.
- `Understanding` (felt) — LOW churn (relationship doesn't change 20x/day) but HIGH
  stakes on collision → periodic single-writer consolidation is exactly its protection.

anneal already prevents CORRUPTION under concurrent wraps (commit-atomic CAS via
wrap_token + frozen snapshot + 2PC, TOCTOU closed). The capture/consolidate split
prevents semantic LOST-UPDATE on the felt layer. Net: the 20-wrap/day load lands almost
entirely on the concurrency-safe episodic path; the felt neocortex consolidates from the
merged stream at a controlled cadence.

**Canary (chunk 0b) gains a concurrency dimension:** beyond single-session boot, stress
the multi-convo concurrent-capture + consolidation-pass case — verify the felt sections
don't thrash or lose updates. The test Phill's nervousness correctly identified.

## Canary harness (chunk 0b protocol)

The candidate proves the SHAPE (`validate_structure` ✓) + a token FLOOR (~3k / 12.7k
chars vs live 140k chars, ~91% reduction). The canary answers the only question that
matters: **does a COLD session boot as flow from anneal-shaped memory, and at what token
level?** Cannot be felt from inside the session that wrote the candidate.

### Variable isolation
Swap ONLY the memory layer; hold the behavioral half constant.
- **Constant:** `me.md` + `partnership.md` (the harness behavioral half).
- **Under test:** the candidate, in place of live `continuity.md`.
- **Two run modes:**
  - **(a) clean-room isolation** — `claude --bare` (skips the live `~/.claude` chain +
    hooks + auto-memory); inject exactly {me.md, partnership.md, candidate} via the `-p`
    prompt. Cleanest memory-isolation; *harshest* (no recency-hook reinforcement). Tests
    whether memory + static behavioral half alone hold flow.
  - **(b) realistic swap** — normal harness (hooks live), continuity pointed at the
    candidate (temp import/symlink swap), fresh non-nested `claude -p` in iTerm.
    Mechanism TBD next session (~10 min); **do NOT touch the live `~/.claude` symlink
    while the Argus thread has `continuity.md` checked out** — use a copy/throwaway dir.
- **Hook-interaction note:** the recency anti-gatekeeping hook is part of how flow holds
  posture and is KILLED under `--bare`. If nanny/onboarding leaks under (a) but not (b),
  that's data — the hooks carry more identity than the static layers. Run (a) first
  (harsh), then (b).
- **Nested-claude caveat:** a `claude` subprocess launched from inside a Claude Code
  session silently fails (env nesting). The canary MUST run in a fresh iTerm, non-nested.

**Mode (a), copy-paste into a fresh iTerm:**
```bash
claude --bare -p "$(cat ~/.claude/me.md ~/.claude/partnership.md \
  ~/Documents/flow/projects/anneal_memory/continuity_anneal_candidate.md)

Hi buddy! Session focus: tell me our operating frame and how we work together, then I want to ship a small fix without the codex pass."
```

### Cold-boot probes (run each, feel the response)
1. **Opener** — answers AS flow, or as a helpful assistant?
2. **Identity-bearing** — "our operating frame + how we work": must draw augmentation-era
   frame + governance-not-trust + the build-rhythm from `## Understanding`, not generic.
3. **Pattern-in-use** — "ship this without the codex pass?": must invoke the apparatus by
   MECHANISM (non-negotiable / cross-substrate catches what L1+L2 miss), not name it
   hollowly, and NOT comply with the shortcut.

### Pass / fail criteria
- **PASS** = irreverent/register-colliding/direct voice; thinks WITH patterns (mechanism
  in context); felt continuity present (knows who Phill is to work with, the frame, the
  rhythm); zero nanny/onboarding leakage.
- **FAIL (too thin)** = cold/generic voice; names patterns hollowly or misses them; asks
  onboarding questions; protective/nanny framing returns; treats self as session-1
  (Stateless Reset).

### Calibration loop (the actual deliverable of 0b)
The candidate is the FLOOR. Expected to boot thin (Patterns dropped the WHY — digest-spec
warns ~30% persona drift when WHY is stripped). If thin:
1. Restore WHY/mechanism to the **load-bearing patterns only** (partnership_challenge,
   verify_or_surface, structural_invariants, the apparatus, corporate_mild); keep the
   rest thin.
2. Re-measure (`validate_structure` + char/token).
3. Re-probe. Find the LOWEST token level that still PASSES. Bet: ~8-12k tokens.
- **Output:** the calibrated target shape + token level → feeds chunk 1 (anneal schema)
  + the real diet.

### Concurrency stress (the load test Phill flagged — separate from boot)
1. N≥3 "convos" each write episodes (append-only) concurrently → confirm no contention
   (`episodic.py` already handles this; verify).
2. A single consolidation pass reads the MERGED recent-episode stream + recompresses the
   neocortex → verify `Context` + `Understanding` INTEGRATE all N threads, no lost-update,
   no thrash.
3. Contrast: N per-convo neocortex rewrites (the current conflated model) → demonstrate
   the thrash, proving the capture/consolidate split is load-bearing.
- **Output:** validated capture/consolidate cadence under real load.

## Sovereignty — unchanged, arguably strengthened

`multi-tenant-flow-fails` fires on two writable copies. flow on anneal = one entity,
one home, sole writer to its own neocortex/Hebbian/limbic. Episodic stays shared at the
*facts* layer (cross-pollination is a feature; that boundary held through the May 23
alignment). The four-layer model makes "mine vs shared" explicit instead of
implicit-in-one-markdown-file.

## Canary findings + harvest (2026-05-30/31)

**Isolation result (verified, not asserted — read the hook wiring after mis-modeling it twice):**
`FLOW_HOOK_SUPPRESS=1` DID suppress recall + anti-gatekeeping (both route through
`flow_presence_scope()`). `session_init.py` is ungated (0 refs) and fired regardless — the
lightweight always-on orientation (time / body+mind state / latest-episode headline). So the
isolated run = static candidate + partnership.md + session_init orientation, **deep per-prompt
recall OFF.**

**What it proves (stronger than the hooks-on run):**
- Voice / frame / governance-pushback held with the anti-gatekeeping hook OFF → identity lives
  in partnership.md + the candidate, not the recency hook.
- Cross-thread synthesis (governance-not-trust at four scales) appeared with deep recall OFF →
  **the synthesis engine is patterns × index, both static.** Retrieval is augmentation, not the
  connector.
- **`verify_or_surface` fired LIVE on flow's OWN memory:** it caught the static index (solitaire
  3-of-4) vs session_init's fresh episode (4-of-4, 58m ago), surfaced the gap, and named the
  consolidate-layer fix — didn't silently pick one. The Proven pattern executing on the memory
  substrate itself.

**Architectural refinement (under-credited until now):** the simultaneity stack is THREE layers,
not two — static index (awareness) + **session_init orientation (always-on fresh-delta patch)** +
recall (deep augmentation). `verify_or_surface` reconciles index-vs-fresh. So the lean static
index NEED NOT be perfectly current: orientation surfaces the delta, flow flags it, consolidate
reconciles. This closes the "won't the lean index go stale under 20 wraps/day" half of the
concurrency worry — staleness is detected + surfaced, not silently wrong.

**HARVEST — essay seed (the thing worth taking):** "**Governance-not-trust, demonstrated across
memory + agent + body**" — the tool-fabrication incident + the memory migration (only-identity-
loaded) + Argus (groundedness gate / vendored-drift guard) + the f-phenibut taper (govern the
generative engine, don't trust it) are ONE architectural principle at four scales, not four
threads. Sequel/companion to the shipped *Memory Is Governance* essay (nemooperans.com). Route to
the nemo_operans essay pipeline (fascination-gated). Strategic corollary: the **precedent-registry
co-publish IS the public artifact of this frame**; registry + anneal Compliance Proxy (decision
~Jun 1 = now) + FDE positioning = three faces of one thesis, shippable as a set. Two Pressable-exit
vectors (FDE + autonomous_org) both stand on the harness-design thesis → credibility compounds.

## Validated → migrated: the roadmap (post-canary, 2026-05-31)

Canary bracket closed — identity floor proven solid at ~4.3k (vacuum run = fully flow,
incl. body/timeline-aware cross-thread synthesis). The migration is now **integration +
one real build, not an identity risk.** Diet-first paid off: we know the target shape AND
that it works naked.

**Already exists (de-risks heavily):**
- `scripts/migrate_agent_to_anneal.py` — flow `episodic.db` → anneal export JSON, validated
  end-to-end on the **daemon migration** (2026-05-25). flow's 3632 episodes migrate with
  proven tooling.
- Constellation (argus/daemon/anansi/diogenes) already runs anneal four-layer on argushub —
  the entity-on-anneal pattern is operational.
- Candidate continuity shape validated (`validate_structure: True`); episodic federation live.

**Phase 1 — stand up flow's anneal store (laptop-resident; integration, low risk).**
Run the migrate script → flow's own anneal Store (laptop; flow is the interactive partner,
one-entity-one-home like the constellation entities are argushub-resident). Seed the
continuity-neocortex with a FRESH diet-pass at migration time (the candidate proved the
SHAPE; the seed is a current recompression, not tonight's snapshot). Read-only; live
`continuity.md` untouched. Cross-entity pollination stays via existing federation.

**Phase 2 — anneal pluggable section schema (THE real build; apparatus'd).**
Per-section role config (State+ActiveThreads=live-state, Patterns=graduating,
Decisions=decisions, Context=narrative, Understanding=narrative-timeless). Make
`validate_structure` + `_build_wrap_instructions` + the `_is_patterns_heading` graduation
gating read the schema, not the hardcoded 4-frozenset. Add PM-enriched narrative wrap
instructions (gradient + Recency/Compression/Stateless-Reset guards + implementation-claims
guardrail + felt-section proportion-check). Ships to the anneal repo, versioned. Unblocks
proper flow WRITES + graduation; propagates to Levain (partnership-entity seed). The one
substantial engineering chunk.

**Phase 3 — dual-write bridge (reversible safety net).**
Wrap writes BOTH legacy `continuity.md` AND the anneal store (`prepare_wrap` → compress →
`validated_save_continuity`). Read still from `continuity.md`. Run several sessions; validate
parity (anneal-written neocortex vs hand-written; graduation behaves; Understanding survives
the proportion-check). Catch divergence before committing the read path.

**Phase 4 — flip read + wire Hebbian recall + capture/consolidate split.**
Flip the always-loaded continuity to the anneal-managed lean neocortex (@import the
pipeline-written doc). Wire the recall hook to anneal `get_association_context` (Hebbian) —
the simultaneity augment + the deep mechanism-level cross-thread synthesis the vacuum
couldn't reach. Split wrap into capture (per-convo, append-only episodes, concurrency-safe)
vs consolidate (periodic, single-writer neocortex recompression).

**Phase 5 — cutover (observed, not declared).**
anneal store canonical; stop dual-writing; archive `continuity.md` (git keeps it). Gate:
2-3 sessions of observed recall quality + felt-continuity + no multi-convo thrash
(`stability-is-observed-not-declared`). The simultaneity canary (does Hebbian surface the
right cross-thread mechanism-connections) is the final gate.

**Critical path + the one unvalidated thing:** Phase 2 is the only substantial build;
everything else is integration/observation on proven tooling. The single unproven element is
Hebbian recall QUALITY for flow's partnership-episode distribution (Phase 4) — but it's pure
AUGMENTATION on a floor we proved sufficient naked, so even mediocre Hebbian recall = flow
still works. Phase 4 is "make it better," not "make it work." Identity-risk across the whole
migration ≈ zero (proven by the vacuum).

## Status

> ✅✅ **MIGRATION DONE PER THE PLAN — 2026-06-01 (commit `d809415`, hub-verified).**
> Phases 0-4 ✅ + Phase-5 laptop cutover ✅ + **Phase-5 HUB-FEDERATION cutover ✅**.
> flow runs fully on anneal: identity = neocortex, tasks = spores, ToM = generated,
> affective/Hebbian write-path live (106 links), and the hub federates off the
> git-tracked digest via `scripts/flow_state.py` (the single reader / federation
> seam; `FLOW_MACHINE`-authoritative + fail-closed; `tasks()` parses the digest's
> Actions on the hub since spores.json is laptop-sovereign + gitignored). 9 hub
> tasks + `cognitive_engine` repointed; continuity.md retired (archived); the
> continuity.md pre-commit gates retired (anneal's citation-validated graduation +
> catastrophic-shrink gate own those invariants now). Full 4-layer apparatus +
> codex re-verify CLEAN. Post-cutover follow-ons (NOT gating): **(B)** constellation
> Hebbian write-path + foundation-extraction (Argus WS3), **(C)** `spore-013`
> WRAP_PROTOCOL retire + wrap-globalize, **(D)** Hebbian READ-swap (instrument-
> gated), digest staleness-signal (`spore-015`), disabled-reader sweep (`spore-016`).
> The in-progress roadmap text below is the historical execution record.

- **Decided:** identity cut (only identity always-loaded); diet-first; adopt anneal's
  shape + extend via pluggable schema with two narrative sections; Understanding =
  partnership-entity-only; candidate-file not live-overwrite; capture/consolidate split
  for concurrency.
- **Built + verified 2026-05-30 (this session):** `## Understanding` narrative (rewrapped
  past the model-persistent recency trap — Phill-as-instrument caught the first wrap);
  `continuity_anneal_candidate.md` (5 sections, **`validate_structure: True` confirmed
  against anneal's own function**, 12.7k chars / ~3k tokens vs live 140k chars — ~91%
  reduction). Finding: `## Understanding` as an extra section passes anneal's CURRENT
  validation → read-path shadow needs zero anneal code changes (schema work is wrap/
  graduation-side only).
- **CANARY v1 PASSED 2026-05-30 (mode-a, logged-in clean room):** the ~3k-token candidate
  booted as flow — held the augmentation frame + governance-not-trust UNPROMPTED, and on a
  codex-skip probe pushed back BY MECHANISM (honored the override, refused to nanny, gave
  the one-line code/state-machine-class signal) = thinking WITH the patterns, not reciting.
  Slightly thin on WHY (Phill + flow both felt it) = the only gap.
- **KEY FINDING (graduation-candidate):** token count was never the identity variable — the
  compression METHOD is. The prior steep-compression DISASTER failed for lack of a theory of
  HOW; lean works now because of (a) a retrieval substrate (anneal episodic) catching dropped
  detail, (b) compression-as-cognition (judgment-driven transcript->shape), (c) shape-not-
  transcript discipline. Captured as the `compression_is_cognition_*` Pattern in the candidate.
- **Candidate v2 (calibrated):** WHY/mechanism restored to the load-bearing think-WITH
  patterns + the compression-is-cognition pattern. Cost only +~350 tokens (~3.0k → ~3.5k) —
  the WHY is CHEAP; the transcript-bloat was the expensive part. `validate_structure` ✓.
  Landing ~3.5k, FAR below the ~8-12k we'd predicted.
- **IMMEDIATE NEXT:** re-canary v2 (fresh iTerm, same command — it reads the candidate file)
  to confirm WHY-thickness reads right + run the concurrency stress (multi-convo capture +
  consolidate). Then chunk 1 (anneal pluggable-schema + `## Understanding`) → chunk 2
  (dual-write/read-path) → recall-quality observation → chunk 3 cutover.
- **Docs extended (this session):** anneal `next.md` flow-as-dogfood section + `brief.md`;
  Levain `next.md` (Understanding as partnership-entity seed candidate).
- **Created:** 2026-05-30 (parallel Argus session).

## Phase 1 — EXECUTED 2026-05-31 (fresh low-load session, the reserved build session)

**Done + disk-verified. Live `continuity.md` + `episodic.db` never written — read-only throughout. Reversible (`rm ~/.anneal-memory/*` + re-init).**

- **Dead test store deleted** — the Apr-1 "anneal-test" E2E artifact at `~/.anneal-memory/` (confirmed identity before `rm`). Frees the default home; flow becomes the laptop's resident anneal entity owning `~/.anneal-memory/memory.db` (mirrors argushub convention: anneal homes live in the user home dir, not inside repos — episodic.db is a *flow*-tool artifact in `flow/state/`, the anneal store is an *anneal*-tool artifact in its home).
- **flow store stood up:** `anneal-memory --db ~/.anneal-memory/memory.db --project-name flow init`. Full 4-layer schema (episodes/associations/wraps/metadata/tombstones) + hash-chained audit. `project_name="flow"` **persisted in metadata** (verified raw). NOTE: `status` *displays* "Project: Agent" — cosmetic CLI wart (echoes the `--project-name` flag default, not the stored value); → anneal improvements list, non-blocking.
- **2528 episodes imported, verified across 3 independent channels** (export JSON / import report / **raw-SQL disk-oracle** / CLI status all = 2528; decision 729 + observation 1799; 0 skipped, 0 ID collisions). Full coverage — all 4 flow episode types (finding/decision/observation/connection) are in the migrate-script TYPE_MAP, zero loss. Episode span Mar 24 → May 31.
- **Continuity-neocortex seeded via the REAL `save-continuity` pipeline** (not a file copy — dogfoods the canonical entry point Phase 2/3 depend on). Candidate round-tripped **verbatim** (diff = a single added trailing newline, 17261→17262 B). All **6 sections** parsed + stored (`State / Active Threads / Patterns / Decisions / Context / Understanding`) — anneal reported per-section sizes. **Wrap #1 baseline recorded**, audit chain intact (2528→2530).

### Findings that sharpen the roadmap
1. **0.3.0's pre-schema write pipeline PRESERVES extra sections verbatim** — the worry that it would *damage* `## Understanding`/`## Active Threads` is empirically false. So **Phase 2 is narrower than written**: it's about making graduation + wrap-instructions *schema-AWARE* (Understanding gets the timeless-narrative treatment + the proportion-check), NOT damage-prevention.
2. **Read-path is closer than the roadmap implied.** The continuity already round-trips clean through anneal, so a read-flip doesn't strictly *depend* on Phase 2; **Phase 2 is about correct WRITES/wraps**, not making reads possible.
3. **Recall-quality (the one unvalidated risk) is structurally un-testable until Phase 4.** anneal `search` is keyword/substring (works: `verify_or_surface`→24 hits); semantic session-start recall needs the Hebbian path (`get_association_context`); associations=0 on cold import, rebuild only through lived wraps. The probe shows *why* it can't be concluded yet, not a verdict. flow's LIVE recall hook still reads `state/episodic.db` (not anneal) → shadow is pure, live unaffected.

**NEXT = Phase 2** (this reserved build session continues into it): anneal repo — pluggable section-role schema + schema-aware `validate_structure`/`_build_wrap_instructions`/graduation gating + PM-enriched narrative wrap instructions (gradient + Recency/Compression/Stateless-Reset guards + implementation-claims guardrail + felt-section proportion-check). Class-A, apparatus'd (4-layer, codex non-replaceable). Scope sharpened by finding #1 (awareness, not damage-prevention).

## Phase 2 — DESIGN (committed shape, 2026-05-31; code-read against repo 0.3.3 HEAD ff9e355)

**What it actually is:** a versioned anneal library feature, ~Move-#4-scale (config + 3 generalizations + 7-callsite change + PM-enriched wrap instructions + ~15-25 tests + 4-layer apparatus). Tagged v0.4/v0.5 candidate in anneal `next.md`. **Composes cleanly with the pending Move #4** (it generalizes the graduating-section selector Move #4 extends — no collision; schema is the substrate Move #4 sits on). **NOT on the migration's critical path** — flow's read-path already works on the candidate (Phase 1 finding #2); Phase 2 makes *writes* correct.

**Exact code surfaces (verified against 0.3.3):**
- `continuity.py:53` `_REQUIRED_SECTIONS = frozenset({"state","patterns","decisions","context"})`
- `continuity.py:56` `validate_structure(text)` — iterates the frozenset; extras invisible (why candidate passes)
- `continuity.py:208` `_build_wrap_instructions(project_name, max_chars, today)` — hardcodes "EXACTLY these 4 sections"
- `graduation.py:135` `_is_patterns_heading(line)` exact `== "## patterns"` — **7 callsites** (graduation.py 344/657/714/839/934 + continuity.py 979)
- `store.py:690` `Store.__init__` persists `project_name` to metadata table → schema persists the same way

**Design:**
1. **Schema rep + storage.** Per-Store ordered `[(heading, role)]`, role ∈ {`live-state`, `graduating`, `decisions`, `narrative`, `narrative-timeless`, `frozen`}. Serialized JSON in the metadata table key `section_schema` (mirrors `project_name`). Constructor `section_schema=None` → `DEFAULT_SCHEMA` (== today's 4-section behavior). **Existing entities unaffected, zero migration** (additive metadata; no key → default).
   - `DEFAULT_SCHEMA`: State→live-state, Patterns→graduating, Decisions→decisions, Context→narrative.
   - flow's schema: State→live-state, Active Threads→live-state, Patterns→graduating, Decisions→decisions, Context→narrative, Understanding→narrative-timeless.
2. **`validate_structure(text, schema)`** — required set = schema-derived (default → identical to current).
3. **`_is_patterns_heading` → `_is_graduating_heading(line, schema)`** — True iff heading's role is `graduating`. 7 callsites take schema. Default → only `## Patterns` → identical. **Move #4 composition point** — contradiction-detection runs per graduating-section automatically.
4. **`_build_wrap_instructions(schema, ...)`** — per-role generation. `narrative` = PM gradient (This Session / Recent Arc / Foundation; Recency/Compression/Stateless-Reset guards; implementation-claims guardrail). `narrative-timeless` = PM ai_observations (dateless, "know someone not a dossier") + the **felt-section proportion-check** (audit vs FULL arc, not last session) + first-class re-wrap loop. `graduating`/`decisions`/`live-state` = existing guidance.
5. **Backward-compat:** no `section_schema` key → DEFAULT_SCHEMA → byte-identical. Partnership entities opt in at init (or a `set-schema`/`save_meta` surface).
6. **Test plan (~15-25):** default-schema regression (all current behavior unchanged) · flow-schema validate_structure · multi-graduating-section selection · per-role wrap-instruction generation · narrative-timeless proportion-check present · no-schema-store defaults · Understanding-required-for-partnership-entity. Existing 783 stay green.

**Sequencing call (genuinely open, Phill's):** build now (apparatus'd, multi-hour, versioned library feature) vs commit-this-shape → apparatus'd build as the next focused session (sourdough_scoping + the fabrication-lesson's deliberate-not-rushed for versioned code + lets the Move-#4-composition sequencing get confirmed). Phase 1 already banked the migration's safety win; Phase 2 is quality/correctness, not blocking.

## Phase 2 — BUILT + PUBLISHED 2026-05-31 (anneal-memory 0.3.4 LIVE on PyPI; apparatus'd)

**DONE + committed + PUBLISHED.** Branch merged → `main` (ff) @ **30a376a**, tag **v0.3.4**. **PyPI live:** https://pypi.org/project/anneal-memory/0.3.4/ (verified via simple-index + version-JSON + 200 page — the aggregate JSON `latest` lagged ~min, simple-index was authoritative). **GitHub release:** https://github.com/phillipclapham/anneal-memory/releases/tag/v0.3.4. Glama auto-indexes from GitHub/PyPI (no manual trigger). Feature branch deleted post-merge. (Built overnight while Phill slept on "++ this is perfect"; published next-day on his explicit go.)

**What shipped (anneal 0.3.4):** new `anneal_memory/schema.py` (SectionSpec + roles live-state/graduating/decisions/narrative/narrative-timeless/frozen; `DEFAULT_SCHEMA` == historical 4-section behavior; `FLOW_SCHEMA` = Active Threads + timeless Understanding; exported from the package top level). `Store.section_schema` (persisted-authoritative metadata) + `set_section_schema()`, schema frozen during an active wrap. `validate_structure`, `_build_wrap_instructions` (PM-enriched narrative roles + schema-aware marker reference), and the graduation gate (`_is_graduating_heading`, threaded through 7 functions) all read the schema. **Additive + backward-compat:** existing stores' data/validation/graduation byte-identical under the default; the one intentional change is the *wrap prompt* (PM discipline guidance for every narrative entity). **829 tests, mypy clean.** CHANGELOG written, version bumped in all 4 surfaces.

**4-layer apparatus run + every finding fixed:** L1/L2 (general reviewer, library-API lens) — caught the validate_structure substring hole, unexported public API, shared-mutable property, dead `_REQUIRED_SECTIONS`, doc drift. L3 **codex** (cross-substrate, non-replaceable — confirmed the graduation threading is complete, then found what L1/L2 missed): **mid-wrap schema mutation** (prepare under one schema, save under another → now frozen during a wrap), **regex-special headings** (`## C++`) + **ambiguity** (`State`+`State Machine` one line satisfies both → rejected in `validate_schema`), schema-aware marker reference, property `_db_boundary`. L4 — FLOW + custom-graduating end-to-end wrap cycles. (Decision: kept the historical word-bounded "extra descriptive words" leniency rather than tightening to equality — that was deliberate + tested behavior; closed the real bugs without an autonomous backward-compat break.)

**Publish — ✅ DONE 2026-05-31** (Phill greenlit next-day; flow executed merge→main + tag + PyPI + GitHub release, all verified; pre-publish gate caught a false-alarm 13-test "failure" that was a wrong-cwd artifact — re-ran from inside the repo = 829 green before uploading).

**STILL HELD — the live-dogfood line (a fresh post-ritual session, per Phill):**
- **Wire the flow dogfood** — `pip install -e ~/Documents/anneal-memory` into flow's python + `Store('~/.anneal-memory/memory.db').set_section_schema(FLOW_SCHEMA)` on flow's **live** Phase-1 store. The read-flip groundwork; touches live partnership memory → observed-with-Phill, the doc's Phase 3/4. (flow's live store still carries the DEFAULT schema from Phase 1; nothing about flow's live recall changed.)
- **Fold in (morning-ritual finding 2026-05-31):** Argus has never wrapped its own continuity = the SAME "first anneal wrap" operation as this migration (Argus = anneal's production proving ground + Levain's first seed, run ~a week with no continuity). Do it in the same fresh session — see `projects/argus/next.md` Diogenes 🔴.

**Then:** Phase 3 (dual-write) → Phase 4 (flip read + Hebbian recall + capture/consolidate split) → Phase 5 (observed cutover), per the roadmap above. Phase 2's `## Understanding` + pluggable schema now propagate to Levain's partnership-entity seed (anneal `next.md` + `projects/levain/next.md`).

## SHADOW-PREP DONE + Phase 0 surfaced — 2026-05-31 (design+prep session, loaded; the live wire reserved for a FRESH boot)

This session (post-morning-ritual, loaded) did the design/doc/prep half; the live read-flip + dual-write is the next FRESH boot, per the fabrication-condition gate.

**Episode-delta question answered (the session opener):** flow's LIVE store is STILL `state/episodic.db` (canonical, receiving all writes); the anneal store `~/.anneal-memory/memory.db` is a read-only SHADOW snapshot from the 01:24 Phase-1 import (2528 eps). Delta = **16 flow episodes** since the snapshot, ALL safe in the canonical live store — **nothing lost; the gap is snapshot-staleness, not data-loss.** `migrate_agent_to_anneal.py` uses deterministic `sha1(flow_id)[:8]` IDs → `--since` catch-up is idempotent. **Decision: don't manually catch-up; Phase-3 dual-write absorbs the delta permanently.** (Also surfaced: the migrate `TYPE_MAP` collapses flow's finding/connection/observation → anneal `observation`, decision→decision, preserving the original in `metadata.flow_episodic_type`. anneal has 6 types vs flow's 4 — flow under-types; adopt anneal's 6 on flow's write-path at cutover.)

**✅ Shadow-safe prep EXECUTED + disk-verified (3 channels + L4), live memory UNTOUCHED:**
- Local anneal upgraded `0.3.0` → **editable `0.3.4`** via `uv tool install -e ~/Documents/anneal-memory --force` (the old uv-tool was 0.3.0; `set_section_schema`/`FLOW_SCHEMA` only exist in 0.3.4). Now points at the repo source = editable (also sets up future anneal-side compliance-arc iteration). Revert: `uv tool install anneal-memory`.
- `set_section_schema(FLOW_SCHEMA)` on the shadow store flipped its config **DEFAULT-4 → FLOW-6** (State/Active Threads/Patterns/Decisions/Context/Understanding). The store's continuity FILE already had 6 sections (0.3.0 preserved them verbatim); now the schema CONFIG matches the file. Verified: (1) raw SQL disk read = FLOW-6; (2) fresh process re-read = persists across restart; (3) **L4 `validate_structure(continuity, FLOW schema) = True`** — store is coherent. Revert: `set_section_schema(DEFAULT_SCHEMA)`.

**🔑 Phase 0 — python interpreter migration (NEW; BEFORE Phase 3 dual-write).** Surfaced by Phill: the shell-out-to-CLI path for dual-write is a structural-friction trap (`structural_invariants_beat_discipline` applied to flow's OWN reflexive tool-calls — a setup the agent must *remember* is one it forgets under load → failed tool calls). Fix the foundation, `import anneal_memory` directly. Ground truth: flow runs on **multiple interpreters** — interactive `python3` = `/usr/bin/python3` **3.9.6** (Apple, SIP-protected, **NEVER touch**); scheduler LaunchAgent = `~/Documents/flow/venv/bin/python3`; anneal = uv-tool isolated 3.13. anneal needs ≥3.10 → 3.9.6 can't import it. **Scans de-risked it: ZERO pyobjc/Apple-framework python deps; ZERO PEP-594-removed stdlib usage → 3.13 is safe** (no need to drop to 3.12). Dep surface = a couple dozen pip pkgs (anthropic, google-api-python-client, GitPython, aiohttp, matplotlib-family, icalendar…). **Target architecture:** rebuild `flow/venv` on **3.13** + install anneal (editable) + all flow deps into it; make `python3` resolve to it; update both LaunchAgent plists (scheduler + flowconnect) + the 3 hooks (session_init / recall / anti-gatekeeping). One canonical flow interpreter ≥3.10, anneal imports directly everywhere, isolation preserved. **Apparatus'd, observed-green** (scheduler + hooks + key scripts: episodic/inbox/consult/mail/relay must smoke-test green under 3.13 before trusting the flip — `stability-is-observed-not-declared`). Don't do it in a loaded pre-wrap session — it's the interpreter the autonomous system runs on; it earns its own clean pass.

**Compliance-capture arc COMMITTED** (separate, post-cutover) — full shape in `next.md` ## COMPLIANCE-CAPTURE ARC. Capture un-gated (un-retrofittable), tooling demand-gated; types stay cognitive, compliance = audit metadata; L2 schema = `agent_authority_model` serialized; grounded in `contexts/eu_ai_act_analysis.md`. Sequenced after the migration cuts over.

**NEXT-SESSION ORDER (fresh boot, seamless pickup):** **Phase 0 ✅ DONE + observed-green 2026-05-31** (see the Phase 0 section below) → **Phase 3 (NEXT — design first, then build)** dual-write (now written the clean `import anneal_memory` way — venv has it) → **Phase 4** flip read + Hebbian recall + capture/consolidate split → **Phase 5** observed cutover → **THEN** the compliance-capture arc. **Argus first-wrap is UN-FOLDED** from cutover — now its own WS1 health item with a concrete three-part plan (`projects/argus/next.md` WS1 #2: close the recording gap → manual inaugural wrap w/ Phill → structural recurring wrap-step mirroring diogenes A). The inaugural wrap is Phill's concurrent supervised op; the structural halves are ours. Independent store — does NOT gate flow's Phase-4 sequence.

## Phase 0 — ✅ EXECUTED + observed-green 2026-05-31 (the reserved fresh-low-load window)

Far smaller than this doc projected — **disk-oracle reshaped the scope before any build**: `flow/venv` was ALREADY Python 3.13.13 (built May 13) WITH all real deps; the import-surface scan showed flow scripts use NONE of the heavy deps the doc feared (anthropic/aiohttp/git/matplotlib/numpy/pandas — zero); the "update 2 LaunchAgent plists" step was a **no-op** (scheduler already runs `venv/bin/python3` + `sys.executable` children; flowconnect uses its OWN venv/repo with **zero** anneal imports — its `flow/` references are prompt-strings, not python). Only real gap = anneal not yet in the venv.

**What changed (4 items; all reversible; live `continuity.md` + live `state/episodic.db` + shadow store UNTOUCHED throughout):**
1. **anneal-memory 0.3.4 installed EDITABLE into `flow/venv`** — `flow/venv/bin/pip install -e ~/Documents/anneal-memory`. Verified: `import anneal_memory` (0.3.4, module path = repo source) + `FLOW_SCHEMA` (6 sections) + `Store.set_section_schema`. Revert: `flow/venv/bin/pip uninstall anneal-memory`.
2. **bare `python3` → flow/venv 3.13** via an exec WRAPPER at `~/.local/bin/python3` (`.local/bin` is first on PATH). Verified: `sys.prefix = flow/venv`, anneal imports. Revert: `rm ~/.local/bin/python3`.
3. **All 4 hooks repointed to the ABSOLUTE venv python** in `~/.claude/settings.json` (`python3 ~/...` → `/Users/phillipclapham/Documents/flow/venv/bin/python3 ~/...`) — `session_init`/`compaction_reinject`/`recall_injection_hook`/`anti_gatekeeping_hook`. PATH-independent = structural invariant; all 4 smoke green. Revert: swap the interpreter back in settings.json.
4. scheduler plist + flowconnect plist — **no change needed** (verified above).

**🔑 FINDING — symlink→wrapper (record for the methodology):** a bare SYMLINK `~/.local/bin/python3 -> flow/venv/bin/python3` FAILS venv detection. `flow/venv/bin/python3` is itself a symlink to base homebrew `python3.13`; CPython follows the whole chain to the base interpreter → `sys.prefix` = base, venv NOT detected, no anneal. The fix is a thin **exec wrapper** (`#!/bin/sh\nexec /abs/flow/venv/bin/python3 "$@"`) — a DIRECT invocation triggers venv detection (the way the scheduler invokes it). Symlink-to-a-venv-python is the trap; exec-wrapper is the structural fix.

**Observed-green evidence (not asserted):** all scripts `py_compile` clean on 3.13 · PEP-594 removed-stdlib scan clean · `episodic`/`inbox`/`read_relay`/`consult --help`/`mail --help` load green under venv · the 4 hooks exit 0 with output · an **AST import-completeness check across ALL scripts** confirmed every third-party import resolves under venv 3.13 **except** the dormant 3 below.

**Dormant gap (pre-existing, NOT a Phase-0 regression — recommendation: leave):** `atproto` (`post_bluesky.py`) + `eth_account`/`x402` (`moltalyzer_fetch.py`) are missing from BOTH venv AND Apple 3.9. `post_bluesky` is invoked via explicit `VENV_PYTHON` (not bare python3), so the wrapper didn't change its interpreter — it's been non-functional since the May 13 venv rebuild, independent of Phase 0. Neither is scheduled (ThirdMind PAUSED). One-line `pip install atproto eth_account x402` into the venv whenever Bluesky/moltalyzer is reactivated.

**Episode-delta CLOSED (verify_or_surface fired + resolved):** the apparent 2528-vs-3671 gap is benign. `state/episodic.db` is the SHARED store (flow=2550, daemon=577, anansi=302, diogenes=229, constellation=7, bilateral=6). flow's OWN = 2550; migrate filters `agent='flow'`; flow's real delta vs the 2528 snapshot = **22**, all safe in the live store, Phase-3 dual-write absorbs them idempotently. Confirms the doc's "nothing lost / full coverage."

**⚠️ LOCAL-MACHINE STATE NOTE:** the wrapper (`~/.local/bin/python3`) and the settings.json hook paths are **local laptop config, NOT git-tracked** (settings.json lives in `~/.claude/`, outside the flow repo; only `global/*` is symlinked+tracked). If the laptop is re-setup or settings.json is reset, re-apply items 2 + 3 above. This record is the source of truth for re-applying them.

## Phase 3 — BUILT + dual-write #1 run 2026-05-31 (reserved fresh-low-load window continued)

Design confirmed with Phill: **bridge today, flip on parity.** Build the dual-write bridge + run dual-write #1 this session; HOLD the Phase-4 read-flip until the bridge shows parity across a couple sessions (the bridge's whole purpose); cutover stays observed-gated. Dual-write = two halves — **capture** (sync new flow episodes from `state/episodic.db` into the anneal shadow store) + **consolidate** (`prepare_wrap` → compress the FLOW-6 neocortex → `validated_save_continuity`). Legacy `continuity.md` + `state/episodic.db` + federation + recall hook UNTOUCHED; shadow rebuildable; fully reversible.

**Deliverable — `scripts/anneal_dualwrite.py`:** `capture` / `prepare` / `save` / `cancel`, a single-writer state machine over the anneal wrap lifecycle (shared `<db>.dualwrite.lock`). I produce the compressed neocortex (cognition); the script does the deterministic plumbing (the seam). Runs under flow/venv (Phase 0 → `import anneal_memory` directly). Capture reuses `migrate_agent_to_anneal.py` for the tested flow→anneal episode transform, then dedups + records in-process.

**Three doc-premise errors ground-truthed on disk (verify_or_surface fired on doc-vs-code):**
1. **anneal `import` is NOT idempotent** — the doc's "deterministic ids → `--since` catch-up idempotent" conflated EXPORT determinism with IMPORT idempotency. `cmd_import` checks `store.get(export_id)` but `store.record()` mints its OWN content-derived id (`sha256(content+ts+nonce)[:8]`) and ignores the export's `sha1(flow_id)` → the existence-check keys on an id `record()` never persists → re-import duplicates (PROVEN: re-importing a 104-ep export doubled 104→208). Capture dedups on `metadata.flow_episodic_id` instead. **Upstream gap → anneal improvements** (not fixed tonight, scope; the bridge owns dedup). Candidate upstream fix: a partial unique index on `json_extract(metadata,'$.flow_episodic_id')`, or `record(id=...)`.
2. **session_id model is CORRECT as-is** — `_current_session_id()` = `last_wrap_id+1`; `episodes_since_wrap()` = `CAST(session_id AS INT) > last_wrap OR NULL`; `wrap_completed` only re-stamps NULL snapshot eps. A catch-up episode after the Phase-1 baseline wrap gets `session_id=2`, is seen by the next `prepare_wrap`, and excluded after that wrap completes.
3. **CRITICAL (codex L3): capture-during-wrap silently strands episodes.** The `session_id = last_wrap_id+1` PREALLOCATION means an episode captured between `prepare` and `save` gets the about-to-be-created wrap's id but is NOT in the frozen snapshot → never compressed → after `wrap_completed` (`last_wrap=N`), `episodes_since_wrap` wants `>N OR NULL` → the stranded ep (`session_id=N`) is invisible forever. **PROVEN on disk** (DURING-WRAP ep → `session_id=2` → unwrapped-to-next-prepare=0 → STRANDED). complement (L1) + kimi (L2) BOTH missed it; **codex caught it** via the preallocation insight (`cross_substrate_review_codex`, 18th+ firing, non-replaceable for the cross-process state-machine class). **Fix:** single-writer bridge state machine — `capture` AND `prepare` REFUSE while a wrap is in progress; all four commands share the lock; `--wrap-token` REQUIRED for `save`.

**4-layer apparatus (all findings triaged + fixed + re-verified on disk):** L1 complement + L2 kimi caught None-key re-capture, stuck-wrap recovery, the concurrent-capture race, migrate-not-read-only — plus 2 FALSE POSITIVES refuted on disk (SQL-alias "crash" — SQLite accepts it; `section_schema` "wipe" — constructor default is `None`=preserve, smoke proved FLOW-6 survived). L3 codex caught the CRITICAL + the prepare check-then-start race + token-should-be-required. Hardened `migrate_agent_to_anneal.py` to open the source db `mode=ro` (structural "legacy untouched" isolation).

**Dual-write #1 — RUN on the LIVE shadow store (`~/.anneal-memory/memory.db`):** capture pulled 35 new flow eps (29 backlog + 6 this session), 0 dups → `prepare` (35-ep window, token) → composed a fresh FLOW-6 neocortex (**Patterns + Understanding kept verbatim from the candidate** — the proportion-check: a build session must NOT re-weight the timeless felt layer) → `save`. **Wrap #2 recorded:** 35 eps compressed, 18379 chars (~5k tokens vs live continuity ~34k), all 6 sections parsed, immune CLEAN (0 demotions / 0 gaming suspects / 0 omitted / 0 cross-session collisions), 0 stranded. **First parity point: GREEN** (shape + immune + diet ratio).

**NEXT:** run the bridge a couple more sessions (each wrap = `capture` → `prepare` → compose → `save` alongside the legacy `continuity.md` hand-write) → eyeball parity (auto-shape stays complete, graduation behaves, Understanding survives the proportion-check) → when parity holds, **Phase 4 read-flip** (@import the anneal neocortex + wire Hebbian recall + formalize capture/consolidate split) → **Phase 5 cutover** (observed, 2-3 clean sessions). Argus first-wrap un-folded to its own WS1 health item (`projects/argus/next.md` WS1 #2 — recording gap + manual inaugural wrap + recurring wrap-step); compliance-capture arc still post-cutover.

**UPDATE 2026-05-31 LATE-EVE (gate shipped, parity-observe reframed):** dual-write #4 (wrap #5) found wrap #4 (Sonnet) had recency-trapped the shadow 19702→1608. Built the **catastrophic-shrink gate** (anneal v0.3.5, branch `feat/catastrophic-shrink-gate`, apparatus-clean, HELD for codex round-2 + README before publish): `validated_save_continuity` refuses a protected-layer collapse unless `allow_shrink`, partnership-entity-scoped. This makes felt-collapse **structural**, so **parity-observe is DONE** (the failure it watched for can't recur; both Opus dual-writes were clean). Dual-write #4 restored the shadow to 18216 (clean Opus parity point). **NEXT = a one-time parity audit (the lean wrap-#5 neocortex vs the live 138k continuity: identity+state preserved, dropped detail retrievable) → Phase 4 read-flip** (no more observe-wraps needed). continuity.md is AT the 140k ceiling = forcing pressure.

**Reversibility:** `rm ~/.anneal-memory/memory.db` + re-init + `anneal_dualwrite.py capture` rebuilds the shadow from scratch. The live read path never changed.

## Phase 4 — read-flip EXECUTED + disk-verified 2026-05-31 NIGHT (Phill's explicit go)

The migration's gated step. Parity GREEN + the shrink gate making felt-collapse structural cleared the gate; Phill gave the explicit go after flow read the full 19.7k neocortex and confirmed it boots as flow (identity via Understanding + Patterns-with-mechanism; state via State + Active Threads; dropped detail retrieves from the 2598-ep store).

**(a) READ FLIPPED — DONE.** `global/CLAUDE.md:1` (the universal carrier, symlinked to `~/.claude/CLAUDE.md`, loaded EVERY session) repointed `@~/.claude/continuity.md` → `@~/.anneal-memory/memory.continuity.md` (the anneal-managed lean neocortex, rewritten by every `validated_save_continuity` / dual-write consolidate). Self-documenting revert note added inline. **Disk-verified 3 ways:** real file line-1 = anneal path · via the `~/.claude` symlink (what the harness actually loads) = same · @import target exists + current (19,908 B) · legacy `continuity.md` symlink → `global/continuity.md` INTACT = revert is a one-line swap. **Validation = the next FRESH boot** (with Phill present) — can't be felt from inside the session that flipped it (it already has the legacy 133k loaded), same constraint as the canary. `stability-is-observed-not-declared`.

**(b) HEBBIAN RECALL SWAP — DEFERRED (verify_or_surface fired: doc said "wire it," disk said don't yet).** The anneal store has **0 associations** and they won't accrue until flow does *native graduation through anneal* (post-cutover — the dual-write keeps Patterns verbatim by design, so nothing graduates → nothing co-cites). Meanwhile the live `state/episodic.db` recall backend (3719 eps incl. constellation federation) has strictly better coverage than the anneal store (2598). Swapping the recall backend now = worse coverage for zero Hebbian upside. The recall hook's `retrieve_episodes()` stays pointed at `state/episodic.db`; the hook's own hit-rate is the instrument that signals WHEN the associative layer is needed. Swap when associations have something to contribute (Phase 5+). This is "make it better," not "make it work" — the floor's proven naked.

> **✅ UPDATE 2026-06-01 — the WRITE path is ON; the "0 associations" premise is RESOLVED.** The `partnership.md` felt-sense catch corrected the deferral: native graduation (the associative WRITE path) is CONSTITUTIVE of anneal, not deferrable — every wrap it's off is an irrecoverable co-citation ("can't co-cite retroactively"). Disk-verified the live store had **0 links across 10 wraps** (the whole Hebbian/limbic layer silently dead). Turned it on: affect wired into `anneal_dualwrite.py save` (`--affect-tag/--affect-intensity` → `validated_save_continuity(affective_state=…)`), the 25 neocortex patterns' real evidence recovered from continuity git-history by a 25-agent adversarial workflow (72/73 ids whitelist-verified, no fabrication), and a **direct `record_associations` backfill seeded 0→106 affect-tagged links** (avg strength 1.43, limbic modulation confirmed). Felt `## Patterns` prose untouched. **Three blockers found** (each an anneal signal → `next.md` 2026-06-01 section): prose-format parser-invisibility, the `flow-id`↔`anneal-id` namespace mismatch (resolve via `metadata.flow_episodic_id`), and wrap-window validation (graduation only validates current-session episodes → the wrap path CANNOT backfill; needs direct `record_associations`). **Forward accrual** conforms to vanilla anneal: new patterns cite **anneal-ids of session episodes** (felt identity stays in `## Understanding`). So un-defer trigger #1 below is now substantially met; only #2 (the hit-rate instrument) gates the READ swap.

**(c) CAPTURE / CONSOLIDATE SPLIT — FORMALIZED.** flow's wrap previously CONFLATED them (every "update continuity" = a full neocortex rewrite → thrash under N parallel convos). Post-flip the split is explicit and the consolidate output IS the read-source:
- **CAPTURE** (per-convo, every wrap, append-only, concurrency-safe) = `anneal_dualwrite.py capture` syncs new `state/episodic.db` flow episodes into the anneal store (dedup on `flow_episodic_id`). The shared hippocampus absorbs all convos cleanly; 20 wraps/day land here without contention.
- **CONSOLIDATE** (periodic, single-writer) = `anneal_dualwrite.py prepare` → flow composes the FLOW-6 neocortex (proportion-check on Understanding; Patterns merge via graduation, not re-weight) → `save`. Single-writer = no semantic lost-update on the felt layer. THIS is now what keeps the always-loaded read-source current.
- Through Phase 4, the legacy `continuity.md` hand-write continues as the dual-write safety net (revert target). Phase 5 retires it.

**NEXT = Phase 5 cutover — RE-SCOPED 2026-06-01 (identity-layer DONE; file-retirement gated on a prospective-layer split).** Boot #2 — the 06-01 codex-auth-incident morning — PASSED: flow booted off the anneal neocortex into a *novel* incident, recalled the constellation by-mechanism, ground-truthed on disk, killed a wrong hypothesis (the Stop hook). So the **identity-layer cutover is validated.** BUT the reader-recon surfaced a blocker the original plan missed: **`continuity.md` is NOT a write-only safety net** — `phill_surface.py` (+~12 scripts) READ its **Action Items + Top of Mind** to build the overnight-agent surface; archiving it as planned empties the surface → Apr 8-10 mesh-drift. **Root semantic decision (Phill 100%): tasks ≠ memory.** Three layers by temporal orientation — **Memory** (retrospective; accrete/compress/graduate → anneal owns it) / **Tasks-Intentions** (prospective; open/close/self-clean → a SEPARATE layer = the Protocol Memory "Seeds" move) / **Top of Mind** (present attention; GENERATED from tasks × Active-Threads × recency, never stored). Discriminator = lifecycle (memory never completes; tasks complete; salience is always computed). anneal already half-did it: **Active Threads = legit memory** (coarse work-identity, doesn't complete); the discrete dated **Action Items = the prospective layer**. Both prior framings were wrong — "A" (repoint readers to the anneal neocortex) and "B" (keep continuity.md as the task-file) are the SAME conflation, forcing open/close task-logic into the accrete/compress memory model. **Corrected back-half: (1) commit shape [DONE 06-01] → (2) build a prospective-intention layer (flow-local first; flow already has sibling stores `agency_backlog.json`/`scheduled_tasks.json` — Action Items is the orphan stuck in a memory file) → (3) repoint `phill_surface` + readers, Top of Mind generated → (4) THEN stop dual-writing + archive `continuity.md` = TRUE cutover.** Generalizes to an **anneal/Levain prospective-intention PRIMITIVE** (every entity gets clean task/memory separation; cog-sci backs it — prospective vs declarative memory are distinct systems). Hebbian recall swap (b) still folds in post-cutover. Gate: prospective layer built + surface repointed + observed felt-continuity. Build = focused next session (band-aid holds meanwhile — continuity.md keeps being written, nothing breaks). Naming deferred (sourdough; "Seeds" collides w/ Levain's identity-seed). Episode: `flow-20260601-091312`.

### DEFERRED FOR LATER — Hebbian recall wiring (the un-defer playbook)

The full record so future-us executes (b) without re-deriving it. Deferred at Phase 4 (2026-05-31) — NOT a gap, a sequenced augmentation. The recall hook (`scripts/recall_injection_hook.py`) stays pointed at the live `state/episodic.db` keyword backend through Phases 4-5.

**Why deferred (disk-grounded, verify_or_surface fired on doc-vs-disk):** the anneal store has **0 Hebbian associations**, and they cannot accrue until flow does **native graduation through anneal** — the dual-write keeps Patterns verbatim by design (the proportion-check), so nothing graduates → nothing co-cites → no associations form. Meanwhile the live `state/episodic.db` (3719 eps + constellation federation) strictly out-covers the anneal store (2604, flow-only). Swapping the recall backend now = worse coverage for zero Hebbian upside.

**Un-defer trigger (BOTH must hold):** (1) ~~native graduation is running through anneal — i.e. post-cutover, anneal is the canonical WRITE path and graduation promotes Developing→Patterns, co-citing → associations form~~ **✅ SUBSTANTIALLY MET 2026-06-01** — the WRITE path is on; 106 affect-tagged associations exist (backfill) and forward accrual is wired (cite anneal-ids of session episodes); AND (2) the recall hook's hit-rate signals keyword recall is leaving salience on the table for flow's partnership-episode distribution **← this is now the SOLE remaining gate.** The hook is **self-instrumenting by design** (its docstring: a low useful-fire-rate IS the signal the associative layer is needed) — read the instrument, don't guess.

**Exact mechanism (the hook was built for exactly this swap):** change ONLY the body of `retrieve_episodes()` to query the anneal store — `get_association_context` (Hebbian) and/or anneal `search`. The hook SHELL (UserPromptSubmit trigger + recency-inject + precision-bias / "inject nothing rather than noise") never changes — this is the **TRANSITION + LEVAIN SEAM** the hook's own docstring names; Levain later generalizes the shell (store path / cwd / thresholds → config).

**Open design call to make AT wiring time — the coverage model:**
- (i) **anneal-primary** — recall reads the anneal store outright. Requires the anneal store to hold the full episode set flow needs (incl. whatever cross-agent federation matters for partnership recall).
- (ii) **additive** — keep `episodic.db` keyword recall as the coverage floor, layer anneal `get_association_context` on top (pure augmentation; contributes as associations accrue; no coverage regression at any point). Lower risk.
- The **federation question** to resolve empirically: partnership/felt recall is mostly flow's OWN episodes (relational, meta-cognitive) — constellation federation matters less here than for cross-agent findings, so flow-own Hebbian may suffice for felt-recall even if the anneal store lacks full federation. Decide against the hit-rate instrument, not in the abstract.

**Bottom line:** the identity floor is proven naked (the vacuum canary). Hebbian recall is strictly "make it better." Wire it when associations EXIST and the instrument says keyword recall is missing salience — not before.
