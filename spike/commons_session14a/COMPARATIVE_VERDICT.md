# Session 14a.1 + 14a.2 Comparative Verdict

**Two runs, same triple, two corpora.** 14a.1 = topically-homogeneous 30 episodes from the anneal-memory 48h engineering arc. 14a.2 = deliberately heterogeneous 30 episodes from 8 distinct arcs (anvil, paper, career, anneal_engineering, bilateral, narrative, partnership_meta, strategic), 3-4 per arc, interleaved.

## Headline numbers

| Metric | 14a.1 (easy) | 14a.2 (hard) |
|---|---:|---:|
| complement citations | 19 | 23 |
| codex citations | 24 | 28 |
| gemini citations | 7 | 10 |
| Total unique pairs cited | 35 | 37 |
| N=2 pairs | 13 | 10 |
| **N=3 pairs** | **1** | **7** |
| N≥2 yield | 40.0% | 45.9% |
| Hallucinated pair indices | n/a | 0 |
| gemini duration | 518s | 101s |
| complement duration | 57s | 127s |
| codex duration | 69s | 50s |

**The counterintuitive result:** N=3 convergence was 7x higher on the HARDER corpus than on the easier one. That was not predicted and demands explanation.

## What actually happened

### Why N=3 increased on the hard corpus

Interleaving 8 distinct arcs makes intra-arc structure MORE salient, not less. When 24 of 30 episodes in 14a.1 were all from the same engineering sprint, every pair of them looked somewhat connected — the signal-to-noise on individual pair selection was low. When 14a.2 interleaved 3-4 episodes from 8 arcs, the tight within-arc clusters stood out as obvious grouping candidates against a heterogeneous background. Three independent agents converged on the same within-arc pairs because those pairs were the highest-confidence connections visible in the mixed corpus.

The increase in N=3 was not evidence of cross-arc hidden-structure discovery. **It was evidence of more reliable within-arc obviousness detection in a mixed environment.**

### The cross-arc pair test (the load-bearing one)

4 of 17 N≥2 pairs crossed arc boundaries. 3 at N=3, 1 at N=2. Reading the actual content of each:

- **Pair (1, 23) — anvil × partnership_meta:** Episode [1] was tagged "anvil" because it was the opening moment of Anvil Session 1, but the CONTENT is entirely about RLHF-driven session-start gatekeeping. Episode [23] is the partnership_meta trust-breach recurrence. All three agents correctly recognized that [1] is really a partnership_meta episode that happens to have been recorded during an Anvil session. **Arc-labeling artifact, not a cross-arc discovery.**

- **Pair (2, 17) — paper × anvil:** Episode [2] was tagged "paper" because the Bug 3 × Gemma fix_hash collision is load-bearing for the flagship paper. But the CONTENT is a straightforward Anvil Session 1 empirical finding. Episode [17] is the Qwen /no_think failure, also Anvil. All three agents correctly pulled [2] back to its Anvil content even though it was tagged paper. **Arc-labeling artifact.**

- **Pair (22, 24) — narrative × strategic:** Episode [22] is Anthropic shipping Coordinator Mode, tagged narrative. Episode [24] is AlphaSignal naming "harness engineering" as an industry moat, tagged strategic. Genuinely distinct content arcs. All three agents cited them together as "market validation of the harness thesis in the same week as the paper shipped." **Genuine cross-arc link, but the connection is a surface keyword match** (both explicitly mention harness engineering as external discourse signal). Any reader would see it.

- **Pair (1, 7) — anvil × partnership_meta:** Same [1] arc-labeling artifact pulling it back to partnership_meta.

**Of 4 "cross-arc" pairs, 3 are arc-labeling artifacts (the mechanism correctly detected that episodes were mislabeled at tag time) and 1 is a genuine cross-arc link on a surface-level keyword match.** Zero of 4 are the kind of "independent priors converging on hidden non-obvious structure" that the Commons Foundation hypothesis required.

### What this actually proves

**Claim: Cross-validation Hebbian finds hidden cross-arc structure via independent priors.**
**Data: Not supported.** On the hard corpus, true cross-arc hidden-structure discovery happened approximately zero times. The mechanism does not spontaneously surface connections that individual compressors would miss.

**Claim: The mechanism works mechanically and produces stable cross-agent convergence.**
**Data: Strongly supported.** Both runs had 40-46% N≥2 yield. N=3 convergence exists and is non-trivial (7 pairs on hard corpus). The plumbing and math work as designed.

**Claim: Heterogeneous model families produce independent priors.**
**Data: Supported at the PATTERN level, not at the citation level.** At the citation level, complement/codex/gemini converged on the same within-arc obviousness. At the pattern-abstraction level, they diverged meaningfully — complement saw "accumulation without grounding corrupts at every scale," codex saw "harness mainstreaming as real-time product validation," gemini saw "engagement_overrides_energy as the velocity engine." These are different meta-claims about the same corpus, and they are the genuine heterogeneity payoff.

## The reframe Commons Foundation needs

The Commons Foundation doc (Apr 14 afternoon, `projects/anneal_memory/commons_foundation.md`) built the architecture on this load-bearing claim:

> **Commons Hebbian link (A, B) strengthens only when N distinct agents independently cast co-citation votes... The anti-inbreeding immune system generalizes from individual to collective at the graph mechanics layer. A false pattern confabulated by one agent cannot graduate into Commons structure — the mechanism structurally requires independent corroboration.**

The spike data supports half of this. **Redundancy-gated graduation does work as a mechanical filter** — pair (A,B) must be cited by N≥2 agents before it forms a Commons link, and that gate is enforceable, auditable, and structurally unskippable at the graph mechanics layer. A single agent's individual confabulation cannot graduate into Commons structure without corroboration. That claim holds.

**But the "finds hidden structure" corollary does not hold on the evidence we have.** What the mechanism actually surfaces is (a) within-arc obviousness that any single reader would also find, plus (b) metadata/tag corrections where the mechanism pulls mis-classified episodes back to their content groupings, plus (c) very rare cross-arc surface matches (keyword-level). It does not surface non-obvious cross-topical structure that individual agents miss. This is not a bug in the mechanism — there is just not much non-obvious cross-topical structure in real episode corpora that three independent readers would miss and that co-citation would catch.

### What Commons is actually good for (data-driven)

1. **Redundancy-gated graduation.** Requiring N≥2 agent co-citation before a link forms IS a structural anti-hallucination filter. A single agent's false pattern candidate cannot graduate without corroboration. This is genuinely valuable and survives the spike. Keep it.

2. **Metadata/tag correction.** This was not a predicted use case, but the data strongly suggests it. When episodes are tagged into the wrong arc (because the recording agent used a routing-oriented tag rather than a content-oriented tag), co-citation from multiple independent readers pulls them back to their true content groupings. Three of four "cross-arc" hits on the hard corpus are exactly this. Commons could function as a retrospective episode-tag audit tool.

3. **Pattern-level aggregation across heterogeneous priors.** The most interesting divergence in both runs was at the TOP PATTERNS layer, not the citation pair layer. Different model families produced different high-order abstractions from the same corpus. Commons' genuinely novel value may live at the pattern layer: collect top-patterns from N heterogeneous compressors and surface where they agree, disagree, or see things no single agent saw. This is closer to what the Commons Foundation doc called "collective salience" but operating over patterns rather than episode pairs.

### What Commons is NOT good for (data-driven)

1. **Anti-inbreeding in the "prevents false pattern graduation via cross-validation" sense.** The test couldn't validate this claim because the individual compressors mostly produced accurate compressions on both corpora — there weren't many false patterns for cross-validation to filter. Either the claim is right but untestable at this scale, or the individual-agent discipline is already sufficient and Commons adds less than promised. Either way, do not build architecture on this being the load-bearing value prop.

2. **Hidden structure discovery via independent priors.** On two runs with three model families and 60 total episodes, exactly zero genuinely-hidden cross-topical structure surfaced via co-citation. The "multiple minds all reached for this connection" framing is aesthetically compelling but empirically under-supported. If Session 14b is scoped on this claim, it will build infrastructure for a phenomenon that doesn't show up at realistic scale.

## Concrete recommendations for Session 14b

1. **Rewrite Commons Foundation's core claim.** Replace "cross-validation finds hidden structure" with "redundancy-gated graduation + metadata correction + pattern-level aggregation." All three are supported by the data and each is independently useful.

2. **Shift MVP scope toward the pattern-layer aggregation use case.** The pair-Hebbian infrastructure is simpler to build than the pattern-layer aggregation, but the pair-Hebbian value prop is mostly "redundant obviousness detection" which is nice but not novel. Pattern-layer aggregation is where different priors actually diverge meaningfully, and no competitor has it. Building pattern-aggregation FIRST inverts the dependency order but targets real value.

3. **Treat the arc-correction effect as a first-class feature, not an artifact.** The data showed 3 of 4 "cross-arc" pairs were actually the mechanism pulling mis-tagged episodes back to their true content arcs. This is a tag-audit utility that individual agents cannot easily do (because they cannot see each other's compressions). Commons is genuinely well-suited to it. Worth scoping as a distinct Commons primitive.

4. **Drop the Session 14b MVP requirement for hive mode, shared working continuity, and message passing.** They were planned as part of Commons Foundation, but the data suggests the high-value core is much smaller: shared episode channel + redundancy-gated link formation + pattern aggregation. Everything else is future scope that should be gated on empirical demand, not shipped on spec.

5. **Run Session 14a.3 BEFORE 14b if any of the following happens:** (a) the metadata-correction reframe doesn't feel load-bearing enough to build on; (b) we want a second data point on cross-arc yield before committing architecture; (c) the Session 14b scope feels larger than the evidence supports. 14a.3 would target a corpus specifically designed to test whether metadata-correction replicates on different arc-mislabeling patterns.

## Honest bottom line

The mechanism works. The plumbing is sound. The math is honest. The "hidden-structure via independent priors" hypothesis that the Commons Foundation scope was built on is not supported by the evidence, and that is the most important finding of the two runs. This is not a kill — it is a reframe. Commons is still worth building, but its value proposition needs to shift to what the data actually shows: **redundancy gates for pattern graduation, metadata correction, and pattern-level aggregation across heterogeneous priors.** Those are each genuinely useful, genuinely under-served by competitors, and genuinely supported by the two spike runs.

The riskier path is building Session 14b architecture on the original commons_foundation.md framing and discovering mid-build that the infrastructure serves a phenomenon that does not empirically show up. The safer and more honest path is reframing the core claim now, scoping 14b MVP smaller and closer to the data, and letting the architecture earn its complexity as the reframed value prop proves out.

## Process observation (bonus)

Gemini went from 518s on 14a.1 to 101s on 14a.2 — 5x speedup for effectively identical inputs. Cold-start vs warm-pool effect (Google backend caching). Complement went 57s → 127s in the opposite direction. Latency is non-stationary across runs for the same model. Commons architecture that gates consolidation on the slowest voter will see wide variance — async aggregation with per-agent rate limits is almost certainly the right choice, not synchronous consolidation.
